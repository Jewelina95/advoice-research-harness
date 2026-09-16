#!/usr/bin/env python3
"""Bounded PREPARE validation selection, never official-test evaluation.

Only frozen embeddings and raw state evidence are consumed. Mixed legacy files
are streamed, but excluded rows are discarded before numeric/label decoding.
Text row IDs were not saved historically: ordering is reconstructed explicitly
from subject_features and checked against the audio cache's stored subject IDs.
This is a retrospective development pilot, not a SpeechCARE reproduction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import re
import shutil
import sys
import time
import warnings
import zipfile
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path

# Set only this process's limits before numerical imports, never global packages.
for _variable in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                  "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"]:
    os.environ[_variable] = "1"

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from advoice.states import build_fold_calibrated_state_frame  # noqa: E402

LABELS = ["HC", "MCI", "AD"]
SEED = 20260917
FINGERPRINTS = {
    "text": "9984b35dc61d38d8f970146127d468dd4955a3fe327ead5334bafc1535b02704",
    "audio": "cda0e1239c735d415e3ee6b339a705181474e14344fccbafc5734e3ee719bdcf",
}
THREAD_LIMIT_WARNINGS = []


@contextmanager
def bounded_cpu():
    try:
        controller = threadpool_limits(limits=1)
    except AttributeError as error:
        # Old threadpoolctl cannot inspect some macOS OpenBLAS builds.
        if "'NoneType' object has no attribute 'split'" not in str(error):
            raise
        message = ("threadpoolctl could not inspect legacy OpenBLAS; using process environment "
                   "thread limits. Pre-imported numerical libraries may retain their prior limits.")
        if message not in THREAD_LIMIT_WARNINGS:
            THREAD_LIMIT_WARNINGS.append(message)
            warnings.warn(message, RuntimeWarning, stacklevel=2)
        yield
    else:
        with controller:
            yield


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def preregistered_plan():
    return {
        "version": 1,
        "seed": SEED,
        "labels": LABELS,
        "expected_public_counts": {"train": 1295, "validation": 327},
        "candidates": [
            {"id": f"{head}_cognition_{'on' if cognition else 'off'}_C{c:g}",
             "head": head, "cognition": cognition, "C": c}
            for head in ["concat", "stack"]
            for cognition in [False, True]
            for c in [0.01, 0.1]
        ],
        "selection": "highest validation accuracy; ties macro_f1, macro_auc_ovr, plan order",
        "safeguards": "report paired macro-F1/macro-OvR-AUC regressions; do not override ACC selection",
        "inner_folds": 3,
        "expert_C": 0.01,
        "head": "train-only StandardScaler then unweighted multinomial L2 logistic, lbfgs, max_iter=1000, tol=1e-4",
        "stack": "3-fold training OOF expert probabilities, clipped log probabilities into meta head; experts refit public train",
        "states": "configured nonempty states; active-task state preferred, otherwise overall; z*reliability and reliability",
        "state_calibration": "raw evidence; recompute source reliability; HC reference only within fitting fold; clip z at configured limit",
        "compute": "one CPU thread; no encoder training, downloads, API, thresholds, feature search, or test evaluator",
        "test_policy": "no official-test feature arrays, labels or predictions decoded/evaluated; mixed containers streamed and hashed",
        "interpretation": "historically exposed dataset and method choices; selected validation score is tuning performance, not untouched evaluation",
        "public_source_commit": "bf1281d2e6e3617b3c81d16220a96e02646a570d",
        "text_order_limit": "legacy reconstructed row order, not original per-row cryptographic provenance",
        "embedding_fingerprints": FINGERPRINTS,
    }


def reserve_output(output, source):
    output, source = Path(output).resolve(), Path(source).resolve()
    protected = [source / name for name in ["artifacts", "configs", "references", "runs", "src"]]
    if output == source or any(output == p or p in output.parents for p in protected):
        raise ValueError("Output cannot be inside historical inputs")
    output.mkdir(parents=True, exist_ok=False)


def public_split(path, expected_counts=(1295, 327)):
    partitions = {"train": [], "validation": []}
    seen = set()
    with Path(path).open(newline="") as stream:
        for row in csv.DictReader(stream):
            uid, partition = row["uid"], row["reference_partition"]
            if uid in seen:
                raise ValueError(f"Duplicate public UID: {uid}")
            seen.add(uid)
            if partition in partitions:
                partitions[partition].append(uid)
            elif partition != "test":
                raise ValueError(f"Unknown partition: {partition}")
    if expected_counts is not None and tuple(map(len, partitions.values())) != expected_counts:
        raise ValueError("Public split counts differ from preregistration")
    return partitions


def selected_csv(path, allowed, columns):
    rows = []
    with Path(path).open(newline="") as stream:
        reader = csv.DictReader(stream)
        if not set(columns).issubset(reader.fieldnames or []):
            raise ValueError(f"Missing columns in {path}")
        for row in reader:
            if row["subject_id"] in allowed:
                rows.append({column: row[column] for column in columns})
    return rows


def selected_npz_rows(path, indices, expected_rows):
    """Decode only requested C-order NPY rows; never materialize the full matrix."""
    if len(indices) != len(set(indices)) or not indices:
        raise ValueError("Embedding indices must be nonempty and unique")
    with zipfile.ZipFile(path) as archive, archive.open("embeddings.npy") as stream:
        version = np.lib.format.read_magic(stream)
        if version == (1, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_1_0(stream)
        elif version == (2, 0):
            shape, fortran, dtype = np.lib.format.read_array_header_2_0(stream)
        else:
            raise ValueError(f"Unsupported NPY version: {version}")
        if fortran or dtype.kind != "f" or len(shape) != 2 or shape[0] != expected_rows:
            raise ValueError("Unexpected embedding shape/order/dtype")
        if min(indices) < 0 or max(indices) >= shape[0]:
            raise ValueError("Embedding index out of bounds")
        start = stream.tell()
        row_bytes = shape[1] * dtype.itemsize
        result = np.empty((len(indices), shape[1]), dtype=dtype)
        for destination in np.argsort(indices):
            stream.seek(start + indices[destination] * row_bytes)
            raw = stream.read(row_bytes)
            if len(raw) != row_bytes:
                raise ValueError("Truncated embedding cache")
            result[destination] = np.frombuffer(raw, dtype=dtype)
    if not np.isfinite(result).all():
        raise ValueError("Selected embeddings contain nonfinite values")
    return result


def state_matrix(evidence, config, reference_ids, ids, tasks):
    if not set(reference_ids).issubset(set(evidence.subject_id)):
        raise ValueError("Reference IDs missing from raw evidence")
    # Validation labels cannot affect calibration or pivot grouping.
    clean = evidence.copy()
    clean.loc[~clean.subject_id.isin(reference_ids), "label"] = "unobserved"
    wide = build_fold_calibrated_state_frame(clean, config, set(reference_ids), "HC")
    if wide.subject_id.duplicated().any():
        raise ValueError("Duplicate state rows")
    wide = wide.set_index("subject_id")
    if not set(ids).issubset(wide.index):
        raise ValueError("Some selected subjects have no state evidence")
    definitions = [s["id"] for s in config["states"] if s["metrics"]]
    result = np.zeros((len(ids), 2 * len(definitions)), dtype=np.float64)
    for i, uid in enumerate(ids):
        row = wide.loc[uid]
        for j, state in enumerate(definitions):
            scoped = f"{state}__task_{tasks[uid]}"
            chosen = scoped if pd.notna(row.get(f"state_{scoped}", np.nan)) else state
            value, reliability = row.get(f"state_{chosen}", 0.0), row.get(f"rel_{chosen}", 0.0)
            if np.isfinite(value) and np.isfinite(reliability):
                result[i, 2 * j:2 * j + 2] = [value * reliability, reliability]
    return result


def fit_head(x, y, c):
    if not np.isfinite(x).all() or set(y) != {0, 1, 2}:
        raise ValueError("Head needs finite features and all three training classes")
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=c, solver="lbfgs",
                           max_iter=1000, tol=1e-4, random_state=SEED),
    )
    with warnings.catch_warnings(), bounded_cpu():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(x, y)
    return model


def score(y, probabilities):
    pred = probabilities.argmax(axis=1)
    onehot = np.eye(3)[y]
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(f1_score(y, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "macro_auc_ovr": float(roc_auc_score(y, probabilities, labels=[0, 1, 2], multi_class="ovr", average="macro")),
        "micro_auc_ovr": float(roc_auc_score(onehot.ravel(), probabilities.ravel())),
        "confusion_matrix_HC_MCI_AD": confusion_matrix(y, pred, labels=[0, 1, 2]).tolist(),
    }


def select_winner(results):
    return max(results, key=lambda r: (r["accuracy"], r["macro_f1"], r["macro_auc_ovr"]))


def load_inputs(source, protocol, partitions):
    artifact = source / "artifacts/PREPARE_DrivenData"
    ids = partitions["train"] + partitions["validation"]
    allowed = set(ids)
    features_path = artifact / "subject_features.csv"
    with features_path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        quality_columns = [c for c in reader.fieldnames if c.endswith("f0_valid_fraction")]
        row_order, features = [], {}
        for row in reader:
            uid = row["subject_id"]
            row_order.append(uid)
            if uid in allowed:
                features[uid] = {c: row[c] for c in ["subject_id", "label", "split"] + quality_columns}
    if len(set(row_order)) != len(row_order) or set(features) != allowed:
        raise ValueError("Subject-feature IDs are duplicated or incomplete")
    if any(row["split"] != "train" for row in features.values()):
        raise ValueError("Selected subjects are not historical development rows")
    labels = np.array([LABELS.index(features[uid]["label"]) for uid in ids])
    embeddings = {}
    with np.load(artifact / "multilingual_audio_embeddings.npz", allow_pickle=False) as cache:
        audio_ids = cache["subject_ids"].astype(str).tolist()
    if audio_ids != row_order:
        raise ValueError("Audio IDs do not establish the legacy text row order")
    index = {uid: i for i, uid in enumerate(row_order)}
    for modality in ["text", "audio"]:
        path = artifact / f"multilingual_{modality}_embeddings.npz"
        with np.load(path, allow_pickle=False) as cache:
            if str(cache["fingerprint"].item()) != FINGERPRINTS[modality]:
                raise ValueError(f"Unexpected {modality} embedding fingerprint")
        embeddings[modality] = selected_npz_rows(path, [index[uid] for uid in ids], len(row_order))
    manifest = selected_csv(artifact / "manifest.csv", allowed, ["subject_id", "task_type", "language"])
    tasks = {}
    for row in manifest:
        task = re.sub(r"[^a-z0-9]+", "_", row["task_type"].lower()).strip("_")
        if row["subject_id"] in tasks and tasks[row["subject_id"]] != task:
            raise ValueError("This bounded pilot expects one active task per subject")
        tasks[row["subject_id"]] = task
    if set(tasks) != allowed:
        raise ValueError("Missing manifest task")
    with (source / "configs/states/audio_states.yaml").open() as stream:
        states_config = yaml.safe_load(stream)
    with (source / "configs/metrics/audio_metrics.yaml").open() as stream:
        definitions = {m["id"]: m for m in yaml.safe_load(stream)["metrics"]}
    columns = ["dataset_id", "subject_id", "language", "metric_id", "metric_instance_id",
               "task_scope", "value", "source_reliability"]
    rows = selected_csv(artifact / "metric_evidence.csv", allowed, columns)
    for row in rows:
        uid, metric = row["subject_id"], row["metric_id"]
        definition = definitions[metric]
        row["value"] = float(row["value"]) if row["value"] else np.nan
        reliability = float(row.pop("source_reliability") or 0) * float(definition["reliability"])
        if metric in {"f0_median_hz", "f0_iqr_hz"}:
            prefix = "" if row["task_scope"] == "overall" else f"task_{row['task_scope']}__"
            quality = features[uid].get(prefix + "f0_valid_fraction", features[uid].get("f0_valid_fraction", "0"))
            reliability *= float(np.clip(float(quality or 0) / 0.45, 0, 1))
        row.update(label=features[uid]["label"], split="development",
                   direction=float(definition["direction"]),
                   reliability=reliability if np.isfinite(reliability) else 0.0)
    evidence = pd.DataFrame(rows)
    if evidence.duplicated(["subject_id", "metric_instance_id"]).any():
        raise ValueError("Duplicate raw metric evidence")
    if not np.isfinite(evidence.value.dropna()).all():
        raise ValueError("Infinite raw state evidence")
    return ids, labels, embeddings, evidence, states_config, tasks


def fit_candidates(ids, labels, embeddings, evidence, states_config, tasks, n_train, plan):
    train_ids, validation_ids = ids[:n_train], ids[n_train:]
    y_train = labels[:n_train]
    if len(set(train_ids) & set(validation_ids)):
        raise ValueError("Training and validation IDs overlap")
    state = state_matrix(evidence, states_config, train_ids, ids, tasks)
    matrices = {**embeddings, "state": state}
    oof = {name: np.zeros((n_train, 3)) for name in matrices}
    validation_experts, full_experts, fold_audit = {}, {}, []
    folds = StratifiedKFold(n_splits=plan["inner_folds"], shuffle=True, random_state=SEED)
    # OOF state normalization is fitted independently inside every training fold.
    for fold, (fit_index, held_index) in enumerate(folds.split(np.zeros(n_train), y_train)):
        fit_ids = [train_ids[i] for i in fit_index]
        fold_state = state_matrix(evidence[evidence.subject_id.isin(train_ids)], states_config,
                                  fit_ids, train_ids, tasks)
        fold_audit.append({"fold": fold, "fit_ids": fit_ids,
                           "held_ids": [train_ids[i] for i in held_index],
                           "HC_reference_ids": [train_ids[i] for i in fit_index if y_train[i] == 0]})
        for name, x in matrices.items():
            current = fold_state if name == "state" else x[:n_train]
            model = fit_head(current[fit_index], y_train[fit_index], plan["expert_C"])
            oof[name][held_index] = model.predict_proba(current[held_index])
    for name, x in matrices.items():
        model = fit_head(x[:n_train], y_train, plan["expert_C"])
        full_experts[name] = model
        validation_experts[name] = model.predict_proba(x[n_train:])
    results, fitted, predictions = [], {}, {}
    for candidate in plan["candidates"]:
        names = ["text", "audio"] + (["state"] if candidate["cognition"] else [])
        if candidate["head"] == "concat":
            x = np.concatenate([matrices[name] for name in names], axis=1)
            x_train, x_val = x[:n_train], x[n_train:]
        else:
            x_train = np.log(np.clip(np.concatenate([oof[name] for name in names], axis=1), 1e-8, 1))
            x_val = np.log(np.clip(np.concatenate([validation_experts[name] for name in names], axis=1), 1e-8, 1))
        model = fit_head(x_train, y_train, candidate["C"])
        probability = model.predict_proba(x_val)
        results.append({"candidate": candidate["id"], **candidate, **score(labels[n_train:], probability)})
        predictions[candidate["id"]] = probability
        fitted[candidate["id"]] = {"head": model, "experts": {name: full_experts[name] for name in names}
                                   if candidate["head"] == "stack" else {}, "modalities": names}
    return results, predictions, fitted, fold_audit


def run(source, output, protocol):
    source, output, protocol = Path(source).resolve(), Path(output).resolve(), Path(protocol).resolve()
    reserve_output(output, source)
    start = time.monotonic()
    plan = preregistered_plan()
    plan["created_utc"] = datetime.now(timezone.utc).isoformat()
    write_json(output / "preregistration.json", plan)
    shutil.copyfile(Path(__file__).resolve(), output / "executed_pilot_source.py")
    print("Preregistration locked: 8 candidates, validation ACC primary, no test evaluator.", flush=True)
    artifact = source / "artifacts/PREPARE_DrivenData"
    inputs = {"public_split": protocol, "script": Path(__file__).resolve(),
              "state_implementation": ROOT / "src/advoice/states.py",
              "evidence_implementation": ROOT / "src/advoice/evidence.py",
              "state_config": source / "configs/states/audio_states.yaml",
              "metric_config": source / "configs/metrics/audio_metrics.yaml"}
    for name in ["subject_features.csv", "manifest.csv", "metric_evidence.csv",
                 "multilingual_text_embeddings.npz", "multilingual_audio_embeddings.npz"]:
        inputs[name] = artifact / name
    hashes = {name: sha256(path) for name, path in inputs.items()}
    write_json(output / "input_manifest.json", {
        "sources": {name: {"path": str(path), "sha256": hashes[name]} for name, path in inputs.items()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__, "joblib": joblib.__version__},
        "historical_text_order": plan["text_order_limit"],
    })
    partitions = public_split(protocol)
    write_json(output / "development_split_ids.json", partitions)
    ids, labels, embeddings, evidence, config, tasks = load_inputs(source, protocol, partitions)
    n_train = len(partitions["train"])
    print(f"Loaded only development arrays: {n_train} train, {len(ids) - n_train} validation.", flush=True)
    with bounded_cpu():
        results, predictions, fitted, folds = fit_candidates(
            ids, labels, embeddings, evidence, config, tasks, n_train, plan)
    for name, path in inputs.items():
        if sha256(path) != hashes[name]:
            raise RuntimeError(f"Input changed during pilot: {name}; no selected config locked")
    winner = select_winner(results)
    paired = []
    for on in [r for r in results if r["cognition"]]:
        off = next(r for r in results if not r["cognition"] and r["head"] == on["head"] and r["C"] == on["C"])
        delta = {metric: on[metric] - off[metric] for metric in ["accuracy", "macro_f1", "macro_auc_ovr"]}
        paired.append({"head": on["head"], "C": on["C"], "on_minus_off": delta,
                       "safeguard_regression": delta["macro_f1"] < 0 or delta["macro_auc_ovr"] < 0})
    for candidate, probability in predictions.items():
        frame = pd.DataFrame(probability, columns=[f"p_{label}" for label in LABELS])
        frame.insert(0, "subject_id", partitions["validation"])
        frame.insert(1, "label", [LABELS[i] for i in labels[n_train:]])
        frame["prediction"] = [LABELS[i] for i in probability.argmax(axis=1)]
        frame.to_csv(output / f"validation_{candidate}.csv", index=False)
    bundle = {"candidate": winner["id"], "model": fitted[winner["id"]], "classes": LABELS,
              "state_config": config, "state_reference_ids": partitions["train"],
              "state_reference_evidence": evidence[evidence.subject_id.isin(partitions["train"]) & evidence.label.eq("HC")],
              "plan": plan, "input_hashes": hashes}
    joblib.dump(bundle, output / "selected_model.joblib")
    write_json(output / "fold_audit.json", folds)
    report = {"status": "validation_selection_complete_test_not_evaluated", "winner": winner,
              "results": results, "cognition_pairs": paired,
              "counts": {part: {label: int(sum(labels[(slice(0, n_train) if part == 'train' else slice(n_train, None))] == i))
                                 for i, label in enumerate(LABELS)} for part in partitions},
              "source_hashes_unchanged": True, "elapsed_seconds": time.monotonic() - start,
              "fit_count": 20, "official_test_evaluated": False,
              "thread_limit_warnings": THREAD_LIMIT_WARNINGS.copy(),
              "interpretation": plan["interpretation"],
              "test_history": "The official test was examined in historical artifacts and comparator audit before this pilot. Future evaluation is retrospective, not an untouched confirmatory test.",
              "limitations": [plan["text_order_limit"], "Single fixed public split and seed, not paper ten-run protocol",
                              "Frozen pooled encoders and local ASR differ from SpeechCARE fine-tuned pipeline",
                              "Cognition means deterministic evidence states, not a paid agent or the latest custom deep method"]}
    write_json(output / "results.json", report)
    pd.DataFrame([{k: v for k, v in r.items() if k != "confusion_matrix_HC_MCI_AD"} for r in results]).to_csv(output / "results.csv", index=False)
    lock = {"candidate": {k: winner[k] for k in ["id", "head", "cognition", "C"]},
            "preregistration_sha256": sha256(output / "preregistration.json"),
            "model_sha256": sha256(output / "selected_model.joblib"), "input_hashes": hashes,
            "development_split_sha256": sha256(output / "development_split_ids.json"),
            "test_evaluation_authorized": False, "selected_on": plan["selection"],
            "validation_metrics": {k: winner[k] for k in ["accuracy", "macro_f1", "macro_auc_ovr", "micro_auc_ovr"]},
            "locked_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "selected_config.lock.json", lock)
    print(json.dumps({"winner": winner, "output": str(output), "seconds": report["elapsed_seconds"]}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True, help="Read-only historical 9.2 root")
    parser.add_argument("--output", type=Path, default=ROOT / ".local/prepare_validation", help="New output directory beneath .local/prepare_validation")
    args = parser.parse_args()
    base = (ROOT / ".local/prepare_validation").resolve()
    output = args.output.resolve()
    if output != base and base not in output.parents:
        parser.error("--output must be .local/prepare_validation or a new child directory")
    run(args.source_root, output, ROOT / "references/speechcare/prepare_protocol_inputs.csv")


if __name__ == "__main__":
    main()
