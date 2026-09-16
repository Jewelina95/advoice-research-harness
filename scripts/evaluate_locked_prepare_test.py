#!/usr/bin/env python3
"""Evaluate one locked PREPARE head once, with explicit retrospective authorization.

No fitting, threshold selection, candidate selection, or encoder calls exist here.
The first-phase lock is immutable. All new artifacts go to test_evaluation/.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import os
import platform
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

for _name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
    os.environ[_name] = "1"

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
LABELS = ["HC", "MCI", "AD"]
RELEASED = {
    "speechcare_raw": ("mhubert_test_predictions.csv", "a1dbbb51002878ad4051f9c1c6caab1aad522c25830a8bd669ca78b1dcafc4a8"),
    "speechcare_bias_mitigated": ("mhubert_test_predictions_after_bias_mitigation.csv", "8760be7b82f66c151bbcf825a17a27affa5d1cac68656577cad0d6137a6ede5d"),
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_hash(path, expected):
    if sha256(path) != expected:
        raise ValueError(f"Hash mismatch: {path}")


def read_json(path):
    with Path(path).open() as stream:
        return json.load(stream)


def write_json(path, value):
    with Path(path).open("x") as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")


def authorize_execution(output, lock_path):
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "authorization.json", {
        "test_evaluation_authorized": True,
        "authorization_source": "Explicit user/parent instruction after inspecting completed eight-candidate results and lock",
        "scope": "One unchanged selected model; 412 official test IDs; no refit, thresholds, candidate changes, or iteration from results",
        "original_lock_sha256": sha256(lock_path),
        "original_lock_preserved": True,
        "interpretation": "Retrospective: test outcomes were already examined historically and in the comparator audit",
        "created_utc": datetime.now(timezone.utc).isoformat(),
    })


def test_ids(protocol, expected_count=412):
    seen, ids = set(), []
    with Path(protocol).open(newline="") as stream:
        for row in csv.DictReader(stream):
            uid = row["uid"]
            if uid in seen:
                raise ValueError(f"Duplicate public ID: {uid}")
            seen.add(uid)
            if row["reference_partition"] == "test":
                ids.append(uid)
    if len(ids) != expected_count:
        raise ValueError("Unexpected official-test count")
    return ids


def validate_lock(pilot_dir):
    lock_path = pilot_dir / "selected_config.lock.json"
    lock = read_json(lock_path)
    manifest = read_json(pilot_dir / "input_manifest.json")
    require_hash(pilot_dir / "preregistration.json", lock["preregistration_sha256"])
    require_hash(pilot_dir / "selected_model.joblib", lock["model_sha256"])
    require_hash(pilot_dir / "development_split_ids.json", lock["development_split_sha256"])
    paths = {}
    for name, digest in lock["input_hashes"].items():
        entry = manifest["sources"][name]
        if entry["sha256"] != digest:
            raise ValueError("Manifest disagrees with locked input hash")
        # The live launcher received a platform-only fix after selection.
        # Execute the exact archived source whose hash the original lock records.
        path = pilot_dir / "executed_pilot_source.py" if name == "script" else Path(entry["path"])
        require_hash(path, digest)
        paths[name] = path
    for name, version in {"python": platform.python_version(), "numpy": np.__version__,
                          "pandas": pd.__version__, "sklearn": sklearn.__version__,
                          "joblib": joblib.__version__}.items():
        if version != manifest["versions"][name]:
            raise ValueError(f"Use the original pilot runtime: {name} {manifest['versions'][name]} required, got {version}")
    plan = read_json(pilot_dir / "preregistration.json")
    if lock["candidate"] not in plan["candidates"]:
        raise ValueError("Locked candidate is not preregistered")
    bundle = joblib.load(pilot_dir / "selected_model.joblib")
    if bundle["candidate"] != lock["candidate"]["id"] or bundle["input_hashes"] != lock["input_hashes"]:
        raise ValueError("Saved model identity disagrees with lock")
    if bundle["classes"] != LABELS:
        raise ValueError("Unexpected model class order")
    specification = importlib.util.spec_from_file_location("executed_prepare_pilot", paths["script"])
    helpers = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(helpers)
    return lock, bundle, paths, helpers


def load_test_inputs(ids, bundle, paths, helpers):
    allowed, features, row_order = set(ids), {}, []
    with paths["subject_features.csv"].open(newline="") as stream:
        reader = csv.DictReader(stream)
        quality = [c for c in reader.fieldnames if c.endswith("f0_valid_fraction")]
        for row in reader:
            uid = row["subject_id"]
            row_order.append(uid)
            if uid in allowed:
                features[uid] = {c: row[c] for c in ["label", "split"] + quality}
    if len(row_order) != len(set(row_order)) or set(features) != allowed:
        raise ValueError("Test features have missing or duplicate IDs")
    if any(row["split"] != "test" for row in features.values()):
        raise ValueError("Official IDs disagree with historical test partition")
    labels = np.array([LABELS.index(features[uid]["label"]) for uid in ids])
    with np.load(paths["multilingual_audio_embeddings.npz"], allow_pickle=False) as cache:
        if cache["subject_ids"].astype(str).tolist() != row_order:
            raise ValueError("Legacy embedding row order changed")
    positions = {uid: i for i, uid in enumerate(row_order)}
    matrices = {name: helpers.selected_npz_rows(paths[f"multilingual_{name}_embeddings.npz"],
                                                [positions[uid] for uid in ids], len(row_order))
                for name in ["text", "audio"]}
    tasks = {}
    for row in helpers.selected_csv(paths["manifest.csv"], allowed, ["subject_id", "task_type"]):
        task = re.sub(r"[^a-z0-9]+", "_", row["task_type"].lower()).strip("_")
        if row["subject_id"] in tasks and tasks[row["subject_id"]] != task:
            raise ValueError("Multiple active tasks are unsupported")
        tasks[row["subject_id"]] = task
    if set(tasks) != allowed:
        raise ValueError("Incomplete test task metadata")
    with paths["metric_config"].open() as stream:
        definitions = {m["id"]: m for m in yaml.safe_load(stream)["metrics"]}
    rows = helpers.selected_csv(paths["metric_evidence.csv"], allowed,
                                ["dataset_id", "subject_id", "language", "metric_id", "metric_instance_id",
                                 "task_scope", "value", "source_reliability"])
    for row in rows:
        metric, uid = row["metric_id"], row["subject_id"]
        definition = definitions[metric]
        row["value"] = float(row["value"]) if row["value"] else np.nan
        reliability = float(row.pop("source_reliability") or 0) * float(definition["reliability"])
        if metric in {"f0_median_hz", "f0_iqr_hz"}:
            prefix = "" if row["task_scope"] == "overall" else f"task_{row['task_scope']}__"
            quality = features[uid].get(prefix + "f0_valid_fraction", features[uid].get("f0_valid_fraction", "0"))
            reliability *= float(np.clip(float(quality or 0) / 0.45, 0, 1))
        row.update(label="unobserved", split="test", direction=float(definition["direction"]),
                   reliability=reliability if np.isfinite(reliability) else 0.0)
    evidence = pd.DataFrame(rows)
    if evidence.duplicated(["subject_id", "metric_instance_id"]).any():
        raise ValueError("Duplicate test metric evidence")
    reference = bundle["state_reference_evidence"]
    reference_ids = reference.subject_id.unique().tolist()
    if not reference.label.eq("HC").all() or set(reference_ids) & allowed:
        raise ValueError("Invalid frozen HC reference")
    if not set(reference_ids).issubset(bundle["state_reference_ids"]):
        raise ValueError("Reference subjects not in locked training split")
    # Reapply fixed training statistics. No test values or labels enter the reference.
    combined = pd.concat([reference, evidence], ignore_index=True)
    matrices["state"] = helpers.state_matrix(combined, bundle["state_config"], reference_ids, ids, tasks)
    return labels, matrices


def infer_locked(bundle, candidate, matrices):
    model = bundle["model"]
    names = ["text", "audio"] + (["state"] if candidate["cognition"] else [])
    if model["modalities"] != names:
        raise ValueError("Saved modality order disagrees with selected config")
    if candidate["head"] == "concat":
        x = np.concatenate([matrices[name] for name in names], axis=1)
    elif candidate["head"] == "stack":
        probability = []
        for name in names:
            expert = model["experts"][name]
            if list(expert.classes_) != [0, 1, 2]:
                raise ValueError("Unexpected expert class order")
            probability.append(expert.predict_proba(matrices[name]))
        x = np.log(np.clip(np.concatenate(probability, axis=1), 1e-8, 1))
    else:
        raise ValueError("Unknown locked head")
    if list(model["head"].classes_) != [0, 1, 2]:
        raise ValueError("Unexpected head class order")
    result = model["head"].predict_proba(x)
    if not np.isfinite(result).all() or not np.allclose(result.sum(axis=1), 1):
        raise ValueError("Invalid model probabilities")
    return result


def paired_correctness(labels, ours, other):
    ours_correct, other_correct = ours.argmax(axis=1) == labels, other.argmax(axis=1) == labels
    wins, losses = int(sum(ours_correct & ~other_correct)), int(sum(~ours_correct & other_correct))
    return {"selected_only_correct": wins, "comparator_only_correct": losses,
            "both_correct": int(sum(ours_correct & other_correct)),
            "both_wrong": int(sum(~ours_correct & ~other_correct)),
            "accuracy_delta": float(ours_correct.mean() - other_correct.mean()),
            "mcnemar_exact_two_sided_p": float(binomtest(wins, wins + losses, p=0.5).pvalue) if wins + losses else 1.0,
            "inference_status": "descriptive retrospective comparison; unadjusted for two comparator checks"}


def released_probabilities(path, ids, labels, require_labels=True):
    frame = pd.read_csv(path, dtype={"uid": str})
    if frame.uid.duplicated().any() or set(frame.uid) != set(ids):
        raise ValueError("Released predictions do not match exactly the 412 official IDs")
    frame = frame.set_index("uid").loc[ids]
    lookup = {"C": 0, "HC": 0, "Control": 0, "MCI": 1, "AD": 2, "ADRD": 2}
    if "label" in frame:
        released_labels = [lookup[value] if value in lookup else int(float(value)) for value in frame.label.astype(str)]
        if not np.array_equal(labels, released_labels):
            raise ValueError("Released diagnoses disagree after subject alignment")
    elif require_labels:
        raise ValueError("Released labels are required for ground-truth verification")
    probabilities = frame[["C", "MCI", "ADRD"]].to_numpy(dtype=float)
    if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1, atol=1e-5):
        raise ValueError("Invalid released probabilities")
    return probabilities


def run(pilot_dir, resume_incomplete=False):
    pilot_dir = Path(pilot_dir).resolve()
    lock_path = pilot_dir / "selected_config.lock.json"
    original_lock_hash = sha256(lock_path)
    lock, bundle, paths, helpers = validate_lock(pilot_dir)
    output = pilot_dir / "test_evaluation"
    if resume_incomplete:
        authorization = read_json(output / "authorization.json")
        if authorization["original_lock_sha256"] != original_lock_hash or not authorization["test_evaluation_authorized"]:
            raise ValueError("Incomplete execution has different authorization or lock")
        if any((output / name).exists() for name in ["results.json", "execution_manifest.json", "selected_test_predictions.csv"]):
            raise ValueError("Cannot resume an execution that already saved predictions or results")
        write_json(output / "incomplete_retry.json", {
            "reason": "Comparator schema-only repair: bias-mitigated release has no label column",
            "model_unchanged": True, "original_lock_sha256": original_lock_hash,
            "created_utc": datetime.now(timezone.utc).isoformat(),
        })
        source_snapshot = output / "executed_evaluator_source_retry.py"
    else:
        authorize_execution(output, lock_path)
        source_snapshot = output / "executed_evaluator_source.py"
    shutil.copyfile(Path(__file__).resolve(), source_snapshot)
    ids = test_ids(paths["public_split"])
    development = read_json(pilot_dir / "development_split_ids.json")
    if set(ids) & set(development["train"] + development["validation"]):
        raise ValueError("Official test overlaps locked development IDs")
    if bundle["state_reference_ids"] != development["train"]:
        raise ValueError("Reference IDs differ from the original training split")
    labels, matrices = load_test_inputs(ids, bundle, paths, helpers)
    probabilities = infer_locked(bundle, lock["candidate"], matrices)
    metrics = helpers.score(labels, probabilities)
    comparison, comparator_hashes = {}, {}
    for name, (filename, digest) in RELEASED.items():
        path = ROOT / "references/speechcare/released_outputs" / filename
        if not path.exists():
            comparison[name] = {"available": False}
            continue
        require_hash(path, digest)
        if name == "speechcare_bias_mitigated" and not comparison.get("speechcare_raw", {}).get("available"):
            raise ValueError("Bias-mitigated comparison requires raw-release ground-truth verification")
        other = released_probabilities(path, ids, labels, require_labels=name == "speechcare_raw")
        comparison[name] = {"available": True, "metrics": helpers.score(labels, other),
                            "paired": paired_correctness(labels, probabilities, other),
                            "source": "https://github.com/SpeechCARE/SpeechCARE-NIA-Phase2/tree/bf1281d2e6e3617b3c81d16220a96e02646a570d",
                            "sha256": digest}
        comparator_hashes[str(path)] = digest
    for name, path in paths.items():
        require_hash(path, lock["input_hashes"][name])
    require_hash(lock_path, original_lock_hash)
    require_hash(pilot_dir / "selected_model.joblib", lock["model_sha256"])
    for path, digest in comparator_hashes.items():
        require_hash(path, digest)
    frame = pd.DataFrame(probabilities, columns=[f"p_{label}" for label in LABELS])
    frame.insert(0, "subject_id", ids)
    frame.insert(1, "label", [LABELS[i] for i in labels])
    frame["prediction"] = [LABELS[i] for i in probabilities.argmax(axis=1)]
    frame.to_csv(output / "selected_test_predictions.csv", index=False)
    report = {"status": "completed_retrospective_locked_test_evaluation", "candidate": lock["candidate"],
              "n_test": len(ids), "correct": int(sum(probabilities.argmax(axis=1) == labels)),
              "class_counts": {label: int(sum(labels == i)) for i, label in enumerate(LABELS)},
              "metrics": metrics, "released_comparisons": comparison,
              "paper_context": {"accuracy_mean": 0.7211, "accuracy_95ci_halfwidth": 0.0044,
                                "micro_auc_ovr_mean": 0.8683, "runs": 10,
                                "accuracy_delta_single_pilot_vs_paper_mean": metrics["accuracy"] - 0.7211,
                                "comparable_protocol": False,
                                "source": "https://www.nature.com/articles/s41746-025-02026-x"},
              "original_lock_sha256": original_lock_hash, "original_lock_unchanged": True,
              "model_sha256": lock["model_sha256"], "source_hashes_unchanged": True,
              "fit_calls": 0, "threshold": "unchanged argmax", "selected_models_evaluated": 1,
              "state_reference": "Only saved public-training HC evidence; test labels masked before state transformation",
              "interpretation": "Retrospective, not untouched confirmatory evidence: official test had prior historical and audit exposure. No iteration authorized from these results.",
              "completed_utc": datetime.now(timezone.utc).isoformat()}
    write_json(output / "results.json", report)
    write_json(output / "execution_manifest.json", {
        "evaluator_sha256": sha256(source_snapshot), "evaluator_source": source_snapshot.name,
        "authorization_sha256": sha256(output / "authorization.json"),
        "original_lock_sha256": original_lock_hash,
        "predictions_sha256": sha256(output / "selected_test_predictions.csv"),
        "results_sha256": sha256(output / "results.json"),
        "pilot_input_hashes": lock["input_hashes"], "comparator_hashes": comparator_hashes,
    })
    print(json.dumps(report, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-dir", type=Path, default=ROOT / ".local/prepare_validation")
    parser.add_argument("--authorize-retrospective-test", action="store_true", required=True)
    parser.add_argument("--resume-incomplete", action="store_true", help="One schema-only retry before any predictions/results were saved")
    args = parser.parse_args()
    expected = (ROOT / ".local/prepare_validation").resolve()
    if args.pilot_dir.resolve() != expected:
        parser.error("This evaluator is bounded to the existing .local/prepare_validation lock")
    run(args.pilot_dir, resume_incomplete=args.resume_incomplete)


if __name__ == "__main__":
    main()
