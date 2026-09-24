"""Resumable, fail-closed orchestration for the evidence-state pilot.

The runner deliberately has no implicit data preparation or provider fallback.
An analytical run starts only from explicitly supplied, truth-free prepared case
records.  A dry run records the DAG only; it never emits predictions or metrics.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Any, Literal

import numpy as np
from scipy.stats import binomtest

from .contracts import EvidenceSnapshot, PilotContractError, SubjectRow
from .runtime import CacheEvent, CacheIdentity, InMemoryAssessmentCache


RUNNER_VERSION = "advoice.pilot.runner.v2"
PREDICTION_LOCK_SCHEMA = "advoice.pilot.prediction-lock.v1"
STAGES = (
    "inventory", "split", "canary", "fit-oof", "assess-oof", "fit-fusion",
    "fit-final", "predict-holdout", "stress", "model-comparison", "score", "render",
)
STAGE_DEPENDENCIES: dict[str, tuple[str, ...]] = {
    "inventory": (),
    "split": ("inventory",),
    "canary": ("split",),
    "fit-oof": ("split",),
    "assess-oof": ("fit-oof", "canary"),
    "fit-fusion": ("assess-oof",),
    "fit-final": ("fit-fusion",),
    "predict-holdout": ("fit-final",),
    "stress": ("predict-holdout",),
    "model-comparison": ("assess-oof",),
    "score": ("predict-holdout", "stress", "model-comparison"),
    "render": ("score",),
}
LABEL_CAPABILITIES: dict[str, frozenset[str]] = {
    "inventory": frozenset(), "split": frozenset({"development"}),
    "canary": frozenset(), "fit-oof": frozenset({"development"}),
    "assess-oof": frozenset(), "fit-fusion": frozenset({"development"}),
    "fit-final": frozenset({"development"}), "predict-holdout": frozenset(),
    "stress": frozenset(), "model-comparison": frozenset({"development"}),
    "score": frozenset({"holdout", "stress"}), "render": frozenset(),
}
_SCORING_ARMS = ("B_raw", "B", "B_matched", "J-A", "J-S", "J-AS")
_REQUIRED_ARMS = ("B_raw", "B", "J-A", "J-S", "J-AS")
_ARTIFACT_NAMES = {
    "split_manifest": "split_manifest.csv",
    "cohort_status": "cohort_status.csv",
    "prepared_cases": "prepared_cases.jsonl",
    "development_labels": "development_labels.jsonl",
    "holdout_labels": "sealed_holdout_labels.jsonl",
    "stress_labels": "sealed_stress_labels.jsonl",
}
_STAGE_ARTIFACTS: dict[str, frozenset[str]] = {
    "inventory": frozenset({"split_manifest", "cohort_status", "prepared_cases"}),
    "split": frozenset({"split_manifest", "cohort_status", "prepared_cases"}),
    "canary": frozenset({"split_manifest", "cohort_status", "prepared_cases"}),
    "fit-oof": frozenset({"split_manifest", "cohort_status", "prepared_cases", "development_labels"}),
    "assess-oof": frozenset({"split_manifest", "cohort_status", "prepared_cases"}),
    "fit-fusion": frozenset({"split_manifest", "cohort_status", "prepared_cases", "development_labels"}),
    "fit-final": frozenset({"split_manifest", "cohort_status", "prepared_cases", "development_labels"}),
    "predict-holdout": frozenset({"split_manifest", "cohort_status", "prepared_cases"}),
    "stress": frozenset({"split_manifest", "cohort_status", "prepared_cases"}),
    "model-comparison": frozenset({"split_manifest", "cohort_status", "prepared_cases", "development_labels"}),
    "score": frozenset({"split_manifest", "holdout_labels", "stress_labels"}),
    "render": frozenset(),
}


class RunnerError(RuntimeError):
    """A stage cannot safely run, resume, or publish an analytical artifact."""


class PreflightError(RunnerError):
    """Required immutable inputs or authorization evidence are absent."""


def _canonical(value: Any) -> bytes:
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


def _plain(value: Any) -> Any:
    """Turn immutable mapping views into the canonical JSON value they represent."""

    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    if isinstance(value, frozenset):
        return sorted(_plain(item) for item in value)
    return value


def _hash(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = _canonical(value) + b"\n"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(body)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RunnerError(f"Malformed runner artifact: {path}") from exc
    if not isinstance(value, dict):
        raise RunnerError(f"Runner artifact must be an object: {path}")
    return value


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ("git", "-C", str(Path(__file__).resolve().parents[3]), "rev-parse", "HEAD"),
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"Resolved config requires mapping: {name}.")
    return value


def _artifact_paths(config: Mapping[str, Any], names: Sequence[str]) -> dict[str, Path]:
    raw_artifacts = config.get("artifacts", {})
    artifacts = _mapping(raw_artifacts, "artifacts")
    result: dict[str, Path] = {}
    missing: list[str] = []
    for key in names:
        expected_name = _ARTIFACT_NAMES[key]
        raw = artifacts.get(key)
        if not isinstance(raw, str) or not raw:
            missing.append(f"artifacts.{key} ({expected_name})")
        else:
            result[key] = Path(raw).expanduser()
    if missing:
        raise PreflightError("Missing required artifacts: " + "; ".join(missing))
    absent = [f"{key}: {path}" for key, path in result.items() if not path.is_file()]
    if absent:
        raise PreflightError("Missing required artifacts: " + "; ".join(absent))
    return result


def _contains_truth(value: Any) -> bool:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = "".join(character for character in str(key).lower() if character.isalnum())
            if any(marker in normalized for marker in ("label", "truth", "diagnosis")):
                return True
            if _contains_truth(item):
                return True
    elif isinstance(value, (tuple, list)):
        return any(_contains_truth(item) for item in value)
    return False


@dataclass(frozen=True, slots=True)
class PreparedCase:
    """A materialized analytical case with no scorer-only field."""

    subject: SubjectRow
    base_features: Mapping[str, float | int | None]
    state_features: Mapping[str, float | int | None]
    evidence_snapshot: EvidenceSnapshot

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "PreparedCase":
        if set(value) != {"subject", "base_features", "state_features", "evidence_snapshot"}:
            raise PreflightError("Prepared cases must contain only subject/features/evidence_snapshot.")
        if _contains_truth(value):
            raise PreflightError("Prepared case includes a truth-bearing field.")
        try:
            subject = SubjectRow.from_mapping(value["subject"])
            snapshot = EvidenceSnapshot.from_mapping(value["evidence_snapshot"])
            # The typed snapshot performs the provider-boundary validation. Feature
            # names are runner-local model inputs and intentionally are not part of
            # the public provider payload allowlist.
            snapshot.to_inference_dict()
        except (PilotContractError, TypeError, ValueError) as exc:
            raise PreflightError("Prepared case violates the inference boundary.") from exc
        if snapshot.subject != subject:
            raise PreflightError("Prepared case snapshot subject does not match its subject record.")
        for name in ("base_features", "state_features"):
            features = value[name]
            if not isinstance(features, Mapping) or not features:
                raise PreflightError(f"Prepared case {name} must be a non-empty feature mapping.")
            for key, item in features.items():
                if not isinstance(key, str) or not key:
                    raise PreflightError("Prepared feature names must be non-empty strings.")
                if item is not None and (isinstance(item, bool) or not isinstance(item, (int, float))):
                    raise PreflightError("Prepared features must be finite numeric values or null.")
                if isinstance(item, float) and not np.isfinite(item):
                    raise PreflightError("Prepared features must be finite numeric values or null.")
        return cls(subject, dict(value["base_features"]), dict(value["state_features"]), snapshot)


def load_prepared_cases(path: str | Path) -> dict[str, PreparedCase]:
    """Load the truth-free prepared case contract; no raw materialization fallback exists."""

    target = Path(path)
    cases: dict[str, PreparedCase] = {}
    try:
        lines = target.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise PreflightError(f"Cannot read prepared cases: {target}") from exc
    for line_number, line in enumerate(lines, start=1):
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as exc:
            raise PreflightError(f"Malformed prepared case at line {line_number}.") from exc
        if not isinstance(raw, Mapping):
            raise PreflightError(f"Prepared case at line {line_number} is not an object.")
        case = PreparedCase.from_mapping(raw)
        subject_id = case.subject.subject_id
        if subject_id in cases:
            raise PreflightError(f"Prepared cases contain duplicate subject: {subject_id}.")
        cases[subject_id] = case
    if not cases:
        raise PreflightError("Prepared cases are empty; no analytical inference is possible.")
    return cases


class DiskAssessmentCache(InMemoryAssessmentCache):
    """Atomic content-addressed cache that survives process restart.

    It remains an ``InMemoryAssessmentCache`` subtype because the bounded runtime
    intentionally accepts only that contract. The on-disk object contains both
    identity fields and response payload, so a key collision or stale record is
    rejected before a cached response reaches the provider parser.
    """

    def __init__(self, directory: str | Path, *, enabled: bool = True) -> None:
        super().__init__(enabled=enabled)
        self.directory = Path(directory)
        if enabled:
            self.directory.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.directory / key[:2] / f"{key}.json"

    def lookup(self, identity: CacheIdentity) -> tuple[Mapping[str, Any] | None, CacheEvent]:
        if not self.enabled:
            return None, CacheEvent(identity.key, "disabled", "cache_disabled")
        path = self._path(identity.key)
        if not path.is_file():
            return None, CacheEvent(identity.key, "miss", "cache_empty")
        record = _read_json(path)
        expected = _hash({"identity": record.get("identity"), "payload": record.get("payload")})
        if record.get("content_hash") != expected or record.get("identity") != dict(identity.fields):
            raise RunnerError("Persistent provider cache identity or content hash mismatch.")
        payload = record.get("payload")
        if not isinstance(payload, Mapping):
            raise RunnerError("Persistent provider cache payload is malformed.")
        return dict(payload), CacheEvent(identity.key, "hit", "exact_identity_match")

    def store(self, identity: CacheIdentity, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        body = {"identity": dict(identity.fields), "payload": dict(payload)}
        body["content_hash"] = _hash(body)
        _atomic_json(self._path(identity.key), body)


def _metric(truth: Sequence[str], probabilities: np.ndarray, class_order: Sequence[str]) -> dict[str, float | None | str]:
    order = tuple(class_order)
    if len(truth) != len(probabilities):
        raise RunnerError("Prediction/label denominator mismatch.")
    predicted = tuple(order[int(index)] for index in np.argmax(probabilities, axis=1))
    accuracy = float(np.mean(np.asarray(predicted, dtype=object) == np.asarray(truth, dtype=object)))
    rows: dict[str, float | None | str] = {"accuracy": accuracy}
    for label in order:
        support = sum(item == label for item in truth)
        true_positive = sum(a == b == label for a, b in zip(predicted, truth, strict=True))
        predicted_positive = sum(item == label for item in predicted)
        if support == 0:
            rows[f"recall_{label}"] = None
            rows[f"recall_{label}_reason"] = "class_support_missing"
        else:
            rows[f"recall_{label}"] = true_positive / support
        if predicted_positive == 0:
            # A class present in truth but never predicted has F1=0, not an
            # undefined score. This is distinct from absent truth support.
            rows[f"precision_{label}"] = 0.0 if support else None
            if not support:
                rows[f"precision_{label}_reason"] = "class_support_missing"
        else:
            rows[f"precision_{label}"] = true_positive / predicted_positive
        precision, recall = rows[f"precision_{label}"], rows[f"recall_{label}"]
        if support == 0:
            rows[f"f1_{label}"] = None
            rows[f"f1_{label}_reason"] = "class_support_missing"
        elif isinstance(precision, float) and isinstance(recall, float):
            rows[f"f1_{label}"] = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        else:
            raise RunnerError("Defined class support must produce defined precision and recall.")
        positives = np.asarray([item == label for item in truth], dtype=bool)
        if positives.all() or not positives.any():
            rows[f"auroc_{label}"] = None
            rows[f"auroc_{label}_reason"] = "class_support_missing"
        else:
            # Mann-Whitney form of one-vs-rest AUROC avoids a scorer dependency.
            scores = probabilities[:, order.index(label)]
            positive_scores = scores[positives]
            negative_scores = scores[~positives]
            wins = sum(
                1.0 if positive > negative else 0.5 if positive == negative else 0.0
                for positive in positive_scores for negative in negative_scores
            )
            rows[f"auroc_{label}"] = float(wins / (len(positive_scores) * len(negative_scores)))
    one_hot = np.zeros_like(probabilities)
    for index, label in enumerate(truth):
        one_hot[index, order.index(label)] = 1.0
    rows["brier"] = float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))
    confidence = probabilities.max(axis=1)
    correct = np.asarray(predicted, dtype=object) == np.asarray(truth, dtype=object)
    bins = np.minimum((confidence * 10).astype(int), 9)
    rows["ece"] = float(sum(
        abs(float(confidence[bins == bucket].mean()) - float(correct[bins == bucket].mean()))
        * int(np.sum(bins == bucket)) / len(truth)
        for bucket in range(10) if np.any(bins == bucket)
    ))
    return rows


def paired_bootstrap(
    base_correct: Sequence[bool], arm_correct: Sequence[bool], *, seed: int, replicates: int = 2000,
) -> dict[str, float | int]:
    if replicates != 2000:
        raise RunnerError("Paired bootstrap replicates are fixed at 2000.")
    base = np.asarray(base_correct, dtype=float)
    arm = np.asarray(arm_correct, dtype=float)
    if base.size == 0 or base.shape != arm.shape:
        raise RunnerError("Paired bootstrap requires equal non-empty subject vectors.")
    generator = np.random.default_rng(seed)
    indices = generator.integers(0, base.size, size=(replicates, base.size))
    deltas = (arm[indices] - base[indices]).mean(axis=1)
    return {
        "estimate": float((arm - base).mean()), "ci_low": float(np.quantile(deltas, 0.025)),
        "ci_high": float(np.quantile(deltas, 0.975)), "replicates": replicates,
    }


def paired_accuracy_summary(
    truth: Sequence[str], base_predictions: Sequence[str], arm_predictions: Sequence[str], *, seed: int,
) -> dict[str, Any]:
    if not truth or not (len(truth) == len(base_predictions) == len(arm_predictions)):
        raise RunnerError("Paired metrics require equal non-empty assigned-subject vectors.")
    base_correct = [prediction == label for prediction, label in zip(base_predictions, truth, strict=True)]
    arm_correct = [prediction == label for prediction, label in zip(arm_predictions, truth, strict=True)]
    helped = sum(not base and arm for base, arm in zip(base_correct, arm_correct, strict=True))
    harmed = sum(base and not arm for base, arm in zip(base_correct, arm_correct, strict=True))
    discordant = helped + harmed
    per_class = {}
    for label in sorted(set(truth)):
        indices = [index for index, item in enumerate(truth) if item == label]
        per_class[label] = {
            "n": len(indices),
            "helped": sum(not base_correct[index] and arm_correct[index] for index in indices),
            "harmed": sum(base_correct[index] and not arm_correct[index] for index in indices),
        }
    return {
        "n": len(truth), "helped": helped, "harmed": harmed,
        "unchanged_correct": sum(base and arm for base, arm in zip(base_correct, arm_correct, strict=True)),
        "unchanged_incorrect": sum(not base and not arm for base, arm in zip(base_correct, arm_correct, strict=True)),
        "bootstrap": paired_bootstrap(base_correct, arm_correct, seed=seed),
        "mcnemar": {
            "base_wrong_arm_right": helped, "base_right_arm_wrong": harmed,
            "p_value": float(binomtest(min(helped, harmed), n=discordant, p=0.5).pvalue) if discordant else 1.0,
        },
        "per_class": per_class,
    }


def prediction_lock(rows: Sequence[Mapping[str, Any]], assigned_subjects: Mapping[str, Sequence[str]]) -> dict[str, str]:
    """Create the immutable lock consumed by the score stage.

    The caller persists this alongside the frozen predictions before opening a
    label file. It deliberately contains hashes only, never labels.
    """

    return {
        "schema_version": PREDICTION_LOCK_SCHEMA,
        "status": "locked",
        "rows_hash": _hash([_plain(row) for row in rows]),
        "assigned_subjects_hash": _hash(_plain(assigned_subjects)),
    }


def _validate_prediction_lock(
    rows: Sequence[Mapping[str, Any]], assigned_subjects: Mapping[str, Sequence[str]], lock: Mapping[str, Any] | None,
) -> None:
    if not isinstance(lock, Mapping):
        raise RunnerError("Score requires a locked prediction manifest.")
    expected = prediction_lock(rows, assigned_subjects)
    for field, value in expected.items():
        if lock.get(field) != value:
            raise RunnerError(f"Prediction lock {field} mismatch.")


def score_locked_predictions(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str], *, seed: int,
    assigned_subjects: Mapping[str, Sequence[str]], lock: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Score immutable predictions. Callers must supply labels only at score stage."""

    if not isinstance(labels, Mapping) or not isinstance(assigned_subjects, Mapping):
        raise RunnerError("Score labels and frozen assigned subjects must be mappings.")
    _validate_prediction_lock(rows, assigned_subjects, lock)
    assigned_by_dataset: dict[str, tuple[str, ...]] = {}
    assigned_all: set[str] = set()
    for dataset, values in assigned_subjects.items():
        if not isinstance(dataset, str) or not dataset or isinstance(values, (str, bytes)):
            raise RunnerError("Frozen assigned subjects are malformed.")
        subject_ids = tuple(str(item) for item in values)
        if not subject_ids or len(set(subject_ids)) != len(subject_ids):
            raise RunnerError("Frozen assigned subjects contain an empty or duplicate denominator.")
        overlap = assigned_all.intersection(subject_ids)
        if overlap:
            raise RunnerError("Frozen assigned subjects occur in more than one dataset.")
        assigned_all.update(subject_ids)
        assigned_by_dataset[dataset] = subject_ids
    if set(labels) != assigned_all:
        raise RunnerError("Scoring labels do not exactly match the frozen assigned denominator.")

    grouped: dict[tuple[str, str, tuple[str, ...]], list[Mapping[str, Any]]] = {}
    for row in rows:
        if _contains_truth(row):
            raise RunnerError("Prediction rows must not contain truth-bearing fields.")
        try:
            subject_id, arm = str(row["subject_id"]), str(row["arm"])
            order = tuple(str(item) for item in row["class_order"])
            probabilities = tuple(float(item) for item in row["probabilities"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RunnerError("Malformed locked prediction row.") from exc
        if arm not in _SCORING_ARMS or not order or len(set(order)) != len(order) or len(order) != len(probabilities):
            raise RunnerError("Locked prediction arm or class order is invalid.")
        if not all(np.isfinite(item) and item >= 0.0 for item in probabilities) or not np.isclose(sum(probabilities), 1.0, atol=1e-9):
            raise RunnerError("Locked prediction probabilities must be finite, non-negative, and sum to one.")
        dataset = str(row.get("dataset_id", ""))
        if dataset not in assigned_by_dataset or subject_id not in assigned_by_dataset[dataset]:
            raise RunnerError("Prediction row is outside the frozen assigned denominator.")
        if labels[subject_id] not in order:
            raise RunnerError("Scoring label is not represented by the locked class order.")
        predicted = order[int(np.argmax(np.asarray(probabilities)))]
        if "predicted" in row and str(row["predicted"]) != predicted:
            raise RunnerError("Locked prediction contradicts its probabilities.")
        grouped.setdefault((dataset, arm, order), []).append(row)
    output: dict[str, Any] = {"metrics": [], "paired": []}
    by_dataset_subject: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for (dataset, arm, order), group in sorted(grouped.items()):
        subject_ids = [str(item["subject_id"]) for item in group]
        if len(set(subject_ids)) != len(subject_ids):
            raise RunnerError("Locked predictions duplicate an assigned subject within an arm.")
        if set(subject_ids) != set(assigned_by_dataset[dataset]):
            raise RunnerError("Locked predictions omit or add an assigned subject within an arm.")
        group = sorted(group, key=lambda item: str(item["subject_id"]))
        subject_ids = [str(item["subject_id"]) for item in group]
        truth = [labels[item] for item in subject_ids]
        probabilities = np.asarray([item["probabilities"] for item in group], dtype=float)
        metrics = _metric(truth, probabilities, order)
        for metric, value in metrics.items():
            if metric.endswith("_reason"):
                continue
            output["metrics"].append({
                "dataset_id": dataset, "arm": arm, "metric": metric, "value": value,
                "n": len(group), "denominator": len(group),
                "undefined_reason": metrics.get(f"{metric}_reason"),
            })
        for row in group:
            by_dataset_subject.setdefault(dataset, {}).setdefault(str(row["subject_id"]), {})[arm] = row
    for dataset, subjects in sorted(by_dataset_subject.items()):
        actual_arms = {arm for arms in subjects.values() for arm in arms}
        missing_arms = set(_REQUIRED_ARMS) - actual_arms
        if missing_arms:
            raise RunnerError("Locked predictions are missing required arms: " + ", ".join(sorted(missing_arms)))
        for arm in ("J-A", "J-S", "J-AS"):
            baseline_arm = "B_matched" if "B_matched" in actual_arms else "B"
            ids = sorted(subject_id for subject_id, arms in subjects.items() if baseline_arm in arms and arm in arms)
            if set(ids) != set(assigned_by_dataset[dataset]):
                raise RunnerError("Paired rows must cover every frozen assigned subject.")
            truth = [labels[subject_id] for subject_id in ids]
            base = [
                tuple(str(item) for item in subjects[subject_id][baseline_arm]["class_order"])[
                    int(np.argmax(np.asarray(subjects[subject_id][baseline_arm]["probabilities"], dtype=float)))
                ] for subject_id in ids
            ]
            candidate = [
                tuple(str(item) for item in subjects[subject_id][arm]["class_order"])[
                    int(np.argmax(np.asarray(subjects[subject_id][arm]["probabilities"], dtype=float)))
                ] for subject_id in ids
            ]
            output["paired"].append({
                "dataset_id": dataset, "arm": arm, "baseline_arm": baseline_arm,
                **paired_accuracy_summary(truth, base, candidate, seed=seed),
            })
    return output


@dataclass(slots=True)
class PilotRunner:
    config: Mapping[str, Any]
    run_dir: Path
    provider: Literal["fake", "configured"] = "configured"
    allow_paid: bool = False
    dry_run: bool = False
    resume: bool = False

    def __post_init__(self) -> None:
        self.run_dir = Path(self.run_dir)
        if self.provider not in ("fake", "configured"):
            raise RunnerError("Provider must be fake or configured.")
        if self.provider == "fake" and not self.dry_run:
            raise PreflightError("Fake provider is test-only and may run only with --dry-run.")
        if self.allow_paid and self.provider != "configured":
            raise PreflightError("--allow-paid requires the configured provider.")
        if not self.dry_run and not self.allow_paid:
            raise PreflightError("Analytical execution is disabled by default; pass --allow-paid after review.")
        if self.allow_paid and self.config.get("paid_execution") is not True:
            raise PreflightError("--allow-paid requires paid_execution: true in the resolved config.")

    @property
    def records_dir(self) -> Path:
        return self.run_dir / "stages"

    def record_path(self, stage: str) -> Path:
        return self.records_dir / f"{stage}.json"

    def _input_hash(self, stage: str) -> str:
        artifact_hashes: dict[str, str | None] = {}
        if not self.dry_run:
            raw_artifacts = self.config.get("artifacts", {})
            artifacts = _mapping(raw_artifacts, "artifacts")
            for name in sorted(_STAGE_ARTIFACTS[stage]):
                raw = artifacts.get(name)
                path = Path(raw).expanduser() if isinstance(raw, str) else None
                artifact_hashes[name] = _file_hash(path) if path is not None and path.is_file() else None
        dependency_hashes: dict[str, str | None] = {}
        for dependency in STAGE_DEPENDENCIES[stage]:
            record = self.record_path(dependency)
            dependency_hashes[dependency] = _file_hash(record) if record.is_file() else None
        return _hash({"runner_version": RUNNER_VERSION, "stage": stage, "config": _plain(self.config),
                      "artifacts": artifact_hashes, "dependencies": dependency_hashes,
                      "provider": self.provider, "dry_run": self.dry_run})

    def _dependency_snapshot(self, stage: str) -> dict[str, dict[str, Any]]:
        snapshot: dict[str, dict[str, Any]] = {}
        for dependency in STAGE_DEPENDENCIES[stage]:
            path = self.record_path(dependency)
            if not path.is_file():
                snapshot[dependency] = {"status": "missing"}
                continue
            record = _read_json(path)
            snapshot[dependency] = {
                "status": record.get("status"), "output_hash": record.get("output_hash"),
                "code_sha": record.get("code_sha"), "config_hash": record.get("config_hash"),
                "execution_mode": record.get("execution_mode"),
            }
        return snapshot

    def _dependency_blockers(self, stage: str) -> list[str]:
        blockers: list[str] = []
        required_status = "dry_run" if self.dry_run else "succeeded"
        required_mode = "dry_run" if self.dry_run else "real"
        for dependency, record in self._dependency_snapshot(stage).items():
            if record.get("status") != required_status or record.get("execution_mode") != required_mode:
                blockers.append(dependency)
        return blockers

    def _config_hash(self) -> str:
        return _hash(_plain(self.config))

    def _require_t7_gate(self) -> dict[str, Any]:
        raw = self.config.get("t7_preflight_gate")
        if not isinstance(raw, str) or not raw:
            raise PreflightError("Real execution requires a t7_preflight_gate artifact path.")
        gate_path = Path(raw).expanduser()
        gate = _read_json(gate_path)
        if gate.get("verdict") != "ACCEPT":
            raise PreflightError("T7 preflight gate is not ACCEPT.")
        if gate.get("code_sha") != _git_sha():
            raise PreflightError("T7 preflight gate code SHA does not match this runner.")
        if gate.get("config_hash") != self._config_hash():
            raise PreflightError("T7 preflight gate config hash does not match the resolved config.")
        return {"path": str(gate_path), "sha256": _file_hash(gate_path), "verdict": "ACCEPT"}

    def _preflight(self, stage: str) -> dict[str, Any]:
        names = _STAGE_ARTIFACTS[stage]
        paths = _artifact_paths(self.config, sorted(names))
        if not self.dry_run:
            t7_gate = self._require_t7_gate()
        else:
            t7_gate = None
        if stage == "render":
            return {"artifacts": {}, "t7_gate": t7_gate}
        required = {"split_manifest", "cohort_status", "prepared_cases"}
        if not required.issubset(paths):
            return {"artifacts": {name: _file_hash(path) for name, path in paths.items()}, "t7_gate": t7_gate}
        cases = load_prepared_cases(paths["prepared_cases"])
        with paths["split_manifest"].open(newline="", encoding="utf-8") as handle:
            manifest = list(csv.DictReader(handle))
        if not manifest:
            raise PreflightError("split_manifest.csv has no assigned subjects.")
        manifest_ids = {str(row.get("subject_id", "")) for row in manifest}
        absent = sorted(manifest_ids - set(cases))
        if absent:
            raise PreflightError("Missing required prepared FoldCaseInput artifacts: " + ", ".join(absent))
        extra = sorted(set(cases) - manifest_ids)
        if extra:
            raise PreflightError("Prepared cases include subjects outside split manifest: " + ", ".join(extra))
        by_subject = {str(row["subject_id"]): row for row in manifest}
        if len(by_subject) != len(manifest):
            raise PreflightError("split_manifest.csv contains duplicate subject IDs.")
        mismatched: list[str] = []
        for subject_id, case in cases.items():
            row = by_subject[subject_id]
            for field in ("dataset_id", "partition", "task", "source_group_id", "channel", "language", "role", "fold_id", "source_version"):
                if str(row.get(field, "")) != str(getattr(case.subject, field)):
                    mismatched.append(f"{subject_id}:{field}")
            for field in ("class_order", "task_ids", "raw_hashes"):
                try:
                    source_value = json.loads(str(row[field]))
                except (KeyError, TypeError, json.JSONDecodeError):
                    mismatched.append(f"{subject_id}:{field}")
                    continue
                target = getattr(case.subject, field)
                if source_value != (dict(target) if isinstance(target, Mapping) else list(target)):
                    mismatched.append(f"{subject_id}:{field}")
        if mismatched:
            raise PreflightError(
                "Prepared FoldCaseInput metadata differs from split manifest: " + ", ".join(sorted(mismatched))
            )
        with paths["cohort_status"].open(newline="", encoding="utf-8") as handle:
            status_rows = list(csv.DictReader(handle))
        status_ids = [str(row.get("cohort_id", "")) for row in status_rows]
        if not status_ids or any(not cohort_id for cohort_id in status_ids):
            raise PreflightError("cohort_status.csv contains an empty cohort ID.")
        if len(set(status_ids)) != len(status_ids):
            raise PreflightError("cohort_status.csv contains duplicate cohort statuses; blocked status cannot be overwritten.")
        statuses = {str(row["cohort_id"]): row for row in status_rows}
        datasets = {case.subject.dataset_id for case in cases.values()}
        unavailable = sorted(dataset for dataset in datasets if dataset not in statuses)
        if unavailable:
            raise PreflightError("Missing cohort status for: " + ", ".join(unavailable))
        ready = sorted(dataset for dataset in datasets if statuses[dataset].get("status") == "ready")
        blocked = sorted(dataset for dataset, row in statuses.items() if row.get("status") != "ready")
        if set(blocked) & datasets:
            # A blocked dataset must have been excluded by T1. Its status is kept in
            # the record, but it cannot silently enter any later training stage.
            included_blocked = sorted(
                dataset for dataset in blocked
                if any(case.subject.dataset_id == dataset for case in cases.values())
            )
            if included_blocked:
                raise PreflightError("Blocked cohorts must not have prepared cases: " + ", ".join(included_blocked))
        if stage in {"canary", "assess-oof", "predict-holdout", "stress"}:
            if self.provider == "configured" and not self.allow_paid:
                raise PreflightError("Configured provider calls require --allow-paid.")
        return {"artifacts": {name: _file_hash(path) for name, path in paths.items()}, "t7_gate": t7_gate,
                "prepared_case_count": len(cases), "manifest_subject_count": len(manifest),
                "ready_cohorts": ready, "blocked_cohorts": blocked}

    def _run_actual_stage(self, stage: str) -> Mapping[str, Any]:
        preflight = self._preflight(stage)
        # This worktree has no injected predictor/provider orchestration. Do not
        # claim successful analysis merely because immutable inputs passed checks.
        return {
            "stage_status": "blocked", "reason": "no_reviewed_stage_adapter",
            "preflight": preflight, "analytical": False,
        }

    def _validate_resume_record(self, stage: str, record: Mapping[str, Any], input_hash: str) -> None:
        expected_mode = "dry_run" if self.dry_run else "real"
        if record.get("schema_version") != RUNNER_VERSION or record.get("stage") != stage:
            raise RunnerError(f"Resume record schema or stage mismatch for {stage}.")
        if record.get("status") not in {"succeeded", "dry_run"}:
            raise RunnerError(f"Resume record for {stage} is not a completed stage.")
        if record.get("execution_mode") != expected_mode or record.get("provider") != self.provider:
            raise RunnerError(f"Resume dependency modality mismatch for stage {stage}.")
        if record.get("code_sha") != _git_sha() or record.get("config_hash") != self._config_hash():
            raise RunnerError(f"Resume code/config SHA mismatch for stage {stage}.")
        output = record.get("output")
        if not isinstance(output, Mapping) or record.get("output_hash") != _hash(output):
            raise RunnerError(f"Resume output hash mismatch for stage {stage}.")
        if record.get("dependencies") != self._dependency_snapshot(stage):
            raise RunnerError(f"Resume dependency modality mismatch for stage {stage}.")
        if record.get("input_hash") != input_hash:
            raise RunnerError(f"Resume hash mismatch for stage {stage}; refuse stale output reuse.")

    def run_stage(self, stage: str) -> dict[str, Any]:
        if stage not in STAGES:
            raise RunnerError(f"Unknown pilot stage: {stage}.")
        input_hash = self._input_hash(stage)
        existing_path = self.record_path(stage)
        if existing_path.is_file() and self.resume:
            existing = _read_json(existing_path)
            self._validate_resume_record(stage, existing, input_hash)
            return existing
        started = _utc_now()
        blockers = self._dependency_blockers(stage)
        if blockers:
            record = self._stage_record(stage, input_hash, started, "blocked", {
                "stage_status": "blocked", "reason": "dependencies_incomplete", "dependencies": blockers,
            })
            _atomic_json(existing_path, record)
            return record
        status = "dry_run" if self.dry_run else "failed"
        output: Mapping[str, Any]
        try:
            if self.dry_run:
                output = {
                    "structural_only": True, "analytical": False,
                    "provider_stamp": "TEST_ONLY_FAKE_PROVIDER" if self.provider == "fake" else "NO_PROVIDER_CALLS",
                    "label_capabilities": sorted(LABEL_CAPABILITIES[stage]),
                }
            else:
                output = self._run_actual_stage(stage)
                status = str(output.get("stage_status", "failed"))
                if status == "succeeded" and output.get("analytical") is not True:
                    raise RunnerError("A real stage may succeed only with materialized analytical output.")
                if status not in {"succeeded", "blocked", "incomplete"}:
                    raise RunnerError("Real stage returned an invalid completion status.")
        except Exception as exc:
            output = {"error": type(exc).__name__, "detail": str(exc)}
            record = self._stage_record(stage, input_hash, started, "failed", output)
            _atomic_json(existing_path, record)
            raise
        record = self._stage_record(stage, input_hash, started, status, output)
        _atomic_json(existing_path, record)
        return record

    def _stage_record(self, stage: str, input_hash: str, started: str, status: str,
                      output: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "schema_version": RUNNER_VERSION, "stage": stage, "status": status,
            "input_hash": input_hash, "output_hash": _hash(output), "code_sha": _git_sha(),
            "started_at": started, "ended_at": _utc_now(), "row_counts": {},
            "config_hash": self._config_hash(), "provider": self.provider,
            "execution_mode": "dry_run" if self.dry_run else "real",
            "dependencies": self._dependency_snapshot(stage),
            "label_capabilities": sorted(LABEL_CAPABILITIES[stage]), "output": _plain(output),
        }

    def run(self, stage: str = "all") -> list[dict[str, Any]]:
        selected = STAGES if stage == "all" else (stage,)
        return [self.run_stage(item) for item in selected]


def run_stage(stage: str, resolved_config: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Contract-level single-stage entry point; paid execution remains disabled."""

    return PilotRunner(resolved_config, Path(run_dir), dry_run=True, provider="fake").run_stage(stage)


__all__ = [
    "DiskAssessmentCache", "LABEL_CAPABILITIES", "PilotRunner", "PreparedCase", "PreflightError",
    "PREDICTION_LOCK_SCHEMA", "RUNNER_VERSION", "RunnerError", "STAGES", "STAGE_DEPENDENCIES",
    "load_prepared_cases", "paired_accuracy_summary", "paired_bootstrap", "prediction_lock", "run_stage",
    "score_locked_predictions",
]
