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


RUNNER_VERSION = "advoice.pilot.runner.v1"
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


class RunnerError(RuntimeError):
    """A stage cannot safely run, resume, or publish an analytical artifact."""


class PreflightError(RunnerError):
    """Required immutable inputs or authorization evidence are absent."""


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False).encode("utf-8")


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
            ("git", "rev-parse", "HEAD"), text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return "unavailable"


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise PreflightError(f"Resolved config requires mapping: {name}.")
    return value


def _artifact_paths(config: Mapping[str, Any]) -> dict[str, Path]:
    raw_artifacts = config.get("artifacts", {})
    artifacts = _mapping(raw_artifacts, "artifacts")
    required = {
        "split_manifest": "split_manifest.csv",
        "cohort_status": "cohort_status.csv",
        "prepared_cases": "prepared_cases.jsonl",
        "development_labels": "development_labels.jsonl",
        "holdout_labels": "sealed_holdout_labels.jsonl",
        "stress_labels": "sealed_stress_labels.jsonl",
    }
    result: dict[str, Path] = {}
    missing: list[str] = []
    for key, expected_name in required.items():
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
            rows[f"precision_{label}"] = None
            rows[f"precision_{label}_reason"] = "predicted_class_missing"
        else:
            rows[f"precision_{label}"] = true_positive / predicted_positive
        precision, recall = rows[f"precision_{label}"], rows[f"recall_{label}"]
        if isinstance(precision, float) and isinstance(recall, float) and precision + recall:
            rows[f"f1_{label}"] = 2 * precision * recall / (precision + recall)
        else:
            rows[f"f1_{label}"] = None
            rows[f"f1_{label}_reason"] = "precision_or_recall_undefined"
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
    return {
        "n": len(truth), "helped": helped, "harmed": harmed,
        "unchanged_correct": sum(base and arm for base, arm in zip(base_correct, arm_correct, strict=True)),
        "unchanged_incorrect": sum(not base and not arm for base, arm in zip(base_correct, arm_correct, strict=True)),
        "bootstrap": paired_bootstrap(base_correct, arm_correct, seed=seed),
        "mcnemar": {
            "base_wrong_arm_right": helped, "base_right_arm_wrong": harmed,
            "p_value": float(binomtest(min(helped, harmed), n=discordant, p=0.5).pvalue) if discordant else 1.0,
        },
    }


def score_locked_predictions(
    rows: Sequence[Mapping[str, Any]], labels: Mapping[str, str], *, seed: int,
) -> dict[str, Any]:
    """Score immutable predictions. Callers must supply labels only at score stage."""

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
        if arm not in {"B_raw", "B", "J-A", "J-S", "J-AS"} or len(order) != len(probabilities):
            raise RunnerError("Locked prediction arm or class order is invalid.")
        if subject_id not in labels:
            raise RunnerError(f"Scorer has no authorized label for subject: {subject_id}.")
        grouped.setdefault((str(row.get("dataset_id", "unspecified")), arm, order), []).append(row)
    output: dict[str, Any] = {"metrics": [], "paired": []}
    by_dataset_subject: dict[str, dict[str, dict[str, Mapping[str, Any]]]] = {}
    for (dataset, arm, order), group in sorted(grouped.items()):
        subject_ids = [str(item["subject_id"]) for item in group]
        if len(set(subject_ids)) != len(subject_ids):
            raise RunnerError("Locked predictions duplicate an assigned subject within an arm.")
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
        for arm in ("J-A", "J-S", "J-AS"):
            ids = sorted(subject_id for subject_id, arms in subjects.items() if "B" in arms and arm in arms)
            if not ids:
                continue
            truth = [labels[subject_id] for subject_id in ids]
            base = [str(subjects[subject_id]["B"]["predicted"]) for subject_id in ids]
            candidate = [str(subjects[subject_id][arm]["predicted"]) for subject_id in ids]
            output["paired"].append({"dataset_id": dataset, "arm": arm, **paired_accuracy_summary(truth, base, candidate, seed=seed)})
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

    @property
    def records_dir(self) -> Path:
        return self.run_dir / "stages"

    def record_path(self, stage: str) -> Path:
        return self.records_dir / f"{stage}.json"

    def _input_hash(self, stage: str) -> str:
        config = dict(self.config)
        artifacts = config.get("artifacts", {})
        artifact_hashes = {}
        if isinstance(artifacts, Mapping):
            for name, raw in sorted(artifacts.items()):
                path = Path(raw).expanduser() if isinstance(raw, str) else None
                artifact_hashes[str(name)] = _file_hash(path) if path is not None and path.is_file() else None
        dependency_hashes = {}
        for dependency in STAGE_DEPENDENCIES[stage]:
            record = self.record_path(dependency)
            dependency_hashes[dependency] = _file_hash(record) if record.is_file() else None
        return _hash({"runner_version": RUNNER_VERSION, "stage": stage, "config": config,
                      "artifacts": artifact_hashes, "dependencies": dependency_hashes,
                      "provider": self.provider, "dry_run": self.dry_run})

    def _require_dependencies(self, stage: str) -> None:
        missing = []
        for dependency in STAGE_DEPENDENCIES[stage]:
            path = self.record_path(dependency)
            if not path.is_file():
                missing.append(dependency)
                continue
            record = _read_json(path)
            if record.get("status") not in {"succeeded", "dry_run"}:
                missing.append(dependency)
        if missing:
            raise RunnerError(f"Stage {stage} requires completed dependencies: {', '.join(missing)}.")

    def _preflight(self, stage: str) -> dict[str, Any]:
        paths = _artifact_paths(self.config)
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
            statuses = {str(row.get("cohort_id")): row for row in csv.DictReader(handle)}
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
        return {"artifacts": {name: _file_hash(path) for name, path in paths.items()},
                "prepared_case_count": len(cases), "manifest_subject_count": len(manifest),
                "ready_cohorts": ready, "blocked_cohorts": blocked}

    def _run_actual_stage(self, stage: str) -> Mapping[str, Any]:
        preflight = self._preflight(stage)
        # Phase A intentionally provides no configured provider construction. The
        # following artifact validation is useful before T7, while an actual call
        # remains impossible without T7's reviewed, injected provider adapter.
        if stage in {"canary", "assess-oof"}:
            raise PreflightError(
                "Configured provider adapter is not constructed by Phase A; T7 gate and a reviewed adapter are required."
            )
        return {"preflight": preflight, "analytical_output": "not_materialized"}

    def run_stage(self, stage: str) -> dict[str, Any]:
        if stage not in STAGES:
            raise RunnerError(f"Unknown pilot stage: {stage}.")
        self._require_dependencies(stage)
        input_hash = self._input_hash(stage)
        existing_path = self.record_path(stage)
        if existing_path.is_file() and self.resume:
            existing = _read_json(existing_path)
            if existing.get("status") in {"succeeded", "dry_run"}:
                if existing.get("input_hash") != input_hash:
                    raise RunnerError(f"Resume hash mismatch for stage {stage}; refuse stale output reuse.")
                return existing
        started = _utc_now()
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
                status = "succeeded"
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
            "label_capabilities": sorted(LABEL_CAPABILITIES[stage]), "output": dict(output),
        }

    def run(self, stage: str = "all") -> list[dict[str, Any]]:
        selected = STAGES if stage == "all" else (stage,)
        return [self.run_stage(item) for item in selected]


def run_stage(stage: str, resolved_config: Mapping[str, Any], run_dir: str | Path) -> dict[str, Any]:
    """Contract-level single-stage entry point; paid execution remains disabled."""

    return PilotRunner(resolved_config, Path(run_dir), dry_run=True, provider="fake").run_stage(stage)


__all__ = [
    "DiskAssessmentCache", "LABEL_CAPABILITIES", "PilotRunner", "PreparedCase", "PreflightError",
    "RUNNER_VERSION", "RunnerError", "STAGES", "STAGE_DEPENDENCIES", "load_prepared_cases",
    "paired_accuracy_summary", "paired_bootstrap", "run_stage", "score_locked_predictions",
]
