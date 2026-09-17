"""End-to-end, label-isolated authority joint-fusion cohort studies.

The study runner deliberately separates inference from evaluation.  A case is
prepared without its outcome label, reviewed exactly once by the supplied
two-pass runtime, compiled into one atomic transaction, replayed through the
frozen Module A state expert, and only then combined with the immutable
Condition C prediction and the first-pass blind Agent likelihood.  Labels are
read after every requested prediction is complete.

Provider, schema, compilation, replay, and fusion failures are recorded as
explicit failed cases.  The frozen Condition C prediction is computed before
each provider call and retained for the entire requested cohort, while a
failed case never receives a synthetic fused prediction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol, Sequence

import pandas as pd

from .authority_review_runtime import AuthorityReviewResult, AuthorityReviewRuntime, REVIEW_AVAILABLE
from .authority_review_transaction_bridge import compile_authority_review_decision
from .authority_study_dataset import AuthorityStudyDataset, PreparedAuthorityStudyCase
from .authority_joint_fusion import (
    AUTHORITY_JOINT_FUSION_SCHEMA_VERSION,
    AuthorityJointFusionConfig,
    AuthorityJointFusionResult,
    fuse_authority_joint,
)
from .decision_lock import canonical_json, hash_artifact
from .evaluation import evaluate_predictions
from .evidence_replay import replay_evidence
from .module_a import ExplanationPacket
from .utils import hash_values


STUDY_SCHEMA_VERSION = "advoice.authority_joint_fusion_study.v1"
DEFAULT_JOINT_FUSION_CONFIG = AuthorityJointFusionConfig(
    state_strength=1.0,
    agent_strength=1.0,
    max_abs_state_delta=0.75,
    ordinal_temperature=1.0,
)


class AuthorityStateDeltaStudyError(RuntimeError):
    """Raised when a cohort study cannot preserve its inference contract."""


class ReviewRuntime(Protocol):
    """Injectable boundary for a real or test two-pass review runtime."""

    cache_dir: Path

    def review(
        self,
        prepared: Any,
        *,
        transcript: Mapping[str, Any] | str | None = None,
    ) -> AuthorityReviewResult: ...


@dataclass(frozen=True, slots=True)
class AuthorityStateDeltaStudyConfig:
    """Predeclared study controls that are never selected from test outcomes."""

    joint_fusion: AuthorityJointFusionConfig = field(
        default_factory=lambda: DEFAULT_JOINT_FUSION_CONFIG
    )
    evaluation_bins: int = 10
    selection_order: str = "longest_first"
    selection_salt: str = "authority-pilot-v1"
    max_cases: int | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.joint_fusion, AuthorityJointFusionConfig):
            raise TypeError("joint_fusion must be an AuthorityJointFusionConfig.")
        if self.evaluation_bins < 2:
            raise ValueError("evaluation_bins must be at least 2.")
        if self.selection_order not in {"longest_first", "subject_id", "stable_hash"}:
            raise ValueError(
                "selection_order must be 'longest_first', 'subject_id', or 'stable_hash'."
            )
        if not str(self.selection_salt).strip():
            raise ValueError("selection_salt must be non-empty.")
        if self.max_cases is not None and self.max_cases < 1:
            raise ValueError("max_cases must be positive when supplied.")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": STUDY_SCHEMA_VERSION,
            "fusion_schema_version": AUTHORITY_JOINT_FUSION_SCHEMA_VERSION,
            "joint_fusion": self.joint_fusion.to_dict(),
            "evaluation_bins": self.evaluation_bins,
            "selection_order": self.selection_order,
            "selection_salt": self.selection_salt,
            "max_cases": self.max_cases,
        }


@dataclass(frozen=True, slots=True)
class AuthorityStateDeltaStudyResult:
    """Paths and metrics emitted by one resumable cohort study."""

    output_dir: Path
    audit_jsonl_path: Path
    audit_json_path: Path
    aggregate_json_path: Path
    attempted_case_ids: tuple[str, ...]
    completed_case_ids: tuple[str, ...]
    failed_case_ids: tuple[str, ...]
    # Full-queue frozen baseline and completed-case fused metrics.  Their
    # comparable counterparts are explicit in ``aggregate.json`` under the
    # ``paired_*`` keys.
    frozen_metrics: Mapping[str, Any] | None
    fused_metrics: Mapping[str, Any] | None
    paired_counts: Mapping[str, int]
    study_hash: str


def build_authority_review_runtime(
    *,
    root: str | Path,
    provider: str | None,
    model: str,
    cache_dir: str | Path,
    skill_path: str | Path | None = None,
) -> AuthorityReviewRuntime:
    """Construct the production runtime with an explicit resumable cache path."""

    return AuthorityReviewRuntime(
        root=Path(root),
        provider=provider,
        model=model,
        skill_path=None if skill_path is None else Path(skill_path),
        cache_dir=Path(cache_dir),
    )


def run_authority_state_delta_cohort(
    dataset: AuthorityStudyDataset,
    runtime: ReviewRuntime,
    *,
    output_dir: str | Path,
    config: AuthorityStateDeltaStudyConfig | None = None,
    cache_dir: str | Path | None = None,
    decision_only: bool = True,
) -> AuthorityStateDeltaStudyResult:
    """Run a deterministic prepared test cohort without outcome-aware tuning.

    ``runtime`` is injected so tests can use a no-network fake.  Production
    callers should use :func:`build_authority_review_runtime`, which passes
    ``cache_dir`` directly into :class:`AuthorityReviewRuntime`.

    This evaluation path is decision-only: both blind assessment and advisor
    reconciliation are decision-relevant and remain enabled. Structured
    rationales/report_trace are audit data, not clinician report generation.
    Report requests fail before any I/O; that workflow is deferred until
    formal testing. This flag does not disable classification provider calls.
    """

    if decision_only is not True:
        raise ValueError("Clinician report generation is deferred until formal testing; decision_only must be True.")
    if not isinstance(dataset, AuthorityStudyDataset):
        raise TypeError("dataset must be an AuthorityStudyDataset.")
    selected_config = config or AuthorityStateDeltaStudyConfig()
    if not isinstance(selected_config, AuthorityStateDeltaStudyConfig):
        raise TypeError("config must be an AuthorityStateDeltaStudyConfig.")

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    if cache_dir is not None:
        _verify_runtime_cache(runtime, Path(cache_dir).expanduser().resolve())

    prepared_cases = dataset.prepare_test_cases(
        max_cases=selected_config.max_cases,
        order=selected_config.selection_order,
        selection_salt=selected_config.selection_salt,
    )
    if not prepared_cases:
        raise AuthorityStateDeltaStudyError("The selected cohort has no prepared test cases.")
    _assert_unique_case_ids(prepared_cases)

    study_hash = _study_hash(dataset, prepared_cases, selected_config)
    audit_jsonl_path = root / "case_audit.jsonl"
    audit_json_path = root / "case_audit.json"
    aggregate_json_path = root / "aggregate.json"
    existing = _load_resumable_audits(audit_jsonl_path, study_hash)

    audits: dict[str, dict[str, Any]] = {}
    for prepared_case in prepared_cases:
        case_id = prepared_case.prepared_case.case_id
        cached = existing.get(case_id)
        if cached is not None and cached.get("status") == "completed":
            audits[case_id] = cached
            continue
        audit = _run_case(
            dataset=dataset,
            runtime=runtime,
            study_config=selected_config,
            prepared_case=prepared_case,
            study_hash=study_hash,
        )
        audits[case_id] = audit
        _append_jsonl(audit_jsonl_path, audit)

    # This is intentionally the first label access in the runner.  At this
    # point every requested provider call, compilation, replay, and fusion has
    # already completed or been recorded as an explicit failure.
    truth = dataset.evaluation_truth(tuple(audits))
    aggregate = _aggregate(
        audits=audits,
        truth=truth,
        class_order=dataset.class_order,
        config=selected_config,
        study_hash=study_hash,
    )
    _write_json(audit_json_path, {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_hash": study_hash,
        "cases": [audits[case_id] for case_id in sorted(audits)],
    })
    _write_json(aggregate_json_path, aggregate)

    return AuthorityStateDeltaStudyResult(
        output_dir=root,
        audit_jsonl_path=audit_jsonl_path,
        audit_json_path=audit_json_path,
        aggregate_json_path=aggregate_json_path,
        attempted_case_ids=tuple(sorted(audits)),
        completed_case_ids=tuple(aggregate["completed_case_ids"]),
        failed_case_ids=tuple(aggregate["failed_case_ids"]),
        frozen_metrics=aggregate["frozen_metrics"],
        fused_metrics=aggregate["fused_metrics"],
        paired_counts=aggregate["paired_counts"],
        study_hash=study_hash,
    )


def _run_case(
    *,
    dataset: AuthorityStudyDataset,
    runtime: ReviewRuntime,
    study_config: AuthorityStateDeltaStudyConfig,
    prepared_case: PreparedAuthorityStudyCase,
    study_hash: str,
) -> dict[str, Any]:
    prepared = prepared_case.prepared_case
    base = _audit_base(prepared, study_config, study_hash)
    frozen_packet: ExplanationPacket | None = None
    try:
        # Freeze the non-Agent prediction before any provider interaction.
        # This keeps the baseline on the full predeclared test queue even when
        # the Agent later fails.  It is inference-only and has no label access.
        frozen_packet = dataset.advisor.explain_subject(
            prepared.case_id,
            evidence_snapshot=prepared.pre_replay.revised_evidence,
        )
        base["frozen"] = _packet_audit(frozen_packet)
        base["provenance"] = {
            "frozen_packet_hash": _packet_hash(frozen_packet),
        }

        # Exactly one runtime invocation per non-resumed case.  The runtime
        # itself owns the two required structured provider requests.
        review = runtime.review(prepared, transcript=prepared_case.transcript)
        base["review"] = _review_audit(review)
        if review.status != REVIEW_AVAILABLE:
            raise AuthorityStateDeltaStudyError(
                f"Authority review ended with explicit status {review.status!r}: {review.error or ''}".strip()
            )

        compiled = compile_authority_review_decision(prepared, review)
        transaction = compiled.transaction
        pre_replay = prepared.pre_replay
        assert frozen_packet is not None
        if transaction is None:
            post_replay = pre_replay
            revision_hash = pre_replay.audit.revision_hash
            revision_action = "no_revision"
        else:
            post_replay = replay_evidence(
                prepared.module_a_evidence,
                transaction,
                states_config=dataset.executor.states_config,
                module_a=dataset.executor.module_a,
                case_context=prepared.case_context,
                correlation_config=dataset.executor.correlation_config,
                dataset_id=str(prepared.case_metadata.get("dataset_id", dataset.advisor.dataset_id)),
                label="unknown",
                split="inference",
            )
            revision_hash = transaction.transaction_hash
            revision_action = "multi_state_evidence_review"

        blind_assessment = review.blind_assessment
        if blind_assessment is None:
            raise AuthorityStateDeltaStudyError(
                "Available authority review has no validated blind assessment."
            )
        fusion = fuse_authority_joint(
            _packet_probabilities(frozen_packet),
            _packet_probabilities(pre_replay.packet),
            _packet_probabilities(post_replay.packet),
            blind_assessment.ordinal_scores,
            class_order=frozen_packet.class_order,
            config=study_config.joint_fusion,
            channel=str(prepared.case_metadata.get("channel", "unknown")),
            provenance={
                "frozen_packet_hash": _packet_hash(frozen_packet),
                "pre_packet_hash": _packet_hash(pre_replay.packet),
                "post_packet_hash": _packet_hash(post_replay.packet),
                "blind_request_hash": review.blind_request_hash,
                "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
                "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
                "revision_hash": revision_hash,
                "revision_action": revision_action,
            },
        )
        if fusion.frozen_parity and not _bits_equal(
            fusion.fused_probabilities, _packet_probabilities(frozen_packet)
        ):
            raise AuthorityStateDeltaStudyError(
                "Neutral joint fusion did not preserve frozen Condition C probabilities bit-for-bit."
            )
        base.update({
            "status": "completed",
            "transaction": None if transaction is None else transaction.to_dict(),
            "decision_hash": compiled.decision.decision_hash,
            "pre_state": _packet_audit(pre_replay.packet),
            "post_state": _packet_audit(post_replay.packet),
            "fusion": _fusion_audit(fusion),
            "provenance": {
                "frozen_packet_hash": _packet_hash(frozen_packet),
                "pre_packet_hash": _packet_hash(pre_replay.packet),
                "post_packet_hash": _packet_hash(post_replay.packet),
                "pre_replay_audit_hash": pre_replay.audit.audit_hash,
                "post_replay_audit_hash": post_replay.audit.audit_hash,
                "revision_hash": revision_hash,
                "transaction_hash": None if transaction is None else transaction.transaction_hash,
            },
        })
    except Exception as error:
        base.update({
            "status": "failed",
            "failure": {
                "type": type(error).__name__,
                "message": str(error),
            },
        })
    return base


def _packet_hash(packet: ExplanationPacket) -> str:
    return hash_values([packet.to_json()])


def _packet_bound_hash(packet: ExplanationPacket, aliases: Sequence[str]) -> str:
    values = {str(packet.hashes[name]) for name in aliases if name in packet.hashes}
    if len(values) != 1:
        raise AuthorityStateDeltaStudyError(
            "Packet hash aliases must expose one consistent value among "
            f"{list(aliases)}."
        )
    return next(iter(values))


def _packet_probabilities(packet: ExplanationPacket) -> Mapping[str, float]:
    return packet.calibrated_probabilities or packet.raw_probabilities


def _bits_equal(left: Mapping[str, float], right: Mapping[str, float]) -> bool:
    # JSON-compatible float hex representations preserve exact IEEE-754 values
    # without importing an additional numeric package.
    return tuple(float(left[key]).hex() for key in left) == tuple(float(right[key]).hex() for key in right)


def _audit_base(prepared: Any, config: AuthorityStateDeltaStudyConfig, study_hash: str) -> dict[str, Any]:
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_hash": study_hash,
        "case_id": prepared.case_id,
        "prepared": {
            "reviewed_packet_hash": prepared.reviewed_packet_hash,
            "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
            "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
            "advisor_packet_hash": prepared.advisor_packet_hash,
            "route": prepared.route.target_route.id,
            "class_order": list(prepared.route.target_route.labels),
        },
        "joint_fusion": config.joint_fusion.to_dict(),
    }


def _review_audit(review: AuthorityReviewResult) -> dict[str, Any]:
    return {
        "status": review.status,
        "case_pseudonym": review.case_id,
        "blind_request_hash": review.blind_request_hash,
        "reconciliation_request_hash": review.reconciliation_request_hash,
        "cache_key": review.cache_key,
        "error": review.error,
    }


def _packet_audit(packet: ExplanationPacket) -> dict[str, Any]:
    probabilities = _packet_probabilities(packet)
    return {
        "packet_hash": _packet_hash(packet),
        "predicted_label": packet.predicted_label,
        "probabilities": {label: float(probabilities[label]) for label in packet.class_order},
        "hashes": dict(packet.hashes),
    }


def _fusion_audit(fusion: AuthorityJointFusionResult) -> dict[str, Any]:
    return {
        "schema_version": fusion.schema_version,
        "audit_hash": fusion.audit_hash,
        "input_hash": fusion.input_hash,
        "predicted_label": fusion.predicted_label,
        "probabilities": dict(fusion.fused_probabilities),
        "correction_applied": not fusion.frozen_parity,
        "frozen_parity": fusion.frozen_parity,
        "state_component_neutral": fusion.state_component_neutral,
        "state_authority_gate": fusion.state_authority_gate,
        "agent_component_neutral": fusion.agent_component_neutral,
        "agent_authority_gate": fusion.agent_authority_gate,
        "state_agent_conflict": fusion.state_agent_conflict,
        "state_delta": dict(fusion.state_log_evidence),
        "clipped_state_delta": dict(fusion.bounded_state_log_evidence),
        "agent_log_evidence": dict(fusion.agent_log_evidence),
        "agent_equal_prior_likelihood": dict(fusion.agent_equal_prior_likelihood),
        "blind_ordinal_scores": dict(fusion.blind_ordinal_scores),
        "config": fusion.config.to_dict(),
        "provenance": dict(fusion.provenance),
    }


def _aggregate(
    *,
    audits: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
    class_order: Sequence[str],
    config: AuthorityStateDeltaStudyConfig,
    study_hash: str,
) -> dict[str, Any]:
    completed = sorted(case_id for case_id, audit in audits.items() if audit.get("status") == "completed")
    failed = sorted(case_id for case_id, audit in audits.items() if audit.get("status") != "completed")
    labels = list(class_order)
    frozen_available = sorted(case_id for case_id, audit in audits.items() if "frozen" in audit)
    frozen_missing = sorted(set(audits) - set(frozen_available))
    frozen_metrics: Mapping[str, Any] | None = None
    paired_frozen_metrics: Mapping[str, Any] | None = None
    fused_metrics: Mapping[str, Any] | None = None
    paired = {"changed": 0, "unchanged": 0, "corrected": 0, "harmed": 0}

    # The immutable baseline is evaluated over every requested case for which
    # it was available before Agent execution.  A provider failure must not
    # remove the case from this baseline.
    if frozen_available:
        frozen_frame = _prediction_frame(audits, truth, frozen_available, "frozen", labels)
        frozen_metrics = _cohort_metrics(
            frozen_frame, bins=config.evaluation_bins, labels=labels, positive=labels[-1],
        )

    # Only cases with a genuine Agent transaction/replay/fusion are eligible
    # for a paired frozen-vs-fused comparison.  Failed cases retain frozen
    # output above but never receive a substituted fused prediction.
    if completed:
        frozen_frame = _prediction_frame(audits, truth, completed, "frozen", labels)
        fused_frame = _prediction_frame(audits, truth, completed, "fusion", labels)
        positive = labels[-1]
        paired_frozen_metrics = _cohort_metrics(
            frozen_frame, bins=config.evaluation_bins, labels=labels, positive=positive,
        )
        fused_metrics = _cohort_metrics(
            fused_frame, bins=config.evaluation_bins, labels=labels, positive=positive,
        )
        for case_id in completed:
            actual = str(truth[case_id])
            frozen_label = str(audits[case_id]["frozen"]["predicted_label"])
            fused_label = str(audits[case_id]["fusion"]["predicted_label"])
            if frozen_label == fused_label:
                paired["unchanged"] += 1
            else:
                paired["changed"] += 1
                if frozen_label != actual and fused_label == actual:
                    paired["corrected"] += 1
                if frozen_label == actual and fused_label != actual:
                    paired["harmed"] += 1
    return {
        "schema_version": STUDY_SCHEMA_VERSION,
        "study_hash": study_hash,
        "config": config.to_dict(),
        "attempted_case_ids": sorted(audits),
        "completed_case_ids": completed,
        "failed_case_ids": failed,
        "failed_case_count": len(failed),
        "agent_coverage": {
            "attempted_case_count": len(audits),
            "completed_case_count": len(completed),
            "failed_case_count": len(failed),
            "coverage_rate": _rate(len(completed), len(audits)),
            "failure_rate": _rate(len(failed), len(audits)),
        },
        "frozen_full_queue_case_ids": frozen_available,
        "frozen_full_queue_missing_case_ids": frozen_missing,
        "frozen_full_queue_complete": len(frozen_missing) == 0,
        "paired_case_ids": completed,
        # Kept for consumers of the original result object.  It now denotes
        # the full-queue frozen baseline, not a complete-case baseline.
        "metrics_cohort_case_ids": frozen_available,
        "metrics_exclude_failed_cases": False,
        "frozen_metrics": frozen_metrics,
        "paired_frozen_metrics": paired_frozen_metrics,
        "fused_metrics": fused_metrics,
        "paired_fused_metrics": fused_metrics,
        "paired_counts": paired,
    }


def _rate(numerator: int, denominator: int) -> float | None:
    if denominator == 0:
        return None
    return float(numerator / denominator)


def _cohort_metrics(
    frame: pd.DataFrame, *, bins: int, labels: list[str], positive: str,
) -> Mapping[str, Any]:
    observed = sorted(set(frame["label"].astype(str)))
    if len(frame) < 2 or len(observed) < 2:
        predicted = frame["predicted_label"].astype(str)
        actual = frame["label"].astype(str)
        matrix = [
            [int(((actual == truth) & (predicted == guess)).sum()) for guess in labels]
            for truth in labels
        ]
        return {
            "n": int(len(frame)),
            "accuracy": float((actual == predicted).mean()),
            "evaluation_status": "insufficient_class_coverage",
            "observed_classes": observed,
            "macro_f1": None,
            "micro_f1": None,
            "weighted_f1": None,
            "macro_auroc_ovr": None,
            "micro_auroc_ovr": None,
            "weighted_auroc_ovr": None,
            "confusion_matrix": matrix,
        }
    result = dict(evaluate_predictions(frame, bins, labels, positive))
    result["evaluation_status"] = "complete"
    result["observed_classes"] = observed
    return result


def _prediction_frame(
    audits: Mapping[str, Mapping[str, Any]],
    truth: Mapping[str, str],
    case_ids: Sequence[str],
    key: str,
    labels: Sequence[str],
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for case_id in case_ids:
        block = audits[case_id][key]
        row: dict[str, Any] = {
            "case_id": case_id,
            "label": truth[case_id],
            "predicted_label": block["predicted_label"],
        }
        for label in labels:
            row[f"prob_{label}"] = block["probabilities"][label]
        rows.append(row)
    return pd.DataFrame(rows)


def _study_hash(
    dataset: AuthorityStudyDataset,
    cases: Sequence[PreparedAuthorityStudyCase],
    config: AuthorityStateDeltaStudyConfig,
) -> str:
    return hash_artifact({
        "schema_version": STUDY_SCHEMA_VERSION,
        "dataset_id": dataset.advisor.dataset_id,
        "class_order": list(dataset.class_order),
        "config": config.to_dict(),
        "cases": [
            {
                "case_id": item.prepared_case.case_id,
                "reviewed_packet_hash": item.prepared_case.reviewed_packet_hash,
                "reviewed_evidence_hash": item.prepared_case.reviewed_evidence_hash,
                "reviewed_state_graph_hash": item.prepared_case.reviewed_state_graph_hash,
            }
            for item in cases
        ],
    })


def _assert_unique_case_ids(cases: Sequence[PreparedAuthorityStudyCase]) -> None:
    identifiers = [item.prepared_case.case_id for item in cases]
    if len(identifiers) != len(set(identifiers)):
        raise AuthorityStateDeltaStudyError("Prepared cohort has duplicate case IDs.")


def _verify_runtime_cache(runtime: ReviewRuntime, expected: Path) -> None:
    actual = getattr(runtime, "cache_dir", None)
    if actual is None:
        raise AuthorityStateDeltaStudyError("A study cache_dir requires a runtime exposing cache_dir.")
    if Path(actual).expanduser().resolve() != expected:
        raise AuthorityStateDeltaStudyError(
            "Provided cache_dir does not match the AuthorityReviewRuntime cache_dir."
        )


def _load_resumable_audits(path: Path, study_hash: str) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    records: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise AuthorityStateDeltaStudyError(
                f"Invalid JSONL audit at {path}:{line_number}."
            ) from error
        if not isinstance(record, dict):
            raise AuthorityStateDeltaStudyError(f"Audit at {path}:{line_number} must be an object.")
        if record.get("study_hash") != study_hash:
            continue
        case_id = str(record.get("case_id", ""))
        if not case_id:
            raise AuthorityStateDeltaStudyError(f"Audit at {path}:{line_number} has no case_id.")
        records[case_id] = record
    return records


def _append_jsonl(path: Path, record: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(canonical_json(dict(record)) + "\n")


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(canonical_json(dict(value)) + "\n", encoding="utf-8")
    temporary.replace(path)


__all__ = [
    "AuthorityStateDeltaStudyConfig",
    "AuthorityStateDeltaStudyError",
    "AuthorityStateDeltaStudyResult",
    "DEFAULT_DELTA_FUSION_CONFIG",
    "build_authority_review_runtime",
    "run_authority_state_delta_cohort",
]
