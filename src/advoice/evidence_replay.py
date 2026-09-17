"""Typed, deterministic replay of accepted evidence revisions.

This module deliberately has no route to a raw measurement or an advisor
probability.  A review can only reduce evidence availability/reliability; the
already fitted Module A remains the sole producer of the replayed packet.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
import pandas as pd

from .evidence import MetricEvidenceV2, ReliabilityComponents
from .module_a import ExplanationPacket, TaskConditionedStatisticalExpert
from .state_graph import StateGraphV2
from .utils import hash_values


REVISION_SCHEMA_VERSION = "advoice.evidence_revision.v1"
REPLAY_SCHEMA_VERSION = "advoice.evidence_replay.v1"
ALLOWED_DOWNWEIGHT_MULTIPLIERS = frozenset({0.25, 0.5, 0.75})
RevisionAction = Literal["downweight", "invalidate", "mark_unavailable", "request_remeasurement"]
GRAPH_FEATURE_PREFIXES = ("state_", "rel_", "available_")


class EvidenceRevisionError(ValueError):
    """A revision was stale, unsupported, or attempted a protected edit."""


def evidence_snapshot_hash(evidence: Sequence[MetricEvidenceV2]) -> str:
    """Hash a canonical V2 snapshot independently of caller ordering."""

    return hash_values([{
        "schema_version": "advoice.metric_evidence_v2.snapshot.v1",
        "evidence": [item.to_dict() for item in sorted(evidence, key=lambda value: value.evidence_id)],
    }])


@dataclass(frozen=True, slots=True)
class EvidenceRevision:
    """The only Agent-editable evidence operation.

    This intentionally does not have fields for value, direction, reference,
    observability or permissions.  Any attempt to supply one is a Python
    constructor error instead of a silently ignored clinical-policy violation.
    """

    evidence_id: str
    action: RevisionAction
    expected_evidence_hash: str
    reliability_multiplier: float | None = None
    rationale: str = ""
    cited_evidence_ids: tuple[str, ...] = ()
    schema_version: str = REVISION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not self.evidence_id:
            raise EvidenceRevisionError("A revision must identify one MetricEvidenceV2 item.")
        if self.action not in {"downweight", "invalidate", "mark_unavailable", "request_remeasurement"}:
            raise EvidenceRevisionError("Unsupported evidence revision action.")
        if not self.expected_evidence_hash:
            raise EvidenceRevisionError("A revision must bind to an evidence snapshot hash.")
        cited = self.cited_evidence_ids
        if isinstance(cited, str):
            cited = (cited,)
        object.__setattr__(self, "cited_evidence_ids", tuple(sorted({str(item) for item in cited})))
        if self.action == "downweight":
            if self.reliability_multiplier not in ALLOWED_DOWNWEIGHT_MULTIPLIERS:
                raise EvidenceRevisionError(
                    "Downweight reliability_multiplier must be one of "
                    f"{sorted(ALLOWED_DOWNWEIGHT_MULTIPLIERS)}."
                )
        elif self.reliability_multiplier is not None:
            raise EvidenceRevisionError("Only downweight revisions may carry a reliability multiplier.")

    @property
    def revision_hash(self) -> str:
        return hash_values([self.to_dict()])

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "evidence_id": self.evidence_id,
            "action": self.action,
            "expected_evidence_hash": self.expected_evidence_hash,
            "reliability_multiplier": self.reliability_multiplier,
            "rationale": self.rationale,
            "cited_evidence_ids": list(self.cited_evidence_ids),
        }


def _unavailable(evidence: MetricEvidenceV2, reason: str) -> MetricEvidenceV2:
    """Withdraw availability without altering the immutable observation route."""

    # ``observable`` describes whether the task can observe a construct.  An
    # adjudicator may withdraw one measurement but must not rewrite that route
    # rule, so unavailability is represented only by its existing dedicated
    # field.
    return replace(evidence, unavailable_reason=reason)


def _downweighted_reliability(reliability: ReliabilityComponents, multiplier: float) -> ReliabilityComponents:
    # The scalar reliability used by StateGraphV2 is the product of the six
    # components.  Scaling every component would therefore apply multiplier^6.
    # Scale one explicit measurement dimension so the target multiplier is
    # applied exactly once while preserving the other diagnostic dimensions.
    return replace(
        reliability,
        measurement_stability=float(reliability.measurement_stability) * multiplier,
    )


def apply_evidence_revision(
    snapshot: Sequence[MetricEvidenceV2], revision: EvidenceRevision | None,
) -> tuple[tuple[MetricEvidenceV2, ...], str]:
    """Validate and apply one monotonic revision to an immutable snapshot."""

    original = tuple(snapshot)
    ids = [item.evidence_id for item in original]
    if len(ids) != len(set(ids)):
        raise EvidenceRevisionError("MetricEvidenceV2 snapshot has duplicate evidence IDs.")
    baseline_hash = evidence_snapshot_hash(original)
    if revision is None:
        return original, baseline_hash
    if revision.expected_evidence_hash != baseline_hash:
        raise EvidenceRevisionError("Revision was created for a stale evidence snapshot.")
    matching = [item for item in original if item.evidence_id == revision.evidence_id]
    if not matching:
        raise EvidenceRevisionError("Revision references evidence absent from this snapshot.")
    changed: list[MetricEvidenceV2] = []
    for item in original:
        if item.evidence_id != revision.evidence_id:
            changed.append(item)
        elif revision.action == "downweight":
            changed.append(replace(
                item,
                reliability_components=_downweighted_reliability(
                    item.reliability_components, float(revision.reliability_multiplier)
                ),
            ))
        elif revision.action == "invalidate":
            changed.append(_unavailable(item, "invalidated_by_evidence_revision"))
        elif revision.action == "mark_unavailable":
            changed.append(_unavailable(item, "marked_unavailable_by_evidence_revision"))
        else:
            changed.append(_unavailable(item, "remeasurement_requested_by_evidence_revision"))
    return tuple(changed), evidence_snapshot_hash(changed)


def _numeric(value: Any, default: float = np.nan) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return default
    return parsed if np.isfinite(parsed) else default


def _scalar_reliability(evidence: MetricEvidenceV2) -> float:
    components = evidence.reliability_components
    return float(np.prod([
        components.source, components.role, components.alignment, components.asr,
        components.reference_support, components.measurement_stability,
    ]))


def _available_to_module_a(evidence: MetricEvidenceV2) -> bool:
    """Return whether this exact revision can affect the replayed scorer."""

    return bool(
        evidence.inference_permission
        and evidence.observable
        and evidence.unavailable_reason is None
        and np.isfinite(_numeric(evidence.value))
        and _scalar_reliability(evidence) > 0.0
    )


def metric_evidence_frame(
    snapshot: Sequence[MetricEvidenceV2],
    *, dataset_id: str = "replay", label: str = "unknown", split: str = "replay",
) -> pd.DataFrame:
    """Adapt typed evidence to the existing StateGraph dataframe interface."""

    rows: list[dict[str, Any]] = []
    for item in snapshot:
        value = _numeric(item.value)
        scale = _numeric(item.reference.scale, 1.0)
        z = (value - float(item.reference.median)) / scale if np.isfinite(value) else np.nan
        directional_z = z * item.direction if np.isfinite(z) else np.nan
        available = bool(item.observable and item.inference_permission and item.unavailable_reason is None)
        rows.append({
            "dataset_id": dataset_id,
            "subject_id": item.subject_id,
            "case_id": item.case_id,
            "label": label,
            "split": split,
            "evidence_id": item.evidence_id,
            "metric_id": item.metric_id,
            "metric_instance_id": item.metric_instance_id or item.metric_id,
            "state_id": item.state_id,
            "task_id": item.task_id,
            "task_scope": item.task_id or "overall",
            "value": item.value,
            "directional_z": directional_z,
            "reliability": _scalar_reliability(item) if available else 0.0,
            "missing": not available,
            "evidence_status": "available" if available else "unavailable",
            "report_permission": item.report_permission,
            "unavailable_reason": item.unavailable_reason or "",
            "reference_label": item.reference.reference_label,
            "reference_median": item.reference.median,
            "reference_scale": item.reference.scale,
            "source_segment_ids": list(item.source_segment_ids),
            "segment_ids": list(item.source_segment_ids),
            "source_asset_id": item.provenance.source_asset_id,
            "transcript_id": item.provenance.transcript_id,
            "method_version": item.provenance.method_version,
            "measurement_version": item.provenance.measurement_version,
            "generated_by": item.provenance.generated_by,
            "provenance": item.provenance.to_dict(),
            "confound_tags": list(item.observed_confound_ids),
            "potential_confounds": list(item.potential_confound_ids),
            "observed_confounds": list(item.observed_confound_ids),
            "ruled_out_confounds": list(item.ruled_out_confound_ids),
            "consumed_by_supervised": item.consumed_by_supervised,
            "incremental_for_agent": item.incremental_for_agent,
        })
    return pd.DataFrame(rows)


def build_state_graph_v2(
    snapshot: Sequence[MetricEvidenceV2],
    states_config: Mapping[str, Any],
    *, correlation_config: Mapping[str, Any] | None = None,
    dataset_id: str = "replay", label: str = "unknown", split: str = "replay",
) -> StateGraphV2:
    frame = metric_evidence_frame(snapshot, dataset_id=dataset_id, label=label, split=split)
    return StateGraphV2.from_evidence_frame(
        frame, dict(states_config), None if correlation_config is None else dict(correlation_config)
    )


@dataclass(frozen=True, slots=True)
class ReplayAudit:
    schema_version: str
    revision_hash: str
    parent_evidence_hash: str
    evidence_hash: str
    state_hash: str
    model_hash: str
    packet_hash: str
    audit_hash: str
    idempotent: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "revision_hash": self.revision_hash,
            "parent_evidence_hash": self.parent_evidence_hash,
            "evidence_hash": self.evidence_hash,
            "state_hash": self.state_hash,
            "model_hash": self.model_hash,
            "packet_hash": self.packet_hash,
            "audit_hash": self.audit_hash,
            "idempotent": self.idempotent,
        }


@dataclass(frozen=True, slots=True)
class EvidenceReplayResult:
    """Full deterministic trace; no calibrated decision is produced here."""

    revised_evidence: tuple[MetricEvidenceV2, ...]
    state_graph: StateGraphV2
    packet: ExplanationPacket
    audit: ReplayAudit


def _case_from_graph(
    graph: StateGraphV2,
    model: TaskConditionedStatisticalExpert,
    case_context: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if len(graph.wide) != 1:
        raise EvidenceRevisionError("Replay requires state evidence for exactly one subject.")
    row = graph.wide.iloc[0].to_dict()
    context = dict(case_context or {})
    # Graph-derived model inputs always win; callers cannot smuggle a different
    # numeric state into a packet that is labelled as replayed.  Absence is a
    # replay failure, not permission to reuse a stale context value.
    for feature in getattr(model, "numeric_features_", ()):
        if feature.startswith(GRAPH_FEATURE_PREFIXES):
            if feature not in row:
                raise EvidenceRevisionError(
                    f"Rebuilt StateGraphV2 is missing trained feature {feature!r}."
                )
            context[feature] = row[feature]
    for column in (getattr(model, "task_column", None), getattr(model, "language_column", None)):
        if column and column in row and column not in context:
            context[column] = row[column]
    return context


def _module_a_consumed_evidence_ids(
    revised: Sequence[MetricEvidenceV2],
    graph: StateGraphV2,
    model: TaskConditionedStatisticalExpert,
) -> list[str]:
    """Return only supervised-consumed evidence that reaches a trained graph feature."""

    trained = {str(feature) for feature in getattr(model, "numeric_features_", ())}
    if not trained or graph.wide.empty:
        return []
    row = graph.wide.iloc[0]
    consumed: list[str] = []
    for item in revised:
        if not item.consumed_by_supervised or not item.state_id or not _available_to_module_a(item):
            continue
        state_id = str(item.state_id)
        candidates = {f"state_{state_id}", f"rel_{state_id}", f"available_{state_id}"}
        if item.task_id:
            task = str(item.task_id)
            candidates.update({
                f"state_{state_id}__task_{task}_residual",
                f"rel_{state_id}__task_{task}_residual",
                f"available_{state_id}__task_{task}_residual",
            })
        reached = [name for name in trained.intersection(candidates) if name in row.index]
        if not reached:
            continue
        if not any(
            isinstance(row[name], (bool, np.bool_))
            or (np.isfinite(_numeric(row[name])) and not pd.isna(row[name]))
            for name in reached
        ):
            continue
        consumed.append(item.evidence_id)
    return sorted(set(consumed))


def replay_evidence(
    snapshot: Sequence[MetricEvidenceV2],
    revision: EvidenceRevision | None,
    *,
    states_config: Mapping[str, Any],
    module_a: TaskConditionedStatisticalExpert,
    case_context: Mapping[str, Any] | None = None,
    correlation_config: Mapping[str, Any] | None = None,
    dataset_id: str = "replay", label: str = "unknown", split: str = "replay",
) -> EvidenceReplayResult:
    """Replay a bounded review through StateGraphV2 and a fitted frozen Module A."""

    if not hasattr(module_a, "artifact_hash_") or not hasattr(module_a, "classifier_"):
        raise EvidenceRevisionError("Replay requires a fitted, frozen Module A artifact.")
    original = tuple(snapshot)
    parent_hash = evidence_snapshot_hash(original)
    revised, revised_hash = apply_evidence_revision(original, revision)
    graph = build_state_graph_v2(
        revised, states_config, correlation_config=correlation_config,
        dataset_id=dataset_id, label=label, split=split,
    )
    model_hash = str(module_a.artifact_hash_)
    case = _case_from_graph(graph, module_a, case_context)
    consumed_ids = _module_a_consumed_evidence_ids(revised, graph, module_a)
    packet = module_a.explain_case(
        case,
        consumed_evidence_ids=consumed_ids,
        evidence_snapshot={"evidence_hash": revised_hash, "revision_hash": revision.revision_hash if revision else "none"},
        state_snapshot={"state_hash": graph.state_hash, "state_wide": graph.wide.to_dict("records")},
    )
    packet_hash = hash_values([packet.to_json()])
    revision_hash = revision.revision_hash if revision is not None else hash_values([{"action": "no_revision"}])
    audit_payload = {
        "schema_version": REPLAY_SCHEMA_VERSION,
        "revision_hash": revision_hash,
        "parent_evidence_hash": parent_hash,
        "evidence_hash": revised_hash,
        "state_hash": graph.state_hash,
        "model_hash": model_hash,
        "packet_hash": packet_hash,
    }
    audit = ReplayAudit(**audit_payload, audit_hash=hash_values([audit_payload]))
    return EvidenceReplayResult(revised_evidence=revised, state_graph=graph, packet=packet, audit=audit)


# Short aliases make the frozen contract easy to discover without obscuring the
# domain names used by the framework document.
replay = replay_evidence
apply_revision = apply_evidence_revision
