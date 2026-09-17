"""Deterministic compilation of a state-level review into metric revisions.

The Agent may reason about a clinical state, but replay operates on individual
``MetricEvidenceV2`` objects.  This module is the narrow, pure boundary between
those two representations.  It does not alter evidence, invoke an advisor, or
perform replay.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

from .evidence import MetricEvidenceV2
from .evidence_replay import (
    ALLOWED_DOWNWEIGHT_MULTIPLIERS,
    EvidenceRevision,
    evidence_snapshot_hash,
)
from .state_graph import StateGraphV2
from .utils import hash_values


BatchAction = Literal["downweight", "invalidate", "mark_unavailable", "retain"]
DEFAULT_DOWNWEIGHT_MULTIPLIER = 0.5
BATCH_SCHEMA_VERSION = "advoice.evidence_revision_batch.v1"


class EvidenceRevisionBatchError(ValueError):
    """Raised when a state-level revision cannot be compiled safely."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _finite_numeric(value: Any) -> bool:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(parsed))


def _scalar_reliability(evidence: MetricEvidenceV2) -> float:
    components = evidence.reliability_components
    return float(np.prod([
        components.source,
        components.role,
        components.alignment,
        components.asr,
        components.reference_support,
        components.measurement_stability,
    ]))


def _inferable(evidence: MetricEvidenceV2) -> bool:
    """Mirror the replay eligibility rule without changing the replay module."""

    return bool(
        evidence.inference_permission
        and evidence.observable
        and evidence.unavailable_reason is None
        and _finite_numeric(evidence.value)
        and _scalar_reliability(evidence) > 0.0
    )


def _records_from_state_graph(state_graph: Any) -> tuple[dict[str, Any], ...]:
    """Extract state-card records from supported graph representations.

    ``StateGraphV2`` is the production representation.  The mapping forms are
    intentionally small compatibility forms for serialized audit artifacts;
    they do not weaken the evidence and case checks performed by the compiler.
    """

    cards: Any
    if isinstance(state_graph, StateGraphV2):
        cards = state_graph.cards
    elif hasattr(state_graph, "cards"):
        cards = getattr(state_graph, "cards")
    elif isinstance(state_graph, Mapping):
        cards = state_graph.get("cards", state_graph.get("state_cards"))
        if cards is None:
            states = state_graph.get("states", state_graph.get("state_ids"))
            if isinstance(states, Mapping):
                return tuple({"state_id": str(key)} for key in states)
            if isinstance(states, (list, tuple, set)):
                return tuple({"state_id": str(value)} for value in states)
    else:
        cards = None

    if cards is None:
        raise EvidenceRevisionBatchError("A StateGraphV2 or serialized state-card graph is required.")
    if hasattr(cards, "to_dict"):
        try:
            cards = cards.to_dict("records")
        except TypeError as exc:
            raise EvidenceRevisionBatchError("State graph cards must be record-like.") from exc
    if isinstance(cards, Mapping):
        cards = [cards]
    if not isinstance(cards, (list, tuple)):
        raise EvidenceRevisionBatchError("State graph cards must be a sequence of records.")
    records: list[dict[str, Any]] = []
    for card in cards:
        if not isinstance(card, Mapping):
            raise EvidenceRevisionBatchError("State graph contains a non-record state card.")
        records.append({str(key): value for key, value in card.items()})
    return tuple(records)


def _validate_state_graph(state_graph: Any, *, case_id: str, state_id: str) -> None:
    records = _records_from_state_graph(state_graph)
    state_records = [
        record for record in records
        if _text(record.get("state_id", record.get("state_base_id"))) == state_id
    ]
    if not state_records:
        raise EvidenceRevisionBatchError(f"Unknown state_id {state_id!r}.")

    observed_cases: set[str] = set()
    for record in state_records:
        for key in ("case_id", "subject_id"):
            value = _text(record.get(key))
            if value and value.lower() != "nan":
                observed_cases.add(value)
    if observed_cases and case_id not in observed_cases:
        raise EvidenceRevisionBatchError(
            f"State {state_id!r} is not bound to case_id {case_id!r}."
        )


def _validate_evidence_snapshot(
    evidence: Sequence[MetricEvidenceV2],
    *,
    case_id: str,
    expected_evidence_hash: str,
) -> tuple[MetricEvidenceV2, ...]:
    if not evidence:
        raise EvidenceRevisionBatchError("An empty MetricEvidence snapshot is not revisable.")
    if not _text(expected_evidence_hash):
        raise EvidenceRevisionBatchError("expected_evidence_hash is required.")
    items = tuple(evidence)
    if any(not isinstance(item, MetricEvidenceV2) for item in items):
        raise EvidenceRevisionBatchError("The compiler accepts typed MetricEvidenceV2 only.")
    ids = [item.evidence_id for item in items]
    if any(not _text(item_id) for item_id in ids):
        raise EvidenceRevisionBatchError("Every MetricEvidenceV2 item needs an evidence_id.")
    if len(ids) != len(set(ids)):
        raise EvidenceRevisionBatchError("MetricEvidenceV2 snapshot contains duplicate evidence IDs.")

    foreign = [
        item.evidence_id
        for item in items
        if _text(item.case_id) != case_id or (
            _text(item.subject_id) and _text(item.subject_id) != case_id
        )
    ]
    if foreign:
        raise EvidenceRevisionBatchError(
            f"Cross-case evidence is not allowed for case_id {case_id!r}: {sorted(foreign)}."
        )
    incremental = [item.evidence_id for item in items if item.incremental_for_agent]
    if incremental:
        raise EvidenceRevisionBatchError(
            "Incremental Agent evidence cannot be compiled into a supervised revision batch: "
            f"{sorted(incremental)}."
        )

    actual_hash = evidence_snapshot_hash(items)
    if actual_hash != expected_evidence_hash:
        raise EvidenceRevisionBatchError(
            "Mixed or stale evidence hash: expected_evidence_hash does not match the complete snapshot."
        )
    return items


def _target_evidence(
    evidence: Sequence[MetricEvidenceV2], *, state_id: str
) -> tuple[MetricEvidenceV2, ...]:
    state_items = tuple(item for item in evidence if _text(item.state_id) == state_id)
    if not state_items:
        raise EvidenceRevisionBatchError(
            f"The evidence snapshot contains no evidence for state_id {state_id!r}."
        )
    consumed = tuple(item for item in state_items if item.consumed_by_supervised)
    unusable = [item.evidence_id for item in consumed if not _inferable(item)]
    if unusable:
        raise EvidenceRevisionBatchError(
            "State revision requires every supervised state evidence item to be inferable; "
            f"unusable evidence: {sorted(unusable)}."
        )
    if not consumed:
        raise EvidenceRevisionBatchError(
            "State revision has no evidence actually consumed by supervised inference."
        )
    return tuple(sorted(consumed, key=lambda item: item.evidence_id))


@dataclass(frozen=True, slots=True)
class EvidenceRevisionBatch:
    """Immutable, hash-bound collection of atomic evidence revisions."""

    case_id: str
    state_id: str
    action: BatchAction
    expected_evidence_hash: str
    revisions: tuple[EvidenceRevision, ...] = ()
    schema_version: str = BATCH_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _text(self.case_id) or not _text(self.state_id):
            raise EvidenceRevisionBatchError("case_id and state_id are required.")
        if self.action not in {"downweight", "invalidate", "mark_unavailable", "retain"}:
            raise EvidenceRevisionBatchError(f"Unsupported batch action {self.action!r}.")
        if not _text(self.expected_evidence_hash):
            raise EvidenceRevisionBatchError("A revision batch needs an evidence snapshot hash.")
        revisions = tuple(self.revisions)
        if any(not isinstance(item, EvidenceRevision) for item in revisions):
            raise EvidenceRevisionBatchError("A batch may contain EvidenceRevision objects only.")
        if self.action == "retain" and revisions:
            raise EvidenceRevisionBatchError("retain is an explicit no-op and cannot contain revisions.")
        if self.action != "retain" and not revisions:
            raise EvidenceRevisionBatchError("Non-retain actions cannot produce an empty batch.")
        if any(item.expected_evidence_hash != self.expected_evidence_hash for item in revisions):
            raise EvidenceRevisionBatchError("A batch cannot contain mixed evidence hashes.")
        if any(item.action != self.action for item in revisions):
            raise EvidenceRevisionBatchError("A batch cannot mix state and atomic revision actions.")
        if tuple(sorted(revisions, key=lambda item: item.evidence_id)) != revisions:
            raise EvidenceRevisionBatchError("Batch revisions must be stably sorted by evidence_id.")
        ids = [item.evidence_id for item in revisions]
        if len(ids) != len(set(ids)):
            raise EvidenceRevisionBatchError("A batch cannot contain duplicate evidence revisions.")
        object.__setattr__(self, "revisions", revisions)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(item.evidence_id for item in self.revisions)

    @property
    def batch_hash(self) -> str:
        return hash_values([self._hash_payload()])

    @property
    def revision_hash(self) -> str:
        """Compatibility alias for callers that call all revision artifacts hashes."""
        return self.batch_hash

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "state_id": self.state_id,
            "action": self.action,
            "expected_evidence_hash": self.expected_evidence_hash,
            "revisions": [item.to_dict() for item in self.revisions],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "batch_hash": self.batch_hash}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())


def compile_evidence_revision_batch(
    case_id: str,
    state_id: str,
    action: BatchAction,
    state_graph: Any,
    evidence: Sequence[MetricEvidenceV2],
    expected_evidence_hash: str,
    *,
    reliability_multiplier: float | None = None,
    rationale: str = "",
    cited_evidence_ids: Iterable[str] = (),
) -> EvidenceRevisionBatch:
    """Compile one state-level Agent action into atomic MetricEvidence revisions.

    The evidence argument is the complete case snapshot used to compute
    ``expected_evidence_hash``.  This prevents a caller from compiling a batch
    against only a convenient subset of the case evidence.
    """

    normalized_case = _text(case_id)
    normalized_state = _text(state_id)
    if not normalized_case or not normalized_state:
        raise EvidenceRevisionBatchError("case_id and state_id are required.")
    if action not in {"downweight", "invalidate", "mark_unavailable", "retain"}:
        raise EvidenceRevisionBatchError(f"Unsupported batch action {action!r}.")
    _validate_state_graph(state_graph, case_id=normalized_case, state_id=normalized_state)
    snapshot = _validate_evidence_snapshot(
        evidence, case_id=normalized_case, expected_evidence_hash=expected_evidence_hash
    )
    candidates = _target_evidence(snapshot, state_id=normalized_state)
    if action == "retain":
        return EvidenceRevisionBatch(
            case_id=normalized_case,
            state_id=normalized_state,
            action="retain",
            expected_evidence_hash=expected_evidence_hash,
        )

    multiplier = (
        DEFAULT_DOWNWEIGHT_MULTIPLIER
        if reliability_multiplier is None
        else reliability_multiplier
    )
    if action == "downweight" and multiplier not in ALLOWED_DOWNWEIGHT_MULTIPLIERS:
        raise EvidenceRevisionBatchError(
            "reliability_multiplier must be one of "
            f"{sorted(ALLOWED_DOWNWEIGHT_MULTIPLIERS)}."
        )
    if action != "downweight" and reliability_multiplier is not None:
        raise EvidenceRevisionBatchError(
            "reliability_multiplier is valid only for downweight batches."
        )
    cited = tuple(sorted({str(item) for item in cited_evidence_ids}))
    revisions = tuple(
        EvidenceRevision(
            evidence_id=item.evidence_id,
            action=action,  # type: ignore[arg-type]
            expected_evidence_hash=expected_evidence_hash,
            reliability_multiplier=float(multiplier) if action == "downweight" else None,
            rationale=str(rationale),
            cited_evidence_ids=cited,
        )
        for item in candidates
    )
    return EvidenceRevisionBatch(
        case_id=normalized_case,
        state_id=normalized_state,
        action=action,
        expected_evidence_hash=expected_evidence_hash,
        revisions=revisions,
    )


# Short names for downstream callers and audit code.
compile_state_revision_batch = compile_evidence_revision_batch
compile_revision_batch = compile_evidence_revision_batch


__all__ = [
    "BATCH_SCHEMA_VERSION",
    "DEFAULT_DOWNWEIGHT_MULTIPLIER",
    "EvidenceRevisionBatch",
    "EvidenceRevisionBatchError",
    "compile_evidence_revision_batch",
    "compile_revision_batch",
    "compile_state_revision_batch",
]
