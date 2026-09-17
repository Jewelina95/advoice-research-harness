"""Atomic case-level transactions across independently reviewed states.

``EvidenceRevisionBatch`` is deliberately limited to one clinical state.  A
case-level Agent review can legitimately produce actions for several states,
but every action must still be evaluated against the same original evidence
snapshot.  This module defines that immutable aggregation boundary; replay
performs the snapshot-dependent checks before applying any individual change.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Mapping, Sequence

from .evidence_revision_batch import EvidenceRevisionBatch, EvidenceRevisionBatchError
from .utils import hash_values

if TYPE_CHECKING:
    from .evidence import MetricEvidenceV2


TRANSACTION_SCHEMA_VERSION = "advoice.evidence_revision_transaction.v1"


class EvidenceRevisionTransactionError(ValueError):
    """Raised when a case-level revision transaction is not safe to replay."""


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


@dataclass(frozen=True, slots=True)
class EvidenceRevisionTransaction:
    """One hash-bound, case-level transaction of one or more state batches.

    The transaction carries no mutable evidence values.  State and case checks
    that require the live typed snapshot are performed by
    :meth:`validate_against_snapshot` immediately before replay.
    """

    case_id: str
    expected_evidence_hash: str
    batches: tuple[EvidenceRevisionBatch, ...]
    schema_version: str = TRANSACTION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not _text(self.case_id):
            raise EvidenceRevisionTransactionError("case_id is required.")
        if not _text(self.expected_evidence_hash):
            raise EvidenceRevisionTransactionError(
                "A revision transaction needs an evidence snapshot hash."
            )
        batches = tuple(self.batches)
        if not batches:
            raise EvidenceRevisionTransactionError(
                "A revision transaction requires one or more state batches."
            )
        if any(not isinstance(batch, EvidenceRevisionBatch) for batch in batches):
            raise EvidenceRevisionTransactionError(
                "A revision transaction may contain EvidenceRevisionBatch objects only."
            )
        if any(batch.case_id != self.case_id for batch in batches):
            raise EvidenceRevisionTransactionError(
                "A revision transaction cannot contain batches from another case."
            )
        if any(batch.expected_evidence_hash != self.expected_evidence_hash for batch in batches):
            raise EvidenceRevisionTransactionError(
                "A revision transaction cannot contain mixed evidence snapshot hashes."
            )
        ordered = tuple(sorted(batches, key=lambda batch: batch.state_id))
        state_ids = [batch.state_id for batch in ordered]
        if len(state_ids) != len(set(state_ids)):
            raise EvidenceRevisionTransactionError(
                "A revision transaction may contain one batch per state only."
            )
        evidence_ids = [
            evidence_id
            for batch in ordered
            for evidence_id in batch.evidence_ids
        ]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise EvidenceRevisionTransactionError(
                "A revision transaction cannot revise one evidence ID in multiple state batches."
            )
        object.__setattr__(self, "batches", ordered)

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return tuple(
            evidence_id
            for batch in self.batches
            for evidence_id in batch.evidence_ids
        )

    @property
    def transaction_hash(self) -> str:
        return hash_values([self._hash_payload()])

    @property
    def revision_hash(self) -> str:
        """Compatibility alias for APIs that consume revision identities."""
        return self.transaction_hash

    def _hash_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "case_id": self.case_id,
            "expected_evidence_hash": self.expected_evidence_hash,
            "batches": [batch.to_dict() for batch in self.batches],
        }

    def to_dict(self) -> dict[str, Any]:
        return {**self._hash_payload(), "transaction_hash": self.transaction_hash}

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_mapping(
        cls, value: "EvidenceRevisionTransaction | Mapping[str, Any]"
    ) -> "EvidenceRevisionTransaction":
        if isinstance(value, cls):
            return value
        if not isinstance(value, Mapping):
            raise EvidenceRevisionTransactionError(
                "revision_transaction must be an EvidenceRevisionTransaction or mapping."
            )
        allowed = {
            "case_id",
            "expected_evidence_hash",
            "batches",
            "schema_version",
            "transaction_hash",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise EvidenceRevisionTransactionError(
                f"revision_transaction includes protected or unknown fields: {unknown}"
            )
        raw_batches = value.get("batches", ())
        if not isinstance(raw_batches, (list, tuple)):
            raise EvidenceRevisionTransactionError(
                "revision_transaction batches must be a sequence."
            )
        batches: list[EvidenceRevisionBatch] = []
        for item in raw_batches:
            if not isinstance(item, Mapping):
                raise EvidenceRevisionTransactionError(
                    "revision_transaction batches must be mappings."
                )
            try:
                batches.append(_batch_from_mapping(item))
            except EvidenceRevisionBatchError as exc:
                raise EvidenceRevisionTransactionError(
                    f"Invalid revision_transaction batch: {exc}"
                ) from exc
        transaction = cls(
            case_id=str(value.get("case_id", "")),
            expected_evidence_hash=str(value.get("expected_evidence_hash", "")),
            batches=tuple(batches),
            schema_version=str(value.get("schema_version", TRANSACTION_SCHEMA_VERSION)),
        )
        supplied_hash = value.get("transaction_hash")
        if supplied_hash is not None and str(supplied_hash) != transaction.transaction_hash:
            raise EvidenceRevisionTransactionError(
                "revision_transaction transaction_hash does not match its contents."
            )
        return transaction

    def validate_against_snapshot(
        self, snapshot: Sequence["MetricEvidenceV2"]
    ) -> None:
        """Verify every sub-batch before any revision is applied.

        The snapshot is intentionally checked as a whole.  A transaction cannot
        be applied to a convenient subset of the case evidence.
        """

        from .evidence_replay import evidence_snapshot_hash

        items = tuple(snapshot)
        if not items:
            raise EvidenceRevisionTransactionError(
                "A revision transaction cannot replay an empty evidence snapshot."
            )
        ids = [item.evidence_id for item in items]
        if len(ids) != len(set(ids)):
            raise EvidenceRevisionTransactionError(
                "MetricEvidenceV2 snapshot has duplicate evidence IDs."
            )
        actual_hash = evidence_snapshot_hash(items)
        if actual_hash != self.expected_evidence_hash:
            raise EvidenceRevisionTransactionError(
                "Revision transaction was created for a stale evidence snapshot."
            )
        case_mismatches = sorted(
            item.evidence_id
            for item in items
            if _text(item.case_id or item.subject_id) != self.case_id
            or (_text(item.subject_id) and _text(item.subject_id) != self.case_id)
        )
        if case_mismatches:
            raise EvidenceRevisionTransactionError(
                "Revision transaction case_id does not match its evidence: "
                f"{case_mismatches}."
            )
        by_id = {item.evidence_id: item for item in items}
        for batch in self.batches:
            unknown = sorted(set(batch.evidence_ids) - set(by_id))
            if unknown:
                raise EvidenceRevisionTransactionError(
                    "Revision transaction references evidence absent from this snapshot: "
                    f"{unknown}."
                )
            wrong_state = sorted(
                evidence_id
                for evidence_id in batch.evidence_ids
                if by_id[evidence_id].state_id != batch.state_id
            )
            if wrong_state:
                raise EvidenceRevisionTransactionError(
                    "Revision transaction batch state_id does not match its evidence: "
                    f"{wrong_state}."
                )


def _batch_from_mapping(value: Mapping[str, Any]) -> EvidenceRevisionBatch:
    """Parse a fully protected serialized ``EvidenceRevisionBatch``."""

    from .evidence_replay import EvidenceRevision

    allowed = {
        "case_id",
        "state_id",
        "action",
        "expected_evidence_hash",
        "revisions",
        "schema_version",
        "batch_hash",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvidenceRevisionBatchError(
            f"revision batch includes protected or unknown fields: {unknown}"
        )
    raw_revisions = value.get("revisions", ())
    if not isinstance(raw_revisions, (list, tuple)):
        raise EvidenceRevisionBatchError("revision batch revisions must be a sequence.")
    revisions = tuple(_revision_from_mapping(item) for item in raw_revisions)
    batch = EvidenceRevisionBatch(
        case_id=str(value.get("case_id", "")),
        state_id=str(value.get("state_id", "")),
        action=str(value.get("action", "")),  # type: ignore[arg-type]
        expected_evidence_hash=str(value.get("expected_evidence_hash", "")),
        revisions=revisions,
        schema_version=str(value.get("schema_version", "advoice.evidence_revision_batch.v1")),
    )
    supplied_hash = value.get("batch_hash")
    if supplied_hash is not None and str(supplied_hash) != batch.batch_hash:
        raise EvidenceRevisionBatchError("revision batch batch_hash does not match its contents.")
    return batch


def _revision_from_mapping(value: Any):
    from .evidence_replay import EvidenceRevision

    if not isinstance(value, Mapping):
        raise EvidenceRevisionBatchError("revision batch cannot contain non-mapping revisions.")
    allowed = {
        "evidence_id",
        "action",
        "expected_evidence_hash",
        "reliability_multiplier",
        "rationale",
        "cited_evidence_ids",
        "schema_version",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise EvidenceRevisionBatchError(
            f"revision includes protected or unknown fields: {unknown}"
        )
    cited = value.get("cited_evidence_ids", ())
    if isinstance(cited, str):
        cited = (cited,)
    if not isinstance(cited, (list, tuple)):
        raise EvidenceRevisionBatchError("revision cited_evidence_ids must be a sequence.")
    return EvidenceRevision(
        evidence_id=str(value.get("evidence_id", "")),
        action=str(value.get("action", "")),  # type: ignore[arg-type]
        expected_evidence_hash=str(value.get("expected_evidence_hash", "")),
        reliability_multiplier=value.get("reliability_multiplier"),
        rationale=str(value.get("rationale", "")),
        cited_evidence_ids=tuple(str(item) for item in cited),
        schema_version=str(value.get("schema_version", "advoice.evidence_revision.v1")),
    )


__all__ = [
    "EvidenceRevisionTransaction",
    "EvidenceRevisionTransactionError",
    "TRANSACTION_SCHEMA_VERSION",
]
