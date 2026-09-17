"""ADvoice reproducible research harness."""

from .evidence import (
    ConfoundSets,
    EvidencePermissions,
    EvidenceProvenance,
    MetricEvidenceV2,
    ReferenceMetadata,
    ReliabilityComponents,
    deserialize_metric_evidence_v2,
    serialize_metric_evidence_v2,
)
from .routing import ObservationRoute, RouteDecision, TargetRoute, route_case
from .evidence_replay import (
    EvidenceRevision,
    EvidenceRevisionError,
    apply_evidence_revision_transaction,
    replay_evidence,
)
from .evidence_revision_transaction import (
    EvidenceRevisionTransaction,
    EvidenceRevisionTransactionError,
)
from .decision_lock import (
    DecisionLock,
    DecisionLockError,
    EvidenceTraceError,
    HashMismatchError,
    ReportTrace,
    ReportTraceEntry,
    UnlockedReportError,
    canonical_json,
    create_decision_lock,
    deserialize_decision_lock,
    hash_artifact,
    lock_decision,
    serialize_decision_lock,
    validate_decision_lock,
    validate_locked_report,
    validate_report_trace,
)

__version__ = "0.2.0"

__all__ = [
    "ConfoundSets", "EvidencePermissions", "EvidenceProvenance", "MetricEvidenceV2",
    "ObservationRoute", "ReferenceMetadata", "ReliabilityComponents", "RouteDecision",
    "TargetRoute", "deserialize_metric_evidence_v2", "route_case",
    "serialize_metric_evidence_v2", "EvidenceRevision", "EvidenceRevisionError",
    "EvidenceRevisionTransaction", "EvidenceRevisionTransactionError",
    "apply_evidence_revision_transaction", "replay_evidence",
    "DecisionLock", "DecisionLockError", "EvidenceTraceError", "HashMismatchError",
    "ReportTrace", "ReportTraceEntry", "UnlockedReportError", "canonical_json",
    "create_decision_lock", "hash_artifact", "lock_decision", "validate_locked_report",
    "validate_report_trace", "serialize_decision_lock", "deserialize_decision_lock",
    "validate_decision_lock",
]
