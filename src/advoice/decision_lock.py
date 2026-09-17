"""Immutable, hash-bound contracts for a final ADvoice decision.

This module is deliberately a boundary object.  It does not run a predictor,
replay a state graph, or choose a class.  It verifies that the artifacts
produced by those stages all refer to the same case and evidence revision
before a report can be considered locked.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, is_dataclass
import hashlib
import json
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .utils import hash_values as _legacy_hash_values


HASH_LENGTH = 64
SCHEMA_VERSION = "advoice.decision_lock.v1"


class DecisionLockError(ValueError):
    """Base error raised when a decision cannot be safely locked."""


class HashMismatchError(DecisionLockError):
    """An artifact's declared identity does not match its content or peers."""


class EvidenceTraceError(DecisionLockError):
    """A report claim cannot be traced through permitted evidence."""


class UnlockedReportError(DecisionLockError):
    """A report is missing the lock binding or explicitly remains unlocked."""


_NO_ARTIFACT_VIEW = object()


def _dataframe_view(value: Any) -> Any:
    """Return a stable table view without importing pandas at module import."""

    if value.__class__.__module__.split(".", 1)[0] != "pandas":
        return _NO_ARTIFACT_VIEW
    class_name = value.__class__.__name__
    if class_name == "DataFrame":
        columns = sorted(str(column) for column in value.columns)
        ordered = value.loc[:, columns]
        return {
            "artifact_type": "pandas.DataFrame",
            "columns": columns,
            "records": ordered.to_dict("records"),
        }
    if class_name == "Series":
        return {
            "artifact_type": "pandas.Series",
            "name": None if value.name is None else str(value.name),
            "values": value.tolist(),
        }
    return _NO_ARTIFACT_VIEW


def _explicit_artifact_view(value: Any) -> Any:
    """Serialize public pipeline contracts through named, versioned views."""

    # Local imports avoid making the boundary module part of the predictors'
    # import graph while still preventing accidental ``str(object)`` hashes.
    from .evidence_replay import ReplayAudit
    from .module_a import ExplanationPacket
    from .module_b import ModuleBPrediction
    from .state_graph import StateGraphV2

    if isinstance(value, StateGraphV2):
        return {
            "artifact_type": "advoice.StateGraphV2",
            "canonical_view_version": 1,
            "evidence_hash": value.evidence_hash,
            "state_hash": value.state_hash,
            "cards": _dataframe_view(value.cards),
            "wide": _dataframe_view(value.wide),
        }
    if isinstance(value, ExplanationPacket):
        return {
            "artifact_type": "advoice.ExplanationPacket",
            "canonical_view_version": 1,
            "payload": value.to_dict(),
        }
    if isinstance(value, ReplayAudit):
        return {
            "artifact_type": "advoice.ReplayAudit",
            "canonical_view_version": 1,
            "payload": value.to_dict(),
        }
    if isinstance(value, ModuleBPrediction):
        return {
            "artifact_type": "advoice.ModuleBPrediction",
            "canonical_view_version": 1,
            "payload": value.to_dict(),
        }
    return _NO_ARTIFACT_VIEW


def _jsonable(value: Any) -> Any:
    """Convert contract values to a deterministic JSON-native tree.

    Non-finite numeric values are represented as JSON ``null``.  Unknown
    objects are rejected instead of being hashed through an unstable repr.
    """

    explicit = _explicit_artifact_view(value)
    if explicit is not _NO_ARTIFACT_VIEW:
        return _jsonable(explicit)
    table = _dataframe_view(value)
    if table is not _NO_ARTIFACT_VIEW:
        return _jsonable(table)
    if (
        value.__class__.__module__.startswith("pandas.")
        and value.__class__.__name__ in {"NAType", "NaTType"}
    ):
        return None

    if is_dataclass(value):
        return _jsonable({item.name: getattr(value, item.name) for item in fields(value)})
    if hasattr(value, "to_dict") and callable(value.to_dict):
        return _jsonable(value.to_dict())
    if isinstance(value, Mapping):
        result: dict[str, Any] = {}
        for key in sorted(value, key=str):
            normalized_key = str(key)
            if normalized_key in result:
                raise TypeError(f"Mapping keys collide after string normalization: {normalized_key!r}.")
            result[normalized_key] = _jsonable(value[key])
        return result
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [_jsonable(item) for item in value]
        return sorted(normalized, key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if hasattr(value, "item") and callable(value.item):
        return _jsonable(value.item())
    raise TypeError(
        f"Unsupported decision-lock artifact type: {value.__class__.__module__}."
        f"{value.__class__.__qualname__}."
    )


def canonical_json(value: Any) -> str:
    """Return compact, insertion-order-independent JSON for a contract."""

    return json.dumps(
        _jsonable(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )


def hash_artifact(value: Any) -> str:
    """Hash one artifact using :func:`canonical_json`."""

    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def _hash_options(value: Any) -> set[str]:
    """Accept the repository's older hash primitive at migration boundaries."""

    options = {hash_artifact(value)}
    # Legacy hashes are accepted only for legacy mapping payloads.  Typed
    # pipeline objects have one canonical identity at this boundary.
    if isinstance(value, Mapping):
        options.add(_legacy_hash_values([value]))
    return options


# Short aliases make the hash primitive convenient for producers and tests.
canonical_hash = hash_artifact
content_hash = hash_artifact
artifact_hash = hash_artifact


def _freeze(value: Any) -> Any:
    """Deep-freeze mappings and sequences stored in public contracts."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(_freeze(item) for item in value)
    return value


def _tuple_strings(value: Any) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(dict.fromkeys(str(item) for item in value))


def _flag(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "unavailable", "none", "null"}:
            return False
        if lowered in {"true", "1", "yes", "available", "observable"}:
            return True
    return bool(value)


def _required_hash(value: Any, name: str) -> str:
    text = str(value or "")
    if len(text) != HASH_LENGTH or any(char not in "0123456789abcdef" for char in text.lower()):
        raise DecisionLockError(f"{name} must be a 64-character hexadecimal SHA-256 hash.")
    return text.lower()


def _artifact_identity(artifact: Any, name: str, supplied: str | None = None) -> str:
    """Resolve an artifact hash and reject a conflicting embedded identity."""

    embedded: str | None = None
    if isinstance(artifact, Mapping):
        candidates = (
            f"{name}_hash",
            "hash" if name not in {"module_a_pre_replay", "module_a_post_replay"} else "packet_hash",
            "snapshot_hash" if name == "revision" else "",
        )
        for key in candidates:
            if key and artifact.get(key) not in (None, ""):
                embedded = str(artifact[key])
                break
    if embedded is not None and isinstance(artifact, Mapping) and name != "revision":
        identity_keys = {f"{name}_hash", "hash", "packet_hash"}
        content = {key: item for key, item in artifact.items() if key not in identity_keys}
        if embedded not in (_hash_options(artifact) | _hash_options(content)):
            raise HashMismatchError(f"{name} embedded hash does not match artifact content.")
    if supplied is not None:
        resolved = _required_hash(supplied, name)
        if embedded is not None and _required_hash(embedded, f"{name} embedded hash") != resolved:
            raise HashMismatchError(f"{name} hash conflicts with its embedded hash.")
        if artifact is not None and embedded is None and resolved not in _hash_options(artifact):
            raise HashMismatchError(f"{name} hash does not match artifact content.")
        return resolved
    if embedded is not None:
        return _required_hash(embedded, f"{name} embedded hash")
    if artifact is None:
        raise DecisionLockError(f"{name} artifact or hash is required.")
    return hash_artifact(artifact)


def _get(item: Any, *names: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        for name in names:
            if name in item:
                return item[name]
    else:
        for name in names:
            if hasattr(item, name):
                return getattr(item, name)
    return default


def _linked_hash(item: Any, *names: str) -> Any:
    """Read a binding from the artifact or its common nested hash maps."""

    direct = _get(item, *names, default=None)
    if direct is not None:
        return direct
    nested = _get(item, "hashes", "hash_bindings", default=None)
    if isinstance(nested, Mapping):
        return _get(nested, *names, default=None)
    return None


def _records(value: Any, *container_names: str) -> list[Any]:
    def keep(item: Any) -> bool:
        return isinstance(item, Mapping) or is_dataclass(item) or hasattr(item, "to_dict")

    if value is None:
        return []
    for name in (*container_names, "cards"):
        nested = _get(value, name, default=None)
        if nested is not None and nested is not value:
            rows = _records(nested)
            if rows:
                return rows
    if hasattr(value, "to_dict") and callable(value.to_dict):
        try:
            rows = value.to_dict("records")
        except (TypeError, ValueError):
            rows = None
        if isinstance(rows, list):
            return [item for item in rows if keep(item)]
    if isinstance(value, Mapping):
        for name in container_names:
            nested = value.get(name)
            if isinstance(nested, Sequence) and not isinstance(nested, (str, bytes)):
                return [item for item in nested if keep(item)]
        return [value]
    return [item for item in value if keep(item)]


@dataclass(frozen=True, slots=True)
class ReportTraceEntry:
    """One report claim and its complete evidence path."""

    claim_id: str
    claim: str
    state_card_id: str
    state_revision_hash: str
    metric_evidence_ids: tuple[str, ...]
    task_id: str
    segment_ids: tuple[str, ...]
    source_asset_id: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "metric_evidence_ids", _tuple_strings(self.metric_evidence_ids))
        object.__setattr__(self, "segment_ids", _tuple_strings(self.segment_ids))
        for name in ("claim_id", "claim", "state_card_id", "state_revision_hash", "task_id", "source_asset_id"):
            if not str(getattr(self, name)).strip():
                raise EvidenceTraceError(f"Report trace field {name!r} is required.")
        if not self.metric_evidence_ids or not self.segment_ids:
            raise EvidenceTraceError("Every report claim must cite metric evidence and a source segment.")

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReportTraceEntry":
        if not isinstance(value, Mapping):
            value = _jsonable(value)
        evidence = value.get("metric_evidence_ids", value.get("evidence_ids", value.get("metric_evidence_id", ())))
        segments = value.get("segment_ids", value.get("source_segment_ids", value.get("segment_id", ())))
        if isinstance(evidence, str):
            evidence = (evidence,)
        if isinstance(segments, str):
            segments = (segments,)
        return cls(
            claim_id=str(value.get("claim_id", value.get("id", ""))),
            claim=str(value.get("claim", value.get("text", ""))),
            state_card_id=str(value.get("state_card_id", value.get("state_id", ""))),
            state_revision_hash=str(value.get("state_revision_hash", value.get("revision_hash", ""))),
            metric_evidence_ids=tuple(str(item) for item in evidence or ()),
            task_id=str(value.get("task_id", value.get("task_scope", ""))),
            segment_ids=tuple(str(item) for item in segments or ()),
            source_asset_id=str(value.get("source_asset_id", value.get("audio_asset_id", ""))),
        )

    @property
    def evidence_ids(self) -> tuple[str, ...]:
        return self.metric_evidence_ids

    @property
    def state_card_revision_hash(self) -> str:
        return self.state_revision_hash

    @property
    def metric_evidence_id(self) -> str:
        return self.metric_evidence_ids[0]

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "claim": self.claim,
            "state_card_id": self.state_card_id,
            "state_revision_hash": self.state_revision_hash,
            "metric_evidence_ids": list(self.metric_evidence_ids),
            "task_id": self.task_id,
            "segment_ids": list(self.segment_ids),
            "source_asset_id": self.source_asset_id,
        }


@dataclass(frozen=True, slots=True)
class ReportTrace:
    """Immutable collection of claim-to-source trace entries."""

    entries: tuple[ReportTraceEntry, ...] = ()

    def __post_init__(self) -> None:
        entries = tuple(
            item if isinstance(item, ReportTraceEntry) else ReportTraceEntry.from_mapping(item)
            for item in self.entries
        )
        if len({item.claim_id for item in entries}) != len(entries):
            raise EvidenceTraceError("Report trace claim IDs must be unique.")
        object.__setattr__(self, "entries", entries)

    @classmethod
    def from_mapping(cls, value: Any) -> "ReportTrace":
        if isinstance(value, ReportTrace):
            return value
        if isinstance(value, Mapping):
            value = value.get("entries", value.get("claims", value.get("report_trace", [])))
        return cls(tuple(ReportTraceEntry.from_mapping(item) for item in (value or [])))

    def to_dict(self) -> dict[str, Any]:
        return {"entries": [item.to_dict() for item in self.entries]}

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    def __iter__(self):
        return iter(self.entries)


def _check_case(artifact: Any, case_id: str, name: str) -> None:
    found = _get(artifact, "case_id", "case", default=None)
    if found is not None and str(found) != case_id:
        raise HashMismatchError(f"{name} belongs to case {found!r}, not {case_id!r}.")


def validate_report_trace(
    report_trace: ReportTrace | Sequence[Mapping[str, Any]],
    *,
    case_id: str,
    revision_hash: str,
    state_graph: Any,
    metric_evidence: Any,
    segments: Any,
    state_cards: Any = None,
) -> ReportTrace:
    """Validate ``claim -> state -> metric -> task/segment -> asset`` links."""

    trace = ReportTrace.from_mapping(report_trace)
    revision_hash = _required_hash(revision_hash, "revision_hash")
    states = _records(
        state_cards if state_cards is not None else state_graph,
        "state_cards", "states", "state_observations", "cards",
    )
    metrics = _records(metric_evidence, "metric_evidence", "evidence", "records")
    segment_rows = _records(segments, "segments", "source_segments")

    def index_records(rows: Sequence[Any], names: tuple[str, ...], kind: str) -> dict[str, Any]:
        indexed: dict[str, Any] = {}
        for item in rows:
            identifier = str(_get(item, *names, default=""))
            if not identifier:
                raise EvidenceTraceError(f"{kind} is missing its required identifier.")
            if identifier in indexed:
                raise EvidenceTraceError(f"Duplicate {kind} identifier {identifier!r}.")
            indexed[identifier] = item
        return indexed

    state_by_id = index_records(states, ("state_card_id", "state_id", "id"), "StateCard")
    evidence_by_id = index_records(metrics, ("evidence_id", "id"), "MetricEvidence")
    segment_by_id = index_records(segment_rows, ("segment_id", "id"), "source segment")
    if not trace.entries:
        raise EvidenceTraceError("A locked report must contain at least one report trace entry.")

    for entry in trace:
        state = state_by_id.get(entry.state_card_id)
        if state is None:
            raise EvidenceTraceError(f"Unknown StateCard {entry.state_card_id!r} in claim {entry.claim_id!r}.")
        state_revision = _get(
            state, "revision_hash", "state_revision_hash", "evidence_revision_hash", default=None,
        )
        if state_revision in (None, ""):
            raise HashMismatchError(
                f"StateCard {entry.state_card_id!r} has no explicit evidence revision binding."
            )
        if str(state_revision) != revision_hash or entry.state_revision_hash != revision_hash:
            raise HashMismatchError(f"Claim {entry.claim_id!r} cites a stale StateCard revision.")
        state_observable = _get(state, "observable", "observability", "available", default=True)
        if str(state_observable).lower() in {"false", "0", "unavailable", "none"}:
            raise EvidenceTraceError(f"StateCard {entry.state_card_id!r} is unavailable.")
        if not _flag(_get(state, "report_permission", "reportable", default=True), default=True):
            raise EvidenceTraceError(f"StateCard {entry.state_card_id!r} is not report-permitted.")
        if str(_get(state, "case_id", default=case_id)) != case_id:
            raise EvidenceTraceError(f"StateCard {entry.state_card_id!r} belongs to another case.")
        state_tasks = {str(_get(state, "task_id", "task_scope", default=""))}
        state_tasks.update(str(item) for item in (_get(state, "task_ids", default=()) or ()))
        if entry.task_id not in state_tasks:
            raise EvidenceTraceError(f"Claim {entry.claim_id!r} crosses the StateCard task boundary.")
        allowed_state_evidence = set(str(item) for item in (_get(state, "supporting_evidence_ids", "metric_evidence_ids", default=()) or ()))
        allowed_state_evidence.update(str(item) for item in (_get(state, "counterevidence_ids", default=()) or ()))
        allowed_state_evidence.discard("")
        if not allowed_state_evidence:
            raise EvidenceTraceError(
                f"StateCard {entry.state_card_id!r} has no explicit state-to-evidence association."
            )
        for evidence_id in entry.metric_evidence_ids:
            evidence = evidence_by_id.get(evidence_id)
            if evidence is None:
                raise EvidenceTraceError(f"Unknown MetricEvidence {evidence_id!r} in claim {entry.claim_id!r}.")
            if evidence_id not in allowed_state_evidence:
                raise EvidenceTraceError(f"MetricEvidence {evidence_id!r} is not attached to the cited StateCard.")
            if str(_get(evidence, "case_id", default=case_id)) != case_id:
                raise EvidenceTraceError(f"MetricEvidence {evidence_id!r} belongs to another case.")
            observable = _get(evidence, "observable", "observability", "available", default=True)
            if str(observable).lower() in {"false", "0", "unavailable", "none"} or _get(evidence, "missing", default=False):
                raise EvidenceTraceError(f"MetricEvidence {evidence_id!r} is unavailable or missing.")
            if not _flag(_get(evidence, "report_permission", "reportable", default=False)):
                raise EvidenceTraceError(f"MetricEvidence {evidence_id!r} is not report-permitted.")
            role = str(_get(evidence, "evidence_role", "role", default="clinical_support"))
            if role in {"quality_control", "model_auxiliary", "planned_unavailable"}:
                raise EvidenceTraceError(f"MetricEvidence {evidence_id!r} is not clinical-support evidence.")
            if str(_get(evidence, "task_id", "task_scope", default=entry.task_id)) != entry.task_id:
                raise EvidenceTraceError(f"MetricEvidence {evidence_id!r} crosses the task boundary.")
            evidence_segments = set(str(item) for item in (_get(evidence, "segment_ids", "source_segment_ids", default=()) or ()))
            if not set(entry.segment_ids).issubset(evidence_segments):
                raise EvidenceTraceError(f"Claim {entry.claim_id!r} cites segments absent from MetricEvidence.")
            provenance = _get(evidence, "provenance", default=None)
            evidence_asset = _get(evidence, "source_asset_id", "audio_asset_id", default=None)
            if evidence_asset is None and provenance is not None:
                evidence_asset = _get(provenance, "source_asset_id", "audio_asset_id", default=None)
            if evidence_asset is not None and str(evidence_asset) != entry.source_asset_id:
                raise EvidenceTraceError(f"MetricEvidence {evidence_id!r} does not resolve to the cited source asset.")
        state_segments = set(str(item) for item in (_get(state, "segment_ids", "evidence_segment_ids", default=()) or ()))
        if state_segments and not set(entry.segment_ids).issubset(state_segments):
            raise EvidenceTraceError(f"Claim {entry.claim_id!r} cites segments absent from the StateCard.")
        for segment_id in entry.segment_ids:
            segment = segment_by_id.get(segment_id)
            if segment is None:
                raise EvidenceTraceError(f"Unknown source segment {segment_id!r} in claim {entry.claim_id!r}.")
            if str(_get(segment, "case_id", default=case_id)) != case_id:
                raise EvidenceTraceError(f"Source segment {segment_id!r} belongs to another case.")
            if str(_get(segment, "task_id", "task_scope", default="")) != entry.task_id:
                raise EvidenceTraceError(f"Source segment {segment_id!r} crosses the task boundary.")
            asset = _get(segment, "audio_asset_id", "source_asset_id", "asset_id", default=None)
            if asset is not None and str(asset) != entry.source_asset_id:
                raise EvidenceTraceError(f"Source segment {segment_id!r} does not resolve to the cited source asset.")
    return trace


@dataclass(frozen=True, slots=True)
class DecisionLock:
    """The only object a report renderer may treat as a final decision."""

    case_id: str
    evidence_snapshot_hash: str
    revision_hash: str
    state_graph_hash: str
    module_a_pre_replay_hash: str
    module_a_post_replay_hash: str
    agent_decision_hash: str
    validator_hash: str
    module_b_output_hash: str
    model_versions: Mapping[str, str]
    skill_versions: Mapping[str, str]
    tool_versions: Mapping[str, str]
    report_trace: ReportTrace
    replay_audit_hash: str | None = None
    lock_id: str = ""
    locked: bool = True
    report_hash: str | None = None

    def __post_init__(self) -> None:
        if not str(self.case_id).strip():
            raise DecisionLockError("case_id is required.")
        for name in (
            "evidence_snapshot_hash", "revision_hash", "state_graph_hash",
            "module_a_pre_replay_hash", "module_a_post_replay_hash",
            "agent_decision_hash", "validator_hash", "module_b_output_hash",
        ):
            object.__setattr__(self, name, _required_hash(getattr(self, name), name))
        if self.replay_audit_hash is not None:
            object.__setattr__(
                self, "replay_audit_hash", _required_hash(self.replay_audit_hash, "replay_audit_hash")
            )
        for name in ("model_versions", "skill_versions", "tool_versions"):
            value = {str(key): str(item) for key, item in dict(getattr(self, name)).items()}
            if not value or any(not key or not item for key, item in value.items()):
                raise DecisionLockError(f"{name} must bind at least one non-empty version.")
            object.__setattr__(self, name, MappingProxyType(value))
        if not isinstance(self.report_trace, ReportTrace):
            object.__setattr__(self, "report_trace", ReportTrace.from_mapping(self.report_trace))
        if not self.locked:
            raise UnlockedReportError("A DecisionLock cannot be created in an unlocked state.")
        if self.report_hash is not None:
            object.__setattr__(self, "report_hash", _required_hash(self.report_hash, "report_hash"))

    @property
    def decision_hash(self) -> str:
        return hash_artifact(self.to_dict(include_report_hash=False))

    @property
    def pre_replay_module_a_hash(self) -> str:
        return self.module_a_pre_replay_hash

    @property
    def post_replay_module_a_hash(self) -> str:
        return self.module_a_post_replay_hash

    @property
    def module_a_pre_replay_packet_hash(self) -> str:
        return self.module_a_pre_replay_hash

    @property
    def module_a_post_replay_packet_hash(self) -> str:
        return self.module_a_post_replay_hash

    @property
    def is_locked(self) -> bool:
        return self.locked

    @classmethod
    def from_artifacts(cls, **kwargs: Any) -> "DecisionLock":
        return create_decision_lock(**kwargs)

    def to_dict(self, *, include_report_hash: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": SCHEMA_VERSION,
            "lock_id": self.lock_id,
            "locked": self.locked,
            "case_id": self.case_id,
            "evidence_snapshot_hash": self.evidence_snapshot_hash,
            "revision_hash": self.revision_hash,
            "state_graph_hash": self.state_graph_hash,
            "module_a_pre_replay_hash": self.module_a_pre_replay_hash,
            "module_a_post_replay_hash": self.module_a_post_replay_hash,
            "agent_decision_hash": self.agent_decision_hash,
            "validator_hash": self.validator_hash,
            "module_b_output_hash": self.module_b_output_hash,
            "replay_audit_hash": self.replay_audit_hash,
            "model_versions": dict(self.model_versions),
            "skill_versions": dict(self.skill_versions),
            "tool_versions": dict(self.tool_versions),
            "report_trace": self.report_trace.to_dict(),
        }
        if include_report_hash:
            result["report_hash"] = self.report_hash
        return result

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "DecisionLock":
        return cls(
            case_id=str(value.get("case_id", "")),
            evidence_snapshot_hash=str(value.get("evidence_snapshot_hash", "")),
            revision_hash=str(value.get("revision_hash", "")),
            state_graph_hash=str(value.get("state_graph_hash", "")),
            module_a_pre_replay_hash=str(value.get("module_a_pre_replay_hash", "")),
            module_a_post_replay_hash=str(value.get("module_a_post_replay_hash", "")),
            agent_decision_hash=str(value.get("agent_decision_hash", "")),
            validator_hash=str(value.get("validator_hash", "")),
            module_b_output_hash=str(value.get("module_b_output_hash", "")),
            model_versions=value.get("model_versions", {}),
            skill_versions=value.get("skill_versions", {}),
            tool_versions=value.get("tool_versions", {}),
            report_trace=ReportTrace.from_mapping(value.get("report_trace", {})),
            replay_audit_hash=value.get("replay_audit_hash"),
            lock_id=str(value.get("lock_id", "")),
            locked=bool(value.get("locked", False)),
            report_hash=value.get("report_hash"),
        )


def create_decision_lock(
    *,
    case_id: str,
    evidence_snapshot: Any = None,
    evidence_snapshot_hash: str | None = None,
    revision: Any = None,
    revision_hash: str | None = None,
    state_graph: Any = None,
    state_graph_hash: str | None = None,
    module_a_pre_replay: Any = None,
    module_a_pre_replay_hash: str | None = None,
    module_a_post_replay: Any = None,
    module_a_post_replay_hash: str | None = None,
    agent_decision: Any = None,
    agent_decision_hash: str | None = None,
    validator_result: Any = None,
    validator_hash: str | None = None,
    module_b_output: Any = None,
    module_b_output_hash: str | None = None,
    replay_audit: Any = None,
    replay_audit_hash: str | None = None,
    model_versions: Mapping[str, str] | None = None,
    skill_versions: Mapping[str, str] | None = None,
    tool_versions: Mapping[str, str] | None = None,
    model_version: str | None = None,
    skill_version: str | None = None,
    tool_version: str | None = None,
    report_trace: ReportTrace | Sequence[Mapping[str, Any]] = (),
    metric_evidence: Any = None,
    segments: Any = None,
    state_cards: Any = None,
    report: Mapping[str, Any] | None = None,
    lock_id: str = "",
) -> DecisionLock:
    """Atomically validate and create a :class:`DecisionLock`.

    All checks happen before the frozen object is constructed.  Callers may
    supply an artifact, its expected hash, or both; supplying both is preferred
    because it detects stale serialized artifacts at the boundary.
    """

    case_id = str(case_id)
    if not case_id.strip():
        raise DecisionLockError("case_id is required.")
    artifacts = {
        "evidence_snapshot": evidence_snapshot,
        "revision": revision,
        "state_graph": state_graph,
        "module_a_pre_replay": module_a_pre_replay,
        "module_a_post_replay": module_a_post_replay,
        "agent_decision": agent_decision,
        "validator": validator_result,
        "module_b_output": module_b_output,
    }
    for name, artifact in artifacts.items():
        _check_case(artifact, case_id, name)
    hashes = {
        "evidence_snapshot": _artifact_identity(evidence_snapshot, "evidence_snapshot", evidence_snapshot_hash),
        "revision": _artifact_identity(revision, "revision", revision_hash),
        "state_graph": _artifact_identity(state_graph, "state_graph", state_graph_hash),
        "module_a_pre_replay": _artifact_identity(module_a_pre_replay, "module_a_pre_replay", module_a_pre_replay_hash),
        "module_a_post_replay": _artifact_identity(module_a_post_replay, "module_a_post_replay", module_a_post_replay_hash),
        "agent_decision": _artifact_identity(agent_decision, "agent_decision", agent_decision_hash),
        "validator": _artifact_identity(validator_result, "validator", validator_hash),
        "module_b_output": _artifact_identity(module_b_output, "module_b_output", module_b_output_hash),
    }
    resolved_replay_audit_hash = None
    if replay_audit is not None or replay_audit_hash is not None:
        resolved_replay_audit_hash = _artifact_identity(
            replay_audit, "replay_audit", replay_audit_hash,
        )
    # A current post-replay packet must be bound to the current snapshot.  A
    # pre-replay packet may point at the parent snapshot when a revision exists.
    post_snapshot = _linked_hash(module_a_post_replay, "evidence_snapshot_hash", "snapshot_hash")
    if post_snapshot is not None and str(post_snapshot) != hashes["evidence_snapshot"]:
        raise HashMismatchError("Post-replay Module A packet is stale for the evidence snapshot.")
    agent_case = _get(agent_decision, "case_id", default=case_id)
    if str(agent_case) != case_id:
        raise HashMismatchError("Agent decision case binding is inconsistent.")
    validator_agent_hash = _linked_hash(validator_result, "agent_decision_hash", "validated_agent_decision_hash")
    if validator_agent_hash is not None and str(validator_agent_hash) != hashes["agent_decision"]:
        raise HashMismatchError("Validator result is bound to a different Agent decision.")
    for artifact_name in ("module_b_output",):
        linked = _linked_hash(artifacts[artifact_name], "module_a_output_hash", "module_a_post_replay_hash")
        if linked is not None and str(linked) != hashes["module_a_post_replay"]:
            raise HashMismatchError("Module B output is bound to a different post-replay Module A packet.")
    if replay_audit is not None:
        audit_state_hash = _linked_hash(replay_audit, "state_hash")
        graph_state_hash = _linked_hash(state_graph, "state_hash")
        if audit_state_hash is None or graph_state_hash is None:
            raise HashMismatchError("Replay audit and StateGraphV2 require explicit state-hash bindings.")
        if str(audit_state_hash) != str(graph_state_hash):
            raise HashMismatchError("Replay audit is bound to a different StateGraphV2 state.")

        audit_revision_hash = _linked_hash(replay_audit, "revision_hash")
        internal_revision_hash = _linked_hash(revision, "revision_hash")
        if audit_revision_hash is None or internal_revision_hash is None:
            raise HashMismatchError("Replay audit and revision require explicit revision-hash bindings.")
        if str(audit_revision_hash) != str(internal_revision_hash):
            raise HashMismatchError("Replay audit is bound to a different evidence revision.")

        audit_packet_hash = _linked_hash(replay_audit, "packet_hash")
        if audit_packet_hash is None or not hasattr(module_a_post_replay, "to_json"):
            raise HashMismatchError("Replay audit requires an explicit post-replay packet binding.")
        expected_packet_hash = _legacy_hash_values([module_a_post_replay.to_json()])
        if str(audit_packet_hash) != expected_packet_hash:
            raise HashMismatchError("Replay audit is bound to a different post-replay Module A packet.")
    trace = validate_report_trace(
        report_trace,
        case_id=case_id,
        revision_hash=hashes["revision"],
        state_graph=state_graph,
        metric_evidence=metric_evidence,
        segments=segments,
        state_cards=state_cards,
    )
    lock = DecisionLock(
        case_id=case_id,
        evidence_snapshot_hash=hashes["evidence_snapshot"],
        revision_hash=hashes["revision"],
        state_graph_hash=hashes["state_graph"],
        module_a_pre_replay_hash=hashes["module_a_pre_replay"],
        module_a_post_replay_hash=hashes["module_a_post_replay"],
        agent_decision_hash=hashes["agent_decision"],
        validator_hash=hashes["validator"],
        module_b_output_hash=hashes["module_b_output"],
        model_versions=model_versions or ({"model": model_version} if model_version else {}),
        skill_versions=skill_versions or ({"skill": skill_version} if skill_version else {}),
        tool_versions=tool_versions or ({"tool": tool_version} if tool_version else {}),
        report_trace=trace,
        replay_audit_hash=resolved_replay_audit_hash,
        lock_id=str(lock_id),
    )
    if report is not None:
        validate_locked_report(report, lock)
        object.__setattr__(lock, "report_hash", hash_artifact(report))
    return lock


def validate_locked_report(report: Mapping[str, Any], lock: DecisionLock) -> None:
    """Reject reports that are not explicitly and correctly bound to a lock."""

    if not isinstance(report, Mapping):
        raise UnlockedReportError("Report must be a mapping.")
    if report.get("locked") is not True and report.get("decision_locked") is not True:
        raise UnlockedReportError("Report is not marked locked.")
    supplied_lock_id = report.get("lock_id", report.get("decision_lock_id"))
    if lock.lock_id and supplied_lock_id != lock.lock_id:
        raise HashMismatchError("Report references a different decision lock.")
    supplied_hash = report.get("decision_lock_hash", report.get("lock_hash"))
    if supplied_hash is not None and str(supplied_hash) != lock.decision_hash:
        raise HashMismatchError("Report references a different decision-lock hash.")
    report_case = report.get("case_id")
    if report_case is not None and str(report_case) != lock.case_id:
        raise HashMismatchError("Report case ID does not match the locked decision.")
    trace = report.get("report_trace", report.get("trace"))
    if trace is None:
        raise UnlockedReportError("Locked report must include the report trace.")
    reported_entries = ReportTrace.from_mapping(trace)
    if canonical_json(reported_entries.to_dict()) != canonical_json(lock.report_trace.to_dict()):
        raise HashMismatchError("Report trace does not match the locked decision.")


def serialize_decision_lock(lock: DecisionLock) -> str:
    """Serialize a lock without exposing mutable implementation containers."""

    if not isinstance(lock, DecisionLock):
        raise TypeError("serialize_decision_lock expects a DecisionLock.")
    return lock.to_json()


def deserialize_decision_lock(payload: str | bytes | Mapping[str, Any]) -> DecisionLock:
    """Deserialize and revalidate the immutable lock envelope."""

    if isinstance(payload, Mapping):
        value = payload
    else:
        value = json.loads(payload)
    if not isinstance(value, Mapping):
        raise DecisionLockError("A serialized decision lock must contain an object.")
    return DecisionLock.from_mapping(value)


validate_decision_lock = validate_locked_report


# Naming aliases used by callers that treat locking as a command.
lock_decision = create_decision_lock
DecisionLockContract = DecisionLock
ReportTraceLink = ReportTraceEntry


__all__ = [
    "DecisionLock", "DecisionLockContract", "DecisionLockError", "EvidenceTraceError",
    "HashMismatchError", "ReportTrace", "ReportTraceEntry", "ReportTraceLink",
    "UnlockedReportError", "canonical_hash", "canonical_json", "content_hash",
    "artifact_hash", "create_decision_lock", "hash_artifact", "lock_decision", "validate_locked_report",
    "validate_report_trace", "serialize_decision_lock", "deserialize_decision_lock",
    "validate_decision_lock",
]
