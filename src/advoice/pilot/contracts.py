"""Versioned, immutable pilot records and fail-closed serialization boundaries.

Truth-bearing records live in the separate scorer-only labels module. Public
records carry pseudonyms and coded reasons, never free-form provider prose.
Hashes cover canonical JSON including schema versions. No provider is called.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
import hashlib
import json
import math
import re
from types import MappingProxyType, UnionType
from typing import Any, ClassVar, Literal, Protocol, Self, Union, get_args, get_origin, get_type_hints, runtime_checkable

from advoice.evidence import (
    ConfoundSets, EvidencePermissions, EvidenceProvenance, MetricEvidenceV2,
    ReferenceMetadata, ReliabilityComponents,
)
from advoice.state_graph import deserialize_state_card_ids


SCHEMA_VERSION = "advoice.pilot.v1"
Partition = Literal["engineering_canary", "development", "holdout", "stress"]
Task = Literal["hc_mci_ad", "hc_ad", "hc_impairment"]
Observation = Literal["observed", "not_observable", "missing", "conflicted"]
Role = Literal["participant", "examiner", "other", "unknown"]
Arm = Literal["B_raw", "B", "J-A", "J-S", "J-AS"]
CLASS_ORDERS = MappingProxyType({
    "hc_mci_ad": ("HC", "MCI", "AD"),
    "hc_ad": ("HC", "AD"),
    "hc_impairment": ("HC", "IMPAIRED"),
})


class PilotContractError(ValueError):
    """Malformed, unsafe, unsupported, or stale pilot contract."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PilotContractError(message)


def _token(value: str) -> None:
    _require(type(value) is str and bool(re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.:+-]*", value)),
             "Expected a non-empty identifier or coded reason, not raw text or a path.")


def _hash(value: str) -> None:
    _require(type(value) is str and bool(re.fullmatch(r"[0-9a-f]{64}", value)),
             "Expected a lowercase SHA-256 hash.")


def _identifiers(values: tuple[str, ...], *, required: bool = False) -> None:
    _require(not required or bool(values), "Identifiers must not be empty.")
    _require(len(set(values)) == len(values), "Identifiers must be unique.")
    for value in values:
        _token(value)


def _pseudonym(value: Any, prefix: str = "sub") -> None:
    _require(isinstance(value, str) and bool(re.fullmatch(prefix + r"_[0-9a-f]{16,64}", value)),
             "Identity fields must use typed pseudonymous identifiers.")


def _task(task: str, class_order: tuple[str, ...]) -> None:
    _require(CLASS_ORDERS.get(task) == class_order, "Task and class order must match exactly.")


def _model(value: str) -> None:
    _token(value)
    _require(not re.search(r"(?:latest|preview|alias)", value, re.I)
             and bool(re.search(r"\d{4}-\d{2}-\d{2}$", value)),
             "model_id must be an exact dated model ID, not a floating alias.")


def _plain(value: Any) -> Any:
    if isinstance(value, Record):
        return value.to_dict()
    if isinstance(value, Mapping):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _json(value: Any) -> str:
    try:
        return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"),
                          ensure_ascii=True, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise PilotContractError("Expected finite JSON data.") from exc


def _digest(value: Any) -> str:
    return hashlib.sha256(_json(value).encode("utf-8")).hexdigest()


def _strict_json_loads(value: str | bytes) -> Any:
    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, item in pairs:
            _require(key not in result, f"Duplicate JSON key: {key}.")
            result[key] = item
        return result

    try:
        return json.loads(value, object_pairs_hook=unique_object)
    except PilotContractError:
        raise
    except (TypeError, ValueError, UnicodeError) as exc:
        raise PilotContractError("Malformed JSON record.") from exc


def _freeze_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        _require(all(type(key) is str for key in value), "JSON keys must be strings.")
        return MappingProxyType({key: _freeze_json(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze_json(item) for item in value)
    _require(value is None or type(value) in (str, int, float, bool), "Expected JSON data.")
    if type(value) is float:
        _require(math.isfinite(value), "Numbers must be finite.")
    return value


def _typed(value: Any, annotation: Any) -> Any:
    """Strictly decode nested records without truthy booleans or numeric coercion."""
    origin, args = get_origin(annotation), get_args(annotation)
    if annotation is Any:
        return _freeze_json(value)
    if origin is Literal:
        _require(any(type(value) is type(item) and value == item for item in args),
                 "Unsupported protocol value.")
        return value
    if origin in (UnionType, Union):
        for candidate in args:
            try:
                return _typed(value, candidate)
            except PilotContractError:
                pass
        raise PilotContractError("Value does not match the declared optional/union type.")
    if origin is tuple:
        _require(isinstance(value, (tuple, list)), "Expected a sequence.")
        return tuple(_typed(item, args[0]) for item in value)
    if origin is Mapping:
        _require(isinstance(value, Mapping), "Expected a mapping.")
        return MappingProxyType({_typed(key, args[0]): _typed(item, args[1])
                                 for key, item in value.items()})
    if isinstance(annotation, type) and issubclass(annotation, Record):
        return value if type(value) is annotation else annotation.from_mapping(value)
    if annotation is float:
        _require(type(value) in (float, int) and math.isfinite(value), "Expected a finite number.")
        return float(value)
    _require(type(value) is annotation, "Value does not match the declared type.")
    return value


def _safe(value: Any, *, inference: bool, dynamic_keys: bool = False) -> None:
    if isinstance(value, Record):
        value = value.to_dict()
    if isinstance(value, Mapping):
        for key, item in value.items():
            _require(type(key) is str, "Payload keys must be strings.")
            normalized = re.sub(r"[^a-z0-9]", "", key.lower())
            forbidden = any(part in normalized for part in (
                "label", "truth", "diagnosis", "transcript", "path", "patient", "split",
                "baseprob", "baseprediction", "email", "address", "fullname", "filename",
            ))
            _require(not forbidden, "Private/truth/base-probability field rejected.")
            _require(dynamic_keys or key in _PUBLIC_KEYS, "Unknown public payload field.")
            if dynamic_keys:
                _token(key)
            if inference:
                _require("probabilit" not in normalized and normalized not in {"probs", "prediction"},
                         "Probabilities cannot enter the inference boundary.")
                _require(not any(normalized.startswith(prefix) for prefix in (
                    "predicted", "calibrator", "fusion", "arm", "fallbackarm",
                    "basefit", "baseprediction",
                )), "Prediction, calibration, fusion, arm, and base-fit fields cannot enter inference.")
            _safe(item, inference=inference, dynamic_keys=key in {
                "ordinal_scores", "state_delta", "replay_delta", "before_state_scores",
                "after_state_scores", "observability", "raw_hashes", "unavailable_reasons",
                "contrast_definitions",
            })
            prefixes = {"subject_id": "sub", "case_id": "case", "session_id": "ses",
                        "source_asset_id": "asset", "source_group_id": "grp", "segment_id": "seg"}
            if key in prefixes and item != "":
                _pseudonym(item, prefixes[key])
            if key == "source_segment_ids":
                _require(isinstance(item, (tuple, list)), "Segment IDs must be a sequence.")
                for identifier in item:
                    _pseudonym(identifier, "seg")
    elif isinstance(value, (tuple, list)):
        for item in value:
            _safe(item, inference=inference)
    elif isinstance(value, str):
        if value in {"(MCI+AD)/2-HC", "AD-MCI", "AD-HC"}:
            return
        if value:
            _token(value)
        _require(not re.search(r"\.(wav|mp3|flac|csv|txt|jsonl|parquet)$", value, re.I),
                 "Raw filenames cannot cross the public boundary.")
    else:
        _freeze_json(value)


def safe_inference_payload(value: Any) -> Any:
    """Reject leakage recursively; do not silently sanitize untrusted providers."""
    _safe(value, inference=True)
    return _plain(value)


@runtime_checkable
class PublishableRecord(Protocol):
    def to_publishable_dict(self) -> dict[str, Any]: ...


@dataclass(frozen=True, kw_only=True)
class Record:
    schema_version: Literal["advoice.pilot.v1"] = SCHEMA_VERSION
    _defer_safety: ClassVar[bool] = False

    def __post_init__(self) -> None:
        hints = get_type_hints(type(self))
        for item in fields(self):
            object.__setattr__(self, item.name, _typed(getattr(self, item.name), hints[item.name]))
        self._validate()
        if not self._defer_safety:
            _safe(self.to_dict(), inference=False)

    def _validate(self) -> None:
        pass

    def to_dict(self) -> dict[str, Any]:
        """Canonical local storage body; not an inference/public projection."""
        return {item.name: _plain(getattr(self, item.name)) for item in fields(self)}

    def to_json(self) -> str:
        return _json(self.to_dict())

    @property
    def content_hash(self) -> str:
        return _digest(self.to_dict())

    def __hash__(self) -> int:
        return int(self.content_hash[:15], 16)

    def __eq__(self, other: object) -> bool:
        return type(self) is type(other) and self.to_json() == other.to_json()

    def to_inference_dict(self) -> dict[str, Any]:
        return safe_inference_payload(self)

    def to_publishable_dict(self) -> dict[str, Any]:
        _safe(self, inference=False)
        return self.to_dict()

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> Self:
        _require(isinstance(value, Mapping), "Expected a record mapping.")
        _require("schema_version" in value, "schema_version is required when decoding records.")
        _require(set(value) <= {item.name for item in fields(cls)}, "Unknown record fields.")
        try:
            return cls(**value)
        except PilotContractError:
            raise
        except (TypeError, ValueError, KeyError, OverflowError) as exc:
            raise PilotContractError("Malformed record.") from exc

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        return cls.from_mapping(_strict_json_loads(value))


@dataclass(frozen=True, kw_only=True)
class SubjectRow(Record):
    dataset_id: str
    subject_id: str
    partition: Partition
    task: Task
    class_order: tuple[str, ...]
    source_group_id: str
    channel: str
    language: str
    task_ids: tuple[str, ...]
    role: Role
    fold_id: str
    raw_hashes: Mapping[str, str]
    source_version: str

    def _validate(self) -> None:
        _pseudonym(self.subject_id)
        _pseudonym(self.source_group_id, "grp")
        for key in ("dataset_id", "channel", "language", "fold_id", "source_version"):
            _token(getattr(self, key))
        _task(self.task, self.class_order)
        _identifiers(self.task_ids, required=True)
        _require(bool(self.raw_hashes), "Source asset hashes are required.")
        for asset, digest in self.raw_hashes.items():
            _pseudonym(asset, "asset")
            _hash(digest)


@dataclass(frozen=True, kw_only=True)
class SourceSegment(Record):
    segment_id: str
    source_asset_id: str
    start_seconds: float
    end_seconds: float
    role: Role

    def _validate(self) -> None:
        _pseudonym(self.segment_id, "seg")
        _pseudonym(self.source_asset_id, "asset")
        _require(0 <= self.start_seconds < self.end_seconds, "Invalid source segment span.")


@dataclass(frozen=True, kw_only=True)
class Observability(Record):
    status: Observation
    reason: str | None

    def _validate(self) -> None:
        _require((self.status == "observed") == (self.reason is None),
                 "Non-observed states require a reason; observed states must not have one.")
        if self.reason is not None:
            _token(self.reason)


@dataclass(frozen=True, kw_only=True)
class MetricEvidenceBody(Record):
    """Frozen serialized MetricEvidenceV2, with a truth-free reference projection."""
    body: Mapping[str, Any]

    @classmethod
    def from_evidence(cls, evidence: MetricEvidenceV2) -> Self:
        _require(isinstance(evidence, MetricEvidenceV2), "Expected MetricEvidenceV2.")
        body = evidence.to_dict()
        # Reference cohort identity is not subject truth, but has no role in an
        # agent payload. Raw transcript identifiers are not public provenance.
        body["reference"].pop("reference_label")
        _require(body["provenance"].pop("transcript_id") is None,
                 "Transcript identifiers require upstream pseudonymization/projection.")
        return cls(body=body)

    def _validate(self) -> None:
        _safe(self.body, inference=True)
        try:
            normalized = MetricEvidenceV2.from_mapping(self.body).to_dict()
        except (TypeError, ValueError, KeyError) as exc:
            raise PilotContractError("Invalid MetricEvidenceV2 body.") from exc
        normalized["reference"].pop("reference_label")
        normalized["provenance"].pop("transcript_id")
        _require(_json(normalized) == _json(self.body),
                 "MetricEvidenceV2 body must be canonical; coercion or dropped fields are forbidden.")


_STATE_CARD_FLOAT_FIELDS = frozenset({
    "state_z", "raw_state_z", "task_state_z", "report_state_z",
    "residual_shrinkage_factor", "severity", "confidence", "report_confidence",
    "missing_fraction", "metric_contribution_clip_z",
})


def _normalize_state_card_json(value: Any, *, key: str | None = None) -> Any:
    """Convert dataframe scalars to canonical, finite JSON before freezing."""
    if isinstance(value, Mapping):
        _require(all(type(item_key) is str for item_key in value),
                 "StateCard keys must be strings.")
        return {item_key: _normalize_state_card_json(item, key=item_key)
                for item_key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_normalize_state_card_json(item) for item in value]
    if not isinstance(value, (str, bytes)) and hasattr(value, "item"):
        try:
            converted = value.item()
        except (TypeError, ValueError):
            converted = value
        if converted is not value:
            return _normalize_state_card_json(converted, key=key)
    if type(value) is float and not math.isfinite(value):
        return None
    if key in _STATE_CARD_FLOAT_FIELDS and value is not None:
        _require(type(value) in (int, float) and type(value) is not bool,
                 f"StateCard {key} must be numeric or null.")
        _require(math.isfinite(value), f"StateCard {key} must be finite or null.")
        return float(value)
    return value


@dataclass(frozen=True, kw_only=True)
class StateCardBody(Record):
    """Lossless local StateCard adapter, checked separately at export boundaries."""
    body: Mapping[str, Any]
    _defer_safety: ClassVar[bool] = True

    @classmethod
    def from_state_card(cls, value: Any) -> Self:
        body = value if isinstance(value, Mapping) else value.to_dict()
        return cls(body=body)

    def __post_init__(self) -> None:
        object.__setattr__(self, "body", _normalize_state_card_json(self.body))
        super().__post_init__()

    def to_state_card(self) -> dict[str, Any]:
        return _plain(self.body)

    @property
    def state_id(self) -> str:
        return self.body["state_id"]

    @property
    def state_z(self) -> float | None:
        return self.body["state_z"]

    @property
    def available(self) -> bool:
        return self.body["available"]

    @property
    def supporting_evidence_ids(self) -> tuple[str, ...]:
        return _card_ids(self.body["supporting_evidence_ids"])

    @property
    def counter_evidence_ids(self) -> tuple[str, ...]:
        return _card_ids(self.body["counter_evidence_ids"])

    def _validate(self) -> None:
        _require({"state_id", "state_z", "available", "supporting_evidence_ids",
                  "counter_evidence_ids"} <= self.body.keys(), "Missing StateCard fields.")
        _token(self.state_id)
        _typed(self.state_z, float | None)
        _typed(self.available, bool)
        _require(self.available == (self.state_z is not None),
                 "Unavailable state values must be null, not neutral zero.")
        self.supporting_evidence_ids
        self.counter_evidence_ids


def _card_ids(value: Any) -> tuple[str, ...]:
    # CSV-backed StateCards contain JSON-encoded ID arrays; retain that encoding.
    try:
        decoded = json.loads(value) if isinstance(value, str) else value
        _require(isinstance(decoded, (tuple, list)), "Expected an ID array.")
        parsed = deserialize_state_card_ids(decoded)
    except (TypeError, ValueError) as exc:
        raise PilotContractError("Invalid StateCard identifiers.") from exc
    _require(len(parsed) == len(decoded), "Duplicate StateCard identifiers.")
    _identifiers(parsed)
    return parsed


def _segment_refs(value: Any) -> set[str]:
    result: set[str] = set()
    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in {"source_segment_ids", "segment_ids"}:
                identifiers = _card_ids(item)
                for identifier in identifiers:
                    _pseudonym(identifier, "seg")
                result.update(identifiers)
            else:
                result.update(_segment_refs(item))
    elif isinstance(value, (tuple, list)):
        for item in value:
            result.update(_segment_refs(item))
    return result


@dataclass(frozen=True, kw_only=True)
class EvidenceSnapshot(Record):
    subject: SubjectRow
    case_id: str
    state_version: str
    reference_fit_id: str
    reference_fit_hash: str
    evidence: tuple[MetricEvidenceBody, ...]
    state_cards: tuple[StateCardBody, ...]
    source_segments: tuple[SourceSegment, ...]
    observability: Mapping[str, Observability]
    confounds: Mapping[str, tuple[str, ...]]
    skill_hash: str
    extractor_hash: str
    snapshot_hash: str = ""
    _defer_safety: ClassVar[bool] = True

    def _validate(self) -> None:
        _pseudonym(self.case_id, "case")
        _token(self.state_version)
        _token(self.reference_fit_id)
        for key in ("reference_fit_hash", "skill_hash", "extractor_hash"):
            _hash(getattr(self, key))
        _require(set(self.confounds) == {"potential", "observed", "ruled_out"},
                 "Confounds require potential, observed, and ruled_out sets.")
        for values in self.confounds.values():
            _identifiers(values)
        confound_sets = [set(values) for values in self.confounds.values()]
        _require(all(not left & right for index, left in enumerate(confound_sets)
                     for right in confound_sets[index + 1:]),
                 "Confound status sets must be pairwise disjoint.")
        evidence_ids = tuple(item.body["evidence_id"] for item in self.evidence)
        _identifiers(evidence_ids)
        segments = {item.segment_id: item for item in self.source_segments}
        _require(len(segments) == len(self.source_segments), "Source segment IDs must be unique.")
        for segment in self.source_segments:
            _require(segment.source_asset_id in self.subject.raw_hashes,
                     "Source segment asset is not bound to the subject.")
        for item in self.evidence:
            _require(item.body["subject_id"] == self.subject.subject_id,
                     "Evidence subject does not match snapshot.")
            _require(item.body["case_id"] in ("", self.case_id),
                     "Evidence case does not match snapshot.")
            _require(item.body["permissions"]["inference"] is True,
                     "Snapshot evidence requires inference permission.")
            provenance = item.body["provenance"]
            refs = _segment_refs(provenance)
            _require(refs <= segments.keys(), "Evidence cites unknown source segments.")
            asset = provenance["source_asset_id"]
            if asset:
                _require(asset in self.subject.raw_hashes, "Evidence source asset is not bound.")
                _require(all(segments[identifier].source_asset_id == asset for identifier in refs),
                         "Evidence source segments do not match its asset.")
        states = tuple(card.state_id for card in self.state_cards)
        _identifiers(states)
        _require(set(self.observability) == set(states), "Observability must cover every state.")
        for card in self.state_cards:
            assigned = set(card.supporting_evidence_ids + card.counter_evidence_ids)
            _require(assigned <= set(evidence_ids),
                     "StateCard cites unknown evidence.")
            _require(all(item.body["state_id"] == card.state_id for item in self.evidence
                         if item.body["evidence_id"] in assigned),
                     "StateCard evidence must belong to the same state.")
            _require(_segment_refs(card.body) <= segments.keys(), "StateCard cites unknown source segments.")
            observed = self.observability[card.state_id].status == "observed"
            _require(card.available == observed,
                     "Observed states require values; non-observed states must be unavailable and null.")
        body = self.to_dict()
        body.pop("snapshot_hash")
        expected = _digest(body)
        _require(not self.snapshot_hash or self.snapshot_hash == expected, "Stale snapshot hash.")
        object.__setattr__(self, "snapshot_hash", expected)

    @property
    def content_hash(self) -> str:
        return self.snapshot_hash


@dataclass(frozen=True, kw_only=True)
class UsageRow(Record):
    request_id: str
    cache_key: str
    cache_status: Literal["hit", "miss", "bypass", "disabled"]
    response_status: Literal[
        "ok", "cache_hit", "transport_failure", "provider_failure", "timeout", "malformed_response"
    ]
    attempt: int
    model_id: str
    input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    cost_usd: float | None
    wall_seconds: float | None
    response_id: str | None
    unavailable_reasons: Mapping[str, str]

    def _validate(self) -> None:
        _token(self.request_id)
        _hash(self.cache_key)
        _model(self.model_id)
        _require(self.attempt >= 1, "Usage attempts start at one.")
        _require((self.cache_status == "hit") == (self.response_status == "cache_hit"),
                 "Cache hits require cache_hit response status and vice versa.")
        nullable = ("input_tokens", "output_tokens", "reasoning_tokens",
                    "cost_usd", "wall_seconds", "response_id")
        _require(set(self.unavailable_reasons) ==
                 {key for key in nullable if getattr(self, key) is None},
                 "Exactly the unavailable usage fields require reasons.")
        for reason in self.unavailable_reasons.values():
            _token(reason)
        for key in nullable:
            value = getattr(self, key)
            if value is not None:
                if key == "response_id":
                    _token(value)
                else:
                    _require(value >= 0, "Usage cannot be negative.")


@dataclass(frozen=True, kw_only=True)
class StateJudgment(Record):
    state_id: str
    status: Observation
    score: int | None
    citations: tuple[str, ...] = ()
    reason: str | None = None

    def _validate(self) -> None:
        _token(self.state_id)
        _identifiers(self.citations)
        if self.status == "observed":
            _require(self.score is not None and 0 <= self.score <= 4 and bool(self.citations)
                     and self.reason is None,
                     "Observed state requires an ordinal score and citations.")
        else:
            _require(self.score is None and bool(self.reason),
                     "Non-observed states require null scores and a reason.")
        if self.reason is not None:
            _token(self.reason)


@dataclass(frozen=True, kw_only=True)
class ReviewOperation(Record):
    operation_id: str
    state_id: str
    action: Literal["deterministic_remeasurement", "source_role_span_correction",
                    "unsupported_interpretation_flag", "confound_flag"]
    citations: tuple[str, ...]
    source_segment_ids: tuple[str, ...]
    reason: str
    method_hash: str | None = None
    corrected_role: Role | None = None
    span_start: float | None = None
    span_end: float | None = None
    flag_code: str | None = None

    def _validate(self) -> None:
        for value in (self.operation_id, self.state_id, self.reason):
            _token(value)
        _identifiers(self.citations, required=True)
        _identifiers(self.source_segment_ids, required=True)
        for identifier in self.source_segment_ids:
            _pseudonym(identifier, "seg")
        has_span = self.span_start is not None or self.span_end is not None
        if has_span:
            _require(self.span_start is not None and self.span_end is not None
                     and 0 <= self.span_start < self.span_end, "Invalid correction span.")
        if self.action == "deterministic_remeasurement":
            _hash(self.method_hash)
            _require(self.corrected_role is None and not has_span and self.flag_code is None,
                     "Remeasurement only carries a deterministic method.")
        elif self.action == "source_role_span_correction":
            _require(self.method_hash is None and self.flag_code is None
                     and (has_span or self.corrected_role is not None),
                     "Source correction requires a role or span correction.")
        else:
            _require(self.method_hash is None and self.corrected_role is None and not has_span,
                     "Flags cannot carry methods or source corrections.")
            _token(self.flag_code)

    def validate_snapshot(self, snapshot: EvidenceSnapshot) -> None:
        states = {card.state_id: card for card in snapshot.state_cards}
        _require(self.state_id in states, "Unknown reviewed state.")
        card = states[self.state_id]
        linked = set(card.supporting_evidence_ids + card.counter_evidence_ids)
        _require(set(self.citations) <= linked, "Operation cites evidence not linked to its state.")
        segments = {item.segment_id: item for item in snapshot.source_segments}
        requested = set(self.source_segment_ids)
        _require(requested <= segments.keys(), "Operation cites unknown source segments.")
        backing = _segment_refs(card.body)
        for evidence in snapshot.evidence:
            if evidence.body["evidence_id"] in self.citations:
                backing.update(_segment_refs(evidence.body["provenance"]))
        _require(requested <= backing, "Operation source segments are not backed by its evidence/state.")
        if self.span_start is not None:
            _require(all(segments[key].start_seconds <= self.span_start
                         and self.span_end <= segments[key].end_seconds for key in requested),
                     "Correction span exceeds its source segment.")


def _operations(items: tuple[ReviewOperation, ...]) -> None:
    _require(len(items) <= 3, "At most three operations are permitted.")
    _identifiers(tuple(item.operation_id for item in items))


@dataclass(frozen=True, kw_only=True)
class AgentAssessment(Record):
    subject_id: str
    case_id: str
    task: Task
    class_order: tuple[str, ...]
    snapshot_hash: str
    revision: Literal["v0", "v1"]
    ordinal_scores: Mapping[str, int] | None
    citations: tuple[str, ...]
    state_judgments: tuple[StateJudgment, ...]
    operations: tuple[ReviewOperation, ...]
    model_id: str
    usage: UsageRow
    usage_id: str
    status: Literal["ok", "failed"]
    failure_reason: str | None = None
    ad_stage: None = None

    def _validate(self) -> None:
        _pseudonym(self.subject_id)
        _pseudonym(self.case_id, "case")
        _task(self.task, self.class_order)
        _hash(self.snapshot_hash)
        _model(self.model_id)
        _require(self.model_id == self.usage.model_id, "Assessment and usage model must match.")
        _hash(self.usage_id)
        _require(self.usage_id == self.usage.content_hash,
                 "Assessment usage_id must bind the exact usage record.")
        _operations(self.operations)
        _require(self.revision != "v1" or not self.operations, "v1 must contain zero operations.")
        _validity(self.status == "ok", self.failure_reason)
        if self.status == "ok":
            _require(self.usage.response_status in {"ok", "cache_hit"},
                     "Successful assessments require a successful provider or cache response.")
        if self.status == "failed":
            _require(self.ordinal_scores is None and not self.citations
                     and not self.state_judgments and not self.operations,
                     "Failed assessments must not claim scores, judgments, citations, or operations.")
            return
        _require(self.ordinal_scores is not None and set(self.ordinal_scores) == set(self.class_order)
                 and all(0 <= score <= 4 for score in self.ordinal_scores.values()),
                 "Ordinal scores must cover the class order with integers from 0 to 4.")
        object.__setattr__(self, "ordinal_scores", MappingProxyType(
            {key: self.ordinal_scores[key] for key in self.class_order}))
        _identifiers(self.citations, required=True)
        _identifiers(tuple(item.state_id for item in self.state_judgments))

    def validate_snapshot(self, snapshot: EvidenceSnapshot) -> None:
        _require(self.snapshot_hash == snapshot.snapshot_hash, "Assessment has a stale snapshot hash.")
        _require(self.case_id == snapshot.case_id, "Assessment case does not match snapshot.")
        _require((self.subject_id, self.task, self.class_order) ==
                 (snapshot.subject.subject_id, snapshot.subject.task, snapshot.subject.class_order),
                 "Assessment does not match snapshot subject/task.")
        if self.status == "failed":
            return
        evidence_ids = {item.body["evidence_id"] for item in snapshot.evidence}
        states = {card.state_id: card for card in snapshot.state_cards}
        _require(set(self.citations) <= evidence_ids, "Assessment cites unknown evidence.")
        judgment_ids = [item.state_id for item in self.state_judgments]
        _require(len(judgment_ids) == len(states) and set(judgment_ids) == set(states),
                 "Every snapshot state requires exactly one judgment.")
        for item in (*self.state_judgments, *self.operations):
            _require(item.state_id in states, "Unknown reviewed state.")
            card = states[item.state_id]
            linked = set(card.supporting_evidence_ids + card.counter_evidence_ids)
            _require(set(item.citations) <= linked & set(self.citations),
                     "State citations must be linked and included in the assessment.")
            if isinstance(item, StateJudgment):
                _require(item.status == snapshot.observability[item.state_id].status,
                         "State judgment status must match snapshot observability.")
                if item.status == "observed":
                    _require(card.available, "Unavailable state cannot be observed.")
            if isinstance(item, ReviewOperation):
                item.validate_snapshot(snapshot)

    @classmethod
    def from_provider_payload(cls, value: Mapping[str, Any], snapshot: EvidenceSnapshot) -> Self:
        safe_inference_payload(value)
        result = cls.from_mapping(value)
        result.validate_snapshot(snapshot)
        return result


@dataclass(frozen=True, kw_only=True)
class RejectedOperation(Record):
    operation: ReviewOperation
    reason: str

    def _validate(self) -> None:
        _token(self.reason)


@dataclass(frozen=True, kw_only=True)
class ReplayResult(Record):
    before: EvidenceSnapshot
    after: EvidenceSnapshot
    parent_hash: str
    child_hash: str
    accepted_ops: tuple[ReviewOperation, ...]
    rejected_ops: tuple[RejectedOperation, ...]
    invalidated_ids: tuple[str, ...]
    recomputed_ids: tuple[str, ...]
    before_state_scores: Mapping[str, float | None]
    after_state_scores: Mapping[str, float | None]
    state_delta: Mapping[str, float | None]
    replay_model_id: str
    repair_batch_id: str | None
    executor_registry_hash: str
    authorized_operations: Mapping[str, str]
    valid: bool
    reason: str | None = None
    _defer_safety: ClassVar[bool] = True

    def _validate(self) -> None:
        _hash(self.parent_hash)
        _hash(self.child_hash)
        _require(self.parent_hash == self.before.snapshot_hash
                 and self.child_hash == self.after.snapshot_hash, "Replay snapshot hashes differ.")
        _token(self.replay_model_id)
        _hash(self.executor_registry_hash)
        _validity(self.valid, self.reason)
        operations = self.accepted_ops + tuple(item.operation for item in self.rejected_ops)
        _operations(operations)
        for operation_id, operation_hash in self.authorized_operations.items():
            _token(operation_id)
            _hash(operation_hash)
        expected_authorization = {
            item.operation_id: item.content_hash for item in self.accepted_ops
        }
        _require(dict(self.authorized_operations) == expected_authorization,
                 "Accepted operations must exactly match content-bound executor authorization.")
        if self.accepted_ops:
            _token(self.repair_batch_id)
        else:
            _require(self.repair_batch_id is None and not self.authorized_operations,
                     "A repair batch and authorization require accepted operations.")
        for operation in self.accepted_ops:
            operation.validate_snapshot(self.before)
        for key in ("subject", "case_id", "reference_fit_id", "reference_fit_hash",
                    "skill_hash", "extractor_hash"):
            _require(getattr(self.before, key) == getattr(self.after, key),
                     "Replay cannot change subject, case, partition, fold, or producing fit.")
        before = {card.state_id: card.state_z for card in self.before.state_cards}
        after = {card.state_id: card.state_z for card in self.after.state_cards}
        _require(before == self.before_state_scores and after == self.after_state_scores,
                 "Replay scores do not match the bound snapshots.")
        _require(set(before) == set(after) == set(self.state_delta), "Replay state identities differ.")
        for key, delta in self.state_delta.items():
            expected = None if before[key] is None or after[key] is None else after[key] - before[key]
            _require(delta == expected, "Replay delta does not match the bound snapshots.")
        before_evidence = {item.body["evidence_id"]: item for item in self.before.evidence}
        after_evidence = {item.body["evidence_id"]: item for item in self.after.evidence}
        _identifiers(self.invalidated_ids)
        _identifiers(self.recomputed_ids)
        _require(set(self.invalidated_ids) <= before_evidence.keys()
                 and set(self.recomputed_ids) <= after_evidence.keys(),
                 "Replay audit IDs cite unknown evidence.")
        invalidated = set(self.invalidated_ids)
        recomputed = set(self.recomputed_ids)
        audited = invalidated | recomputed
        evidence_operations = tuple(
            item for item in self.accepted_ops
            if item.action in {"deterministic_remeasurement", "source_role_span_correction"}
        )
        directly_authorized = {
            identifier for item in evidence_operations for identifier in item.citations
        }
        descendant_states = {item.state_id for item in evidence_operations}
        descendant_recomputations = {
            identifier for identifier in recomputed
            if after_evidence[identifier].body["state_id"] in descendant_states
        }
        _require(invalidated <= directly_authorized,
                 "Invalidated evidence must be cited by an evidence-mutating operation.")
        _require(recomputed <= directly_authorized | descendant_recomputations,
                 "Recomputed evidence must be cited or be a same-state dependency descendant.")
        changed_evidence = {key for key in before_evidence.keys() | after_evidence.keys()
                            if before_evidence.get(key) != after_evidence.get(key)}
        _require(changed_evidence <= audited,
                 "Changed evidence must be recorded in the replay audit.")
        _require(changed_evidence <= directly_authorized | descendant_recomputations,
                 "Changed evidence is outside the accepted operation domains.")

        accepted_states = {item.state_id for item in self.accepted_ops}
        before_cards = {item.state_id: item for item in self.before.state_cards}
        after_cards = {item.state_id: item for item in self.after.state_cards}
        changed_cards = {key for key in before_cards if before_cards[key] != after_cards[key]}
        changed_observability = {key for key in self.before.observability
                                 if self.before.observability[key] != self.after.observability[key]}
        _require(changed_cards | changed_observability <= accepted_states,
                 "Changed StateCards and observability require accepted operations for those states.")

        before_segments = {item.segment_id: item for item in self.before.source_segments}
        after_segments = {item.segment_id: item for item in self.after.source_segments}
        changed_segments = {key for key in before_segments.keys() | after_segments.keys()
                            if before_segments.get(key) != after_segments.get(key)}
        authorized_segments = {segment_id for item in self.accepted_ops
                               if item.action == "source_role_span_correction"
                               for segment_id in item.source_segment_ids}
        _require(changed_segments <= authorized_segments,
                 "Source segment changes require a source_role_span_correction on each segment.")
        _require(self.before.confounds == self.after.confounds
                 or any(item.action == "confound_flag" for item in self.accepted_ops),
                 "Confound changes require an accepted confound_flag.")
        _require(not self.meaningful_change or (self.valid and bool(self.accepted_ops)),
                 "Meaningful snapshot changes require valid accepted operations.")
        _require(self.before.state_version == self.after.state_version
                 or (self.meaningful_change and self.valid and bool(self.accepted_ops)),
                 "State version may change only with a meaningful accepted change.")
        _require(self.parent_hash == self.child_hash
                 or self.meaningful_change
                 or self.before.state_version != self.after.state_version,
                 "Snapshot hashes changed without an auditable replay change.")

    @property
    def meaningful_change(self) -> bool:
        return any(getattr(self.before, key) != getattr(self.after, key) for key in (
            "evidence", "state_cards", "observability", "confounds", "source_segments",
        ))


def _validity(valid: bool, reason: str | None) -> None:
    _require(valid == (reason is None), "Invalid results require a reason; valid results must not have one.")
    if reason is not None:
        _token(reason)


def _probabilities(values: tuple[float, ...], order: tuple[str, ...]) -> None:
    _require(len(values) == len(order) and all(0 <= value <= 1 for value in values)
             and math.isclose(sum(values), 1.0, abs_tol=1e-9, rel_tol=0),
             "Probabilities must match class order and sum to one.")


@dataclass(frozen=True, kw_only=True)
class ArmAvailability(Record):
    available: bool
    valid: bool
    fallback_reason: str | None
    fallback_arm: Arm | None

    def _validate(self) -> None:
        _require(not self.valid or self.available, "Valid arms must be available.")
        _validity(self.valid, self.fallback_reason)
        if self.valid:
            _require(self.fallback_arm is None, "Valid arms do not fall back.")


@dataclass(frozen=True, kw_only=True)
class FusionRow(Record):
    """Local fusion/storage record; base probabilities never enter provider payloads."""
    subject: SubjectRow
    dataset_id: str
    subject_id: str
    partition: Partition
    fold_id: str
    base_fit_id: str
    base_fit_hash: str
    reference_fit_id: str
    reference_fit_hash: str
    base_probabilities: tuple[float, ...]
    assessment_v0: AgentAssessment | None
    assessment_v1: AgentAssessment | None
    replay: ReplayResult | None
    parent_snapshot_hash: str
    child_snapshot_hash: str
    replay_delta: Mapping[str, float | None]
    v1_required: bool
    j_a: ArmAvailability
    j_s: ArmAvailability
    j_as: ArmAvailability
    validity: Literal["valid", "partial", "invalid"]
    _defer_safety: ClassVar[bool] = True

    def _validate(self) -> None:
        for key in ("fold_id", "base_fit_id", "reference_fit_id"):
            _token(getattr(self, key))
        for key in ("base_fit_hash", "reference_fit_hash", "parent_snapshot_hash",
                    "child_snapshot_hash"):
            _hash(getattr(self, key))
        for key in ("dataset_id", "subject_id", "partition", "fold_id"):
            _require(getattr(self, key) == getattr(self.subject, key),
                     "Fusion subject metadata differs.")
        _probabilities(self.base_probabilities, self.subject.class_order)
        for assessment, revision in ((self.assessment_v0, "v0"), (self.assessment_v1, "v1")):
            if assessment is not None:
                _require(assessment.revision == revision, "Fusion assessment revision differs.")
        replay_ops: tuple[ReviewOperation, ...] = ()
        required = False
        if self.replay is not None:
            for key in ("subject", "reference_fit_id", "reference_fit_hash"):
                _require(getattr(self, key) == getattr(self.replay.before, key),
                         "Fusion replay binding differs.")
            _require(self.parent_snapshot_hash == self.replay.parent_hash
                     and self.child_snapshot_hash == self.replay.child_hash
                     and self.replay_delta == self.replay.state_delta, "Fusion replay audit differs.")
            if self.assessment_v0 is not None:
                self.assessment_v0.validate_snapshot(self.replay.before)
            if self.assessment_v1 is not None:
                self.assessment_v1.validate_snapshot(self.replay.after)
            replay_ops = self.replay.accepted_ops + tuple(
                item.operation for item in self.replay.rejected_ops)
            required = (self.replay.valid and bool(self.replay.accepted_ops)
                        and self.replay.meaningful_change)
        else:
            _require(self.assessment_v1 is None, "Missing replay cannot carry a v1 assessment.")
            if self.assessment_v0 is not None:
                _require(
                    self.assessment_v0.snapshot_hash == self.parent_snapshot_hash
                    and self.assessment_v0.subject_id == self.subject_id
                    and self.assessment_v0.task == self.subject.task
                    and self.assessment_v0.class_order == self.subject.class_order,
                    "Replay-free J-A assessment must bind the fusion subject and parent snapshot.",
                )
        proposed = self.assessment_v0.operations if self.assessment_v0 is not None else ()
        if self.replay is not None:
            _require({item.operation_id: item for item in replay_ops} ==
                     {item.operation_id: item for item in proposed},
                     "Replay operations differ from the v0 assessment.")
        _require(self.v1_required == required, "v1 is required only for accepted state changes.")
        _require(required or self.assessment_v1 is None, "No-op replay must not include v1.")

        ja_reason = ("v0_missing" if self.assessment_v0 is None else
                     "v0_failed" if self.assessment_v0.status == "failed" else None)
        replay_inputs_missing = (
            self.replay is not None
            and all(value is None for values in (
                self.replay.before_state_scores, self.replay.after_state_scores,
                self.replay.state_delta,
            ) for value in values.values())
        )
        js_reason = ("replay_missing" if self.replay is None else
                     "replay_failed" if not self.replay.valid else
                     "replay_inputs_missing" if replay_inputs_missing else None)
        jas_reason = ja_reason
        if jas_reason is None and required:
            jas_reason = ("v1_missing" if self.assessment_v1 is None else
                          "v1_failed" if self.assessment_v1.status == "failed" else None)
        jas_reason = jas_reason or js_reason
        self._check_arm(self.j_a, ja_reason, "B")
        self._check_arm(self.j_s, js_reason, "B")
        self._check_arm(self.j_as, jas_reason, "B")
        arm_validity = (self.j_a.valid, self.j_s.valid, self.j_as.valid)
        expected = "valid" if all(arm_validity) else ("partial" if any(arm_validity) else "invalid")
        _require(self.validity == expected, "Fusion validity must reflect independent arm validity.")

    @staticmethod
    def _check_arm(arm: ArmAvailability, reason: str | None, fallback: Arm) -> None:
        _require(arm.available == arm.valid == (reason is None)
                 and arm.fallback_reason == reason
                 and arm.fallback_arm == (fallback if reason is not None else None),
                 "Arm availability/fallback does not match its dependencies.")

    def to_fusion_dict(self) -> dict[str, Any]:
        return self.to_dict()

    def to_inference_dict(self) -> dict[str, Any]:
        raise PilotContractError("Fusion records are storage/fusion-only.")

    def to_publishable_dict(self) -> dict[str, Any]:
        raise PilotContractError("Fusion records are storage/fusion-only.")


@dataclass(frozen=True, kw_only=True)
class PredictionRow(Record):
    subject: SubjectRow
    fusion_hash: str
    arm: Arm
    class_order: tuple[str, ...]
    probabilities: tuple[float, ...] | None
    predicted: str | None
    status: Literal["ok", "fallback", "unavailable"]
    calibrator_id: str | None
    trace_ids: tuple[str, ...]
    fallback_reason: str | None = None
    fallback_arm: Arm | None = None
    ad_stage: None = None

    def _validate(self) -> None:
        _hash(self.fusion_hash)
        _require(self.class_order == self.subject.class_order, "Prediction class order differs.")
        _identifiers(self.trace_ids, required=True)
        if self.status == "ok":
            _require(self.fallback_reason is None and self.fallback_arm is None,
                     "Successful predictions cannot carry fallback metadata.")
        else:
            _token(self.fallback_reason)
        if self.status == "unavailable":
            _require(self.probabilities is None and self.predicted is None
                     and self.calibrator_id is None and self.fallback_arm is None,
                     "Unavailable predictions must remain null without a fallback arm.")
            return
        if self.status == "fallback":
            allowed = {"B_raw": (), "B": (), "J-A": ("B",), "J-S": ("B",), "J-AS": ("B",)}
            _require(self.fallback_arm in allowed[self.arm], "Invalid prediction fallback arm.")
        _token(self.calibrator_id)
        _require(self.probabilities is not None, "Available predictions require probabilities.")
        _probabilities(self.probabilities, self.class_order)
        selected = self.class_order[max(range(len(self.probabilities)), key=self.probabilities.__getitem__)]
        _require(self.predicted == selected, "Prediction must match the class-ordered argmax.")

    def validate_fusion(self, fusion: FusionRow) -> None:
        _require(self.fusion_hash == fusion.content_hash and self.subject == fusion.subject,
                 "Prediction does not match its bound fusion.")
        if self.arm in ("B_raw", "B"):
            _require(self.status == "ok", "The bound base arm is available.")
            return
        availability = {"J-A": fusion.j_a, "J-S": fusion.j_s, "J-AS": fusion.j_as}[self.arm]
        expected = "ok" if availability.valid else (
            "fallback" if availability.fallback_arm is not None else "unavailable")
        _require(self.status == expected
                 and self.fallback_arm == availability.fallback_arm
                 and self.fallback_reason == availability.fallback_reason,
                 "Prediction status/fallback differs from its fusion arm.")


@dataclass(frozen=True, kw_only=True)
class DatasetManifest(Record):
    dataset_id: str
    manifest_id: str
    manifest_hash: str
    task: Task
    class_order: tuple[str, ...]

    def _validate(self) -> None:
        _token(self.dataset_id)
        _token(self.manifest_id)
        _hash(self.manifest_hash)
        _task(self.task, self.class_order)


@dataclass(frozen=True, kw_only=True)
class Route(Record):
    dataset_id: str
    observation_route: str
    target_route: Literal["diagnosis"]

    def _validate(self) -> None:
        _token(self.dataset_id)
        _token(self.observation_route)


@dataclass(frozen=True, kw_only=True)
class FusionConfig(Record):
    family: Literal["regularized_two_head_logit_stacking"]
    loss: Literal["mean_cross_entropy"]
    l2_lambda: float
    prior: tuple[float, ...]
    non_intercept_nonnegative: bool
    base_log_odds_scaling: Literal["none"]
    other_features_scaling: Literal["development_only"]
    hyperparameter_search: bool
    optimizer: Literal["constrained"]
    probability_clip: tuple[float, ...]
    contrast_definitions: Mapping[str, str]
    feature_order: tuple[str, ...]

    def _validate(self) -> None:
        _require(self.l2_lambda == 1.0
                 and self.prior == (1.0, 0.0, 0.0, 0.0)
                 and self.non_intercept_nonnegative is True
                 and self.hyperparameter_search is False
                 and self.probability_clip == (1e-6, 1 - 1e-6)
                 and self.contrast_definitions == {
                     "impaired": "(MCI+AD)/2-HC", "stage": "AD-MCI", "binary": "AD-HC",
                 }
                 and self.feature_order == (
                     "base_log_odds", "agent_contrast", "replay_log_odds_delta", "intercept",
                 ),
                 "Fusion policy is fixed and cannot be relaxed.")


@dataclass(frozen=True, kw_only=True)
class CacheConfig(Record):
    enabled: bool
    namespace: str

    def _validate(self) -> None:
        _token(self.namespace)


@dataclass(frozen=True, kw_only=True)
class Pricing(Record):
    source: str
    date: str
    input_usd_per_million: float
    output_usd_per_million: float
    reasoning_usd_per_million: float

    def _validate(self) -> None:
        _require(bool(self.source) and bool(self.date), "Pricing source and date are required.")
        _require(min(self.input_usd_per_million, self.output_usd_per_million,
                     self.reasoning_usd_per_million) >= 0,
                 "Pricing values must be nonnegative.")


@dataclass(frozen=True, kw_only=True)
class ExecutionCaps(Record):
    concurrency: int
    request_timeout_seconds: int
    semantic_retry_max: int
    pause_after_consecutive_transport_failures: int
    core_call_cap: int
    comparison_call_cap: int
    canary_call_cap: int
    total_call_cap: int
    comparison_structured_cell_max_calls: int
    comparison_flat_cell_max_calls: int
    comparison_subject_max_calls: int
    max_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_reasoning_tokens: int
    max_total_tokens: int
    max_usd: float

    def _validate(self) -> None:
        _require(1 <= self.concurrency <= 2
                 and 1 <= self.request_timeout_seconds <= 180
                 and 0 <= self.semantic_retry_max <= 1
                 and 1 <= self.pause_after_consecutive_transport_failures <= 2,
                 "Execution safety settings exceed their allowed ceilings.")
        maxima = {
            "core_call_cap": 742,
            "comparison_call_cap": 144,
            "canary_call_cap": 24,
            "total_call_cap": 910,
            "comparison_structured_cell_max_calls": 2,
            "comparison_flat_cell_max_calls": 1,
            "comparison_subject_max_calls": 6,
        }
        _require(all(0 < getattr(self, key) <= maximum for key, maximum in maxima.items()),
                 "Pilot semantic call limits must be positive and within frozen ceilings.")
        _require(self.core_call_cap + self.comparison_call_cap + self.canary_call_cap
                 <= self.total_call_cap,
                 "Component call caps cannot exceed the total semantic call cap.")
        _require(min(self.max_calls, self.max_input_tokens, self.max_output_tokens,
                     self.max_reasoning_tokens, self.max_total_tokens, self.max_usd) >= 0,
                 "Execution caps must be nonnegative.")
        _require(self.max_calls <= self.total_call_cap,
                 "Runtime max_calls cannot exceed the frozen total call cap.")


@dataclass(frozen=True, kw_only=True)
class ResolvedConfig(Record):
    """Fully resolved execution identity. A disabled YAML template is not resolved."""
    dataset_manifests: tuple[DatasetManifest, ...]
    routes: tuple[Route, ...]
    provider: str
    model_id: str
    skill_hash: str
    prompt_hash: str
    seed: int
    pricing: Pricing
    fusion: FusionConfig
    cache: CacheConfig
    limits: ExecutionCaps
    paid_execution: bool = False

    def _validate(self) -> None:
        _token(self.provider)
        _model(self.model_id)
        _hash(self.skill_hash)
        _hash(self.prompt_hash)
        _require(self.seed >= 0, "Seed must be nonnegative.")
        datasets = [item.dataset_id for item in self.dataset_manifests]
        routes = [item.dataset_id for item in self.routes]
        _require(bool(datasets) and len(set(datasets)) == len(datasets)
                 and len(routes) == len(set(routes)) and set(datasets) == set(routes),
                 "Each unique dataset manifest requires exactly one route.")
        if self.paid_execution:
            _require(min(self.limits.max_calls, self.limits.max_input_tokens,
                         self.limits.max_output_tokens, self.limits.max_reasoning_tokens,
                         self.limits.max_total_tokens, self.limits.max_usd) > 0,
                     "Paid execution requires explicit positive caps.")
        else:
            _require(self.limits.max_calls == 0, "Disabled paid execution requires max_calls=0.")


# An allowlist, in addition to leakage-key rejection, keeps novel provider
# metadata out. Only score/delta maps accept dynamic, still-screened keys.
_RECORD_TYPES = (
    SubjectRow, SourceSegment, Observability, MetricEvidenceBody, StateCardBody, EvidenceSnapshot,
    UsageRow, StateJudgment, ReviewOperation, RejectedOperation, AgentAssessment, ReplayResult,
    FusionRow, ArmAvailability, PredictionRow, DatasetManifest, Route, FusionConfig, CacheConfig,
    Pricing, ExecutionCaps, ResolvedConfig,
)
for _record_type in _RECORD_TYPES:
    _record_type.__eq__ = Record.__eq__
    _record_type.__hash__ = Record.__hash__


_PUBLIC_KEYS = frozenset(
    item.name
    for cls in (
        *_RECORD_TYPES, MetricEvidenceV2, ReferenceMetadata,
        ReliabilityComponents, ConfoundSets, EvidenceProvenance, EvidencePermissions,
    )
    for item in fields(cls)
) | {
    "inference_permission", "report_permission", "state_z", "available",
    "supporting_evidence_ids", "counter_evidence_ids", "provenance_trace",
    "confidence", "raw_state_z", "task_scope",
}
