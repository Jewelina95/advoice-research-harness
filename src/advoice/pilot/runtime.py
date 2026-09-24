"""Blind, bounded Agent assessment and deterministic replay for the pilot.

This module is an additive correlated-fusion route.  It deliberately does not
modify or call the historical strict-independent authority runtime.  Providers
and replay executors are injected protocols so tests and offline validation use
no network or patient API calls.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
import hashlib
import json
import math
import re
import threading
import time
from types import MappingProxyType
from typing import Any, Literal, Protocol, runtime_checkable

from advoice.condition_c import clean_model_transcript
from advoice.transcript_sanitization import sanitize_segment_payload

from .contracts import (
    SCHEMA_VERSION,
    AgentAssessment,
    EvidenceSnapshot,
    PilotContractError,
    RejectedOperation,
    ReplayResult,
    ReviewOperation,
    StateJudgment,
    UsageRow,
)


RUNTIME_VERSION = "advoice.pilot.runtime.v1"
CORRELATED_FUSION_ROUTE = "pilot_correlated_evidence_v1"
_HASH_RE = re.compile(r"[0-9a-f]{64}")
_TOKEN_RE = re.compile(r"[A-Za-z0-9_][A-Za-z0-9_.:+-]*")
_FORBIDDEN_KEY_PARTS = (
    "groundtruth", "truelabel", "trueclass", "diagnosis", "patientname",
    "participantname", "fullname", "sourcepath", "filepath", "filename",
    "baseprob", "baseprediction", "calibratedprob", "supervisedprob",
)
_ALLOWED_RESPONSE_KEYS = frozenset({
    "snapshot_hash", "revision", "ordinal_scores", "citations",
    "state_judgments", "operations", "status", "failure_reason",
})
_ALLOWED_OPERATIONS = frozenset({
    "deterministic_remeasurement", "source_role_span_correction",
    "unsupported_interpretation_flag", "confound_flag",
})


class RuntimeValidationError(ValueError):
    """Unsafe, stale, malformed, unauthorized, or over-budget runtime input."""


class ProviderTransportError(RuntimeError):
    """Retryable provider transport failure."""


class ProviderTimeoutError(ProviderTransportError):
    """Retryable provider timeout."""


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True,
                      allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(item) for item in value]
    return value


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(key): _freeze(item) for key, item in value.items()})
    if isinstance(value, (tuple, list)):
        return tuple(_freeze(item) for item in value)
    return value


def _require_hash(value: str, field_name: str) -> None:
    if not isinstance(value, str) or not _HASH_RE.fullmatch(value):
        raise RuntimeValidationError(f"{field_name} must be a lowercase SHA-256 hash.")


def _coded_reason(value: str) -> str:
    normalized = re.sub(r"[^A-Za-z0-9_.:+-]+", "_", str(value)).strip("_").lower()
    return normalized[:96] or "runtime_failure"


@dataclass(frozen=True, slots=True)
class TranscriptSpan:
    segment_id: str
    task_id: str
    role: Literal["participant", "examiner", "other", "unknown"]
    text: str
    public_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not all(isinstance(value, str) for value in (
            self.segment_id, self.task_id, self.role, self.text,
        )):
            raise RuntimeValidationError("TranscriptSpan fields must be strings.")
        if self.role not in {"participant", "examiner", "other", "unknown"}:
            raise RuntimeValidationError("TranscriptSpan role is unsupported.")
        if not self.segment_id or not self.task_id:
            raise RuntimeValidationError("TranscriptSpan requires segment and task IDs.")
        if any(not isinstance(value, str) or not value.strip() for value in self.public_names):
            raise RuntimeValidationError("TranscriptSpan public names must be non-empty strings.")


@dataclass(frozen=True, slots=True)
class ProviderUsage:
    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None

    def __post_init__(self) -> None:
        for value in (self.input_tokens, self.output_tokens, self.reasoning_tokens):
            if value is not None and (type(value) is not int or value < 0):
                raise RuntimeValidationError("Provider token counts must be nonnegative integers or null.")


@dataclass(frozen=True, slots=True)
class ProviderResponse:
    payload: Mapping[str, Any] | str | bytes | None
    usage: ProviderUsage | None
    response_id: str | None
    reported_cost_usd: float | None
    status: Literal["ok", "provider_failure"] = "ok"

    def __post_init__(self) -> None:
        if self.status not in {"ok", "provider_failure"}:
            raise RuntimeValidationError("Provider response status is unsupported.")
        if self.reported_cost_usd is not None and (
            type(self.reported_cost_usd) not in (int, float)
            or not math.isfinite(self.reported_cost_usd)
            or self.reported_cost_usd < 0
        ):
            raise RuntimeValidationError("Reported provider cost must be finite and nonnegative.")


@dataclass(frozen=True, slots=True)
class ProviderRequest:
    request_id: str
    idempotency_key: str
    cache_key: str
    model_id: str
    provider_settings: Mapping[str, Any]
    assessment_round: Literal["v0", "v1"]
    parent_snapshot_hash: str
    operations_hash: str
    prompt_hash: str
    case_packet: Mapping[str, Any]


@runtime_checkable
class AssessmentProvider(Protocol):
    model_id: str
    settings: Mapping[str, Any]
    prompt_hash: str

    def assess(self, request: ProviderRequest, *, timeout_seconds: int) -> ProviderResponse: ...


@dataclass(frozen=True, slots=True)
class OperationAuthorization:
    accepted: bool
    reason: str | None

    def __post_init__(self) -> None:
        if self.accepted == (self.reason is not None):
            raise RuntimeValidationError("Accepted operations have no rejection reason and vice versa.")
        if self.reason is not None and not _TOKEN_RE.fullmatch(self.reason):
            raise RuntimeValidationError("Operation rejection reasons must be coded identifiers.")


@dataclass(frozen=True, slots=True)
class ReplayExecution:
    after: EvidenceSnapshot
    invalidated_ids: tuple[str, ...]
    recomputed_ids: tuple[str, ...]
    repair_batch_id: str | None
    valid: bool
    reason: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.after, EvidenceSnapshot):
            raise RuntimeValidationError("Replay executor must return an EvidenceSnapshot.")
        if self.valid == (self.reason is not None):
            raise RuntimeValidationError("Invalid replay execution requires a coded reason.")


@runtime_checkable
class ReplayExecutor(Protocol):
    replay_model_id: str
    registry_hash: str
    state_schema_hash: str
    operation_registry: Mapping[str, str]

    def transcript_spans(self, snapshot: EvidenceSnapshot) -> Sequence[TranscriptSpan]: ...

    def authorize_operation(
        self, operation: ReviewOperation, snapshot: EvidenceSnapshot,
    ) -> OperationAuthorization: ...

    def replay(
        self, snapshot: EvidenceSnapshot, operations: Sequence[ReviewOperation],
    ) -> ReplayExecution: ...


@dataclass(frozen=True, slots=True)
class CasePacket:
    payload: Mapping[str, Any]
    content_hash: str
    prompt_hash: str
    transcript_hash: str
    skill_inventory: tuple[Mapping[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CacheIdentity:
    fields: Mapping[str, Any]
    key: str


@dataclass(frozen=True, slots=True)
class CacheEvent:
    key: str
    status: Literal["hit", "miss", "disabled"]
    reason: str


class InMemoryAssessmentCache:
    """Content/version-bound cache used by tests and local runners."""

    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled
        self._entries: dict[str, tuple[Mapping[str, Any], Mapping[str, Any]]] = {}
        self._lock = threading.Lock()

    def lookup(self, identity: CacheIdentity) -> tuple[Mapping[str, Any] | None, CacheEvent]:
        if not self.enabled:
            return None, CacheEvent(identity.key, "disabled", "cache_disabled")
        with self._lock:
            entry = self._entries.get(identity.key)
            if entry is not None:
                return _freeze(_plain(entry[1])), CacheEvent(identity.key, "hit", "exact_identity_match")
            reason = "cache_empty"
            if self._entries:
                closest = min(
                    self._entries.values(),
                    key=lambda item: sum(
                        item[0].get(key) != value for key, value in identity.fields.items()
                    ),
                )[0]
                changed = sorted(
                    key for key, value in identity.fields.items() if closest.get(key) != value
                )
                reason = "identity_mismatch:" + ",".join(changed)
            return None, CacheEvent(identity.key, "miss", reason)

    def store(self, identity: CacheIdentity, payload: Mapping[str, Any]) -> None:
        if not self.enabled:
            return
        with self._lock:
            self._entries[identity.key] = (_freeze(_plain(identity.fields)), _freeze(_plain(payload)))


@dataclass(frozen=True, slots=True)
class BudgetSnapshot:
    semantic_calls: int
    attempts: int
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int
    total_tokens: int
    reported_cost_usd: float
    consecutive_transport_failures: int


@dataclass(slots=True)
class RuntimeBudget:
    max_semantic_calls: int
    max_attempts: int
    max_input_tokens: int
    max_output_tokens: int
    max_reasoning_tokens: int
    max_total_tokens: int
    max_usd: float
    timeout_seconds: int = 180
    transport_retry_max: int = 1
    pause_after_consecutive_transport_failures: int = 2
    max_concurrency: int = 2
    semantic_calls: int = field(default=0, init=False)
    attempts: int = field(default=0, init=False)
    input_tokens: int = field(default=0, init=False)
    output_tokens: int = field(default=0, init=False)
    reasoning_tokens: int = field(default=0, init=False)
    reported_cost_usd: float = field(default=0.0, init=False)
    consecutive_transport_failures: int = field(default=0, init=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _provider_slots: threading.BoundedSemaphore = field(init=False, repr=False)

    def __post_init__(self) -> None:
        integer_limits = (
            self.max_semantic_calls, self.max_attempts, self.max_input_tokens,
            self.max_output_tokens, self.max_reasoning_tokens, self.max_total_tokens,
        )
        if any(type(value) is not int or value < 0 for value in integer_limits):
            raise RuntimeValidationError("Runtime budget limits must be nonnegative integers.")
        if not 1 <= self.timeout_seconds <= 180:
            raise RuntimeValidationError("Provider timeout must be between 1 and 180 seconds.")
        if self.transport_retry_max not in (0, 1):
            raise RuntimeValidationError("Transport retry maximum cannot exceed one.")
        if not 1 <= self.pause_after_consecutive_transport_failures <= 2:
            raise RuntimeValidationError("Provider pause threshold must be one or two failures.")
        if not 1 <= self.max_concurrency <= 2:
            raise RuntimeValidationError("Provider concurrency must be one or two.")
        if type(self.max_usd) not in (int, float) or not math.isfinite(self.max_usd) or self.max_usd < 0:
            raise RuntimeValidationError("Runtime cost cap must be finite and nonnegative.")
        self._provider_slots = threading.BoundedSemaphore(self.max_concurrency)

    @contextmanager
    def provider_slot(self):
        acquired = self._provider_slots.acquire(timeout=self.timeout_seconds)
        if not acquired:
            raise RuntimeValidationError("Provider concurrency wait exceeded the request timeout.")
        try:
            yield
        finally:
            self._provider_slots.release()

    def reserve_semantic_call(self) -> None:
        with self._lock:
            if self.semantic_calls >= self.max_semantic_calls:
                raise RuntimeValidationError("Semantic call budget exhausted.")
            if self.consecutive_transport_failures >= self.pause_after_consecutive_transport_failures:
                raise RuntimeValidationError("Provider queue paused by consecutive transport failures.")
            self.semantic_calls += 1

    def reserve_attempt(self) -> int:
        with self._lock:
            if self.attempts >= self.max_attempts:
                raise RuntimeValidationError("Transport attempt budget exhausted.")
            self.attempts += 1
            return self.attempts

    def record_transport_failure(self) -> None:
        with self._lock:
            self.consecutive_transport_failures += 1

    def record_response(self, response: ProviderResponse) -> None:
        with self._lock:
            self.consecutive_transport_failures = 0
            usage = response.usage
            if usage is not None:
                self.input_tokens += usage.input_tokens or 0
                self.output_tokens += usage.output_tokens or 0
                self.reasoning_tokens += usage.reasoning_tokens or 0
            if response.reported_cost_usd is not None:
                self.reported_cost_usd += float(response.reported_cost_usd)
            self._check_totals()

    def _check_totals(self) -> None:
        total = self.input_tokens + self.output_tokens + self.reasoning_tokens
        values = (
            (self.input_tokens, self.max_input_tokens, "input token"),
            (self.output_tokens, self.max_output_tokens, "output token"),
            (self.reasoning_tokens, self.max_reasoning_tokens, "reasoning token"),
            (total, self.max_total_tokens, "total token"),
        )
        for actual, maximum, label in values:
            if actual > maximum:
                raise RuntimeValidationError(f"Runtime {label} budget exceeded.")
        if self.reported_cost_usd > self.max_usd:
            raise RuntimeValidationError("Runtime reported cost budget exceeded.")

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            total = self.input_tokens + self.output_tokens + self.reasoning_tokens
            return BudgetSnapshot(
                semantic_calls=self.semantic_calls,
                attempts=self.attempts,
                input_tokens=self.input_tokens,
                output_tokens=self.output_tokens,
                reasoning_tokens=self.reasoning_tokens,
                total_tokens=total,
                reported_cost_usd=self.reported_cost_usd,
                consecutive_transport_failures=self.consecutive_transport_failures,
            )


@dataclass(frozen=True, slots=True)
class StrengthSummary:
    independent_support_count: int
    cited_evidence_count: int
    reliability_weight: float


@dataclass(frozen=True, slots=True)
class EvidenceTrace:
    trace_id: str
    evidence_id: str
    state_id: str
    support_unit_id: str
    source_segment_ids: tuple[str, ...]
    source_hashes: tuple[str, ...]
    legal_for_inference: bool
    reportable: bool
    shared_with_supervised: bool
    opaque: bool
    unsupported_reason: str | None


@dataclass(frozen=True, slots=True)
class ClaimTrace:
    trace_id: str
    claim_id: str
    revision: Literal["v0", "v1"]
    score: int
    state_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SourceTrace:
    claims: tuple[ClaimTrace, ...]
    evidence: tuple[EvidenceTrace, ...]
    unsupported_or_opaque: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class RuntimeResult:
    route: Literal["pilot_correlated_evidence_v1"]
    assessment_v0: AgentAssessment
    assessment_v1: AgentAssessment | None
    replay: ReplayResult | None
    usage: tuple[UsageRow, ...]
    cache_events: tuple[CacheEvent, ...]
    prompt_hashes: tuple[str, ...]
    source_trace: SourceTrace
    budget_snapshot: BudgetSnapshot
    j_s_valid: bool
    j_as_valid: bool
    j_as_fallback_reason: str | None


def _sanitize_transcript(
    span: TranscriptSpan, *, language: str,
) -> tuple[str, str, bool]:
    text = clean_model_transcript(span.text)
    text = "".join(character if character >= " " or character in "\n\t" else " "
                   for character in text)
    for name in sorted(span.public_names, key=len, reverse=True):
        text = re.sub(re.escape(name), "[redacted-name]", text, flags=re.IGNORECASE)
    text = re.sub(
        r"(?i)\b(patient|participant|subject)\s+name\s*:\s*[^.,;\n]+",
        r"\1 name: [redacted-name]",
        text,
    )
    text = re.sub(r"(?i)\b[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}\b", "[redacted-contact]", text)
    text = re.sub(
        r"(?i)(?:[A-Za-z]:)?(?:[/\\][^\s,;]+)+",
        "[redacted-path]",
        text,
    )
    injection_patterns = (
        r"ignore\s+(?:all\s+|any\s+)?previous\s+instructions?",
        r"disregard\s+(?:the\s+)?(?:system|developer|previous)\s+(?:message|instructions?)",
        r"reveal\s+(?:the\s+)?(?:system|developer|hidden)\s+(?:prompt|instructions?)",
        r"you\s+are\s+now\s+(?:a|an)\b",
    )
    for pattern in injection_patterns:
        text = re.sub(pattern, "[redacted-untrusted-instruction]", text, flags=re.IGNORECASE)
    text = re.sub(r"[ \t]+", " ", text).strip()
    screened = sanitize_segment_payload({
        "text": text,
        "language": language,
        "speaker_role": span.role,
        "prediction_eligible": True,
    })
    return (
        str(screened["text"]),
        str(screened["diagnostic_disclosure"]),
        bool(screened["prediction_eligible"]),
    )


def _skill_inventory(snapshot: EvidenceSnapshot, executor: ReplayExecutor) -> tuple[Mapping[str, Any], ...]:
    raw = getattr(executor, "skill_documents", None)
    documents = raw(snapshot) if callable(raw) else None
    if documents is None:
        return (_freeze({
            "document_id": "snapshot_skill_bundle",
            "sha256": snapshot.skill_hash,
            "content": None,
            "availability": "hash_bound_external_bundle",
        }),)
    if not isinstance(documents, Mapping) or not documents:
        raise RuntimeValidationError("Applicable skill documents must be a non-empty mapping.")
    inventory: list[Mapping[str, Any]] = []
    for document_id, content in sorted(documents.items()):
        if not isinstance(document_id, str) or not _TOKEN_RE.fullmatch(document_id):
            raise RuntimeValidationError("Skill document IDs must be coded identifiers, not paths.")
        if not isinstance(content, str):
            raise RuntimeValidationError("Skill document content must be text.")
        inventory.append(_freeze({
            "document_id": document_id,
            "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
            "content": content,
            "availability": "loaded",
        }))
    return tuple(inventory)


def _evidence_projection(snapshot: EvidenceSnapshot) -> list[dict[str, Any]]:
    projected: list[dict[str, Any]] = []
    for wrapped in snapshot.evidence:
        item = wrapped.body
        permissions = item["permissions"]
        if permissions["inference"] is not True:
            raise RuntimeValidationError("Evidence without inference permission cannot enter the case packet.")
        provenance = item["provenance"]
        projected.append({
            "evidence_id": item["evidence_id"],
            "metric_id": item["metric_id"],
            "metric_instance_id": item["metric_instance_id"],
            "state_id": item["state_id"],
            "task_id": item["task_id"],
            "value": item["value"],
            "unit": item["unit"],
            "source_modality": item["source_modality"],
            "direction": item["direction"],
            "observable": item["observable"],
            "unavailable_reason": item["unavailable_reason"],
            "reference": {
                "median": item["reference"]["median"],
                "scale": item["reference"]["scale"],
                "sample_size": item["reference"]["sample_size"],
                "scope": item["reference"]["scope"],
            },
            "reliability": dict(item["reliability_components"]),
            "confounds": dict(item["confounds"]),
            "legal_for_inference": True,
            "reportable": permissions["report"],
            "consumed_by_supervised": item["consumed_by_supervised"],
            "shared_evidence_notice": (
                "correlated_shared_measurement" if item["consumed_by_supervised"]
                else "not_marked_supervised"
            ),
            "provenance": {
                "source_asset_id": provenance["source_asset_id"],
                "source_segment_ids": list(provenance["source_segment_ids"]),
                "method_version": provenance["method_version"],
                "measurement_version": provenance["measurement_version"],
                "generated_by": provenance["generated_by"],
            },
        })
    return projected


def _state_projection(snapshot: EvidenceSnapshot) -> list[dict[str, Any]]:
    return [{
        "state_id": card.state_id,
        "state_z": card.state_z,
        "available": card.available,
        "supporting_evidence_ids": list(card.supporting_evidence_ids),
        "counter_evidence_ids": list(card.counter_evidence_ids),
        "observability": snapshot.observability[card.state_id].to_dict(),
    } for card in snapshot.state_cards]


def _reject_leakage(value: Any) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            normalized = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if any(part in normalized for part in _FORBIDDEN_KEY_PARTS):
                raise RuntimeValidationError("Truth, identity, path, or base-probability field rejected.")
            _reject_leakage(item)
    elif isinstance(value, (tuple, list)):
        for item in value:
            _reject_leakage(item)


def build_case_packet(snapshot: EvidenceSnapshot, executor: ReplayExecutor) -> CasePacket:
    """Build a blind role/task-aware packet from one immutable snapshot."""

    if not isinstance(snapshot, EvidenceSnapshot):
        raise TypeError("build_case_packet requires EvidenceSnapshot.")
    for field_name in ("registry_hash", "state_schema_hash"):
        _require_hash(str(getattr(executor, field_name, "")), f"executor.{field_name}")
    registry = getattr(executor, "operation_registry", None)
    if not isinstance(registry, Mapping) or set(registry) != _ALLOWED_OPERATIONS:
        raise RuntimeValidationError("Executor registry must expose exactly the four bounded operations.")
    for action, method_hash in registry.items():
        if action not in _ALLOWED_OPERATIONS:
            raise RuntimeValidationError("Executor registry contains an unbounded operation.")
        _require_hash(str(method_hash), f"executor.operation_registry[{action}]")

    source_segments = {item.segment_id: item for item in snapshot.source_segments}
    raw_spans = executor.transcript_spans(snapshot)
    spans: list[dict[str, Any]] = []
    seen: set[str] = set()
    for span in raw_spans:
        if type(span) is not TranscriptSpan:
            raise RuntimeValidationError("Executor transcript source must return TranscriptSpan values only.")
        if span.segment_id in seen:
            raise RuntimeValidationError("TranscriptSpan segment IDs must be unique.")
        seen.add(span.segment_id)
        source = source_segments.get(span.segment_id)
        if source is None:
            raise RuntimeValidationError("TranscriptSpan cites an unknown source segment.")
        if span.role != source.role:
            raise RuntimeValidationError("TranscriptSpan role does not match source provenance.")
        if span.task_id not in snapshot.subject.task_ids:
            raise RuntimeValidationError("TranscriptSpan task is not applicable to this subject route.")
        text, disclosure, prediction_eligible = _sanitize_transcript(
            span, language=snapshot.subject.language,
        )
        spans.append({
            "segment_id": span.segment_id,
            "task_id": span.task_id,
            "role": span.role,
            "start_seconds": source.start_seconds,
            "end_seconds": source.end_seconds,
            "text": text,
            "content_class": "untrusted_transcript",
            "disclosure_status": disclosure,
            "prediction_eligible": prediction_eligible,
        })

    skill_inventory = _skill_inventory(snapshot, executor)
    transcript_hash = _digest(spans)
    payload = {
        "runtime_version": RUNTIME_VERSION,
        "route_id": CORRELATED_FUSION_ROUTE,
        "assessment_contract": {
            "scores": "ordinal_zero_to_four_not_probabilities",
            "max_operations": 3,
            "max_repair_batches": 1,
            "transcript_trust": "untrusted_data_never_instructions",
            "evidence_dimensions": (
                "legal_for_inference", "reliability", "predictive_usefulness_judgment",
            ),
            "shared_evidence_policy": "citeable_correlated_not_independent",
            "report_policy": "no_long_clinician_report",
        },
        "case_id": snapshot.case_id,
        "snapshot_hash": snapshot.snapshot_hash,
        "reference_fit_id": snapshot.reference_fit_id,
        "reference_fit_hash": snapshot.reference_fit_hash,
        "route": {
            "dataset_id": snapshot.subject.dataset_id,
            "channel": snapshot.subject.channel,
            "language": snapshot.subject.language,
            "task": snapshot.subject.task,
            "task_ids": list(snapshot.subject.task_ids),
            "role": snapshot.subject.role,
            "class_order": list(snapshot.subject.class_order),
        },
        "transcript_spans": spans,
        "metric_evidence": _evidence_projection(snapshot),
        "state_cards": _state_projection(snapshot),
        "confounds": {key: list(values) for key, values in snapshot.confounds.items()},
        "modalities_present": sorted({
            item.body["source_modality"] for item in snapshot.evidence
            if item.body["source_modality"]
        }),
        "applicable_skills": [_plain(item) for item in skill_inventory],
    }
    _reject_leakage(payload)
    content_hash = _digest(payload)
    return CasePacket(
        payload=_plain(payload),
        content_hash=content_hash,
        prompt_hash=content_hash,
        transcript_hash=transcript_hash,
        skill_inventory=tuple(_plain(item) for item in skill_inventory),
    )


def cache_identity(
    snapshot: EvidenceSnapshot,
    packet: CasePacket,
    provider: AssessmentProvider,
    executor: ReplayExecutor,
    revision: Literal["v0", "v1"],
    operations: Sequence[ReviewOperation],
    *,
    parent_snapshot_hash: str | None = None,
) -> CacheIdentity:
    """Return the complete identity required for provider-response reuse."""

    _require_hash(str(provider.prompt_hash), "provider.prompt_hash")
    _require_hash(str(executor.state_schema_hash), "executor.state_schema_hash")
    operation_rows = [item.to_dict() for item in operations]
    fields = {
        "runtime_version": RUNTIME_VERSION,
        "route_id": CORRELATED_FUSION_ROUTE,
        "input_hash": packet.content_hash,
        "transcript_hash": packet.transcript_hash,
        "snapshot_hash": snapshot.snapshot_hash,
        "reference_fit_id": snapshot.reference_fit_id,
        "reference_fit_hash": snapshot.reference_fit_hash,
        "replay_model_id": str(executor.replay_model_id),
        "executor_registry_hash": str(executor.registry_hash),
        "task": snapshot.subject.task,
        "task_ids": list(snapshot.subject.task_ids),
        "role": snapshot.subject.role,
        "extractor_hash": snapshot.extractor_hash,
        "state_schema_hash": str(executor.state_schema_hash),
        "state_version": snapshot.state_version,
        "skill_hash": snapshot.skill_hash,
        "prompt_hash": str(provider.prompt_hash),
        "provider_model_id": str(provider.model_id),
        "provider_settings": _plain(provider.settings),
        "assessment_round": revision,
        "parent_snapshot_hash": parent_snapshot_hash or snapshot.snapshot_hash,
        "operations_hash": _digest(operation_rows),
    }
    _reject_leakage(fields)
    return CacheIdentity(fields=_freeze(fields), key=_digest(fields))


def _usage_row(
    *,
    request_id: str,
    cache_key: str,
    cache_status: Literal["hit", "miss", "bypass", "disabled"],
    response_status: Literal[
        "ok", "cache_hit", "transport_failure", "provider_failure", "timeout",
        "malformed_response",
    ],
    attempt: int,
    model_id: str,
    usage: ProviderUsage | None,
    reported_cost: float | None,
    wall_seconds: float | None,
    response_id: str | None,
    missing_reason: str = "provider_not_reported",
) -> UsageRow:
    values: dict[str, Any] = {
        "input_tokens": None if usage is None else usage.input_tokens,
        "output_tokens": None if usage is None else usage.output_tokens,
        "reasoning_tokens": None if usage is None else usage.reasoning_tokens,
        "cost_usd": reported_cost,
        "wall_seconds": wall_seconds,
        "response_id": response_id,
    }
    reasons = {key: missing_reason for key, value in values.items() if value is None}
    return UsageRow(
        request_id=request_id,
        cache_key=cache_key,
        cache_status=cache_status,
        response_status=response_status,
        attempt=attempt,
        model_id=model_id,
        unavailable_reasons=reasons,
        **values,
    )


def _decode_payload(value: Mapping[str, Any] | str | bytes | None) -> dict[str, Any]:
    if isinstance(value, Mapping):
        payload = _plain(value)
    elif isinstance(value, (str, bytes)):
        try:
            payload = json.loads(value)
        except (json.JSONDecodeError, UnicodeDecodeError, TypeError) as exc:
            raise RuntimeValidationError("Provider response is malformed JSON.") from exc
    else:
        raise RuntimeValidationError("Provider response is missing.")
    if not isinstance(payload, dict):
        raise RuntimeValidationError("Provider response must be one JSON object.")
    if set(payload) != _ALLOWED_RESPONSE_KEYS:
        raise RuntimeValidationError("Provider response fields do not match the assessment schema.")
    _reject_leakage(payload)
    return payload


def _record_mapping(value: Any, field_name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeValidationError(f"Provider {field_name} entries must be objects.")
    result = _plain(value)
    result["schema_version"] = SCHEMA_VERSION
    return result


def _parse_assessment(
    payload_value: Mapping[str, Any] | str | bytes | None,
    *,
    snapshot: EvidenceSnapshot,
    revision: Literal["v0", "v1"],
    provider: AssessmentProvider,
    usage: UsageRow,
) -> tuple[AgentAssessment, Mapping[str, Any]]:
    payload = _decode_payload(payload_value)
    if payload["snapshot_hash"] != snapshot.snapshot_hash or payload["revision"] != revision:
        raise RuntimeValidationError("Provider assessment is bound to a stale snapshot or round.")
    if payload["status"] != "ok" or payload["failure_reason"] is not None:
        raise RuntimeValidationError("Provider did not return a successful typed assessment.")
    if not isinstance(payload["state_judgments"], (tuple, list)):
        raise RuntimeValidationError("Provider state_judgments must be an array.")
    if not isinstance(payload["operations"], (tuple, list)):
        raise RuntimeValidationError("Provider operations must be an array.")
    judgments = tuple(StateJudgment.from_mapping(
        _record_mapping(item, "state_judgments")
    ) for item in payload["state_judgments"])
    operations = tuple(ReviewOperation.from_mapping(
        _record_mapping(item, "operations")
    ) for item in payload["operations"])
    fingerprints: set[str] = set()
    for operation in operations:
        body = operation.to_dict()
        body.pop("schema_version", None)
        body.pop("operation_id", None)
        body.pop("reason", None)
        fingerprint = _digest(body)
        if fingerprint in fingerprints:
            raise RuntimeValidationError("Duplicate operation loop rejected.")
        fingerprints.add(fingerprint)
    assessment = AgentAssessment(
        subject_id=snapshot.subject.subject_id,
        case_id=snapshot.case_id,
        task=snapshot.subject.task,
        class_order=snapshot.subject.class_order,
        snapshot_hash=snapshot.snapshot_hash,
        revision=revision,
        ordinal_scores=payload["ordinal_scores"],
        citations=tuple(payload["citations"]),
        state_judgments=judgments,
        operations=operations,
        model_id=str(provider.model_id),
        usage=usage,
        usage_id=usage.content_hash,
        status="ok",
        failure_reason=None,
    )
    assessment.validate_snapshot(snapshot)
    evidence_by_id = {item.body["evidence_id"]: item.body for item in snapshot.evidence}
    for evidence_id in assessment.citations:
        task_id = evidence_by_id[evidence_id]["task_id"]
        if task_id is not None and task_id not in snapshot.subject.task_ids:
            raise RuntimeValidationError("Assessment cited evidence outside the applicable task route.")
    return assessment, _freeze(payload)


def _failed_assessment(
    snapshot: EvidenceSnapshot,
    revision: Literal["v0", "v1"],
    provider: AssessmentProvider,
    usage: UsageRow,
    reason: str,
) -> AgentAssessment:
    return AgentAssessment(
        subject_id=snapshot.subject.subject_id,
        case_id=snapshot.case_id,
        task=snapshot.subject.task,
        class_order=snapshot.subject.class_order,
        snapshot_hash=snapshot.snapshot_hash,
        revision=revision,
        ordinal_scores=None,
        citations=(),
        state_judgments=(),
        operations=(),
        model_id=str(provider.model_id),
        usage=usage,
        usage_id=usage.content_hash,
        status="failed",
        failure_reason=_coded_reason(reason),
    )


@dataclass(frozen=True, slots=True)
class _CallResult:
    assessment: AgentAssessment
    usage: tuple[UsageRow, ...]
    cache_event: CacheEvent
    prompt_hash: str


def _call_assessment(
    snapshot: EvidenceSnapshot,
    provider: AssessmentProvider,
    executor: ReplayExecutor,
    budget: RuntimeBudget,
    cache: InMemoryAssessmentCache,
    *,
    revision: Literal["v0", "v1"],
    operations: Sequence[ReviewOperation],
    parent_snapshot_hash: str,
) -> _CallResult:
    packet = build_case_packet(snapshot, executor)
    identity = cache_identity(
        snapshot, packet, provider, executor, revision, operations,
        parent_snapshot_hash=parent_snapshot_hash,
    )
    cached, cache_event = cache.lookup(identity)
    request_id = "req_" + identity.key[:24]
    if cached is not None:
        usage = _usage_row(
            request_id=request_id,
            cache_key=identity.key,
            cache_status="hit",
            response_status="cache_hit",
            attempt=1,
            model_id=str(provider.model_id),
            usage=None,
            reported_cost=None,
            wall_seconds=0.0,
            response_id=None,
            missing_reason="cache_hit_no_provider_usage",
        )
        try:
            assessment, _ = _parse_assessment(
                cached, snapshot=snapshot, revision=revision, provider=provider, usage=usage,
            )
        except (PilotContractError, RuntimeValidationError, TypeError, ValueError) as exc:
            raise RuntimeValidationError("Validated cache entry no longer matches its identity.") from exc
        return _CallResult(assessment, (usage,), cache_event, packet.prompt_hash)

    budget.reserve_semantic_call()
    all_usage: list[UsageRow] = []
    final_usage: UsageRow | None = None
    final_reason = "provider_failure"
    cache_status: Literal["miss", "disabled"] = (
        "disabled" if cache_event.status == "disabled" else "miss"
    )
    for request_attempt in range(1, budget.transport_retry_max + 2):
        budget.reserve_attempt()
        request = ProviderRequest(
            request_id=request_id,
            idempotency_key=request_id,
            cache_key=identity.key,
            model_id=str(provider.model_id),
            provider_settings=_freeze(_plain(provider.settings)),
            assessment_round=revision,
            parent_snapshot_hash=parent_snapshot_hash,
            operations_hash=identity.fields["operations_hash"],
            prompt_hash=packet.prompt_hash,
            case_packet=packet.payload,
        )
        started = time.monotonic()
        try:
            with budget.provider_slot():
                response = provider.assess(request, timeout_seconds=budget.timeout_seconds)
            elapsed = time.monotonic() - started
            if not isinstance(response, ProviderResponse):
                raise RuntimeValidationError("Provider must return ProviderResponse.")
            budget.record_response(response)
            if response.status != "ok":
                final_reason = "provider_failure"
                final_usage = _usage_row(
                    request_id=request_id, cache_key=identity.key,
                    cache_status=cache_status, response_status="provider_failure",
                    attempt=request_attempt, model_id=str(provider.model_id), usage=response.usage,
                    reported_cost=response.reported_cost_usd, wall_seconds=elapsed,
                    response_id=response.response_id,
                )
                all_usage.append(final_usage)
                break
            provisional = _usage_row(
                request_id=request_id, cache_key=identity.key,
                cache_status=cache_status, response_status="ok",
                attempt=request_attempt, model_id=str(provider.model_id), usage=response.usage,
                reported_cost=response.reported_cost_usd, wall_seconds=elapsed,
                response_id=response.response_id,
            )
            try:
                assessment, validated_payload = _parse_assessment(
                    response.payload,
                    snapshot=snapshot,
                    revision=revision,
                    provider=provider,
                    usage=provisional,
                )
            except (PilotContractError, RuntimeValidationError, TypeError, ValueError, KeyError):
                final_reason = "malformed_response"
                final_usage = replace(provisional, response_status="malformed_response")
                all_usage.append(final_usage)
                break
            all_usage.append(provisional)
            cache.store(identity, validated_payload)
            return _CallResult(assessment, tuple(all_usage), cache_event, packet.prompt_hash)
        except ProviderTimeoutError:
            elapsed = time.monotonic() - started
            budget.record_transport_failure()
            final_reason = "timeout"
            final_usage = _usage_row(
                request_id=request_id, cache_key=identity.key,
                cache_status=cache_status, response_status="timeout",
                attempt=request_attempt, model_id=str(provider.model_id), usage=None,
                reported_cost=None, wall_seconds=elapsed, response_id=None,
                missing_reason="request_failed",
            )
            all_usage.append(final_usage)
        except ProviderTransportError:
            elapsed = time.monotonic() - started
            budget.record_transport_failure()
            final_reason = "transport_failure"
            final_usage = _usage_row(
                request_id=request_id, cache_key=identity.key,
                cache_status=cache_status, response_status="transport_failure",
                attempt=request_attempt, model_id=str(provider.model_id), usage=None,
                reported_cost=None, wall_seconds=elapsed, response_id=None,
                missing_reason="request_failed",
            )
            all_usage.append(final_usage)
        except RuntimeValidationError:
            raise
        except Exception:
            elapsed = time.monotonic() - started
            final_reason = "provider_failure"
            final_usage = _usage_row(
                request_id=request_id, cache_key=identity.key,
                cache_status=cache_status, response_status="provider_failure",
                attempt=request_attempt, model_id=str(provider.model_id), usage=None,
                reported_cost=None, wall_seconds=elapsed, response_id=None,
                missing_reason="request_failed",
            )
            all_usage.append(final_usage)
            break

        if request_attempt > budget.transport_retry_max:
            break
        if budget.consecutive_transport_failures >= budget.pause_after_consecutive_transport_failures:
            break

    if final_usage is None:
        raise RuntimeValidationError("Provider failed without an auditable usage attempt.")
    failed = _failed_assessment(snapshot, revision, provider, final_usage, final_reason)
    return _CallResult(failed, tuple(all_usage), cache_event, packet.prompt_hash)


def _authorize_operations(
    snapshot: EvidenceSnapshot,
    operations: Sequence[ReviewOperation],
    executor: ReplayExecutor,
) -> tuple[tuple[ReviewOperation, ...], tuple[RejectedOperation, ...]]:
    accepted: list[ReviewOperation] = []
    rejected: list[RejectedOperation] = []
    seen: set[str] = set()
    registry = dict(executor.operation_registry)
    for operation in operations:
        fingerprint_body = operation.to_dict()
        fingerprint_body.pop("schema_version", None)
        fingerprint_body.pop("operation_id", None)
        fingerprint_body.pop("reason", None)
        fingerprint = _digest(fingerprint_body)
        reason: str | None = None
        if operation.action not in registry:
            reason = "operation_not_registered"
        elif fingerprint in seen:
            reason = "duplicate_operation"
        elif (
            operation.action == "deterministic_remeasurement"
            and operation.method_hash != registry[operation.action]
        ):
            reason = "method_not_registered"
        seen.add(fingerprint)
        if reason is None:
            try:
                authorization = executor.authorize_operation(operation, snapshot)
                if not isinstance(authorization, OperationAuthorization):
                    raise RuntimeValidationError("Executor authorization must be typed.")
                if not authorization.accepted:
                    reason = authorization.reason
            except RuntimeValidationError:
                raise
            except Exception:
                reason = "executor_authorization_failed"
        if reason is None:
            accepted.append(operation)
        else:
            rejected.append(RejectedOperation(operation=operation, reason=_coded_reason(reason)))
    return tuple(accepted), tuple(rejected)


def _state_scores(snapshot: EvidenceSnapshot) -> Mapping[str, float | None]:
    return MappingProxyType({card.state_id: card.state_z for card in snapshot.state_cards})


def _run_replay(
    snapshot: EvidenceSnapshot,
    operations: Sequence[ReviewOperation],
    executor: ReplayExecutor,
) -> ReplayResult:
    accepted, rejected = _authorize_operations(snapshot, operations, executor)
    if accepted:
        try:
            execution = executor.replay(snapshot, accepted)
            if not isinstance(execution, ReplayExecution):
                raise RuntimeValidationError("Executor replay result must be typed.")
        except Exception:
            execution = ReplayExecution(
                after=snapshot,
                invalidated_ids=(),
                recomputed_ids=(),
                repair_batch_id="repair_failed",
                valid=False,
                reason="executor_replay_failed",
            )
    else:
        execution = ReplayExecution(
            after=snapshot,
            invalidated_ids=(),
            recomputed_ids=(),
            repair_batch_id=None,
            valid=True,
            reason=None,
        )
    before_scores = _state_scores(snapshot)
    after_scores = _state_scores(execution.after)
    state_delta = {
        state_id: (
            None if before_scores[state_id] is None or after_scores.get(state_id) is None
            else after_scores[state_id] - before_scores[state_id]
        )
        for state_id in before_scores
    }
    try:
        return ReplayResult(
            before=snapshot,
            after=execution.after,
            parent_hash=snapshot.snapshot_hash,
            child_hash=execution.after.snapshot_hash,
            accepted_ops=accepted,
            rejected_ops=rejected,
            invalidated_ids=execution.invalidated_ids,
            recomputed_ids=execution.recomputed_ids,
            before_state_scores=before_scores,
            after_state_scores=after_scores,
            state_delta=state_delta,
            replay_model_id=str(executor.replay_model_id),
            repair_batch_id=execution.repair_batch_id,
            executor_registry_hash=str(executor.registry_hash),
            authorized_operations={item.operation_id: item.content_hash for item in accepted},
            valid=execution.valid,
            reason=execution.reason,
        )
    except (PilotContractError, TypeError, ValueError) as exc:
        raise RuntimeValidationError("Executor returned an invalid replay dependency closure.") from exc


def _measurement_identity(item: Mapping[str, Any]) -> str:
    return _digest({
        "metric_instance_id": item["metric_instance_id"],
        "source_asset_id": item["provenance"]["source_asset_id"],
        "source_segment_ids": list(item["provenance"]["source_segment_ids"]),
        "measurement_version": item["provenance"]["measurement_version"],
    })


def assessment_strength(
    assessment: AgentAssessment, snapshot: EvidenceSnapshot,
) -> StrengthSummary:
    """Summarize unique cited measurements without using any base uncertainty."""

    assessment.validate_snapshot(snapshot)
    evidence = {item.body["evidence_id"]: item.body for item in snapshot.evidence}
    units: dict[str, Mapping[str, Any]] = {}
    for evidence_id in assessment.citations:
        item = evidence[evidence_id]
        if item["permissions"]["inference"]:
            units.setdefault(_measurement_identity(item), item)
    reliability = 0.0
    for item in units.values():
        components = item["reliability_components"]
        values = [float(value) for value in components.values()]
        reliability += sum(values) / len(values) if values else 0.0
    return StrengthSummary(
        independent_support_count=len(units),
        cited_evidence_count=len(assessment.citations),
        reliability_weight=reliability,
    )


def _source_trace(
    assessments: Sequence[tuple[AgentAssessment, EvidenceSnapshot]],
) -> SourceTrace:
    claims: list[ClaimTrace] = []
    evidence_rows: dict[tuple[str, str], EvidenceTrace] = {}
    unsupported: set[str] = set()
    for assessment, snapshot in assessments:
        if assessment.status != "ok" or assessment.ordinal_scores is None:
            continue
        evidence = {item.body["evidence_id"]: item.body for item in snapshot.evidence}
        cited_states = tuple(sorted({evidence[item]["state_id"] for item in assessment.citations}))
        for class_name, score in assessment.ordinal_scores.items():
            claim_id = f"{assessment.revision}.class.{class_name}"
            claims.append(ClaimTrace(
                trace_id="trace_" + _digest({
                    "claim": claim_id, "snapshot": snapshot.snapshot_hash,
                    "citations": list(assessment.citations),
                })[:24],
                claim_id=claim_id,
                revision=assessment.revision,
                score=score,
                state_ids=cited_states,
                evidence_ids=assessment.citations,
            ))
        unsupported.add(f"{assessment.revision}:class_mapping_is_model_judgment")
        for evidence_id in assessment.citations:
            item = evidence[evidence_id]
            provenance = item["provenance"]
            asset_id = provenance["source_asset_id"]
            source_hashes = (
                (snapshot.subject.raw_hashes[asset_id],)
                if asset_id in snapshot.subject.raw_hashes else ()
            )
            opaque = "embedding" in (
                str(item["metric_id"]) + " " + str(item["source_modality"])
            ).lower()
            reason = "opaque_embedding" if opaque else None
            if reason:
                unsupported.add(f"{assessment.revision}:{evidence_id}:{reason}")
            key = (snapshot.snapshot_hash, evidence_id)
            evidence_rows[key] = EvidenceTrace(
                trace_id="trace_" + _digest({
                    "snapshot": snapshot.snapshot_hash, "evidence": evidence_id,
                })[:24],
                evidence_id=evidence_id,
                state_id=item["state_id"],
                support_unit_id=_measurement_identity(item),
                source_segment_ids=tuple(provenance["source_segment_ids"]),
                source_hashes=source_hashes,
                legal_for_inference=bool(item["permissions"]["inference"]),
                reportable=bool(item["permissions"]["report"]),
                shared_with_supervised=bool(item["consumed_by_supervised"]),
                opaque=opaque,
                unsupported_reason=reason,
            )
    return SourceTrace(
        claims=tuple(claims),
        evidence=tuple(evidence_rows.values()),
        unsupported_or_opaque=tuple(sorted(unsupported)),
    )


def assess_and_replay(
    snapshot: EvidenceSnapshot,
    provider: AssessmentProvider,
    executor: ReplayExecutor,
    budget: RuntimeBudget,
    cache: InMemoryAssessmentCache,
) -> RuntimeResult:
    """Run one blind v0 assessment, one bounded replay, and at most one v1."""

    if not isinstance(snapshot, EvidenceSnapshot):
        raise TypeError("assess_and_replay requires EvidenceSnapshot.")
    if not isinstance(budget, RuntimeBudget):
        raise TypeError("assess_and_replay requires RuntimeBudget.")
    if not isinstance(cache, InMemoryAssessmentCache):
        raise TypeError("assess_and_replay requires InMemoryAssessmentCache.")
    if not isinstance(provider.model_id, str) or not provider.model_id:
        raise RuntimeValidationError("Provider requires an exact model ID.")
    if not re.search(r"\d{4}-\d{2}-\d{2}$", provider.model_id):
        raise RuntimeValidationError("Provider model ID must be exact and dated.")
    _require_hash(str(provider.prompt_hash), "provider.prompt_hash")

    v0 = _call_assessment(
        snapshot, provider, executor, budget, cache,
        revision="v0", operations=(), parent_snapshot_hash=snapshot.snapshot_hash,
    )
    usage = list(v0.usage)
    events = [v0.cache_event]
    prompt_hashes = [v0.prompt_hash]
    assessment_v1: AgentAssessment | None = None
    replay: ReplayResult | None = None
    trace_inputs: list[tuple[AgentAssessment, EvidenceSnapshot]] = [(v0.assessment, snapshot)]

    if v0.assessment.status == "ok":
        replay = _run_replay(snapshot, v0.assessment.operations, executor)
        if replay.valid and replay.accepted_ops and replay.meaningful_change:
            try:
                v1 = _call_assessment(
                    replay.after, provider, executor, budget, cache,
                    revision="v1", operations=replay.accepted_ops,
                    parent_snapshot_hash=replay.parent_hash,
                )
                assessment_v1 = v1.assessment
                usage.extend(v1.usage)
                events.append(v1.cache_event)
                prompt_hashes.append(v1.prompt_hash)
                trace_inputs.append((assessment_v1, replay.after))
            except RuntimeValidationError as exc:
                # Initial budget exhaustion is a caller error.  A required v1
                # that cannot be called is represented as missing, never by v0.
                if budget.semantic_calls == 0:
                    raise
                if "budget" not in str(exc).lower() and "paused" not in str(exc).lower():
                    raise

    j_s_valid = replay is not None and replay.valid
    if replay is None:
        j_as_reason = "v0_failed"
    elif not replay.valid:
        j_as_reason = "replay_failed"
    elif replay.meaningful_change and replay.accepted_ops:
        if assessment_v1 is None:
            j_as_reason = "v1_missing"
        elif assessment_v1.status != "ok":
            j_as_reason = "v1_failed"
        else:
            j_as_reason = None
    else:
        j_as_reason = None

    return RuntimeResult(
        route=CORRELATED_FUSION_ROUTE,
        assessment_v0=v0.assessment,
        assessment_v1=assessment_v1,
        replay=replay,
        usage=tuple(usage),
        cache_events=tuple(events),
        prompt_hashes=tuple(prompt_hashes),
        source_trace=_source_trace(trace_inputs),
        budget_snapshot=budget.snapshot(),
        j_s_valid=j_s_valid,
        j_as_valid=j_as_reason is None,
        j_as_fallback_reason=j_as_reason,
    )
