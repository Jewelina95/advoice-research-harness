"""Strict bridge from an Agent state review to conditional-authority inputs."""
from __future__ import annotations

from dataclasses import dataclass
import math
import re
from types import MappingProxyType
from typing import Any, Literal, Mapping, Sequence

from .conditional_authority import AgentAuthorityDecision, PreparedAuthorityCase
from .decision_lock import canonical_json, hash_artifact
from .evidence_revision_batch import (
    EvidenceRevisionBatch,
    EvidenceRevisionBatchError,
    compile_evidence_revision_batch,
)


AgentStateAction = Literal["retain", "downweight", "invalidate", "mark_unavailable"]

_REVIEW_FIELDS = frozenset(
    {
        "case_id",
        "reviewed_packet_hash",
        "reviewed_evidence_hash",
        "reviewed_state_graph_hash",
        "state_id",
        "action",
        "reliability_multiplier",
        "rationale",
        "cited_evidence_ids",
        "ordinal_scores",
        "report_trace",
    }
)
_OPTIONAL_REVIEW_FIELDS = frozenset({"reliability_multiplier"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_FORBIDDEN_FIELD_PARTS = frozenset({"label", "labels", "split", "truth"})
_ACTION_MAP = {
    "retain": "retain",
    "downweight": "downweight",
    "invalidate": "invalidate",
    "mark_unavailable": "mark_unavailable",
}


class AgentStateReviewError(ValueError):
    """Raised when an Agent state review cannot cross the authority boundary."""


def _field_parts(value: Any) -> set[str]:
    normalized = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", str(value))
    return {part for part in re.split(r"[^a-z0-9]+", normalized.lower()) if part}


def _reject_forbidden_fields(value: Any, *, path: str = "review") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            if _field_parts(key) & _FORBIDDEN_FIELD_PARTS:
                raise AgentStateReviewError(
                    f"Agent state review contains forbidden raw field {str(key)!r} at {path}."
                )
            _reject_forbidden_fields(item, path=f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_fields(item, path=f"{path}[{index}]")


def _freeze_json(value: Any, *, path: str) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_json(value[key], path=f"{path}.{key}")
                for key in sorted(value, key=str)
            }
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float) and math.isfinite(value):
        return value
    raise AgentStateReviewError(f"{path} must contain JSON-compatible finite values only.")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def _required_text(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise AgentStateReviewError(f"Agent state review {field} must be a non-empty string.")
    return value.strip()


def _sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value) is None:
        raise AgentStateReviewError(f"Agent state review {field} must be a lowercase SHA-256 hash.")
    return value


def _string_tuple(value: Any, *, field: str) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AgentStateReviewError(f"Agent state review {field} must be a sequence of IDs.")
    result: list[str] = []
    for item in value:
        if not isinstance(item, str) or not item.strip():
            raise AgentStateReviewError(f"Agent state review {field} must contain non-empty strings.")
        result.append(item.strip())
    if len(result) != len(set(result)):
        raise AgentStateReviewError(f"Agent state review {field} cannot contain duplicate IDs.")
    return tuple(sorted(result))


@dataclass(frozen=True, slots=True)
class AgentStateReview:
    """Immutable, label-blind review of one prepared state."""

    case_id: str
    reviewed_packet_hash: str
    reviewed_evidence_hash: str
    reviewed_state_graph_hash: str
    state_id: str
    action: AgentStateAction
    rationale: str
    cited_evidence_ids: tuple[str, ...]
    ordinal_scores: Mapping[str, int]
    report_trace: tuple[Mapping[str, Any], ...]
    reliability_multiplier: float | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "case_id", _required_text(self.case_id, field="case_id"))
        object.__setattr__(self, "state_id", _required_text(self.state_id, field="state_id"))
        object.__setattr__(self, "rationale", _required_text(self.rationale, field="rationale"))
        for field in (
            "reviewed_packet_hash",
            "reviewed_evidence_hash",
            "reviewed_state_graph_hash",
        ):
            object.__setattr__(self, field, _sha256(getattr(self, field), field=field))
        if self.action not in _ACTION_MAP:
            raise AgentStateReviewError(
                "Agent state review action must be retain, downweight, invalidate, "
                "or mark_unavailable."
            )
        multiplier = self.reliability_multiplier
        if multiplier is not None:
            if isinstance(multiplier, bool) or not isinstance(multiplier, (int, float)):
                raise AgentStateReviewError("reliability_multiplier must be a finite number or null.")
            multiplier = float(multiplier)
            if not math.isfinite(multiplier):
                raise AgentStateReviewError("reliability_multiplier must be a finite number or null.")
            if self.action != "downweight":
                raise AgentStateReviewError(
                    "reliability_multiplier is valid only for a downweight review."
                )
        object.__setattr__(self, "reliability_multiplier", multiplier)
        object.__setattr__(
            self,
            "cited_evidence_ids",
            _string_tuple(self.cited_evidence_ids, field="cited_evidence_ids"),
        )

        if not isinstance(self.ordinal_scores, Mapping) or not self.ordinal_scores:
            raise AgentStateReviewError(
                "Agent state review ordinal_scores must be a non-empty mapping."
            )
        scores: dict[str, int] = {}
        for key, score in self.ordinal_scores.items():
            if not isinstance(key, str) or not key.strip():
                raise AgentStateReviewError("Agent state review ordinal_scores keys must be strings.")
            if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
                raise AgentStateReviewError(
                    "Agent state review ordinal_scores must be integers from 0 through 4."
                )
            scores[key.strip()] = score
        if len(scores) != len(self.ordinal_scores):
            raise AgentStateReviewError("Agent state review ordinal_scores contains duplicate keys.")
        object.__setattr__(self, "ordinal_scores", MappingProxyType(dict(sorted(scores.items()))))

        if isinstance(self.report_trace, (str, bytes)) or not isinstance(self.report_trace, Sequence):
            raise AgentStateReviewError("Agent state review report_trace must be a sequence of records.")
        _reject_forbidden_fields(self.report_trace, path="review.report_trace")
        trace: list[Mapping[str, Any]] = []
        for index, item in enumerate(self.report_trace):
            if not isinstance(item, Mapping):
                raise AgentStateReviewError("Agent state review report_trace must contain records only.")
            trace.append(_freeze_json(item, path=f"report_trace[{index}]"))
        object.__setattr__(self, "report_trace", tuple(trace))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentStateReview":
        if not isinstance(value, Mapping):
            raise AgentStateReviewError("Agent state review must be a mapping.")
        _reject_forbidden_fields(value)
        fields = {str(key) for key in value}
        unknown = sorted(fields - _REVIEW_FIELDS)
        if unknown:
            raise AgentStateReviewError(f"Agent state review includes unknown fields: {unknown}")
        missing = sorted((_REVIEW_FIELDS - _OPTIONAL_REVIEW_FIELDS) - fields)
        if missing:
            raise AgentStateReviewError(f"Agent state review is missing required fields: {missing}")
        return cls(
            case_id=value["case_id"],
            reviewed_packet_hash=value["reviewed_packet_hash"],
            reviewed_evidence_hash=value["reviewed_evidence_hash"],
            reviewed_state_graph_hash=value["reviewed_state_graph_hash"],
            state_id=value["state_id"],
            action=value["action"],
            rationale=value["rationale"],
            cited_evidence_ids=value["cited_evidence_ids"],
            ordinal_scores=value["ordinal_scores"],
            report_trace=value["report_trace"],
            reliability_multiplier=value.get("reliability_multiplier"),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "reviewed_packet_hash": self.reviewed_packet_hash,
            "reviewed_evidence_hash": self.reviewed_evidence_hash,
            "reviewed_state_graph_hash": self.reviewed_state_graph_hash,
            "state_id": self.state_id,
            "action": self.action,
            "reliability_multiplier": self.reliability_multiplier,
            "rationale": self.rationale,
            "cited_evidence_ids": list(self.cited_evidence_ids),
            "ordinal_scores": dict(self.ordinal_scores),
            "report_trace": [_thaw_json(item) for item in self.report_trace],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def review_hash(self) -> str:
        return hash_artifact(self.to_dict())


@dataclass(frozen=True, slots=True)
class CompiledAgentStateReview:
    """Immutable bridge output for the current one-revision executor."""

    review: AgentStateReview
    decision: AgentAuthorityDecision
    batch: EvidenceRevisionBatch

    @property
    def agent_decision(self) -> AgentAuthorityDecision:
        return self.decision

    @property
    def evidence_revision_batch(self) -> EvidenceRevisionBatch:
        return self.batch

    @property
    def revision_batch(self) -> EvidenceRevisionBatch:
        return self.batch

    def to_dict(self) -> dict[str, Any]:
        return {
            "review": self.review.to_dict(),
            "review_hash": self.review.review_hash,
            "agent_decision": self.decision.to_dict(),
            "evidence_revision_batch": self.batch.to_dict(),
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict())

    @property
    def compiled_hash(self) -> str:
        return hash_artifact(self.to_dict())

    @property
    def result_hash(self) -> str:
        return self.compiled_hash


def compile_agent_state_review(
    prepared: PreparedAuthorityCase,
    review: AgentStateReview | Mapping[str, Any],
) -> CompiledAgentStateReview:
    """Compile one state review without reading labels or truth artifacts."""

    if not isinstance(prepared, PreparedAuthorityCase):
        raise AgentStateReviewError("prepared must be a PreparedAuthorityCase.")
    parsed = review if isinstance(review, AgentStateReview) else AgentStateReview.from_mapping(review)
    if parsed.case_id != prepared.case_id:
        raise AgentStateReviewError("Agent state review is bound to a different case.")
    for field in (
        "reviewed_packet_hash",
        "reviewed_evidence_hash",
        "reviewed_state_graph_hash",
    ):
        if getattr(parsed, field) != getattr(prepared, field):
            raise AgentStateReviewError(f"Agent state review contains a stale {field}.")

    target_evidence_ids = {
        item.evidence_id
        for item in prepared.module_a_evidence
        if item.state_id == parsed.state_id
    }
    if not target_evidence_ids:
        raise AgentStateReviewError(
            f"Agent state review state mismatch: {parsed.state_id!r} is not in the prepared snapshot."
        )
    foreign_citations = sorted(set(parsed.cited_evidence_ids) - target_evidence_ids)
    if foreign_citations:
        raise AgentStateReviewError(
            "Agent state review cited evidence IDs outside the target state: "
            f"{foreign_citations}."
        )

    try:
        batch = compile_evidence_revision_batch(
            prepared.case_id,
            parsed.state_id,
            _ACTION_MAP[parsed.action],  # type: ignore[arg-type]
            prepared.pre_replay.state_graph,
            prepared.module_a_evidence,
            prepared.reviewed_evidence_hash,
            reliability_multiplier=parsed.reliability_multiplier,
            rationale=parsed.rationale,
            cited_evidence_ids=parsed.cited_evidence_ids,
        )
    except EvidenceRevisionBatchError as exc:
        raise AgentStateReviewError(f"Agent state review batch compilation failed: {exc}") from exc

    if parsed.action != "retain" and len(batch.revisions) != 1:
        raise AgentStateReviewError(
            "batch execution not yet supported: non-retain state reviews must compile to exactly one revision."
        )
    revision = None if parsed.action == "retain" else batch.revisions[0]
    decision = AgentAuthorityDecision(
        case_id=prepared.case_id,
        reviewed_packet_hash=prepared.reviewed_packet_hash,
        reviewed_evidence_hash=prepared.reviewed_evidence_hash,
        reviewed_state_graph_hash=prepared.reviewed_state_graph_hash,
        advisor_packet_hash=prepared.advisor_packet_hash,
        advisor_current=True,
        ordinal_scores=parsed.ordinal_scores,
        revision=revision,
        action_type=parsed.action,
        incremental_evidence_ids=(),
        report_trace=parsed.report_trace,
    )
    return CompiledAgentStateReview(review=parsed, decision=decision, batch=batch)


__all__ = [
    "AgentStateAction",
    "AgentStateReview",
    "AgentStateReviewError",
    "CompiledAgentStateReview",
    "compile_agent_state_review",
]
