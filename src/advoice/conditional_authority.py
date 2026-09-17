"""Opt-in end-to-end conditional authority execution.

This module deliberately does not alter the legacy ``condition_c`` path.  It
only composes the typed contracts introduced for evidence-governed inference:
route selection, immutable MetricEvidenceV2, StateGraphV2 replay, Module A,
the bounded Module B arbitrator, and DecisionLock.

The boundary is intentionally strict.  An Agent may identify a bounded
evidence revision or supply an ordinal assessment of *validated incremental*
evidence.  It cannot write a probability, alter a measurement, replace Module
A, or emit a report without a DecisionLock.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from .decision_lock import (
    DecisionLock,
    HashMismatchError,
    create_decision_lock,
    hash_artifact,
    validate_locked_report,
)
from .evidence import MetricEvidenceV2
from .evidence_replay import (
    EvidenceReplayResult,
    EvidenceRevision,
    replay_evidence,
)
from .evidence_revision_batch import EvidenceRevisionBatch, EvidenceRevisionBatchError
from .module_a import ExplanationPacket, TaskConditionedStatisticalExpert
from .module_b import ConditionalArbitrator, ModuleBPrediction
from .routing import RouteDecision, route_case
from .state_graph import deserialize_state_card_ids


class ConditionalAuthorityError(ValueError):
    """Raised when a case does not satisfy the opt-in authority contract."""


class StaleAuthorityError(ConditionalAuthorityError):
    """Raised when a decision, advisor, validator, or packet is stale."""


def _canonical_mapping(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType({str(key): value[key] for key in sorted(value or {}, key=str)})


def _tuple(value: Iterable[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        value = (value,)
    return tuple(sorted({str(item) for item in value}))


def _finite_unit(value: Any, name: str) -> float:
    parsed = float(value)
    if not np.isfinite(parsed) or not 0.0 <= parsed <= 1.0:
        raise ConditionalAuthorityError(f"{name} must be a finite value in [0, 1].")
    return parsed


def _strict_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise ConditionalAuthorityError(f"{name} must be a boolean, not a coercible value.")
    return value


def _revision_from(value: EvidenceRevision | Mapping[str, Any] | None) -> EvidenceRevision | None:
    if value is None or isinstance(value, EvidenceRevision):
        return value
    if not isinstance(value, Mapping):
        raise ConditionalAuthorityError("revision must be an EvidenceRevision, mapping, or None.")
    allowed = {
        "evidence_id", "action", "expected_evidence_hash", "reliability_multiplier",
        "rationale", "cited_evidence_ids", "schema_version",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConditionalAuthorityError(f"revision includes protected or unknown fields: {unknown}")
    return EvidenceRevision(
        evidence_id=str(value.get("evidence_id", "")),
        action=str(value.get("action", "")),  # type: ignore[arg-type]
        expected_evidence_hash=str(value.get("expected_evidence_hash", "")),
        reliability_multiplier=value.get("reliability_multiplier"),
        rationale=str(value.get("rationale", "")),
        cited_evidence_ids=_tuple(value.get("cited_evidence_ids")),
        schema_version=str(value.get("schema_version", "advoice.evidence_revision.v1")),
    )


def _revision_batch_from(
    value: EvidenceRevisionBatch | Mapping[str, Any] | None,
) -> EvidenceRevisionBatch | None:
    if value is None or isinstance(value, EvidenceRevisionBatch):
        return value
    if not isinstance(value, Mapping):
        raise ConditionalAuthorityError(
            "revision_batch must be an EvidenceRevisionBatch, mapping, or None."
        )
    allowed = {
        "case_id", "state_id", "action", "expected_evidence_hash", "revisions",
        "schema_version", "batch_hash",
    }
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ConditionalAuthorityError(
            f"revision_batch includes protected or unknown fields: {unknown}"
        )
    raw_revisions = value.get("revisions", ())
    if not isinstance(raw_revisions, (list, tuple)):
        raise ConditionalAuthorityError("revision_batch revisions must be a sequence.")
    revisions: list[EvidenceRevision] = []
    for item in raw_revisions:
        parsed = _revision_from(item)
        if parsed is None:
            raise ConditionalAuthorityError("revision_batch cannot contain null revisions.")
        revisions.append(parsed)
    try:
        batch = EvidenceRevisionBatch(
            case_id=str(value.get("case_id", "")),
            state_id=str(value.get("state_id", "")),
            action=str(value.get("action", "")),  # type: ignore[arg-type]
            expected_evidence_hash=str(value.get("expected_evidence_hash", "")),
            revisions=tuple(revisions),
            schema_version=str(value.get("schema_version", "advoice.evidence_revision_batch.v1")),
        )
    except EvidenceRevisionBatchError as exc:
        raise ConditionalAuthorityError(f"Invalid revision_batch: {exc}") from exc
    supplied_hash = value.get("batch_hash")
    if supplied_hash is not None and str(supplied_hash) != batch.batch_hash:
        raise ConditionalAuthorityError("revision_batch batch_hash does not match its contents.")
    return batch


@dataclass(frozen=True, slots=True)
class AgentAuthorityDecision:
    """Structured Agent output accepted by the conditional authority path.

    The Agent's review is bound to the exact pre-Agent Module A packet and
    state graph.  ``ordinal_scores`` are evidence likelihood assessments, not
    probabilities; only :class:`ConditionalArbitrator` can use them to make a
    bounded correction.
    """

    case_id: str
    reviewed_packet_hash: str
    reviewed_evidence_hash: str
    reviewed_state_graph_hash: str
    advisor_packet_hash: str
    advisor_current: bool
    ordinal_scores: Mapping[str, int]
    revision: EvidenceRevision | None = None
    revision_batch: EvidenceRevisionBatch | None = None
    action_type: str = "review"
    incremental_evidence_ids: tuple[str, ...] = ()
    report_trace: tuple[Mapping[str, Any], ...] = ()

    def __post_init__(self) -> None:
        if not str(self.case_id).strip():
            raise ConditionalAuthorityError("Agent decision requires case_id.")
        if not isinstance(self.advisor_current, bool):
            raise ConditionalAuthorityError("advisor_current must be a boolean.")
        for field in (
            "reviewed_packet_hash", "reviewed_evidence_hash", "reviewed_state_graph_hash", "advisor_packet_hash",
        ):
            if len(str(getattr(self, field))) != 64:
                raise ConditionalAuthorityError(f"Agent decision {field} must be a SHA-256 hash.")
        if not str(self.action_type).strip():
            raise ConditionalAuthorityError("Agent decision action_type is required.")
        if self.revision is not None and not isinstance(self.revision, EvidenceRevision):
            raise ConditionalAuthorityError("revision must be an EvidenceRevision or None.")
        if self.revision_batch is not None and not isinstance(self.revision_batch, EvidenceRevisionBatch):
            raise ConditionalAuthorityError("revision_batch must be an EvidenceRevisionBatch or None.")
        if self.revision is not None and self.revision_batch is not None:
            raise ConditionalAuthorityError(
                "Agent decision may carry a legacy revision or a revision batch, never both."
            )
        scores: dict[str, int] = {}
        for key, value in dict(self.ordinal_scores).items():
            if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4:
                raise ConditionalAuthorityError("Agent ordinal_scores must be integers from 0 through 4.")
            scores[str(key)] = value
        if not scores:
            raise ConditionalAuthorityError("Agent ordinal_scores must be non-empty integers from 0 through 4.")
        object.__setattr__(self, "ordinal_scores", MappingProxyType(scores))
        object.__setattr__(self, "incremental_evidence_ids", _tuple(self.incremental_evidence_ids))
        object.__setattr__(
            self,
            "report_trace",
            tuple(_canonical_mapping(item) for item in self.report_trace),
        )

    @property
    def decision_hash(self) -> str:
        return hash_artifact(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "reviewed_packet_hash": self.reviewed_packet_hash,
            "reviewed_evidence_hash": self.reviewed_evidence_hash,
            "reviewed_state_graph_hash": self.reviewed_state_graph_hash,
            "advisor_packet_hash": self.advisor_packet_hash,
            "advisor_current": bool(self.advisor_current),
            "ordinal_scores": dict(self.ordinal_scores),
            "revision": None if self.revision is None else self.revision.to_dict(),
            "revision_batch": (
                None if self.revision_batch is None else self.revision_batch.to_dict()
            ),
            "action_type": self.action_type,
            "incremental_evidence_ids": list(self.incremental_evidence_ids),
            "report_trace": [dict(item) for item in self.report_trace],
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AgentAuthorityDecision":
        allowed = {
            "case_id", "reviewed_packet_hash", "reviewed_evidence_hash",
            "reviewed_state_graph_hash", "advisor_packet_hash", "advisor_current",
            "ordinal_scores", "revision", "revision_batch", "action_type",
            "incremental_evidence_ids", "report_trace",
        }
        unknown = sorted(set(value) - allowed)
        if unknown:
            raise ConditionalAuthorityError(
                f"Agent decision includes protected or unknown fields: {unknown}"
            )
        return cls(
            case_id=str(value.get("case_id", "")),
            reviewed_packet_hash=str(value.get("reviewed_packet_hash", "")),
            reviewed_evidence_hash=str(value.get("reviewed_evidence_hash", "")),
            reviewed_state_graph_hash=str(value.get("reviewed_state_graph_hash", "")),
            advisor_packet_hash=str(value.get("advisor_packet_hash", "")),
            advisor_current=_strict_bool(value.get("advisor_current"), "advisor_current"),
            ordinal_scores=value.get("ordinal_scores", {}),
            revision=_revision_from(value.get("revision")),
            revision_batch=_revision_batch_from(value.get("revision_batch")),
            action_type=str(value.get("action_type", "review")),
            incremental_evidence_ids=_tuple(value.get("incremental_evidence_ids")),
            report_trace=tuple(value.get("report_trace", ())),
        )


@dataclass(frozen=True, slots=True)
class AuthorityValidation:
    """Independent validation contract for Agent authority and incremental data."""

    case_id: str
    agent_decision_hash: str
    approved: bool
    incremental_evidence_ids: tuple[str, ...] = ()
    evidence_coverage: float = 0.0
    evidence_reliability: float = 0.0
    confound_burden: float = 1.0
    ood: float = 1.0
    agreement: bool = False
    route_supported: bool = False
    cross_fit_fold: str | int | None = None

    def __post_init__(self) -> None:
        if not str(self.case_id).strip() or len(str(self.agent_decision_hash)) != 64:
            raise ConditionalAuthorityError("Validator requires case_id and Agent decision hash.")
        for field in ("approved", "agreement", "route_supported"):
            if not isinstance(getattr(self, field), bool):
                raise ConditionalAuthorityError(f"{field} must be a boolean.")
        object.__setattr__(self, "incremental_evidence_ids", _tuple(self.incremental_evidence_ids))
        for field in ("evidence_coverage", "evidence_reliability", "confound_burden", "ood"):
            object.__setattr__(self, field, _finite_unit(getattr(self, field), field))

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "agent_decision_hash": self.agent_decision_hash,
            "approved": bool(self.approved),
            "incremental_evidence_ids": list(self.incremental_evidence_ids),
            "evidence_coverage": self.evidence_coverage,
            "evidence_reliability": self.evidence_reliability,
            "confound_burden": self.confound_burden,
            "ood": self.ood,
            "agreement": bool(self.agreement),
            "route_supported": bool(self.route_supported),
            "cross_fit_fold": self.cross_fit_fold,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "AuthorityValidation":
        return cls(
            case_id=str(value.get("case_id", "")),
            agent_decision_hash=str(value.get("agent_decision_hash", "")),
            approved=_strict_bool(value.get("approved"), "approved"),
            incremental_evidence_ids=_tuple(value.get("incremental_evidence_ids")),
            evidence_coverage=value.get("evidence_coverage", 0.0),
            evidence_reliability=value.get("evidence_reliability", 0.0),
            confound_burden=value.get("confound_burden", 1.0),
            ood=value.get("ood", 1.0),
            agreement=_strict_bool(value.get("agreement"), "agreement"),
            route_supported=_strict_bool(value.get("route_supported"), "route_supported"),
            cross_fit_fold=value.get("cross_fit_fold"),
        )


@dataclass(frozen=True, slots=True)
class ConditionalAuthorityResult:
    """Locked result of one opt-in conditional-authority execution."""

    route: RouteDecision
    pre_replay: EvidenceReplayResult
    post_replay: EvidenceReplayResult
    agent_decision: AgentAuthorityDecision
    validation: AuthorityValidation
    module_b: ModuleBPrediction
    decision_lock: DecisionLock
    pre_packet_artifact: Mapping[str, Any]
    post_packet_artifact: Mapping[str, Any]
    module_b_artifact: Mapping[str, Any]
    report_trace: tuple[Mapping[str, Any], ...]

    @property
    def final_probabilities(self) -> Mapping[str, float]:
        return self.module_b.probabilities


@dataclass(frozen=True, slots=True)
class PreparedAuthorityCase:
    """Immutable pre-Agent case packet produced by :meth:`prepare_case`.

    The packet is the complete handoff boundary for an Agent.  In particular,
    its four public hashes are the exact values a later
    :class:`AgentAuthorityDecision` must echo; finalization never recomputes a
    different pre-Agent snapshot behind the Agent's back.
    """

    case_id: str
    route: RouteDecision
    case_metadata: Mapping[str, Any]
    case_context: Mapping[str, Any]
    evidence: tuple[MetricEvidenceV2, ...]
    module_a_evidence: tuple[MetricEvidenceV2, ...]
    incremental_evidence: tuple[MetricEvidenceV2, ...]
    pre_replay: EvidenceReplayResult
    pre_state_cards: tuple[Mapping[str, Any], ...]
    pre_evidence_artifact: Mapping[str, Any]
    pre_state_artifact: Mapping[str, Any]
    pre_packet_artifact: Mapping[str, Any]
    reviewed_evidence_hash: str
    reviewed_state_graph_hash: str
    reviewed_packet_hash: str
    advisor_packet_hash: str


class ConditionalAuthorityExecutor:
    """Compose, but never replace, the existing typed numerical components."""

    def __init__(
        self,
        *,
        states_config: Mapping[str, Any],
        module_a: TaskConditionedStatisticalExpert,
        module_b: ConditionalArbitrator,
        module_a_state_feature_whitelist: Sequence[str],
        correlation_config: Mapping[str, Any] | None = None,
        observation_route_config: Mapping[str, Mapping[str, Any]] | None = None,
        target_route_config: Mapping[str, Mapping[str, Any]] | None = None,
        model_versions: Mapping[str, str] | None = None,
        skill_versions: Mapping[str, str] | None = None,
        tool_versions: Mapping[str, str] | None = None,
    ) -> None:
        self.states_config = dict(states_config)
        self.module_a = module_a
        self.module_b = module_b
        self.module_a_state_feature_whitelist = tuple(
            dict.fromkeys(str(value) for value in module_a_state_feature_whitelist)
        )
        if not self.module_a_state_feature_whitelist:
            raise ConditionalAuthorityError("module_a_state_feature_whitelist must be explicit and non-empty.")
        trained = tuple(getattr(module_a, "numeric_features_", ()))
        if not trained:
            raise ConditionalAuthorityError("Conditional authority requires a fitted Module A artifact.")
        if set(trained) != set(self.module_a_state_feature_whitelist):
            raise ConditionalAuthorityError(
                "Module A numeric features must exactly match module_a_state_feature_whitelist."
            )
        invalid_features = [
            feature for feature in self.module_a_state_feature_whitelist
            if not feature.startswith(("state_", "rel_", "available_"))
        ]
        if invalid_features:
            raise ConditionalAuthorityError(
                "Module A whitelist may contain only StateGraphV2 state_, rel_, or available_ features: "
                f"{invalid_features}"
            )
        self.correlation_config = None if correlation_config is None else dict(correlation_config)
        self.observation_route_config = (
            None if observation_route_config is None else dict(observation_route_config)
        )
        self.target_route_config = (
            None if target_route_config is None else dict(target_route_config)
        )
        self.model_versions = dict(model_versions or {
            "module_a": str(getattr(module_a, "module_version", "unknown")),
            "module_b": str(getattr(module_b, "module_version", "unknown")),
        })
        self.skill_versions = dict(skill_versions or {"ad_evidence_skill": "required"})
        self.tool_versions = dict(tool_versions or {"conditional_authority": "v1"})

    @staticmethod
    def _case_id(metadata: Mapping[str, Any]) -> str:
        case_id = str(metadata.get("case_id", metadata.get("subject_id", "")))
        if not case_id:
            raise ConditionalAuthorityError("case_metadata requires case_id or subject_id.")
        return case_id

    @staticmethod
    def _evidence_artifact(case_id: str, evidence: Sequence[MetricEvidenceV2]) -> dict[str, Any]:
        return {
            "schema_version": "advoice.metric_evidence_v2.snapshot.v1",
            "case_id": case_id,
            "evidence": [item.to_dict() for item in sorted(evidence, key=lambda item: item.evidence_id)],
        }

    @staticmethod
    def _state_artifact(
        case_id: str,
        replay: EvidenceReplayResult,
        state_cards: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "state_graph_v2_hash": replay.state_graph.state_hash,
            "evidence_hash": replay.audit.evidence_hash,
            "state_cards": [dict(card) for card in state_cards],
        }

    @staticmethod
    def _packet_artifact(
        case_id: str,
        packet: ExplanationPacket,
        *,
        evidence_lock_hash: str,
        state_lock_hash: str,
    ) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "packet": packet.to_dict(),
            "evidence_snapshot_hash": evidence_lock_hash,
            "state_graph_hash": state_lock_hash,
        }

    @staticmethod
    def _revision_artifact(
        case_id: str,
        replay: EvidenceReplayResult,
        revision: EvidenceRevision | EvidenceRevisionBatch | None,
    ) -> dict[str, Any]:
        return {
            "case_id": case_id,
            "snapshot_hash": replay.audit.revision_hash,
            "revision": {"action": "no_revision"} if revision is None else revision.to_dict(),
        }

    @staticmethod
    def _validate_subjects(case_id: str, evidence: Sequence[MetricEvidenceV2]) -> None:
        if not evidence:
            raise ConditionalAuthorityError("Conditional authority execution requires MetricEvidenceV2 items.")
        if any(not isinstance(item, MetricEvidenceV2) for item in evidence):
            raise TypeError("Conditional authority accepts only typed MetricEvidenceV2 inputs.")
        subjects = {str(item.subject_id) for item in evidence}
        if subjects != {case_id}:
            raise ConditionalAuthorityError(
                f"Every MetricEvidenceV2 subject_id must equal case_id {case_id!r}; got {sorted(subjects)}."
            )
        if len({item.evidence_id for item in evidence}) != len(evidence):
            raise ConditionalAuthorityError("MetricEvidenceV2 evidence_id values must be unique per case.")

    @staticmethod
    def _state_cards(
        *,
        case_id: str,
        replay: EvidenceReplayResult,
        revision_hash: str,
    ) -> list[dict[str, Any]]:
        evidence = list(replay.revised_evidence)
        cards: list[dict[str, Any]] = []
        for index, row in replay.state_graph.cards.iterrows():
            required = {"supporting_evidence_ids"}
            missing = sorted(required - set(row.index))
            if missing:
                raise ConditionalAuthorityError(
                    "StateGraphV2 must expose revision-bound evidence IDs before this opt-in path can run: "
                    f"{missing}"
                )
            state_revision = row.get("revision_hash", row.get("state_revision_hash"))
            if state_revision is None:
                raise ConditionalAuthorityError(
                    "StateGraphV2 must expose revision_hash or state_revision_hash before this opt-in path can run."
                )
            if str(state_revision) != revision_hash:
                raise StaleAuthorityError("StateGraphV2 card is bound to a stale evidence revision.")
            linked_ids = deserialize_state_card_ids(
                row["supporting_evidence_ids"],
                field="StateCard.supporting_evidence_ids",
            )
            counter_ids = deserialize_state_card_ids(
                row.get("counter_evidence_ids", row.get("counterevidence_ids", ())),
                field="StateCard.counter_evidence_ids",
            )
            linked = [item for item in evidence if item.evidence_id in set(linked_ids)]
            task_ids = sorted({str(item.task_id or "overall") for item in linked})
            segment_ids = sorted({segment for item in linked for segment in item.source_segment_ids})
            cards.append({
                "case_id": case_id,
                "state_card_id": f"{row['state_id']}:{row['task_scope']}:{index}",
                "state_id": str(row["state_id"]),
                "revision_hash": revision_hash,
                "task_id": str(row["task_scope"]),
                "task_ids": task_ids,
                "supporting_evidence_ids": list(linked_ids),
                "counter_evidence_ids": list(counter_ids),
                "segment_ids": segment_ids,
                "available": bool(row.get("available", False)),
                "report_permission": bool(row.get("report_permission", False)),
            })
        return cards

    @staticmethod
    def _evidence_for_lock(case_id: str, evidence: Iterable[MetricEvidenceV2]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for item in evidence:
            row = item.to_dict()
            row["case_id"] = case_id
            row["segment_ids"] = list(item.source_segment_ids)
            row["source_asset_id"] = item.provenance.source_asset_id
            row["evidence_role"] = "clinical_support"
            rows.append(row)
        return rows

    @staticmethod
    def _validate_agent_context(
        decision: AgentAuthorityDecision,
        *,
        case_id: str,
        packet_hash: str,
        replay_evidence_hash: str,
        state_hash: str,
        labels: Sequence[str],
    ) -> None:
        if decision.case_id != case_id:
            raise StaleAuthorityError("Agent decision is bound to a different case.")
        if decision.reviewed_packet_hash != packet_hash or decision.advisor_packet_hash != packet_hash:
            raise StaleAuthorityError("Agent decision or advisor references a stale Module A packet.")
        if not decision.advisor_current:
            raise StaleAuthorityError("Agent advisor is marked stale.")
        if decision.reviewed_evidence_hash != replay_evidence_hash:
            raise StaleAuthorityError("Agent decision references a stale evidence snapshot.")
        if decision.reviewed_state_graph_hash != state_hash:
            raise StaleAuthorityError("Agent decision references a stale StateGraphV2 artifact.")
        if tuple(sorted(decision.ordinal_scores)) != tuple(sorted(str(label) for label in labels)):
            raise ConditionalAuthorityError("Agent ordinal_scores must contain exactly the configured target labels.")
        if decision.revision is not None and decision.revision.expected_evidence_hash != replay_evidence_hash:
            raise StaleAuthorityError("Agent revision references a stale evidence snapshot.")
        if decision.revision_batch is not None:
            if decision.revision_batch.case_id != case_id:
                raise StaleAuthorityError("Agent revision batch is bound to a different case.")
            if decision.revision_batch.expected_evidence_hash != replay_evidence_hash:
                raise StaleAuthorityError("Agent revision batch references a stale evidence snapshot.")

    @staticmethod
    def _validate_incremental(
        decision: AgentAuthorityDecision,
        validation: AuthorityValidation,
        incremental: Sequence[MetricEvidenceV2],
        module_a_consumed_ids: Sequence[str],
    ) -> bool:
        declared = set(decision.incremental_evidence_ids)
        validated = set(validation.incremental_evidence_ids)
        actual = {item.evidence_id for item in incremental}
        if declared != validated:
            raise StaleAuthorityError("Validator and Agent disagree about incremental evidence IDs.")
        if declared - actual:
            raise ConditionalAuthorityError("Agent declared incremental evidence absent from the validated snapshot.")
        if declared & set(module_a_consumed_ids):
            raise ConditionalAuthorityError("Incremental evidence was already consumed by Module A.")
        selected = [item for item in incremental if item.evidence_id in declared]
        if any(
            not item.incremental_for_agent
            or item.consumed_by_supervised
            or not item.observable
            or not item.inference_permission
            or item.unavailable_reason is not None
            for item in selected
        ):
            raise ConditionalAuthorityError("Incremental evidence does not satisfy its declared authority lane.")
        return bool(validation.approved and selected)

    def prepare_case(
        self,
        *,
        case_metadata: Mapping[str, Any],
        evidence: Sequence[MetricEvidenceV2],
        case_context: Mapping[str, Any] | None = None,
    ) -> PreparedAuthorityCase:
        """Freeze one case before the Agent receives its authority packet."""

        case_id = self._case_id(case_metadata)
        route = route_case(
            case_metadata,
            observation_config=self.observation_route_config,
            target_config=self.target_route_config,
        )
        if (
            tuple(route.target_route.labels) != tuple(self.module_a.labels)
            or tuple(self.module_a.labels) != tuple(self.module_b.labels)
        ):
            raise ConditionalAuthorityError("Route, Module A, and Module B must use the same ordered labels.")
        self._validate_subjects(case_id, evidence)
        frozen_evidence = tuple(evidence)
        if route.observation_route.allowed_states:
            unsupported = sorted(
                {item.state_id for item in frozen_evidence} - set(route.observation_route.allowed_states)
            )
            if unsupported:
                raise ConditionalAuthorityError(
                    "Observation route forbids evidence states: " + ", ".join(unsupported)
                )
        module_a_evidence = tuple(item for item in frozen_evidence if item.consumed_by_supervised)
        incremental = tuple(item for item in frozen_evidence if item.incremental_for_agent)
        if not module_a_evidence:
            raise ConditionalAuthorityError(
                "At least one MetricEvidenceV2 item with consumed_by_supervised=True is required for Module A."
            )

        context = dict(case_context or {})
        injected = sorted(set(context) & set(self.module_a_state_feature_whitelist))
        if injected:
            raise ConditionalAuthorityError(
                "Module A state features must come only from StateGraphV2, not case_context: "
                f"{injected}"
            )
        if self.module_a.task_column and self.module_a.task_column not in context:
            context[self.module_a.task_column] = route.observation_route.task_id or route.observation_route.id
        if self.module_a.language_column and self.module_a.language_column not in context:
            context[self.module_a.language_column] = route.observation_route.language or "unknown"
        pre = replay_evidence(
            module_a_evidence,
            None,
            states_config=self.states_config,
            module_a=self.module_a,
            case_context=context,
            correlation_config=self.correlation_config,
            dataset_id=str(case_metadata.get("dataset_id", "conditional_authority")),
            label="unknown",
            split="inference",
        )
        pre_evidence_artifact = self._evidence_artifact(case_id, pre.revised_evidence)
        pre_evidence_hash = hash_artifact(pre_evidence_artifact)
        pre_state_cards = self._state_cards(
            case_id=case_id,
            replay=pre,
            revision_hash=pre.audit.revision_hash,
        )
        pre_state_artifact = self._state_artifact(case_id, pre, pre_state_cards)
        pre_state_hash = hash_artifact(pre_state_artifact)
        pre_packet_artifact = self._packet_artifact(
            case_id,
            pre.packet,
            evidence_lock_hash=pre_evidence_hash,
            state_lock_hash=pre_state_hash,
        )
        pre_packet_hash = hash_artifact(pre_packet_artifact)
        return PreparedAuthorityCase(
            case_id=case_id,
            route=route,
            case_metadata=MappingProxyType(dict(case_metadata)),
            case_context=MappingProxyType(context),
            evidence=frozen_evidence,
            module_a_evidence=module_a_evidence,
            incremental_evidence=incremental,
            pre_replay=pre,
            pre_state_cards=tuple(_canonical_mapping(card) for card in pre_state_cards),
            pre_evidence_artifact=MappingProxyType(pre_evidence_artifact),
            pre_state_artifact=MappingProxyType(pre_state_artifact),
            pre_packet_artifact=MappingProxyType(pre_packet_artifact),
            reviewed_evidence_hash=pre.audit.evidence_hash,
            reviewed_state_graph_hash=pre_state_hash,
            reviewed_packet_hash=pre_packet_hash,
            advisor_packet_hash=pre_packet_hash,
        )

    def finalize_case(
        self,
        *,
        prepared: PreparedAuthorityCase,
        agent_decision: AgentAuthorityDecision | Mapping[str, Any],
        validator: AuthorityValidation | Mapping[str, Any],
        segments: Sequence[Mapping[str, Any]],
        lock_id: str = "",
    ) -> ConditionalAuthorityResult:
        """Validate an Agent decision against a frozen :class:`PreparedAuthorityCase`."""

        if not isinstance(prepared, PreparedAuthorityCase):
            raise TypeError("finalize_case requires a PreparedAuthorityCase from prepare_case.")
        return self._finalize_prepared(
            prepared=prepared,
            agent_decision=agent_decision,
            validator=validator,
            segments=segments,
            lock_id=lock_id,
        )

    def execute(
        self,
        *,
        case_metadata: Mapping[str, Any],
        evidence: Sequence[MetricEvidenceV2],
        agent_decision: AgentAuthorityDecision | Mapping[str, Any],
        validator: AuthorityValidation | Mapping[str, Any],
        segments: Sequence[Mapping[str, Any]],
        case_context: Mapping[str, Any] | None = None,
        lock_id: str = "",
    ) -> ConditionalAuthorityResult:
        """Backward-compatible prepare/finalize convenience wrapper."""

        return self.finalize_case(
            prepared=self.prepare_case(
                case_metadata=case_metadata,
                evidence=evidence,
                case_context=case_context,
            ),
            agent_decision=agent_decision,
            validator=validator,
            segments=segments,
            lock_id=lock_id,
        )

    def _finalize_prepared(
        self,
        *,
        prepared: PreparedAuthorityCase,
        agent_decision: AgentAuthorityDecision | Mapping[str, Any],
        validator: AuthorityValidation | Mapping[str, Any],
        segments: Sequence[Mapping[str, Any]],
        lock_id: str = "",
    ) -> ConditionalAuthorityResult:
        """Complete one previously frozen Agent handoff."""

        case_id = prepared.case_id
        route = prepared.route
        decision = (
            agent_decision if isinstance(agent_decision, AgentAuthorityDecision)
            else AgentAuthorityDecision.from_mapping(agent_decision)
        )
        validation = validator if isinstance(validator, AuthorityValidation) else AuthorityValidation.from_mapping(validator)
        pre = prepared.pre_replay
        supervised = prepared.module_a_evidence
        incremental = prepared.incremental_evidence
        context = dict(prepared.case_context)
        pre_evidence_artifact = dict(prepared.pre_evidence_artifact)
        pre_evidence_lock_hash = hash_artifact(pre_evidence_artifact)
        pre_state_artifact = dict(prepared.pre_state_artifact)
        pre_state_lock_hash = prepared.reviewed_state_graph_hash
        pre_packet_artifact = dict(prepared.pre_packet_artifact)
        pre_packet_hash = prepared.reviewed_packet_hash
        self._validate_agent_context(
            decision,
            case_id=case_id,
            packet_hash=pre_packet_hash,
            replay_evidence_hash=prepared.reviewed_evidence_hash,
            state_hash=pre_state_lock_hash,
            labels=self.module_a.labels,
        )
        if validation.case_id != case_id or validation.agent_decision_hash != decision.decision_hash:
            raise StaleAuthorityError("Validator result is stale or bound to a different Agent decision.")
        proposal = decision.revision if decision.revision is not None else decision.revision_batch
        proposal_ids = (
            {decision.revision.evidence_id}
            if decision.revision is not None
            else set(decision.revision_batch.evidence_ids) if decision.revision_batch is not None else set()
        )
        if proposal_ids - {item.evidence_id for item in supervised}:
            raise ConditionalAuthorityError("Agent may only revise evidence consumed by the Module A replay snapshot.")

        # Validation governs authority to mutate the evidence snapshot.  A
        # rejected proposal remains visible in the validator artifact, but it
        # cannot reach replay or alter Module A's numerical packet.
        accepted_revision = proposal if validation.approved else None
        post = pre if accepted_revision is None else replay_evidence(
            supervised, accepted_revision, states_config=self.states_config, module_a=self.module_a,
            case_context=context, correlation_config=self.correlation_config,
            dataset_id=str(prepared.case_metadata.get("dataset_id", "conditional_authority")),
            label="unknown", split="inference",
        )
        post_evidence_artifact = self._evidence_artifact(case_id, post.revised_evidence)
        post_evidence_lock_hash = hash_artifact(post_evidence_artifact)
        revision_lock_hash = post.audit.revision_hash
        state_cards = self._state_cards(case_id=case_id, replay=post, revision_hash=revision_lock_hash)
        post_state_artifact = self._state_artifact(case_id, post, state_cards)
        post_state_lock_hash = hash_artifact(post_state_artifact)
        post_packet_artifact = self._packet_artifact(
            case_id, post.packet,
            evidence_lock_hash=post_evidence_lock_hash,
            state_lock_hash=post_state_lock_hash,
        )
        post_packet_hash = hash_artifact(post_packet_artifact)
        revision_artifact = self._revision_artifact(case_id, post, accepted_revision)
        consumed_by_module_a = tuple(post.packet.consumed_evidence_ids)
        validated_incremental = self._validate_incremental(
            decision, validation, incremental, post.packet.consumed_evidence_ids,
        )
        # A consumed revision is represented only through replay.  Separately
        # validated incremental IDs remain eligible for Module B under its
        # mixed-consumed/incremental contract; the two ID sets stay disjoint.
        b_incremental_ids = decision.incremental_evidence_ids if validated_incremental else ()
        b_input = {
            "module_a_pre_replay": pre.packet.calibrated_probabilities or pre.packet.raw_probabilities,
            "module_a_post_replay": post.packet.calibrated_probabilities or post.packet.raw_probabilities,
            "agent_ordinal_scores": dict(decision.ordinal_scores),
            "agent_scores_validated": bool(validation.approved),
            "eligible": bool(b_incremental_ids),
            "revision_type": "none" if accepted_revision is None else accepted_revision.action,
            "action_type": decision.action_type,
            "agreement": bool(validation.agreement),
            "evidence_coverage": validation.evidence_coverage,
            "evidence_reliability": validation.evidence_reliability,
            "confound_burden": validation.confound_burden,
            "route": route.observation_route.id,
            "language": route.observation_route.language or "unknown",
            "ood": validation.ood,
            "incremental_evidence_declared": bool(b_incremental_ids),
            "evidence_consumed_by_module_a": bool(consumed_by_module_a),
            "incremental_evidence_ids": list(b_incremental_ids),
            "consumed_evidence_ids": list(consumed_by_module_a),
            "route_supported": bool(validation.route_supported),
            "replay_performed": accepted_revision is not None,
        }
        module_b = self.module_b.predict_one(b_input)
        module_b_artifact = {
            "case_id": case_id,
            "module_a_post_replay_hash": post_packet_hash,
            "output": module_b.to_dict(),
        }
        metric_evidence = self._evidence_for_lock(case_id, tuple(post.revised_evidence) + incremental)
        agent_artifact = decision.to_dict()
        validator_artifact = validation.to_dict()
        lock = create_decision_lock(
            case_id=case_id,
            evidence_snapshot=post_evidence_artifact,
            evidence_snapshot_hash=post_evidence_lock_hash,
            revision=revision_artifact,
            revision_hash=revision_lock_hash,
            state_graph=post_state_artifact,
            state_graph_hash=post_state_lock_hash,
            module_a_pre_replay=pre_packet_artifact,
            module_a_pre_replay_hash=pre_packet_hash,
            module_a_post_replay=post_packet_artifact,
            module_a_post_replay_hash=post_packet_hash,
            agent_decision=agent_artifact,
            agent_decision_hash=decision.decision_hash,
            validator_result=validator_artifact,
            validator_hash=hash_artifact(validator_artifact),
            module_b_output=module_b_artifact,
            module_b_output_hash=hash_artifact(module_b_artifact),
            model_versions=self.model_versions,
            skill_versions=self.skill_versions,
            tool_versions=self.tool_versions,
            report_trace=[dict(item) for item in decision.report_trace],
            metric_evidence=metric_evidence,
            segments=segments,
            state_cards=state_cards,
            lock_id=lock_id,
        )
        return ConditionalAuthorityResult(
            route=route,
            pre_replay=pre,
            post_replay=post,
            agent_decision=decision,
            validation=validation,
            module_b=module_b,
            decision_lock=lock,
            pre_packet_artifact=MappingProxyType(pre_packet_artifact),
            post_packet_artifact=MappingProxyType(post_packet_artifact),
            module_b_artifact=MappingProxyType(module_b_artifact),
            report_trace=tuple(_canonical_mapping(item) for item in decision.report_trace),
        )

    @staticmethod
    def render_locked_report(result: ConditionalAuthorityResult) -> dict[str, Any]:
        """Render a report only after verifying the immutable DecisionLock."""

        lock = result.decision_lock
        if hash_artifact(result.module_b_artifact) != lock.module_b_output_hash:
            raise HashMismatchError("Result Module B output no longer matches its DecisionLock.")
        report = {
            "locked": True,
            "lock_id": lock.lock_id,
            "decision_lock_hash": lock.decision_hash,
            "case_id": lock.case_id,
            "report_trace": lock.report_trace.to_dict(),
            "prediction": {
                "label": result.module_b.predicted_label,
                "probabilities": dict(result.module_b.probabilities),
                "source": "conditional_authority",
            },
        }
        validate_locked_report(report, lock)
        return report


__all__ = [
    "AgentAuthorityDecision", "AuthorityValidation", "ConditionalAuthorityError",
    "ConditionalAuthorityExecutor", "ConditionalAuthorityResult", "PreparedAuthorityCase",
    "StaleAuthorityError",
]
