"""Bounded bridge from a two-pass Agent review to a case transaction.

The runtime deliberately keeps Agent actions at the state level.  This module
is the only adapter that turns the effective non-retain actions into the
case-level atomic revision accepted by conditional authority.  It does not
read labels, probabilities, or raw case metadata.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .agent_runtime import case_pseudonym
from .authority_agent_bridge import (
    AgentStateReview,
    CompiledAgentStateReview,
    compile_agent_state_review,
)
from .authority_review_runtime import (
    AuthorityReviewResult,
    REVIEW_AVAILABLE,
    StateReviewAction,
)
from .conditional_authority import AgentAuthorityDecision, PreparedAuthorityCase
from .decision_lock import canonical_json
from .evidence_revision_batch import inferable_supervised_state_evidence
from .evidence_revision_transaction import EvidenceRevisionTransaction


class AuthorityReviewTransactionBridgeError(ValueError):
    """Raised when a review cannot be bound to one prepared case transaction."""


@dataclass(frozen=True, slots=True)
class CompiledAuthorityReview:
    """The transaction and decision produced from one validated review."""

    decision: AgentAuthorityDecision
    compiled_reviews: tuple[CompiledAgentStateReview, ...]

    @property
    def transaction(self) -> EvidenceRevisionTransaction | None:
        return self.decision.revision_transaction


def _require_available(
    prepared: PreparedAuthorityCase,
    result: AuthorityReviewResult,
) -> None:
    if not isinstance(prepared, PreparedAuthorityCase):
        raise TypeError("prepared must be a PreparedAuthorityCase.")
    if not isinstance(result, AuthorityReviewResult):
        raise TypeError("result must be an AuthorityReviewResult.")
    if result.status != REVIEW_AVAILABLE:
        raise AuthorityReviewTransactionBridgeError(
            f"Cannot compile review status {result.status!r}; only {REVIEW_AVAILABLE!r} is accepted."
        )
    expected_case = case_pseudonym(prepared.case_id)
    if result.case_id != expected_case:
        raise AuthorityReviewTransactionBridgeError(
            "Authority review is bound to a different pseudonymous case."
        )
    blind = result.blind_assessment
    reconciliation = result.reconciliation
    if blind is None or reconciliation is None or result.effective_state_actions is None:
        raise AuthorityReviewTransactionBridgeError(
            "Available authority review is missing one of its validated passes."
        )
    expected = {
        "case_id": expected_case,
        "reviewed_packet_hash": prepared.reviewed_packet_hash,
        "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
        "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
    }
    if blind.case_id != expected_case:
        raise AuthorityReviewTransactionBridgeError(
            "Blind assessment is bound to a different pseudonymous case."
        )
    for name, value in expected.items():
        if name != "case_id" and getattr(blind, name) != value:
            raise AuthorityReviewTransactionBridgeError(
                f"Blind assessment contains stale {name}."
            )
        if getattr(reconciliation, name) != value:
            raise AuthorityReviewTransactionBridgeError(
                f"Advisor reconciliation contains stale {name}."
            )
    if reconciliation.advisor_packet_hash != prepared.advisor_packet_hash:
        raise AuthorityReviewTransactionBridgeError(
            "Advisor reconciliation contains a stale advisor packet hash."
        )


def _effective_actions(
    result: AuthorityReviewResult,
) -> tuple[StateReviewAction, ...]:
    actions = result.effective_state_actions
    if actions is None:
        raise AuthorityReviewTransactionBridgeError(
            "Available authority review has no effective state action mapping."
        )
    if not isinstance(actions, Mapping):
        raise AuthorityReviewTransactionBridgeError(
            "Effective state actions must be a mapping."
        )
    parsed: list[StateReviewAction] = []
    for state_id in sorted(actions, key=str):
        action = actions[state_id]
        if not isinstance(action, StateReviewAction):
            raise AuthorityReviewTransactionBridgeError(
                f"Effective action for {state_id!r} is not a StateReviewAction."
            )
        if action.state_id != str(state_id):
            raise AuthorityReviewTransactionBridgeError(
                f"Effective action key {state_id!r} does not match action state_id {action.state_id!r}."
            )
        if action.action != "retain":
            parsed.append(action)
    return tuple(parsed)


def _as_state_review(
    prepared: PreparedAuthorityCase,
    action: StateReviewAction,
    ordinal_scores: Mapping[str, int],
) -> AgentStateReview:
    blind = action
    # StateReviewAction is already validated by AuthorityReviewRuntime.  The
    # bridge adds the raw case binding only after the runtime's pseudonymous
    # boundary has been checked above.
    return AgentStateReview(
        case_id=prepared.case_id,
        reviewed_packet_hash=prepared.reviewed_packet_hash,
        reviewed_evidence_hash=prepared.reviewed_evidence_hash,
        reviewed_state_graph_hash=prepared.reviewed_state_graph_hash,
        state_id=blind.state_id,
        action=blind.action,
        rationale=blind.rationale,
        cited_evidence_ids=blind.cited_metric_evidence_ids,
        ordinal_scores=ordinal_scores,
        report_trace=(),
        reliability_multiplier=(
            blind.reliability_multiplier if blind.action == "downweight" else None
        ),
    )


def _compile_states(
    prepared: PreparedAuthorityCase,
    actions: tuple[StateReviewAction, ...],
    ordinal_scores: Mapping[str, int],
) -> tuple[CompiledAgentStateReview, ...]:
    compiled: list[CompiledAgentStateReview] = []
    for action in actions:
        if not inferable_supervised_state_evidence(
            prepared.module_a_evidence, state_id=action.state_id,
        ):
            # The report trace still records this action, but there is no
            # replayable evidence left in the state.  Avoid fabricating a
            # transaction or requiring a graph row that was already removed.
            continue
        review = _as_state_review(prepared, action, ordinal_scores)
        try:
            compiled.append(compile_agent_state_review(prepared, review))
        except ValueError as exc:
            raise AuthorityReviewTransactionBridgeError(
                f"Failed to compile effective action for state {action.state_id!r}: {exc}"
            ) from exc
    return tuple(sorted(compiled, key=lambda item: item.review.state_id))


def _merge_report_trace(
    result: AuthorityReviewResult,
    actions: tuple[StateReviewAction, ...],
) -> tuple[Mapping[str, Any], ...]:
    """Create a stable, state-addressable trace without discarding rationale."""

    blind = result.blind_assessment
    reconciliation = result.reconciliation
    if blind is None or reconciliation is None:
        raise AuthorityReviewTransactionBridgeError("Available review passes are required.")
    records: list[dict[str, Any]] = []
    for index, text in enumerate(blind.report_trace):
        records.append({"source": "blind_assessment", "index": index, "text": str(text)})
    for action in actions:
        records.append({
            "source": "state_action",
            "state_id": action.state_id,
            "action": action.action,
            "cited_metric_evidence_ids": list(action.cited_metric_evidence_ids),
            "rationale": action.rationale,
        })
    records.append({
        "source": "advisor_reconciliation",
        "disposition": reconciliation.disposition,
        "cited_metric_evidence_ids": list(reconciliation.cited_metric_evidence_ids),
        "rationale": reconciliation.rationale,
    })
    unique: dict[str, Mapping[str, Any]] = {
        canonical_json(record): record for record in records
    }
    return tuple(unique[key] for key in sorted(unique))


def compile_authority_review_transaction(
    prepared: PreparedAuthorityCase,
    result: AuthorityReviewResult,
) -> EvidenceRevisionTransaction | None:
    """Compile every effective non-retain action into one atomic transaction.

    An empty effective action mapping is an explicit no-op and returns
    ``None``.  Any unavailable/failed review, stale provenance, or compilation
    failure raises rather than silently falling back to a partial transaction.
    """

    _require_available(prepared, result)
    actions = _effective_actions(result)
    if not actions:
        return None
    blind = result.blind_assessment
    if blind is None:
        raise AuthorityReviewTransactionBridgeError("Available review requires a blind assessment.")
    compiled = _compile_states(prepared, actions, blind.ordinal_scores)
    batches = tuple(
        item.batch for item in compiled if item.batch.action != "retain"
    )
    if not batches:
        return None
    try:
        return EvidenceRevisionTransaction(
            case_id=prepared.case_id,
            expected_evidence_hash=prepared.reviewed_evidence_hash,
            batches=batches,
        )
    except ValueError as exc:
        raise AuthorityReviewTransactionBridgeError(
            f"Failed to assemble the atomic multi-state transaction: {exc}"
        ) from exc


def compile_authority_review_decision(
    prepared: PreparedAuthorityCase,
    result: AuthorityReviewResult,
) -> CompiledAuthorityReview:
    """Return the transaction plus the merged bounded decision metadata."""

    _require_available(prepared, result)
    actions = _effective_actions(result)
    blind = result.blind_assessment
    if blind is None:
        raise AuthorityReviewTransactionBridgeError("Available review requires a blind assessment.")
    compiled = _compile_states(prepared, actions, blind.ordinal_scores)
    transaction = None
    active_batches = tuple(
        item.batch for item in compiled if item.batch.action != "retain"
    )
    if active_batches:
        transaction = EvidenceRevisionTransaction(
            case_id=prepared.case_id,
            expected_evidence_hash=prepared.reviewed_evidence_hash,
            batches=active_batches,
        )
    decision = AgentAuthorityDecision(
        case_id=prepared.case_id,
        reviewed_packet_hash=prepared.reviewed_packet_hash,
        reviewed_evidence_hash=prepared.reviewed_evidence_hash,
        reviewed_state_graph_hash=prepared.reviewed_state_graph_hash,
        advisor_packet_hash=prepared.advisor_packet_hash,
        advisor_current=True,
        ordinal_scores=dict(sorted(blind.ordinal_scores.items())),
        revision_transaction=transaction,
        action_type="multi_state_evidence_review" if transaction is not None else "evidence_review_noop",
        incremental_evidence_ids=(),
        report_trace=_merge_report_trace(result, actions),
    )
    return CompiledAuthorityReview(decision=decision, compiled_reviews=compiled)


__all__ = [
    "AuthorityReviewTransactionBridgeError",
    "CompiledAuthorityReview",
    "compile_authority_review_decision",
    "compile_authority_review_transaction",
]
