from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import pandas as pd
import pytest

from advoice.agent_runtime import case_pseudonym
from advoice.authority_review_runtime import (
    AdvisorReconciliation,
    AuthorityReviewResult,
    BlindEvidenceAssessment,
    REVIEW_AVAILABLE,
    REVIEW_PROVIDER_ERROR,
    REVIEW_UNAVAILABLE,
    StateReviewAction,
)
from advoice.authority_review_transaction_bridge import (
    AuthorityReviewTransactionBridgeError,
    compile_authority_review_decision,
    compile_authority_review_transaction,
)
from advoice.conditional_authority import PreparedAuthorityCase
from advoice.decision_lock import hash_artifact
from advoice.evidence import EvidencePermissions, EvidenceProvenance, MetricEvidenceV2
from advoice.routing import ObservationRoute, RouteDecision, TargetRoute
from advoice.state_graph import StateGraphV2
from advoice.evidence_replay import evidence_snapshot_hash


LABELS = ("HC", "MCI", "AD")


def _evidence(evidence_id: str, state_id: str) -> MetricEvidenceV2:
    return MetricEvidenceV2(
        evidence_id=evidence_id,
        metric_id=f"metric:{state_id}",
        metric_instance_id=evidence_id,
        subject_id="case-1",
        session_id="session-1",
        case_id="case-1",
        state_id=state_id,
        task_id="picture_description",
        value=1.0,
        direction=1,
        provenance=EvidenceProvenance(source_asset_id="fixture"),
        permissions=EvidencePermissions(inference=True, report=True),
        consumed_by_supervised=True,
    )


def _graph(evidence: tuple[MetricEvidenceV2, ...]) -> StateGraphV2:
    frame = pd.DataFrame([
        {
            "dataset_id": "fixture",
            "subject_id": item.subject_id,
            "case_id": item.case_id,
            "label": "hidden",
            "split": "inference",
            "evidence_id": item.evidence_id,
            "metric_id": item.metric_id,
            "metric_instance_id": item.metric_instance_id,
            "state_id": item.state_id,
            "task_scope": item.task_id,
            "directional_z": 1.0,
            "reliability": 1.0,
            "missing": False,
            "evidence_status": "available",
            "report_permission": True,
        }
        for item in evidence
    ])
    return StateGraphV2.from_evidence_frame(
        frame,
        {"states": [
            {
                "id": state_id,
                "metrics": [item.metric_id for item in evidence if item.state_id == state_id],
                "weights": [1.0 for item in evidence if item.state_id == state_id],
            }
            for state_id in sorted({item.state_id for item in evidence})
        ]},
    )


def _prepared() -> PreparedAuthorityCase:
    evidence = (_evidence("metric:s01", "S01"), _evidence("metric:s02", "S02"))
    state_artifact = {"evidence_hash": evidence_snapshot_hash(evidence), "states": ["S01", "S02"]}
    evidence_artifact = {"evidence": [item.to_dict() for item in evidence]}
    state_hash = hash_artifact(state_artifact)
    packet_artifact = {
        "state_graph_hash": state_hash,
        "evidence_snapshot_hash": hash_artifact(evidence_artifact),
        "packet": {"fixture": True},
    }
    packet_hash = hash_artifact(packet_artifact)
    route = RouteDecision(
        ObservationRoute("picture_description", "picture_description", task_id="cookie", language="en"),
        TargetRoute("diagnosis", "cross_sectional_diagnosis", LABELS),
    )
    packet = SimpleNamespace(
        class_order=LABELS,
        raw_probabilities={label: 1 / len(LABELS) for label in LABELS},
        calibrated_probabilities=None,
        feature_contributions={label: {} for label in LABELS},
        branch_contributions={label: {} for label in LABELS},
        uncertainty={},
        hashes={},
    )
    cards = tuple({
        "case_id": "case-1",
        "state_card_id": f"{state_id}:cookie:0",
        "state_id": state_id,
        "task_id": "cookie",
        "task_ids": ["cookie"],
        "supporting_evidence_ids": [item.evidence_id for item in evidence if item.state_id == state_id],
        "counter_evidence_ids": [],
        "segment_ids": [],
        "available": True,
        "report_permission": True,
    } for state_id in ("S01", "S02"))
    return PreparedAuthorityCase(
        case_id="case-1",
        route=route,
        case_metadata={},
        case_context={},
        evidence=evidence,
        module_a_evidence=evidence,
        incremental_evidence=(),
        pre_replay=SimpleNamespace(state_graph=_graph(evidence), packet=packet),
        pre_state_cards=cards,
        pre_evidence_artifact=evidence_artifact,
        pre_state_artifact=state_artifact,
        pre_packet_artifact=packet_artifact,
        reviewed_evidence_hash=evidence_snapshot_hash(evidence),
        reviewed_state_graph_hash=state_hash,
        reviewed_packet_hash=packet_hash,
        advisor_packet_hash=packet_hash,
    )


def _action(prepared: PreparedAuthorityCase, state_id: str, action: str) -> StateReviewAction:
    evidence_id = next(item.evidence_id for item in prepared.evidence if item.state_id == state_id)
    return StateReviewAction(
        state_id=state_id,
        action=action,
        cited_metric_evidence_ids=(evidence_id,),
        reliability_multiplier=0.5 if action == "downweight" else 0.0,
        rationale=f"Review action for {state_id}.",
    )


def _result(prepared: PreparedAuthorityCase, actions: dict[str, StateReviewAction]) -> AuthorityReviewResult:
    pseudo = case_pseudonym(prepared.case_id)
    blind = BlindEvidenceAssessment(
        case_id=pseudo,
        reviewed_packet_hash=prepared.reviewed_packet_hash,
        reviewed_evidence_hash=prepared.reviewed_evidence_hash,
        reviewed_state_graph_hash=prepared.reviewed_state_graph_hash,
        state_actions=actions,
        ordinal_scores={"HC": 2, "MCI": 2, "AD": 1},
        report_trace=("Evidence was reviewed state by state.",),
    )
    reconciliation = AdvisorReconciliation(
        case_id=pseudo,
        reviewed_packet_hash=prepared.reviewed_packet_hash,
        reviewed_evidence_hash=prepared.reviewed_evidence_hash,
        reviewed_state_graph_hash=prepared.reviewed_state_graph_hash,
        advisor_packet_hash=prepared.advisor_packet_hash,
        disposition="retain",
        amendments={},
        cited_metric_evidence_ids=(prepared.evidence[0].evidence_id,),
        rationale="No further evidence amendment was required.",
    )
    return AuthorityReviewResult(
        status=REVIEW_AVAILABLE,
        case_id=pseudo,
        blind_request_hash="a" * 64,
        reconciliation_request_hash="b" * 64,
        blind_assessment=blind,
        reconciliation=reconciliation,
        effective_state_actions=actions,
    )


def test_two_non_retain_states_compile_into_one_stably_ordered_transaction() -> None:
    prepared = _prepared()
    result = _result(prepared, {
        "S02": _action(prepared, "S02", "invalidate"),
        "S01": _action(prepared, "S01", "downweight"),
    })

    transaction = compile_authority_review_transaction(prepared, result)

    assert transaction is not None
    assert transaction.case_id == prepared.case_id
    assert transaction.expected_evidence_hash == prepared.reviewed_evidence_hash
    assert tuple(batch.state_id for batch in transaction.batches) == ("S01", "S02")
    assert transaction.evidence_ids == ("metric:s01", "metric:s02")


def test_empty_or_retain_only_actions_return_no_transaction_and_merge_metadata() -> None:
    prepared = _prepared()
    result = _result(prepared, {})

    assert compile_authority_review_transaction(prepared, result) is None
    compiled = compile_authority_review_decision(prepared, result)

    assert compiled.transaction is None
    assert compiled.decision.revision_transaction is None
    assert dict(compiled.decision.ordinal_scores) == {"AD": 1, "HC": 2, "MCI": 2}
    assert [item["source"] for item in compiled.decision.report_trace] == [
        "advisor_reconciliation", "blind_assessment",
    ]


def test_action_on_already_unavailable_state_is_audited_noop() -> None:
    prepared = _prepared()
    unavailable = replace(
        prepared.evidence[0],
        permissions=EvidencePermissions(inference=False, report=False),
        observable=False,
        unavailable_reason="already_unavailable",
    )
    prepared = replace(
        prepared,
        evidence=(unavailable,) + prepared.evidence[1:],
        module_a_evidence=(unavailable,) + prepared.module_a_evidence[1:],
    )
    result = _result(prepared, {"S01": _action(prepared, "S01", "mark_unavailable")})

    compiled = compile_authority_review_decision(prepared, result)

    assert compiled.transaction is None
    assert compiled.compiled_reviews == ()
    assert compiled.decision.action_type == "evidence_review_noop"
    assert any(
        item.get("state_id") == "S01" and item.get("action") == "mark_unavailable"
        for item in compiled.decision.report_trace
    )


@pytest.mark.parametrize("status", [REVIEW_UNAVAILABLE, REVIEW_PROVIDER_ERROR])
def test_unavailable_or_failed_review_is_rejected(status: str) -> None:
    prepared = _prepared()
    result = AuthorityReviewResult(
        status=status,
        case_id=case_pseudonym(prepared.case_id),
        blind_request_hash="a" * 64,
        reconciliation_request_hash=None,
        error="fixture",
    )

    with pytest.raises(AuthorityReviewTransactionBridgeError, match="only 'available'"):
        compile_authority_review_transaction(prepared, result)


def test_stale_provenance_is_rejected_before_compilation() -> None:
    prepared = _prepared()
    result = _result(prepared, {"S01": _action(prepared, "S01", "downweight")})
    stale = BlindEvidenceAssessment(
        case_id=result.case_id,
        reviewed_packet_hash="d" * 64,
        reviewed_evidence_hash=prepared.reviewed_evidence_hash,
        reviewed_state_graph_hash=prepared.reviewed_state_graph_hash,
        state_actions=result.blind_assessment.state_actions,
        ordinal_scores=result.blind_assessment.ordinal_scores,
        report_trace=result.blind_assessment.report_trace,
    )
    stale_result = AuthorityReviewResult(
        status=REVIEW_AVAILABLE,
        case_id=result.case_id,
        blind_request_hash=result.blind_request_hash,
        reconciliation_request_hash=result.reconciliation_request_hash,
        blind_assessment=stale,
        reconciliation=result.reconciliation,
        effective_state_actions=result.effective_state_actions,
    )

    with pytest.raises(AuthorityReviewTransactionBridgeError, match="stale reviewed_packet_hash"):
        compile_authority_review_transaction(prepared, stale_result)


def test_decision_trace_is_deterministic_and_contains_each_compiled_state() -> None:
    prepared = _prepared()
    result = _result(prepared, {
        "S02": _action(prepared, "S02", "invalidate"),
        "S01": _action(prepared, "S01", "downweight"),
    })

    left = compile_authority_review_decision(prepared, result)
    right = compile_authority_review_decision(prepared, result)

    assert left.decision.to_dict() == right.decision.to_dict()
    assert [item.review.state_id for item in left.compiled_reviews] == ["S01", "S02"]
    assert {item["state_id"] for item in left.decision.report_trace if item["source"] == "state_action"} == {"S01", "S02"}
