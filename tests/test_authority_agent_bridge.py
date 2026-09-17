from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError
from types import SimpleNamespace

import pandas as pd
import pytest

from advoice.authority_agent_bridge import (
    AgentStateReview,
    AgentStateReviewError,
    compile_agent_state_review,
)
from advoice.conditional_authority import PreparedAuthorityCase
from advoice.evidence import EvidencePermissions, MetricEvidenceV2
from advoice.evidence_replay import evidence_snapshot_hash
from advoice.state_graph import StateGraphV2


def _evidence(evidence_id: str, *, state_id: str = "S01") -> MetricEvidenceV2:
    return MetricEvidenceV2(
        evidence_id=evidence_id,
        metric_id=evidence_id,
        metric_instance_id=evidence_id,
        subject_id="case-1",
        case_id="case-1",
        state_id=state_id,
        task_id="picture",
        value=1.0,
        direction=1,
        consumed_by_supervised=True,
        permissions=EvidencePermissions(inference=True, report=True),
    )


def _graph(evidence: tuple[MetricEvidenceV2, ...]) -> StateGraphV2:
    frame = pd.DataFrame(
        [
            {
                "dataset_id": "fixture",
                "subject_id": item.subject_id,
                "case_id": item.case_id,
                "label": "unknown",
                "split": "inference",
                "evidence_id": item.evidence_id,
                "metric_id": item.metric_id,
                "metric_instance_id": item.metric_instance_id,
                "state_id": item.state_id,
                "task_scope": item.task_id or "overall",
                "directional_z": 1.0,
                "reliability": 1.0,
                "missing": False,
                "evidence_status": "available",
                "report_permission": True,
            }
            for item in evidence
        ]
    )
    states = sorted({item.state_id for item in evidence})
    return StateGraphV2.from_evidence_frame(
        frame,
        {
            "states": [
                {
                    "id": state_id,
                    "metrics": [item.metric_id for item in evidence if item.state_id == state_id],
                    "weights": [1.0 for item in evidence if item.state_id == state_id],
                }
                for state_id in states
            ]
        },
    )


def _prepared(*evidence: MetricEvidenceV2) -> PreparedAuthorityCase:
    items = tuple(evidence) or (_evidence("metric:pause"),)
    return PreparedAuthorityCase(
        case_id="case-1",
        route=None,  # type: ignore[arg-type]
        case_metadata={},
        case_context={},
        evidence=items,
        module_a_evidence=items,
        incremental_evidence=(),
        pre_replay=SimpleNamespace(state_graph=_graph(items)),  # type: ignore[arg-type]
        pre_state_cards=(),
        pre_evidence_artifact={},
        pre_state_artifact={},
        pre_packet_artifact={},
        reviewed_evidence_hash=evidence_snapshot_hash(items),
        reviewed_state_graph_hash="b" * 64,
        reviewed_packet_hash="a" * 64,
        advisor_packet_hash="c" * 64,
    )


def _review(prepared: PreparedAuthorityCase, **overrides: object) -> dict[str, object]:
    value: dict[str, object] = {
        "case_id": prepared.case_id,
        "reviewed_packet_hash": prepared.reviewed_packet_hash,
        "reviewed_evidence_hash": prepared.reviewed_evidence_hash,
        "reviewed_state_graph_hash": prepared.reviewed_state_graph_hash,
        "state_id": "S01",
        "action": "retain",
        "reliability_multiplier": None,
        "rationale": "The state evidence remains reliable.",
        "cited_evidence_ids": [prepared.module_a_evidence[0].evidence_id],
        "ordinal_scores": {"HC": 3, "MCI": 1, "AD": 0},
        "report_trace": [{"claim_id": "claim-1", "metric_evidence_ids": [prepared.module_a_evidence[0].evidence_id]}],
    }
    value.update(overrides)
    return value


def test_valid_retain_compiles_hash_bound_noop_decision() -> None:
    prepared = _prepared()

    compiled = compile_agent_state_review(prepared, _review(prepared))

    assert compiled.batch.action == "retain"
    assert compiled.batch.revisions == ()
    assert compiled.decision.revision is None
    assert compiled.decision.case_id == prepared.case_id
    assert compiled.decision.reviewed_packet_hash == prepared.reviewed_packet_hash
    assert compiled.decision.reviewed_evidence_hash == prepared.reviewed_evidence_hash
    assert compiled.decision.reviewed_state_graph_hash == prepared.reviewed_state_graph_hash
    assert compiled.decision.advisor_packet_hash == prepared.advisor_packet_hash
    assert compiled.decision.incremental_evidence_ids == ()


def test_valid_one_metric_downweight_binds_the_compiled_revision() -> None:
    prepared = _prepared()
    review = _review(prepared, action="downweight", reliability_multiplier=0.5)

    compiled = compile_agent_state_review(prepared, review)

    assert compiled.batch.action == "downweight"
    assert len(compiled.batch.revisions) == 1
    assert compiled.decision.revision is compiled.batch.revisions[0]
    assert compiled.decision.revision.reliability_multiplier == 0.5


@pytest.mark.parametrize("action", ["invalidate", "mark_unavailable"])
def test_canonical_exclusion_action_compiles_to_one_atomic_revision(action: str) -> None:
    prepared = _prepared()

    compiled = compile_agent_state_review(prepared, _review(prepared, action=action))

    assert compiled.review.action == action
    assert compiled.batch.action == action
    assert compiled.decision.action_type == action
    assert compiled.decision.revision is compiled.batch.revisions[0]


def test_unknown_review_field_is_rejected() -> None:
    prepared = _prepared()

    with pytest.raises(AgentStateReviewError, match="unknown fields"):
        AgentStateReview.from_mapping(_review(prepared, probability=0.9))


def test_missing_required_review_field_is_rejected() -> None:
    prepared = _prepared()
    value = _review(prepared)
    del value["reviewed_packet_hash"]

    with pytest.raises(AgentStateReviewError, match="missing required fields"):
        AgentStateReview.from_mapping(value)


@pytest.mark.parametrize("field", ["label", "split", "truth", "true_label", "ground_truth"])
def test_raw_label_split_or_truth_field_is_rejected_at_any_depth(field: str) -> None:
    prepared = _prepared()
    value = _review(
        prepared,
        report_trace=[{"claim_id": "claim-1", "nested": {field: "AD"}}],
    )

    with pytest.raises(AgentStateReviewError, match="forbidden raw field"):
        AgentStateReview.from_mapping(value)


@pytest.mark.parametrize(
    "field",
    ["reviewed_packet_hash", "reviewed_evidence_hash", "reviewed_state_graph_hash"],
)
def test_stale_review_hashes_are_rejected(field: str) -> None:
    prepared = _prepared()

    with pytest.raises(AgentStateReviewError, match="stale"):
        compile_agent_state_review(prepared, _review(prepared, **{field: "d" * 64}))


def test_cited_evidence_must_belong_to_the_target_state() -> None:
    prepared = _prepared(_evidence("metric:s01"), _evidence("metric:s02", state_id="S02"))

    with pytest.raises(AgentStateReviewError, match="target state"):
        compile_agent_state_review(
            prepared,
            _review(prepared, cited_evidence_ids=["metric:s02"]),
        )


def test_non_retain_multi_metric_state_fails_closed_without_dropping_revisions() -> None:
    prepared = _prepared(_evidence("metric:one"), _evidence("metric:two"))

    with pytest.raises(AgentStateReviewError, match="batch execution not yet supported"):
        compile_agent_state_review(
            prepared,
            _review(
                prepared,
                action="downweight",
                reliability_multiplier=0.5,
                cited_evidence_ids=["metric:one", "metric:two"],
            ),
        )


def test_review_and_compiled_hashes_are_deterministic() -> None:
    prepared = _prepared()
    left_input = _review(prepared)
    right_input = dict(reversed(list(left_input.items())))
    right_input["ordinal_scores"] = {"AD": 0, "MCI": 1, "HC": 3}

    left_review = AgentStateReview.from_mapping(left_input)
    right_review = AgentStateReview.from_mapping(right_input)
    left = compile_agent_state_review(prepared, left_review)
    right = compile_agent_state_review(prepared, right_review)

    assert left_review.to_json() == right_review.to_json()
    assert left_review.review_hash == right_review.review_hash
    assert left.to_json() == right.to_json()
    assert left.compiled_hash == right.compiled_hash


def test_parsing_and_compilation_do_not_mutate_or_alias_input() -> None:
    prepared = _prepared()
    value = _review(prepared)
    before = deepcopy(value)

    review = AgentStateReview.from_mapping(value)
    compiled = compile_agent_state_review(prepared, review)
    assert value == before

    value["ordinal_scores"]["HC"] = 0  # type: ignore[index]
    value["report_trace"][0]["claim_id"] = "changed"  # type: ignore[index]

    assert before == _review(prepared)
    assert review.ordinal_scores["HC"] == 3
    assert review.report_trace[0]["claim_id"] == "claim-1"
    assert compiled.decision.ordinal_scores["HC"] == 3
    assert compiled.decision.report_trace[0]["claim_id"] == "claim-1"
    with pytest.raises(FrozenInstanceError):
        review.action = "invalidate"  # type: ignore[misc]


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"action": "exclude"}, "action"),
        ({"ordinal_scores": {"HC": 5}}, "ordinal_scores"),
        ({"ordinal_scores": {"HC": True}}, "ordinal_scores"),
        ({"reviewed_packet_hash": "not-a-hash"}, "SHA-256"),
    ],
)
def test_invalid_review_values_are_rejected(overrides: dict[str, object], match: str) -> None:
    prepared = _prepared()

    with pytest.raises(AgentStateReviewError, match=match):
        AgentStateReview.from_mapping(_review(prepared, **overrides))


def test_case_and_state_mismatches_are_rejected() -> None:
    prepared = _prepared()

    with pytest.raises(AgentStateReviewError, match="different case"):
        compile_agent_state_review(prepared, _review(prepared, case_id="case-2"))
    with pytest.raises(AgentStateReviewError, match="state mismatch"):
        compile_agent_state_review(prepared, _review(prepared, state_id="S99"))
