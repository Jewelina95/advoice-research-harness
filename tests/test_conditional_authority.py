from __future__ import annotations

from dataclasses import replace

import pytest
import pandas as pd

from advoice.conditional_authority import (
    AgentAuthorityDecision,
    AuthorityValidation,
    ConditionalAuthorityError,
    ConditionalAuthorityExecutor,
    StaleAuthorityError,
)
from advoice.decision_lock import hash_artifact
from advoice.evidence import EvidencePermissions, EvidenceProvenance, MetricEvidenceV2, ReferenceMetadata
from advoice.evidence_replay import EvidenceRevision, replay_evidence
from advoice.module_a import TaskConditionedStatisticalExpert
from advoice.module_b import ConditionalArbitrator


LABELS = ("HC", "MCI", "AD")
STATES = {"states": [{"id": "S01", "metrics": ["pause_a", "pause_b"], "weights": [1.0, 1.0]}]}
CASE = {
    "case_id": "case-1",
    "dataset_id": "synthetic",
    "channel": "picture_description",
    "target": "diagnosis",
    "task_id": "cookie",
    "language": "en",
    "allowed_states": ["S01"],
}


def _expert() -> TaskConditionedStatisticalExpert:
    frame = pd.DataFrame({"state_S01": [-3.0, -2.0, -1.0, 0.8, 1.8, 2.8, 4.0, 5.0, 6.0]})
    return TaskConditionedStatisticalExpert(LABELS, c=0.6).fit(
        frame,
        ["HC", "HC", "HC", "MCI", "MCI", "MCI", "AD", "AD", "AD"],
        feature_columns=["state_S01"],
        artifact_snapshot={"fold": "synthetic"},
    )


def _module_b() -> ConditionalArbitrator:
    rows = []
    probabilities = ((.7, .2, .1), (.2, .65, .15), (.1, .2, .7))
    for index in range(18):
        post = dict(zip(LABELS, probabilities[index % 3], strict=True))
        scores = {
            "HC": 4 if index % 3 == 0 else 0,
            "MCI": 4 if index % 3 == 1 else 1,
            "AD": 4 if index % 3 == 2 else 0,
        }
        rows.append({
            "module_a_pre_replay": post,
            "module_a_post_replay": post,
            "agent_ordinal_scores": scores,
            "agent_scores_validated": True,
            "eligible": True,
            "revision_type": "none",
            "action_type": "review",
            "agreement": True,
            "evidence_coverage": .95,
            "evidence_reliability": .95,
            "confound_burden": .02,
            "route": "picture_description",
            "language": "en",
            "ood": .0,
            "incremental_evidence_declared": True,
            "evidence_consumed_by_module_a": False,
            "incremental_evidence_ids": [f"metric:incremental-{index}"],
            "consumed_evidence_ids": [],
            "route_supported": True,
            "replay_performed": False,
            "cross_fit_fold": index % 3,
            "true_label": LABELS[index % 3],
        })
    return ConditionalArbitrator(LABELS, ridge_alpha=.5, max_logit_correction=.15).fit(rows)


def _executor() -> ConditionalAuthorityExecutor:
    return ConditionalAuthorityExecutor(
        states_config=STATES,
        module_a=_expert(),
        module_b=_module_b(),
        module_a_state_feature_whitelist=("state_S01",),
        model_versions={"module_a": "synthetic-a", "module_b": "synthetic-b"},
        skill_versions={"ad_evidence_skill": "synthetic-skill"},
        tool_versions={"conditional_authority": "synthetic-tool"},
    )


def _evidence(*, incremental: bool = False) -> tuple[MetricEvidenceV2, ...]:
    reference = ReferenceMetadata(median=0.0, scale=1.0, sample_size=40)
    common = {
        "subject_id": "case-1",
        "session_id": "session-1",
        "state_id": "S01",
        "task_id": "cookie",
        "direction": 1,
        "reference": reference,
        "permissions": EvidencePermissions(inference=True, report=True),
        "provenance": EvidenceProvenance(
            source_asset_id="audio-1", source_segment_ids=("segment-1",), method_version="synthetic"
        ),
    }
    result = (
        MetricEvidenceV2(evidence_id="metric:pause-a", metric_id="pause_a", value=1.0, **common),
        MetricEvidenceV2(evidence_id="metric:pause-b", metric_id="pause_b", value=4.0, **common),
    )
    if not incremental:
        return result
    return result + (
        MetricEvidenceV2(
            evidence_id="metric:incremental", metric_id="agent_only", value=1.0,
            incremental_for_agent=True, **common,
        ),
    )


def _require_bound_state_cards(executor: ConditionalAuthorityExecutor, evidence: tuple[MetricEvidenceV2, ...]) -> None:
    """Do not emulate missing bottom-layer contracts in this orchestrator test."""

    replay = replay_evidence(evidence[:2], None, states_config=STATES, module_a=executor.module_a)
    required = {"supporting_evidence_ids"}
    revision_columns = {"revision_hash", "state_revision_hash"}
    if not required.issubset(set(replay.state_graph.cards.columns)) or not revision_columns.intersection(replay.state_graph.cards.columns):
        pytest.skip("StateGraphV2 revision-bound evidence-ID contract is pending merge.")


def _prepared_decision(
    executor: ConditionalAuthorityExecutor,
    evidence: tuple[MetricEvidenceV2, ...],
    *,
    revision: EvidenceRevision | None = None,
    incremental_ids: tuple[str, ...] = (),
    stale_packet: bool = False,
) -> AgentAuthorityDecision:
    _require_bound_state_cards(executor, evidence)
    supervised = tuple(item for item in evidence if not item.incremental_for_agent)
    replay_kwargs = {
        "dataset_id": CASE["dataset_id"], "label": "unknown", "split": "inference",
    }
    pre = replay_evidence(supervised, None, states_config=STATES, module_a=executor.module_a, **replay_kwargs)
    pre_evidence_hash = hash_artifact(executor._evidence_artifact("case-1", pre.revised_evidence))
    pre_cards = executor._state_cards(case_id="case-1", replay=pre, revision_hash=pre.audit.revision_hash)
    pre_state_hash = hash_artifact(executor._state_artifact("case-1", pre, pre_cards))
    pre_packet_hash = hash_artifact(
        executor._packet_artifact("case-1", pre.packet, evidence_lock_hash=pre_evidence_hash, state_lock_hash=pre_state_hash)
    )
    post = pre if revision is None else replay_evidence(
        supervised, revision, states_config=STATES, module_a=executor.module_a, **replay_kwargs
    )
    cards = executor._state_cards(case_id="case-1", replay=post, revision_hash=post.audit.revision_hash)
    card = next(item for item in cards if item["available"] and item["report_permission"])
    evidence_id = card["supporting_evidence_ids"][0]
    trace = ({
        "claim_id": "claim-1",
        "claim": "Synthetic pause burden evidence.",
        "state_card_id": card["state_card_id"],
        "state_revision_hash": post.audit.revision_hash,
        "metric_evidence_ids": [evidence_id],
        "task_id": "cookie",
        "segment_ids": ["segment-1"],
        "source_asset_id": "audio-1",
    },)
    return AgentAuthorityDecision(
        case_id="case-1",
        reviewed_packet_hash="0" * 64 if stale_packet else pre_packet_hash,
        reviewed_evidence_hash=pre.audit.evidence_hash,
        reviewed_state_graph_hash=pre_state_hash,
        advisor_packet_hash=pre_packet_hash,
        advisor_current=True,
        ordinal_scores={"HC": 0, "MCI": 2, "AD": 4},
        revision=revision,
        action_type="review",
        incremental_evidence_ids=incremental_ids,
        report_trace=trace,
    )


def _validation(decision: AgentAuthorityDecision, *, incremental_ids: tuple[str, ...] = ()) -> AuthorityValidation:
    return AuthorityValidation(
        case_id="case-1",
        agent_decision_hash=decision.decision_hash,
        approved=True,
        incremental_evidence_ids=incremental_ids,
        evidence_coverage=.95,
        evidence_reliability=.95,
        confound_burden=.02,
        ood=.0,
        agreement=True,
        route_supported=True,
    )


def _segments() -> tuple[dict[str, str], ...]:
    return ({
        "case_id": "case-1", "segment_id": "segment-1", "task_id": "cookie", "audio_asset_id": "audio-1",
    },)


def test_explicit_module_a_state_feature_whitelist_is_required() -> None:
    with pytest.raises(ConditionalAuthorityError, match="whitelist"):
        ConditionalAuthorityExecutor(states_config=STATES, module_a=_expert(), module_b=_module_b(), module_a_state_feature_whitelist=())


def test_dataset_specific_binary_target_route_is_supported() -> None:
    labels = ("HC", "AD")
    frame = pd.DataFrame({"state_S01": [-2.0, -1.0, 1.0, 2.0]})
    module_a = TaskConditionedStatisticalExpert(labels, require_explicit_feature_whitelist=True).fit(
        frame,
        ["HC", "HC", "AD", "AD"],
        feature_columns=["state_S01"],
    )
    rows = []
    for index in range(8):
        probability = {"HC": .8, "AD": .2} if index % 2 == 0 else {"HC": .2, "AD": .8}
        rows.append({
            "module_a_pre_replay": probability,
            "module_a_post_replay": probability,
            "agent_ordinal_scores": {"HC": 4 - (index % 2) * 4, "AD": (index % 2) * 4},
            "agent_scores_validated": True,
            "eligible": True,
            "revision_type": "none",
            "action_type": "review",
            "agreement": True,
            "evidence_coverage": .9,
            "evidence_reliability": .9,
            "confound_burden": .0,
            "route": "picture_description",
            "language": "en",
            "ood": .0,
            "incremental_evidence_declared": True,
            "evidence_consumed_by_module_a": False,
            "incremental_evidence_ids": [f"metric:incremental-{index}"],
            "consumed_evidence_ids": [],
            "route_supported": True,
            "replay_performed": False,
            "cross_fit_fold": index % 2,
            "true_label": labels[index % 2],
        })
    module_b = ConditionalArbitrator(labels).fit(rows)
    executor = ConditionalAuthorityExecutor(
        states_config=STATES,
        module_a=module_a,
        module_b=module_b,
        module_a_state_feature_whitelist=("state_S01",),
        target_route_config={
            "diagnosis": {"endpoint": "cross_sectional_diagnosis", "labels": labels},
        },
    )
    assert executor.target_route_config["diagnosis"]["labels"] == labels


def test_no_agent_revision_keeps_module_a_and_reports_only_after_lock() -> None:
    executor = _executor()
    evidence = _evidence()
    decision = _prepared_decision(executor, evidence)
    result = executor.execute(
        case_metadata=CASE, evidence=evidence, agent_decision=decision,
        validator=_validation(decision), segments=_segments(), lock_id="lock-no-revision",
    )
    assert result.pre_replay.packet.to_json() == result.post_replay.packet.to_json()
    assert result.module_b.probabilities == (result.post_replay.packet.calibrated_probabilities or result.post_replay.packet.raw_probabilities)
    assert not result.module_b.additive_correction_applied
    report = executor.render_locked_report(result)
    assert report["locked"] is True
    assert report["decision_lock_hash"] == result.decision_lock.decision_hash


def test_consumed_revision_replays_module_a_without_module_b_double_vote() -> None:
    executor = _executor()
    evidence = _evidence()
    baseline = replay_evidence(evidence, None, states_config=STATES, module_a=executor.module_a)
    revision = EvidenceRevision(
        evidence_id="metric:pause-a", action="downweight", expected_evidence_hash=baseline.audit.evidence_hash,
        reliability_multiplier=.25,
    )
    decision = _prepared_decision(executor, evidence, revision=revision)
    result = executor.execute(
        case_metadata=CASE, evidence=evidence, agent_decision=decision,
        validator=_validation(decision), segments=_segments(), lock_id="lock-revision",
    )
    assert result.pre_replay.packet.raw_probabilities != result.post_replay.packet.raw_probabilities
    assert result.module_b.probabilities == (result.post_replay.packet.calibrated_probabilities or result.post_replay.packet.raw_probabilities)
    assert not result.module_b.additive_correction_applied


def test_only_validated_incremental_evidence_can_apply_bounded_module_b_correction() -> None:
    executor = _executor()
    evidence = _evidence(incremental=True)
    decision = _prepared_decision(executor, evidence, incremental_ids=("metric:incremental",))
    result = executor.execute(
        case_metadata=CASE, evidence=evidence, agent_decision=decision,
        validator=_validation(decision, incremental_ids=("metric:incremental",)), segments=_segments(),
    )
    assert result.module_b.additive_correction_applied
    assert max(abs(value) for value in result.module_b.additive_logit_correction.values()) <= executor.module_b.max_logit_correction
    assert result.module_b.incremental_evidence_ids == ("metric:incremental",)
    assert result.module_b.consumed_evidence_ids == ()


def test_stale_packet_and_validator_fail_closed() -> None:
    executor = _executor()
    evidence = _evidence()
    stale = _prepared_decision(executor, evidence, stale_packet=True)
    with pytest.raises(StaleAuthorityError, match="stale Module A packet"):
        executor.execute(case_metadata=CASE, evidence=evidence, agent_decision=stale, validator=_validation(stale), segments=_segments())

    current = _prepared_decision(executor, evidence)
    stale_validator = AuthorityValidation(
        case_id="case-1", agent_decision_hash="f" * 64, approved=True,
        evidence_coverage=.95, evidence_reliability=.95, confound_burden=.02, ood=.0,
        agreement=True, route_supported=True,
    )
    with pytest.raises(StaleAuthorityError, match="Validator result is stale"):
        executor.execute(case_metadata=CASE, evidence=evidence, agent_decision=current, validator=stale_validator, segments=_segments())

    stale_advisor = replace(current, advisor_packet_hash="0" * 64)
    with pytest.raises(StaleAuthorityError, match="stale Module A packet"):
        executor.execute(case_metadata=CASE, evidence=evidence, agent_decision=stale_advisor, validator=_validation(stale_advisor), segments=_segments())

    stale_revision = EvidenceRevision(
        evidence_id="metric:pause-a", action="invalidate", expected_evidence_hash="f" * 64,
    )
    with pytest.raises(StaleAuthorityError, match="stale evidence snapshot"):
        executor.execute(
            case_metadata=CASE, evidence=evidence, agent_decision=replace(current, revision=stale_revision),
            validator=_validation(replace(current, revision=stale_revision)), segments=_segments(),
        )
