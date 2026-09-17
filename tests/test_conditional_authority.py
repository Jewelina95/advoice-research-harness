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
from advoice.evidence_revision_batch import EvidenceRevisionBatch
from advoice.module_a import TaskConditionedStatisticalExpert
from advoice.module_b import ConditionalArbitrator, compute_fit_subject_hash


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
        fold = index % 3
        fit_subject_ids = [f"module-b-fold-{fold}-train-{offset}" for offset in range(6)]
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
            "subject_id": f"module-b-target-{index}",
            "fold": fold,
            "cross_fit_fold": fold,
            "fit_subject_ids": fit_subject_ids,
            "fit_subject_hash": compute_fit_subject_hash(fit_subject_ids),
            "reference_hash": f"reference-fold-{fold}",
            "module_a_hash": f"module-a-fold-{fold}",
            "agent_version": "agent-v1",
            "validator_version": "validator-v1",
            "selection_independent": True,
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
        MetricEvidenceV2(
            evidence_id="metric:pause-a", metric_id="pause_a", value=1.0,
            consumed_by_supervised=True, **common,
        ),
        MetricEvidenceV2(
            evidence_id="metric:pause-b", metric_id="pause_b", value=4.0,
            consumed_by_supervised=True, **common,
        ),
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
    revision_batch: EvidenceRevisionBatch | None = None,
    incremental_ids: tuple[str, ...] = (),
    stale_packet: bool = False,
    trace_post_revision: bool = True,
) -> AgentAuthorityDecision:
    _require_bound_state_cards(executor, evidence)
    supervised = tuple(item for item in evidence if item.consumed_by_supervised)
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
    proposed_revision = revision if revision is not None else revision_batch
    post = pre if proposed_revision is None or not trace_post_revision else replay_evidence(
        supervised, proposed_revision, states_config=STATES, module_a=executor.module_a, **replay_kwargs
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
        revision_batch=revision_batch,
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


def _rejected_validation(decision: AgentAuthorityDecision) -> AuthorityValidation:
    return replace(_validation(decision), approved=False)


def _revision_batch(evidence: tuple[MetricEvidenceV2, ...]) -> EvidenceRevisionBatch:
    supervised = tuple(item for item in evidence if item.consumed_by_supervised)
    expected_hash = replay_evidence(
        supervised, None, states_config=STATES, module_a=_expert(),
    ).audit.evidence_hash
    revisions = tuple(
        EvidenceRevision(
            evidence_id=evidence_id,
            action="downweight",
            expected_evidence_hash=expected_hash,
            reliability_multiplier=multiplier,
        )
        for evidence_id, multiplier in (
            ("metric:pause-a", .25),
            ("metric:pause-b", .75),
        )
    )
    return EvidenceRevisionBatch(
        case_id="case-1",
        state_id="S01",
        action="downweight",
        expected_evidence_hash=expected_hash,
        revisions=revisions,
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
        fold = index % 2
        fit_subject_ids = [f"binary-fold-{fold}-train-{offset}" for offset in range(4)]
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
            "subject_id": f"binary-target-{index}",
            "fold": fold,
            "cross_fit_fold": fold,
            "fit_subject_ids": fit_subject_ids,
            "fit_subject_hash": compute_fit_subject_hash(fit_subject_ids),
            "reference_hash": f"binary-reference-fold-{fold}",
            "module_a_hash": f"binary-module-a-fold-{fold}",
            "agent_version": "agent-v1",
            "validator_version": "validator-v1",
            "selection_independent": True,
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
    prepared = executor.prepare_case(case_metadata=CASE, evidence=evidence)
    decision = _prepared_decision(executor, evidence)
    result = executor.finalize_case(
        prepared=prepared, agent_decision=decision,
        validator=_validation(decision), segments=_segments(), lock_id="lock-no-revision",
    )
    assert result.pre_replay is prepared.pre_replay
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


def test_consumed_revision_batch_replays_all_metrics_through_module_a_once(monkeypatch) -> None:
    executor = _executor()
    evidence = _evidence()
    batch = _revision_batch(evidence)
    decision = _prepared_decision(executor, evidence, revision_batch=batch)
    prepared = executor.prepare_case(case_metadata=CASE, evidence=evidence)
    calls = 0
    explain_case = executor.module_a.explain_case

    def counted_explain_case(*args, **kwargs):
        nonlocal calls
        calls += 1
        return explain_case(*args, **kwargs)

    monkeypatch.setattr(executor.module_a, "explain_case", counted_explain_case)
    result = executor.finalize_case(
        prepared=prepared,
        agent_decision=decision,
        validator=_validation(decision),
        segments=_segments(),
        lock_id="lock-revision-batch",
    )

    assert calls == 1
    assert result.pre_replay.packet.raw_probabilities != result.post_replay.packet.raw_probabilities
    assert result.post_replay.audit.revision_hash not in {
        item.revision_hash for item in batch.revisions
    }
    assert result.module_b.consumed_evidence_ids == ("metric:pause-a", "metric:pause-b")
    assert result.module_b.incremental_evidence_ids == ()
    assert not result.module_b.additive_correction_applied


def test_agent_decision_batch_mapping_round_trip_and_legacy_revision_unchanged() -> None:
    executor = _executor()
    evidence = _evidence()
    batch_decision = _prepared_decision(executor, evidence, revision_batch=_revision_batch(evidence))
    restored_batch = AgentAuthorityDecision.from_mapping(batch_decision.to_dict())

    assert restored_batch == batch_decision
    assert restored_batch.revision is None
    assert restored_batch.revision_batch == batch_decision.revision_batch

    baseline = replay_evidence(evidence, None, states_config=STATES, module_a=executor.module_a)
    legacy_revision = EvidenceRevision(
        evidence_id="metric:pause-a",
        action="invalidate",
        expected_evidence_hash=baseline.audit.evidence_hash,
    )
    legacy = _prepared_decision(
        executor, evidence, revision=legacy_revision, trace_post_revision=False,
    )
    restored_legacy = AgentAuthorityDecision.from_mapping(legacy.to_dict())
    assert restored_legacy.revision == legacy_revision
    assert restored_legacy.revision_batch is None


def test_agent_decision_rejects_both_revision_forms_and_unknown_batch_fields() -> None:
    executor = _executor()
    evidence = _evidence()
    batch = _revision_batch(evidence)
    legacy = batch.revisions[0]
    decision = _prepared_decision(executor, evidence)

    with pytest.raises(ConditionalAuthorityError, match="never both"):
        replace(decision, revision=legacy, revision_batch=batch)

    mapping = decision.to_dict()
    mapping["revision_batch"] = {**batch.to_dict(), "protected": "forbidden"}
    with pytest.raises(ConditionalAuthorityError, match="unknown"):
        AgentAuthorityDecision.from_mapping(mapping)


def test_rejected_validation_never_replays_revision_batch(monkeypatch) -> None:
    executor = _executor()
    evidence = _evidence()
    batch = _revision_batch(evidence)
    decision = _prepared_decision(
        executor, evidence, revision_batch=batch, trace_post_revision=False,
    )
    prepared = executor.prepare_case(case_metadata=CASE, evidence=evidence)

    def forbidden_replay(*args, **kwargs):
        raise AssertionError("rejected batch reached replay")

    monkeypatch.setattr("advoice.conditional_authority.replay_evidence", forbidden_replay)
    result = executor.finalize_case(
        prepared=prepared,
        agent_decision=decision,
        validator=_rejected_validation(decision),
        segments=_segments(),
        lock_id="lock-rejected-batch",
    )

    assert result.post_replay is prepared.pre_replay


def test_prepare_case_freezes_every_hash_required_by_agent_decision() -> None:
    prepared = _executor().prepare_case(case_metadata=CASE, evidence=_evidence())

    assert prepared.case_id == "case-1"
    assert len(prepared.reviewed_packet_hash) == 64
    assert len(prepared.reviewed_evidence_hash) == 64
    assert len(prepared.reviewed_state_graph_hash) == 64
    assert prepared.reviewed_packet_hash == prepared.advisor_packet_hash


def test_rejected_revision_cannot_replay_or_change_module_a() -> None:
    executor = _executor()
    evidence = _evidence()
    baseline = replay_evidence(evidence, None, states_config=STATES, module_a=executor.module_a)
    revision = EvidenceRevision(
        evidence_id="metric:pause-a", action="downweight", expected_evidence_hash=baseline.audit.evidence_hash,
        reliability_multiplier=.25,
    )
    decision = _prepared_decision(executor, evidence, revision=revision, trace_post_revision=False)

    result = executor.execute(
        case_metadata=CASE, evidence=evidence, agent_decision=decision,
        validator=_rejected_validation(decision), segments=_segments(), lock_id="lock-rejected-revision",
    )

    assert result.pre_replay.packet.to_json() == result.post_replay.packet.to_json()
    assert result.pre_replay.audit.evidence_hash == result.post_replay.audit.evidence_hash
    assert result.module_b.consumed_evidence_ids == result.pre_replay.packet.consumed_evidence_ids


def test_unconsumed_evidence_cannot_enter_module_a_or_consumption_audit() -> None:
    executor = _executor()
    evidence = list(_evidence())
    evidence[1] = replace(evidence[1], consumed_by_supervised=False)
    decision = _prepared_decision(executor, tuple(evidence))

    result = executor.execute(
        case_metadata=CASE, evidence=tuple(evidence), agent_decision=decision,
        validator=_validation(decision), segments=_segments(), lock_id="lock-unconsumed-evidence",
    )

    assert result.pre_replay.packet.consumed_evidence_ids == ("metric:pause-a",)
    assert "metric:pause-b" not in result.module_b.consumed_evidence_ids


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
    assert result.module_b.consumed_evidence_ids == ("metric:pause-a", "metric:pause-b")


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
