from __future__ import annotations

from dataclasses import FrozenInstanceError
import json

import pytest

from advoice.decision_lock import (
    DecisionLock,
    EvidenceTraceError,
    HashMismatchError,
    UnlockedReportError,
    canonical_json,
    create_decision_lock,
    hash_artifact,
    validate_locked_report,
)


def _inputs() -> dict:
    case_id = "case_000000000001"
    snapshot = {"case_id": case_id, "evidence": ["metric:e1"]}
    revision = {"case_id": case_id, "actions": [{"action": "keep", "state_id": "S01"}]}
    snapshot_hash = hash_artifact(snapshot)
    revision_hash = hash_artifact(revision)
    state_card = {
        "case_id": case_id,
        "state_id": "S01",
        "revision_hash": revision_hash,
        "task_id": "task_cookie",
        "supporting_evidence_ids": ["metric:e1"],
    }
    state_graph = {"case_id": case_id, "state_cards": [state_card]}
    metric = {
        "case_id": case_id,
        "evidence_id": "metric:e1",
        "task_id": "task_cookie",
        "segment_ids": ["segment_0000000001"],
        "observable": True,
        "report_permission": True,
        "evidence_role": "clinical_support",
    }
    segment = {
        "case_id": case_id,
        "segment_id": "segment_0000000001",
        "task_id": "task_cookie",
        "audio_asset_id": "asset_000000000001",
    }
    pre = {"case_id": case_id, "evidence_snapshot_hash": hash_artifact({"parent": True})}
    post = {"case_id": case_id, "evidence_snapshot_hash": snapshot_hash}
    agent = {"case_id": case_id, "decision_id": "decision-1"}
    validator = {"case_id": case_id, "agent_decision_hash": hash_artifact(agent)}
    module_b = {"case_id": case_id, "module_a_post_replay_hash": hash_artifact(post)}
    trace = [{
        "claim_id": "claim-1",
        "claim": "Reduced output efficiency",
        "state_card_id": "S01",
        "state_revision_hash": revision_hash,
        "metric_evidence_ids": ["metric:e1"],
        "task_id": "task_cookie",
        "segment_ids": ["segment_0000000001"],
        "source_asset_id": "asset_000000000001",
    }]
    return {
        "case_id": case_id,
        "evidence_snapshot": snapshot,
        "revision": revision,
        "state_graph": state_graph,
        "module_a_pre_replay": pre,
        "module_a_post_replay": post,
        "agent_decision": agent,
        "validator_result": validator,
        "module_b_output": module_b,
        "report_trace": trace,
        "metric_evidence": [metric],
        "segments": [segment],
        "lock_id": "lock-1",
        "model_versions": {"module_a": "a-1", "module_b": "b-1"},
        "skill_versions": {"ad_evidence_diagnostic": "skill-1"},
        "tool_versions": {"segment_tool": "tool-1"},
    }


def test_lock_binds_all_artifacts_and_serializes_deterministically() -> None:
    first = create_decision_lock(**_inputs())
    second = create_decision_lock(**_inputs())

    assert first.to_json() == second.to_json()
    assert DecisionLock.from_mapping(json.loads(first.to_json())).to_json() == first.to_json()
    assert first.evidence_snapshot_hash == hash_artifact(_inputs()["evidence_snapshot"])
    with pytest.raises(FrozenInstanceError):
        first.case_id = "other"  # type: ignore[misc]
    with pytest.raises(TypeError):
        first.model_versions["module_a"] = "changed"  # type: ignore[index]
    assert canonical_json({"b": 2, "a": 1}) == '{"a":1,"b":2}'


def test_stale_post_replay_packet_is_rejected() -> None:
    values = _inputs()
    values["module_a_post_replay"] = {
        "case_id": values["case_id"],
        "evidence_snapshot_hash": hash_artifact({"different": True}),
    }
    with pytest.raises(HashMismatchError, match="Post-replay"):
        create_decision_lock(**values)


def test_unavailable_or_report_forbidden_metric_cannot_enter_report_trace() -> None:
    values = _inputs()
    values["metric_evidence"][0]["report_permission"] = False
    with pytest.raises(EvidenceTraceError, match="not report-permitted"):
        create_decision_lock(**values)


def test_locked_report_must_match_the_immutable_trace() -> None:
    lock = create_decision_lock(**_inputs())
    report = {
        "locked": True,
        "lock_id": "lock-1",
        "case_id": lock.case_id,
        "report_trace": lock.report_trace.to_dict(),
    }
    validate_locked_report(report, lock)
    with pytest.raises(UnlockedReportError):
        validate_locked_report({**report, "locked": False}, lock)
    with pytest.raises(HashMismatchError):
        validate_locked_report({**report, "case_id": "case_000000000002"}, lock)
