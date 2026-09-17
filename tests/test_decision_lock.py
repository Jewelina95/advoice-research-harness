from __future__ import annotations

from dataclasses import FrozenInstanceError, replace
import json

import pandas as pd
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
from advoice.evidence_replay import EvidenceRevision, ReplayAudit
from advoice.module_a import ExplanationPacket
from advoice.module_b import ModuleBPrediction
from advoice.state_graph import StateGraphV2
from advoice.utils import hash_values


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


def _packet(snapshot_hash: str, *, label: str = "HC") -> ExplanationPacket:
    probabilities = {"HC": 0.7, "MCI": 0.2, "AD": 0.1}
    return ExplanationPacket(
        schema_version="module-a-v1",
        module_version="module-a-test",
        class_order=("HC", "MCI", "AD"),
        predicted_label=label,
        raw_probabilities=probabilities,
        calibrated_probabilities=probabilities,
        calibration_status="calibrated",
        logits={"HC": 1.0, "MCI": 0.0, "AD": -1.0},
        intercepts={"HC": 0.1, "MCI": 0.0, "AD": -0.1},
        feature_contributions={name: {"state_S01": 0.1} for name in ("HC", "MCI", "AD")},
        branch_contributions={name: {"cognition": 0.1} for name in ("HC", "MCI", "AD")},
        consumed_evidence_ids=("metric:e1",),
        uncertainty={"entropy": 0.8, "margin": 0.5},
        fold_disagreement=None,
        ood={"is_ood": False, "score": 0.1},
        applicability_status="applicable",
        hashes={"evidence_snapshot_hash": snapshot_hash, "model_hash": "m" * 64},
    )


def _typed_inputs() -> dict:
    values = _inputs()
    case_id = values["case_id"]
    snapshot_hash = hash_artifact(values["evidence_snapshot"])
    revision = EvidenceRevision(
        evidence_id="metric:e1",
        action="downweight",
        expected_evidence_hash="e" * 64,
        reliability_multiplier=0.75,
        rationale="recording noise",
        cited_evidence_ids=("metric:e1",),
    )
    lock_revision_hash = hash_artifact(revision)
    cards = pd.DataFrame([{
        "case_id": case_id,
        "state_card_id": "S01",
        "revision_hash": lock_revision_hash,
        "task_id": "task_cookie",
        "supporting_evidence_ids": ["metric:e1"],
        "segment_ids": ["segment_0000000001"],
        "state_z": float("nan"),
    }])
    graph = StateGraphV2(
        evidence_hash="e" * 64,
        state_hash="s" * 64,
        cards=cards,
        wide=pd.DataFrame([{"subject_id": case_id, "state_S01": float("nan")}]),
    )
    pre = _packet(hash_artifact({"parent": True}))
    post = _packet(snapshot_hash)
    audit_payload = {
        "schema_version": "replay-v1",
        "revision_hash": revision.revision_hash,
        "parent_evidence_hash": "p" * 64,
        "evidence_hash": "e" * 64,
        "state_hash": graph.state_hash,
        "model_hash": "m" * 64,
        "packet_hash": hash_values([post.to_json()]),
    }
    audit = ReplayAudit(**audit_payload, audit_hash=hash_values([audit_payload]))
    module_b = ModuleBPrediction(
        class_order=("HC", "MCI", "AD"),
        probabilities={"HC": 0.7, "MCI": 0.2, "AD": 0.1},
        module_a_post_replay={"HC": 0.7, "MCI": 0.2, "AD": 0.1},
        additive_logit_correction={"HC": 0.0, "MCI": 0.0, "AD": 0.0},
        additive_correction_applied=False,
        fallback="module_a",
        authority=0.0,
        reason="no incremental evidence",
        consumed_evidence_ids=("metric:e1",),
    )
    values.update({
        "revision": revision,
        "state_graph": graph,
        "module_a_pre_replay": pre,
        "module_a_post_replay": post,
        "module_b_output": module_b,
        "replay_audit": audit,
    })
    values["report_trace"][0]["state_revision_hash"] = lock_revision_hash
    return values


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


def test_report_trace_reads_json_encoded_state_card_id_lists() -> None:
    values = _inputs()
    values["state_graph"]["state_cards"][0]["supporting_evidence_ids"] = '["metric:e1"]'
    values["state_graph"]["state_cards"][0]["task_ids"] = '["task_cookie"]'
    values["metric_evidence"][0]["segment_ids"] = '["segment_0000000001"]'
    lock = create_decision_lock(**values)
    assert lock.locked is True


def test_canonical_hash_accepts_real_pipeline_objects_and_normalizes_nonfinite_values() -> None:
    values = _typed_inputs()
    first = create_decision_lock(**values)
    second = create_decision_lock(**_typed_inputs())

    assert first.to_json() == second.to_json()
    assert first.state_graph_hash == hash_artifact(values["state_graph"])
    assert first.module_a_post_replay_hash == hash_artifact(values["module_a_post_replay"])
    assert first.module_b_output_hash == hash_artifact(values["module_b_output"])
    assert first.replay_audit_hash == hash_artifact(values["replay_audit"])
    assert '"state_z":null' in canonical_json(values["state_graph"])
    assert canonical_json({"positive": float("inf"), "negative": float("-inf"), "nan": float("nan")}) == (
        '{"nan":null,"negative":null,"positive":null}'
    )
    assert canonical_json({"pandas_missing": pd.NA, "not_a_time": pd.NaT}) == (
        '{"not_a_time":null,"pandas_missing":null}'
    )


def test_unknown_object_cannot_fall_back_to_repr_hash() -> None:
    class UnversionedArtifact:
        pass

    with pytest.raises(TypeError, match="Unsupported decision-lock artifact type"):
        hash_artifact(UnversionedArtifact())


def test_state_card_without_revision_binding_fails_closed() -> None:
    values = _inputs()
    del values["state_graph"]["state_cards"][0]["revision_hash"]
    with pytest.raises(HashMismatchError, match="no explicit evidence revision binding"):
        create_decision_lock(**values)


def test_state_card_without_evidence_association_fails_closed() -> None:
    values = _inputs()
    del values["state_graph"]["state_cards"][0]["supporting_evidence_ids"]
    with pytest.raises(EvidenceTraceError, match="no explicit state-to-evidence association"):
        create_decision_lock(**values)


def test_real_replay_audit_rejects_stale_state_revision_and_packet_bindings() -> None:
    values = _typed_inputs()
    audit = values["replay_audit"]

    values["replay_audit"] = replace(audit, state_hash="x" * 64)
    with pytest.raises(HashMismatchError, match="different StateGraphV2 state"):
        create_decision_lock(**values)

    values = _typed_inputs()
    values["replay_audit"] = replace(values["replay_audit"], revision_hash="x" * 64)
    with pytest.raises(HashMismatchError, match="different evidence revision"):
        create_decision_lock(**values)

    values = _typed_inputs()
    values["replay_audit"] = replace(values["replay_audit"], packet_hash="x" * 64)
    with pytest.raises(HashMismatchError, match="different post-replay Module A packet"):
        create_decision_lock(**values)
