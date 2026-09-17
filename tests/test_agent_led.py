from copy import deepcopy

import pytest

from advoice.agent_led import EvidenceSession, run_agent_session, evidence_snapshot, response_schema
from advoice.utils import hash_values


def workspace():
    return {
        "case_id": "synthetic", "label": "AD", "base_probabilities": {"HC": .99, "AD": .01},
        "corrected_probabilities": {"HC": .98, "AD": .02},
        "case_context": {"languages": ["English"], "label": "AD"},
        "state_observations": [
            {"evidence_id": "state:S01", "state_id": "S01", "task_scope": "overall",
             "confidence": .8, "missing_fraction": 0, "state_z": 2,
             "metric_evidence_ids": ["metric:m1"], "report_permission": True,
             "supporting_metrics": [{"metric_id": "m1", "value": 2.}],
             "counter_evidence": [], "evidence_segments": []},
            {"evidence_id": "state:S07", "state_id": "S07", "task_scope": "overall",
             "confidence": .9, "missing_fraction": 0, "state_z": 1.5,
             "metric_evidence_ids": [], "report_permission": True,
             "evidence_segments": []},
        ],
        "selected_supporting_evidence": [{"evidence_id": "metric:m1", "state_id": "S01", "task_scope": "overall"}],
        "quality_observations": [{"evidence_id": "qc:role", "value": .8}],
        "selected_counterevidence": [],
        "evidence_registry": [
            {"evidence_id": "state:S01", "evidence_type": "state"},
            {"evidence_id": "state:S07", "evidence_type": "state"},
            {"evidence_id": "metric:m1", "evidence_type": "metric"},
            {"evidence_id": "qc:role", "evidence_type": "quality"},
        ],
    }


def session():
    source = workspace()
    source["advisor_provenance"] = {"evidence_hash": hash_values([evidence_snapshot(source)]),
                                    "artifacts": {"module_a": "fixture-a-v1", "module_b": "fixture-b-v1"}}
    return EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")


def reply(s, action, **kwargs):
    return {"action": action, "revision": s.revision, "target_id": "", "state_action": "none",
            "evidence_ids": [], "counterevidence_ids": [], "predicted_label": "undetermined",
            "scores": {"HC": 0, "AD": 0}, "rationale": "Evidence-linked research judgment.", "limitations": [], **kwargs}


def inspect(s, target="state:S01"):
    s.step(reply(s, "inspect_quality"))
    s.step(reply(s, "inspect_counterevidence"))
    s.step(reply(s, "inspect_state", target_id=target))
    return s.step(reply(s, "record_hypothesis", evidence_ids=[target], rationale="Task-specific evidence is present."))


def test_agent_final_not_prior_or_scalar_correction():
    s = session()
    inspect(s)
    advisor = s.step(reply(s, "consult_models"))
    assert advisor["outputs"]["module_a"]["HC"] == .99
    s.step(reply(s, "finalize", predicted_label="AD", scores={"HC": 0, "AD": 3}, evidence_ids=["state:S01"]))
    r = s.finish()
    assert r["predicted_label"] == "AD"
    assert r["prediction_source"] == "agent"
    assert r["probabilities"] is None
    assert r["supervised_modules_consulted"] is True
    assert r["supervised_outputs_available"] is True
    assert not r["clinical_release"]


def test_blinding_recursive_and_advisor_requires_hypothesis():
    s = session()
    assert "label" not in s.workspace["case_context"]
    assert "base_probabilities" not in s.workspace
    assert s.step(reply(s, "consult_models"))["status"] == "rejected"
    assert not s.hypothesis_recorded


def test_revision_invalidates_advisors_and_requires_replay_before_judgment():
    s = session()
    inspect(s)
    before = s.revision
    out = s.step(reply(s, "revise_state", target_id="state:S01", state_action="invalidate",
                       evidence_ids=["qc:role"], rationale="Timing source is uncertain."))
    assert out["status"] == "revised", out
    assert s.revision != before
    assert not s.hypothesis_recorded
    assert "state:S01" not in s._objects()
    assert "metric:m1" not in s._objects()
    assert s.step(reply(s, "inspect_quality", revision=before))["status"] == "rejected"
    inspect(s, "state:S07")
    assert s.step(reply(s, "consult_models"))["status"] == "unavailable"
    final = reply(s, "finalize", evidence_ids=["state:S07"], predicted_label="HC", scores={"HC": 3, "AD": 0})
    blocked = s.step(final)
    assert blocked["status"] == "rejected"
    assert "replay" in blocked["reason"].lower()
    assert s.finish()["state_revisions"] == 1


@pytest.mark.parametrize("evidence", [["state:invented"], ["qc:role"], ["state:S07"]])
def test_no_unknown_quality_or_uninspected_support(evidence):
    s = session()
    inspect(s)
    out = s.step(reply(s, "finalize", evidence_ids=evidence, predicted_label="AD", scores={"HC": 0, "AD": 4}))
    assert out["status"] == "rejected"
    assert s.result is None


def test_quality_is_not_fabricated_clinical_assessment():
    s = session()
    q = s.step(reply(s, "inspect_quality"))
    assert q["clinical_confound_assessment"] == "unknown"
    assert "correction_gate" not in q


def test_model_change_invalidates_calibration_identity():
    s = session()
    other = EvidenceSession(workspace(), s.labels, model_id="next-model", skill_hash="fixture")
    assert s.fingerprint != other.fingerprint
    modified = workspace()
    modified["state_observations"][0]["state_z"] = 4
    assert s.revision != EvidenceSession(modified, s.labels, model_id="test", skill_hash="fixture").revision


def test_provider_failure_and_budget_never_fallback_to_prior():
    def broken(_):
        raise RuntimeError("provider failed")
    result = run_agent_session(workspace(), ["HC", "AD"], broken, model_id="test", skill_hash="fixture")
    assert result["status"] == "provider_error"
    assert result["provider_error_type"] == "RuntimeError"
    assert result["provider_error_message"] == "provider failed"
    assert "RuntimeError: provider failed" in result["provider_error_traceback"]
    assert result["predicted_label"] is None
    result = run_agent_session(workspace(), ["HC", "AD"], lambda _: {}, model_id="test", skill_hash="fixture", max_steps=2)
    assert result["status"] == "budget_exhausted"
    assert len(result["trace"]) == 2
    assert result["probabilities"] is None


def test_input_workspace_not_mutated():
    source = workspace()
    before = deepcopy(source)
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    inspect(s)
    s.step(reply(s, "revise_state", target_id="state:S01", state_action="downweight", evidence_ids=["qc:role"], rationale="Review timing."))
    assert source == before


def test_withdraw_shared_metric_invalidates_dependent_aggregate():
    source = workspace()
    source["state_observations"][1]["metric_evidence_ids"] = ["metric:m1"]
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    inspect(s)
    out = s.step(reply(s, "revise_state", target_id="state:S01", state_action="invalidate",
                       evidence_ids=["qc:role"], rationale="Shared timing measurement is unreliable."))
    assert out["status"] == "revised"
    assert out["dependent_states_withdrawn"] == ["state:S07"]
    assert "state:S07" not in s._objects()


def test_evidence_hash_includes_values_not_only_dictionary_keys():
    from advoice.evidence_review import apply_reviewed_snapshot
    a = workspace()
    b = deepcopy(a)
    b["state_observations"][1]["state_z"] = 9
    candidate = {"state_updates": [{"state_id": "S01", "action": "downweight"}]}
    audit = {"valid": True}
    left = apply_reviewed_snapshot(a, candidate, audit)["evidence_revision"]
    right = apply_reviewed_snapshot(b, candidate, audit)["evidence_revision"]
    assert left["parent_hash"] != right["parent_hash"]
    assert left["snapshot_hash"] != right["snapshot_hash"]


def test_imported_reviewed_snapshot_does_not_reactivate_stale_advisors():
    from advoice.evidence_review import apply_reviewed_snapshot
    source = apply_reviewed_snapshot(workspace(), {"state_updates": [{"state_id": "S01", "action": "downweight"}]}, {"valid": True})
    source["advisor_provenance"] = {"evidence_hash": hash_values([evidence_snapshot(source)]),
                                    "artifacts": {"module_a": "fixture-a-v1"}}
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    inspect(s)
    assert s.step(reply(s, "consult_models"))["status"] == "unavailable"


def test_unbound_advisors_are_not_presented_as_current():
    s = EvidenceSession(workspace(), ["HC", "AD"], model_id="test", skill_hash="fixture")
    inspect(s)
    assert s.step(reply(s, "consult_models"))["status"] == "unavailable"


def test_finalization_rejects_stale_bound_advisors():
    source = workspace()
    source["advisor_provenance"] = {
        "evidence_hash": "stale-hash", "artifacts": {"module_a": "fixture-a-v1"},
    }
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    inspect(s)
    result = s.step(reply(
        s, "finalize", predicted_label="AD", scores={"HC": 0, "AD": 3}, evidence_ids=["state:S01"],
    ))
    assert result["status"] == "rejected"
    assert "stale" in result["reason"].lower()


def test_nested_counterevidence_is_inspected_and_required():
    source = workspace()
    metric = {"metric_id": "nested", "directional_z": -3}
    source["state_observations"][0]["counter_evidence"] = [metric]
    source["evidence_registry"].append({"evidence_id": "metric:nested", "evidence_type": "metric"})
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    inspect(s)
    assert "metric:nested" in s.observed
    out = s.step(reply(s, "finalize", predicted_label="AD", scores={"HC": 0, "AD": 3}, evidence_ids=["state:S01"]))
    assert out["status"] == "rejected"
    out = s.step(reply(s, "finalize", predicted_label="AD", scores={"HC": 0, "AD": 3}, evidence_ids=["state:S01"], counterevidence_ids=["metric:nested"]))
    assert out["status"] == "decided"


def test_abstention_cannot_cite_uninspected_findings():
    s = session()
    assert s.step(reply(s, "abstain", evidence_ids=["state:S01"]))["status"] == "rejected"
    inspect(s)
    result = s.step(reply(s, "abstain", predicted_label="AD"))
    assert result["status"] == "abstained"
    assert s.finish()["research_most_likely_class"] == "AD"


def test_provider_and_advisor_versions_change_calibration_identity():
    source = workspace()
    left = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture", provider="api")
    right = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture", provider="custom")
    assert left.fingerprint != right.fingerprint
    source["advisor_provenance"] = {"artifacts": {"module_a": "new-artifact"}}
    newer = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture", provider="api")
    assert left.fingerprint != newer.fingerprint


def test_benchmark_mode_forces_classification_but_preserves_uncertainty_fields():
    source = workspace()
    source["case_transcript"] = {
        "text": "The patient gives a short task response.",
        "character_count": 40,
        "whitespace_token_count": 7,
        "truncated": False,
    }
    s = EvidenceSession(
        source, ["HC", "AD"], model_id="test", skill_hash="fixture",
        decision_mode="benchmark_forced_choice",
    )
    assert "abstain" not in s.observation()["available_tools"]
    assert "abstain" not in response_schema(["HC", "AD"], "benchmark_forced_choice")["properties"]["action"]["enum"]
    transcript = s.step(reply(s, "inspect_transcript"))
    assert transcript["status"] == "observed"
    assert transcript["transcript"]["character_count"] == 40
    assert s.step(reply(s, "abstain"))["status"] == "rejected"
    inspect(s)
    result = s.step(reply(
        s, "finalize", predicted_label="HC", scores={"HC": 2, "AD": 1},
        evidence_ids=["state:S01"], limitations=["Evidence is limited; retest is recommended."],
    ))
    assert result["status"] == "decided"
    assert s.finish()["decision_mode"] == "benchmark_forced_choice"


def test_benchmark_requires_bound_advisor_review_after_blind_hypothesis():
    source = workspace()
    source["advisor_provenance"] = {
        "evidence_hash": hash_values([evidence_snapshot(source)]),
        "artifacts": {"module_a": "fixture-a-v1", "module_b": "fixture-b-v1"},
    }
    s = EvidenceSession(
        source, ["HC", "AD"], model_id="test", skill_hash="fixture",
        decision_mode="benchmark_forced_choice",
    )
    inspect(s)
    final = reply(
        s, "finalize", predicted_label="AD", scores={"HC": 0, "AD": 3},
        evidence_ids=["state:S01"],
    )
    rejected = s.step(final)
    assert rejected["status"] == "rejected"
    assert "consult" in rejected["reason"].lower()
    advisor = s.step(reply(s, "consult_models"))
    assert advisor["status"] == "available"
    assert s.step(final)["status"] == "decided"


def test_inspecting_state_marks_returned_child_evidence_as_observed():
    source = workspace()
    source["state_observations"][0]["evidence_segments"] = [
        {"segment_id": "segment:seg1", "report_permission": True},
    ]
    source["evidence_registry"].append(
        {"evidence_id": "segment:seg1", "evidence_type": "segment"},
    )
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    s.step(reply(s, "inspect_state", target_id="state:S01"))
    assert {"state:S01", "metric:m1", "segment:seg1"}.issubset(s.observed)


def test_hypothesis_rejects_uninspected_counterevidence_immediately():
    source = workspace()
    source["selected_counterevidence"] = [
        {"evidence_id": "metric:c1", "state_id": "S02", "report_permission": True},
    ]
    source["evidence_registry"].append(
        {"evidence_id": "metric:c1", "evidence_type": "metric"},
    )
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    s.step(reply(s, "inspect_state", target_id="state:S01"))
    out = s.step(reply(
        s, "record_hypothesis", evidence_ids=["state:S01"],
        counterevidence_ids=["metric:c1"],
    ))
    assert out["status"] == "rejected"
    assert "metric:c1" in out["reason"]


def test_legacy_untyped_state_metric_segment_and_quality_ids_are_canonicalized():
    source = workspace()
    state = source["state_observations"][0]
    state["evidence_id"] = "S01"
    state["metric_evidence_ids"] = ["m1"]
    state["supporting_metrics"][0]["evidence_id"] = "m1"
    state["evidence_segments"] = [{"segment_id": "seg1", "report_permission": True}]
    source["quality_observations"][0]["evidence_id"] = "role"
    source["evidence_registry"] = [
        {"evidence_id": "S01", "evidence_type": "state"},
        {"evidence_id": "S07", "evidence_type": "state"},
        {"evidence_id": "m1", "evidence_type": "metric"},
        {"evidence_id": "seg1", "evidence_type": "segment"},
        {"evidence_id": "role", "evidence_type": "quality"},
    ]
    s = EvidenceSession(source, ["HC", "AD"], model_id="test", skill_hash="fixture")
    assert {"state:S01", "metric:m1", "segment:seg1", "qc:role"}.issubset(s._objects())
    assert s.observation()["states"][0]["evidence_id"] == "state:S01"
    assert s.step(reply(s, "inspect_state", target_id="state:S01"))["status"] == "observed"
