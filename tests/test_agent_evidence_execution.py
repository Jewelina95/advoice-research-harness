from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from advoice import cognitive_agent
from advoice.agent_runtime import case_pseudonym
from advoice.diagnostic_agent import build_case_workspace


def _workspace(confound_tags: tuple[str, ...] = (), confound_assessment=None) -> dict:
    cards = pd.DataFrame([
        {
            "subject_id": "synthetic",
            "state_id": state_id,
            "state_base_id": state_id,
            "branch": "speech_behavior",
            "state_z": 1.0,
            "confidence": 1.0,
            "missing_fraction": 0.0,
            "task_scope": "overall",
            "report_permission": True,
            "supporting_metrics": json.dumps(
                [{"metric_id": "m9", "metric_instance_id": "m9"}]
                if state_id == "S1" else []
            ),
            "counter_evidence": "[]",
            "evidence_segments": "[]",
        }
        for state_id in ["S1", "S2"]
    ])
    metrics = pd.DataFrame([
        {
            "subject_id": "synthetic",
            "metric_id": f"m{index}",
            "metric_instance_id": f"m{index}",
            "state_id": "S1",
            "branch": "speech_behavior",
            "directional_z": 10.0 - index,
            "value": 10.0 - index,
            "reliability": 1.0,
            "task_scope": "overall",
            "evidence_role": "clinical_support",
            "report_permission": True,
            "missing": False,
            "confound_tags": json.dumps(confound_tags),
        }
        for index in range(10)
    ])
    return build_case_workspace(
        subject_id="synthetic",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=cards,
        metric_evidence=metrics,
        class_support={},
        confound_assessment=confound_assessment,
    )


def _candidate(evidence_id: str = "state:S1") -> dict:
    return {
        "case_id": case_pseudonym("synthetic"),
        "action": "classify",
        "evidence_class": "AD",
        "evidence_scores": {"HC": 0, "AD": 4},
        "used_evidence_ids": [evidence_id],
        "counterevidence_ids": [],
        "quality_evidence_ids": [],
        "state_updates": [],
        "counterevidence_checked": True,
    }


def test_embedded_metric_outside_display_limit_is_a_valid_reference() -> None:
    workspace = _workspace()
    selected = {item["evidence_id"] for item in workspace["selected_supporting_evidence"]}
    assert "metric:m9" not in selected
    assert "metric:m9" in workspace["state_observations"][0]["metric_evidence_ids"]
    audit = cognitive_agent.validate_candidate(_candidate("metric:m9"), workspace, ["HC", "AD"])
    assert audit["valid"], audit["violations"]
    assert not cognitive_agent.validate_candidate(
        _candidate("metric:invented"), workspace, ["HC", "AD"]
    )["valid"]


@pytest.mark.parametrize("tags", [(), ("ASR_error",), ("ASR_error", "language", "task_type")])
def test_potential_confound_tags_are_unknown_not_observed_absence(tags) -> None:
    workspace = _workspace(tags)
    assert workspace["potential_confound_tags"] == sorted(tags)
    assert workspace["observed_confound_findings"] is None
    assert workspace["confound_assessment_status"] == "unknown"
    assert workspace["confound_burden"] is None
    assert workspace["correction_gate"] == 0.0
    assert workspace["correction_gate_reason"] == "unknown_case_level_confounds"
    metric = workspace["selected_supporting_evidence"][0]
    assert metric["potential_confound_tags"] == list(tags)
    assert metric["confound_tags"] == list(tags)
    assert metric["confound_tag_semantics"] == "potential_only"
    candidate = _candidate()
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    case_id = candidate["case_id"]
    prior = pd.DataFrame([{
        "subject_id": "synthetic", "predicted_label": "AD", "prob_HC": 0.3, "prob_AD": 0.7,
    }])
    predictions, _ = cognitive_agent._prediction_rows(
        prior, ["HC", "AD"], {case_id: candidate}, {case_id: audit}, {case_id: workspace}, 1.0,
    )
    np.testing.assert_array_equal(predictions[["prob_HC", "prob_AD"]], [[0.3, 0.7]])


def test_archived_numeric_gate_cannot_bypass_unknown_confound_assessment(tmp_path) -> None:
    workspace = _workspace(("ASR_error",))
    for key in ("confound_assessment_status", "observed_confound_findings", "correction_gate_reason"):
        workspace.pop(key)
    workspace.update(confound_burden=0.0, correction_gate=0.9)
    path = tmp_path / "archived.jsonl"
    path.write_text(json.dumps(workspace) + "\n")
    loaded = cognitive_agent._read_workspaces(path)[workspace["case_id"]]
    assert loaded["confound_assessment_status"] == "unknown"
    assert loaded["confound_burden"] is None
    assert loaded["correction_gate"] == 0.0
    assert loaded["selected_supporting_evidence"][0]["confound_tags"] == ["ASR_error"]


@pytest.mark.parametrize("burden", [0.0, 0.25, 1.0])
def test_explicit_confound_assessment_survives_workspace_roundtrip(tmp_path, burden) -> None:
    assessment = {"status": "assessed", "burden": burden, "provenance": "synthetic-test:assessment-v1"}
    workspace = _workspace(("ASR_error", "language", "task_type"), assessment)
    assert workspace["confound_assessment"] == assessment
    assert workspace["confound_assessment_status"] == "assessed"
    assert workspace["confound_burden"] == burden
    assert workspace["correction_gate"] == pytest.approx(np.sqrt(1.0 - burden))
    assert workspace["observed_confound_findings"] is None
    path = tmp_path / "assessed.jsonl"
    workspace["correction_gate"] = 0.123  # The loader must recompute, not trust a cached gate.
    path.write_text(json.dumps(workspace) + "\n")
    loaded = cognitive_agent._read_workspaces(path)[workspace["case_id"]]
    assert loaded["correction_gate"] == pytest.approx(np.sqrt(1.0 - burden))
    assert loaded["confound_assessment"] == assessment


def test_explicit_assessment_permits_synthetic_correction_without_removing_warnings() -> None:
    workspace = _workspace(("ASR_error",), {
        "status": "assessed", "burden": 0.25, "provenance": "synthetic-test:assessment-v1",
    })
    candidate = _candidate()
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    case_id = candidate["case_id"]
    prior = pd.DataFrame([{
        "subject_id": "synthetic", "predicted_label": "AD", "prob_HC": 0.3, "prob_AD": 0.7,
    }])
    predictions, _ = cognitive_agent._prediction_rows(
        prior, ["HC", "AD"], {case_id: candidate}, {case_id: audit}, {case_id: workspace}, 1.0,
    )
    assert predictions.iloc[0]["agent_correction_applied"]
    assert workspace["potential_confound_tags"] == ["ASR_error"]


@pytest.mark.parametrize("assessment", [
    {}, [], "assessed",
    {"status": "unknown", "burden": 0.0, "provenance": "test"},
    {"status": "assessed", "burden": -0.01, "provenance": "test"},
    {"status": "assessed", "burden": 1.01, "provenance": "test"},
    {"status": "assessed", "burden": float("nan"), "provenance": "test"},
    {"status": "assessed", "burden": float("inf"), "provenance": "test"},
    {"status": "assessed", "burden": True, "provenance": "test"},
    {"status": "assessed", "burden": "0", "provenance": "test"},
    {"status": "assessed", "burden": 0.0},
    {"status": "assessed", "burden": 0.0, "provenance": " "},
    {"status": "assessed", "burden": 0.0, "provenance": []},
])
def test_malformed_confound_assessment_is_rejected_at_build_and_load(tmp_path, assessment) -> None:
    with pytest.raises(ValueError, match="confound assessment"):
        _workspace(confound_assessment=assessment)
    workspace = _workspace()
    workspace["confound_assessment"] = assessment
    path = tmp_path / "malformed.jsonl"
    path.write_text(json.dumps(workspace) + "\n")
    with pytest.raises(ValueError, match="confound assessment"):
        cognitive_agent._read_workspaces(path)


@pytest.mark.parametrize("action", ["invalidate", "mark_unavailable"])
@pytest.mark.parametrize("reference", ["state:S1", "metric:m0", "metric:m9"])
def test_removed_state_evidence_cannot_correct_prior(action: str, reference: str) -> None:
    workspace = _workspace()
    candidate = _candidate(reference)
    candidate["state_updates"] = [{
        "state_id": "S1", "task_scope": "overall", "action": action,
        "reason": "Rejected measurement", "evidence_ids": ["state:S1"],
    }]
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    assert not audit["valid"]
    assert "V04_INVALIDATED_EVIDENCE_CITED" in audit["violations"]
    prior = pd.DataFrame([{
        "subject_id": "synthetic", "predicted_label": "AD", "prob_HC": 0.3, "prob_AD": 0.7,
    }])
    case_id = candidate["case_id"]
    predictions, _ = cognitive_agent._prediction_rows(
        prior, ["HC", "AD"], {case_id: candidate}, {case_id: audit}, {case_id: workspace}, 1.0,
    )
    np.testing.assert_array_equal(predictions[["prob_HC", "prob_AD"]], [[0.3, 0.7]])
    assert predictions.iloc[0]["agent_decision_status"] == "rolled_back_to_prior"


def test_invalidation_does_not_reject_independent_retained_evidence() -> None:
    workspace = _workspace()
    candidate = _candidate("state:S2")
    candidate["state_updates"] = [{
        "state_id": "S1", "task_scope": "overall", "action": "invalidate",
        "reason": "Rejected measurement", "evidence_ids": ["state:S1"],
    }]
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    assert audit["valid"], audit["violations"]
    assert audit["state_update_factor"] == 0.5


def test_task_alias_invalidation_rejects_its_segment_but_not_another_task() -> None:
    workspace = _workspace()
    state = workspace["state_observations"][0]
    state.update(state_id="S1__task_reading", task_scope="reading")
    state["evidence_segments"] = [{"segment_id": "segment:reading"}]
    workspace["evidence_registry"].append({
        "evidence_id": "segment:reading", "evidence_type": "segment",
    })
    candidate = _candidate("segment:reading")
    candidate["state_updates"] = [{
        "state_id": "S1", "task_scope": "reading", "action": "invalidate",
        "reason": "Rejected segment", "evidence_ids": ["segment:reading"],
    }]
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    assert "V04_INVALIDATED_EVIDENCE_CITED" in audit["violations"]
    candidate["used_evidence_ids"] = ["metric:m0"]
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    assert audit["valid"], audit["violations"]


def test_invalidated_counterevidence_is_not_required_as_retained_counterevidence() -> None:
    workspace = _workspace()
    counter = workspace["selected_supporting_evidence"].pop()
    workspace["selected_counterevidence"] = [counter]
    candidate = _candidate("state:S2")
    candidate["state_updates"] = [{
        "state_id": "S1", "task_scope": "overall", "action": "invalidate",
        "reason": "Rejected measurement", "evidence_ids": [counter["evidence_id"]],
    }]
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    assert audit["valid"], audit["violations"]
    candidate["counterevidence_ids"] = [counter["evidence_id"]]
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    assert "V04_INVALIDATED_EVIDENCE_CITED" in audit["violations"]
    candidate["counterevidence_ids"] = []
    candidate["state_updates"][0]["evidence_ids"] = []
    audit = cognitive_agent.validate_candidate(candidate, workspace, ["HC", "AD"])
    assert "V04_UNSUPPORTED_STATE_UPDATE" in audit["violations"]


def test_duplicate_case_responses_are_rejected_instead_of_overwritten() -> None:
    indexed, rejected = cognitive_agent._index_expected_candidates(
        [{"case_id": "A", "score": 0}, {"case_id": "A", "score": 4},
         {"case_id": "A", "score": 2}, {"case_id": "B"}],
        {"A", "B"},
    )
    assert indexed == {"B": {"case_id": "B"}}
    assert "A" in rejected


@pytest.mark.parametrize("cross_batch_response", [False, True])
def test_development_routes_use_calibrated_oof_evidence(
    tmp_path, monkeypatch, cross_batch_response: bool,
) -> None:
    labels = ["HC", "MCI", "AD"]
    prior = pd.DataFrame([
        {"subject_id": f"synthetic-{index}", "label": labels[index % 3],
         "predicted_label": "HC", "prob_HC": 0.9, "prob_MCI": 0.05, "prob_AD": 0.05}
        for index in range(15)
    ])
    prior["selection_independent"] = True
    prior["dedicated_calibration_holdout"] = True
    workspaces = []
    for subject_id in prior["subject_id"]:
        workspace = _workspace(confound_assessment={
            "status": "assessed", "burden": 0.0, "provenance": "synthetic-test:routing-v1",
        })
        workspace["case_id"] = case_pseudonym(subject_id)
        workspace["correction_gate"] = 1.0
        workspace["oof_provenance"] = {"selection_independent": True, "dedicated_calibration_holdout": True}
        workspaces.append(workspace)
    prior_path = tmp_path / "prior.csv"
    prior.to_csv(prior_path, index=False)
    test_prior_path = tmp_path / "test_prior.csv"
    test_prior = prior.copy()
    test_prior["subject_id"] = "test-" + test_prior["subject_id"]
    test_prior.to_csv(test_prior_path, index=False)
    workspace_path = tmp_path / "workspaces.jsonl"
    workspace_path.write_text("\n".join(json.dumps(item) for item in workspaces))
    monkeypatch.setattr(cognitive_agent, "_skill_text", lambda root: "")

    def request(**kwargs):
        cases = [dict(
            _candidate(), case_id=case_id, screening_class="HC",
            screening_scores={"HC": 3, "impaired": 2},
            staging_action="insufficient", staging_class="undetermined",
            staging_scores={"MCI": 2, "AD": 2},
        ) for case_id in kwargs["case_ids"]]
        first_case = workspaces[0]["case_id"]
        if cross_batch_response and first_case not in kwargs["case_ids"]:
            cases.append(dict(cases[0], case_id=first_case, action="abstain"))
        return {"cases": cases}

    monkeypatch.setattr(cognitive_agent, "_run_batch", request)
    fits = []

    def calibrate(truth, likelihood, usable):
        fitted = np.tile([0.2, 0.8] if not fits else [0.5, 0.5], (len(truth), 1))
        fits.append(fitted)
        return {"status": "calibrated_on_development_oof", "coefficient": 1.0,
                "intercept": 0.0, "oof_likelihood": fitted}

    monkeypatch.setattr(cognitive_agent, "fit_binary_evidence_calibrator", calibrate)
    captured = {}

    def select(*args, **kwargs):
        captured["multipliers"] = args[6]
        return {"selected_screening_strength": 0.0, "selected_staging_strength": 0.0,
                "selection_status": "failed_closed_no_joint_gain"}

    monkeypatch.setattr(cognitive_agent, "fit_agent_two_stage_strengths", select)
    cognitive_agent.run_cognitive_diagnostic_agent(
        root=tmp_path, prior_predictions_path=test_prior_path, workspaces_path=workspace_path,
        agents_config={"labels": labels, "model": "mock", "diagnostic_agent_min_calibration_cases": 15},
        provider="openai_api", predictions_path=tmp_path / "predictions.csv",
        decisions_path=tmp_path / "decisions.jsonl", audit_path=tmp_path / "audit.jsonl",
        locked_workspaces_path=tmp_path / "locked.jsonl", status_path=tmp_path / "status.json",
        prompt_path=tmp_path / "prompt.txt", calibration_predictions_path=prior_path,
        calibration_workspaces_path=workspace_path,
    )
    np.testing.assert_array_equal(captured["multipliers"], np.ones(15))
    status = json.loads((tmp_path / "status.json").read_text())
    assert not status["test_agent_gate_passed"]
    result = pd.read_csv(tmp_path / "predictions.csv")
    np.testing.assert_array_equal(result[["prob_HC", "prob_MCI", "prob_AD"]], prior[["prob_HC", "prob_MCI", "prob_AD"]])
