import json

import pandas as pd
import pytest

import advoice.cognitive_agent as agent
from advoice.agent_runtime import case_pseudonym
from advoice.cognitive_agent import _calibration_partition_error


def _inputs():
    calibration = pd.DataFrame({"subject_id": ["cal"], "selection_independent": [True],
                                "dedicated_calibration_holdout": [True]})
    test = pd.DataFrame({"subject_id": ["test"]})
    workspaces = {case_pseudonym("cal"): {"oof_provenance": {
        "selection_independent": True, "dedicated_calibration_holdout": True}}}
    return calibration, test, workspaces


def test_verified_disjoint_holdout_is_accepted():
    assert _calibration_partition_error(*_inputs()) is None


@pytest.mark.parametrize("column", ["selection_independent", "dedicated_calibration_holdout"])
def test_legacy_or_selection_dependent_calibration_is_rejected(column):
    calibration, test, workspaces = _inputs()
    assert _calibration_partition_error(calibration.drop(columns=column), test, workspaces)
    calibration[column] = False
    assert _calibration_partition_error(calibration, test, workspaces)


def test_overlapping_subjects_and_missing_workspace_are_rejected():
    calibration, test, workspaces = _inputs()
    assert _calibration_partition_error(calibration, calibration, workspaces) == "calibration_test_overlap"
    assert _calibration_partition_error(calibration, test, {}) == "unverified_calibration_workspace"


@pytest.mark.parametrize("independent", [False, True])
def test_invalid_partition_does_not_make_api_requests(tmp_path, monkeypatch, independent):
    calibration, test, workspaces = _inputs()
    calibration["selection_independent"] = independent
    for frame in (calibration, test):
        frame["label"] = "HC"
        frame["predicted_label"] = "HC"
        frame["prob_HC"], frame["prob_AD"] = .8, .2
    for case_id, workspace in workspaces.items():
        workspace["case_id"] = case_id
    calibration.to_csv(tmp_path / "cal.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    (tmp_path / "workspaces.jsonl").write_text("\n".join(json.dumps(w) for w in workspaces.values()))
    monkeypatch.setattr(agent, "_skill_text", lambda root: "test skill")
    def forbidden(**kwargs):
        pytest.fail("Invalid calibration must be blocked before a paid API request")
    monkeypatch.setattr(agent, "_run_batch", forbidden)
    agent.run_cognitive_diagnostic_agent(
        root=tmp_path, prior_predictions_path=tmp_path / "test.csv",
        workspaces_path=tmp_path / "workspaces.jsonl",
        agents_config={"labels": ["HC", "AD"], "model": "mock"}, provider="openai_api",
        predictions_path=tmp_path / "predictions.csv", decisions_path=tmp_path / "decisions.jsonl",
        audit_path=tmp_path / "audit.jsonl", locked_workspaces_path=tmp_path / "locked.jsonl",
        status_path=tmp_path / "status.json", prompt_path=tmp_path / "prompt.txt",
        calibration_predictions_path=tmp_path / "cal.csv",
        calibration_workspaces_path=tmp_path / "workspaces.jsonl",
        calibration_result_path=tmp_path / "calibration.json",
    )
    status = json.loads((tmp_path / "calibration.json").read_text())
    assert status["status"] == "failed_closed_invalid_calibration_partition"
    assert status["partition_error"] == (
        "insufficient_calibration_subjects" if independent else "unverified_independent_calibration")
    predictions = pd.read_csv(tmp_path / "predictions.csv")
    assert predictions.prob_HC.tolist() == [.8]
