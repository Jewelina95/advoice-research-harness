import copy

import numpy as np
import pandas as pd

from advoice.cognitive_agent import _prediction_rows, validate_candidate
from advoice.diagnostic_agent_report import _fallback_report, _sanitize_workspace, _report_validation_errors
from test_agent_evidence_execution import _workspace, _candidate


def _review():
    workspace = _workspace(confound_assessment={"status": "assessed", "burden": 0., "provenance": "synthetic:test"})
    candidate = _candidate("state:S2")
    candidate["state_updates"] = [{"state_id": "S1", "task_scope": "overall", "action": "invalidate",
                                   "reason": "Incorrect extraction", "evidence_ids": ["state:S1"]}]
    audit = validate_candidate(candidate, workspace, ["HC", "AD"])
    assert audit["valid"], audit
    return workspace, candidate, audit


def test_invalidated_metrics_cannot_reappear_in_locked_report():
    workspace, candidate, audit = _review()
    original = copy.deepcopy(workspace)
    case_id = workspace["case_id"]
    prior = pd.DataFrame([{"subject_id": "synthetic", "predicted_label": "AD", "prob_HC": .3, "prob_AD": .7}])
    _, locked = _prediction_rows(prior, ["HC", "AD"], {case_id: candidate}, {case_id: audit}, {case_id: workspace}, 1.)
    sanitized = _sanitize_workspace(locked[0], case_id)
    report = _fallback_report(sanitized, ["HC", "AD"])
    assert "metric:m0" not in report["used_evidence_ids"]
    assert "state:S1" not in {v["evidence_id"] for v in sanitized["state_observations"]}
    forged = {**report, "used_evidence_ids": ["metric:m0"]}
    assert _report_validation_errors(forged, sanitized, "synthetic")
    assert workspace == original


def test_unreplayed_state_edits_withhold_clinical_probability_not_fake_recompute():
    workspace, candidate, audit = _review()
    case_id = workspace["case_id"]
    prior = pd.DataFrame([{"subject_id": "synthetic", "predicted_label": "AD", "prob_HC": .3, "prob_AD": .7}])
    rows, locked = _prediction_rows(prior, ["HC", "AD"], {case_id: candidate}, {case_id: audit}, {case_id: workspace}, 1.)
    assert not rows.iloc[0]["prediction_released"]
    assert rows.iloc[0]["agent_decision_status"] == "state_replay_required"
    np.testing.assert_array_equal(rows[["prob_HC", "prob_AD"]], [[.3, .7]])
    report = _fallback_report(_sanitize_workspace(locked[0], case_id), ["HC", "AD"])
    assert "%" not in report["report_zh"]
    assert locked[0]["released_probabilities"] is None


def test_rejected_update_does_not_modify_evidence_snapshot():
    workspace, candidate, _ = _review()
    candidate["used_evidence_ids"] = ["state:S1"]
    audit = validate_candidate(candidate, workspace, ["HC", "AD"])
    assert not audit["valid"]
    case_id = workspace["case_id"]
    prior = pd.DataFrame([{"subject_id": "synthetic", "predicted_label": "AD", "prob_HC": .3, "prob_AD": .7}])
    _, locked = _prediction_rows(prior, ["HC", "AD"], {case_id: candidate}, {case_id: audit}, {case_id: workspace}, 1.)
    assert locked[0]["state_observations"] == workspace["state_observations"]
