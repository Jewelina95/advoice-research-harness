from __future__ import annotations

import copy
import json
from pathlib import Path

import pandas as pd
import pytest

from advoice import diagnostic_agent_report as reports
from advoice.agent_runtime import case_pseudonym


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["HC", "AD"]


def _workspace(subject_id: str, *, released: bool) -> dict:
    metric = {
        "evidence_id": "metric:retained", "metric_id": "retained",
        "metric_instance_id": "retained", "value": 2.0, "reliability": 0.9,
        "reference_median": 1.0, "report_permission": True,
    }
    return {
        "case_id": case_pseudonym(subject_id),
        "subject_id": subject_id,
        "prediction_released": released,
        "released_probabilities": {"HC": 0.17, "AD": 0.83} if released else None,
        "final_prediction": "AD",
        "final_probabilities": {"HC": 0.17, "AD": 0.83},
        "supervised_prior_probabilities": {"HC": 0.23, "AD": 0.77},
        "selected_supporting_evidence": [metric],
        "selected_counterevidence": [],
        "quality_observations": [{"evidence_id": "quality:retained"}],
        "state_observations": [{
            "evidence_id": "state:retained", "state_id": "S01",
            "state_z": 1.0, "confidence": 0.9, "category": "borderline",
            "report_permission": True, "metric_evidence_ids": ["metric:retained"],
            "supporting_metrics": [metric], "counter_evidence": [],
            "evidence_segments": [{
                "segment_id": f"{subject_id}:segment", "case_id": subject_id,
                "start_sec": 1.0, "end_sec": 2.0,
            }],
        }],
        "evidence_registry": [
            {"evidence_id": evidence_id}
            for evidence_id in ["metric:retained", "state:retained", "quality:retained"]
        ],
        "evidence_revision": {
            "schema_version": "reviewed-evidence-v1",
            "excluded_evidence_ids": ["metric:rejected", "state:rejected"],
            "state_replay_required": not released,
            "snapshot_hash": "reviewed-snapshot",
        },
    }


def _run(tmp_path, workspaces, provider, *, include_model=True):
    predictions = pd.DataFrame([
        {"subject_id": item["subject_id"], "predicted_label": item["final_prediction"],
         "prob_HC": item["final_probabilities"]["HC"], "prob_AD": item["final_probabilities"]["AD"]}
        for item in workspaces
    ])
    predictions.to_csv(tmp_path / "predictions.csv", index=False)
    workspace_text = "\n".join(json.dumps(item) for item in workspaces)
    (tmp_path / "workspaces.jsonl").write_text(workspace_text)
    config = {"labels": LABELS, "max_report_cases": 10}
    if include_model:
        config["model"] = "mock-model"
    reports.run_diagnostic_agent_reports(
        ROOT, tmp_path / "predictions.csv", tmp_path / "workspaces.jsonl",
        config, provider, tmp_path / "reports.csv", tmp_path / "status.json",
        tmp_path / "prompt.txt",
    )
    assert (tmp_path / "workspaces.jsonl").read_text() == workspace_text
    return pd.read_csv(tmp_path / "reports.csv", keep_default_na=False), json.loads(
        (tmp_path / "status.json").read_text()
    )


def _mock_provider(monkeypatch, *, extra=None, use_rejected=False, use_segment=False):
    calls = []

    def run(root, prompt, schema_path, output_path, model, provider):
        cases = json.loads(prompt.rsplit("\n", 1)[-1])
        calls.append(cases)
        result = []
        for case in cases:
            item = reports._fallback_report(case, LABELS)
            if use_rejected:
                item["used_evidence_ids"] = ["metric:rejected"]
                item["report_zh"] = "REJECTED_FINDING"
            if use_segment:
                item["used_evidence_ids"] = [case["state_observations"][0]["evidence_segments"][0]["segment_id"]]
            result.append(item)
        if extra is not None:
            result.append(extra)
        return {"cases": result}

    monkeypatch.setattr(reports, "run_structured_batch", run)
    return calls


def _assert_withheld(row):
    assert row["predicted_label"] == "withheld"
    assert row["model"] == "deterministic_policy_fallback"
    assert json.loads(row["evidence"]) == {
        "used_evidence_ids": [], "counterevidence_ids": [], "quality_evidence_ids": [],
    }
    assert row["validation_status"] == "validated"
    assert json.loads(row["validation_errors"]) == []
    text = "\n".join(row[key] for key in ["report_zh", "patient_summary_zh", "uncertainty_zh"])
    assert text.strip()
    assert not any(token in text for token in ["%", "AD", "HC", "0.83", "0.77", "retained", "rejected", "REJECTED_FINDING"])


@pytest.mark.parametrize("provider", ["disabled", "openai_api", "codex_cli"])
def test_withheld_reports_never_call_provider(tmp_path, monkeypatch, provider):
    calls = _mock_provider(monkeypatch)
    workspace = _workspace("withheld-subject", released=False)
    frame, status = _run(tmp_path, [workspace], provider)
    assert calls == []
    assert len(frame) == 1
    _assert_withheld(frame.iloc[0])
    assert status["status"] == "completed_policy_fallback"
    assert status["replaced_unsafe_reports"] == 0


def test_all_withheld_does_not_require_provider_model(tmp_path, monkeypatch):
    calls = _mock_provider(monkeypatch)
    frame, _ = _run(
        tmp_path, [_workspace("withheld-subject", released=False)], "openai_api",
        include_model=False,
    )
    assert calls == []
    _assert_withheld(frame.iloc[0])


def test_mixed_batch_sends_only_released_cases_and_ignores_forged_withheld_response(tmp_path, monkeypatch):
    withheld = _workspace("withheld-subject", released=False)
    released = _workspace("released-subject", released=True)
    forged = {
        **reports._fallback_report(released, LABELS),
        "case_id": withheld["case_id"], "report_zh": "AD risk 83% REJECTED_FINDING",
    }
    calls = _mock_provider(monkeypatch, extra=forged, use_segment=True)
    frame, status = _run(tmp_path, [withheld, released], "openai_api")
    assert len(calls) == 1
    assert [case["case_id"] for case in calls[0]] == [released["case_id"]]
    assert len(frame) == 2
    indexed = frame.set_index("case_id")
    _assert_withheld(indexed.loc[withheld["case_id"]])
    live = indexed.loc[released["case_id"]]
    assert live["predicted_label"] == "AD"
    assert live["model"] == "mock-model"
    assert live["validation_status"] == "validated"
    assert json.loads(live["evidence"])["used_evidence_ids"] == [
        reports._segment_alias("released-subject:segment")
    ]
    assert "released-subject" not in json.dumps(calls)
    assert status["generated_cases"] == 2
    assert status["replaced_unsafe_reports"] == 0


def test_released_report_rejects_ids_removed_from_reviewed_snapshot(tmp_path, monkeypatch):
    calls = _mock_provider(monkeypatch, use_rejected=True)
    workspace = _workspace("released-subject", released=True)
    frame, status = _run(tmp_path, [workspace], "openai_api")
    assert len(calls) == 1
    submitted = calls[0][0]
    assert {item["evidence_id"] for item in submitted["selected_supporting_evidence"]} == {"metric:retained"}
    assert {item["evidence_id"] for item in submitted["state_observations"]} == {"state:retained"}
    row = frame.iloc[0]
    assert row["validation_status"] == "fallback_replaced"
    assert "unknown_evidence" in json.loads(row["validation_errors"])
    assert json.loads(row["evidence"])["used_evidence_ids"] == ["metric:retained"]
    assert "REJECTED_FINDING" not in row["report_zh"]
    assert status["unknown_evidence_violations"] == 1


def test_withheld_fallback_is_independent_of_evaluation_priors():
    workspace = _workspace("withheld-subject", released=False)
    original = copy.deepcopy(workspace)
    changed = copy.deepcopy(workspace)
    changed["final_prediction"] = "HC"
    changed["final_probabilities"] = {"HC": 0.99, "AD": 0.01}
    changed["supervised_prior_probabilities"] = {"HC": 0.98, "AD": 0.02}
    first = reports._fallback_report(workspace, LABELS)
    assert first == reports._fallback_report(changed, LABELS)
    assert first["predicted_label"] == "withheld"
    assert workspace == original


def test_validation_rejects_risk_or_evidence_in_withheld_report():
    workspace = _workspace("withheld-subject", released=False)
    safe = reports._fallback_report(workspace, LABELS)
    assert reports._report_validation_errors(safe, workspace, "withheld-subject") == []
    unsafe = {
        **safe, "report_zh": "AD risk 83%", "used_evidence_ids": ["metric:retained"],
    }
    assert reports._report_validation_errors(unsafe, workspace, "withheld-subject")
