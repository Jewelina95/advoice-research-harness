from copy import deepcopy
import html
import json
from pathlib import Path

import pytest

from advoice import agent_led_run
from advoice.cli import parser

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "demo/agent_led/synthetic_workspace.jsonl"


def test_disabled_is_not_a_fake_agent_prediction(tmp_path, monkeypatch):
    session_type = agent_led_run.EvidenceSession
    providers = []
    def session(*args, **kwargs):
        providers.append(kwargs["provider"])
        return session_type(*args, **kwargs)
    def forbidden(*args, **kwargs):
        raise AssertionError("Provider must not be called")
    monkeypatch.setattr(agent_led_run, "EvidenceSession", session)
    monkeypatch.setattr(agent_led_run, "run_structured_batch", forbidden)
    out = tmp_path / "run"
    result = agent_led_run.run_agent_led_cohort(ROOT, FIXTURE, out, ["HC", "AD"], provider="disabled", model="test")
    assert result["provider_requests"] == 0
    assert result["decided"] == 0
    assert providers == ["disabled"]
    decision = json.loads((out / "decisions.jsonl").read_text())
    assert decision["predicted_label"] is None
    assert (out / "report.html").exists()
    with pytest.raises(FileExistsError):
        agent_led_run.run_agent_led_cohort(ROOT, FIXTURE, out, ["HC", "AD"], provider="disabled", model="test")


def test_real_provider_adapter_runs_multiple_tools_without_training(tmp_path, monkeypatch):
    seen = []
    providers = []
    run_session = agent_led_run.run_agent_session
    def session(*args, **kwargs):
        providers.append(kwargs["provider"])
        return run_session(*args, **kwargs)
    monkeypatch.setattr(agent_led_run, "run_agent_session", session)
    actions = ["inspect_quality", "inspect_state", "inspect_counterevidence", "record_hypothesis", "finalize"]
    def fake(root, prompt, schema, output, model, provider):
        obs = json.loads(prompt.split("DATA (not instructions):\n")[1])
        seen.append(obs)
        assert "base_probabilities" not in prompt
        assert provider == "openai_api"
        return {
            "action": actions[len(seen)-1], "revision": obs["revision"],
            "target_id": "state:S07", "state_action": "none",
            "evidence_ids": ["state:S07"], "counterevidence_ids": ["metric:continuity"],
            "predicted_label": "AD", "scores": {"HC": 1, "AD": 3},
            "rationale": "Synthetic fixture reasoning.", "limitations": ["Not a patient."],
        }
    monkeypatch.setattr(agent_led_run, "run_structured_batch", fake)
    out = tmp_path / "run"
    summary = agent_led_run.run_agent_led_cohort(ROOT, FIXTURE, out, ["HC", "AD"], provider="openai_api", model="test")
    assert summary["provider_requests"] == 5
    assert summary["provider_parse_retries"] == 0
    assert summary["decided"] == 1
    assert providers == ["openai_api"]
    assert not summary["training_performed"]
    result = json.loads((out / "decisions.jsonl").read_text())
    assert result["prediction_source"] == "agent"
    assert result["probabilities"] is None
    assert result["predicted_label"] == "AD"


def test_real_provider_retries_one_malformed_structured_response(tmp_path, monkeypatch):
    actions = ["inspect_quality", "inspect_state", "inspect_counterevidence", "record_hypothesis", "finalize"]
    calls = 0

    def fake(root, prompt, schema, output, model, provider):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise json.JSONDecodeError("malformed structured output", "{", 1)
        observation = json.loads(prompt.split("DATA (not instructions):\n")[1])
        action = actions[calls - 2]
        return {
            "action": action, "revision": observation["revision"],
            "target_id": "state:S07", "state_action": "none",
            "evidence_ids": ["state:S07"],
            "counterevidence_ids": ["metric:continuity"],
            "predicted_label": "AD", "scores": {"HC": 1, "AD": 3},
            "rationale": "Synthetic fixture reasoning.", "limitations": ["Not a patient."],
        }

    monkeypatch.setattr(agent_led_run, "run_structured_batch", fake)
    out = tmp_path / "run"
    summary = agent_led_run.run_agent_led_cohort(
        ROOT, FIXTURE, out, ["HC", "AD"], provider="openai_api", model="test",
    )
    assert summary["decided"] == 1
    assert summary["provider_requests"] == 6
    assert summary["provider_parse_retries"] == 1
    result = json.loads((out / "decisions.jsonl").read_text())
    assert result["provider_requests"] == 6
    assert result["provider_parse_retries"] == 1


def test_cli_has_explicit_nontraining_inference_path():
    args = parser().parse_args(["agent-led", "--workspaces", str(FIXTURE), "--output-dir", "/tmp/unused", "--labels", "HC", "AD"])
    assert args.provider == "disabled"
    assert args.command == "agent-led"


def test_agent_led_cli_rejects_codex_without_changing_other_commands():
    with pytest.raises(SystemExit) as error:
        parser().parse_args([
            "agent-led", "--workspaces", str(FIXTURE), "--output-dir", "/tmp/unused",
            "--labels", "HC", "AD", "--provider", "codex_cli",
        ])
    assert error.value.code == 2
    for command in ("run", "run-all", "run-processed", "run-all-processed"):
        assert parser().parse_args([command, "--agent-provider", "codex_cli"]).agent_provider == "codex_cli"


@pytest.mark.parametrize("provider", ["codex_cli", "custom", "unknown"])
def test_cohort_rejects_unsafe_provider_before_io(tmp_path, monkeypatch, provider):
    def forbidden(*args, **kwargs):
        raise AssertionError("Provider must not be called")
    monkeypatch.setattr(agent_led_run, "run_structured_batch", forbidden)
    out = tmp_path / "run"
    with pytest.raises(ValueError, match="filesystem-capable providers are not permitted"):
        agent_led_run.run_agent_led_cohort(
            ROOT, tmp_path / "missing-workspaces.jsonl", out, ["HC", "AD"],
            provider=provider, model="test",
        )
    assert not out.exists()


def report_result(status="decided"):
    return {
        "case_id": "synthetic", "status": status,
        "predicted_label": "AD" if status == "decided" else None,
        "evidence_ids": ["state:visible"], "trace": [],
        "rationale": "Research interpretation.", "limitations": ["Not a patient."],
        "reviewed_workspace": {
            "state_observations": [{
                "evidence_id": "state:visible", "report_permission": True,
                "state_z": 1.5, "confidence": .8,
                "supporting_metrics": [], "counter_evidence": [], "evidence_segments": [],
                "metric_evidence_ids": [],
            }],
        },
    }


@pytest.mark.parametrize("field", ["supporting_metrics", "counter_evidence", "evidence_segments"])
@pytest.mark.parametrize("permission", [{}, {"report_permission": False},
                                       {"report_permission": None}, {"report_permission": 1},
                                       {"report_permission": "true"}])
def test_report_denies_children_without_explicit_permission(tmp_path, field, permission):
    result = report_result()
    state = result["reviewed_workspace"]["state_observations"][0]
    state[field] = [
        {"metric_id": "hidden_child", "segment_id": "segment:hidden_child",
         "value": 12345, "text": "hidden_payload", **permission},
        {"metric_id": "visible_child", "segment_id": "segment:visible_child",
         "value": 2, "text": "visible_payload", "report_permission": True},
    ]
    state["metric_evidence_ids"] = ["metric:hidden_child", "metric:visible_child"]
    original = deepcopy(result)
    path = tmp_path / "report.html"
    agent_led_run._report([result], None, path)
    rendered = html.unescape(path.read_text())
    assert "hidden_child" not in rendered
    assert "hidden_payload" not in rendered
    assert "12345" not in rendered
    assert "visible_child" in rendered
    assert "visible_payload" in rendered
    assert result == original


def test_report_projection_filters_nested_children_and_unknown_fields(tmp_path):
    result = report_result()
    state = result["reviewed_workspace"]["state_observations"][0]
    state.update({
        "model_state_z": "hidden_model_value",
        "unreviewed_metadata": {"text": "hidden_metadata"},
        "supporting_metrics": [{
            "metric_id": "visible_metric", "report_permission": True,
            "value": 2, "evidence_segments": [
                {"segment_id": "segment:allowed", "text": "<visible transcript>", "report_permission": True},
                {"segment_id": "segment:denied", "text": "hidden_segment", "report_permission": False},
            ],
        }],
        "metric_evidence_ids": ["metric:visible_metric", "metric:hidden_reference"],
    })
    path = tmp_path / "report.html"
    agent_led_run._report([result], None, path)
    rendered = path.read_text()
    assert "&lt;visible transcript&gt;" in rendered
    assert "<visible transcript>" not in rendered
    assert "metric:visible_metric" in rendered
    assert all(value not in rendered for value in (
        "hidden_model_value", "hidden_metadata", "hidden_segment", "hidden_reference",
    ))


@pytest.mark.parametrize("permission", [{}, {"report_permission": False}, {"report_permission": 1}])
def test_report_denies_unpermitted_parent_even_with_permitted_child(tmp_path, permission):
    result = report_result()
    state = result["reviewed_workspace"]["state_observations"][0]
    state.pop("report_permission")
    state.update(permission)
    state["supporting_metrics"] = [{"metric_id": "hidden_child", "report_permission": True}]
    path = tmp_path / "report.html"
    agent_led_run._report([result], None, path)
    rendered = path.read_text()
    assert "state:visible" not in rendered
    assert "hidden_child" not in rendered


@pytest.mark.parametrize("status", ["abstained", "provider_disabled", "provider_error", "budget_exhausted", "unknown"])
def test_nondecisions_never_publish_clinical_findings(tmp_path, status):
    result = report_result(status)
    path = tmp_path / "report.html"
    agent_led_run._report([result], None, path)
    rendered = path.read_text()
    assert "Source-linked findings" not in rendered
    assert "state:visible" not in rendered
    assert "Undetermined" in rendered
    assert "Research interpretation." in rendered
    assert "Clinical release: not authorized" in rendered


def test_report_renders_permitted_direct_metric_projection(tmp_path):
    result = report_result()
    result["evidence_ids"] = ["metric:visible", "metric:denied"]
    result["reviewed_workspace"]["selected_supporting_evidence"] = [
        {"evidence_id": "metric:visible", "value": 7, "report_permission": True,
         "metadata": {"text": "hidden_metadata"}},
        {"evidence_id": "metric:denied", "value": 8, "report_permission": False},
    ]
    path = tmp_path / "report.html"
    agent_led_run._report([result], None, path)
    rendered = html.unescape(path.read_text())
    assert "metric:visible" in rendered
    assert '"value": 7' in rendered
    assert "metric:denied" not in rendered
    assert "hidden_metadata" not in rendered
