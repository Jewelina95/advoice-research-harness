from __future__ import annotations

import json
from pathlib import Path

import pytest

from advoice.pilot.reporting import ReportInputError, aggregate_lock, render_run


def _aggregate() -> dict:
    payload = {
        "schema_version": "advoice.pilot.aggregate.v1",
        "status": "incomplete",
        "metadata": {"run_id": "run_fixture", "title": "Fixture pilot"},
        "datasets": [{"dataset_id": "PREPARE", "channel": "speech", "partition": "holdout", "subjects": 4, "status": "complete"}],
        "pipeline": {"description": "Fold-fitted evidence-state pilot", "agent": "bounded"},
        "versions": {"code": "abc123", "config": "cfg123"},
        "trace_example": {
            "case_id": "case_0123456789abcdef",
            "snapshot_hash": "a" * 64,
            "state_version": "state_v1",
            "evidence_ids": ["evidence_1"],
            "transcript": "must not be rendered",
        },
        "layer_a": [
            {"dataset_id": "PREPARE", "partition": "holdout", "arm": "B", "metric": "accuracy", "value": 0.75, "n": 4, "denominator": 4, "ci_low": 0.4, "ci_high": 0.95},
            {"dataset_id": "PREPARE", "partition": "holdout", "arm": "J-AS", "metric": "macro_auroc", "value": None, "n": 4, "denominator": 4, "undefined_reason": "class_support_missing"},
            {"dataset_id": "IAEAV", "partition": "stress", "arm": "B", "metric": "accuracy", "value": 0.5, "n": 2, "denominator": 2},
        ],
        "layer_b": [
            {"dataset_id": "PREPARE", "partition": "holdout", "arm": "B", "metric": "citation_validity", "value": 1.0, "n": 4, "denominator": 4},
            {"dataset_id": "IAEAV", "partition": "stress", "arm": "B", "metric": "api_failure_rate", "value": 0.25, "n": 4, "denominator": 4},
        ],
        "paired": [{"dataset_id": "PREPARE", "baseline_arm": "B_matched", "arm": "J-AS", "n": 4, "helped": 1, "harmed": 0,
                    "per_class": {"AD": {"n": 2, "helped": 1, "harmed": 0}}}],
        "failures": [{"name": "provider_failure", "detail": "one bounded request failed; baseline fallback retained"}, {"name": "cost", "detail": "tokens=120 cost_usd=0.004"}],
    }
    payload["lock"] = aggregate_lock(payload)
    return payload


def test_render_run_preserves_locked_values_and_sections(tmp_path: Path) -> None:
    run = tmp_path / "run"
    output = tmp_path / "reports"
    run.mkdir()
    (run / "aggregate.json").write_text(json.dumps(_aggregate()), encoding="utf-8")

    paths = render_run(run, output)
    evaluation = paths["evaluation_report"].read_text(encoding="utf-8")
    system = paths["system_report"].read_text(encoding="utf-8")

    assert "Layer A | Medical prediction and screening" in evaluation
    assert "Layer B | Technical framework proxies" in evaluation
    assert "0.750" in evaluation
    assert "class_support_missing" in evaluation
    assert "IAEAV" in evaluation and "Stress results" in evaluation
    assert "provider_failure" in evaluation and "tokens=120 cost_usd=0.004" in evaluation
    assert "SpeechCARE superiority is not evaluated" in evaluation
    assert "Paired help and harm" in evaluation
    assert "B_matched" in evaluation
    assert "must not be rendered" not in system
    assert "must not be rendered" not in system.lower()
    assert "case_0123456789abcdef" in system


def test_empty_run_is_honest_and_does_not_invent_zeros(tmp_path: Path) -> None:
    run = tmp_path / "run"
    output = tmp_path / "reports"
    run.mkdir()
    payload = {"schema_version": "advoice.pilot.aggregate.v1", "status": "incomplete", "metadata": {"run_id": "empty"}, "failures": ["no holdout metrics"]}
    payload["lock"] = aggregate_lock(payload)
    (run / "aggregate.json").write_text(json.dumps(payload), encoding="utf-8")

    paths = render_run(run, output)
    evaluation = paths["evaluation_report"].read_text(encoding="utf-8")

    assert "No Layer A metrics were locked" in evaluation
    assert "No Layer B metrics were locked" in evaluation
    assert "no holdout metrics" in evaluation
    assert "0.000" not in evaluation


def test_render_rejects_unlocked_tampered_or_non_incomplete_empty_aggregate(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = _aggregate()
    payload["layer_a"][0]["value"] = 0.1
    (run / "aggregate.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReportInputError, match="aggregate_hash mismatch"):
        render_run(run, tmp_path / "reports")

    payload = {"schema_version": "advoice.pilot.aggregate.v1", "status": "complete", "layer_a": [], "layer_b": []}
    payload["lock"] = aggregate_lock(payload)
    (run / "aggregate.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReportInputError, match="explicitly incomplete"):
        render_run(run, tmp_path / "reports")


def test_render_rejects_metric_rows_without_actual_denominators(tmp_path: Path) -> None:
    run = tmp_path / "run"
    run.mkdir()
    payload = _aggregate()
    del payload["layer_a"][0]["denominator"]
    payload["lock"] = aggregate_lock(payload)
    (run / "aggregate.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ReportInputError, match="lacks required metric fields"):
        render_run(run, tmp_path / "reports")
