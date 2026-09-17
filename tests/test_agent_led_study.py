from __future__ import annotations

import json
import math
from pathlib import Path

import pandas as pd
import pytest

from advoice.agent_led import evidence_snapshot
from advoice.agent_led_study import prepare_agent_led_study, run_agent_led_study
from advoice.agent_runtime import case_pseudonym
from advoice.utils import hash_values


ROOT = Path(__file__).resolve().parents[1]
LABELS = ["HC", "AD"]


def _prediction_rows() -> list[dict]:
    return [
        {"dataset_id": "fixture", "subject_id": "s1", "label": "HC", "split": "test",
         "prob_HC": 0.8, "prob_AD": 0.2, "predicted_label": "HC"},
        {"dataset_id": "fixture", "subject_id": "s2", "label": "AD", "split": "test",
         "prob_HC": 0.3, "prob_AD": 0.7, "predicted_label": "AD"},
        {"dataset_id": "fixture", "subject_id": "s3", "label": "HC", "split": "test",
         "prob_HC": 0.4, "prob_AD": 0.6, "predicted_label": "AD"},
    ]


def _artifact_dir(tmp_path: Path) -> Path:
    artifact = tmp_path / "artifacts"
    artifact.mkdir()
    rows = _prediction_rows()
    workspaces = []
    for row in rows:
        workspaces.append({
            "workspace_version": "fixture-v1",
            "case_id": case_pseudonym(row["subject_id"]),
            "case_context": {"task_scope": ["picture_description"], "languages": ["en"]},
            "quality_observations": [],
            "state_observations": [{
                "evidence_id": "state:S01", "state_id": "S01", "state_z": 0.5,
                "confidence": 0.8, "report_permission": True,
                "supporting_metrics": [], "counter_evidence": [], "evidence_segments": [],
            }],
            "selected_supporting_evidence": ["state:S01"],
            "selected_counterevidence": [],
            "base_probabilities": {"HC": row["prob_HC"], "AD": row["prob_AD"]},
            "corrected_probabilities": {"HC": row["prob_HC"], "AD": row["prob_AD"]},
        })
    (artifact / "diagnostic_agent_workspaces.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in workspaces), encoding="utf-8"
    )
    for name in ("b1_predictions.csv", "b2_predictions.csv", "ours_predictions.csv"):
        pd.DataFrame(rows).to_csv(artifact / name, index=False)
    (artifact / "ours_model.joblib").write_bytes(b"frozen-model")
    (artifact / "ours_model.json").write_text(json.dumps({
        "labels": LABELS, "base_architecture": "fixture", "base_selected_c": 1.0,
        "correction_selected_c": 0.5, "alpha_cv": {"alpha": 0.25},
    }), encoding="utf-8")
    pd.DataFrame([
        {"subject_id": "s1", "transcript": "short response", "recording_count": 1, "asr_model": "fixture"},
        {"subject_id": "s2", "transcript": "a much longer patient response with several task details", "recording_count": 1, "asr_model": "fixture"},
        {"subject_id": "s3", "transcript": "medium patient response with details", "recording_count": 1, "asr_model": "fixture"},
    ]).to_csv(artifact / "subject_transcripts.csv", index=False)
    return artifact


def test_prepare_is_label_blind_and_binds_exact_advisor_provenance(tmp_path):
    artifact = _artifact_dir(tmp_path)
    study = tmp_path / "study"
    prepared = prepare_agent_led_study(
        artifact, study, dataset_id="fixture", labels=LABELS,
        max_cases=2, selection_seed=17,
    )

    selected = [json.loads(line) for line in prepared["workspaces_path"].read_text().splitlines()]
    truth = json.loads(prepared["truth_path"].read_text())
    expected_ids = sorted(
        [case_pseudonym(row["subject_id"]) for row in _prediction_rows()],
        key=lambda case_id: hash_values([17, case_id]),
    )[:2]
    assert [row["case_id"] for row in selected] == expected_ids
    assert set(truth) == set(expected_ids)
    for workspace in selected:
        provenance = workspace["advisor_provenance"]
        assert provenance["evidence_hash"] == hash_values([evidence_snapshot(workspace)])
        assert set(provenance["artifacts"]) == {"module_a", "module_b"}
        assert provenance["artifacts"]["module_a"] != provenance["artifacts"]["module_b"]

    manifest = json.loads((study / "study_manifest.json").read_text())
    assert manifest["selection"]["uses_labels"] is False
    assert manifest["truth_access"] == "evaluation_after_inference_only"
    assert manifest["selected_case_ids"] == expected_ids


def test_external_provider_requires_explicit_permission_before_study_creation(tmp_path):
    artifact = _artifact_dir(tmp_path)
    study = tmp_path / "study"
    with pytest.raises(ValueError, match="explicit external-data permission"):
        run_agent_led_study(
            ROOT, artifact, study, dataset_id="fixture", labels=LABELS,
            provider="openai_api", model="test", max_cases=1,
            confirm_external_data_permission=False,
        )
    assert not study.exists()


def test_disabled_study_compares_exact_same_cohort_and_counts_nondecisions(tmp_path):
    artifact = _artifact_dir(tmp_path)
    study = tmp_path / "study"
    summary = run_agent_led_study(
        ROOT, artifact, study, dataset_id="fixture", labels=LABELS,
        provider="disabled", model="test", max_cases=2, selection_seed=17,
    )

    comparison = json.loads((study / "comparison.json").read_text())
    assert summary["cases"] == 2
    assert comparison["cohort_case_ids"] == comparison["agent_led"]["case_ids"]
    assert comparison["baselines"]["ours"]["n"] == 2
    assert comparison["agent_led"]["accuracy"] == 0.0
    assert comparison["agent_led"]["coverage"] == 0.0
    assert comparison["speechcare"]["direct_comparison_valid"] is False
    assert (study / "study_report.html").exists()


def test_missing_subject_mapping_fails_closed(tmp_path):
    artifact = _artifact_dir(tmp_path)
    frame = pd.read_csv(artifact / "ours_predictions.csv")
    frame.loc[0, "subject_id"] = "unmapped"
    frame.to_csv(artifact / "ours_predictions.csv", index=False)
    with pytest.raises(ValueError, match="case IDs do not match"):
        prepare_agent_led_study(
            artifact, tmp_path / "study", dataset_id="fixture", labels=LABELS,
            max_cases=None, selection_seed=17,
        )


def test_nonfinite_legacy_values_become_unobserved_nulls(tmp_path):
    artifact = _artifact_dir(tmp_path)
    rows = [json.loads(line) for line in (artifact / "diagnostic_agent_workspaces.jsonl").read_text().splitlines()]
    rows[0]["state_observations"][0]["state_z"] = math.nan
    (artifact / "diagnostic_agent_workspaces.jsonl").write_text(
        "".join(json.dumps(row, allow_nan=True) + "\n" for row in rows), encoding="utf-8"
    )
    study = tmp_path / "study"
    prepared = prepare_agent_led_study(
        artifact, study, dataset_id="fixture", labels=LABELS,
        max_cases=None, selection_seed=17,
    )
    selected = [json.loads(line) for line in prepared["workspaces_path"].read_text().splitlines()]
    affected = next(row for row in selected if row["case_id"] == rows[0]["case_id"])
    assert affected["state_observations"][0]["state_z"] is None
    manifest = json.loads((study / "study_manifest.json").read_text())
    assert manifest["input_normalization"]["nonfinite_values_replaced_with_null"] == 1


def test_baseline_truth_disagreement_fails_closed(tmp_path):
    artifact = _artifact_dir(tmp_path)
    frame = pd.read_csv(artifact / "b1_predictions.csv")
    frame.loc[0, "label"] = "AD"
    frame.to_csv(artifact / "b1_predictions.csv", index=False)
    with pytest.raises(ValueError, match="truth labels do not match"):
        run_agent_led_study(
            ROOT, artifact, tmp_path / "study", dataset_id="fixture", labels=LABELS,
            provider="disabled", model="test", max_cases=None, selection_seed=17,
        )


def test_frozen_model_label_order_must_match_study(tmp_path):
    artifact = _artifact_dir(tmp_path)
    metadata = json.loads((artifact / "ours_model.json").read_text())
    metadata["labels"] = ["AD", "HC"]
    (artifact / "ours_model.json").write_text(json.dumps(metadata), encoding="utf-8")
    with pytest.raises(ValueError, match="label order"):
        prepare_agent_led_study(
            artifact, tmp_path / "study", dataset_id="fixture", labels=LABELS,
            max_cases=None, selection_seed=17,
        )


def test_long_transcript_selection_is_label_blind_and_attaches_transcript(tmp_path):
    artifact = _artifact_dir(tmp_path)
    prepared = prepare_agent_led_study(
        artifact, tmp_path / "study", dataset_id="fixture", labels=LABELS,
        max_cases=1, selection_seed=17, selection_method="longest_transcript",
    )
    selected = [json.loads(line) for line in prepared["workspaces_path"].read_text().splitlines()]
    assert selected[0]["case_id"] == case_pseudonym("s2")
    assert selected[0]["case_transcript"]["character_count"] > 40
    manifest = json.loads((tmp_path / "study" / "study_manifest.json").read_text())
    assert manifest["selection"]["uses_labels"] is False
    assert manifest["selection"]["selection_method"] == "longest_transcript"
    assert manifest["selection"]["selected_with_transcript"] == 1
