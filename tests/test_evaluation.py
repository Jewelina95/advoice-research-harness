from __future__ import annotations

import pandas as pd

from advoice.evaluation import (
    _available_state_cards,
    _report_permission_rate,
    _quality_reference_rate,
    _segment_faithfulness_rate,
    _trace_presence_rate,
    evaluate_predictions,
    paired_prediction_comparison,
)


def test_multiclass_metrics_perfect() -> None:
    frame = pd.DataFrame(
        {
            "label": ["HC", "MCI", "AD", "HC", "MCI", "AD"],
            "predicted_label": ["HC", "MCI", "AD", "HC", "MCI", "AD"],
            "prob_HC": [0.9, 0.05, 0.05, 0.9, 0.05, 0.05],
            "prob_MCI": [0.05, 0.9, 0.05, 0.05, 0.9, 0.05],
            "prob_AD": [0.05, 0.05, 0.9, 0.05, 0.05, 0.9],
        }
    )
    result = evaluate_predictions(frame, 10, ["HC", "MCI", "AD"], "AD")
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["macro_auroc_ovr"] == 1.0
    assert result["per_class"]["AD"]["sensitivity"] == 1.0
    assert result["referral_auroc"] == 1.0
    assert result["referral_specificity_at_locked_threshold"] == 1.0


def test_paired_prediction_comparison_uses_matched_subjects() -> None:
    baseline = pd.DataFrame(
        {
            "subject_id": ["1", "2", "3", "4"],
            "label": ["HC", "HC", "AD", "AD"],
            "predicted_label": ["HC", "AD", "HC", "AD"],
            "prob_HC": [0.8, 0.4, 0.6, 0.2],
            "prob_AD": [0.2, 0.6, 0.4, 0.8],
        }
    )
    ours = baseline.copy()
    ours["predicted_label"] = ["HC", "HC", "AD", "AD"]
    ours["prob_HC"] = [0.9, 0.7, 0.2, 0.1]
    ours["prob_AD"] = [0.1, 0.3, 0.8, 0.9]
    result = paired_prediction_comparison(
        ours, baseline, 5, 50, ["HC", "AD"], "AD"
    )
    assert result["status"] == "completed"
    assert result["n_matched"] == 4
    assert result["delta_ours_minus_baseline"]["accuracy"] == 0.5
    assert result["mcnemar_exact"]["ours_correct_baseline_wrong"] == 2
    assert result["mcnemar_exact"]["ours_wrong_baseline_correct"] == 0


def test_paired_prediction_comparison_rejects_different_subject_cohorts() -> None:
    baseline = pd.DataFrame(
        {
            "subject_id": ["1", "2", "3", "4"],
            "label": ["HC", "HC", "AD", "AD"],
            "predicted_label": ["HC", "HC", "AD", "AD"],
            "prob_HC": [0.8, 0.7, 0.2, 0.1],
            "prob_AD": [0.2, 0.3, 0.8, 0.9],
        }
    )
    ours = baseline.iloc[:-1].copy()
    result = paired_prediction_comparison(
        ours, baseline, 5, 20, ["HC", "AD"], "AD"
    )
    assert result["status"] == "invalid"
    assert result["baseline_only_subjects"] == ["4"]


def test_binary_screening_metrics_are_computed() -> None:
    frame = pd.DataFrame(
        {
            "label": ["HC", "HC", "AD", "AD"],
            "predicted_label": ["HC", "HC", "AD", "AD"],
            "prob_HC": [0.9, 0.8, 0.2, 0.1],
            "prob_AD": [0.1, 0.2, 0.8, 0.9],
        }
    )
    result = evaluate_predictions(frame, 10, ["HC", "AD"], "AD")
    assert result["accuracy"] == 1.0
    assert result["macro_auroc_ovr"] == 1.0
    assert result["screening_threshold"] == 0.5
    assert result["sensitivity_at_locked_threshold"] == 1.0
    assert result["specificity_at_locked_threshold"] == 1.0
    assert "near_threshold_uncertainty_rate" in result
    assert "high_confidence_error_rate" in result
    assert result["tp"] + result["tn"] + result["fp"] + result["fn"] == len(frame)


def test_trace_rate_excludes_states_unavailable_for_the_source_task() -> None:
    cards = pd.DataFrame(
        {
            "missing_fraction": [0.0, 1.0, 0.25],
            "evidence_segments": ['[{"segment_id":"S1"}]', "[]", "[]"],
        }
    )
    available = _available_state_cards(cards)
    assert len(available) == 2
    assert _trace_presence_rate(available) == 0.5


def test_report_permission_audit_rejects_qc_citations() -> None:
    reports = pd.DataFrame(
        {
            "subject_id": ["case-1"],
            "evidence": [
                '{"used_evidence_ids":["pause_rate","snr_proxy_db"],"counterevidence_ids":[]}'
            ],
        }
    )
    evidence = pd.DataFrame(
        {
            "subject_id": ["case-1", "case-1"],
            "metric_id": ["pause_rate", "snr_proxy_db"],
            "metric_instance_id": ["pause_rate", "snr_proxy_db"],
            "task_scope": ["overall", "overall"],
            "report_permission": [True, False],
            "evidence_role": ["clinical_support", "qc_only"],
            "missing": [False, False],
        }
    )
    assert _report_permission_rate(reports, evidence) == 0.5


def test_report_permission_audit_uses_state_level_report_view() -> None:
    reports = pd.DataFrame(
        {
            "subject_id": ["case-1"],
            "evidence": [
                '{"used_evidence_ids":["S01","S06"],"counterevidence_ids":[]}'
            ],
        }
    )
    evidence = pd.DataFrame(
        columns=[
            "subject_id",
            "metric_id",
            "metric_instance_id",
            "task_scope",
            "report_permission",
            "evidence_role",
            "missing",
        ]
    )
    cards = pd.DataFrame(
        {
            "subject_id": ["case-1", "case-1"],
            "state_id": ["S01", "S06"],
            "report_permission": [True, False],
            "report_confidence": [0.8, 0.0],
            "missing_fraction": [0.0, 0.0],
        }
    )
    assert _report_permission_rate(reports, evidence, cards) == 0.5


def test_quality_evidence_must_use_qc_role_and_separate_field() -> None:
    reports = pd.DataFrame(
        {
            "subject_id": ["case-1"],
            "evidence": [
                '{"used_evidence_ids":[],"counterevidence_ids":[],'
                '"quality_evidence_ids":["snr_proxy_db"]}'
            ],
        }
    )
    evidence = pd.DataFrame(
        {
            "subject_id": ["case-1"],
            "metric_instance_id": ["snr_proxy_db"],
            "evidence_role": ["qc_only"],
            "report_permission": [False],
        }
    )
    assert _quality_reference_rate(reports, evidence) == 1.0
    evidence.loc[0, "evidence_role"] = "clinical_support"
    assert _quality_reference_rate(reports, evidence) == 0.0


def test_report_audits_resolve_typed_evidence_identifiers() -> None:
    reports = pd.DataFrame(
        {
            "subject_id": ["case-1"],
            "evidence": [
                '{"used_evidence_ids":["state:S01","metric:pause_rate"],'
                '"counterevidence_ids":[],"quality_evidence_ids":["qc:snr_proxy_db"]}'
            ],
        }
    )
    evidence = pd.DataFrame(
        {
            "subject_id": ["case-1", "case-1"],
            "metric_id": ["pause_rate", "snr_proxy_db"],
            "metric_instance_id": ["pause_rate", "snr_proxy_db"],
            "task_scope": ["overall", "overall"],
            "report_permission": [True, False],
            "evidence_role": ["clinical_support", "qc_only"],
            "missing": [False, False],
        }
    )
    cards = pd.DataFrame(
        {
            "subject_id": ["case-1"],
            "state_id": ["S01"],
            "report_permission": [True],
            "report_confidence": [0.8],
            "missing_fraction": [0.0],
        }
    )
    assert _report_permission_rate(reports, evidence, cards) == 1.0
    assert _quality_reference_rate(reports, evidence) == 1.0


def test_segment_faithfulness_checks_source_and_state_selection_rule() -> None:
    segments = pd.DataFrame(
        {
            "case_id": ["case-1"],
            "segment_id": ["case-1:S01"],
            "start_sec": [0.0],
            "end_sec": [10.0],
            "source_spans": ["[[0.0,10.0]]"],
            "silence_fraction": [0.6],
        }
    )
    item = (
        '[{"segment_id":"case-1:S01","case_id":"case-1",'
        '"start_sec":0.0,"end_sec":10.0,"source_spans":"[[0.0,10.0]]",'
        '"selection_basis":"highest_silence_fraction"}]'
    )
    cards = pd.DataFrame(
        {
            "state_base_id": ["S01"],
            "trace_resolution": ["segment"],
            "missing_fraction": [0.0],
            "evidence_segments": [item],
        }
    )
    assert _segment_faithfulness_rate(cards, segments) == 1.0
    cards.loc[0, "evidence_segments"] = item.replace(
        "highest_silence_fraction", "lowest_rms"
    )
    assert _segment_faithfulness_rate(cards, segments) == 0.0
