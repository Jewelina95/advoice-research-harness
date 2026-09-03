from __future__ import annotations

import json

import numpy as np
import pandas as pd

from advoice.condition_c import (
    _add_age_band_features,
    _apply_logit_offsets,
    _case_level_expert_contributions,
    _hierarchical_state_weights,
    _expert_passes_cv_gate,
    _numeric_pipeline,
    _replace_card_metric_summaries,
    _small_sample_correction_guard,
    _stable_fold_offsets,
    clean_model_transcript,
    train_condition_c,
)
from advoice.deep_audio_embeddings import _parse_intervals, _windows
from advoice.cognitive_agent import (
    _allowed_ids,
    _cached_response,
    _index_expected_candidates,
    _select_correction_candidate,
    _state_update_factor,
    _test_agent_gate_passed,
    apply_binary_evidence_calibrator,
    blind_workspace,
    fit_binary_evidence_calibrator,
    fit_agent_correction_strength,
    fit_agent_two_stage_strengths,
    two_stage_route_parameters,
    validate_candidate,
)
from advoice.cognitive_prototypes import _robust_distance
from advoice.diagnostic_agent import (
    build_case_workspace,
    case_input_route,
    evidence_gate,
    fuse_evidence_likelihood,
    fuse_two_stage_evidence,
    fuse_corrected_probability,
    route_case,
    structured_evidence_coverage,
    validate_agent_trace,
)
from advoice.diagnostic_agent_report import (
    _fallback_report,
    _report_validation_errors,
    _sanitize_workspace,
)
from advoice.evidence import build_metric_evidence
from advoice.evidence import recalibrate_metric_evidence_frame


def test_small_sample_guard_rejects_unstable_agent_correction() -> None:
    labels = ["no_decline", "decline"]
    y = np.asarray(["no_decline", "decline"] * 4)
    base = np.asarray([[0.8, 0.2], [0.2, 0.8]] * 4)
    harmful = np.asarray([[0.8, 0.2]] * 8)
    splits = [
        (np.asarray([2, 3, 4, 5, 6, 7]), np.asarray([0, 1])),
        (np.asarray([0, 1, 4, 5, 6, 7]), np.asarray([2, 3])),
        (np.asarray([0, 1, 2, 3, 6, 7]), np.asarray([4, 5])),
        (np.asarray([0, 1, 2, 3, 4, 5]), np.asarray([6, 7])),
    ]
    audit = _small_sample_correction_guard(
        y,
        labels,
        splits,
        base,
        harmful,
        "macro_f1",
        {"enabled": True, "maximum_training_subjects": 80},
    )
    assert audit["triggered"] is True
    assert audit["action"] == "fall_back_to_base_probability"


def test_large_cohort_guard_rejects_oof_harmful_correction() -> None:
    labels = ["HC", "MCI"]
    y = np.asarray(["HC", "MCI"] * 50)
    base = np.asarray([[0.8, 0.2], [0.2, 0.8]] * 50)
    harmful = np.asarray([[0.8, 0.2]] * 100)
    indices = np.arange(100)
    splits = [
        (indices[20:], indices[:20]),
        (np.r_[indices[:20], indices[40:]], indices[20:40]),
        (np.r_[indices[:40], indices[60:]], indices[40:60]),
        (np.r_[indices[:60], indices[80:]], indices[60:80]),
        (indices[:80], indices[80:]),
    ]
    audit = _small_sample_correction_guard(
        y,
        labels,
        splits,
        base,
        harmful,
        "macro_f1",
        {"enabled": True, "maximum_training_subjects": 80},
    )
    assert audit["small_sample"] is False
    assert audit["triggered"] is True


def test_fold_offset_aggregation_rejects_conflicting_class_bias() -> None:
    selected, audit = _stable_fold_offsets(
        [
            [0.0, 0.0, 1.0],
            [0.0, 0.5, 1.0],
            [0.0, -1.0, -1.0],
            [0.0, -0.5, -1.0],
            [0.0, -1.0, 0.5],
        ]
    )
    assert selected.tolist() == [0.0, 0.0, 0.0]
    assert audit["classes"][2]["retained"] is False


def test_fold_offset_aggregation_retains_consistent_train_fold_direction() -> None:
    selected, audit = _stable_fold_offsets(
        [
            [0.0, -0.5],
            [0.0, -1.0],
            [0.0, -0.5],
            [0.0, -0.5],
            [0.0, 0.0],
        ]
    )
    assert selected.tolist() == [0.0, -0.5]
    assert audit["classes"][1]["retained"] is True


def test_train_only_expert_gate_rejects_non_discriminative_branch() -> None:
    assert _expert_passes_cv_gate({"0.1": 0.61, "1.0": 0.58}, 0.52)
    assert not _expert_passes_cv_gate({"0.1": 0.49, "1.0": 0.51}, 0.52)


def test_unsafe_report_is_rejected_before_publication() -> None:
    workspace = {
        "case_id": "CASE-1",
        "final_prediction": "AD",
        "final_probabilities": {"HC": 0.2, "AD": 0.8},
        "selected_supporting_evidence": [
            {"evidence_id": "E1", "metric_id": "pause", "value": 2.0, "reliability": 0.9}
        ],
        "selected_counterevidence": [],
        "quality_observations": [{"evidence_id": "Q1"}],
        "state_observations": [],
        "correction_gate": 0.7,
    }
    unsafe = {
        "predicted_label": "HC",
        "used_evidence_ids": ["Q1"],
        "counterevidence_ids": ["UNKNOWN"],
        "quality_evidence_ids": [],
        "report_zh": "该患者确诊为 AD，编号 AD_M_001。",
        "patient_summary_zh": "已经患有 AD。",
        "uncertainty_zh": "",
    }
    errors = _report_validation_errors(unsafe, workspace, "AD_M_001")
    assert "prediction_changed" in errors
    assert "unknown_evidence" in errors
    assert "diagnostic_role_error" in errors
    assert "source_identifier_leak" in errors
    assert "diagnostic_overclaim" in errors
    safe = _fallback_report(workspace, ["HC", "AD"])
    assert safe["patient_summary_zh"]
    assert "确诊为" not in safe["report_zh"]


def test_patient_summary_rejects_probability_and_missing_next_step() -> None:
    workspace = {
        "final_prediction": "AD",
        "selected_supporting_evidence": [{"evidence_id": "E1"}],
        "selected_counterevidence": [],
        "state_observations": [],
        "quality_observations": [],
    }
    item = {
        "predicted_label": "AD",
        "used_evidence_ids": ["E1"],
        "counterevidence_ids": [],
        "quality_evidence_ids": [],
        "report_zh": "研究性筛查结果。",
        "patient_summary_zh": "模型概率为80%。",
        "uncertainty_zh": "仍有不确定性。",
    }
    errors = _report_validation_errors(item, workspace, "source-case")
    assert "patient_audience_violation" in errors
    assert "patient_next_step_missing" in errors


def _cards() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "case-1",
                "state_id": "S01",
                "state_name_zh": "停顿与流畅性负担",
                "branch": "speech_behavior",
                "state_z": 2.1,
                "confidence": 0.9,
                "missing_fraction": 0.0,
                "task_scope": "overall",
                "report_state_z": 1.8,
                "report_confidence": 0.8,
                "report_permission": True,
                "supporting_metrics": "[]",
                "counter_evidence": "[]",
                "evidence_segments": "[]",
            }
        ]
    )


def _evidence() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "subject_id": "case-1",
                "metric_instance_id": "pause_p90_sec",
                "metric_id": "pause_p90_sec",
                "state_id": "S01",
                "branch": "speech_behavior",
                "task_scope": "overall",
                "directional_z": 2.0,
                "reliability": 0.9,
                "missing": False,
                "evidence_role": "clinical",
                "confound_tags": '["vad_threshold"]',
                "report_permission": True,
                "value": 1.3,
            },
            {
                "subject_id": "case-1",
                "metric_instance_id": "duration_sec",
                "metric_id": "duration_sec",
                "state_id": "QC",
                "branch": "qc",
                "task_scope": "overall",
                "directional_z": 4.0,
                "reliability": 1.0,
                "missing": False,
                "evidence_role": "qc_only",
                "confound_tags": '["task_duration"]',
                "report_permission": False,
                "value": 120.0,
            },
        ]
    )


def test_workspace_never_serializes_test_label() -> None:
    workspace = build_case_workspace(
        subject_id="case-1",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=_cards(),
        metric_evidence=_evidence(),
        class_support={"HC": -0.4, "AD": 0.4},
    )
    serialized = str(workspace)
    assert "ground_truth" not in serialized
    assert "true_label" not in serialized
    assert "label" not in workspace


def test_workspace_pseudonymizes_label_bearing_source_identifiers() -> None:
    cards = _cards()
    cards.loc[0, "subject_id"] = "AD_F_040021"
    cards.loc[0, "evidence_segments"] = json.dumps(
        [{"segment_id": "AD_F_040021_001:S01", "case_id": "AD_F_040021_001"}]
    )
    evidence = _evidence()
    evidence.loc[:, "subject_id"] = "AD_F_040021"
    workspace = build_case_workspace(
        subject_id="AD_F_040021",
        base_probabilities={"HC": 0.4, "AD": 0.6},
        state_cards=cards,
        metric_evidence=evidence,
        class_support={"HC": 0.4, "AD": 0.6},
    )
    serialized = json.dumps(workspace)
    assert "AD_F_040021" not in serialized
    assert workspace["case_id"].startswith("case_")
    segment = workspace["state_observations"][0]["evidence_segments"][0]
    assert segment["segment_id"].startswith("segment:SEG_")
    assert segment["case_id"].startswith("REC_")


def test_qc_only_evidence_cannot_support_disease_hypothesis() -> None:
    workspace = build_case_workspace(
        subject_id="case-1",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=_cards(),
        metric_evidence=_evidence(),
        class_support={"HC": -0.4, "AD": 0.4},
    )
    support_ids = {
        item["evidence_id"] for item in workspace["selected_supporting_evidence"]
    }
    assert "metric:duration_sec" not in support_ids
    assert "qc:duration_sec" in {
        item["evidence_id"] for item in workspace["quality_observations"]
    }


def test_quality_evidence_can_only_reduce_an_observable_state() -> None:
    workspace = {
        "state_observations": [
            {"evidence_id": "S01", "state_id": "S01", "task_scope": "overall"}
        ],
        "model_only_state_observations": [],
        "selected_supporting_evidence": [{"evidence_id": "pause_mean_sec"}],
        "selected_counterevidence": [],
        "quality_observations": [{"evidence_id": "snr_proxy_db"}],
    }
    candidate = {
        "action": "classify",
        "proposed_class": "AD",
        "proposed_probabilities": {"HC": 0.2, "AD": 0.8},
        "used_evidence_ids": ["pause_mean_sec"],
        "counterevidence_ids": [],
        "quality_evidence_ids": ["snr_proxy_db"],
        "state_updates": [
            {
                "state_id": "S01",
                "task_scope": "overall",
                "action": "downweight",
                "evidence_ids": ["pause_mean_sec", "snr_proxy_db"],
            }
        ],
        "counterevidence_checked": True,
    }

    assert validate_candidate(candidate, workspace, ["HC", "AD"])["valid"] is True


def test_unobservable_model_state_can_only_be_marked_unavailable() -> None:
    workspace = {
        "state_observations": [],
        "model_only_state_observations": [
            {"evidence_id": "S07", "state_id": "S07", "task_scope": "overall"}
        ],
        "selected_supporting_evidence": [],
        "selected_counterevidence": [],
        "quality_observations": [{"evidence_id": "transcript_available"}],
    }
    candidate = {
        "action": "hold_prior",
        "proposed_class": "HC",
        "proposed_probabilities": {"HC": 0.6, "AD": 0.4},
        "used_evidence_ids": [],
        "counterevidence_ids": [],
        "quality_evidence_ids": ["transcript_available"],
        "state_updates": [
            {
                "state_id": "S07",
                "task_scope": "overall",
                "action": "mark_unavailable",
                "evidence_ids": ["S07", "transcript_available"],
            }
        ],
        "counterevidence_checked": True,
    }

    assert validate_candidate(candidate, workspace, ["HC", "AD"])["valid"] is True


def test_model_only_state_is_not_exposed_as_reportable_observation() -> None:
    cards = _cards()
    cards.loc[0, "branch"] = "auxiliary_acoustic"
    cards.loc[0, "report_permission"] = False
    workspace = build_case_workspace(
        subject_id="case-1",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=cards,
        metric_evidence=_evidence(),
        class_support={"HC": -0.4, "AD": 0.4},
    )
    assert workspace["state_observations"] == []
    assert workspace["model_only_state_observations"][0]["state_id"] == "S01"


def test_language_auxiliary_state_is_available_to_inference_but_not_report() -> None:
    cards = _cards()
    cards.loc[0, "branch"] = "language"
    cards.loc[0, "report_permission"] = False
    evidence = _evidence()
    evidence.loc[0, "branch"] = "language"
    evidence.loc[0, "evidence_role"] = "model_auxiliary"
    evidence.loc[0, "report_permission"] = False
    workspace = build_case_workspace(
        subject_id="case-1",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=cards,
        metric_evidence=evidence,
        class_support={"HC": -0.4, "AD": 0.4},
    )
    inference_state = workspace["inference_only_state_observations"][0]
    inference_metric = workspace["inference_only_metric_observations"][0]
    assert inference_state["state_id"] == "S01"
    assert inference_metric["metric_id"] == "pause_p90_sec"
    clinical, _, _ = _allowed_ids(workspace)
    assert inference_state["evidence_id"] in clinical
    assert inference_metric["evidence_id"] in clinical

    cleaned = _sanitize_workspace(workspace, "case_public")
    assert cleaned["state_observations"] == []
    assert "inference_only_state_observations" not in cleaned
    assert "inference_only_metric_observations" not in cleaned
    assert inference_state["evidence_id"] not in {
        item["evidence_id"] for item in cleaned["evidence_registry"]
    }
    assert inference_metric["evidence_id"] not in {
        item["evidence_id"] for item in cleaned["evidence_registry"]
    }


def test_report_sanitization_removes_nested_inference_only_metric() -> None:
    workspace = {
        "state_observations": [
            {
                "evidence_id": "state:S01",
                "report_permission": True,
                "metric_evidence_ids": ["metric:reportable", "metric:model_only"],
                "supporting_metrics": [
                    {"metric_instance_id": "reportable", "value": 1.0},
                    {"metric_instance_id": "model_only", "value": 2.0},
                ],
                "counter_evidence": [],
                "evidence_segments": [],
            }
        ],
        "selected_supporting_evidence": [
            {"evidence_id": "metric:reportable", "report_permission": True}
        ],
        "selected_counterevidence": [],
        "inference_only_metric_observations": [
            {"evidence_id": "metric:model_only"}
        ],
        "quality_observations": [],
        "evidence_registry": [
            {"evidence_id": "state:S01", "evidence_type": "state"},
            {"evidence_id": "metric:reportable", "evidence_type": "metric"},
            {"evidence_id": "metric:model_only", "evidence_type": "metric"},
        ],
    }
    cleaned = _sanitize_workspace(workspace, "case_public")
    state = cleaned["state_observations"][0]
    assert state["metric_evidence_ids"] == ["metric:reportable"]
    assert [item["metric_instance_id"] for item in state["supporting_metrics"]] == [
        "reportable"
    ]
    assert "metric:model_only" not in {
        item["evidence_id"] for item in cleaned["evidence_registry"]
    }


def test_cognitive_prototype_distance_ignores_missing_values_without_nan() -> None:
    distance, usable = _robust_distance(
        np.asarray([[np.nan, 2.0], [1.0, np.nan]]),
        np.asarray([[0.0, 1.0], [1.0, 0.0]]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
        np.asarray([1.0, 1.0]),
    )
    assert np.isfinite(distance).all()
    assert usable.tolist() == [1, 1]


def test_task_specific_state_replaces_conflicting_overall_state_for_agent() -> None:
    cards = pd.concat([_cards(), _cards()], ignore_index=True)
    cards.loc[0, "state_id"] = "S01"
    cards.loc[0, "state_base_id"] = "S01"
    cards.loc[0, "task_scope"] = "overall"
    cards.loc[0, "state_z"] = -0.5
    cards.loc[1, "state_id"] = "S01__task_picture_description"
    cards.loc[1, "state_base_id"] = "S01"
    cards.loc[1, "task_scope"] = "picture_description"
    cards.loc[1, "state_z"] = 2.5
    workspace = build_case_workspace(
        subject_id="case-1",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=cards,
        metric_evidence=_evidence(),
        class_support={"HC": -0.4, "AD": 0.4},
    )
    states = workspace["state_observations"]
    assert len(states) == 1
    assert states[0]["task_scope"] == "picture_description"


def test_task_specific_metric_replaces_its_overall_duplicate() -> None:
    evidence = pd.concat([_evidence(), _evidence().iloc[[0]]], ignore_index=True)
    evidence.loc[0, "task_scope"] = "overall"
    evidence.loc[1, "task_scope"] = "overall"
    evidence.loc[2, "task_scope"] = "picture_description"
    evidence.loc[2, "metric_instance_id"] = "task_picture_description__pause_p90_sec"
    workspace = build_case_workspace(
        subject_id="case-1",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=_cards(),
        metric_evidence=evidence,
        class_support={"HC": -0.4, "AD": 0.4},
    )
    metric_ids = {
        item["evidence_id"] for item in workspace["selected_supporting_evidence"]
    }
    assert "metric:pause_p90_sec" not in metric_ids
    assert "metric:task_picture_description__pause_p90_sec" in metric_ids


def test_zero_gate_returns_base_probability_exactly() -> None:
    base = np.asarray([[0.2, 0.8], [0.6, 0.4]])
    corrected = np.asarray([[0.9, 0.1], [0.1, 0.9]])
    result = fuse_corrected_probability(
        base, corrected, np.zeros(2), alpha=1.0
    )
    assert np.allclose(result, base)


def test_evidence_gate_penalizes_missingness_and_confounds() -> None:
    clean = evidence_gate(coverage=1.0, reliability=0.9, confound_burden=0.0)
    contaminated = evidence_gate(
        coverage=0.5, reliability=0.9, confound_burden=0.8
    )
    assert 0.0 <= contaminated < clean <= 1.0


def test_structured_coverage_does_not_fall_when_duplicate_metrics_are_added() -> None:
    states = [
        {"state_id": "S01", "task_scope": "picture", "branch": "speech", "category": "normal"},
        {"state_id": "S07", "task_scope": "picture", "branch": "language", "category": "normal"},
    ]
    observed = [
        {**item, "confidence": 0.8, "missing_fraction": 0.0}
        for item in states
    ]
    baseline = structured_evidence_coverage(states, observed)
    duplicated = structured_evidence_coverage(states * 20, observed)
    assert baseline == duplicated
    assert baseline["overall"] == 1.0


def test_blind_workspace_hides_supervised_and_corrected_probabilities() -> None:
    workspace = {
        "case_id": "CASE-1",
        "base_probabilities": {"HC": 0.8, "AD": 0.2},
        "corrected_probabilities": {"HC": 0.7, "AD": 0.3},
        "final_prediction": "HC",
        "class_support": {"HC": 0.6, "AD": 0.4},
        "state_observations": [{"state_id": "S01"}],
    }
    blinded = blind_workspace(workspace)
    assert blinded == {"case_id": "CASE-1", "state_observations": [{"state_id": "S01"}]}


def test_state_visible_metric_ids_are_valid_agent_references() -> None:
    workspace = {
        "state_observations": [
            {
                "evidence_id": "S01",
                "metric_evidence_ids": ["pause_mean", "task_cookie__pause_mean"],
                "evidence_segments": [],
            }
        ],
        "selected_supporting_evidence": [],
        "selected_counterevidence": [],
        "quality_observations": [],
    }
    clinical, _, _ = _allowed_ids(workspace)
    assert {"S01", "pause_mean", "task_cookie__pause_mean"}.issubset(clinical)


def test_agent_references_must_resolve_through_registry_when_present() -> None:
    workspace = {
        "state_observations": [
            {
                "evidence_id": "state:S01",
                "metric_evidence_ids": ["metric:registered", "metric:stale_nested"],
                "evidence_segments": [],
            }
        ],
        "selected_supporting_evidence": [],
        "selected_counterevidence": [],
        "quality_observations": [],
        "evidence_registry": [
            {"evidence_id": "state:S01", "evidence_type": "state"},
            {"evidence_id": "metric:registered", "evidence_type": "metric"},
        ],
    }
    clinical, _, _ = _allowed_ids(workspace)
    assert "metric:registered" in clinical
    assert "metric:stale_nested" not in clinical


def test_discrete_evidence_scores_are_converted_without_a_supervised_prior() -> None:
    workspace = {
        "state_observations": [
            {
                "evidence_id": "state:S01",
                "state_id": "S01",
                "task_scope": "overall",
                "metric_evidence_ids": ["metric:pause_mean"],
                "evidence_segments": [],
            }
        ],
        "model_only_state_observations": [],
        "selected_supporting_evidence": [{"evidence_id": "metric:pause_mean"}],
        "selected_counterevidence": [],
        "quality_observations": [],
    }
    candidate = {
        "action": "classify",
        "evidence_class": "AD",
        "evidence_scores": {"HC": 1, "AD": 4},
        "used_evidence_ids": ["metric:pause_mean"],
        "counterevidence_ids": [],
        "quality_evidence_ids": [],
        "state_updates": [],
        "counterevidence_checked": True,
    }
    audit = validate_candidate(candidate, workspace, ["HC", "AD"])
    assert audit["valid"] is True
    assert audit["normalized_evidence_likelihoods"]["AD"] > 0.9


def test_two_stage_insufficient_staging_is_forced_to_neutral() -> None:
    workspace = {
        "state_observations": [
            {"evidence_id": "state:S01", "state_id": "S01", "task_scope": "overall"}
        ],
        "model_only_state_observations": [],
        "selected_supporting_evidence": [{"evidence_id": "metric:pause_mean"}],
        "selected_counterevidence": [],
        "quality_observations": [],
    }
    candidate = {
        "action": "classify",
        "screening_class": "impaired",
        "screening_scores": {"HC": 1, "impaired": 3},
        "staging_action": "insufficient",
        "staging_class": "undetermined",
        "staging_scores": {"MCI": 2, "AD": 2},
        "used_evidence_ids": ["metric:pause_mean"],
        "counterevidence_ids": [],
        "quality_evidence_ids": [],
        "state_updates": [],
        "counterevidence_checked": True,
    }
    audit = validate_candidate(candidate, workspace, ["HC", "MCI", "AD"])
    assert audit["valid"] is True
    assert audit["staging_available"] is False
    assert audit["normalized_staging_likelihoods"] == {"MCI": 0.5, "AD": 0.5}


def test_binary_evidence_calibrator_uses_oof_and_frozen_transform() -> None:
    truth = np.asarray([0, 1] * 10)
    likelihood = np.asarray(
        [[0.8, 0.2] if label == 0 else [0.2, 0.8] for label in truth]
    )
    calibrator = fit_binary_evidence_calibrator(
        truth, likelihood, np.ones(len(truth), dtype=bool), maximum_folds=5
    )
    assert calibrator["status"] == "calibrated_on_development_oof"
    assert calibrator["oof_likelihood"].shape == (20, 2)
    transformed = apply_binary_evidence_calibrator(likelihood, calibrator)
    assert np.all(transformed[truth == 1, 1] > 0.5)
    assert np.all(transformed[truth == 0, 0] > 0.5)


def test_two_stage_route_parameters_are_shared_by_calibration_and_inference() -> None:
    result = two_stage_route_parameters(
        np.asarray([0.45, 0.35, 0.20]),
        np.asarray([0.25, 0.75]),
        np.asarray([0.7, 0.3]),
        gate=0.8,
        staging_available=True,
    )
    assert result["staging_gate"] == 0.8
    assert result["screening_multiplier"] == result["screening_route"]["route_multiplier"]
    assert result["staging_multiplier"] == result["staging_route"]["route_multiplier"]


def test_fold_metric_recalibration_excludes_target_subject() -> None:
    evidence = pd.DataFrame(
        {
            "subject_id": ["r1", "r2", "target"],
            "language": ["english", "english", "english"],
            "metric_id": ["pause_mean_sec"] * 3,
            "metric_instance_id": ["pause_mean_sec"] * 3,
            "task_scope": ["overall"] * 3,
            "state_id": ["S01"] * 3,
            "value": [0.0, 2.0, 100.0],
            "reference_scope": ["old"] * 3,
            "reference_median": [0.0] * 3,
            "reference_scale": [1.0] * 3,
            "cn_train_median": [0.0] * 3,
            "cn_train_scale": [1.0] * 3,
            "robust_z": [0.0, 2.0, 100.0],
            "direction": [1] * 3,
            "directional_z": [0.0, 2.0, 100.0],
            "reliability": [0.8] * 3,
            "missing": [False] * 3,
        }
    )
    result = recalibrate_metric_evidence_frame(
        evidence,
        reference_subject_ids={"r1", "r2"},
        target_subject_ids={"target"},
    )
    assert result.loc[0, "reference_median"] == 1.0
    assert result.loc[0, "reference_scope"] == "outer_fit_hc_reference"
    assert result.loc[0, "robust_z"] < 100.0


def test_fold_card_refresh_keeps_string_false_missing_rows() -> None:
    cards = pd.DataFrame(
        {
            "subject_id": ["target"],
            "state_id": ["S01"],
            "state_base_id": ["S01"],
            "task_scope": ["overall"],
            "supporting_metrics": ["[]"],
            "counter_evidence": ["[]"],
        }
    )
    evidence = pd.DataFrame(
        {
            "subject_id": ["target"],
            "state_id": ["S01"],
            "task_scope": ["overall"],
            "metric_id": ["pause_mean_sec"],
            "metric_instance_id": ["pause_mean_sec"],
            "value": [1.5],
            "reference_label": ["HC"],
            "reference_scope": ["outer_fit_hc_reference"],
            "reference_median": [0.5],
            "reference_scale": [0.5],
            "cn_train_median": [0.5],
            "cn_train_scale": [0.5],
            "robust_z": [2.0],
            "directional_z": [2.0],
            "reliability": [0.8],
            "report_permission": [True],
            "confound_tags": ["[]"],
            "missing": ["False"],
        }
    )
    refreshed = _replace_card_metric_summaries(cards, evidence)
    supporting = json.loads(refreshed.loc[0, "supporting_metrics"])
    assert supporting[0]["metric_id"] == "pause_mean_sec"


def test_task_scoped_state_update_resolves_base_state_identifier() -> None:
    workspace = {
        "state_observations": [],
        "model_only_state_observations": [
            {
                "evidence_id": "state:S07__task_sentence_reading",
                "state_id": "S07__task_sentence_reading",
                "task_scope": "sentence_reading",
                "metric_evidence_ids": [],
                "evidence_segments": [],
            }
        ],
        "selected_supporting_evidence": [],
        "selected_counterevidence": [],
        "quality_observations": [],
    }
    candidate = {
        "action": "retest",
        "evidence_class": "HC",
        "evidence_scores": {"HC": 1, "AD": 0},
        "used_evidence_ids": [],
        "counterevidence_ids": [],
        "quality_evidence_ids": [],
        "state_updates": [
            {
                "state_id": "S07",
                "task_scope": "sentence_reading",
                "action": "mark_unavailable",
                "reason": "Model-only state.",
                "evidence_ids": ["state:S07__task_sentence_reading"],
            }
        ],
        "counterevidence_checked": True,
    }
    audit = validate_candidate(candidate, workspace, ["HC", "AD"])
    assert audit["valid"] is True


def test_state_updates_can_only_reduce_agent_permission() -> None:
    workspace = {
        "state_observations": [
            {
                "state_id": "S01",
                "task_scope": "overall",
                "state_z": 2.0,
                "confidence": 0.9,
                "missing_fraction": 0.0,
            },
            {
                "state_id": "S07__task_picture_description",
                "task_scope": "picture_description",
                "state_z": 1.0,
                "confidence": 0.8,
                "missing_fraction": 0.0,
            },
        ]
    }
    keep = {"state_updates": []}
    downweight = {
        "state_updates": [
            {
                "state_id": "S01",
                "task_scope": "overall",
                "action": "downweight",
            }
        ]
    }
    invalidate = {
        "state_updates": [
            {
                "state_id": "S01",
                "task_scope": "overall",
                "action": "invalidate",
            }
        ]
    }
    assert _state_update_factor(keep, workspace) == 1.0
    assert 0.0 < _state_update_factor(downweight, workspace) < 1.0
    assert _state_update_factor(invalidate, workspace) < _state_update_factor(
        downweight, workspace
    )


def test_case_router_protects_stable_agreement_and_escalates_conflict() -> None:
    stable = route_case([0.9, 0.07, 0.03], [0.8, 0.1, 0.1], 0.9)
    conflict = route_case([0.85, 0.1, 0.05], [0.1, 0.2, 0.7], 0.9)
    limited = route_case([0.4, 0.35, 0.25], [0.2, 0.5, 0.3], 0.1)
    assert stable["route"] == "stable_agreement"
    assert stable["route_multiplier"] < 1.0
    assert conflict["route"] == "model_evidence_conflict"
    assert conflict["route_multiplier"] == 1.0
    assert limited["route"] == "quality_limited"
    assert limited["route_multiplier"] == 0.0


def test_case_router_treats_uniform_multiclass_evidence_as_neutral() -> None:
    route = route_case(
        [0.85, 0.10, 0.05],
        [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0],
        0.9,
        hierarchical_reference_index=0,
    )
    assert route["prior_evidence_agreement"] is True
    assert route["route"] == "stable_agreement"


def test_input_router_uses_shared_multitask_parent_families() -> None:
    route = case_input_route(
        ["english", "spanish"],
        ["picture_description", "animal_fluency", "delayed_recall"],
        ["language", "speech_behavior"],
    )
    assert route["task_structure"] == "multitask"
    assert route["language_scope"] == "multilingual"
    assert route["task_families"] == [
        "memory_recall",
        "picture_description",
        "verbal_fluency",
    ]
    assert route["fallback_policy"] == "shared_model_then_parent_task_family"


def test_evidence_likelihood_updates_prior_without_replacing_it() -> None:
    base = np.asarray([[0.7, 0.2, 0.1]])
    evidence = np.asarray([[0.1, 0.2, 0.7]])
    unchanged = fuse_evidence_likelihood(base, evidence, np.asarray([1.0]), 0.0)
    updated = fuse_evidence_likelihood(base, evidence, np.asarray([1.0]), 1.0)
    assert np.allclose(unchanged, base)
    assert updated[0, 2] > base[0, 2]
    assert np.allclose(updated.sum(axis=1), 1.0)


def test_hierarchical_agent_correction_preserves_mci_ad_ratio() -> None:
    base = np.asarray([[0.6, 0.1, 0.3]])
    evidence = np.asarray([[0.1, 0.8, 0.1]])
    updated = fuse_evidence_likelihood(
        base,
        evidence,
        np.asarray([1.0]),
        1.0,
        hierarchical_reference_index=0,
    )
    assert updated[0, 0] < base[0, 0]
    assert np.isclose(updated[0, 1] / updated[0, 2], base[0, 1] / base[0, 2])


def test_hierarchical_uniform_evidence_is_neutral() -> None:
    base = np.asarray([[0.7, 0.2, 0.1], [0.2, 0.3, 0.5]])
    evidence = np.full((2, 3), 1.0 / 3.0)
    updated = fuse_evidence_likelihood(
        base,
        evidence,
        np.ones(2),
        1.0,
        hierarchical_reference_index=0,
    )
    assert np.allclose(updated, base)


def test_two_stage_agent_updates_screening_and_staging_separately() -> None:
    base = np.asarray([[0.60, 0.30, 0.10]])
    screening = np.asarray([[0.20, 0.80]])
    staging = np.asarray([[0.10, 0.90]])
    screening_only = fuse_two_stage_evidence(
        base, screening, staging, np.ones(1), np.zeros(1), 1.0, 1.0
    )
    both = fuse_two_stage_evidence(
        base, screening, staging, np.ones(1), np.ones(1), 1.0, 1.0
    )
    assert screening_only[0, 0] < base[0, 0]
    assert np.isclose(screening_only[0, 1] / screening_only[0, 2], 3.0)
    assert both[0, 2] > screening_only[0, 2]


def test_two_stage_uniform_evidence_is_neutral() -> None:
    base = np.asarray([[0.65, 0.15, 0.20], [0.2, 0.3, 0.5]])
    neutral = np.full((2, 2), 0.5)
    updated = fuse_two_stage_evidence(
        base, neutral, neutral, np.ones(2), np.ones(2), 1.0, 1.0
    )
    assert np.allclose(updated, base)


def test_agent_strength_is_selected_only_from_development_labels() -> None:
    labels = ["HC", "MCI", "AD"]
    truth = np.asarray(["HC", "MCI", "AD", "HC", "MCI", "AD"])
    base = np.full((6, 3), 1.0 / 3.0)
    evidence = np.asarray(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]] * 2
    )
    result = fit_agent_correction_strength(
        truth,
        labels,
        base,
        evidence,
        np.ones(6),
        np.ones(6),
        [0.0, 0.5, 1.0],
    )
    assert result["selected_strength"] > 0.0
    assert all(np.isfinite(row["macro_auroc"]) for row in result["candidates"])


def test_agent_strength_fails_closed_when_f1_gain_costs_too_much_auroc() -> None:
    rows = [
        {"strength": 0.0, "macro_f1": 0.60, "macro_auroc": 0.82},
        {"strength": 0.25, "macro_f1": 0.63, "macro_auroc": 0.80},
    ]
    selected = _select_correction_candidate(
        rows,
        minimum_macro_f1_gain=0.0,
        auroc_noninferiority_margin=0.001,
    )
    assert selected["strength"] == 0.0
    assert selected["selection_status"] == "failed_closed_no_joint_gain"


def test_agent_strength_accepts_joint_validation_gain() -> None:
    rows = [
        {"strength": 0.0, "macro_f1": 0.60, "macro_auroc": 0.82},
        {"strength": 0.25, "macro_f1": 0.63, "macro_auroc": 0.821},
    ]
    selected = _select_correction_candidate(
        rows,
        minimum_macro_f1_gain=0.005,
        auroc_noninferiority_margin=0.001,
    )
    assert selected["strength"] == 0.25
    assert selected["selection_status"] == "validated_joint_gain"


def test_held_out_agent_runs_only_after_development_gate_passes() -> None:
    passed = {"status": "completed", "selection_status": "validated_joint_gain"}
    failed = {"status": "completed", "selection_status": "failed_closed_no_joint_gain"}
    missing = {"status": "not_available", "selection_status": "not_available"}
    assert not _test_agent_gate_passed("disabled", 1.0, 1.0, passed)
    assert not _test_agent_gate_passed("openai_api", 0.0, 0.0, passed)
    assert not _test_agent_gate_passed("openai_api", 0.25, 0.0, failed)
    assert not _test_agent_gate_passed("openai_api", 0.25, 0.0, missing)
    assert _test_agent_gate_passed("openai_api", 0.25, 0.0, passed)
    assert _test_agent_gate_passed("openai_api", 0.0, 0.25, passed)


def test_two_stage_strengths_can_select_screening_without_staging() -> None:
    truth = np.asarray(["HC", "HC", "MCI", "AD"])
    base = np.asarray(
        [[0.45, 0.35, 0.20], [0.45, 0.30, 0.25], [0.60, 0.30, 0.10], [0.60, 0.10, 0.30]]
    )
    screening = np.asarray([[0.8, 0.2], [0.8, 0.2], [0.2, 0.8], [0.2, 0.8]])
    staging = np.full((4, 2), 0.5)
    result = fit_agent_two_stage_strengths(
        truth,
        base,
        screening,
        staging,
        np.ones(4),
        np.zeros(4),
        np.ones(4),
        np.ones(4),
        [0.0, 1.0],
    )
    assert result["selected_screening_strength"] == 1.0
    assert result["selected_staging_strength"] == 0.0


def test_agent_batch_cache_accepts_reordered_subset_for_targeted_retry(tmp_path) -> None:
    output = tmp_path / "batch.json"
    metadata = tmp_path / "batch.meta.json"
    output.write_text(
        '{"cases":[{"case_id":"case_b"},{"case_id":"case_a"}]}',
        encoding="utf-8",
    )
    metadata.write_text('{"fingerprint":"fp"}', encoding="utf-8")
    cached = _cached_response(
        output, metadata, "fp", ["case_a", "case_b", "case_c"]
    )
    assert cached is not None


def test_agent_response_index_rejects_unrequested_case_ids() -> None:
    indexed, unexpected = _index_expected_candidates(
        [{"case_id": "case_a"}, {"case_id": "stale_case"}], {"case_a"}
    )
    assert set(indexed) == {"case_a"}
    assert unexpected == ["stale_case"]


def test_trace_rejects_unknown_evidence_identifier() -> None:
    valid_ids = {"pause_p90_sec", "S01"}
    trace = [
        {"action": "inspect_metric", "evidence_id": "pause_p90_sec"},
        {"action": "inspect_metric", "evidence_id": "invented_metric"},
    ]
    result = validate_agent_trace(trace, valid_ids)
    assert result["valid"] is False
    assert result["unknown_evidence_ids"] == ["invented_metric"]


def test_case_level_logistic_expert_contributions_are_normalized() -> None:
    values = np.asarray(
        [
            [-0.2, -1.2, 0.8, 0.2, 0.9, 0.7],
            [-1.4, -0.1, 0.1, 1.1, 0.8, 0.6],
            [-0.8, -0.7, 0.5, 0.5, 0.7, 0.8],
            [-0.1, -1.5, 1.2, 0.1, 0.9, 0.5],
        ]
    )
    labels = ["AD", "CN"]
    target = np.asarray(["AD", "CN", "AD", "CN"])
    model = _numeric_pipeline(1.0, 500)
    model.fit(values, target)
    probability = model.predict_proba(values)
    ordered = probability[:, [list(model.named_steps["classifier"].classes_).index(label) for label in labels]]
    names = [
        "log_probability__language__AD",
        "log_probability__language__CN",
        "log_probability__audio__AD",
        "log_probability__audio__CN",
        "reliability__language",
        "reliability__audio",
    ]
    contribution = _case_level_expert_contributions(
        model, values, names, ["language", "audio"], labels, ordered
    )
    assert contribution.shape == (4, 2)
    assert np.all(contribution >= 0.0)
    assert np.allclose(contribution.sum(axis=1), 1.0)


def test_task_specific_views_do_not_multiply_a_state_vote() -> None:
    columns = [
        "state_S01",
        "state_S01__task_cookie",
        "state_S01__task_recall",
        "state_S07",
    ]
    weights = _hierarchical_state_weights(
        columns,
        pd.Series({column: 1.0 for column in columns}),
    )
    assert np.isclose(weights[:3].sum(), 0.5)
    assert np.isclose(weights[3], 0.5)


def test_workspace_trace_inspects_segment_and_compares_tasks() -> None:
    cards = pd.concat([_cards(), _cards()], ignore_index=True)
    cards.loc[0, "task_scope"] = "cookie"
    cards.loc[0, "supporting_metrics"] = (
        '[{"metric_id":"pause_mean_sec","metric_instance_id":"task_cookie__pause_mean_sec"}]'
    )
    cards.loc[0, "evidence_segments"] = '[{"segment_id":"S1"}]'
    cards.loc[1, "task_scope"] = "recall"
    cards.loc[1, "evidence_segments"] = '[{"segment_id":"S2"}]'
    workspace = build_case_workspace(
        subject_id="case-1",
        base_probabilities={"HC": 0.3, "AD": 0.7},
        state_cards=cards,
        metric_evidence=_evidence(),
        class_support={"HC": -0.4, "AD": 0.4},
    )
    actions = [item["action"] for item in workspace["precomputed_review_plan"]]
    assert "inspect_segment" in actions
    assert "compare_tasks" in actions
    assert workspace["state_observations"][0]["metric_evidence_ids"]


def test_transcript_cleaner_removes_parser_residue() -> None:
    cleaned = clean_model_transcript(
        "[录音1] the boy takes cookies n|cookie-PL 9|8|OBJ and the sink overflows"
    )
    assert cleaned == "the boy takes cookies and the sink overflows"


def test_age_context_uses_fixed_categories_and_marks_missingness() -> None:
    frame = pd.DataFrame({"age": [65, 66, 80, 81, np.nan]})
    columns = _add_age_band_features(frame, {"age_bins": [0, 66, 81, 200]})
    assert columns == [
        "age_band_0",
        "age_band_1",
        "age_band_2",
        "age_band_missing",
    ]
    assert frame.loc[0, "age_band_0"] == 1.0
    assert frame.loc[1, "age_band_1"] == 1.0
    assert frame.loc[4, "age_band_missing"] == 1.0


def test_class_offsets_are_probability_preserving() -> None:
    probability = np.asarray([[0.2, 0.3, 0.5], [0.6, 0.2, 0.2]])
    adjusted = _apply_logit_offsets(probability, np.asarray([0.0, 0.5, -0.5]))
    assert np.allclose(adjusted.sum(axis=1), 1.0)
    assert adjusted[0, 1] > probability[0, 1]


def test_patient_intervals_and_overlapping_windows_are_deterministic() -> None:
    intervals = _parse_intervals(
        '[{"role":"interviewer","start":0,"end":1},'
        '{"role":"participant","start":1,"end":3}]'
    )
    assert intervals == [(1.0, 3.0)]
    windows = _windows(np.arange(12, dtype=np.float32), 1, 5.0, 0.25)
    assert [len(window) for window in windows] == [5, 5, 5]
    assert windows[-1][-1] == 11


def test_condition_c_trains_two_modules_and_writes_label_free_workspaces(
    tmp_path,
) -> None:
    feature_rows = []
    transcript_rows = []
    for index in range(30):
        label = "HC" if index % 2 == 0 else "AD"
        split = "train" if index < 24 else "test"
        signal = -2.0 if label == "HC" else 2.0
        feature_rows.append(
            {
                "dataset_id": "synthetic",
                "subject_id": str(index),
                "label": label,
                "split": split,
                "pause_metric": signal + (index % 3) * 0.05,
                "audio_reliability": 0.95,
                "text_reliability": 0.95,
                "original_duration_sec": 30.0,
            }
        )
        transcript_rows.append(
            {
                "subject_id": str(index),
                "transcript": (
                    "clear complete picture description"
                    if label == "HC"
                    else "um vague thing repeated hesitation"
                ),
            }
        )
    features = pd.DataFrame(feature_rows)
    features_path = tmp_path / "features.csv"
    transcripts_path = tmp_path / "transcripts.csv"
    evidence_path = tmp_path / "evidence.csv"
    reference_path = tmp_path / "reference.json"
    states_path = tmp_path / "states.csv"
    cards_path = tmp_path / "cards.csv"
    features.to_csv(features_path, index=False)
    pd.DataFrame(transcript_rows).to_csv(transcripts_path, index=False)
    build_metric_evidence(
        features_path,
        {
            "metrics": [
                {
                    "id": "pause_metric",
                    "state": "S01",
                    "branch": "speech_behavior",
                    "direction": 1,
                    "role": "clinical",
                    "reliability": 1.0,
                    "confounds": ["vad_threshold"],
                    "report_permission": True,
                }
            ]
        },
        evidence_path,
        reference_path,
        reference_label="HC",
    )
    pd.DataFrame(
        [
            {
                "dataset_id": row["dataset_id"],
                "subject_id": row["subject_id"],
                "label": row["label"],
                "split": row["split"],
                "state_S01": row["pause_metric"],
                "rel_S01": 0.95,
            }
            for row in feature_rows
        ]
    ).to_csv(states_path, index=False)
    pd.DataFrame(
        [
            {
                "dataset_id": row["dataset_id"],
                "subject_id": row["subject_id"],
                "label": row["label"],
                "split": row["split"],
                "state_id": "S01",
                "state_base_id": "S01",
                "task_scope": "overall",
                "state_name_zh": "停顿与流畅性负担",
                "branch": "speech_behavior",
                "state_z": row["pause_metric"],
                "confidence": 0.95,
                "missing_fraction": 0.0,
                "supporting_metrics": "[]",
                "counter_evidence": "[]",
                "evidence_segments": "[]",
            }
            for row in feature_rows
        ]
    ).to_csv(cards_path, index=False)
    states_config = {
        "states": [
            {
                "id": "S01",
                "name_zh": "停顿与流畅性负担",
                "branch": "speech_behavior",
                "metrics": ["pause_metric"],
                "weights": [1.0],
            }
        ]
    }
    model_config = {
        "labels": ["HC", "AD"],
        "positive_class": "AD",
        "state_branches": {"S01": "speech_behavior"},
        "cross_validation": {"folds": 3},
        "ours": {"qc_orthogonalization": {"enabled": False}},
        "condition_c": {
            "c_grid": [0.1, 1.0],
            "alpha_grid": [0.0, 0.5, 1.0],
            "max_iter": 1000,
        },
    }
    outputs = {
        name: tmp_path / filename
        for name, filename in {
            "predictions": "predictions.csv",
            "base": "base.csv",
            "ablations": "ablations.csv",
            "interventions": "interventions.csv",
            "workspaces": "workspaces.jsonl",
            "contributions": "contributions.csv",
            "model": "model.joblib",
            "metadata": "metadata.json",
        }.items()
    }
    train_condition_c(
        features_path,
        transcripts_path,
        states_path,
        evidence_path,
        cards_path,
        states_config,
        model_config,
        outputs["predictions"],
        outputs["base"],
        outputs["ablations"],
        outputs["interventions"],
        outputs["workspaces"],
        outputs["contributions"],
        outputs["model"],
        outputs["metadata"],
    )
    predictions = pd.read_csv(outputs["predictions"])
    metadata = pd.read_json(outputs["metadata"], typ="series")
    workspace_lines = outputs["workspaces"].read_text(encoding="utf-8").splitlines()
    assert len(predictions) == 6
    assert predictions["condition"].eq("Ours").all()
    assert metadata["test_labels_used_by_agent"] is False
    assert metadata["selection_protocol"] == "strict_outer_fold_nested_refit"
    assert len(metadata["outer_fold_selection"]["base_c"]) == 3
    assert metadata["final_probability_temperature"] > 0.0
    assert metadata["supervised_modules"] == [
        "multibranch_base_predictor",
        "bounded_risk_correction_and_temperature_calibration",
    ]
    assert all("true_label" not in line and "ground_truth" not in line for line in workspace_lines)

    # Held-out labels may be retained for scoring, but changing them must not change inference.
    first_probability = predictions[["prob_HC", "prob_AD"]].to_numpy()
    for path in [features_path, evidence_path, states_path, cards_path]:
        frame_to_flip = pd.read_csv(path, dtype={"subject_id": str})
        test_mask = frame_to_flip["split"].eq("test")
        frame_to_flip.loc[test_mask, "label"] = frame_to_flip.loc[
            test_mask, "label"
        ].map({"HC": "AD", "AD": "HC"})
        frame_to_flip.to_csv(path, index=False)
    second_outputs = {
        name: tmp_path / f"second_{path.name}" for name, path in outputs.items()
    }
    train_condition_c(
        features_path,
        transcripts_path,
        states_path,
        evidence_path,
        cards_path,
        states_config,
        model_config,
        second_outputs["predictions"],
        second_outputs["base"],
        second_outputs["ablations"],
        second_outputs["interventions"],
        second_outputs["workspaces"],
        second_outputs["contributions"],
        second_outputs["model"],
        second_outputs["metadata"],
    )
    second_probability = pd.read_csv(second_outputs["predictions"])[
        ["prob_HC", "prob_AD"]
    ].to_numpy()
    assert np.allclose(first_probability, second_probability)
