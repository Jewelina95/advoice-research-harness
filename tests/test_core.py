from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from advoice import asr
from advoice.cache import StageCache
from advoice.config import ProjectPaths
from advoice.data import (
    NCMMSC_NAME,
    _balanced_acquisition_group_holdout,
    build_adresso_manifest,
    build_pitt_manifest,
)
from advoice import direct_agent
from advoice.direct_agent import (
    _cached_batch,
    _canonicalize_case_ids,
    _run_or_load_batch,
)
from advoice.evaluation import expected_calibration_error
from advoice.dynamic_gate import fit_dynamic_reliability_gate
from advoice.evidence import _robust_reference as evidence_robust_reference
from advoice.evidence import build_metric_evidence
from advoice.features import (
    _voice_activity_mask,
    aggregate_subject_features,
    extract_features,
)
from advoice.models import (
    QCOrthogonalizer,
    _apply_caps,
    _branch_specifications,
    _paired_gain_summary,
    train_ours,
)
from advoice.pipeline import _PROCESSED_REUSE_FILES
from advoice.states import (
    _bounded_metric_contribution,
    _normalize_evidence_schema,
    _robust_reference as state_robust_reference,
    _state_category,
    build_fold_calibrated_state_frame,
    build_state_cards,
)


def test_constant_reference_is_unavailable_in_evidence_and_state_calibration() -> None:
    values = pd.Series([0.0, 0.0, 0.0, 0.0])
    evidence_median, evidence_scale, evidence_available = evidence_robust_reference(values)
    state_median, state_scale, state_available = state_robust_reference(values)
    assert evidence_median == state_median == 0.0
    assert evidence_scale == state_scale == 1.0
    assert evidence_available is False
    assert state_available is False


def test_voice_activity_mask_preserves_a_long_silence() -> None:
    sample_rate = 16000
    time = np.arange(sample_rate, dtype=float) / sample_rate
    speech_like = (0.35 * np.sin(2.0 * np.pi * 180.0 * time)).astype(np.float32)
    audio = np.concatenate(
        [speech_like, np.zeros(sample_rate, dtype=np.float32), speech_like]
    )
    voiced, backend = _voice_activity_mask(audio, sample_rate)
    middle = voiced[len(voiced) // 3 : 2 * len(voiced) // 3]
    edges = np.concatenate([voiced[: len(voiced) // 3], voiced[2 * len(voiced) // 3 :]])
    assert middle.mean() < 0.10
    assert edges.mean() > 0.50
    assert backend in {"energy", "webrtc_energy_hybrid"}
from advoice.report_agent import _retained_state_features, _sanitized_segments
from advoice.reporting import _expert_cv_values, plot_branch_weights
from advoice.agent_runtime import select_agent_cohort
from advoice.aggregate_reporting import _evaluation_payload
from advoice.condition_c import _text_pipeline
from advoice.config import load_all
from advoice.failure_analysis import _prediction_metrics
from advoice.transcripts import _tokens, canonical_language, transcript_metrics


def test_ncmmsc_filename_contract() -> None:
    match = NCMMSC_NAME.match("AD_F_040349_003")
    assert match is not None
    assert match.groups() == ("AD", "F", "040349", "003")
    assert NCMMSC_NAME.match("MCI_M_061901-002") is not None
    assert NCMMSC_NAME.match("001") is None


def test_chinese_tokenization_uses_words_not_single_characters() -> None:
    tokens = _tokens("患者正在描述厨房里的男孩和母亲", "zh")
    assert "患者" in tokens
    assert "厨房" in tokens
    assert len(tokens) < len("患者正在描述厨房里的男孩和母亲")


def test_picture_description_metrics_are_task_scoped(tmp_path: Path) -> None:
    transcript = tmp_path / "cookie.txt"
    transcript.write_text(
        "The boy reaches for cookies while the mother dries dishes and the sink overflows.",
        encoding="utf-8",
    )
    picture = transcript_metrics(
        str(transcript), "english", 10.0, "picture_description"
    )
    narrative = transcript_metrics(
        str(transcript), "english", 10.0, "personal_narrative"
    )
    assert picture["picture_content_unit_coverage"] > 0.30
    assert picture["picture_information_density"] > 0.0
    assert np.isnan(narrative["picture_content_unit_coverage"])


def test_picture_description_metric_aliases_are_supported(tmp_path: Path) -> None:
    transcript = tmp_path / "picture.txt"
    transcript.write_text(
        "The boy takes a cookie while the mother washes dishes at the sink.",
        encoding="utf-8",
    )

    cookie = transcript_metrics(
        str(transcript), "english", 10.0, "cookie_theft_picture_description"
    )
    process_ctd = transcript_metrics(str(transcript), "english", 10.0, "ctd")

    assert cookie["picture_content_unit_coverage"] > 0
    assert process_ctd["picture_content_unit_coverage"] > 0


def test_acquisition_group_holdout_keeps_interviewer_out_of_training() -> None:
    frame = pd.DataFrame(
        {
            "subject_id": ["a1", "a2", "b1", "b2", "b3", "c1", "c2"],
            "label": ["AD", "HC", "AD", "AD", "HC", "HC", "HC"],
            "acquisition_group": ["inv01", "inv01", "inv02", "inv02", "inv02", "inv03", "inv03"],
        }
    )
    split = _balanced_acquisition_group_holdout(frame)
    assert set(split.loc[split["split"].eq("test"), "acquisition_group"]) == {"inv01"}
    assert not (
        set(split.loc[split["split"].eq("train"), "acquisition_group"])
        & set(split.loc[split["split"].eq("test"), "acquisition_group"])
    )


def test_subject_aggregation_preserves_bilingual_evidence_scopes(tmp_path: Path) -> None:
    recording = pd.DataFrame(
        [
            {
                "dataset_id": "demo", "subject_id": "s1", "label": "HC", "split": "train",
                "sex": "U", "case_id": "c1", "task_type": "spontaneous_speech", "language": "en",
                "channel": "audio", "duration_sec": 10.0, "recording_index": 1, "word_count": 20.0,
            },
            {
                "dataset_id": "demo", "subject_id": "s1", "label": "HC", "split": "train",
                "sex": "U", "case_id": "c2", "task_type": "spontaneous_speech", "language": "zh",
                "channel": "audio", "duration_sec": 10.0, "recording_index": 2, "word_count": 12.0,
            },
        ]
    )
    output = tmp_path / "subject_features.csv"
    aggregate_subject_features(recording, output)
    subject = pd.read_csv(output).iloc[0]
    assert subject["language"] == "multilingual"
    assert subject["task_language_en__word_count"] == 20.0
    assert subject["task_language_zh__word_count"] == 12.0


def test_state_category_honors_reliability() -> None:
    assert _state_category(3.0, 0.2, 0.0) == "unreliable"
    assert _state_category(2.2, 0.9, 0.0) == "impaired"
    assert _state_category(1.2, 0.9, 0.0) == "borderline"
    assert _state_category(0.3, 0.9, 0.0) == "normal"


def test_stage_cache_schema_version_invalidates_old_output(tmp_path: Path) -> None:
    output = tmp_path / "output.txt"
    calls: list[str] = []

    def write() -> None:
        calls.append("executed")
        output.write_text(str(len(calls)), encoding="utf-8")

    StageCache(tmp_path / "cache", schema_version="v1").execute(
        "stage", ["same input"], [output], write
    )
    StageCache(tmp_path / "cache", schema_version="v1").execute(
        "stage", ["same input"], [output], write
    )
    StageCache(tmp_path / "cache", schema_version="v2").execute(
        "stage", ["same input"], [output], write
    )
    assert calls == ["executed", "executed"]


def test_direct_agent_batch_cache_requires_matching_request(tmp_path: Path) -> None:
    output = tmp_path / "batch.json"
    metadata = tmp_path / "batch.meta.json"
    output.write_text(
        json.dumps({"cases": [{"case_id": "CASE-1"}]}), encoding="utf-8"
    )
    metadata.write_text(
        json.dumps({"request_fingerprint": "current"}), encoding="utf-8"
    )
    assert _cached_batch(output, metadata, "current", ["CASE-1"]) is not None
    assert _cached_batch(output, metadata, "changed", ["CASE-1"]) is None
    assert _cached_batch(output, metadata, "current", ["CASE-2"]) is None


def test_direct_agent_retries_incomplete_batch(tmp_path: Path, monkeypatch) -> None:
    calls = 0

    def fake_run(*args, **kwargs):
        nonlocal calls
        calls += 1
        case_ids = ["CASE-1"] if calls == 1 else ["CASE-1", "CASE-2"]
        return {"cases": [{"case_id": case_id} for case_id in case_ids]}

    monkeypatch.setattr(direct_agent, "run_structured_batch", fake_run)
    response = _run_or_load_batch(
        tmp_path,
        "prompt",
        tmp_path / "schema.json",
        tmp_path / "output.json",
        tmp_path / "metadata.json",
        "model",
        ["HC", "AD"],
        ["CASE-1", "CASE-2"],
        "openai_api",
    )
    assert calls == 2
    assert [case["case_id"] for case in response["cases"]] == ["CASE-1", "CASE-2"]


def test_direct_agent_recovers_only_unambiguous_case_id_prefix() -> None:
    response = {"cases": [{"case_id": "P-4326C8F24"}, {"case_id": "P-B8791FAFBE"}]}
    expected = ["P-4326C8F24B", "P-B8791FAFBE"]
    assert _canonicalize_case_ids(response, expected) == expected
    assert [case["case_id"] for case in response["cases"]] == expected

    ambiguous = {"cases": [{"case_id": "P-123456"}, {"case_id": "P-ABCDEF00"}]}
    assert _canonicalize_case_ids(
        ambiguous, ["P-123456AA", "P-123456BB"]
    ) is None


def test_fixed_expert_cv_metadata_is_reportable() -> None:
    values, selected = _expert_cv_values(
        {"macro_f1": 0.55, "macro_auroc": 0.77, "fold_scores": []}
    )
    assert values == {"fixed": 0.77}
    assert selected == "fixed"


def test_dynamic_reliability_gate_cross_fits_probabilities() -> None:
    labels = ["HC", "MCI", "AD"]
    y = np.asarray(labels * 6)
    rng = np.random.default_rng(20260821)
    probability = rng.uniform(0.05, 0.4, size=(len(y), 2, len(labels)))
    for row, label in enumerate(y):
        probability[row, 0, labels.index(label)] += 0.8
    probability = probability / probability.sum(axis=2, keepdims=True)
    reliability = np.column_stack(
        [np.full(len(y), 0.95), np.full(len(y), 0.25)]
    )
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=20260821)
    bundle, oof, test, oof_weights, test_weights, metadata = (
        fit_dynamic_reliability_gate(
            probability,
            reliability,
            probability[:4],
            reliability[:4],
            y,
            labels,
            list(splitter.split(probability, y)),
            {
                "epochs": 3,
                "batch_size": 8,
                "hidden_dimension": 8,
                "device": "cpu",
            },
        )
    )
    assert bundle.expert_count == 2
    assert oof.shape == (len(y), len(labels))
    assert test.shape == (4, len(labels))
    assert oof_weights.shape == (len(y), 2)
    assert test_weights.shape == (4, 2)
    assert np.allclose(oof.sum(axis=1), 1.0, atol=1e-5)
    assert np.allclose(test_weights.sum(axis=1), 1.0, atol=1e-5)
    assert metadata["parameter_count"] > 0


def test_paired_task_gain_requires_stable_fold_improvement() -> None:
    summary = _paired_gain_summary(
        np.asarray([-0.006, 0.050, 0.015, -0.119, 0.076])
    )
    assert summary["mean"] > 0.0
    assert summary["lower_95"] < 0.0


def test_state_fusion_bounds_metric_contribution_without_hiding_raw_value() -> None:
    raw = pd.Series([-19.0, -1.0, 2.0, 17.0])
    assert _bounded_metric_contribution(raw, 5.0).tolist() == [-5.0, -1.0, 2.0, 5.0]


def test_report_prediction_roles_follow_fitted_branch_specification() -> None:
    metadata = {
        "branches": [
            {
                "kind": "clinical_state",
                "state_features": ["state_S01", "state_S02__task_cookie"],
            },
            {
                "kind": "low_interpretability_auxiliary",
                "state_features": ["state_S06"],
            },
        ]
    }
    assert _retained_state_features(metadata) == {
        "state_S01",
        "state_S02__task_cookie",
    }


def test_legacy_evidence_schema_gets_overall_task_scope() -> None:
    frame = _normalize_evidence_schema(pd.DataFrame({"metric_id": ["m1"]}))
    assert frame.loc[0, "task_scope"] == "overall"
    assert "reference_label" in frame.columns


def test_ece_is_zero_for_perfect_predictions() -> None:
    labels = np.array(["HC", "MCI", "AD"])
    probability = np.eye(3)
    assert expected_calibration_error(labels, probability, 10, labels.tolist()) == 0.0


def test_stage_cache_skips_unchanged_inputs(tmp_path: Path) -> None:
    output = tmp_path / "value.txt"
    calls = []

    def write() -> None:
        calls.append(1)
        output.write_text("ok", encoding="utf-8")

    cache = StageCache(tmp_path / "cache")
    first = cache.execute("stage", [{"x": 1}], [output], write)
    second = cache.execute("stage", [{"x": 1}], [output], write)
    assert first.status == "executed"
    assert second.status == "cached"
    assert len(calls) == 1


def test_report_segments_remove_source_label_identifiers() -> None:
    raw = '[{"segment_id":"HC_F_019239_002:S06","case_id":"HC_F_019239_002","start_sec":50,"end_sec":60,"silence_fraction":0.8,"rms_db_mean":-32}]'
    result = _sanitized_segments(raw)
    assert result[0]["segment_id"].startswith("SEG-")
    assert "HC_F" not in str(result)


def test_agent_cohort_is_capped_deterministically_without_using_labels() -> None:
    truth = pd.DataFrame(
        {
            "subject_id": [f"H{i}" for i in range(10)] + [f"A{i}" for i in range(10)],
            "label": ["HC"] * 10 + ["AD"] * 10,
        }
    )
    selected = select_agent_cohort(truth, 8)
    assert len(selected) == 8
    relabeled = truth.copy()
    relabeled["label"] = relabeled["label"].map({"HC": "AD", "AD": "HC"})
    selected_after_relabel = select_agent_cohort(relabeled, 8)
    assert selected["subject_id"].tolist() == selected_after_relabel["subject_id"].tolist()


def test_chat_metrics_use_patient_turns_only(tmp_path: Path) -> None:
    transcript = tmp_path / "case.cha"
    transcript.write_text(
        "@Begin\n*INV: tell me what you see .\n*PAR: um the boy gets cookies [/] cookie .\n*PAR: he is there .\n@End\n",
        encoding="utf-8",
    )
    metrics = transcript_metrics(str(transcript), "en", 60.0)
    assert metrics["patient_turn_count"] == 2
    assert metrics["interviewer_turn_count"] == 1
    assert metrics["filler_rate_100w"] > 0
    assert metrics["repair_rate_100w"] > 0


def test_dataset_language_aliases_use_supported_codes() -> None:
    assert canonical_language("English") == "en"
    assert canonical_language("Spanish") == "es"
    assert canonical_language("Mandarin") == "zh"
    assert canonical_language("zh-CN") == "zh"
    assert canonical_language("multilingual") == ""


def test_auto_asr_uses_known_manifest_language() -> None:
    assert asr._effective_asr_language({"language": "Mandarin"}, "auto") == "zh"
    assert asr._effective_asr_language({"language": "Spanish"}, "auto") == "es"
    assert asr._effective_asr_language({"language": "multilingual"}, "auto") is None


def test_channel_profiles_change_metric_evidence_design() -> None:
    audio_only = load_all("NCMMSC2021_AD")
    interview = load_all("IAEAV")
    audio_ids = {item["id"] for item in audio_only["metrics"]["metrics"]}
    interview_ids = {item["id"] for item in interview["metrics"]["metrics"]}
    assert "patient_turn_share" not in audio_ids
    assert "patient_turn_share" in interview_ids
    assert len(interview_ids) > len(audio_ids)


def test_processed_bootstrap_never_overwrites_rebuilt_state_outputs() -> None:
    assert "state_cards.csv" not in _PROCESSED_REUSE_FILES
    assert "state_wide.csv" not in _PROCESSED_REUSE_FILES


def test_aggregate_payload_reports_tied_winner_without_false_loss() -> None:
    layer_a = pd.DataFrame(
        [
            {
                "dataset_name": "Tie task",
                "condition": condition,
                "metric": "macro_auroc_ovr",
                "value": value,
                "analysis_scope": "matched_three_arm",
                "preferred_scope": "matched_three_arm",
            }
            for condition, value in (("B1", 1.0), ("B2", 0.8), ("Ours", 1.0))
        ]
    )
    layer_b = pd.DataFrame(
        [{"condition": "Ours", "passed": True}]
    )
    row = _evaluation_payload(layer_a, layer_b)["rows"][0]
    assert row["winner"] == "B1/Ours"
    assert "并列最高" in row["conclusion"]


def test_aggregate_payload_invalidates_high_shortcut_control() -> None:
    layer_a = pd.DataFrame(
        [
            {
                "dataset_name": "Shortcut task",
                "condition": condition,
                "metric": "macro_auroc_ovr",
                "value": value,
                "analysis_scope": "full_available_cohort",
                "preferred_scope": "full_available_cohort",
            }
            for condition, value in (
                ("B1", 0.80),
                ("B2", 0.75),
                ("Ours", 0.99),
                ("QC_only", 0.96),
            )
        ]
    )
    row = _evaluation_payload(
        layer_a, pd.DataFrame([{"condition": "Ours", "passed": True}])
    )["rows"][0]
    assert row["shortcut_invalidated"] is True
    assert "不能作为临床有效性" in row["conclusion"]


def test_metric_reference_uses_task_specific_training_label(tmp_path: Path) -> None:
    subjects = pd.DataFrame(
        {
            "dataset_id": ["progression"] * 4,
            "subject_id": ["1", "2", "3", "4"],
            "label": ["no_decline", "decline", "no_decline", "decline"],
            "split": ["train", "train", "test", "test"],
            "audio_reliability": [1.0] * 4,
            "text_reliability": [1.0] * 4,
            "pause_rate": [2.0, 20.0, 3.0, 18.0],
        }
    )
    subjects_path = tmp_path / "subjects.csv"
    evidence_path = tmp_path / "evidence.csv"
    reference_path = tmp_path / "reference.json"
    subjects.to_csv(subjects_path, index=False)
    build_metric_evidence(
        subjects_path,
        {
            "metrics": [
                {
                    "id": "pause_rate",
                    "state": "S01",
                    "branch": "speech_behavior",
                    "direction": 1,
                    "role": "clinical",
                    "reliability": 1.0,
                    "confounds": [],
                    "report_permission": True,
                }
            ]
        },
        evidence_path,
        reference_path,
        reference_label="no_decline",
    )
    evidence = pd.read_csv(evidence_path)
    assert evidence["reference_label"].eq("no_decline").all()
    assert evidence["reference_median"].eq(2.0).all()
    assert evidence["cn_train_median"].eq(2.0).all()
    target = evidence.loc[evidence["subject_id"].eq(2)].iloc[0]
    assert bool(target["missing"])
    assert target["reliability"] == 0.0
    assert target["directional_z"] == 0.0


def test_language_dependent_metrics_require_same_language_reference(tmp_path: Path) -> None:
    subjects = pd.DataFrame(
        {
            "dataset_id": ["multilingual"] * 11,
            "subject_id": [str(index) for index in range(11)],
            "label": ["HC"] * 9 + ["AD", "AD"],
            "split": ["train"] * 9 + ["test", "test"],
            "language": ["en"] * 8 + ["zh", "en", "zh"],
            "audio_reliability": [1.0] * 11,
            "text_reliability": [1.0] * 11,
            "lexical_ttr": [0.50] * 8 + [0.80, 0.40, 0.90],
        }
    )
    subjects_path = tmp_path / "subjects.csv"
    evidence_path = tmp_path / "evidence.csv"
    reference_path = tmp_path / "reference.json"
    subjects.to_csv(subjects_path, index=False)
    build_metric_evidence(
        subjects_path,
        {
            "metrics": [
                {
                    "id": "lexical_ttr",
                    "state": "S08",
                    "branch": "language",
                    "direction": -1,
                    "role": "clinical",
                    "reliability": 1.0,
                    "confounds": [],
                    "report_permission": True,
                }
            ]
        },
        evidence_path,
        reference_path,
        reference_label="HC",
    )
    evidence = pd.read_csv(evidence_path)
    english_case = evidence[evidence["subject_id"].eq(9)].iloc[0]
    chinese_case = evidence[evidence["subject_id"].eq(10)].iloc[0]
    assert english_case["reference_scope"] == "training_reference_language:en"
    assert bool(english_case["missing"])
    assert english_case["reliability"] == 0.0
    assert chinese_case["reference_scope"] == "training_reference_language:zh"
    assert bool(chinese_case["missing"])
    assert chinese_case["reliability"] == 0.0


def test_fold_state_calibration_uses_same_language_controls() -> None:
    rows = []
    for index in range(8):
        for language, value in (("english", 10.0 + index), ("spanish", 100.0 + index)):
            rows.append(
                {
                    "dataset_id": "multilingual",
                    "subject_id": f"{language}-{index}",
                    "label": "HC",
                    "split": "train",
                    "language": language,
                    "metric_id": "word_count",
                    "metric_instance_id": "word_count",
                    "task_scope": "overall",
                    "value": value,
                    "direction": -1,
                    "reliability": 1.0,
                }
            )
    rows.append(
        {
            "dataset_id": "multilingual",
            "subject_id": "spanish-case",
            "label": "AD",
            "split": "test",
            "language": "spanish",
            "metric_id": "word_count",
            "metric_instance_id": "word_count",
            "task_scope": "overall",
            "value": 103.5,
            "direction": -1,
            "reliability": 1.0,
        }
    )
    evidence = pd.DataFrame(rows)
    states = build_fold_calibrated_state_frame(
        evidence,
        {"states": [{"id": "S02", "metrics": ["word_count"], "weights": [1.0]}]},
        {str(row["subject_id"]) for row in rows if row["label"] == "HC"},
        "HC",
    )
    case = states.loc[states["subject_id"].eq("spanish-case")].iloc[0]
    assert abs(float(case["state_S02"])) < 1e-9


def test_multitask_states_retain_task_calibration_and_segment_trace(tmp_path: Path) -> None:
    subjects = pd.DataFrame(
        {
            "dataset_id": ["multitask"] * 4,
            "subject_id": ["1", "2", "3", "4"],
            "label": ["HC", "HC", "AD", "AD"],
            "split": ["train", "train", "test", "test"],
            "pause_rate": [6.0, 9.0, 15.0, 18.0],
            "audio_reliability": [0.95] * 4,
            "text_reliability": [0.95] * 4,
            "task_cookie__pause_rate": [2.0, 4.0, 8.0, 9.0],
            "task_cookie__audio_reliability": [0.90] * 4,
            "task_cookie__text_reliability": [0.90] * 4,
            "task_recall__pause_rate": [10.0, 14.0, 22.0, 24.0],
            "task_recall__audio_reliability": [0.80] * 4,
            "task_recall__text_reliability": [0.80] * 4,
        }
    )
    subjects_path = tmp_path / "subjects.csv"
    evidence_path = tmp_path / "evidence.csv"
    reference_path = tmp_path / "reference.json"
    subjects.to_csv(subjects_path, index=False)
    metric_config = {
        "metrics": [
            {
                "id": "pause_rate",
                "state": "S01",
                "branch": "speech_behavior",
                "direction": 1,
                "role": "clinical",
                "reliability": 1.0,
                "confounds": [],
                "report_permission": True,
            }
        ]
    }
    build_metric_evidence(
        subjects_path,
        metric_config,
        evidence_path,
        reference_path,
        reference_label="HC",
    )
    evidence = pd.read_csv(evidence_path)
    assert set(evidence["task_scope"]) == {"overall", "cookie", "recall"}
    task_reference = (
        evidence[evidence["subject_id"].eq(1)]
        .set_index("task_scope")["reference_median"]
        .to_dict()
    )
    assert task_reference["cookie"] == 3.0
    assert task_reference["recall"] == 12.0
    fold_states = build_fold_calibrated_state_frame(
        evidence,
        {
            "states": [
                {
                    "id": "S01",
                    "metrics": ["pause_rate"],
                    "weights": [1.0],
                }
            ]
        },
        {"1", "2"},
        "HC",
    )
    assert {
        "state_S01",
        "state_S01__task_cookie",
        "state_S01__task_recall",
    }.issubset(fold_states.columns)

    recordings = pd.DataFrame(
        [
            {
                "dataset_id": "multitask",
                "subject_id": subject_id,
                "case_id": f"{subject_id}-{task}",
                "task_type": task,
            }
            for subject_id in subjects["subject_id"]
            for task in ("cookie", "recall")
        ]
    )
    segments = pd.DataFrame(
        [
            {
                "segment_id": f"{row.case_id}:S01",
                "case_id": row.case_id,
                "start_sec": 0.0,
                "end_sec": 5.0,
                "silence_fraction": 0.2 if row.task_type == "cookie" else 0.6,
                "rms_db_mean": -24.0,
                "source_spans": "[]",
            }
            for row in recordings.itertuples()
        ]
    )
    recording_path = tmp_path / "recordings.csv"
    segments_path = tmp_path / "segments.csv"
    cards_path = tmp_path / "cards.csv"
    wide_path = tmp_path / "states.csv"
    recordings.to_csv(recording_path, index=False)
    segments.to_csv(segments_path, index=False)
    build_state_cards(
        evidence_path,
        recording_path,
        segments_path,
        {
            "states": [
                {
                    "id": "S01",
                    "name_zh": "停顿负担",
                    "branch": "speech_behavior",
                    "clinical_question": "是否出现异常停顿？",
                    "metrics": ["pause_rate"],
                    "weights": [1.0],
                }
            ]
        },
        cards_path,
        wide_path,
    )
    cards = pd.read_csv(cards_path, dtype={"subject_id": str})
    case_cards = cards[cards["subject_id"].eq("3")]
    assert set(case_cards["state_id"]) == {
        "S01",
        "S01__task_cookie",
        "S01__task_recall",
    }
    cookie_segments = case_cards.loc[
        case_cards["task_scope"].eq("cookie"), "evidence_segments"
    ].iloc[0]
    assert all("cookie" in item["case_id"] for item in json.loads(cookie_segments))
    assert case_cards.loc[
        case_cards["task_scope"].eq("cookie"), "trace_resolution"
    ].eq("task_and_segment").all()
    assert case_cards["report_permission"].all()
    assert np.allclose(case_cards["report_state_z"], case_cards["state_z"])
    assert case_cards["report_confidence"].gt(0.0).all()

    wide = pd.read_csv(wide_path)
    specifications = _branch_specifications(
        wide,
        subjects,
        {
            "state_branches": {"S01": "speech_behavior"},
            "ours": {"qc_orthogonalization": {"enabled": False}},
        },
    )
    branch = next(item for item in specifications if item["name"] == "speech_behavior")
    assert set(branch["state_features"]) == {
        "state_S01",
        "state_S01__task_cookie",
        "state_S01__task_recall",
    }


def test_branch_caps_are_enforced_simultaneously() -> None:
    weights = np.array([[0.75, 0.20, 0.05], [0.05, 0.90, 0.05]])
    capped = _apply_caps(weights, np.array([0.0, 1.0, 0.35]))
    assert np.allclose(capped.sum(axis=1), 1.0)
    assert np.allclose(capped[:, 0], 0.0)
    assert np.all(capped[:, 2] <= 0.35 + 1e-12)


def test_generated_transcript_updates_audio_only_manifest(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "case.wav"
    audio.write_bytes(b"placeholder")
    manifest = pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "subject_id": "subject-1",
                "audio_path": str(audio),
                "transcript_path": np.nan,
                "language": "zh",
                "analysis_intervals": "[]",
            }
        ]
    )
    manifest_path = tmp_path / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    monkeypatch.setattr(
        asr,
        "_transcribe_mlx",
        lambda record, language, model: {
            "case_id": record["case_id"],
            "subject_id": record["subject_id"],
            "language": "zh",
            "language_probability": 0.99,
            "text": "这是测试转录",
            "segments_json": "[]",
            "asr_model": model,
        },
    )
    asr.prepare_analysis_transcripts(
        manifest_path,
        tmp_path / "analysis_manifest.csv",
        tmp_path / "recording_transcripts.csv",
        tmp_path / "subject_transcripts.csv",
        {"asr_backend": "mlx_whisper", "asr_model": "test", "asr_language": "auto"},
        generate_missing=True,
    )

    updated = pd.read_csv(tmp_path / "analysis_manifest.csv")
    assert updated.loc[0, "transcript_origin"] == "generated_asr"
    assert 0.50 <= updated.loc[0, "transcript_reliability"] <= 0.75
    transcript = Path(updated.loc[0, "transcript_path"])
    assert transcript.read_text(encoding="utf-8") == "这是测试转录"


def test_generated_transcript_preserves_trusted_manifest_language(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "case.wav"
    audio.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "subject_id": "subject-1",
                "audio_path": str(audio),
                "transcript_path": np.nan,
                "language": "en",
                "analysis_intervals": "[]",
            }
        ]
    ).to_csv(manifest_path, index=False)
    monkeypatch.setattr(
        asr,
        "_transcribe_mlx",
        lambda record, language, model: {
            "case_id": record["case_id"],
            "subject_id": record["subject_id"],
            "language": "id",
            "language_probability": 0.81,
            "text": "This is an English transcript.",
            "segments_json": "[]",
            "asr_model": model,
        },
    )
    analysis_manifest_path = tmp_path / "analysis_manifest.csv"
    asr.prepare_analysis_transcripts(
        manifest_path,
        analysis_manifest_path,
        tmp_path / "recording_transcripts.csv",
        tmp_path / "subject_transcripts.csv",
        {"asr_backend": "mlx_whisper", "asr_model": "test", "asr_language": "auto"},
        generate_missing=True,
    )
    updated = pd.read_csv(analysis_manifest_path)
    assert updated.loc[0, "language"] == "en"


def test_generated_transcript_resolves_multilingual_manifest_language(tmp_path: Path, monkeypatch) -> None:
    audio = tmp_path / "case.wav"
    audio.write_bytes(b"placeholder")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [
            {
                "case_id": "case-1",
                "subject_id": "subject-1",
                "audio_path": str(audio),
                "transcript_path": np.nan,
                "language": "zh-en",
                "analysis_intervals": "[]",
            }
        ]
    ).to_csv(manifest_path, index=False)
    monkeypatch.setattr(
        asr,
        "_transcribe_mlx",
        lambda record, language, model: {
            "case_id": record["case_id"],
            "subject_id": record["subject_id"],
            "language": "zh",
            "language_probability": 0.98,
            "text": "这是测试转录",
            "segments_json": "[]",
            "asr_model": model,
        },
    )
    analysis_manifest_path = tmp_path / "analysis_manifest.csv"
    asr.prepare_analysis_transcripts(
        manifest_path,
        analysis_manifest_path,
        tmp_path / "recording_transcripts.csv",
        tmp_path / "subject_transcripts.csv",
        {"asr_backend": "mlx_whisper", "asr_model": "test", "asr_language": "auto"},
        generate_missing=True,
    )
    updated = pd.read_csv(analysis_manifest_path)
    assert updated.loc[0, "language"] == "zh"


def test_empty_distributed_transcript_is_not_marked_human_reliable(tmp_path: Path) -> None:
    audio = tmp_path / "case.wav"
    audio.write_bytes(b"placeholder")
    empty_transcript = tmp_path / "empty.txt"
    empty_transcript.write_text("", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(
        [{
            "case_id": "case-1",
            "subject_id": "subject-1",
            "audio_path": str(audio),
            "transcript_path": str(empty_transcript),
            "language": "English",
            "analysis_intervals": "[]",
        }]
    ).to_csv(manifest_path, index=False)
    analysis_manifest_path = tmp_path / "analysis_manifest.csv"
    asr.prepare_analysis_transcripts(
        manifest_path,
        analysis_manifest_path,
        tmp_path / "recording_transcripts.csv",
        tmp_path / "subject_transcripts.csv",
        {"asr_backend": "mlx_whisper", "asr_model": "test", "asr_language": "auto"},
        generate_missing=False,
    )
    updated = pd.read_csv(analysis_manifest_path)
    assert updated.loc[0, "transcript_origin"] == "missing"
    assert updated.loc[0, "transcript_reliability"] == 0.0


def test_adresso_segmentation_is_not_used_as_transcript(tmp_path: Path) -> None:
    root = tmp_path / "raw" / "train"
    for folder in ("cn", "ad"):
        (root / "segmentation" / folder).mkdir(parents=True, exist_ok=True)
        (root / "audio" / folder).mkdir(parents=True, exist_ok=True)
        for index in range(4):
            case_id = f"{folder}{index}"
            pd.DataFrame([{"begin": 0, "end": 100, "speaker": "PAR"}]).to_csv(
                root / "segmentation" / folder / f"{case_id}.csv", index=False
            )
            import soundfile as sf

            offset = index + (10 if folder == "ad" else 0)
            waveform = np.zeros(1600)
            waveform[offset] = 0.1 + offset / 100.0
            sf.write(root / "audio" / folder / f"{case_id}.wav", waveform, 16000)

    manifest_path = tmp_path / "manifest.csv"
    build_adresso_manifest(
        ProjectPaths(tmp_path),
        {
            "dataset_id": "test_adresso",
            "raw_path": "raw",
            "managed_train_root": "train",
            "folder_labels": {"cn": "HC", "ad": "AD"},
            "task_type": "picture_description",
            "language": "en",
            "channel": "picture_description_audio_only",
            "speaker_roles": "distributed_segmentation",
            "test_size": 0.25,
            "split_seed": 7,
        },
        manifest_path,
        tmp_path / "audit.json",
    )

    manifest = pd.read_csv(manifest_path)
    assert manifest["transcript_path"].fillna("").eq("").all()
    assert manifest["segmentation_path"].str.endswith(".csv").all()
    assert manifest["analysis_intervals"].eq("[[0.0, 0.1]]").all()


def test_pitt_case_ids_are_unique_across_tasks(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    rows = []
    import soundfile as sf

    for group, subject_ids in (("Dementia", ["001", "002"]), ("Control", ["101", "102"])):
        for subject_id in subject_ids:
            for task in ("cookie", "fluency"):
                source_case_id = f"{subject_id}-0"
                media = raw / "Pitt_noise_removed" / group / task / f"{source_case_id}.wav"
                transcript = raw / "Pitt_noise_removed" / group / task / f"{source_case_id}.cha"
                media.parent.mkdir(parents=True, exist_ok=True)
                waveform = np.zeros(1600)
                waveform[0] = (len(rows) + 1) / 100.0
                sf.write(media, waveform, 16000)
                transcript.write_text("@Begin\n*PAR: test sample .\n@End\n", encoding="utf-8")
                rows.append(
                    {
                        "pairing_status": "paired",
                        "all_media_paths": str(media),
                        "preferred_media_path": str(media),
                        "transcript_path": str(transcript),
                        "case_id": source_case_id,
                        "group": group,
                        "task": task,
                    }
                )
    pd.DataFrame(rows).to_csv(raw / "talkbank_media_pairing_manifest.csv", index=False)
    manifest_path = tmp_path / "manifest.csv"
    build_pitt_manifest(
        ProjectPaths(tmp_path),
        {
            "dataset_id": "Pitt",
            "raw_path": "raw",
            "channel": "structured_multitask",
            "task_type": "neuropsychological_multitask",
            "speaker_roles": "CHAT_PAR_INV",
            "test_size": 0.5,
            "split_seed": 7,
        },
        manifest_path,
        tmp_path / "audit.json",
    )
    manifest = pd.read_csv(manifest_path)
    assert manifest["case_id"].is_unique
    assert manifest["case_id"].str.contains("cookie|fluency").all()
    assert manifest["subject_id"].nunique() == 4


def test_unreviewed_asr_profile_blocks_unsupported_clinical_text_claims() -> None:
    config = load_all("ADReSSo_2021_diagnosis")
    metrics = {item["id"]: item for item in config["metrics"]["metrics"]}
    enabled_states = {item["id"] for item in config["states"]["states"]}

    assert "S07" in enabled_states and "S08" in enabled_states
    assert "S11" not in enabled_states and "S12" not in enabled_states
    assert metrics["lexical_mattr50"]["report_permission"] is False
    assert "ASR_error" in metrics["lexical_mattr50"]["confounds"]
    assert "filler_rate_100w" not in metrics


def test_qc_orthogonalizer_removes_train_fold_qc_component() -> None:
    duration = np.linspace(10.0, 100.0, 40)
    frame = pd.DataFrame(
        {
            "state_S01": 0.08 * duration + np.sin(duration),
            "original_duration_sec": duration,
        }
    )
    transformer = QCOrthogonalizer(("state_S01",), ("original_duration_sec",), alpha=1e-6)
    residual = transformer.fit_transform(frame)[:, 0]
    assert abs(np.corrcoef(residual, duration)[0, 1]) < 0.02


def test_branch_specifications_do_not_duplicate_qc_as_disease_evidence() -> None:
    frame = pd.DataFrame(
        {
            "state_S01": [0.1, 0.2],
            "rel_S01": [0.9, 0.8],
            "rms_db_mean": [-22.0, -18.0],
            "rms_db_std": [2.0, 3.0],
            "mfcc_01_mean": [0.4, 0.5],
            "audio_reliability": [0.9, 0.8],
        }
    )
    config = {
        "state_branches": {"S01": "speech_behavior"},
        "ours": {
            "qc_orthogonalization": {
                "enabled": True,
                "features": ["rms_db_mean", "rms_db_std", "audio_reliability"],
            }
        },
    }

    specifications = _branch_specifications(frame, frame, config)

    for specification in specifications:
        assert len(specification["features"]) == len(set(specification["features"]))
        assert not (
            set(specification["state_features"])
            & set(specification["qc_features"])
        )
    auxiliary = next(item for item in specifications if item["name"] == "auxiliary_acoustic")
    assert "mfcc_01_mean" in auxiliary["state_features"]
    assert "rms_db_mean" not in auxiliary["state_features"]


def test_branch_weight_plot_is_compatible_with_current_matplotlib(tmp_path: Path) -> None:
    contributions = pd.DataFrame(
        {
            "label": ["HC", "AD", "AD"],
            "agent_correction_gate": [0.1, 0.3, 0.4],
        }
    )
    output = tmp_path / "branch_weights.png"

    plot_branch_weights({}, contributions, output, ["HC", "AD"])

    assert output.exists()
    assert output.stat().st_size > 0


def test_condition_c_text_expert_supports_multiclass_labels() -> None:
    texts = np.asarray(
        [
            "clear detailed description",
            "clear organized response",
            "some hesitation and repair",
            "frequent searching for words",
            "markedly sparse output",
            "severe loss of content",
        ],
        dtype=object,
    )
    labels = np.asarray(["HC", "HC", "MCI", "MCI", "AD", "AD"])
    model = _text_pipeline(c=1.0, max_iter=500, min_df=1)

    model.fit(texts, labels)

    assert set(model.classes_) == {"HC", "MCI", "AD"}


def test_failure_metrics_treat_empty_predictions_as_unavailable() -> None:
    metrics = _prediction_metrics(
        pd.DataFrame(columns=["label", "predicted_label"])
    )

    assert metrics["n"] == 0.0
    assert np.isnan(metrics["macro_auroc"])


def test_reference_state_intervention_is_limited_to_misclassified_cases(tmp_path: Path) -> None:
    feature_rows = []
    state_rows = []
    for index in range(24):
        split = "train" if index < 20 else "test"
        label = "HC" if index % 2 == 0 else "AD"
        state_value = -2.0 if label == "HC" else 2.0
        if split == "test":
            state_value = -state_value
        identity = {
            "dataset_id": "synthetic",
            "subject_id": str(index),
            "label": label,
            "split": split,
        }
        feature_rows.append(
            {
                **identity,
                "original_duration_sec": 40.0 + index,
                "audio_reliability": 0.95,
                "text_reliability": 0.95,
            }
        )
        state_rows.append({**identity, "state_S01": state_value, "rel_S01": 0.95})

    features_path = tmp_path / "features.csv"
    states_path = tmp_path / "states.csv"
    evidence_path = tmp_path / "evidence.csv"
    predictions_path = tmp_path / "predictions.csv"
    interventions_path = tmp_path / "interventions.csv"
    pd.DataFrame(feature_rows).to_csv(features_path, index=False)
    pd.DataFrame(state_rows).to_csv(states_path, index=False)
    pd.DataFrame(
        [
            {
                **{key: row[key] for key in ["dataset_id", "subject_id", "label", "split"]},
                "metric_id": "synthetic_state",
                "metric_instance_id": "synthetic_state",
                "task_scope": "overall",
                "value": row["state_S01"],
                "direction": 1,
                "reliability": row["rel_S01"],
            }
            for row in state_rows
        ]
    ).to_csv(evidence_path, index=False)
    states_config = {
        "states": [
            {
                "id": "S01",
                "name_zh": "停顿负担",
                "branch": "speech_behavior",
                "clinical_question": "是否出现异常停顿？",
                "metrics": ["synthetic_state"],
                "weights": [1.0],
            }
        ]
    }
    config = {
        "labels": ["HC", "AD"],
        "positive_class": "AD",
        "state_branches": {"S01": "speech_behavior"},
        "cross_validation": {"folds": 5},
        "ours": {
            "c_grid": [0.1, 1.0],
            "max_iter": 1000,
            "branch_gate_l2": 0.05,
            "min_branch_cv_auroc": 0.52,
            "quality_prior_strength": 4.0,
            "auxiliary_weight_cap": 0.35,
            "qc_orthogonalization": {
                "enabled": True,
                "alpha": 1.0,
                "features": ["original_duration_sec", "audio_reliability", "text_reliability"],
            },
        },
    }
    train_ours(
        features_path,
        states_path,
        evidence_path,
        states_config,
        config,
        predictions_path,
        tmp_path / "ablations.csv",
        tmp_path / "contributions.csv",
        interventions_path,
        tmp_path / "model.joblib",
        tmp_path / "metadata.json",
    )

    predictions = pd.read_csv(predictions_path)
    interventions = pd.read_csv(interventions_path)
    error_count = int((predictions["label"] != predictions["predicted_label"]).sum())
    assert len(interventions) == error_count
    assert set(interventions["subject_id"].astype(str)) == set(
        predictions.loc[
            predictions["label"] != predictions["predicted_label"], "subject_id"
        ].astype(str)
    )
    assert interventions["correction_source"].eq("training-set true-class state mean").all()


def test_recording_feature_cache_rebinds_current_case_identity(tmp_path: Path) -> None:
    import soundfile as sf

    sample_rate = 16000
    time = np.arange(sample_rate * 2) / sample_rate
    audio = tmp_path / "sample.wav"
    sf.write(audio, 0.1 * np.sin(2 * np.pi * 180 * time), sample_rate)
    transcript = tmp_path / "sample.txt"
    transcript.write_text("the boy takes a cookie", encoding="utf-8")
    manifest_path = tmp_path / "manifest.csv"
    base = {
        "dataset_id": "synthetic",
        "subject_id": "subject-1",
        "label": "HC",
        "split": "train",
        "sex": "F",
        "task_type": "picture_description",
        "language": "en",
        "channel": "picture_description",
        "audio_path": str(audio),
        "audio_sha256": "stable-audio-content",
        "analysis_intervals": "[]",
        "transcript_path": str(transcript),
        "transcript_reliability": 0.95,
    }
    pd.DataFrame([{**base, "case_id": "case-old"}]).to_csv(manifest_path, index=False)
    outputs = [tmp_path / "recording.csv", tmp_path / "subject.csv", tmp_path / "segments.csv"]
    extract_features(manifest_path, *outputs, workers=1)
    cache_file = next((tmp_path / "feature_cache").glob("*.pkl"))
    cache_mtime = cache_file.stat().st_mtime_ns

    pd.DataFrame([{**base, "case_id": "case-new"}]).to_csv(manifest_path, index=False)
    extract_features(manifest_path, *outputs, workers=1)
    recording = pd.read_csv(outputs[0])
    segments = pd.read_csv(outputs[2])
    assert recording.loc[0, "case_id"] == "case-new"
    assert segments["case_id"].eq("case-new").all()
    assert segments["segment_id"].str.startswith("case-new:S").all()
    assert cache_file.stat().st_mtime_ns == cache_mtime
