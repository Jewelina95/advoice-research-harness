from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from advoice import states
from advoice.config import load_all
from advoice.evidence import build_metric_evidence, recalibrate_metric_evidence_frame


def _subjects() -> pd.DataFrame:
    return pd.DataFrame({
        "dataset_id": ["coverage"] * 3,
        "subject_id": ["hc1", "hc2", "case"],
        "label": ["HC", "HC", "AD"],
        "split": ["train", "train", "test"],
        "language": ["en"] * 3,
        "audio_reliability": [1.0] * 3,
        "text_reliability": [1.0] * 3,
        "silence_fraction": [0.1, 0.3, 0.8],
    })


def _configs():
    config = load_all("NCMMSC2021_AD")
    state = next(item for item in config["states"]["states"] if item["id"] == "S01")
    metrics = [item for item in config["metrics"]["metrics"] if item["id"] in state["metrics"]]
    return {"metrics": metrics}, {"states": [state]}


def _build(tmp_path, subjects, metrics):
    subjects.to_csv(tmp_path / "subjects.csv", index=False)
    build_metric_evidence(
        tmp_path / "subjects.csv", metrics,
        tmp_path / "evidence.csv", tmp_path / "reference.json",
    )
    return pd.read_csv(tmp_path / "evidence.csv", dtype={"subject_id": str})


def _cards(evidence, config, **kwargs):
    helper = getattr(states, "build_state_cards_frame", None)
    assert callable(helper), "Calibration needs the same dataframe aggregation as inference"
    return helper(evidence, config, **kwargs)


def test_absent_columns_remain_in_configured_evidence_grid(tmp_path: Path):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects(), metrics)
    case = evidence[evidence["subject_id"].eq("case")].set_index("metric_id")
    assert set(case.index) == set(config["states"][0]["metrics"])
    absent = case.drop(index="silence_fraction")
    assert absent["missing"].all()
    assert absent["value"].isna().all()
    assert absent["robust_z"].isna().all()
    assert absent["directional_z"].eq(0).all()
    assert absent["reliability"].eq(0).all()
    assert absent["report_permission"].all()
    references = json.loads((tmp_path / "reference.json").read_text())["metrics"]
    assert all(not references[metric]["available"] for metric in absent.index)


def test_s01_coverage_uses_all_four_expected_metrics(tmp_path: Path):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects(), metrics)
    cards, wide = _cards(evidence, config)
    case = cards[cards["subject_id"].eq("case")].iloc[0]
    assert case["missing_fraction"] == pytest.approx(0.75)
    assert 1 - case["missing_fraction"] == pytest.approx(0.25)
    assert case["confidence"] == pytest.approx(0.30 * 0.85)
    assert case["report_confidence"] == pytest.approx(case["confidence"])
    assert case["category"] == "unreliable"
    assert [item["metric_id"] for item in json.loads(case["supporting_metrics"])] == ["silence_fraction"]
    assert wide.set_index("subject_id").loc["case", "rel_S01"] == case["confidence"]


def test_no_observed_metrics_still_produces_unreliable_state(tmp_path: Path):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects().drop(columns="silence_fraction"), metrics)
    cards, wide = _cards(evidence, config)
    assert len(evidence) == 12
    assert len(cards) == len(wide) == 3
    assert cards["missing_fraction"].eq(1).all()
    assert cards["confidence"].eq(0).all()
    assert cards["category"].eq("unreliable").all()
    assert not cards["report_permission"].any()


def test_channel_disabled_and_planned_states_are_not_invented(tmp_path: Path):
    config = load_all("NCMMSC2021_AD")
    evidence = _build(tmp_path, _subjects(), config["metrics"])
    assert set(evidence["metric_id"]) == {item["id"] for item in config["metrics"]["metrics"]}
    cards, _ = _cards(evidence, config["states"])
    expected = {item["id"] for item in config["states"]["states"] if item["metrics"]}
    assert set(cards["state_id"]) == expected
    assert not set(cards["state_id"]) & {"S09", "S13", "S14"}


def test_task_grid_retains_missing_metrics_but_excludes_unperformed_tasks(tmp_path: Path):
    metrics, config = _configs()
    subjects = _subjects()
    subjects["task_cookie__duration_sec"] = [10.0, 10.0, 10.0]
    subjects["task_cookie__silence_fraction"] = [0.1, 0.2, 0.5]
    subjects["task_recall__duration_sec"] = [12.0, 12.0, np.nan]
    subjects["task_recall__silence_fraction"] = [0.3, 0.4, np.nan]
    evidence = _build(tmp_path, subjects, metrics)
    case = evidence[evidence["subject_id"].eq("case")]
    assert case.groupby("task_scope").size().to_dict() == {"overall": 4, "cookie": 4}
    cards, _ = _cards(evidence, config)
    case_cards = cards[cards["subject_id"].eq("case")]
    assert set(case_cards["state_id"]) == {"S01", "S01__task_cookie"}
    assert case_cards["missing_fraction"].eq(0.75).all()
    assert "S01__task_recall" in set(cards["state_id"])


def test_single_task_keeps_overall_only_behavior(tmp_path: Path):
    metrics, _ = _configs()
    subjects = _subjects()
    subjects["task_cookie__duration_sec"] = 10.0
    evidence = _build(tmp_path, subjects, metrics)
    assert set(evidence["task_scope"]) == {"overall"}
    assert len(evidence) == 12


def test_missing_value_is_distinct_from_unavailable_reference(tmp_path: Path):
    metrics, _ = _configs()
    subjects = _subjects()
    subjects["long_pause_rate_min"] = 2.0
    subjects["pause_mean_sec"] = [1.0, 2.0, np.nan]
    evidence = _build(tmp_path, subjects, metrics)
    assert {"value_missing", "reference_available", "evidence_status"} <= set(evidence.columns)
    case = evidence[evidence["subject_id"].eq("case")].set_index("metric_id")
    assert case.loc["silence_fraction", "evidence_status"] == "available"
    assert case.loc["long_pause_rate_min", "evidence_status"] == "unavailable"
    assert not case.loc["long_pause_rate_min", "value_missing"]
    assert not case.loc["long_pause_rate_min", "reference_available"]
    assert case.loc["pause_mean_sec", "evidence_status"] == "missing"
    assert case.loc["pause_mean_sec", "reference_available"]
    assert case.loc["pause_p90_sec", "evidence_status"] == "missing"
    assert case.drop(index="silence_fraction")["missing"].all()
    assert case["report_permission"].all()


def test_frame_helper_matches_file_entrypoint_and_preserves_input(tmp_path: Path):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects(), metrics)
    original = evidence.copy(deep=True)
    recordings = pd.DataFrame([{"subject_id": "case", "case_id": "case-cookie", "task_type": "cookie"}])
    segments = pd.DataFrame([{
        "case_id": "case-cookie", "segment_id": "segment-1", "start_sec": 0.0,
        "end_sec": 5.0, "silence_fraction": 0.8, "rms_db_mean": -24.0,
    }])
    recordings.to_csv(tmp_path / "recordings.csv", index=False)
    segments.to_csv(tmp_path / "segments.csv", index=False)
    cards, wide = _cards(evidence, config, recordings=recordings, segments=segments)
    states.build_state_cards(
        tmp_path / "evidence.csv", tmp_path / "recordings.csv", tmp_path / "segments.csv",
        config, tmp_path / "cards.csv", tmp_path / "wide.csv",
    )
    pd.testing.assert_frame_equal(cards, pd.read_csv(tmp_path / "cards.csv"), check_dtype=False)
    pd.testing.assert_frame_equal(
        wide, pd.read_csv(tmp_path / "wide.csv"), check_dtype=False, check_names=False,
    )
    pd.testing.assert_frame_equal(evidence, original)
    assert cards.set_index("subject_id").loc["case", "trace_resolution"] == "segment"
    no_segments, _ = _cards(evidence, config)
    pd.testing.assert_frame_equal(
        cards.drop(columns=["evidence_segments", "trace_resolution"]),
        no_segments.drop(columns=["evidence_segments", "trace_resolution"]),
    )
    assert no_segments["evidence_segments"].eq("[]").all()


def test_frame_helper_keeps_model_and_report_scores_separate_and_parses_flags(tmp_path: Path):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects(), metrics)
    evidence = evidence[evidence["subject_id"].eq("case")].copy()
    # Two usable metrics, one model-only; two missing reportable metrics.
    evidence["missing"] = evidence["metric_id"].isin(["pause_mean_sec", "pause_p90_sec"])
    evidence["reliability"] = 1.0
    evidence["directional_z"] = evidence["metric_id"].map({"silence_fraction": 2.0, "long_pause_rate_min": -20.0}).fillna(0.0)
    evidence["report_permission"] = ~evidence["metric_id"].eq("long_pause_rate_min")
    evidence["missing"] = evidence["missing"].astype(str)
    evidence["report_permission"] = evidence["report_permission"].astype(str)
    cards, wide = _cards(evidence, config)
    card = cards.iloc[0]
    assert card["state_z"] == pytest.approx(-1.5)
    assert card["raw_state_z"] == pytest.approx(-9.0)
    assert card["report_state_z"] == pytest.approx(2.0)
    assert card["confidence"] == pytest.approx(0.6)
    assert card["report_confidence"] == pytest.approx(0.3 / 0.7)
    assert card["missing_fraction"] == pytest.approx(0.5)
    assert wide.iloc[0]["state_S01"] == pytest.approx(-1.5)
    assert {item["metric_id"] for item in json.loads(card["supporting_metrics"])} == {"silence_fraction"}
    assert json.loads(card["counter_evidence"]) == []


@pytest.mark.parametrize("target_value", [np.inf, -np.inf, np.nan, 0.8])
def test_fold_recalibration_refreshes_availability_and_uses_shared_aggregation(tmp_path: Path, target_value):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects(), metrics)
    evidence.loc[evidence["subject_id"].eq("case") & evidence["metric_id"].eq("silence_fraction"), "value"] = target_value
    calibrated = recalibrate_metric_evidence_frame(
        evidence, reference_subject_ids={"hc1", "hc2"}, target_subject_ids={"case"},
    )
    assert calibrated["report_permission"].all()
    silence = calibrated[calibrated["metric_id"].eq("silence_fraction")].iloc[0]
    assert bool(silence["missing"]) == (not np.isfinite(target_value))
    assert "evidence_status" in calibrated
    assert silence["evidence_status"] == ("available" if np.isfinite(target_value) else "missing")
    cards, _ = _cards(calibrated, config)
    assert cards.iloc[0]["missing_fraction"] == pytest.approx(0.75 if np.isfinite(target_value) else 1.0)
    assert np.isfinite(cards.iloc[0]["state_z"])
