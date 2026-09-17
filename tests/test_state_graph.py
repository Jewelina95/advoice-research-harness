from __future__ import annotations

import json

import numpy as np
import pandas as pd

from advoice.state_graph import build_state_graph_frame


def _states(metrics: list[str]) -> dict:
    return {
        "metric_contribution_clip_z": 5.0,
        "states": [
            {
                "id": "S01",
                "name_zh": "pause burden",
                "branch": "speech_behavior",
                "clinical_question": "pause?",
                "metrics": metrics,
                "weights": [1.0] * len(metrics),
            }
        ],
    }


def _families(metrics: list[str], *, prior: float = 1.0) -> dict:
    return {
        "defaults": {
            "robust_huber_delta": 1.5,
            "task_residual_prior_precision": prior,
            "min_tasks_for_residual": 2,
        },
        "states": {
            "S01": {
                "families": [
                    {"id": "pause", "metrics": metrics, "budget": 1.0}
                ]
            }
        },
    }


def _row(
    metric: str,
    score: float,
    task: str = "overall",
    *,
    reliability: float = 0.8,
    missing: bool = False,
    status: str = "available",
    report: bool = True,
    confounds: str = "[]",
    segments: str = "[]",
) -> dict:
    return {
        "dataset_id": "d",
        "subject_id": "1",
        "label": "HC",
        "split": "test",
        "metric_id": metric,
        "metric_instance_id": f"task_{task}__{metric}" if task != "overall" else metric,
        "task_scope": task,
        "value": score,
        "reference_label": "HC",
        "reference_median": 0.0,
        "reference_scale": 1.0,
        "cn_train_median": 0.0,
        "robust_z": score,
        "directional_z": score,
        "reliability": reliability,
        "missing": missing,
        "evidence_status": status,
        "report_permission": report,
        "confound_tags": confounds,
        "evidence_segments": segments,
    }


def test_correlated_duplicates_share_one_budget_and_are_order_invariant() -> None:
    single, _ = build_state_graph_frame(
        pd.DataFrame([_row("pause_a", 2.0)]),
        _states(["pause_a"]),
        _families(["pause_a"]),
    )
    duplicate_evidence = pd.DataFrame(
        [_row("pause_a", 2.0), _row("pause_b", 2.0)]
    )
    duplicate, _ = build_state_graph_frame(
        duplicate_evidence,
        _states(["pause_a", "pause_b"]),
        _families(["pause_a", "pause_b"]),
    )
    shuffled, _ = build_state_graph_frame(
        duplicate_evidence.sample(frac=1.0, random_state=9),
        _states(["pause_a", "pause_b"]),
        _families(["pause_a", "pause_b"]),
    )
    single_shared = single.query("graph_level == 'shared'").iloc[0]
    duplicate_shared = duplicate.query("graph_level == 'shared'").iloc[0]
    assert duplicate_shared["state_z"] == single_shared["state_z"] == 2.0
    assert np.isclose(duplicate_shared["confidence"], 0.8)
    assert np.isclose(single_shared["confidence"], 0.8)
    assert duplicate_shared["independent_family_count"] == 1
    pd.testing.assert_frame_equal(duplicate, shuffled)


def test_task_residuals_are_centered_and_shrunk() -> None:
    evidence = pd.DataFrame(
        [_row("pause", value, task, reliability=1.0) for task, value in [("a", 0.0), ("b", 2.0), ("c", 4.0)]]
    )
    cards, _ = build_state_graph_frame(
        evidence,
        _states(["pause"]),
        _families(["pause"], prior=1.0),
    )
    shared = cards.query("graph_level == 'shared'").iloc[0]
    residuals = cards.query("graph_level == 'task_residual'").sort_values("task_scope")
    assert shared["state_z"] == 2.0
    assert np.isclose(residuals["state_z"].sum(), 0.0)
    raw_differences = residuals["task_state_z"] - shared["state_z"]
    assert np.all(np.abs(residuals["state_z"]) <= np.abs(raw_differences))
    assert np.allclose(residuals["residual_shrinkage_factor"], 0.5)


def test_family_score_robustly_limits_one_outlying_metric() -> None:
    evidence = pd.DataFrame(
        [_row("pause_a", 1.0), _row("pause_b", 1.0), _row("pause_c", 20.0)]
    )
    cards, _ = build_state_graph_frame(
        evidence,
        _states(["pause_a", "pause_b", "pause_c"]),
        _families(["pause_a", "pause_b", "pause_c"]),
    )
    shared = cards.query("graph_level == 'shared'").iloc[0]
    assert 1.0 <= shared["state_z"] < 2.0
    assert shared["independent_family_count"] == 1


def test_single_task_residual_is_explicitly_unavailable() -> None:
    cards, wide = build_state_graph_frame(
        pd.DataFrame([_row("pause", 1.5, "cookie", reliability=1.0)]),
        _states(["pause"]),
        _families(["pause"]),
    )
    shared = cards.query("graph_level == 'shared'").iloc[0]
    residual = cards.query("graph_level == 'task_residual'").iloc[0]
    assert shared["available"] and shared["state_z"] == 1.5
    assert not residual["available"]
    assert np.isnan(residual["state_z"])
    assert residual["unavailable_reason"] == "single_task_residual_not_identifiable"
    assert not wide["available_S01__task_cookie_residual"].iloc[0]


def test_unobservable_evidence_stays_unavailable_with_metric_trace() -> None:
    evidence = pd.DataFrame(
        [
            _row("pause", 0.0, "cookie", reliability=0.0, missing=True, status="unobservable"),
            _row("pause", 0.0, "recall", reliability=0.0, missing=True, status="unavailable"),
        ]
    )
    cards, _ = build_state_graph_frame(
        evidence,
        _states(["pause"]),
        _families(["pause"]),
    )
    shared = cards.query("graph_level == 'shared'").iloc[0]
    assert not shared["available"]
    assert shared["evidence_status"] == "unavailable"
    assert shared["unavailable_reason"] == "no_observable_family_evidence"
    trace = json.loads(shared["metric_trace"])
    assert {item["evidence_status"] for item in trace} == {"unavailable", "unobservable"}
    assert cards.query("graph_level == 'task_residual'")["available"].eq(False).all()


def test_support_counterevidence_confounds_and_segments_survive_graph_build() -> None:
    evidence = pd.DataFrame(
        [
            _row(
                "pause_a",
                2.0,
                "cookie",
                confounds='["short_recording"]',
                segments='[{"segment_id":"seg-2"}]',
            ),
            _row(
                "pause_b",
                -1.0,
                "cookie",
                confounds='["asr_noise"]',
                segments='[{"segment_id":"seg-1"}]',
            ),
        ]
    )
    config = {
        "defaults": {"task_residual_prior_precision": 1.0},
        "states": {
            "S01": {
                "families": [
                    {"id": "a", "metrics": ["pause_a"], "budget": 0.5},
                    {"id": "b", "metrics": ["pause_b"], "budget": 0.5},
                ]
            }
        },
    }
    cards, _ = build_state_graph_frame(
        evidence,
        _states(["pause_a", "pause_b"]),
        config,
    )
    shared = cards.query("graph_level == 'shared'").iloc[0]
    assert [item["metric_id"] for item in json.loads(shared["supporting_metrics"])] == ["pause_a"]
    assert [item["metric_id"] for item in json.loads(shared["counter_evidence"])] == ["pause_b"]
    confounds = json.loads(shared["confounds"])
    assert {item["metric_id"] for item in confounds} == {"pause_a", "pause_b"}
    assert [item["segment_id"] for item in json.loads(shared["evidence_segments"])] == ["seg-1", "seg-2"]


def test_legacy_state_config_uses_singleton_families() -> None:
    evidence = pd.DataFrame([_row("a", 1.0), _row("b", 3.0)])
    cards, _ = build_state_graph_frame(evidence, _states(["a", "b"]))
    shared = cards.query("graph_level == 'shared'").iloc[0]
    assert shared["independent_family_count"] == 2
    assert shared["state_z"] == 2.0
