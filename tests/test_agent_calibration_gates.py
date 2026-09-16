from __future__ import annotations

import itertools

import numpy as np
import pytest

from advoice import cognitive_agent


LABELS = np.asarray(["HC", "MCI", "AD"])


def _fit(truth, base, **kwargs):
    n = len(truth)
    return cognitive_agent.fit_agent_two_stage_strengths(
        np.asarray(truth), base, np.full((n, 2), 0.5), np.full((n, 2), 0.5),
        np.ones(n), np.ones(n), np.ones(n), np.ones(n), [0.0, 1.0],
        **kwargs,
    )


@pytest.mark.parametrize("present", [
    subset for size in range(3) for subset in itertools.combinations(LABELS, size)
])
def test_missing_calibration_class_fails_closed_without_roc_crash(present) -> None:
    truth = np.asarray(list(present) * 3)
    result = _fit(truth, np.full((len(truth), 3), 1.0 / 3.0))
    assert result["selection_status"] == "failed_closed_missing_classes"
    assert result["missing_classes"] == sorted(set(LABELS) - set(present))
    assert result["selected_screening_strength"] == 0.0
    assert result["selected_staging_strength"] == 0.0
    assert not cognitive_agent._test_agent_gate_passed(
        "openai_api", 1.0, 1.0, {"status": "completed", **result},
    )


@pytest.mark.parametrize("margins", [
    {},
    {"log_loss_noninferiority_margin": 100.0, "brier_noninferiority_margin": 0.0},
    {"log_loss_noninferiority_margin": 0.0, "brier_noninferiority_margin": 100.0},
])
def test_f1_auroc_gain_cannot_override_either_probability_score_gate(monkeypatch, margins) -> None:
    truth = np.tile(LABELS, 2)
    one_hot = np.eye(3)[np.tile(np.arange(3), 2)]
    base = one_hot * 0.98 + 0.01
    base[0] = [0.4, 0.59, 0.01]
    candidate = one_hot * 0.01 + 0.33

    def fuse(*args):
        return base.copy() if args[5] == args[6] == 0.0 else candidate.copy()

    monkeypatch.setattr(cognitive_agent, "fuse_two_stage_evidence", fuse)
    result = _fit(truth, base, minimum_macro_f1_gain=0.01, **margins)
    assert result["selection_status"] == "failed_closed_no_joint_gain"
    improved = result["candidates"][1]
    assert improved["macro_f1"] > result["baseline_macro_f1"] + 0.01
    assert improved["macro_auroc"] >= result["baseline_macro_auroc"]
    assert improved["log_loss"] > result["baseline_log_loss"]
    assert improved["brier"] > result["baseline_brier"]
    assert result["selected_screening_strength"] == result["selected_staging_strength"] == 0.0


def test_joint_gain_retains_existing_f1_requirement_and_reports_score_definitions(monkeypatch) -> None:
    truth = np.tile(LABELS, 2)
    one_hot = np.eye(3)[np.tile(np.arange(3), 2)]
    base = np.full((len(truth), 3), 1.0 / 3.0)
    candidate = one_hot * 0.7 + 0.1

    def fuse(*args):
        return base.copy() if args[5] == args[6] == 0.0 else candidate.copy()

    monkeypatch.setattr(cognitive_agent, "fuse_two_stage_evidence", fuse)
    result = _fit(truth, base, minimum_macro_f1_gain=0.01)
    assert result["selection_status"] == "validated_joint_gain"
    assert result["log_loss_noninferiority_margin"] == 0.0
    assert result["brier_noninferiority_margin"] == 0.0
    assert result["brier_definition"] == "mean_sum_squared_class_probability_error"
    assert result["baseline_log_loss"] == pytest.approx(np.log(3.0))
    assert result["baseline_brier"] == pytest.approx(2.0 / 3.0)
    result = _fit(truth, base, minimum_macro_f1_gain=0.9)
    assert result["selection_status"] == "failed_closed_no_joint_gain"


@pytest.mark.parametrize("name", ["log_loss_noninferiority_margin", "brier_noninferiority_margin"])
@pytest.mark.parametrize("value", [-0.01, float("nan"), float("inf")])
def test_probability_score_margins_must_be_finite_and_nonnegative(name, value) -> None:
    with pytest.raises(ValueError, match="finite and nonnegative"):
        _fit(LABELS, np.full((3, 3), 1.0 / 3.0), **{name: value})
