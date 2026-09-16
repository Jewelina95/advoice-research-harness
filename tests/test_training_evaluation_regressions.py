from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import advoice.condition_c as condition_c
import advoice.evaluation as evaluation
from advoice.evidence import build_metric_evidence
from advoice.models import _oof_reliability
from advoice.states import build_fold_calibrated_state_frame


def _training_inputs(root: Path, labels: list[str], train_size: int = 24) -> tuple[list[Path], dict, dict]:
    rows = []
    for index in range(train_size + 6):
        rows.append({
            "dataset_id": "synthetic",
            "subject_id": str(index),
            "label": labels[index % 2],
            "split": "train" if index < train_size else "test",
            "pause_metric": (-2.0 if index % 2 == 0 else 2.0) + index % 3 * 0.05,
            "audio_reliability": 0.95,
            "text_reliability": 0.95,
        })
    features = pd.DataFrame(rows)
    features_path = root / "features.csv"
    features.to_csv(features_path, index=False)
    transcripts_path = root / "transcripts.csv"
    pd.DataFrame({
        "subject_id": features.subject_id,
        "transcript": ["clear complete description" if i % 2 == 0 else "vague repeated hesitation" for i in range(len(features))],
    }).to_csv(transcripts_path, index=False)
    evidence_path = root / "evidence.csv"
    build_metric_evidence(features_path, {"metrics": [{
        "id": "pause_metric", "state": "S01", "branch": "speech_behavior",
        "direction": 1, "role": "clinical", "reliability": 1.0,
        "confounds": [], "report_permission": True,
    }]}, evidence_path, root / "reference.json", reference_label=labels[0])
    states = features[condition_c.IDENTITY].copy()
    states["state_S01"] = features.pause_metric
    states["rel_S01"] = 0.95
    states_path = root / "states.csv"
    states.to_csv(states_path, index=False)
    cards = features[condition_c.IDENTITY].copy()
    for column, value in {
        "state_id": "S01", "state_base_id": "S01", "task_scope": "overall",
        "state_name_zh": "Pause burden", "branch": "speech_behavior",
        "confidence": 0.95, "missing_fraction": 0.0,
        "supporting_metrics": "[]", "counter_evidence": "[]", "evidence_segments": "[]",
    }.items():
        cards[column] = value
    cards["state_z"] = features.pause_metric
    cards_path = root / "cards.csv"
    cards.to_csv(cards_path, index=False)
    state_config = {"states": [{"id": "S01", "metrics": ["pause_metric"], "weights": [1.0]}]}
    model_config = {
        "labels": labels, "positive_class": labels[-1],
        "state_branches": {"S01": "speech_behavior"},
        "cross_validation": {"folds": 3},
        "ours": {"qc_orthogonalization": {"enabled": False}},
        "condition_c": {"c_grid": [0.1], "alpha_grid": [0.0], "class_offset_grid": [0.0], "max_iter": 500},
    }
    return [features_path, transcripts_path, states_path, evidence_path, cards_path], state_config, model_config


def _train(root: Path, inputs: tuple[list[Path], dict, dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    root.mkdir()
    paths, states, models = inputs
    condition_c.train_condition_c(
        *paths, states, models,
        *[root / name for name in ["predictions.csv", "base.csv", "ablations.csv", "interventions.csv", "workspaces.jsonl", "contributions.csv", "model.joblib", "metadata.json"]],
        agent_calibration_predictions_path=root / "calibration.csv",
        agent_calibration_workspaces_path=root / "calibration.jsonl",
    )
    return pd.read_csv(root / "predictions.csv"), pd.read_csv(root / "calibration.csv")


@pytest.mark.parametrize("train_size,folds", [(24, 3), (35, 5)])
def test_progression_calibration_uses_task_reference_label(tmp_path: Path, train_size: int, folds: int) -> None:
    inputs = _training_inputs(tmp_path, ["no_decline", "decline"], train_size)
    inputs[2]["cross_validation"]["folds"] = folds
    predictions, calibration = _train(tmp_path / "run", inputs)
    assert len(predictions) == 6
    assert len(calibration) == train_size // folds
    assert set(calibration.label) == {"no_decline", "decline"}
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    provenance = metadata["oof_provenance"]
    assert provenance["status"] == "selection_dependent_development_predictions"
    assert provenance["selection_independent"] is False
    assert provenance["dedicated_calibration_holdout"] is False
    assert "expert_eligibility" in provenance["selection_dependencies"]
    assert calibration["oof_status"].eq(provenance["status"]).all()
    assert calibration["selection_independent"].eq(False).all()
    for line in (tmp_path / "run" / "calibration.jsonl").read_text().splitlines():
        assert json.loads(line)["oof_provenance"] == provenance


def test_deployment_temperature_cannot_rewrite_calibration_prior(tmp_path: Path, monkeypatch) -> None:
    inputs = _training_inputs(tmp_path, ["HC", "AD"])
    original = condition_c._fit_temperature
    deployment_temperature = 1.0

    def fit(probability, y, labels):
        if len(y) == 24:
            return deployment_temperature
        return original(probability, y, labels)

    monkeypatch.setattr(condition_c, "_fit_temperature", fit)
    predictions_one, calibration_one = _train(tmp_path / "one", inputs)
    deployment_temperature = 5.0
    predictions_five, calibration_five = _train(tmp_path / "five", inputs)
    columns = ["prob_HC", "prob_AD"]
    np.testing.assert_allclose(calibration_one[columns], calibration_five[columns])
    assert not np.allclose(predictions_one[columns], predictions_five[columns])


def test_evaluation_rejects_duplicate_subjects_before_scoring(tmp_path: Path, monkeypatch) -> None:
    predictions_path = tmp_path / "predictions.csv"
    pd.DataFrame({"subject_id": ["same", "same"], "predicted_label": ["HC", "AD"]}).to_csv(predictions_path, index=False)

    def unexpected_score(*args, **kwargs):
        pytest.fail("Duplicate patients reached scoring")

    monkeypatch.setattr(evaluation, "evaluate_predictions", unexpected_score)
    with pytest.raises(ValueError, match="subject_id.*unique"):
        evaluation.run_evaluation(
            {"Ours": predictions_path}, *[tmp_path / "unused" for _ in range(12)],
            {"labels": ["HC", "AD"], "ece_bins": 5, "bootstrap_iterations": 2},
            *[tmp_path / name for name in ["a.csv", "b.csv", "summary.json"]],
        )


def test_bootstrap_still_allows_repeated_subject_draws() -> None:
    frame = pd.DataFrame({
        "subject_id": ["1", "2", "3", "4"], "label": ["HC", "HC", "AD", "AD"],
        "predicted_label": ["HC", "AD", "AD", "AD"],
        "prob_HC": [0.8, 0.4, 0.2, 0.1], "prob_AD": [0.2, 0.6, 0.8, 0.9],
    })
    intervals = evaluation.bootstrap_intervals(frame, 5, 4, ["HC", "AD"], "AD")
    assert intervals["__effective_resamples__"] == [4.0, 4.0]


def test_oof_reliability_uses_the_prediction_fold_reference() -> None:
    frame = pd.DataFrame({"rel_S01": [1.0, 1.0, 1.0, 1.0]})
    splits = [(np.array([2, 3]), np.array([0, 1])), (np.array([0, 1]), np.array([2, 3]))]
    fold_frames = [frame.copy(), frame.copy()]
    fold_frames[0].loc[[0, 1], "rel_S01"] = 0.0
    fold_frames[1].loc[[2, 3], "rel_S01"] = 0.5
    np.testing.assert_array_equal(
        _oof_reliability(frame, ["rel_S01"], splits, fold_frames), [0.0, 0.0, 0.5, 0.5]
    )
    np.testing.assert_array_equal(_oof_reliability(frame, ["rel_S01"], splits, None), np.ones(4))


def test_nested_branches_recalibrate_inside_inner_folds(tmp_path: Path, monkeypatch) -> None:
    inputs = _training_inputs(tmp_path, ["HC", "AD"])
    evidence = pd.read_csv(inputs[0][3], dtype={"subject_id": str})
    original = condition_c._fit_branch
    nested_checks = []

    def fit(frame, y, c_grid, max_iter, splitter, labels, specification, qc_alpha, fold_frames=None):
        if len(frame) < 24 and specification["kind"] == "clinical_state":
            assert fold_frames is not None
            for (fit_index, _), fold_frame in zip(splitter.split(frame, y), fold_frames, strict=True):
                expected = build_fold_calibrated_state_frame(
                    evidence, inputs[1], set(frame.iloc[fit_index].subject_id), labels[0]
                ).set_index("subject_id").loc[frame.subject_id]
                np.testing.assert_allclose(fold_frame[["state_S01", "rel_S01"]], expected[["state_S01", "rel_S01"]])
                assert fold_frame.subject_id.tolist() == frame.subject_id.tolist()
            nested_checks.append(len(frame))
        return original(frame, y, c_grid, max_iter, splitter, labels, specification, qc_alpha, fold_frames)

    monkeypatch.setattr(condition_c, "_fit_branch", fit)
    _train(tmp_path / "nested", inputs)
    assert nested_checks == [16, 16, 16]


def _evaluation_frame() -> pd.DataFrame:
    return pd.DataFrame({
        "subject_id": ["1", "2", "3", "4"],
        "label": ["HC", "HC", "AD", "AD"],
        "predicted_label": ["HC", "HC", "AD", "AD"],
        "prob_HC": [0.8, 0.7, 0.2, 0.1],
        "prob_AD": [0.2, 0.3, 0.8, 0.9],
    })


@pytest.mark.parametrize("invalid", [
    [np.nan, 0.5], [np.inf, 0.5], [-np.inf, 0.5],
    [-0.1, 1.1], [-1e-12, 1.0], [1.1, 0.0],
    [0.0, 0.0], [0.2, 0.2], [0.8, 0.8],
])
@pytest.mark.parametrize("entrypoint", ["point", "bootstrap", "paired"])
def test_invalid_probabilities_fail_before_scoring(invalid, entrypoint) -> None:
    frame = _evaluation_frame()
    frame.loc[0, ["prob_HC", "prob_AD"]] = invalid
    with pytest.raises(ValueError, match="[Pp]robabilit"):
        if entrypoint == "point":
            evaluation.evaluate_predictions(frame, 5, ["HC", "AD"], "AD")
        elif entrypoint == "bootstrap":
            evaluation.bootstrap_intervals(frame, 5, 0, ["HC", "AD"], "AD")
        else:
            evaluation.paired_prediction_comparison(
                _evaluation_frame(), frame, 5, 0, ["HC", "AD"], "AD"
            )


def test_probability_validation_accepts_boundaries_and_does_not_normalize() -> None:
    frame = _evaluation_frame()
    frame.loc[0, ["prob_HC", "prob_AD"]] = [1.0, 0.0]
    frame.loc[1, ["prob_HC", "prob_AD"]] = [0.7, 0.30000001]
    before = frame.copy(deep=True)
    values = evaluation._validated_probabilities(frame, ["HC", "AD"])
    np.testing.assert_array_equal(values, before[["prob_HC", "prob_AD"]].to_numpy())
    pd.testing.assert_frame_equal(frame, before)


@pytest.mark.parametrize("column", ["label", "predicted_label"])
@pytest.mark.parametrize("invalid", ["UNKNOWN", "", " ", None, np.nan])
def test_evaluation_rejects_labels_outside_configured_classes(column, invalid, monkeypatch) -> None:
    frame = _evaluation_frame()
    frame.loc[0, column] = invalid

    def unexpected_confusion(*args, **kwargs):
        pytest.fail("Invalid class reached confusion-matrix scoring")

    monkeypatch.setattr(evaluation, "confusion_matrix", unexpected_confusion)
    with pytest.raises(ValueError, match=f"{column}.*configured labels"):
        evaluation.evaluate_predictions(frame, 5, ["HC", "AD"], "AD")


@pytest.mark.parametrize("blank", ["", " ", "\t\n"])
def test_artifact_guard_rejects_blank_subject_ids(blank) -> None:
    frame = _evaluation_frame()
    frame.loc[0, "subject_id"] = blank
    with pytest.raises(ValueError, match="subject_id.*blank"):
        evaluation._validate_subject_cohort(frame, "Ours")
