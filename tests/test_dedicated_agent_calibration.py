import json

import joblib
import numpy as np
import pandas as pd

import advoice.condition_c as condition_c
from test_training_evaluation_regressions import _training_inputs, _train


def test_calibration_subjects_are_excluded_from_all_model_fits(tmp_path, monkeypatch):
    inputs = _training_inputs(tmp_path, ["HC", "AD"], train_size=40)
    inputs[2]["condition_c"]["dedicated_calibration"] = {
        "enabled": True, "fraction": .25, "minimum_training_subjects": 30,
        "minimum_calibration_subjects": 10, "seed": 17,
    }
    seen = []
    original = condition_c._fit_branch
    def fit(frame, *args, **kwargs):
        seen.extend(frame.subject_id.astype(str))
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(condition_c, "_fit_branch", fit)
    predictions, calibration = _train(tmp_path / "run", inputs)
    assert len(predictions) == 6
    assert len(calibration) == 10
    assert not set(calibration.subject_id.astype(str)) & set(seen)
    assert calibration.selection_independent.all()
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    partition = metadata["dedicated_calibration"]
    assert set(partition["calibration_subject_ids"]) == set(calibration.subject_id.astype(str))
    assert not set(partition["fit_subject_ids"]) & set(partition["calibration_subject_ids"])
    bundle = joblib.load(tmp_path / "run" / "model.joblib")
    assert bundle["dedicated_calibration"] == partition


def test_external_test_labels_do_not_change_model_parameters(tmp_path):
    inputs = _training_inputs(tmp_path, ["HC", "AD"], train_size=40)
    inputs[2]["condition_c"]["dedicated_calibration"] = {
        "enabled": True, "fraction": .25, "minimum_training_subjects": 30,
        "minimum_calibration_subjects": 10, "seed": 17,
    }
    first, cal1 = _train(tmp_path / "one", inputs)
    for file in [inputs[0][0], inputs[0][2], inputs[0][3], inputs[0][4]]:
        frame = pd.read_csv(file)
        frame.loc[frame.split.eq("test"), "label"] = frame.loc[frame.split.eq("test"), "label"].map({"HC": "AD", "AD": "HC"})
        frame.to_csv(file, index=False)
    second, cal2 = _train(tmp_path / "two", inputs)
    np.testing.assert_allclose(first[["prob_HC", "prob_AD"]], second[["prob_HC", "prob_AD"]])
    np.testing.assert_allclose(cal1[["prob_HC", "prob_AD"]], cal2[["prob_HC", "prob_AD"]])


def test_insufficient_calibration_preserves_supervised_training_subjects(tmp_path, monkeypatch):
    inputs = _training_inputs(tmp_path, ["HC", "AD"], train_size=40)
    inputs[2]["condition_c"]["dedicated_calibration"] = {
        "enabled": True, "fraction": .25, "minimum_training_subjects": 30,
        "minimum_calibration_subjects": 12, "seed": 17,
    }
    seen = set()
    original = condition_c._fit_branch
    def fit(frame, *args, **kwargs):
        seen.update(frame.subject_id.astype(str))
        return original(frame, *args, **kwargs)
    monkeypatch.setattr(condition_c, "_fit_branch", fit)
    _train(tmp_path / "run", inputs)
    assert seen == {str(index) for index in range(40)}
    metadata = json.loads((tmp_path / "run" / "metadata.json").read_text())
    assert metadata["dedicated_calibration"]["reason"] == "insufficient_calibration_subjects"
