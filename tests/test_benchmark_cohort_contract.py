import runpy
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


evaluate = runpy.run_path(str(Path(__file__).resolve().parents[1] / "scripts/check_prepare_speechcare_gate.py"))["evaluate_prediction_file"]


def test_benchmark_rejects_wrong_cohort_duplicates_and_invalid_probabilities(tmp_path):
    path = tmp_path / "predictions.csv"
    ids = {f"s{i}" for i in range(412)}
    frame = pd.DataFrame({"subject_id": sorted(ids), "label": ["HC", "MCI", "AD", "HC"] * 103,
                          "prob_HC": 0.4, "prob_MCI": 0.3, "prob_AD": 0.3})
    frame.to_csv(path, index=False)
    assert np.isfinite(evaluate(path, ids)["micro_f1"])
    for column, value, message in [
        ("subject_id", "outsider", "cohort"),
        ("subject_id", frame.iloc[1]["subject_id"], "one prediction"),
        ("prob_HC", -0.2, "Invalid class probabilities"),
        ("prob_HC", np.nan, "Invalid class probabilities"),
        ("prob_HC", 0.8, "Invalid class probabilities"),
    ]:
        bad = frame.copy()
        bad.loc[0, column] = value
        bad.to_csv(path, index=False)
        with pytest.raises(ValueError, match=message):
            evaluate(path, ids)
