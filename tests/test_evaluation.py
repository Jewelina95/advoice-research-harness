from __future__ import annotations

import pandas as pd

from advoice.evaluation import evaluate_predictions


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
    result = evaluate_predictions(frame, 10)
    assert result["accuracy"] == 1.0
    assert result["macro_f1"] == 1.0
    assert result["macro_auroc_ovr"] == 1.0
    assert result["per_class"]["AD"]["sensitivity"] == 1.0

