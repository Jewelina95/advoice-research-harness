import numpy as np
import pandas as pd

from advoice.cognitive_representation_fusion import (
    active_task_state_matrix,
    apply_logit_offsets,
    fit_class_logit_offsets,
    protocol_metrics,
    hierarchical_auxiliary_loss,
)
from advoice.transcripts import repair_utf8_mojibake


def test_active_task_state_matrix_uses_only_observed_task() -> None:
    states = pd.DataFrame(
        {
            "subject_id": ["a", "b"],
            "state_S01": [1.0, 2.0],
            "state_S01__task_picture_description": [3.0, np.nan],
            "state_S01__task_sentence_reading": [np.nan, 4.0],
            "rel_S01": [0.8, 0.7],
            "rel_S01__task_picture_description": [0.6, np.nan],
            "rel_S01__task_sentence_reading": [np.nan, 0.5],
        }
    )
    values, reliability, names = active_task_state_matrix(
        states, {"a": "picture_description", "b": "sentence_reading"}
    )
    assert names == ["S01_overall", "S01_active_task"]
    np.testing.assert_allclose(values, [[1.0, 3.0], [2.0, 4.0]])
    np.testing.assert_allclose(reliability, [[0.8, 0.6], [0.7, 0.5]])


def test_active_task_state_matrix_removes_exact_duplicate_vote() -> None:
    states = pd.DataFrame(
        {
            "subject_id": ["a", "b"],
            "state_S14": [1.0, 2.0],
            "state_S14__task_picture_description": [1.0, 2.0],
            "rel_S14": [0.8, 0.7],
            "rel_S14__task_picture_description": [0.8, 0.7],
        }
    )
    values, reliability, names = active_task_state_matrix(
        states, {"a": "picture_description", "b": "picture_description"}
    )
    assert names == ["S14_overall"]
    np.testing.assert_allclose(values, [[1.0], [2.0]])
    np.testing.assert_allclose(reliability, [[0.8], [0.7]])


def test_protocol_metrics_reports_benchmark_and_clinical_averages() -> None:
    labels = np.asarray(["HC", "MCI", "AD", "HC"])
    probability = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.2, 0.6, 0.2],
            [0.1, 0.2, 0.7],
            [0.6, 0.3, 0.1],
        ]
    )
    metrics = protocol_metrics(labels, probability)
    assert metrics["accuracy"] == 1.0
    assert metrics["micro_f1"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["micro_auroc_ovr"] == 1.0
    assert metrics["macro_auroc_ovo"] == 1.0


def test_hierarchical_auxiliary_loss_rewards_both_clinical_boundaries() -> None:
    import torch

    target = torch.tensor([0, 1, 2])
    good = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]])
    bad = torch.tensor([[0.0, 0.0, 5.0], [5.0, 0.0, 0.0], [0.0, 5.0, 0.0]])
    loss = torch.nn.CrossEntropyLoss()
    assert hierarchical_auxiliary_loss(good, target, loss) < hierarchical_auxiliary_loss(
        bad, target, loss
    )


def test_threshold_offsets_are_fit_without_changing_input_probabilities() -> None:
    labels = np.asarray(["HC", "HC", "MCI", "AD"])
    probability = np.asarray(
        [
            [0.51, 0.45, 0.04],
            [0.52, 0.44, 0.04],
            [0.47, 0.49, 0.04],
            [0.20, 0.20, 0.60],
        ]
    )
    original = probability.copy()
    offsets, metrics = fit_class_logit_offsets(labels, probability)
    adjusted = apply_logit_offsets(probability, offsets)
    np.testing.assert_allclose(probability, original)
    np.testing.assert_allclose(adjusted.sum(axis=1), 1.0)
    assert metrics["micro_f1"] >= protocol_metrics(labels, probability)["micro_f1"]


def test_repair_utf8_mojibake_is_conservative() -> None:
    assert repair_utf8_mojibake("RocÃ­n") == "Rocín"
    assert repair_utf8_mojibake("texto español") == "texto español"
