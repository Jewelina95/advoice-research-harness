from __future__ import annotations

import numpy as np

from advoice.cognitive_prototypes import (
    build_case_prototype_reference,
    fit_cognitive_prototypes,
    predict_cognitive_prototypes,
)


def _training_example() -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    values = np.asarray(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [1.8, 0.9],
            [2.0, 1.1],
            [3.8, 2.8],
            [4.0, 3.0],
        ],
        dtype=float,
    )
    reliability = np.ones_like(values)
    labels = np.asarray(["HC", "HC", "MCI", "MCI", "AD", "AD"])
    return values, reliability, labels, ["S01_overall", "S07_active_task"]


def test_cognitive_prototypes_separate_screening_and_staging() -> None:
    values, reliability, labels, names = _training_example()
    model = fit_cognitive_prototypes(
        values,
        reliability,
        labels,
        names,
        minimum_class_support=2,
    )
    probability, details = predict_cognitive_prototypes(
        model,
        np.asarray([[0.0, 0.0], [3.9, 2.9]]),
        np.ones((2, 2)),
    )
    assert probability[0, 0] > probability[0, 1:].sum()
    assert probability[1, 2] > probability[1, 1]
    assert details[0]["screening_evidence"] < 0
    assert details[1]["staging_evidence"] > 0


def test_prototype_reference_contains_statistics_not_probabilities() -> None:
    values, reliability, labels, names = _training_example()
    model = fit_cognitive_prototypes(
        values,
        reliability,
        labels,
        names,
        minimum_class_support=2,
    )
    reference = build_case_prototype_reference(
        model,
        values[2],
        reliability[2],
        maximum_states=2,
    )
    serialized = str(reference).lower()
    assert "prob" not in serialized
    assert "prevalence" not in serialized
    assert reference["screening_state_references"]
    assert reference["staging_state_references"]
    assert reference["staging_reference_available"] is True


def test_unreliable_states_do_not_contribute_to_prototype_evidence() -> None:
    values, reliability, labels, names = _training_example()
    model = fit_cognitive_prototypes(
        values,
        reliability,
        labels,
        names,
        minimum_class_support=2,
    )
    probability, details = predict_cognitive_prototypes(
        model,
        np.asarray([[4.0, 3.0]]),
        np.zeros((1, 2)),
    )
    np.testing.assert_allclose(probability[0], [0.5, 0.25, 0.25])
    assert details[0]["screening_usable_states"] == 0
    assert details[0]["staging_usable_states"] == 0
