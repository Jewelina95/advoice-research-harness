import numpy as np
import pytest

from advoice.cognitive_extension import benchmark_metrics, bounded_cognitive_extension


def test_bounded_extension_is_normalized_and_uses_fixed_weight() -> None:
    backbone = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.6, 0.3]])
    cognition = np.asarray([[0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    result = bounded_cognitive_extension(backbone, cognition, cognitive_weight=0.2)
    np.testing.assert_allclose(result, 0.8 * backbone + 0.2 * cognition)
    np.testing.assert_allclose(result.sum(axis=1), 1.0)


def test_bounded_extension_rejects_unbounded_correction() -> None:
    probability = np.asarray([[0.8, 0.1, 0.1]])
    with pytest.raises(ValueError):
        bounded_cognitive_extension(probability, probability, cognitive_weight=0.21)


def test_benchmark_metrics_reports_published_endpoints() -> None:
    probability = np.asarray(
        [[0.8, 0.1, 0.1], [0.1, 0.8, 0.1], [0.1, 0.1, 0.8]]
    )
    metrics = benchmark_metrics(np.asarray([0, 1, 2]), probability)
    assert metrics["micro_f1"] == 1.0
    assert metrics["micro_auroc_ovr"] == 1.0
    assert metrics["micro_auprc"] == 1.0
