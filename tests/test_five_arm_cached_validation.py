from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).parents[1] / "scripts" / "run_five_arm_cached_validation.py"
SPEC = spec_from_file_location("five_arm_cached_validation", SCRIPT)
MODULE = module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_log_pool_zero_auxiliary_weight_preserves_base() -> None:
    base = np.array([[0.8, 0.2], [0.3, 0.7]])
    auxiliary = np.array([[0.1, 0.9], [0.9, 0.1]])
    pooled = MODULE.log_pool((base, 1.0), (auxiliary, 0.0))
    np.testing.assert_allclose(pooled, base)


def test_metrics_respect_hc_ad_probability_order() -> None:
    y = np.array([0, 1])
    probability = np.array([[0.9, 0.1], [0.2, 0.8]])
    result = MODULE.metrics(y, probability)
    assert result["accuracy"] == 1.0
    assert result["auroc"] == 1.0
