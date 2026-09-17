from __future__ import annotations

import math
import struct
from types import MappingProxyType

import pytest

from advoice.authority_joint_fusion import (
    AuthorityJointFusionConfig,
    AuthorityJointFusionError,
    fuse_authority_joint,
)


LABELS = ("HC", "MCI", "AD")


def _config(**overrides: float) -> AuthorityJointFusionConfig:
    values = {
        "state_strength": 0.5,
        "agent_strength": 0.75,
        "max_abs_state_delta": 2.0,
        "ordinal_temperature": 1.0,
    }
    values.update(overrides)
    return AuthorityJointFusionConfig(**values)


def _fuse(**overrides):
    values = {
        "frozen_probabilities": {"HC": 0.6, "MCI": 0.3, "AD": 0.1},
        "pre_state_probabilities": {"HC": 0.4, "MCI": 0.4, "AD": 0.2},
        "post_state_probabilities": {"HC": 0.2, "MCI": 0.3, "AD": 0.5},
        "blind_ordinal_scores": {"HC": 0, "MCI": 2, "AD": 4},
        "class_order": LABELS,
        "config": _config(),
        "provenance": {"case_hash": "a" * 64, "blind_review_hash": "b" * 64},
    }
    values.update(overrides)
    return fuse_authority_joint(**values)


def _bits(values) -> tuple[bytes, ...]:
    return tuple(struct.pack("!d", item) for item in values)


def test_zero_strengths_return_frozen_probabilities_bit_for_bit() -> None:
    frozen = {"HC": 0.7, "MCI": 0.2, "AD": 0.1}
    result = _fuse(
        frozen_probabilities=frozen,
        config=_config(state_strength=0.0, agent_strength=0.0),
    )

    assert _bits(result.fused_probabilities.values()) == _bits(frozen.values())
    assert result.frozen_parity is True
    assert result.predicted_label == "HC"


def test_equal_state_and_equal_ordinal_components_are_strictly_neutral() -> None:
    frozen = {"HC": 0.7, "MCI": 0.2, "AD": 0.1}
    pre = {"HC": 0.2, "MCI": 0.3, "AD": 0.5}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities=pre,
        post_state_probabilities=dict(pre),
        blind_ordinal_scores={"HC": 2, "MCI": 2, "AD": 2},
    )

    assert _bits(result.fused_probabilities.values()) == _bits(frozen.values())
    assert result.state_component_neutral is True
    assert result.agent_component_neutral is True
    assert tuple(result.agent_log_evidence.values()) == (0.0, 0.0, 0.0)
    assert tuple(result.agent_equal_prior_likelihood.values()) == pytest.approx((1 / 3, 1 / 3, 1 / 3))


def test_joint_log_linear_arithmetic_uses_centered_state_and_equal_prior_agent_likelihood() -> None:
    result = _fuse()
    state_raw = tuple(
        math.log(after) - math.log(before)
        for before, after in zip((0.4, 0.4, 0.2), (0.2, 0.3, 0.5), strict=True)
    )
    state = tuple(value - sum(state_raw) / 3 for value in state_raw)
    agent = (-2.0, 0.0, 2.0)
    logits = tuple(
        math.log(base) + 0.5 * state_value + 0.75 * agent_value
        for base, state_value, agent_value in zip((0.6, 0.3, 0.1), state, agent, strict=True)
    )
    maximum = max(logits)
    expected = tuple(math.exp(item - maximum) for item in logits)
    expected = tuple(item / sum(expected) for item in expected)

    assert tuple(result.state_log_evidence.values()) == pytest.approx(state)
    assert tuple(result.agent_log_evidence.values()) == pytest.approx(agent)
    assert tuple(result.agent_equal_prior_likelihood.values()) == pytest.approx(_softmax(agent))
    assert tuple(result.fused_probabilities.values()) == pytest.approx(expected)
    assert result.predicted_label == "AD"


def _softmax(values: tuple[float, ...]) -> tuple[float, ...]:
    maximum = max(values)
    weights = tuple(math.exp(item - maximum) for item in values)
    return tuple(item / sum(weights) for item in weights)


def test_state_delta_is_bounded_and_zero_probabilities_remain_numerically_stable() -> None:
    result = _fuse(
        frozen_probabilities={"HC": 1.0, "MCI": 0.0, "AD": 0.0},
        pre_state_probabilities={"HC": 1.0, "MCI": 0.0, "AD": 0.0},
        post_state_probabilities={"HC": 0.0, "MCI": 0.0, "AD": 1.0},
        blind_ordinal_scores={"HC": 0, "MCI": 0, "AD": 4},
        config=_config(max_abs_state_delta=0.25),
    )

    assert all(math.isfinite(item) for item in result.state_log_evidence.values())
    assert all(abs(item) <= 0.25 for item in result.bounded_state_log_evidence.values())
    assert math.isclose(sum(result.fused_probabilities.values()), 1.0)
    assert all(math.isfinite(item) for item in result.fused_probabilities.values())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"frozen_probabilities": {"HC": 0.5, "MCI": 0.5}}, "frozen_probabilities must contain"),
        ({"pre_state_probabilities": {"HC": 0.5, "MCI": 0.3, "AD": 0.3}}, "must sum"),
        ({"blind_ordinal_scores": {"HC": 0, "MCI": 2, "AD": 5}}, "0 through 4"),
        ({"blind_ordinal_scores": {"HC": 0, "MCI": 2}}, "must contain"),
        ({"class_order": ("HC", "HC")}, "duplicate"),
    ],
)
def test_invalid_or_incomplete_class_inputs_fail_closed(kwargs, message: str) -> None:
    with pytest.raises(AuthorityJointFusionError, match=message):
        _fuse(**kwargs)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"state_strength": -0.1}, "state_strength"),
        ({"agent_strength": -0.1}, "agent_strength"),
        ({"max_abs_state_delta": -0.1}, "max_abs_state_delta"),
        ({"ordinal_temperature": 0.0}, "ordinal_temperature"),
    ],
)
def test_invalid_configuration_fails_closed(kwargs, message: str) -> None:
    with pytest.raises(AuthorityJointFusionError, match=message):
        _config(**kwargs)


def test_audit_hash_covers_inputs_config_and_provenance_deterministically() -> None:
    left = _fuse()
    same = _fuse()
    changed = _fuse(provenance={"case_hash": "c" * 64, "blind_review_hash": "b" * 64})

    assert left.input_hash == same.input_hash
    assert left.audit_hash == same.audit_hash
    assert left.input_hash != changed.input_hash
    assert left.audit_hash != changed.audit_hash
    assert left.to_dict()["audit_hash"] == left.audit_hash
    assert isinstance(left.provenance, MappingProxyType)


def test_provenance_rejects_truth_or_target_labels() -> None:
    with pytest.raises(AuthorityJointFusionError, match="truth or target labels"):
        _fuse(provenance={"truth": "AD"})
