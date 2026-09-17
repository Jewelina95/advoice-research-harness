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


def _config(**overrides) -> AuthorityJointFusionConfig:
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


def test_three_class_log_linear_arithmetic_uses_screening_evidence() -> None:
    screening_delta = math.log(0.8 / 0.2) - math.log(0.6 / 0.4)
    state = (-2 * screening_delta / 3, screening_delta / 3, screening_delta / 3)
    agent = (-8 / 3, 4 / 3, 4 / 3)
    result = _fuse(config=_config(conflict_aware_gating=False))
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
    assert result.predicted_label == "MCI"


def test_conflict_aware_gate_keeps_agreement_at_exact_frozen_parity() -> None:
    frozen = {"HC": 0.98, "MCI": 0.01, "AD": 0.01}
    gated = _fuse(
        frozen_probabilities=frozen,
        blind_ordinal_scores={"HC": 4, "MCI": 1, "AD": 0},
        config=_config(state_strength=0.0, agent_strength=1.0),
    )
    assert gated.agent_authority_gate == 0.0
    assert _bits(gated.fused_probabilities.values()) == _bits(frozen.values())


def test_conflict_aware_gate_preserves_clear_counter_evidence() -> None:
    frozen = {"HC": 0.8, "MCI": 0.1, "AD": 0.1}
    result = _fuse(
        frozen_probabilities=frozen,
        blind_ordinal_scores={"HC": 0, "MCI": 0, "AD": 4},
        config=_config(state_strength=0.0, agent_strength=1.0),
    )

    assert result.agent_authority_gate == pytest.approx(1.0)
    assert result.fused_probabilities["MCI"] + result.fused_probabilities["AD"] > 0.5
    assert result.fused_probabilities["MCI"] / result.fused_probabilities["AD"] == pytest.approx(
        frozen["MCI"] / frozen["AD"]
    )


def test_conflict_aware_gate_rejects_weak_counterevidence_against_confident_prior() -> None:
    frozen = {"HC": 0.05, "MCI": 0.05, "AD": 0.9}
    result = _fuse(
        frozen_probabilities=frozen,
        blind_ordinal_scores={"HC": 3, "MCI": 1, "AD": 1},
        config=_config(state_strength=0.0, agent_strength=1.0),
    )

    assert result.agent_authority_gate == 0.0
    assert _bits(result.fused_probabilities.values()) == _bits(frozen.values())


def test_uncertain_prior_does_not_amplify_weak_agent_counterevidence() -> None:
    frozen = {"HC": 0.208014684670914, "MCI": 0.3988898183405164, "AD": 0.3930954969885695}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities=frozen,
        post_state_probabilities=frozen,
        blind_ordinal_scores={"HC": 3, "MCI": 2, "AD": 1},
        config=_config(state_strength=0.0, agent_strength=1.0),
    )

    # Stage ambiguity must not masquerade as HC-versus-impairment uncertainty.
    assert result.agent_authority_gate == 0.0
    assert result.predicted_label in {"MCI", "AD"}
    assert result.fused_probabilities["MCI"] / result.fused_probabilities["AD"] == pytest.approx(
        frozen["MCI"] / frozen["AD"]
    )


def test_state_gate_requires_conflict_and_frozen_uncertainty() -> None:
    confident = {"HC": 0.95, "MCI": 0.03, "AD": 0.02}
    blocked = _fuse(
        frozen_probabilities=confident,
        pre_state_probabilities={"HC": 0.8, "MCI": 0.1, "AD": 0.1},
        post_state_probabilities={"HC": 0.05, "MCI": 0.05, "AD": 0.9},
        blind_ordinal_scores={"HC": 2, "MCI": 2, "AD": 2},
        config=_config(state_strength=1.0, agent_strength=0.0),
    )
    uncertain = {"HC": 0.55, "MCI": 0.3, "AD": 0.15}
    active = _fuse(
        frozen_probabilities=uncertain,
        pre_state_probabilities={"HC": 0.8, "MCI": 0.1, "AD": 0.1},
        post_state_probabilities={"HC": 0.05, "MCI": 0.05, "AD": 0.9},
        blind_ordinal_scores={"HC": 2, "MCI": 2, "AD": 2},
        config=_config(state_strength=1.0, agent_strength=0.0),
    )

    assert blocked.state_authority_gate == 0.0
    assert _bits(blocked.fused_probabilities.values()) == _bits(confident.values())
    assert active.state_authority_gate == 1.0
    assert active.fused_probabilities["MCI"] + active.fused_probabilities["AD"] > 0.45


def test_three_class_state_correction_changes_screening_not_staging_ratio() -> None:
    frozen = {"HC": 0.51, "MCI": 0.36, "AD": 0.13}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities={"HC": 0.2, "MCI": 0.41, "AD": 0.39},
        post_state_probabilities={"HC": 0.04, "MCI": 0.14, "AD": 0.82},
        blind_ordinal_scores={"HC": 1, "MCI": 3, "AD": 2},
        config=_config(state_strength=1.0, agent_strength=1.0),
        channel="structured_multitask",
    )

    assert result.state_authority_gate == 1.0
    assert result.agent_authority_gate == 0.0
    assert result.predicted_label == "MCI"
    assert result.fused_probabilities["MCI"] / result.fused_probabilities["AD"] == pytest.approx(
        frozen["MCI"] / frozen["AD"]
    )


def test_validated_staging_strength_can_change_mci_ad_odds_separately() -> None:
    frozen = {"HC": 0.20, "MCI": 0.40, "AD": 0.40}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities=frozen,
        post_state_probabilities=frozen,
        blind_ordinal_scores={"HC": 0, "MCI": 1, "AD": 4},
        config=_config(
            state_strength=0.0,
            agent_strength=1.0,
            staging_strength=1.0,
            conflict_aware_gating=False,
        ),
    )

    assert result.fused_probabilities["AD"] > result.fused_probabilities["MCI"]


def test_route_specific_cognitive_stages_are_not_dropped() -> None:
    labels = ("HC", "SCD", "MCI", "mild_dementia", "moderate_dementia", "severe_dementia")
    result = fuse_authority_joint(
        frozen_probabilities=dict.fromkeys(labels, 1.0 / len(labels)),
        pre_state_probabilities=dict.fromkeys(labels, 1.0 / len(labels)),
        post_state_probabilities=dict.fromkeys(labels, 1.0 / len(labels)),
        blind_ordinal_scores={label: min(index, 4) for index, label in enumerate(labels)},
        class_order=labels,
        config=_config(state_strength=0.0, agent_strength=0.0),
    )
    assert result.class_order == labels
    assert tuple(result.fused_probabilities) == labels
    assert sum(result.fused_probabilities.values()) == pytest.approx(1.0)


def test_public_speech_is_report_only_and_preserves_frozen_prediction() -> None:
    frozen = {"HC": 0.78, "MCI": 0.12, "AD": 0.10}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities={"HC": 0.8, "MCI": 0.1, "AD": 0.1},
        post_state_probabilities={"HC": 0.02, "MCI": 0.08, "AD": 0.9},
        blind_ordinal_scores={"HC": 0, "MCI": 1, "AD": 4},
        config=_config(state_strength=1.0, agent_strength=1.0),
        channel="public_speech",
    )

    assert result.state_authority_gate == 0.0
    assert result.agent_authority_gate == 0.0
    assert _bits(result.fused_probabilities.values()) == _bits(frozen.values())


def test_state_gate_uses_delta_not_absolute_post_state_class() -> None:
    # Both state outputs favour impairment, but the revision reduces that evidence.
    frozen = {"HC": 0.55, "MCI": 0.30, "AD": 0.15}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities={"HC": 0.05, "MCI": 0.50, "AD": 0.45},
        post_state_probabilities={"HC": 0.30, "MCI": 0.40, "AD": 0.30},
        blind_ordinal_scores={"HC": 4, "MCI": 1, "AD": 0},
    )
    assert result.state_authority_gate == 0.0
    assert result.frozen_parity


def test_counter_delta_can_be_used_before_post_state_crosses_class_boundary() -> None:
    result = _fuse(
        frozen_probabilities={"HC": 0.55, "MCI": 0.30, "AD": 0.15},
        pre_state_probabilities={"HC": 0.95, "MCI": 0.03, "AD": 0.02},
        post_state_probabilities={"HC": 0.60, "MCI": 0.25, "AD": 0.15},
        blind_ordinal_scores={"HC": 0, "MCI": 4, "AD": 3},
    )
    assert result.state_authority_gate > 0.0
    assert result.agent_authority_gate == 0.0
    assert result.fused_probabilities["HC"] < 0.55


@pytest.mark.parametrize("scores", [{"HC": 4, "MCI": 0, "AD": 0}, {"HC": 2, "MCI": 1, "AD": 0}])
def test_uncertain_supervisor_state_agent_opposition_does_not_silence_agent(scores) -> None:
    frozen = {"HC": 0.30, "MCI": 0.40, "AD": 0.30}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities={"HC": 0.95, "MCI": 0.03, "AD": 0.02},
        post_state_probabilities={"HC": 0.60, "MCI": 0.25, "AD": 0.15},
        blind_ordinal_scores=scores,
    )
    assert result.state_authority_gate == 0.0
    assert result.agent_authority_gate > 0.0
    assert result.agent_authority_gate <= (scores["HC"] - max(scores["MCI"], scores["AD"])) / 4
    assert result.fused_probabilities["HC"] > frozen["HC"]


def test_binary_opposing_state_revision_cannot_override_agent_agreement() -> None:
    frozen = {"HC": 0.55, "AD": 0.45}
    result = _fuse(
        class_order=("HC", "AD"),
        frozen_probabilities=frozen,
        pre_state_probabilities={"HC": 0.80, "AD": 0.20},
        post_state_probabilities={"HC": 0.20, "AD": 0.80},
        blind_ordinal_scores={"HC": 4, "AD": 0},
    )
    assert result.state_authority_gate == 0.0
    assert result.frozen_parity


def test_stage_uncertainty_does_not_unlock_weak_screening_counterevidence() -> None:
    frozen = {"HC": 0.05, "MCI": 0.475, "AD": 0.475}
    result = _fuse(
        frozen_probabilities=frozen,
        pre_state_probabilities=frozen,
        post_state_probabilities=frozen,
        blind_ordinal_scores={"HC": 3, "MCI": 2, "AD": 1},
    )
    assert result.agent_authority_gate == 0.0
    assert result.frozen_parity


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
        ({"min_frozen_uncertainty": 1.1}, "min_frozen_uncertainty"),
        ({"min_state_uncertainty": 1.1}, "min_state_uncertainty"),
        ({"min_counterevidence_margin": -0.1}, "min_counterevidence_margin"),
        ({"staging_strength": -0.1}, "staging_strength"),
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
