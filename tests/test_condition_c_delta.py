from __future__ import annotations

from copy import deepcopy
from dataclasses import FrozenInstanceError, replace
import math
import struct

import pytest

from advoice.condition_c_delta import (
    ConditionCDeltaError,
    DeltaFusionConfig,
    DeltaFusionHashExpectations,
    fuse_condition_c_state_delta,
)
from advoice.module_a import ExplanationPacket
from advoice.utils import hash_values


REVISION_HASH = "9" * 64


def _packet(
    probabilities: tuple[float, ...],
    *,
    labels: tuple[str, ...] = ("HC", "AD"),
    evidence_hash: str = "e" * 64,
    state_hash: str | None = "5" * 64,
    calibrated: bool = True,
) -> ExplanationPacket:
    values = dict(zip(labels, probabilities, strict=True))
    hashes = {"evidence_hash": evidence_hash}
    if state_hash is not None:
        hashes["state_hash"] = state_hash
    return ExplanationPacket(
        schema_version="packet-v1",
        module_version="test-module",
        class_order=labels,
        predicted_label=labels[max(range(len(labels)), key=probabilities.__getitem__)],
        raw_probabilities=dict(values),
        calibrated_probabilities=dict(values) if calibrated else None,
        calibration_status="calibrated" if calibrated else "not_calibrated",
        logits={},
        intercepts={},
        feature_contributions={label: {} for label in labels},
        branch_contributions={label: {} for label in labels},
        consumed_evidence_ids=(),
        uncertainty={},
        fold_disagreement=None,
        ood={},
        applicability_status="applicable",
        hashes=hashes,
    )


def _hashes(
    frozen: ExplanationPacket,
    pre: ExplanationPacket,
    post: ExplanationPacket,
) -> DeltaFusionHashExpectations:
    return DeltaFusionHashExpectations(
        frozen_packet_hash=hash_values([frozen.to_json()]),
        pre_packet_hash=hash_values([pre.to_json()]),
        post_packet_hash=hash_values([post.to_json()]),
        frozen_evidence_hash=frozen.hashes["evidence_hash"],
        pre_evidence_hash=pre.hashes["evidence_hash"],
        post_evidence_hash=post.hashes["evidence_hash"],
        pre_state_hash=pre.hashes["state_hash"],
        post_state_hash=post.hashes["state_hash"],
        revision_hash=REVISION_HASH,
    )


def _fuse(
    frozen: ExplanationPacket,
    pre: ExplanationPacket,
    post: ExplanationPacket,
    *,
    alpha: float = 0.5,
    max_abs_delta: float = 2.0,
    revision_action: str = "downweight",
    expected_hashes: DeltaFusionHashExpectations | None = None,
    revision_hash: str = REVISION_HASH,
):
    return fuse_condition_c_state_delta(
        frozen,
        pre,
        post,
        config=DeltaFusionConfig(alpha=alpha, max_abs_delta=max_abs_delta),
        expected_hashes=expected_hashes or _hashes(frozen, pre, post),
        revision_hash=revision_hash,
        revision_action=revision_action,
    )


def _bits(values) -> tuple[bytes, ...]:
    return tuple(struct.pack("!d", value) for value in values)


@pytest.mark.parametrize("revision_action", ["retain", "no_op", "no_revision", "unchanged"])
def test_explicit_noop_preserves_frozen_probabilities_bit_for_bit(revision_action: str) -> None:
    frozen = _packet((0.8, 0.2), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.5, 0.5), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.01, 0.99), evidence_hash="d" * 64, state_hash="f" * 64)

    result = _fuse(frozen, pre, post, revision_action=revision_action)

    assert _bits(result.fused_probabilities.values()) == _bits(
        frozen.calibrated_probabilities.values()
    )
    assert result.predicted_label == frozen.predicted_label
    assert result.correction_applied is False


def test_equal_pre_and_post_state_preserves_frozen_probabilities_exactly() -> None:
    frozen = _packet((0.7, 0.3), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.4, 0.6), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.4, 0.6), evidence_hash="d" * 64, state_hash="f" * 64)

    result = _fuse(frozen, pre, post)

    assert tuple(result.fused_probabilities.values()) == (0.7, 0.3)
    assert result.correction_applied is False


def test_known_binary_log_probability_delta_arithmetic() -> None:
    frozen = _packet((0.8, 0.2), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.5, 0.5), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.25, 0.75), evidence_hash="d" * 64, state_hash="f" * 64)

    result = _fuse(frozen, pre, post, alpha=0.5, max_abs_delta=10.0)

    logits = (
        math.log(0.8) + 0.5 * (math.log(0.25) - math.log(0.5)),
        math.log(0.2) + 0.5 * (math.log(0.75) - math.log(0.5)),
    )
    denominator = sum(math.exp(value) for value in logits)
    assert tuple(result.state_delta.values()) == pytest.approx((math.log(0.5), math.log(1.5)))
    assert tuple(result.fused_probabilities.values()) == pytest.approx(
        tuple(math.exp(value) / denominator for value in logits)
    )
    assert result.correction_applied is True


def test_multiclass_delta_uses_ordered_probability_vectors() -> None:
    labels = ("HC", "MCI", "AD")
    frozen = _packet((0.6, 0.3, 0.1), labels=labels, evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.2, 0.5, 0.3), labels=labels, evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.4, 0.4, 0.2), labels=labels, evidence_hash="d" * 64, state_hash="f" * 64)

    result = _fuse(frozen, pre, post, alpha=0.25, max_abs_delta=4.0)

    delta = tuple(math.log(after) - math.log(before) for before, after in zip(
        (0.2, 0.5, 0.3), (0.4, 0.4, 0.2), strict=True
    ))
    logits = tuple(math.log(base) + 0.25 * change for base, change in zip(
        (0.6, 0.3, 0.1), delta, strict=True
    ))
    largest = max(logits)
    weights = tuple(math.exp(value - largest) for value in logits)
    expected = tuple(value / sum(weights) for value in weights)
    assert tuple(result.fused_probabilities) == labels
    assert tuple(result.fused_probabilities.values()) == pytest.approx(expected)


def test_alpha_zero_is_exact_frozen_parity() -> None:
    frozen = _packet((0.9, 0.1), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((1.0, 0.0), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.0, 1.0), evidence_hash="d" * 64, state_hash="f" * 64)

    result = _fuse(frozen, pre, post, alpha=0.0, max_abs_delta=0.25)

    assert _bits(result.fused_probabilities.values()) == _bits((0.9, 0.1))
    assert result.correction_applied is False


def test_alpha_one_clips_zero_to_one_extremes_before_fusion() -> None:
    frozen = _packet((0.5, 0.5), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((1.0, 0.0), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.0, 1.0), evidence_hash="d" * 64, state_hash="f" * 64)

    result = _fuse(frozen, pre, post, alpha=1.0, max_abs_delta=0.4)

    assert tuple(result.clipped_state_delta.values()) == pytest.approx((-0.4, 0.4))
    assert all(math.isfinite(value) for value in result.state_delta.values())
    assert result.fused_probabilities["AD"] == pytest.approx(
        math.exp(0.4) / (math.exp(-0.4) + math.exp(0.4))
    )


def test_label_order_mismatch_fails_closed() -> None:
    frozen = _packet((0.7, 0.3), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.6, 0.4), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet(
        (0.4, 0.6), labels=("AD", "HC"), evidence_hash="d" * 64, state_hash="f" * 64
    )

    with pytest.raises(ConditionCDeltaError, match="class order"):
        _fuse(frozen, pre, post)


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("frozen_packet_hash", "frozen packet"),
        ("pre_packet_hash", "pre packet"),
        ("post_packet_hash", "post packet"),
        ("frozen_evidence_hash", "frozen evidence"),
        ("pre_evidence_hash", "pre evidence"),
        ("post_evidence_hash", "post evidence"),
        ("pre_state_hash", "pre state"),
        ("post_state_hash", "post state"),
        ("revision_hash", "revision"),
    ],
)
def test_stale_hashes_fail_closed(field: str, message: str) -> None:
    frozen = _packet((0.7, 0.3), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.6, 0.4), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.4, 0.6), evidence_hash="d" * 64, state_hash="f" * 64)
    stale = replace(_hashes(frozen, pre, post), **{field: "0" * 64})

    with pytest.raises(ConditionCDeltaError, match=message):
        _fuse(frozen, pre, post, expected_hashes=stale)


def test_missing_packet_hash_binding_fails_closed() -> None:
    frozen = _packet((0.7, 0.3), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.6, 0.4), evidence_hash="b" * 64, state_hash=None)
    post = _packet((0.4, 0.6), evidence_hash="d" * 64, state_hash="f" * 64)
    expected = replace(
        _hashes(frozen, _packet((0.6, 0.4), evidence_hash="b" * 64), post),
        pre_packet_hash=hash_values([pre.to_json()]),
    )

    with pytest.raises(ConditionCDeltaError, match="pre state"):
        _fuse(frozen, pre, post, expected_hashes=expected)


@pytest.mark.parametrize(
    ("alpha", "max_abs_delta"),
    [(-0.01, 1.0), (1.01, 1.0), (math.nan, 1.0), (math.inf, 1.0), (0.5, 0.0), (0.5, math.inf)],
)
def test_invalid_config_is_rejected(alpha: float, max_abs_delta: float) -> None:
    with pytest.raises(ConditionCDeltaError):
        DeltaFusionConfig(alpha=alpha, max_abs_delta=max_abs_delta)


@pytest.mark.parametrize("bad_value", [math.nan, math.inf, -math.inf])
def test_nonfinite_probabilities_are_rejected(bad_value: float) -> None:
    frozen = _packet((0.7, 0.3), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.6, 0.4), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((bad_value, 0.6), evidence_hash="d" * 64, state_hash="f" * 64)

    with pytest.raises(ConditionCDeltaError, match="finite"):
        _fuse(
            frozen,
            pre,
            post,
            expected_hashes=DeltaFusionHashExpectations(
                frozen_packet_hash="1" * 64,
                pre_packet_hash="2" * 64,
                post_packet_hash="3" * 64,
                frozen_evidence_hash="a" * 64,
                pre_evidence_hash="b" * 64,
                post_evidence_hash="d" * 64,
                pre_state_hash="c" * 64,
                post_state_hash="f" * 64,
                revision_hash=REVISION_HASH,
            ),
        )


def test_audit_hash_is_deterministic_and_binds_configuration() -> None:
    frozen = _packet((0.8, 0.2), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.5, 0.5), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.25, 0.75), evidence_hash="d" * 64, state_hash="f" * 64)

    first = _fuse(frozen, pre, post)
    second = _fuse(frozen, pre, post)
    changed = _fuse(frozen, pre, post, alpha=0.6)

    assert first.audit_hash == second.audit_hash
    assert first.audit_hash != changed.audit_hash
    assert first.provenance.frozen_packet_hash == _hashes(frozen, pre, post).frozen_packet_hash
    assert first.provenance.revision_hash == REVISION_HASH


def test_inputs_are_not_mutated_and_result_is_immutable() -> None:
    frozen = _packet((0.8, 0.2), evidence_hash="a" * 64, state_hash=None)
    pre = _packet((0.5, 0.5), evidence_hash="b" * 64, state_hash="c" * 64)
    post = _packet((0.25, 0.75), evidence_hash="d" * 64, state_hash="f" * 64)
    before = tuple(deepcopy(packet.to_dict()) for packet in (frozen, pre, post))

    result = _fuse(frozen, pre, post)

    assert tuple(packet.to_dict() for packet in (frozen, pre, post)) == before
    with pytest.raises(TypeError):
        result.fused_probabilities["HC"] = 0.0
    with pytest.raises(FrozenInstanceError):
        result.alpha = 1.0
