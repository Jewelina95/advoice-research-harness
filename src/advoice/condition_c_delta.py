"""Pure bounded state-delta fusion for frozen Condition C predictions.

This module applies a caller-configured correction; it neither selects
``alpha`` nor makes a performance claim.  Only pre/post probability changes
from a validated state replay may alter the frozen Condition C probabilities.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
import math
from numbers import Real
import re
from types import MappingProxyType
from typing import Any

from .module_a import ExplanationPacket
from .utils import hash_values


DELTA_FUSION_SCHEMA_VERSION = "advoice.condition_c.state_delta_fusion.v1"
PROBABILITY_FLOOR = 1e-12
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}", flags=re.IGNORECASE)
_NOOP_ACTIONS = frozenset({"retain", "no_op", "noop", "no_revision", "unchanged"})


class ConditionCDeltaError(ValueError):
    """The fusion inputs were invalid, stale, or incompletely hash-bound."""


def _finite_real(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ConditionCDeltaError(f"{name} must be a finite real number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ConditionCDeltaError(f"{name} must be a finite real number.")
    return parsed


def _sha256(value: Any, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ConditionCDeltaError(f"{name} must be a complete SHA-256 hash.")
    return value.lower()


@dataclass(frozen=True, slots=True)
class DeltaFusionConfig:
    """Predeclared, non-learned controls for one fusion operation."""

    alpha: float
    max_abs_delta: float

    def __post_init__(self) -> None:
        alpha = _finite_real(self.alpha, "alpha")
        maximum = _finite_real(self.max_abs_delta, "max_abs_delta")
        if not 0.0 <= alpha <= 1.0:
            raise ConditionCDeltaError("alpha must be in the closed interval [0, 1].")
        if maximum <= 0.0:
            raise ConditionCDeltaError("max_abs_delta must be greater than zero.")
        object.__setattr__(self, "alpha", alpha)
        object.__setattr__(self, "max_abs_delta", maximum)


@dataclass(frozen=True, slots=True)
class DeltaFusionHashExpectations:
    """Caller-held locks that must match all artifacts used by fusion."""

    frozen_packet_hash: str
    pre_packet_hash: str
    post_packet_hash: str
    frozen_evidence_hash: str
    pre_evidence_hash: str
    post_evidence_hash: str
    pre_state_hash: str
    post_state_hash: str
    revision_hash: str

    def __post_init__(self) -> None:
        for field in fields(self):
            object.__setattr__(self, field.name, _sha256(getattr(self, field.name), field.name))

    def to_dict(self) -> dict[str, str]:
        return {field.name: getattr(self, field.name) for field in fields(self)}


@dataclass(frozen=True, slots=True)
class DeltaFusionResult:
    """Immutable probabilities, correction trace, and validated provenance."""

    frozen_probabilities: Mapping[str, float]
    pre_state_probabilities: Mapping[str, float]
    post_state_probabilities: Mapping[str, float]
    state_delta: Mapping[str, float]
    clipped_state_delta: Mapping[str, float]
    fused_probabilities: Mapping[str, float]
    predicted_label: str
    alpha: float
    max_abs_delta: float
    correction_applied: bool
    revision_action: str
    provenance: DeltaFusionHashExpectations
    audit_hash: str
    schema_version: str = DELTA_FUSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in (
            "frozen_probabilities",
            "pre_state_probabilities",
            "post_state_probabilities",
            "state_delta",
            "clipped_state_delta",
            "fused_probabilities",
        ):
            object.__setattr__(self, name, MappingProxyType(dict(getattr(self, name))))


def _class_order(packet: ExplanationPacket, name: str) -> tuple[str, ...]:
    if not isinstance(packet, ExplanationPacket):
        raise ConditionCDeltaError(f"{name} must be an ExplanationPacket.")
    labels = tuple(packet.class_order)
    if (
        len(labels) < 2
        or any(not isinstance(label, str) or not label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ConditionCDeltaError(f"{name} has an invalid class order.")
    return labels


def _probabilities(
    packet: ExplanationPacket,
    labels: Sequence[str],
    name: str,
) -> tuple[float, ...]:
    selected = (
        packet.calibrated_probabilities
        if packet.calibrated_probabilities is not None
        else packet.raw_probabilities
    )
    if not isinstance(selected, Mapping) or tuple(selected) != tuple(labels):
        raise ConditionCDeltaError(
            f"{name} probability keys must exactly match its ordered class labels."
        )
    values = tuple(
        _finite_real(selected[label], f"{name} probability for {label!r}")
        for label in labels
    )
    if any(value < 0.0 or value > 1.0 for value in values):
        raise ConditionCDeltaError(f"{name} probabilities must be between zero and one.")
    if not math.isclose(math.fsum(values), 1.0, rel_tol=1e-9, abs_tol=1e-9):
        raise ConditionCDeltaError(f"{name} probabilities must sum to one.")
    return values


def _packet_hash(packet: ExplanationPacket, name: str) -> str:
    try:
        return hash_values([packet.to_json()])
    except (TypeError, ValueError) as exc:
        raise ConditionCDeltaError(f"{name} packet is not deterministically serializable.") from exc


def _validate_packet_hash(packet: ExplanationPacket, expected: str, name: str) -> None:
    if _packet_hash(packet, name) != expected:
        raise ConditionCDeltaError(f"Stale or mismatched {name} packet hash.")


def _bound_hash(
    packet: ExplanationPacket,
    aliases: Sequence[str],
    expected: str,
    name: str,
) -> None:
    if not isinstance(packet.hashes, Mapping):
        raise ConditionCDeltaError(f"Missing {name} hash binding.")
    found = [packet.hashes[key] for key in aliases if key in packet.hashes]
    if not found:
        raise ConditionCDeltaError(f"Missing {name} hash binding.")
    actual = {_sha256(value, f"{name} hash") for value in found}
    if len(actual) != 1 or next(iter(actual)) != expected:
        raise ConditionCDeltaError(f"Stale or mismatched {name} hash.")


def _validate_hashes(
    frozen: ExplanationPacket,
    pre: ExplanationPacket,
    post: ExplanationPacket,
    expected: DeltaFusionHashExpectations,
    revision_hash: str,
) -> None:
    _validate_packet_hash(frozen, expected.frozen_packet_hash, "frozen")
    _validate_packet_hash(pre, expected.pre_packet_hash, "pre")
    _validate_packet_hash(post, expected.post_packet_hash, "post")
    evidence_aliases = ("evidence_hash", "evidence_snapshot_hash", "evidence_snapshot_sha256")
    state_aliases = ("state_hash", "state_snapshot_hash")
    _bound_hash(frozen, evidence_aliases, expected.frozen_evidence_hash, "frozen evidence")
    _bound_hash(pre, evidence_aliases, expected.pre_evidence_hash, "pre evidence")
    _bound_hash(post, evidence_aliases, expected.post_evidence_hash, "post evidence")
    _bound_hash(pre, state_aliases, expected.pre_state_hash, "pre state")
    _bound_hash(post, state_aliases, expected.post_state_hash, "post state")
    if _sha256(revision_hash, "revision hash") != expected.revision_hash:
        raise ConditionCDeltaError("Stale or mismatched revision hash.")


def _ordered(labels: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    return dict(zip(labels, values, strict=True))


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    maximum = max(logits)
    weights = tuple(math.exp(value - maximum) for value in logits)
    denominator = math.fsum(weights)
    return tuple(value / denominator for value in weights)


def _normalized_action(value: Any) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ConditionCDeltaError("revision_action must be a non-empty string.")
    return value.strip().lower().replace("-", "_").replace(" ", "_")


def fuse_condition_c_state_delta(
    frozen: ExplanationPacket,
    pre_state: ExplanationPacket,
    post_state: ExplanationPacket,
    *,
    config: DeltaFusionConfig,
    expected_hashes: DeltaFusionHashExpectations,
    revision_hash: str,
    revision_action: str,
) -> DeltaFusionResult:
    """Apply only a bounded replay state change to frozen Condition C output.

    Predicted labels, truth labels, split metadata, feature evidence, and raw
    model contributions are never numeric inputs to this operation.
    """

    if not isinstance(config, DeltaFusionConfig):
        raise ConditionCDeltaError("config must be a predeclared DeltaFusionConfig.")
    if not isinstance(expected_hashes, DeltaFusionHashExpectations):
        raise ConditionCDeltaError("expected_hashes must be DeltaFusionHashExpectations.")

    frozen_labels = _class_order(frozen, "frozen")
    pre_labels = _class_order(pre_state, "pre state")
    post_labels = _class_order(post_state, "post state")
    if frozen_labels != pre_labels or frozen_labels != post_labels:
        raise ConditionCDeltaError("All packets must have the exact same class order.")

    frozen_values = _probabilities(frozen, frozen_labels, "frozen")
    pre_values = _probabilities(pre_state, frozen_labels, "pre state")
    post_values = _probabilities(post_state, frozen_labels, "post state")
    action = _normalized_action(revision_action)
    _validate_hashes(frozen, pre_state, post_state, expected_hashes, revision_hash)

    pre_log = tuple(math.log(max(value, PROBABILITY_FLOOR)) for value in pre_values)
    post_log = tuple(math.log(max(value, PROBABILITY_FLOOR)) for value in post_values)
    state_delta = tuple(after - before for before, after in zip(pre_log, post_log, strict=True))
    clipped_delta = tuple(
        max(-config.max_abs_delta, min(config.max_abs_delta, value)) for value in state_delta
    )
    no_correction = action in _NOOP_ACTIONS or pre_values == post_values or config.alpha == 0.0
    if no_correction:
        fused_values = frozen_values
        correction_applied = False
    else:
        base_log = tuple(math.log(max(value, PROBABILITY_FLOOR)) for value in frozen_values)
        fused_values = _softmax(tuple(
            base + config.alpha * change
            for base, change in zip(base_log, clipped_delta, strict=True)
        ))
        correction_applied = any(value != 0.0 for value in clipped_delta)

    predicted_label = frozen_labels[max(
        range(len(frozen_labels)), key=fused_values.__getitem__
    )]
    payload = {
        "schema_version": DELTA_FUSION_SCHEMA_VERSION,
        "class_order": list(frozen_labels),
        "frozen_probabilities": list(frozen_values),
        "pre_state_probabilities": list(pre_values),
        "post_state_probabilities": list(post_values),
        "state_delta": list(state_delta),
        "clipped_state_delta": list(clipped_delta),
        "fused_probabilities": list(fused_values),
        "predicted_label": predicted_label,
        "alpha": config.alpha,
        "max_abs_delta": config.max_abs_delta,
        "probability_floor": PROBABILITY_FLOOR,
        "correction_applied": correction_applied,
        "revision_action": action,
        "provenance": expected_hashes.to_dict(),
    }
    return DeltaFusionResult(
        frozen_probabilities=_ordered(frozen_labels, frozen_values),
        pre_state_probabilities=_ordered(frozen_labels, pre_values),
        post_state_probabilities=_ordered(frozen_labels, post_values),
        state_delta=_ordered(frozen_labels, state_delta),
        clipped_state_delta=_ordered(frozen_labels, clipped_delta),
        fused_probabilities=_ordered(frozen_labels, fused_values),
        predicted_label=predicted_label,
        alpha=config.alpha,
        max_abs_delta=config.max_abs_delta,
        correction_applied=correction_applied,
        revision_action=action,
        provenance=expected_hashes,
        audit_hash=hash_values([payload]),
    )
