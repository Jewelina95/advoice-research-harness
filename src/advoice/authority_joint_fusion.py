"""Pure, auditable joint fusion for frozen predictions and blind Agent evidence.

This module deliberately has no access to labels, splits, training artifacts,
or provider clients.  It combines three already-produced, class-aligned
objects in log space:

* frozen class probabilities;
* a bounded likelihood ratio from post-state versus pre-state probabilities;
* equal-prior likelihood evidence derived only from blind ordinal scores.

The caller owns all coefficient selection.  This function only applies an
explicit configuration and returns a deterministic audit record.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import json
import math
from numbers import Real
from types import MappingProxyType
from typing import Any

from .utils import hash_values


AUTHORITY_JOINT_FUSION_SCHEMA_VERSION = "advoice.authority.joint_fusion.v1"
PROBABILITY_FLOOR = 1e-12
_FORBIDDEN_PROVENANCE_TOKENS = frozenset({"label", "labels", "truth", "ground_truth", "target"})


class AuthorityJointFusionError(ValueError):
    """Raised when a label-blind joint fusion input is malformed."""


def _finite_real(value: Any, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AuthorityJointFusionError(f"{name} must be a finite real number.")
    parsed = float(value)
    if not math.isfinite(parsed):
        raise AuthorityJointFusionError(f"{name} must be a finite real number.")
    return parsed


def _class_order(value: Sequence[str]) -> tuple[str, ...]:
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise AuthorityJointFusionError("class_order must be an ordered sequence of class names.")
    labels = tuple(value)
    if len(labels) < 2 or any(not isinstance(label, str) or not label.strip() for label in labels):
        raise AuthorityJointFusionError("class_order must contain at least two non-empty class names.")
    if len(set(labels)) != len(labels):
        raise AuthorityJointFusionError("class_order must not contain duplicate class names.")
    return labels


def _probability_vector(
    value: Mapping[str, float],
    *,
    class_order: Sequence[str],
    name: str,
) -> tuple[float, ...]:
    if not isinstance(value, Mapping):
        raise AuthorityJointFusionError(f"{name} must be a class-probability mapping.")
    expected = set(class_order)
    actual = set(value)
    if actual != expected or len(value) != len(class_order):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise AuthorityJointFusionError(
            f"{name} must contain exactly class_order; missing={missing}, unexpected={unexpected}."
        )
    probabilities = tuple(
        _finite_real(value[label], name=f"{name}[{label!r}]") for label in class_order
    )
    if any(item < 0.0 or item > 1.0 for item in probabilities):
        raise AuthorityJointFusionError(f"{name} values must be in [0, 1].")
    if not math.isclose(math.fsum(probabilities), 1.0, rel_tol=1e-10, abs_tol=1e-10):
        raise AuthorityJointFusionError(f"{name} values must sum to one.")
    return probabilities


def _ordinal_vector(
    value: Mapping[str, int],
    *,
    class_order: Sequence[str],
) -> tuple[int, ...]:
    if not isinstance(value, Mapping):
        raise AuthorityJointFusionError("blind_ordinal_scores must be a class-score mapping.")
    expected = set(class_order)
    actual = set(value)
    if actual != expected or len(value) != len(class_order):
        missing = sorted(expected - actual)
        unexpected = sorted(actual - expected)
        raise AuthorityJointFusionError(
            "blind_ordinal_scores must contain exactly class_order; "
            f"missing={missing}, unexpected={unexpected}."
        )
    scores: list[int] = []
    for label in class_order:
        score = value[label]
        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 4:
            raise AuthorityJointFusionError(
                "blind_ordinal_scores values must be integer scores from 0 through 4."
            )
        scores.append(score)
    return tuple(scores)


def _softmax(logits: Sequence[float]) -> tuple[float, ...]:
    maximum = max(logits)
    weights = tuple(math.exp(value - maximum) for value in logits)
    denominator = math.fsum(weights)
    return tuple(value / denominator for value in weights)


def _center(values: Sequence[float]) -> tuple[float, ...]:
    average = math.fsum(values) / len(values)
    return tuple(value - average for value in values)


def _ordered(labels: Sequence[str], values: Sequence[float | int]) -> dict[str, float | int]:
    return dict(zip(labels, values, strict=True))


def _freeze_json(value: Any, *, path: str = "provenance") -> Any:
    """Validate and recursively freeze JSON provenance without hidden labels."""
    if isinstance(value, Mapping):
        frozen: dict[str, Any] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise AuthorityJointFusionError(f"{path} keys must be non-empty strings.")
            normalized = key.strip().lower().replace("-", "_").replace(" ", "_")
            if normalized in _FORBIDDEN_PROVENANCE_TOKENS:
                raise AuthorityJointFusionError(f"{path} must not contain truth or target labels.")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    if isinstance(value, bool) or value is None or isinstance(value, str) or isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise AuthorityJointFusionError(f"{path} must be JSON-serializable with finite floats.")
        return value
    raise AuthorityJointFusionError(f"{path} must contain JSON-compatible values only.")


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class AuthorityJointFusionConfig:
    """Explicit non-learned coefficients for one label-blind fusion operation."""

    state_strength: float
    agent_strength: float
    max_abs_state_delta: float
    ordinal_temperature: float

    def __post_init__(self) -> None:
        for field_name in ("state_strength", "agent_strength", "max_abs_state_delta"):
            parsed = _finite_real(getattr(self, field_name), name=field_name)
            if parsed < 0.0:
                raise AuthorityJointFusionError(f"{field_name} must be non-negative.")
            object.__setattr__(self, field_name, parsed)
        temperature = _finite_real(self.ordinal_temperature, name="ordinal_temperature")
        if temperature <= 0.0:
            raise AuthorityJointFusionError("ordinal_temperature must be greater than zero.")
        object.__setattr__(self, "ordinal_temperature", temperature)

    def to_dict(self) -> dict[str, float]:
        return {
            "state_strength": self.state_strength,
            "agent_strength": self.agent_strength,
            "max_abs_state_delta": self.max_abs_state_delta,
            "ordinal_temperature": self.ordinal_temperature,
        }


@dataclass(frozen=True, slots=True)
class AuthorityJointFusionResult:
    """Immutable probabilities, component evidence, and deterministic audit hashes."""

    class_order: tuple[str, ...]
    frozen_probabilities: Mapping[str, float]
    pre_state_probabilities: Mapping[str, float]
    post_state_probabilities: Mapping[str, float]
    blind_ordinal_scores: Mapping[str, int]
    state_log_evidence: Mapping[str, float]
    bounded_state_log_evidence: Mapping[str, float]
    agent_log_evidence: Mapping[str, float]
    agent_equal_prior_likelihood: Mapping[str, float]
    fused_probabilities: Mapping[str, float]
    predicted_label: str
    state_component_neutral: bool
    agent_component_neutral: bool
    frozen_parity: bool
    config: AuthorityJointFusionConfig
    provenance: Mapping[str, Any]
    input_hash: str
    audit_hash: str
    schema_version: str = AUTHORITY_JOINT_FUSION_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for field_name in (
            "frozen_probabilities",
            "pre_state_probabilities",
            "post_state_probabilities",
            "blind_ordinal_scores",
            "state_log_evidence",
            "bounded_state_log_evidence",
            "agent_log_evidence",
            "agent_equal_prior_likelihood",
            "fused_probabilities",
        ):
            object.__setattr__(self, field_name, MappingProxyType(dict(getattr(self, field_name))))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "class_order": list(self.class_order),
            "frozen_probabilities": dict(self.frozen_probabilities),
            "pre_state_probabilities": dict(self.pre_state_probabilities),
            "post_state_probabilities": dict(self.post_state_probabilities),
            "blind_ordinal_scores": dict(self.blind_ordinal_scores),
            "state_log_evidence": dict(self.state_log_evidence),
            "bounded_state_log_evidence": dict(self.bounded_state_log_evidence),
            "agent_log_evidence": dict(self.agent_log_evidence),
            "agent_equal_prior_likelihood": dict(self.agent_equal_prior_likelihood),
            "fused_probabilities": dict(self.fused_probabilities),
            "predicted_label": self.predicted_label,
            "state_component_neutral": self.state_component_neutral,
            "agent_component_neutral": self.agent_component_neutral,
            "frozen_parity": self.frozen_parity,
            "config": self.config.to_dict(),
            "provenance": _thaw_json(self.provenance),
            "input_hash": self.input_hash,
            "audit_hash": self.audit_hash,
        }


def fuse_authority_joint(
    frozen_probabilities: Mapping[str, float],
    pre_state_probabilities: Mapping[str, float],
    post_state_probabilities: Mapping[str, float],
    blind_ordinal_scores: Mapping[str, int],
    *,
    class_order: Sequence[str],
    config: AuthorityJointFusionConfig,
    provenance: Mapping[str, Any] | None = None,
) -> AuthorityJointFusionResult:
    """Fuse frozen, replay-state, and blind-Agent evidence in log space.

    ``blind_ordinal_scores`` must be produced before any frozen prediction is
    shown to the Agent.  They are transformed into a centered equal-prior log
    likelihood, so an all-equal score vector is exactly neutral.  This pure
    function does not read labels and has no model/provider side effects.
    """
    if not isinstance(config, AuthorityJointFusionConfig):
        raise AuthorityJointFusionError("config must be an AuthorityJointFusionConfig.")
    labels = _class_order(class_order)
    frozen = _probability_vector(frozen_probabilities, class_order=labels, name="frozen_probabilities")
    pre_state = _probability_vector(
        pre_state_probabilities, class_order=labels, name="pre_state_probabilities"
    )
    post_state = _probability_vector(
        post_state_probabilities, class_order=labels, name="post_state_probabilities"
    )
    ordinal = _ordinal_vector(blind_ordinal_scores, class_order=labels)
    frozen_provenance = _freeze_json({} if provenance is None else provenance)

    raw_state = tuple(
        math.log(max(after, PROBABILITY_FLOOR)) - math.log(max(before, PROBABILITY_FLOOR))
        for before, after in zip(pre_state, post_state, strict=True)
    )
    centered_state = _center(raw_state)
    bounded_state = tuple(
        max(-config.max_abs_state_delta, min(config.max_abs_state_delta, item))
        for item in centered_state
    )
    state_neutral = (
        config.state_strength == 0.0
        or pre_state == post_state
        or all(item == 0.0 for item in bounded_state)
    )

    ordinal_as_float = tuple(float(item) for item in ordinal)
    agent_log = tuple(item / config.ordinal_temperature for item in _center(ordinal_as_float))
    agent_likelihood = _softmax(agent_log)
    agent_neutral = config.agent_strength == 0.0 or len(set(ordinal)) == 1

    inputs = {
        "schema_version": AUTHORITY_JOINT_FUSION_SCHEMA_VERSION,
        "class_order": list(labels),
        "frozen_probabilities": list(frozen),
        "pre_state_probabilities": list(pre_state),
        "post_state_probabilities": list(post_state),
        "blind_ordinal_scores": list(ordinal),
        "config": config.to_dict(),
        "provenance": _thaw_json(frozen_provenance),
    }
    input_hash = hash_values([inputs])

    frozen_parity = state_neutral and agent_neutral
    if frozen_parity:
        fused = frozen
    else:
        logits = tuple(
            math.log(max(base, PROBABILITY_FLOOR))
            + (0.0 if state_neutral else config.state_strength * state)
            + (0.0 if agent_neutral else config.agent_strength * agent)
            for base, state, agent in zip(frozen, bounded_state, agent_log, strict=True)
        )
        fused = _softmax(logits)

    predicted_label = labels[max(range(len(labels)), key=fused.__getitem__)]
    audit_payload = {
        **inputs,
        "input_hash": input_hash,
        "state_log_evidence": list(centered_state),
        "bounded_state_log_evidence": list(bounded_state),
        "agent_log_evidence": list(agent_log),
        "agent_equal_prior_likelihood": list(agent_likelihood),
        "state_component_neutral": state_neutral,
        "agent_component_neutral": agent_neutral,
        "frozen_parity": frozen_parity,
        "fused_probabilities": list(fused),
        "predicted_label": predicted_label,
    }
    return AuthorityJointFusionResult(
        class_order=labels,
        frozen_probabilities=_ordered(labels, frozen),
        pre_state_probabilities=_ordered(labels, pre_state),
        post_state_probabilities=_ordered(labels, post_state),
        blind_ordinal_scores=_ordered(labels, ordinal),
        state_log_evidence=_ordered(labels, centered_state),
        bounded_state_log_evidence=_ordered(labels, bounded_state),
        agent_log_evidence=_ordered(labels, agent_log),
        agent_equal_prior_likelihood=_ordered(labels, agent_likelihood),
        fused_probabilities=_ordered(labels, fused),
        predicted_label=predicted_label,
        state_component_neutral=state_neutral,
        agent_component_neutral=agent_neutral,
        frozen_parity=frozen_parity,
        config=config,
        provenance=frozen_provenance,
        input_hash=input_hash,
        audit_hash=hash_values([audit_payload]),
    )
