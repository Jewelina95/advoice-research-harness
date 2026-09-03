from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


LABELS = ("HC", "MCI", "AD")


@dataclass(frozen=True)
class CognitivePrototypeModel:
    feature_names: tuple[str, ...]
    scale: np.ndarray
    class_medians: dict[str, np.ndarray]
    screening_weights: np.ndarray
    staging_weights: np.ndarray
    minimum_reliability: float


def _class_median(
    values: np.ndarray,
    reliability: np.ndarray,
    labels: np.ndarray,
    label: str,
    minimum_reliability: float,
) -> tuple[np.ndarray, np.ndarray]:
    selected = labels == label
    observed = selected[:, None] & (reliability >= minimum_reliability)
    masked = np.where(observed, values, np.nan)
    return np.nanmedian(masked, axis=0), observed.sum(axis=0)


def fit_cognitive_prototypes(
    values: np.ndarray,
    reliability: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    *,
    minimum_class_support: int = 8,
    minimum_reliability: float = 0.2,
) -> CognitivePrototypeModel:
    """Fit robust, class-balanced state references without using class prevalence."""

    x = np.asarray(values, dtype=float)
    rel = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    y = np.asarray(labels, dtype=str)
    if x.shape != rel.shape or x.ndim != 2:
        raise ValueError("values and reliability must be aligned two-dimensional arrays")
    if x.shape[1] != len(feature_names):
        raise ValueError("feature_names must match the state matrix")
    if set(LABELS) - set(y):
        raise ValueError("HC, MCI, and AD examples are required to fit prototypes")

    x = np.clip(x, -6.0, 6.0)
    observed = rel >= minimum_reliability
    masked = np.where(observed, x, np.nan)
    global_median = np.nanmedian(masked, axis=0)
    global_median = np.nan_to_num(global_median, nan=0.0)
    q75 = np.nanpercentile(masked, 75, axis=0)
    q25 = np.nanpercentile(masked, 25, axis=0)
    scale = np.nan_to_num((q75 - q25) / 1.349, nan=1.0, posinf=1.0, neginf=1.0)
    scale = np.maximum(scale, 0.75)

    medians: dict[str, np.ndarray] = {}
    support: dict[str, np.ndarray] = {}
    for label in LABELS:
        median, count = _class_median(x, rel, y, label, minimum_reliability)
        median = np.where(np.isfinite(median), median, global_median)
        medians[label] = median
        support[label] = count

    impaired = (medians["MCI"] + medians["AD"]) / 2.0
    screen_available = (
        (support["HC"] >= minimum_class_support)
        & (support["MCI"] >= minimum_class_support)
        & (support["AD"] >= minimum_class_support)
    )
    stage_available = (
        (support["MCI"] >= minimum_class_support)
        & (support["AD"] >= minimum_class_support)
    )
    coverage = observed.mean(axis=0)
    screening_weights = (
        np.abs(impaired - medians["HC"]) / scale * coverage * screen_available
    )
    staging_weights = (
        np.abs(medians["AD"] - medians["MCI"]) / scale
        * coverage
        * stage_available
    )
    screening_weights = np.clip(screening_weights, 0.0, 3.0)
    staging_weights = np.clip(staging_weights, 0.0, 3.0)

    return CognitivePrototypeModel(
        feature_names=tuple(feature_names),
        scale=scale,
        class_medians=medians,
        screening_weights=screening_weights,
        staging_weights=staging_weights,
        minimum_reliability=float(minimum_reliability),
    )


def _robust_distance(
    values: np.ndarray,
    reliability: np.ndarray,
    prototype: np.ndarray,
    scale: np.ndarray,
    feature_weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = (
        np.isfinite(values)
        & np.isfinite(reliability)
        & np.isfinite(prototype[None, :])
        & np.isfinite(scale[None, :])
        & (scale[None, :] > 0.0)
        & np.isfinite(feature_weights[None, :])
        & (reliability > 0.0)
        & (feature_weights[None, :] > 0.0)
    )
    standardized = np.zeros_like(values, dtype=float)
    np.divide(
        values - prototype[None, :],
        scale[None, :],
        out=standardized,
        where=valid,
    )
    absolute = np.abs(standardized)
    huber = np.where(absolute <= 2.0, standardized**2, 4.0 * absolute - 4.0)
    effective = np.where(valid, reliability * feature_weights[None, :], 0.0)
    denominator = effective.sum(axis=1)
    distance = np.divide(
        (huber * effective).sum(axis=1),
        denominator,
        out=np.zeros(len(values), dtype=float),
        where=denominator > 0,
    )
    usable = valid.sum(axis=1)
    return distance, usable


def _sigmoid(values: np.ndarray) -> np.ndarray:
    clipped = np.clip(values, -12.0, 12.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def predict_cognitive_prototypes(
    model: CognitivePrototypeModel,
    values: np.ndarray,
    reliability: np.ndarray,
) -> tuple[np.ndarray, list[dict[str, float | int]]]:
    """Return hierarchical probabilities and auditable signed evidence scores."""

    x = np.clip(np.asarray(values, dtype=float), -6.0, 6.0)
    rel = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    if x.shape != rel.shape or x.shape[1] != len(model.feature_names):
        raise ValueError("prototype input does not match the fitted state matrix")
    rel = np.where(rel >= model.minimum_reliability, rel, 0.0)
    impaired = (model.class_medians["MCI"] + model.class_medians["AD"]) / 2.0
    hc_distance, screen_usable = _robust_distance(
        x, rel, model.class_medians["HC"], model.scale, model.screening_weights
    )
    impaired_distance, _ = _robust_distance(
        x, rel, impaired, model.scale, model.screening_weights
    )
    mci_distance, stage_usable = _robust_distance(
        x, rel, model.class_medians["MCI"], model.scale, model.staging_weights
    )
    ad_distance, _ = _robust_distance(
        x, rel, model.class_medians["AD"], model.scale, model.staging_weights
    )
    screening_evidence = hc_distance - impaired_distance
    staging_evidence = mci_distance - ad_distance
    impaired_probability = _sigmoid(screening_evidence)
    ad_given_impaired = _sigmoid(staging_evidence)
    probability = np.column_stack(
        [
            1.0 - impaired_probability,
            impaired_probability * (1.0 - ad_given_impaired),
            impaired_probability * ad_given_impaired,
        ]
    )
    details = [
        {
            "screening_evidence": float(screening_evidence[index]),
            "staging_evidence": float(staging_evidence[index]),
            "screening_usable_states": int(screen_usable[index]),
            "staging_usable_states": int(stage_usable[index]),
        }
        for index in range(len(x))
    ]
    return probability, details


def _state_identity(feature_name: str) -> tuple[str, str]:
    name = feature_name.removeprefix("state_")
    if name.endswith("_active_task"):
        return name.removesuffix("_active_task"), "active_task"
    if "__task_" in name:
        return tuple(name.split("__task_", 1))  # type: ignore[return-value]
    return name.removesuffix("_overall"), "overall"


def build_case_prototype_reference(
    model: CognitivePrototypeModel,
    values: np.ndarray,
    reliability: np.ndarray,
    *,
    maximum_states: int = 6,
) -> dict[str, Any]:
    """Build a label-aware training reference that contains no model probabilities."""

    x = np.asarray(values, dtype=float)
    rel = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    if x.shape != rel.shape or x.ndim != 1 or len(x) != len(model.feature_names):
        raise ValueError("case reference must match the fitted state features")

    def rows(weights: np.ndarray) -> list[dict[str, Any]]:
        order = np.argsort(weights * rel)[::-1]
        result: list[dict[str, Any]] = []
        for index in order:
            if weights[index] <= 0.0 or rel[index] < model.minimum_reliability:
                continue
            state_id, task_scope = _state_identity(model.feature_names[index])
            result.append(
                {
                    "state_id": state_id,
                    "task_scope": task_scope,
                    "case_state_z": float(x[index]),
                    "case_reliability": float(rel[index]),
                    "HC_train_median": float(model.class_medians["HC"][index]),
                    "MCI_train_median": float(model.class_medians["MCI"][index]),
                    "AD_train_median": float(model.class_medians["AD"][index]),
                    "train_robust_scale": float(model.scale[index]),
                    "reference_strength": float(weights[index]),
                }
            )
            if len(result) >= maximum_states:
                break
        return result

    screening = rows(model.screening_weights)
    staging = rows(model.staging_weights)
    return {
        "reference_scope": "training_fold_balanced_state_prototypes",
        "reference_note": (
            "The impaired reference gives equal weight to MCI and AD. Task and language "
            "identify applicable references but are not disease evidence."
        ),
        "screening_state_references": screening,
        "staging_state_references": staging,
        "screening_reference_available": len(screening) >= 2,
        "staging_reference_available": len(staging) >= 2,
    }
