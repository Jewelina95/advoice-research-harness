"""Fold-safe fitting and fixed joint calibration for the evidence-state pilot.

The production model identities intentionally remain distinct:

* the full base training entry point is ``advoice.condition_c.train_condition_c``;
* the replayable state model is
  ``advoice.module_a.TaskConditionedStatisticalExpert``;
* the historical pure fusion primitive is
  ``advoice.authority_joint_fusion.fuse_authority_joint``.

This module does not load historical prediction tables.  It supplies a small,
serializable linear adapter for fold orchestration and pure calibration code.
The runner may replace the adapter with the full model implementation while
preserving the fit/provenance boundary defined here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Literal, Self

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .contracts import EvidenceSnapshot, FusionRow, PredictionRow, SubjectRow


LEARNING_SCHEMA_VERSION = "advoice.pilot.learning.v1"
FULL_BASE_PREDICTOR_SYMBOL = "advoice.condition_c.train_condition_c"
STATE_REPLAY_PREDICTOR_SYMBOL = "advoice.module_a.TaskConditionedStatisticalExpert"
EXISTING_FUSION_SYMBOL = "advoice.authority_joint_fusion.fuse_authority_joint"
PROBABILITY_FLOOR = 1e-6
ARMS = ("B_raw", "B", "J-A", "J-S", "J-AS")
CALIBRATED_ARMS = ("B", "J-A", "J-S", "J-AS")
HEADS_BY_TASK = {
    "hc_mci_ad": ("impairment", "stage"),
    "hc_ad": ("binary",),
    "hc_impairment": ("binary",),
}
PRIOR_COEFFICIENTS = (1.0, 0.0, 0.0, 0.0)

Arm = Literal["B_raw", "B", "J-A", "J-S", "J-AS"]
HeadName = Literal["binary", "impairment", "stage"]


class LearningError(ValueError):
    """Invalid learning input, provenance, class semantics, or artifact."""


class FoldLeakageError(LearningError):
    """A fold includes a validation identity or identity group in fitting."""


def model_identity_manifest() -> dict[str, dict[str, Any]]:
    """Describe the production symbols discovered in the current source tree."""

    return {
        "base": {
            "symbol": FULL_BASE_PREDICTOR_SYMBOL,
            "feature_signature": (
                "subject_features", "subject_transcripts", "fold_calibrated_states",
                "metric_evidence", "task_and_reliability_adapters",
            ),
            "training_recipe": (
                "fold-local branch experts; nested OOF selection; multinomial logistic "
                "stacking or validated dynamic reliability gate; frozen probability calibration"
            ),
        },
        "replay": {
            "symbol": STATE_REPLAY_PREDICTOR_SYMBOL,
            "feature_signature": (
                "explicit_state_feature_whitelist", "task_adapter", "language_adapter",
            ),
            "training_recipe": (
                "fit-only median imputation and standardization; class-balanced fixed-C "
                "logistic regression; identical frozen model before and after replay"
            ),
        },
        "existing_fusion": {
            "symbol": EXISTING_FUSION_SYMBOL,
            "feature_signature": (
                "frozen_probabilities", "pre_post_state_probabilities", "blind_ordinal_scores",
            ),
            "training_recipe": "pure scoring primitive with caller-owned coefficients",
        },
    }


def _canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise LearningError(f"{name} must be a finite number.")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise LearningError(f"{name} must be a finite number.") from exc
    if not math.isfinite(result):
        raise LearningError(f"{name} must be a finite number.")
    return result


def _ordered_ids(values: Sequence[str], name: str, *, allow_empty: bool = False) -> tuple[str, ...]:
    result = tuple(str(value) for value in values)
    if (not allow_empty and not result) or any(not value for value in result):
        raise LearningError(f"{name} must contain non-empty identifiers.")
    if len(set(result)) != len(result):
        raise LearningError(f"{name} must not contain duplicates.")
    return result


def _validate_class_order(class_order: Sequence[str]) -> tuple[str, ...]:
    order = tuple(str(value) for value in class_order)
    if order not in (("HC", "AD"), ("HC", "IMPAIRED"), ("HC", "MCI", "AD")):
        raise LearningError("Unsupported binary or three-class order.")
    return order


def _probability_vector(values: Sequence[float], class_order: Sequence[str]) -> tuple[float, ...]:
    order = tuple(class_order)
    result = tuple(_finite_float(value, "probability") for value in values)
    if len(result) != len(order) or any(value < 0.0 or value > 1.0 for value in result):
        raise LearningError("Probabilities must match class_order and lie in [0, 1].")
    total = math.fsum(result)
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise LearningError("Probabilities must sum to one.")
    return result


def _clip_probability(value: float) -> float:
    return min(1.0 - PROBABILITY_FLOOR, max(PROBABILITY_FLOOR, float(value)))


def _logit(value: float) -> float:
    probability = _clip_probability(value)
    return math.log(probability) - math.log1p(-probability)


def _sigmoid(value: float) -> float:
    if value >= 0.0:
        inverse = math.exp(-value)
        return 1.0 / (1.0 + inverse)
    exponential = math.exp(value)
    return exponential / (1.0 + exponential)


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    weights = np.exp(shifted)
    return weights / weights.sum(axis=1, keepdims=True)


@dataclass(frozen=True, slots=True)
class FoldCaseInput:
    """Truth-free features and same-fold evidence for one subject."""

    subject: SubjectRow
    base_features: Mapping[str, float | int | None]
    state_features: Mapping[str, float | int | None]
    evidence_snapshot: EvidenceSnapshot

    def __post_init__(self) -> None:
        if self.evidence_snapshot.subject != self.subject:
            raise LearningError("Evidence snapshot and fold subject differ.")


@dataclass(frozen=True, slots=True)
class FoldInputs:
    """Cases plus a scorer-side label mapping; only declared fit IDs are read."""

    cases: Mapping[str, FoldCaseInput]
    labels: Mapping[str, str]

    def __post_init__(self) -> None:
        if not self.cases:
            raise LearningError("Fold inputs require cases.")
        for subject_id, case in self.cases.items():
            if subject_id != case.subject.subject_id:
                raise LearningError("Fold case key and subject_id differ.")


@dataclass(frozen=True, slots=True)
class FrozenFoldConfig:
    """Fixed fitting recipe; no data-driven hyperparameter search is allowed."""

    class_order: tuple[str, ...]
    base_feature_names: tuple[str, ...]
    state_feature_names: tuple[str, ...]
    base_model_id: str
    replay_model_id: str
    seed: int = 20260924
    c: float = 1.0
    max_iter: int = 2000
    excluded_ids: tuple[str, ...] = ()
    base_source_symbol: str = FULL_BASE_PREDICTOR_SYMBOL
    replay_source_symbol: str = STATE_REPLAY_PREDICTOR_SYMBOL
    final_refit: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "class_order", _validate_class_order(self.class_order))
        for name in ("base_feature_names", "state_feature_names"):
            values = _ordered_ids(getattr(self, name), name)
            object.__setattr__(self, name, values)
        if set(self.base_feature_names) <= set(self.state_feature_names):
            raise LearningError(
                "The full base signature must include non-state inputs; a state-only baseline is forbidden."
            )
        object.__setattr__(self, "excluded_ids", _ordered_ids(
            self.excluded_ids, "excluded_ids", allow_empty=True,
        ))
        if not self.base_model_id or not self.replay_model_id:
            raise LearningError("Base and replay model IDs are required and distinct.")
        if self.base_model_id == self.replay_model_id:
            raise LearningError("Base and replay model identities must remain distinct.")
        if self.base_source_symbol == self.replay_source_symbol:
            raise LearningError("Base and replay source symbols must remain distinct.")
        if self.c <= 0 or self.max_iter < 1:
            raise LearningError("The fixed linear fitting recipe is invalid.")


@dataclass(frozen=True, slots=True)
class LinearModelArtifact:
    """Portable fixed-feature multinomial or binary logistic model."""

    model_id: str
    source_symbol: str
    class_order: tuple[str, ...]
    feature_names: tuple[str, ...]
    impute_values: tuple[float, ...]
    means: tuple[float, ...]
    scales: tuple[float, ...]
    learned_classes: tuple[str, ...]
    coefficients: tuple[tuple[float, ...], ...]
    intercepts: tuple[float, ...]
    fit_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    seed: int
    artifact_id: str = ""

    def __post_init__(self) -> None:
        order = _validate_class_order(self.class_order)
        object.__setattr__(self, "class_order", order)
        width = len(self.feature_names)
        if not self.model_id or not self.source_symbol or width == 0:
            raise LearningError("Model identity, source symbol, and features are required.")
        if not (len(self.impute_values) == len(self.means) == len(self.scales) == width):
            raise LearningError("Preprocessing vectors must match feature_names.")
        if set(self.learned_classes) != set(order):
            raise LearningError("Learned classes differ from class_order.")
        if len(self.coefficients) != len(self.intercepts):
            raise LearningError("Coefficient rows and intercepts differ.")
        if len(order) == 2 and len(self.coefficients) != 1:
            raise LearningError("Binary logistic artifacts require one coefficient row.")
        if len(order) == 3 and len(self.coefficients) != 3:
            raise LearningError("Three-class artifacts require one row per class.")
        if any(len(row) != width for row in self.coefficients):
            raise LearningError("Coefficient width differs from feature_names.")
        if any(scale <= 0.0 for scale in self.scales):
            raise LearningError("Feature scales must be positive.")
        payload = self.to_dict(include_id=False)
        expected = "model_" + _digest(payload)[:24]
        if self.artifact_id and self.artifact_id != expected:
            raise LearningError("Stale model artifact ID.")
        object.__setattr__(self, "artifact_id", expected)

    def _matrix(self, rows: Sequence[Mapping[str, float | int | None]]) -> np.ndarray:
        matrix = np.empty((len(rows), len(self.feature_names)), dtype=float)
        for row_index, row in enumerate(rows):
            for column_index, name in enumerate(self.feature_names):
                value = row.get(name)
                try:
                    matrix[row_index, column_index] = np.nan if value is None else float(value)
                except (TypeError, ValueError):
                    matrix[row_index, column_index] = np.nan
        imputed = np.where(np.isfinite(matrix), matrix, np.asarray(self.impute_values))
        return (imputed - np.asarray(self.means)) / np.asarray(self.scales)

    def predict_proba(
        self, rows: Sequence[Mapping[str, float | int | None]],
    ) -> tuple[tuple[float, ...], ...]:
        design = self._matrix(rows)
        coefficients = np.asarray(self.coefficients, dtype=float)
        intercepts = np.asarray(self.intercepts, dtype=float)
        if len(self.class_order) == 2:
            positive = _sigmoid_array(design @ coefficients[0] + intercepts[0])
            learned = np.column_stack((1.0 - positive, positive))
        else:
            learned = _softmax(design @ coefficients.T + intercepts)
        index = {label: idx for idx, label in enumerate(self.learned_classes)}
        ordered = learned[:, [index[label] for label in self.class_order]]
        return tuple(tuple(float(value) for value in row) for row in ordered)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": LEARNING_SCHEMA_VERSION,
            "model_id": self.model_id,
            "source_symbol": self.source_symbol,
            "class_order": list(self.class_order),
            "feature_names": list(self.feature_names),
            "impute_values": list(self.impute_values),
            "means": list(self.means),
            "scales": list(self.scales),
            "learned_classes": list(self.learned_classes),
            "coefficients": [list(row) for row in self.coefficients],
            "intercepts": list(self.intercepts),
            "fit_ids": list(self.fit_ids),
            "excluded_ids": list(self.excluded_ids),
            "seed": self.seed,
        }
        if include_id:
            result["artifact_id"] = self.artifact_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        if value.get("schema_version") != LEARNING_SCHEMA_VERSION:
            raise LearningError("Unknown learning artifact schema.")
        return cls(
            model_id=str(value["model_id"]), source_symbol=str(value["source_symbol"]),
            class_order=tuple(value["class_order"]), feature_names=tuple(value["feature_names"]),
            impute_values=tuple(value["impute_values"]), means=tuple(value["means"]),
            scales=tuple(value["scales"]), learned_classes=tuple(value["learned_classes"]),
            coefficients=tuple(tuple(row) for row in value["coefficients"]),
            intercepts=tuple(value["intercepts"]), fit_ids=tuple(value["fit_ids"]),
            excluded_ids=tuple(value["excluded_ids"]), seed=int(value["seed"]),
            artifact_id=str(value["artifact_id"]),
        )


def _sigmoid_array(values: np.ndarray) -> np.ndarray:
    result = np.empty_like(values, dtype=float)
    positive = values >= 0.0
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exponential = np.exp(values[~positive])
    result[~positive] = exponential / (1.0 + exponential)
    return result


def _fit_linear_model(
    *,
    cases: Mapping[str, FoldCaseInput],
    labels: Mapping[str, str],
    fit_ids: tuple[str, ...],
    excluded_ids: tuple[str, ...],
    feature_names: tuple[str, ...],
    feature_source: Literal["base", "state"],
    model_id: str,
    source_symbol: str,
    class_order: tuple[str, ...],
    seed: int,
    c: float,
    max_iter: int,
) -> LinearModelArtifact:
    rows = [
        cases[subject_id].base_features if feature_source == "base"
        else cases[subject_id].state_features
        for subject_id in fit_ids
    ]
    matrix = np.empty((len(rows), len(feature_names)), dtype=float)
    for row_index, row in enumerate(rows):
        for column_index, name in enumerate(feature_names):
            value = row.get(name)
            try:
                matrix[row_index, column_index] = np.nan if value is None else float(value)
            except (TypeError, ValueError):
                matrix[row_index, column_index] = np.nan
    impute = np.asarray([
        float(np.median(column[np.isfinite(column)])) if np.isfinite(column).any() else 0.0
        for column in matrix.T
    ])
    imputed = np.where(np.isfinite(matrix), matrix, impute)
    means = imputed.mean(axis=0)
    scales = imputed.std(axis=0)
    scales = np.where(scales > 1e-12, scales, 1.0)
    design = (imputed - means) / scales
    target = np.asarray([labels[subject_id] for subject_id in fit_ids], dtype=object)
    if set(target) != set(class_order):
        raise LearningError("Every configured class must occur in the fold fit IDs.")
    estimator = LogisticRegression(
        C=c, max_iter=max_iter, class_weight="balanced", solver="lbfgs", random_state=seed,
    ).fit(design, target)
    return LinearModelArtifact(
        model_id=model_id, source_symbol=source_symbol, class_order=class_order,
        feature_names=feature_names, impute_values=tuple(float(value) for value in impute),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        learned_classes=tuple(str(value) for value in estimator.classes_),
        coefficients=tuple(tuple(float(value) for value in row) for row in estimator.coef_),
        intercepts=tuple(float(value) for value in estimator.intercept_), fit_ids=fit_ids,
        excluded_ids=excluded_ids, seed=seed,
    )


@dataclass(frozen=True, slots=True)
class FoldArtifact:
    class_order: tuple[str, ...]
    fold_id: str
    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    base_model: LinearModelArtifact
    replay_model: LinearModelArtifact
    base_fit_id: str
    reference_fit_id: str
    reference_fit_hash: str
    final_refit: bool
    artifact_id: str = ""

    def __post_init__(self) -> None:
        if self.base_model.source_symbol == self.replay_model.source_symbol:
            raise LearningError("Base and replay artifacts cannot share a source identity.")
        if set(self.fit_ids) & set(self.validation_ids):
            raise FoldLeakageError("Fit and validation IDs overlap.")
        if set(self.fit_group_ids) & set(self.validation_group_ids):
            raise FoldLeakageError("Fit and validation identity groups overlap.")
        payload = self.to_dict(include_id=False)
        expected = "fold_" + _digest(payload)[:24]
        if self.artifact_id and self.artifact_id != expected:
            raise LearningError("Stale fold artifact ID.")
        object.__setattr__(self, "artifact_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": LEARNING_SCHEMA_VERSION,
            "class_order": list(self.class_order), "fold_id": self.fold_id,
            "fit_ids": list(self.fit_ids), "validation_ids": list(self.validation_ids),
            "excluded_ids": list(self.excluded_ids), "fit_group_ids": list(self.fit_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "base_model": self.base_model.to_dict(), "replay_model": self.replay_model.to_dict(),
            "base_fit_id": self.base_fit_id, "reference_fit_id": self.reference_fit_id,
            "reference_fit_hash": self.reference_fit_hash, "final_refit": self.final_refit,
        }
        if include_id:
            result["artifact_id"] = self.artifact_id
        return result


@dataclass(frozen=True, slots=True)
class OOFProvenance:
    fold_artifact_id: str
    fit_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    validation_id: str
    validation_group_id: str
    excluded_ids: tuple[str, ...]
    final_fit: bool = False

    def __post_init__(self) -> None:
        if self.final_fit:
            raise FoldLeakageError("Final-refit predictions cannot be OOF provenance.")
        if self.validation_id in self.fit_ids:
            raise FoldLeakageError("OOF subject occurs in fit IDs.")
        if self.validation_group_id in self.fit_group_ids:
            raise FoldLeakageError("OOF identity group occurs in fit groups.")
        if self.validation_id not in self.excluded_ids:
            raise FoldLeakageError("OOF subject must be explicit in excluded IDs.")


@dataclass(frozen=True, slots=True)
class ReplayContext:
    replay_model: LinearModelArtifact
    reference_fit_id: str
    reference_fit_hash: str
    before_probabilities: tuple[float, ...]

    def score_after(self, state_features: Mapping[str, float | int | None]) -> tuple[float, ...]:
        return self.replay_model.predict_proba((state_features,))[0]


@dataclass(frozen=True, slots=True)
class FoldPrediction:
    subject: SubjectRow
    base_probabilities: tuple[float, ...]
    state_probabilities: tuple[float, ...]
    evidence_snapshot: EvidenceSnapshot
    replay_context: ReplayContext
    provenance: OOFProvenance | None
    base_fit_id: str
    base_fit_hash: str


def fit_fold(
    inputs: FoldInputs,
    fit_ids: Sequence[str],
    validation_ids: Sequence[str],
    frozen_config: FrozenFoldConfig,
) -> FoldArtifact:
    """Fit both branches on explicit IDs and reject subject/group leakage."""

    fit = _ordered_ids(fit_ids, "fit_ids")
    validation = _ordered_ids(
        validation_ids, "validation_ids", allow_empty=frozen_config.final_refit,
    )
    excluded = tuple(dict.fromkeys((*frozen_config.excluded_ids, *validation)))
    known = set(inputs.cases)
    missing = sorted((set(fit) | set(validation)) - known)
    if missing:
        raise LearningError(f"Unknown fold subject IDs: {missing}")
    if set(fit) & set(validation) or set(fit) & set(excluded):
        raise FoldLeakageError("Fit IDs overlap validation or excluded IDs.")
    fit_groups = tuple(dict.fromkeys(inputs.cases[item].subject.source_group_id for item in fit))
    validation_groups = tuple(
        dict.fromkeys(inputs.cases[item].subject.source_group_id for item in validation)
    )
    if set(fit_groups) & set(validation_groups):
        raise FoldLeakageError("A validation identity group occurs in fit IDs.")
    missing_labels = [item for item in fit if item not in inputs.labels]
    if missing_labels:
        raise LearningError(f"Missing fit labels: {missing_labels}")
    fit_labels = {inputs.labels[item] for item in fit}
    if not fit_labels <= set(frozen_config.class_order):
        raise LearningError("Fit labels are outside class_order.")
    if validation:
        fold_ids = {inputs.cases[item].subject.fold_id for item in validation}
        if len(fold_ids) != 1:
            raise LearningError("A fold artifact requires one validation fold ID.")
        fold_id = next(iter(fold_ids))
    else:
        fold_id = "full_development"
    base = _fit_linear_model(
        cases=inputs.cases, labels=inputs.labels, fit_ids=fit, excluded_ids=excluded,
        feature_names=frozen_config.base_feature_names, feature_source="base",
        model_id=frozen_config.base_model_id, source_symbol=frozen_config.base_source_symbol,
        class_order=frozen_config.class_order, seed=frozen_config.seed, c=frozen_config.c,
        max_iter=frozen_config.max_iter,
    )
    replay = _fit_linear_model(
        cases=inputs.cases, labels=inputs.labels, fit_ids=fit, excluded_ids=excluded,
        feature_names=frozen_config.state_feature_names, feature_source="state",
        model_id=frozen_config.replay_model_id, source_symbol=frozen_config.replay_source_symbol,
        class_order=frozen_config.class_order, seed=frozen_config.seed, c=frozen_config.c,
        max_iter=frozen_config.max_iter,
    )
    reference_hash = _digest({
        "fit_ids": fit, "fit_group_ids": fit_groups,
        "replay_model_artifact_id": replay.artifact_id,
        "state_features": frozen_config.state_feature_names,
    })
    return FoldArtifact(
        class_order=frozen_config.class_order, fold_id=fold_id, fit_ids=fit,
        validation_ids=validation, excluded_ids=excluded, fit_group_ids=fit_groups,
        validation_group_ids=validation_groups, base_model=base, replay_model=replay,
        base_fit_id=f"base_{fold_id}", reference_fit_id=f"reference_{fold_id}",
        reference_fit_hash=reference_hash, final_refit=frozen_config.final_refit,
    )


def predict_fold(fold_artifact: FoldArtifact, case_inputs: FoldCaseInput) -> FoldPrediction:
    """Predict with the paired base/replay fit and bind same-fold evidence."""

    subject = case_inputs.subject
    if not fold_artifact.final_refit:
        if subject.subject_id not in fold_artifact.validation_ids:
            raise LearningError("OOF prediction subject is not assigned to this validation fold.")
        if subject.subject_id in fold_artifact.fit_ids:
            raise FoldLeakageError("OOF prediction subject occurs in fit IDs.")
        if subject.source_group_id in fold_artifact.fit_group_ids:
            raise FoldLeakageError("OOF prediction identity group occurs in fit groups.")
    snapshot = case_inputs.evidence_snapshot
    if (
        snapshot.reference_fit_id != fold_artifact.reference_fit_id
        or snapshot.reference_fit_hash != fold_artifact.reference_fit_hash
    ):
        raise LearningError("Evidence snapshot is not from the paired fold reference fit.")
    base_probability = fold_artifact.base_model.predict_proba((case_inputs.base_features,))[0]
    state_probability = fold_artifact.replay_model.predict_proba((case_inputs.state_features,))[0]
    provenance = None
    if not fold_artifact.final_refit:
        provenance = OOFProvenance(
            fold_artifact_id=fold_artifact.artifact_id, fit_ids=fold_artifact.fit_ids,
            fit_group_ids=fold_artifact.fit_group_ids, validation_id=subject.subject_id,
            validation_group_id=subject.source_group_id, excluded_ids=fold_artifact.excluded_ids,
        )
    return FoldPrediction(
        subject=subject, base_probabilities=base_probability,
        state_probabilities=state_probability, evidence_snapshot=snapshot,
        replay_context=ReplayContext(
            replay_model=fold_artifact.replay_model,
            reference_fit_id=fold_artifact.reference_fit_id,
            reference_fit_hash=fold_artifact.reference_fit_hash,
            before_probabilities=state_probability,
        ),
        provenance=provenance, base_fit_id=fold_artifact.base_fit_id,
        base_fit_hash=fold_artifact.base_model.artifact_id,
    )


def refit_full_development(
    inputs: FoldInputs,
    development_ids: Sequence[str],
    oof_predictions: Sequence[FoldPrediction],
    frozen_config: FrozenFoldConfig,
) -> FoldArtifact:
    """Refit only after every development subject has exactly one OOF row."""

    expected = _ordered_ids(development_ids, "development_ids")
    observed = [
        item.subject.subject_id for item in oof_predictions if item.provenance is not None
    ]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise LearningError("Full development refit requires exactly one OOF prediction per subject.")
    return fit_fold(
        inputs, expected, (), replace(frozen_config, excluded_ids=(), final_refit=True),
    )


@dataclass(frozen=True, slots=True)
class CalibrationRow:
    """Truth-free projection used by calibration and pure prediction."""

    subject: SubjectRow
    fusion_hash: str
    base_probabilities: tuple[float, ...]
    agent_v0_scores: Mapping[str, int] | None
    agent_v1_scores: Mapping[str, int] | None
    v1_required: bool
    state_probabilities_before: tuple[float, ...] | None
    state_probabilities_after: tuple[float, ...] | None
    arm_refusal_reasons: Mapping[str, str | None]
    oof_provenance: OOFProvenance | None = None

    def __post_init__(self) -> None:
        _probability_vector(self.base_probabilities, self.subject.class_order)
        for value in (self.state_probabilities_before, self.state_probabilities_after):
            if value is not None:
                _probability_vector(value, self.subject.class_order)
        if (self.state_probabilities_before is None) != (self.state_probabilities_after is None):
            raise LearningError("State replay probabilities must be supplied as a before/after pair.")
        expected = {"J-A", "J-S", "J-AS"}
        if set(self.arm_refusal_reasons) != expected:
            raise LearningError("Joint arm refusal reasons are incomplete.")
        for scores in (self.agent_v0_scores, self.agent_v1_scores):
            if scores is not None:
                if set(scores) != set(self.subject.class_order):
                    raise LearningError("Agent scores must match class_order.")
                if any(isinstance(value, bool) or not isinstance(value, int) or not 0 <= value <= 4
                       for value in scores.values()):
                    raise LearningError("Agent scores must be integer ordinals from zero through four.")
        if not self.v1_required and self.agent_v1_scores is not None:
            raise LearningError("A no-op replay must use v0 and cannot carry v1 scores.")

    @classmethod
    def from_fusion(
        cls,
        fusion: FusionRow,
        *,
        state_probabilities_before: Sequence[float] | None = None,
        state_probabilities_after: Sequence[float] | None = None,
        oof_provenance: OOFProvenance | None = None,
    ) -> Self:
        v0 = None
        if fusion.assessment_v0 is not None and fusion.assessment_v0.status == "ok":
            v0 = dict(fusion.assessment_v0.ordinal_scores or {})
        v1 = None
        if fusion.assessment_v1 is not None and fusion.assessment_v1.status == "ok":
            v1 = dict(fusion.assessment_v1.ordinal_scores or {})
        reasons = {
            "J-A": fusion.j_a.fallback_reason,
            "J-S": fusion.j_s.fallback_reason,
            "J-AS": fusion.j_as.fallback_reason,
        }
        before = None if state_probabilities_before is None else tuple(state_probabilities_before)
        after = None if state_probabilities_after is None else tuple(state_probabilities_after)
        return cls(
            subject=fusion.subject, fusion_hash=fusion.content_hash,
            base_probabilities=fusion.base_probabilities, agent_v0_scores=v0,
            agent_v1_scores=v1, v1_required=fusion.v1_required,
            state_probabilities_before=before,
            state_probabilities_after=after, arm_refusal_reasons=reasons,
            oof_provenance=oof_provenance,
        )


@dataclass(frozen=True, slots=True)
class CalibrationConfig:
    seed: int = 20260924
    optimizer_maxiter: int = 2000

    def __post_init__(self) -> None:
        if self.optimizer_maxiter < 1:
            raise LearningError("optimizer_maxiter must be positive.")


@dataclass(frozen=True, slots=True)
class HeadCalibrator:
    arm: str
    head: str
    class_order: tuple[str, ...]
    estimable: bool
    refusal_reason: str | None
    optimizer_success: bool
    objective: float | None
    coefficients: tuple[float, float, float, float] | None
    agent_mean: float
    agent_scale: float
    state_mean: float
    state_scale: float
    fit_ids: tuple[str, ...]
    fit_count: int
    seed: int
    calibrator_id: str = ""

    def __post_init__(self) -> None:
        _validate_class_order(self.class_order)
        if self.estimable != (self.refusal_reason is None):
            raise LearningError("Estimable heads cannot carry refusal reasons and vice versa.")
        if self.estimable:
            if not self.optimizer_success or self.objective is None or self.coefficients is None:
                raise LearningError("Estimable heads require successful optimizer output.")
            if any(value < 0.0 for value in self.coefficients[:3]):
                raise LearningError("Non-intercept calibrator coefficients must be nonnegative.")
        else:
            if self.coefficients is not None:
                raise LearningError("Blocked heads cannot contain invented coefficients.")
        if self.agent_scale <= 0.0 or self.state_scale <= 0.0:
            raise LearningError("Calibrator feature scales must be positive.")
        if self.fit_count != len(self.fit_ids):
            raise LearningError("Calibrator fit count differs from fit IDs.")
        payload = self.to_dict(include_id=False)
        expected = "cal_" + _digest(payload)[:24]
        if self.calibrator_id and self.calibrator_id != expected:
            raise LearningError("Stale calibrator ID.")
        object.__setattr__(self, "calibrator_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "arm": self.arm, "head": self.head, "class_order": list(self.class_order),
            "estimable": self.estimable, "refusal_reason": self.refusal_reason,
            "optimizer_success": self.optimizer_success, "objective": self.objective,
            "coefficients": None if self.coefficients is None else list(self.coefficients),
            "agent_mean": self.agent_mean, "agent_scale": self.agent_scale,
            "state_mean": self.state_mean, "state_scale": self.state_scale,
            "fit_ids": list(self.fit_ids), "fit_count": self.fit_count, "seed": self.seed,
        }
        if include_id:
            result["calibrator_id"] = self.calibrator_id
        return result

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> Self:
        coefficients = value["coefficients"]
        return cls(
            arm=str(value["arm"]), head=str(value["head"]),
            class_order=tuple(value["class_order"]), estimable=bool(value["estimable"]),
            refusal_reason=value["refusal_reason"], optimizer_success=bool(value["optimizer_success"]),
            objective=value["objective"],
            coefficients=None if coefficients is None else tuple(coefficients),
            agent_mean=float(value["agent_mean"]), agent_scale=float(value["agent_scale"]),
            state_mean=float(value["state_mean"]), state_scale=float(value["state_scale"]),
            fit_ids=tuple(value["fit_ids"]), fit_count=int(value["fit_count"]),
            seed=int(value["seed"]), calibrator_id=str(value["calibrator_id"]),
        )


@dataclass(frozen=True, slots=True)
class JointCalibrators:
    task: str
    class_order: tuple[str, ...]
    arms: Mapping[str, Mapping[str, HeadCalibrator]]
    matched_baselines: Mapping[str, Mapping[str, HeadCalibrator]]
    seed: int
    artifact_id: str = ""

    def __post_init__(self) -> None:
        order = _validate_class_order(self.class_order)
        expected_heads = set(HEADS_BY_TASK[self.task])
        if set(self.arms) != set(CALIBRATED_ARMS):
            raise LearningError("Calibrator artifact must contain B and all three joint arms.")
        if set(self.matched_baselines) != {"J-A", "J-S", "J-AS"}:
            raise LearningError("Matched baseline calibrators are incomplete.")
        for heads in (*self.arms.values(), *self.matched_baselines.values()):
            if set(heads) != expected_heads:
                raise LearningError("Calibrator head set differs from task semantics.")
        object.__setattr__(self, "class_order", order)
        payload = self.to_dict(include_id=False)
        expected = "joint_" + _digest(payload)[:24]
        if self.artifact_id and self.artifact_id != expected:
            raise LearningError("Stale joint calibrator artifact ID.")
        object.__setattr__(self, "artifact_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": LEARNING_SCHEMA_VERSION, "task": self.task,
            "class_order": list(self.class_order), "seed": self.seed,
            "arms": {
                arm: {head: model.to_dict() for head, model in sorted(heads.items())}
                for arm, heads in sorted(self.arms.items())
            },
            "matched_baselines": {
                arm: {head: model.to_dict() for head, model in sorted(heads.items())}
                for arm, heads in sorted(self.matched_baselines.items())
            },
        }
        if include_id:
            result["artifact_id"] = self.artifact_id
        return result

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_json(cls, value: str | bytes) -> Self:
        try:
            data = json.loads(value)
        except (TypeError, ValueError, UnicodeError) as exc:
            raise LearningError("Malformed calibrator JSON.") from exc
        if data.get("schema_version") != LEARNING_SCHEMA_VERSION:
            raise LearningError("Unknown learning artifact schema.")
        return cls(
            task=str(data["task"]), class_order=tuple(data["class_order"]), seed=int(data["seed"]),
            arms={
                arm: {head: HeadCalibrator.from_dict(model) for head, model in heads.items()}
                for arm, heads in data["arms"].items()
            },
            matched_baselines={
                arm: {head: HeadCalibrator.from_dict(model) for head, model in heads.items()}
                for arm, heads in data["matched_baselines"].items()
            },
            artifact_id=str(data["artifact_id"]),
        )


def agent_contrast(scores: Mapping[str, int], head: str) -> float:
    """Return the frozen ordinal contrast for one diagnostic head."""

    if head == "impairment":
        required = {"HC", "MCI", "AD"}
        if set(scores) != required:
            raise LearningError("Impairment contrast requires HC/MCI/AD scores.")
        return (float(scores["MCI"]) + float(scores["AD"])) / 2.0 - float(scores["HC"])
    if head == "stage":
        if not {"MCI", "AD"} <= set(scores):
            raise LearningError("Stage contrast requires MCI and AD scores.")
        return float(scores["AD"] - scores["MCI"])
    if head == "binary":
        if "HC" not in scores or len(scores) != 2:
            raise LearningError("Binary contrast requires HC and one positive class.")
        positive = next(label for label in scores if label != "HC")
        return float(scores[positive] - scores["HC"])
    raise LearningError(f"Unknown calibration head: {head}")


def head_probability(probabilities: Sequence[float], class_order: Sequence[str], head: str) -> float:
    order = tuple(class_order)
    values = _probability_vector(probabilities, order)
    by_class = dict(zip(order, values, strict=True))
    if head == "impairment":
        return by_class["MCI"] + by_class["AD"]
    if head == "stage":
        impaired = by_class["MCI"] + by_class["AD"]
        return 0.5 if impaired <= 0.0 else by_class["AD"] / impaired
    if head == "binary":
        positive = next(label for label in order if label != "HC")
        return by_class[positive]
    raise LearningError(f"Unknown calibration head: {head}")


def replay_log_odds_delta(row: CalibrationRow, head: str) -> float | None:
    if row.state_probabilities_before is None or row.state_probabilities_after is None:
        return None
    before = head_probability(row.state_probabilities_before, row.subject.class_order, head)
    after = head_probability(row.state_probabilities_after, row.subject.class_order, head)
    return _logit(after) - _logit(before)


def _agent_scores_for_arm(row: CalibrationRow, arm: str) -> Mapping[str, int] | None:
    if arm == "J-A":
        return row.agent_v0_scores
    if arm == "J-AS":
        return row.agent_v1_scores if row.v1_required else row.agent_v0_scores
    return None


def _features_for_arm(row: CalibrationRow, arm: str, head: str) -> tuple[float, float, float] | None:
    base = _logit(head_probability(row.base_probabilities, row.subject.class_order, head))
    agent = 0.0
    state = 0.0
    if arm in ("J-A", "J-AS"):
        scores = _agent_scores_for_arm(row, arm)
        if scores is None:
            return None
        agent = agent_contrast(scores, head)
    if arm in ("J-S", "J-AS"):
        delta = replay_log_odds_delta(row, head)
        if delta is None:
            return None
        state = delta
    return base, agent, state


def _target(label: str, head: str) -> float | None:
    if head == "impairment":
        return float(label != "HC")
    if head == "stage":
        if label == "HC":
            return None
        return float(label == "AD")
    if head == "binary":
        return float(label != "HC")
    raise LearningError(f"Unknown calibration head: {head}")


def _blocked_head(
    *, arm: str, head: str, class_order: tuple[str, ...], reason: str,
    fit_ids: tuple[str, ...], seed: int, optimizer_success: bool = False,
    objective: float | None = None,
) -> HeadCalibrator:
    return HeadCalibrator(
        arm=arm, head=head, class_order=class_order, estimable=False,
        refusal_reason=reason, optimizer_success=optimizer_success, objective=objective,
        coefficients=None, agent_mean=0.0, agent_scale=1.0, state_mean=0.0,
        state_scale=1.0, fit_ids=fit_ids, fit_count=len(fit_ids), seed=seed,
    )


def _fit_binary_head(
    rows: Sequence[CalibrationRow], labels: Mapping[str, str], *, arm: str,
    head: str, class_order: tuple[str, ...], seed: int, optimizer_maxiter: int,
) -> HeadCalibrator:
    fit_ids: list[str] = []
    raw_features: list[tuple[float, float, float]] = []
    targets: list[float] = []
    for row in rows:
        subject_id = row.subject.subject_id
        if subject_id not in labels:
            raise LearningError(f"Missing development label for {subject_id}.")
        target = _target(labels[subject_id], head)
        if target is None:
            continue
        if arm != "B" and row.arm_refusal_reasons[arm] is not None:
            continue
        features = _features_for_arm(row, arm, head)
        if features is None:
            continue
        fit_ids.append(subject_id)
        raw_features.append(features)
        targets.append(target)
    ids = tuple(fit_ids)
    if not ids:
        return _blocked_head(
            arm=arm, head=head, class_order=class_order, reason="no_valid_fit_rows",
            fit_ids=ids, seed=seed,
        )
    y = np.asarray(targets, dtype=float)
    if len(np.unique(y)) != 2:
        return _blocked_head(
            arm=arm, head=head, class_order=class_order, reason="absent_training_class",
            fit_ids=ids, seed=seed,
        )
    raw = np.asarray(raw_features, dtype=float)
    agent_mean = float(raw[:, 1].mean()) if arm in ("J-A", "J-AS") else 0.0
    state_mean = float(raw[:, 2].mean()) if arm in ("J-S", "J-AS") else 0.0
    agent_scale = float(raw[:, 1].std()) if arm in ("J-A", "J-AS") else 1.0
    state_scale = float(raw[:, 2].std()) if arm in ("J-S", "J-AS") else 1.0
    if agent_scale <= 1e-12:
        agent_scale = 1.0
    if state_scale <= 1e-12:
        state_scale = 1.0
    design = np.column_stack((
        raw[:, 0], (raw[:, 1] - agent_mean) / agent_scale,
        (raw[:, 2] - state_mean) / state_scale, np.ones(len(raw)),
    ))
    active_agent = arm in ("J-A", "J-AS")
    active_state = arm in ("J-S", "J-AS")

    def objective(theta: np.ndarray) -> tuple[float, np.ndarray]:
        logits = design @ theta
        loss = float(np.mean(np.logaddexp(0.0, logits) - y * logits))
        delta = theta - np.asarray(PRIOR_COEFFICIENTS)
        value = loss + float(delta @ delta)
        probability = _sigmoid_array(logits)
        gradient = design.T @ (probability - y) / len(y) + 2.0 * delta
        return value, gradient

    bounds = [(0.0, None), (0.0, None) if active_agent else (0.0, 0.0),
              (0.0, None) if active_state else (0.0, 0.0), (None, None)]
    try:
        result = minimize(
            objective, np.asarray(PRIOR_COEFFICIENTS), method="L-BFGS-B", jac=True,
            bounds=bounds, options={"maxiter": optimizer_maxiter, "ftol": 1e-12},
        )
    except Exception:
        return _blocked_head(
            arm=arm, head=head, class_order=class_order, reason="optimizer_failure",
            fit_ids=ids, seed=seed,
        )
    objective_value = float(result.fun) if math.isfinite(float(result.fun)) else None
    if not result.success or objective_value is None or not np.isfinite(result.x).all():
        return _blocked_head(
            arm=arm, head=head, class_order=class_order, reason="optimizer_failure",
            fit_ids=ids, seed=seed, optimizer_success=False, objective=objective_value,
        )
    coefficients = tuple(float(value) for value in result.x)
    return HeadCalibrator(
        arm=arm, head=head, class_order=class_order, estimable=True, refusal_reason=None,
        optimizer_success=True, objective=objective_value, coefficients=coefficients,
        agent_mean=agent_mean, agent_scale=agent_scale, state_mean=state_mean,
        state_scale=state_scale, fit_ids=ids, fit_count=len(ids), seed=seed,
    )


def _validate_oof_rows(rows: Sequence[CalibrationRow], labels: Mapping[str, str]) -> tuple[str, tuple[str, ...]]:
    if not rows:
        raise LearningError("Calibration requires OOF rows.")
    subject_ids = tuple(row.subject.subject_id for row in rows)
    if len(subject_ids) != len(set(subject_ids)):
        raise FoldLeakageError("Every development subject must have exactly one OOF row.")
    if set(subject_ids) != set(labels):
        missing = sorted(set(labels) - set(subject_ids))
        unexpected = sorted(set(subject_ids) - set(labels))
        raise LearningError(f"OOF rows and development labels differ: missing={missing}, unexpected={unexpected}.")
    tasks = {row.subject.task for row in rows}
    orders = {row.subject.class_order for row in rows}
    if len(tasks) != 1 or len(orders) != 1:
        raise LearningError("Calibration rows must share one task and class order.")
    for row in rows:
        if row.subject.partition != "development":
            raise LearningError("Only development OOF rows may fit calibrators.")
        if row.oof_provenance is None:
            raise FoldLeakageError("Calibration rows require OOF provenance.")
        if row.oof_provenance.validation_id != row.subject.subject_id:
            raise FoldLeakageError("OOF provenance belongs to a different subject.")
        if row.oof_provenance.validation_group_id != row.subject.source_group_id:
            raise FoldLeakageError("OOF provenance belongs to a different identity group.")
    return next(iter(tasks)), next(iter(orders))


def fit_joint_calibrators(
    oof_rows: Sequence[CalibrationRow],
    development_labels: Mapping[str, str],
    config: CalibrationConfig,
) -> JointCalibrators:
    """Fit B and each joint arm independently using fixed OOF-only heads."""

    task, class_order = _validate_oof_rows(oof_rows, development_labels)
    allowed = set(class_order)
    if any(label not in allowed for label in development_labels.values()):
        raise LearningError("Development labels are outside class_order.")
    heads = HEADS_BY_TASK[task]
    arms: dict[str, dict[str, HeadCalibrator]] = {}
    for arm in CALIBRATED_ARMS:
        arms[arm] = {
            head: _fit_binary_head(
                oof_rows, development_labels, arm=arm, head=head, class_order=class_order,
                seed=config.seed, optimizer_maxiter=config.optimizer_maxiter,
            )
            for head in heads
        }
    matched: dict[str, dict[str, HeadCalibrator]] = {}
    for joint_arm in ("J-A", "J-S", "J-AS"):
        matched[joint_arm] = {}
        for head in heads:
            valid_ids = set(arms[joint_arm][head].fit_ids)
            matched_rows = [row for row in oof_rows if row.subject.subject_id in valid_ids]
            model = _fit_binary_head(
                matched_rows, development_labels, arm="B", head=head, class_order=class_order,
                seed=config.seed, optimizer_maxiter=config.optimizer_maxiter,
            )
            matched[joint_arm][head] = replace(model, arm=f"B_matched_{joint_arm}", calibrator_id="")
    return JointCalibrators(
        task=task, class_order=class_order, arms=arms,
        matched_baselines=matched, seed=config.seed,
    )


def apply_head(
    calibrator: HeadCalibrator, *, base_log_odds: float,
    agent_value: float = 0.0, state_value: float = 0.0,
) -> float:
    """Apply one fitted binary head without labels, I/O, or provider calls."""

    if not calibrator.estimable or calibrator.coefficients is None:
        raise LearningError("Cannot score an unestimable calibration head.")
    values = np.asarray((
        _finite_float(base_log_odds, "base_log_odds"),
        (_finite_float(agent_value, "agent_value") - calibrator.agent_mean) / calibrator.agent_scale,
        (_finite_float(state_value, "state_value") - calibrator.state_mean) / calibrator.state_scale,
        1.0,
    ))
    return _sigmoid(float(values @ np.asarray(calibrator.coefficients)))


def combine_head_probabilities(
    class_order: Sequence[str], *, impairment: float | None = None,
    stage: float | None = None, binary: float | None = None,
) -> tuple[float, ...]:
    """Combine fixed heads and return probabilities in the requested class order."""

    order = tuple(class_order)
    _validate_class_order(order)
    if len(order) == 2:
        if binary is None:
            raise LearningError("Binary class order requires a binary head probability.")
        positive = _clip_probability(binary)
        values = {"HC": 1.0 - positive, next(label for label in order if label != "HC"): positive}
    else:
        if impairment is None or stage is None:
            raise LearningError("Three-class output requires impairment and stage heads.")
        impaired = _clip_probability(impairment)
        ad_given_impaired = _clip_probability(stage)
        values = {
            "HC": 1.0 - impaired,
            "MCI": impaired * (1.0 - ad_given_impaired),
            "AD": impaired * ad_given_impaired,
        }
    result = tuple(float(values[label]) for label in order)
    total = math.fsum(result)
    return tuple(value / total for value in result)


def _score_arm(row: CalibrationRow, calibrators: JointCalibrators, arm: str) -> tuple[float, ...] | None:
    probabilities: dict[str, float] = {}
    for head, model in calibrators.arms[arm].items():
        if not model.estimable:
            return None
        features = _features_for_arm(row, arm, head)
        if features is None:
            return None
        probabilities[head] = apply_head(
            model, base_log_odds=features[0], agent_value=features[1], state_value=features[2],
        )
    return combine_head_probabilities(
        calibrators.class_order, impairment=probabilities.get("impairment"),
        stage=probabilities.get("stage"), binary=probabilities.get("binary"),
    )


def predict_joint(
    fusion_row: FusionRow | CalibrationRow,
    calibrators: JointCalibrators,
    *,
    state_probabilities_before: Sequence[float] | None = None,
    state_probabilities_after: Sequence[float] | None = None,
) -> tuple[PredictionRow, ...]:
    """Score B_raw/B/J-A/J-S/J-AS with direct B fallback for invalid inputs."""

    row = fusion_row if isinstance(fusion_row, CalibrationRow) else CalibrationRow.from_fusion(
        fusion_row, state_probabilities_before=state_probabilities_before,
        state_probabilities_after=state_probabilities_after,
    )
    if row.subject.class_order != calibrators.class_order or row.subject.task != calibrators.task:
        raise LearningError("Class order or task differs from calibrator artifact.")
    predictions: list[PredictionRow] = []

    def prediction(
        arm: str, probabilities: tuple[float, ...] | None, *, status: str,
        calibrator_id: str | None, reason: str | None = None, fallback_arm: str | None = None,
    ) -> PredictionRow:
        predicted = None
        if probabilities is not None:
            predicted = row.subject.class_order[int(np.argmax(np.asarray(probabilities)))]
        traces = [row.fusion_hash]
        if calibrator_id is not None:
            traces.append(calibrator_id)
        else:
            traces.append("calibrator_unavailable")
        return PredictionRow(
            subject=row.subject, fusion_hash=row.fusion_hash, arm=arm,
            class_order=row.subject.class_order, probabilities=probabilities,
            predicted=predicted, status=status, calibrator_id=calibrator_id,
            trace_ids=tuple(traces), fallback_reason=reason, fallback_arm=fallback_arm,
        )

    raw = tuple(row.base_probabilities)
    predictions.append(prediction(
        "B_raw", raw, status="ok", calibrator_id="raw_base_probabilities",
    ))
    baseline = _score_arm(row, calibrators, "B")
    baseline_id = calibrators.artifact_id if baseline is not None else None
    if baseline is None:
        predictions.append(prediction(
            "B", None, status="unavailable", calibrator_id=None,
            reason="baseline_head_unestimable",
        ))
    else:
        predictions.append(prediction(
            "B", baseline, status="ok", calibrator_id=baseline_id,
        ))
    for arm in ("J-A", "J-S", "J-AS"):
        refusal = row.arm_refusal_reasons[arm]
        joint = None if refusal is not None else _score_arm(row, calibrators, arm)
        if joint is not None:
            predictions.append(prediction(
                arm, joint, status="ok", calibrator_id=calibrators.artifact_id,
            ))
        elif baseline is not None:
            predictions.append(prediction(
                arm, baseline, status="fallback", calibrator_id=baseline_id,
                reason=refusal or "calibrator_unestimable", fallback_arm="B",
            ))
        else:
            predictions.append(prediction(
                arm, None, status="unavailable", calibrator_id=None,
                reason=refusal or "baseline_head_unestimable",
            ))
    return tuple(predictions)


__all__ = [
    "ARMS", "CALIBRATED_ARMS", "EXISTING_FUSION_SYMBOL", "FULL_BASE_PREDICTOR_SYMBOL",
    "STATE_REPLAY_PREDICTOR_SYMBOL", "CalibrationConfig", "CalibrationRow",
    "FoldArtifact", "FoldCaseInput", "FoldInputs", "FoldLeakageError", "FoldPrediction",
    "FrozenFoldConfig", "HeadCalibrator", "JointCalibrators", "LearningError",
    "LinearModelArtifact", "OOFProvenance", "ReplayContext", "agent_contrast",
    "apply_head", "combine_head_probabilities", "fit_fold", "fit_joint_calibrators",
    "head_probability", "model_identity_manifest", "predict_fold", "predict_joint", "refit_full_development",
    "replay_log_odds_delta",
]
