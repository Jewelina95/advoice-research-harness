"""Fold-safe fitting and fixed joint calibration for the evidence-state pilot.

The production model identities intentionally remain distinct:

* the full base training entry point is ``advoice.condition_c.train_condition_c``;
* the replayable state model is
  ``advoice.module_a.TaskConditionedStatisticalExpert``;
* the historical pure fusion primitive is
  ``advoice.authority_joint_fusion.fuse_authority_joint``.

This module does not load historical prediction tables.  Its built-in linear
adapter is explicitly synthetic/test-only.  Analytical fitting fails closed
unless the runner supplies typed production-capable base and replay adapters
while preserving the fit/provenance boundary defined here.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
import hashlib
import json
import math
from typing import Any, Literal, Protocol, Self, runtime_checkable

import numpy as np
from scipy.optimize import minimize
from sklearn.linear_model import LogisticRegression

from .contracts import EvidenceSnapshot, FusionRow, PredictionRow, SubjectRow


LEARNING_SCHEMA_VERSION = "advoice.pilot.learning.v3"
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
_FOLD_MANIFEST_SEAL = object()
_DEVELOPMENT_MANIFEST_SEAL = object()
_ADAPTER_REGISTRY_SEAL = object()
_ADAPTER_ATTESTATION_SEAL = object()
_PREDICTION_RECEIPT_SEAL = object()

Arm = Literal["B_raw", "B", "J-A", "J-S", "J-AS"]
HeadName = Literal["binary", "impairment", "stage"]


class LearningError(ValueError):
    """Invalid learning input, provenance, class semantics, or artifact."""


class FoldLeakageError(LearningError):
    """A fold includes a validation identity or identity group in fitting."""


@runtime_checkable
class PredictorAdapter(Protocol):
    """Typed runner-supplied boundary for a fold-fitted predictor."""

    role: Literal["base", "replay"]
    implementation_id: str
    implementation_version: str

    def feature_pipeline_hash(
        self, feature_names: tuple[str, ...], class_order: tuple[str, ...],
    ) -> str: ...

    def fit(
        self,
        *,
        cases: Mapping[str, "FoldCaseInput"],
        labels: Mapping[str, str],
        fit_ids: tuple[str, ...],
        excluded_ids: tuple[str, ...],
        feature_names: tuple[str, ...],
        model_id: str,
        class_order: tuple[str, ...],
        seed: int,
        c: float,
        max_iter: int,
        adapter_attestation_id: str | None,
    ) -> "LinearModelArtifact": ...


@dataclass(frozen=True, slots=True, init=False)
class TrustedAdapterAttestation:
    """Non-forgeable-in-API attestation for an exact allowlisted adapter type."""

    role: Literal["base", "replay"]
    implementation_id: str
    implementation_version: str
    feature_pipeline_hash: str
    adapter_type: str
    adapter_object_id: int
    attestation_id: str

    def __init__(
        self,
        *,
        role: Literal["base", "replay"],
        implementation_id: str,
        implementation_version: str,
        feature_pipeline_hash: str,
        adapter_type: str,
        adapter_object_id: int,
        attestation_id: str,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _ADAPTER_ATTESTATION_SEAL:
            raise LearningError("Adapter attestations are issued only by the trusted registry.")
        object.__setattr__(self, "role", role)
        object.__setattr__(self, "implementation_id", implementation_id)
        object.__setattr__(self, "implementation_version", implementation_version)
        object.__setattr__(self, "feature_pipeline_hash", feature_pipeline_hash)
        object.__setattr__(self, "adapter_type", adapter_type)
        object.__setattr__(self, "adapter_object_id", adapter_object_id)
        object.__setattr__(self, "attestation_id", attestation_id)


@dataclass(frozen=True, slots=True, init=False)
class TrustedAdapterRegistry:
    """Closed runner-created registry; arbitrary adapter claims are never trusted."""

    attestations: tuple[TrustedAdapterAttestation, ...]

    def __init__(
        self,
        attestations: tuple[TrustedAdapterAttestation, ...],
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _ADAPTER_REGISTRY_SEAL:
            raise LearningError(
                "TrustedAdapterRegistry must be created by create_trusted_adapter_registry()."
            )
        object.__setattr__(self, "attestations", attestations)

    def resolve(
        self,
        adapter: PredictorAdapter,
        role: Literal["base", "replay"],
        feature_pipeline_hash: str,
    ) -> TrustedAdapterAttestation:
        matches = [
            item for item in self.attestations
            if item.adapter_object_id == id(adapter) and item.role == role
        ]
        if len(matches) != 1:
            raise LearningError(f"{role} adapter lacks a trusted registry attestation.")
        attestation = matches[0]
        adapter_type = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
        if (
            attestation.implementation_id != adapter.implementation_id
            or attestation.implementation_version != adapter.implementation_version
            or attestation.feature_pipeline_hash != feature_pipeline_hash
            or attestation.adapter_type != adapter_type
        ):
            raise LearningError(f"{role} adapter differs from its trusted registry attestation.")
        return attestation


# T2 owns no production fitting implementation. Exact production adapter types must
# be added here by the runner integration task; an empty allowlist fails closed.
_TRUSTED_ANALYTICAL_ADAPTER_TYPES: dict[str, tuple[type[Any], ...]] = {
    "base": (),
    "replay": (),
}


def create_trusted_adapter_registry(
    adapters: Sequence[PredictorAdapter],
    *,
    class_order: Sequence[str] | None = None,
    feature_names_by_role: Mapping[str, Sequence[str]] | None = None,
) -> TrustedAdapterRegistry:
    """Issue attestations only for exact production adapter types in the closed registry."""

    attestations: list[TrustedAdapterAttestation] = []
    for adapter in adapters:
        role = getattr(adapter, "role", None)
        if role not in ("base", "replay") or type(adapter) not in _TRUSTED_ANALYTICAL_ADAPTER_TYPES[role]:
            raise LearningError("Adapter type is not a trusted analytical adapter.")
        if class_order is None or feature_names_by_role is None or role not in feature_names_by_role:
            raise LearningError("Trusted adapter attestation requires the frozen model recipe.")
        order = _validate_class_order(class_order)
        names = _ordered_ids(feature_names_by_role[role], f"{role}_feature_names")
        pipeline_hash = adapter.feature_pipeline_hash(names, order)
        adapter_type = f"{type(adapter).__module__}.{type(adapter).__qualname__}"
        payload = {
            "role": role,
            "implementation_id": adapter.implementation_id,
            "implementation_version": adapter.implementation_version,
            "feature_pipeline_hash": pipeline_hash,
            "adapter_type": adapter_type,
        }
        attestations.append(TrustedAdapterAttestation(
            role=role, implementation_id=adapter.implementation_id,
            implementation_version=adapter.implementation_version,
            feature_pipeline_hash=pipeline_hash, adapter_type=adapter_type,
            adapter_object_id=id(adapter),
            attestation_id="adapter_attestation_" + _digest(payload)[:24],
            _seal=_ADAPTER_ATTESTATION_SEAL,
        ))
    roles = tuple(item.role for item in attestations)
    if set(roles) != {"base", "replay"} or len(roles) != 2:
        raise LearningError("Trusted adapter registry requires one base and one replay adapter.")
    return TrustedAdapterRegistry(tuple(attestations), _seal=_ADAPTER_REGISTRY_SEAL)


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
class DevelopmentSubjectProof:
    subject_id: str
    source_group_id: str
    fold_id: str
    subject_hash: str

    @classmethod
    def from_subject(cls, subject: SubjectRow) -> Self:
        if subject.partition != "development":
            raise FoldLeakageError("Development manifests cannot contain non-development subjects.")
        return cls(
            subject_id=subject.subject_id, source_group_id=subject.source_group_id,
            fold_id=subject.fold_id, subject_hash=subject.content_hash,
        )

    def to_dict(self) -> dict[str, str]:
        return {
            "subject_id": self.subject_id, "source_group_id": self.source_group_id,
            "fold_id": self.fold_id, "subject_hash": self.subject_hash,
        }


@dataclass(frozen=True, slots=True, init=False)
class DevelopmentManifest:
    """Canonical immutable development cohort derived from SubjectRow contracts."""

    subjects: tuple[DevelopmentSubjectProof, ...]
    manifest_id: str

    def __init__(
        self,
        subjects: tuple[DevelopmentSubjectProof, ...],
        manifest_id: str = "",
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _DEVELOPMENT_MANIFEST_SEAL:
            raise LearningError(
                "DevelopmentManifest must be created by seal_development_manifest()."
            )
        if not subjects:
            raise LearningError("Canonical development manifest cannot be empty.")
        ordered = tuple(sorted(subjects, key=lambda item: item.subject_id))
        subject_ids = tuple(item.subject_id for item in ordered)
        if len(set(subject_ids)) != len(subject_ids):
            raise LearningError("Canonical development manifest contains duplicate subjects.")
        expected = "development_manifest_" + _digest(
            [item.to_dict() for item in ordered]
        )[:24]
        if manifest_id and manifest_id != expected:
            raise LearningError("Stale development manifest ID.")
        object.__setattr__(self, "subjects", ordered)
        object.__setattr__(self, "manifest_id", expected)

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(item.subject_id for item in self.subjects)


def seal_development_manifest(inputs: FoldInputs) -> DevelopmentManifest:
    """Seal every development SubjectRow in the supplied canonical data contracts."""

    proofs = tuple(
        DevelopmentSubjectProof.from_subject(case.subject)
        for case in inputs.cases.values()
        if case.subject.partition == "development"
    )
    return DevelopmentManifest(proofs, _seal=_DEVELOPMENT_MANIFEST_SEAL)


def _validate_development_manifest(
    inputs: FoldInputs, manifest: DevelopmentManifest,
) -> tuple[str, ...]:
    if not isinstance(manifest, DevelopmentManifest):
        raise LearningError("A sealed canonical development manifest is required.")
    actual = tuple(sorted(
        (
            DevelopmentSubjectProof.from_subject(case.subject)
            for case in inputs.cases.values()
            if case.subject.partition == "development"
        ),
        key=lambda item: item.subject_id,
    ))
    if actual != manifest.subjects:
        raise FoldLeakageError("Inputs differ from the sealed canonical development manifest.")
    return manifest.subject_ids


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
    analytical_run: bool = True
    base_adapter: PredictorAdapter | None = None
    replay_adapter: PredictorAdapter | None = None
    development_manifest: DevelopmentManifest | None = None
    trusted_adapter_registry: TrustedAdapterRegistry | None = None

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
        if not isinstance(self.analytical_run, bool):
            raise LearningError("analytical_run must be boolean.")
        if self.development_manifest is not None and not isinstance(
            self.development_manifest, DevelopmentManifest,
        ):
            raise LearningError("development_manifest must be sealed.")
        if self.trusted_adapter_registry is not None and not isinstance(
            self.trusted_adapter_registry, TrustedAdapterRegistry,
        ):
            raise LearningError("trusted_adapter_registry must be runner-created.")
        if self.c <= 0 or self.max_iter < 1:
            raise LearningError("The fixed linear fitting recipe is invalid.")


@dataclass(frozen=True, slots=True)
class LinearModelArtifact:
    """Portable fixed-feature multinomial or binary logistic model."""

    model_id: str
    adapter_implementation: str
    adapter_version: str
    feature_pipeline_hash: str
    adapter_attestation_id: str | None
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
        if not self.model_id or not self.adapter_implementation or not self.adapter_version or width == 0:
            raise LearningError("Model and adapter identities plus features are required.")
        if len(self.feature_pipeline_hash) != 64:
            raise LearningError("Feature pipeline hash must be SHA-256.")
        if self.adapter_attestation_id is not None and not (
            isinstance(self.adapter_attestation_id, str)
            and self.adapter_attestation_id.startswith("adapter_attestation_")
        ):
            raise LearningError("Adapter attestation ID is invalid.")
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
            "adapter_implementation": self.adapter_implementation,
            "adapter_version": self.adapter_version,
            "feature_pipeline_hash": self.feature_pipeline_hash,
            "adapter_attestation_id": self.adapter_attestation_id,
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
            model_id=str(value["model_id"]),
            adapter_implementation=str(value["adapter_implementation"]),
            adapter_version=str(value["adapter_version"]),
            feature_pipeline_hash=str(value["feature_pipeline_hash"]),
            adapter_attestation_id=value["adapter_attestation_id"],
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
    adapter_implementation: str,
    adapter_version: str,
    feature_pipeline_hash: str,
    adapter_attestation_id: str | None,
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
        model_id=model_id, adapter_implementation=adapter_implementation,
        adapter_version=adapter_version, feature_pipeline_hash=feature_pipeline_hash,
        adapter_attestation_id=adapter_attestation_id, class_order=class_order,
        feature_names=feature_names, impute_values=tuple(float(value) for value in impute),
        means=tuple(float(value) for value in means),
        scales=tuple(float(value) for value in scales),
        learned_classes=tuple(str(value) for value in estimator.classes_),
        coefficients=tuple(tuple(float(value) for value in row) for row in estimator.coef_),
        intercepts=tuple(float(value) for value in estimator.intercept_), fit_ids=fit_ids,
        excluded_ids=excluded_ids, seed=seed,
    )


@dataclass(frozen=True, slots=True)
class SyntheticLinearPredictorAdapter:
    """Deterministic synthetic-test adapter; forbidden for analytical runs."""

    role: Literal["base", "replay"]
    implementation_version: str = "synthetic_linear_v1"

    @property
    def implementation_id(self) -> str:
        return f"advoice.pilot.learning.synthetic_{self.role}_linear"

    def feature_pipeline_hash(
        self, feature_names: tuple[str, ...], class_order: tuple[str, ...],
    ) -> str:
        return _digest({
            "implementation_id": self.implementation_id,
            "implementation_version": self.implementation_version,
            "role": self.role,
            "feature_names": feature_names,
            "class_order": class_order,
            "recipe": "fit_only_median_impute_standardize_fixed_c_balanced_logistic",
        })

    def fit(
        self,
        *,
        cases: Mapping[str, FoldCaseInput],
        labels: Mapping[str, str],
        fit_ids: tuple[str, ...],
        excluded_ids: tuple[str, ...],
        feature_names: tuple[str, ...],
        model_id: str,
        class_order: tuple[str, ...],
        seed: int,
        c: float,
        max_iter: int,
        adapter_attestation_id: str | None,
    ) -> LinearModelArtifact:
        if adapter_attestation_id is not None:
            raise LearningError("Synthetic adapters cannot receive analytical attestations.")
        return _fit_linear_model(
            cases=cases, labels=labels, fit_ids=fit_ids, excluded_ids=excluded_ids,
            feature_names=feature_names,
            feature_source="base" if self.role == "base" else "state",
            model_id=model_id, adapter_implementation=self.implementation_id,
            adapter_version=self.implementation_version,
            feature_pipeline_hash=self.feature_pipeline_hash(feature_names, class_order),
            adapter_attestation_id=None, class_order=class_order,
            seed=seed, c=c, max_iter=max_iter,
        )


@dataclass(frozen=True, slots=True)
class _ResolvedAdapter:
    adapter: PredictorAdapter
    pipeline_hash: str
    attestation_id: str | None


def _resolve_adapter(
    config: FrozenFoldConfig, role: Literal["base", "replay"],
) -> _ResolvedAdapter:
    adapter = config.base_adapter if role == "base" else config.replay_adapter
    if adapter is None:
        if config.analytical_run:
            raise LearningError(
                f"Analytical {role} fitting requires a runner-supplied production predictor adapter."
            )
        adapter = SyntheticLinearPredictorAdapter(role)
    if not isinstance(adapter, PredictorAdapter):
        raise LearningError(f"{role} adapter does not implement PredictorAdapter.")
    if adapter.role != role:
        raise LearningError(f"{role} adapter declares the wrong model role.")
    pipeline_hash = adapter.feature_pipeline_hash(
        config.base_feature_names if role == "base" else config.state_feature_names,
        config.class_order,
    )
    if len(pipeline_hash) != 64:
        raise LearningError(f"{role} adapter returned an invalid feature pipeline hash.")
    attestation_id = None
    if config.analytical_run:
        if config.trusted_adapter_registry is None:
            raise LearningError("Analytical fitting requires a runner-created trusted adapter registry.")
        attestation = config.trusted_adapter_registry.resolve(adapter, role, pipeline_hash)
        attestation_id = attestation.attestation_id
    return _ResolvedAdapter(adapter, pipeline_hash, attestation_id)


def _model_recipe_hash(
    config: FrozenFoldConfig,
    base: _ResolvedAdapter,
    replay: _ResolvedAdapter,
) -> str:
    return _digest({
        "class_order": config.class_order,
        "base_feature_names": config.base_feature_names,
        "state_feature_names": config.state_feature_names,
        "base_model_id": config.base_model_id,
        "replay_model_id": config.replay_model_id,
        "seed": config.seed,
        "c": config.c,
        "max_iter": config.max_iter,
        "excluded_ids": config.excluded_ids,
        "analytical_run": config.analytical_run,
        "development_manifest_id": (
            None if config.development_manifest is None
            else config.development_manifest.manifest_id
        ),
        "base_adapter": {
            "implementation_id": base.adapter.implementation_id,
            "implementation_version": base.adapter.implementation_version,
            "pipeline_hash": base.pipeline_hash,
            "attestation_id": base.attestation_id,
        },
        "replay_adapter": {
            "implementation_id": replay.adapter.implementation_id,
            "implementation_version": replay.adapter.implementation_version,
            "pipeline_hash": replay.pipeline_hash,
            "attestation_id": replay.attestation_id,
        },
    })


@dataclass(frozen=True, slots=True)
class FoldArtifact:
    class_order: tuple[str, ...]
    development_manifest_id: str
    recipe_hash: str
    fold_id: str
    fit_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    validation_group_ids: tuple[str, ...]
    validation_subject_groups: tuple[tuple[str, str], ...]
    base_model: LinearModelArtifact
    replay_model: LinearModelArtifact
    base_fit_id: str
    reference_fit_id: str
    reference_fit_hash: str
    final_refit: bool
    purpose: Literal["analytical", "synthetic_test"]
    artifact_id: str = ""

    def __post_init__(self) -> None:
        if self.base_model.model_id == self.replay_model.model_id:
            raise LearningError("Base and replay artifacts require distinct model identities.")
        if self.base_model.feature_pipeline_hash == self.replay_model.feature_pipeline_hash:
            raise LearningError("Base and replay feature pipelines must remain distinct.")
        if self.purpose == "analytical" and (
            self.base_model.adapter_attestation_id is None
            or self.replay_model.adapter_attestation_id is None
        ):
            raise LearningError("Analytical fold artifacts require trusted adapter attestations.")
        if not self.development_manifest_id or len(self.recipe_hash) != 64:
            raise LearningError("Fold artifacts require a development manifest and model recipe hash.")
        if set(self.fit_ids) & set(self.validation_ids):
            raise FoldLeakageError("Fit and validation IDs overlap.")
        if set(self.fit_group_ids) & set(self.validation_group_ids):
            raise FoldLeakageError("Fit and validation identity groups overlap.")
        if dict(self.validation_subject_groups).keys() != set(self.validation_ids):
            raise LearningError("Validation subject/group proof differs from validation IDs.")
        if set(dict(self.validation_subject_groups).values()) != set(self.validation_group_ids):
            raise LearningError("Validation subject/group proof differs from validation groups.")
        if self.final_refit != (not self.validation_ids):
            raise LearningError("Only a validation-free private refit may be final.")
        payload = self.to_dict(include_id=False)
        expected = "fold_" + _digest(payload)[:24]
        if self.artifact_id and self.artifact_id != expected:
            raise LearningError("Stale fold artifact ID.")
        object.__setattr__(self, "artifact_id", expected)

    def to_dict(self, *, include_id: bool = True) -> dict[str, Any]:
        result = {
            "schema_version": LEARNING_SCHEMA_VERSION,
            "development_manifest_id": self.development_manifest_id,
            "recipe_hash": self.recipe_hash,
            "class_order": list(self.class_order), "fold_id": self.fold_id,
            "fit_ids": list(self.fit_ids), "validation_ids": list(self.validation_ids),
            "excluded_ids": list(self.excluded_ids), "fit_group_ids": list(self.fit_group_ids),
            "validation_group_ids": list(self.validation_group_ids),
            "validation_subject_groups": [list(item) for item in self.validation_subject_groups],
            "base_model": self.base_model.to_dict(), "replay_model": self.replay_model.to_dict(),
            "base_fit_id": self.base_fit_id, "reference_fit_id": self.reference_fit_id,
            "reference_fit_hash": self.reference_fit_hash, "final_refit": self.final_refit,
            "purpose": self.purpose,
        }
        if include_id:
            result["artifact_id"] = self.artifact_id
        return result


@dataclass(frozen=True, slots=True)
class FoldArtifactProof:
    fold_artifact_id: str
    development_manifest_id: str
    recipe_hash: str
    fold_id: str
    fit_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    validation_ids: tuple[str, ...]
    validation_subject_groups: tuple[tuple[str, str], ...]
    excluded_ids: tuple[str, ...]
    base_model_artifact_hash: str
    replay_model_artifact_hash: str
    reference_fit_id: str
    reference_fit_hash: str
    final_refit: bool
    artifact_content_hash: str

    @classmethod
    def from_artifact(cls, artifact: FoldArtifact) -> Self:
        return cls(
            fold_artifact_id=artifact.artifact_id,
            development_manifest_id=artifact.development_manifest_id,
            recipe_hash=artifact.recipe_hash, fold_id=artifact.fold_id,
            fit_ids=artifact.fit_ids, fit_group_ids=artifact.fit_group_ids,
            validation_ids=artifact.validation_ids,
            validation_subject_groups=artifact.validation_subject_groups,
            excluded_ids=artifact.excluded_ids,
            base_model_artifact_hash=artifact.base_model.artifact_id,
            replay_model_artifact_hash=artifact.replay_model.artifact_id,
            reference_fit_id=artifact.reference_fit_id,
            reference_fit_hash=artifact.reference_fit_hash,
            final_refit=artifact.final_refit,
            artifact_content_hash=_digest(artifact.to_dict()),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "fold_artifact_id": self.fold_artifact_id,
            "development_manifest_id": self.development_manifest_id,
            "recipe_hash": self.recipe_hash, "fold_id": self.fold_id,
            "fit_ids": list(self.fit_ids), "fit_group_ids": list(self.fit_group_ids),
            "validation_ids": list(self.validation_ids),
            "validation_subject_groups": [list(item) for item in self.validation_subject_groups],
            "excluded_ids": list(self.excluded_ids),
            "base_model_artifact_hash": self.base_model_artifact_hash,
            "replay_model_artifact_hash": self.replay_model_artifact_hash,
            "reference_fit_id": self.reference_fit_id,
            "reference_fit_hash": self.reference_fit_hash,
            "final_refit": self.final_refit,
            "artifact_content_hash": self.artifact_content_hash,
        }


def _canonical_feature_values(
    values: Mapping[str, float | int | None], feature_names: Sequence[str],
) -> tuple[tuple[str, float | None], ...]:
    result: list[tuple[str, float | None]] = []
    for name in feature_names:
        value = values.get(name)
        try:
            normalized = None if value is None else float(value)
        except (TypeError, ValueError):
            normalized = None
        if normalized is not None and not math.isfinite(normalized):
            normalized = None
        result.append((name, normalized))
    return tuple(result)


@dataclass(frozen=True, slots=True, init=False)
class FoldPredictionReceipt:
    """Private receipt for the exact fold-artifact execution and output payload."""

    subject_id: str
    subject_hash: str
    snapshot_hash: str
    fold_artifact_id: str
    base_features: tuple[tuple[str, float | None], ...]
    state_features: tuple[tuple[str, float | None], ...]
    base_probabilities: tuple[float, ...]
    state_probabilities: tuple[float, ...]
    receipt_id: str

    def __init__(
        self,
        *,
        subject_id: str,
        subject_hash: str,
        snapshot_hash: str,
        fold_artifact_id: str,
        base_features: tuple[tuple[str, float | None], ...],
        state_features: tuple[tuple[str, float | None], ...],
        base_probabilities: tuple[float, ...],
        state_probabilities: tuple[float, ...],
        receipt_id: str = "",
        _seal: object | None = None,
    ) -> None:
        if _seal is not _PREDICTION_RECEIPT_SEAL:
            raise LearningError("Fold prediction receipts are issued only by predict_fold().")
        payload = {
            "subject_id": subject_id, "subject_hash": subject_hash,
            "snapshot_hash": snapshot_hash, "fold_artifact_id": fold_artifact_id,
            "base_features": base_features, "state_features": state_features,
            "base_probabilities": base_probabilities,
            "state_probabilities": state_probabilities,
        }
        expected = "fold_output_" + _digest(payload)[:24]
        if receipt_id and receipt_id != expected:
            raise LearningError("Stale fold prediction receipt ID.")
        for name, value in payload.items():
            object.__setattr__(self, name, value)
        object.__setattr__(self, "receipt_id", expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject_id": self.subject_id, "subject_hash": self.subject_hash,
            "snapshot_hash": self.snapshot_hash, "fold_artifact_id": self.fold_artifact_id,
            "base_features": [list(item) for item in self.base_features],
            "state_features": [list(item) for item in self.state_features],
            "base_probabilities": list(self.base_probabilities),
            "state_probabilities": list(self.state_probabilities),
            "receipt_id": self.receipt_id,
        }


@dataclass(frozen=True, slots=True, init=False)
class FoldArtifactManifest:
    proofs: tuple[FoldArtifactProof, ...]
    outputs: tuple[FoldPredictionReceipt, ...]
    _artifacts: tuple[FoldArtifact, ...]
    manifest_id: str = ""

    def __init__(
        self,
        proofs: tuple[FoldArtifactProof, ...],
        outputs: tuple[FoldPredictionReceipt, ...] = (),
        artifacts: tuple[FoldArtifact, ...] = (),
        manifest_id: str = "",
        *,
        _seal: object | None = None,
    ) -> None:
        if _seal is not _FOLD_MANIFEST_SEAL:
            raise LearningError("FoldArtifactManifest must be created by seal_fold_artifacts().")
        object.__setattr__(self, "proofs", proofs)
        object.__setattr__(self, "outputs", outputs)
        object.__setattr__(self, "_artifacts", artifacts)
        object.__setattr__(self, "manifest_id", manifest_id)
        self.__post_init__()

    def __post_init__(self) -> None:
        if not self.proofs:
            raise LearningError("A sealed fold-artifact manifest cannot be empty.")
        if not self.outputs:
            raise LearningError("A sealed fold-artifact manifest requires persisted fold outputs.")
        if len(self.proofs) != len(self._artifacts):
            raise LearningError("Fold-artifact proofs and immutable artifacts differ.")
        artifact_ids = tuple(item.fold_artifact_id for item in self.proofs)
        fold_ids = tuple(item.fold_id for item in self.proofs)
        if len(set(artifact_ids)) != len(artifact_ids) or len(set(fold_ids)) != len(fold_ids):
            raise LearningError("Fold-artifact manifest contains duplicate artifact or fold IDs.")
        if any(item.final_refit for item in self.proofs):
            raise FoldLeakageError("Final-refit artifacts cannot enter an OOF manifest.")
        if len({item.development_manifest_id for item in self.proofs}) != 1:
            raise FoldLeakageError("OOF artifacts do not share one development manifest.")
        if len({item.recipe_hash for item in self.proofs}) != 1:
            raise FoldLeakageError("OOF artifacts do not share one frozen model recipe.")
        for proof, artifact in zip(self.proofs, self._artifacts, strict=True):
            if proof != FoldArtifactProof.from_artifact(artifact):
                raise LearningError("Fold-artifact proof differs from its immutable artifact.")
        output_keys = tuple((item.subject_id, item.fold_artifact_id) for item in self.outputs)
        if len(set(output_keys)) != len(output_keys):
            raise LearningError("Fold-output manifest contains duplicate subject outputs.")
        if not set(item.fold_artifact_id for item in self.outputs) <= set(artifact_ids):
            raise LearningError("Fold output refers to an artifact outside the sealed manifest.")
        validation_ids = tuple(
            subject_id for proof in self.proofs for subject_id in proof.validation_ids
        )
        output_ids = tuple(item.subject_id for item in self.outputs)
        if len(set(validation_ids)) != len(validation_ids) or set(output_ids) != set(validation_ids):
            raise FoldLeakageError(
                "Sealed fold outputs must cover every validation subject exactly once."
            )
        expected = "fold_manifest_" + _digest({
            "proofs": [item.to_dict() for item in self.proofs],
            "outputs": [item.to_dict() for item in self.outputs],
        })[:24]
        if self.manifest_id and self.manifest_id != expected:
            raise LearningError("Stale fold-artifact manifest ID.")
        object.__setattr__(self, "manifest_id", expected)

    def resolve(self, artifact_id: str) -> FoldArtifactProof:
        matches = [item for item in self.proofs if item.fold_artifact_id == artifact_id]
        if len(matches) != 1:
            raise FoldLeakageError("OOF prediction does not resolve to exactly one sealed fold artifact.")
        return matches[0]

    def resolve_artifact(self, artifact_id: str) -> FoldArtifact:
        matches = [item for item in self._artifacts if item.artifact_id == artifact_id]
        if len(matches) != 1:
            raise FoldLeakageError("OOF prediction does not resolve to one immutable fold artifact.")
        return matches[0]

    def resolve_output(self, subject_id: str, artifact_id: str) -> FoldPredictionReceipt:
        matches = [
            item for item in self.outputs
            if item.subject_id == subject_id and item.fold_artifact_id == artifact_id
        ]
        if len(matches) != 1:
            raise FoldLeakageError("OOF prediction does not resolve to one sealed fold output.")
        return matches[0]


def seal_fold_artifacts(
    artifacts: Sequence[FoldArtifact], predictions: Sequence["FoldPrediction"],
) -> FoldArtifactManifest:
    """Seal immutable fold artifacts and their exact predict-fold output receipts."""

    artifact_tuple = tuple(artifacts)
    by_id = {item.artifact_id: item for item in artifact_tuple}
    receipts: list[FoldPredictionReceipt] = []
    for prediction in predictions:
        artifact = by_id.get(prediction.fold_artifact_id)
        if artifact is None:
            raise FoldLeakageError("Fold output refers to an unsealed fold artifact.")
        _validate_prediction_receipt_against_artifact(prediction, artifact)
        receipts.append(prediction.output_receipt)
    return FoldArtifactManifest(
        tuple(FoldArtifactProof.from_artifact(item) for item in artifact_tuple),
        tuple(receipts), artifact_tuple, _seal=_FOLD_MANIFEST_SEAL,
    )


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
    fold_artifact_id: str
    fold_id: str
    fit_ids: tuple[str, ...]
    fit_group_ids: tuple[str, ...]
    excluded_ids: tuple[str, ...]
    validation_id: str | None
    validation_group_id: str | None
    final_refit: bool
    base_fit_id: str
    base_fit_hash: str
    reference_fit_id: str
    reference_fit_hash: str
    output_receipt: FoldPredictionReceipt
    prediction_hash: str = ""

    def __post_init__(self) -> None:
        if self.subject != self.evidence_snapshot.subject:
            raise LearningError("Fold prediction subject differs from its evidence snapshot.")
        if not isinstance(self.output_receipt, FoldPredictionReceipt):
            raise LearningError("Fold prediction requires a predict_fold output receipt.")
        _probability_vector(self.base_probabilities, self.subject.class_order)
        _probability_vector(self.state_probabilities, self.subject.class_order)
        if self.base_fit_hash == self.replay_context.replay_model.artifact_id:
            raise LearningError("Base and replay model artifact hashes must remain distinct.")
        if (
            self.evidence_snapshot.reference_fit_id != self.reference_fit_id
            or self.evidence_snapshot.reference_fit_hash != self.reference_fit_hash
            or self.replay_context.reference_fit_id != self.reference_fit_id
            or self.replay_context.reference_fit_hash != self.reference_fit_hash
            or self.replay_context.before_probabilities != self.state_probabilities
        ):
            raise LearningError("Fold prediction reference or replay context is not internally bound.")
        if self.final_refit:
            if self.validation_id is not None or self.validation_group_id is not None:
                raise LearningError("Final-refit predictions cannot claim OOF validation provenance.")
        else:
            if (self.validation_id, self.validation_group_id) != (
                self.subject.subject_id, self.subject.source_group_id,
            ):
                raise FoldLeakageError("OOF validation proof differs from the prediction subject.")
        payload = {
            "subject_hash": self.subject.content_hash,
            "base_probabilities": self.base_probabilities,
            "state_probabilities": self.state_probabilities,
            "snapshot_hash": self.evidence_snapshot.snapshot_hash,
            "fold_artifact_id": self.fold_artifact_id,
            "fold_id": self.fold_id,
            "fit_ids": self.fit_ids,
            "fit_group_ids": self.fit_group_ids,
            "excluded_ids": self.excluded_ids,
            "validation_id": self.validation_id,
            "validation_group_id": self.validation_group_id,
            "final_refit": self.final_refit,
            "base_fit_id": self.base_fit_id,
            "base_fit_hash": self.base_fit_hash,
            "replay_model_artifact_hash": self.replay_context.replay_model.artifact_id,
            "reference_fit_id": self.reference_fit_id,
            "reference_fit_hash": self.reference_fit_hash,
            "output_receipt_id": self.output_receipt.receipt_id,
        }
        expected = _digest(payload)
        if self.prediction_hash and self.prediction_hash != expected:
            raise LearningError("Stale fold prediction hash.")
        object.__setattr__(self, "prediction_hash", expected)


def _validate_prediction_receipt_against_artifact(
    prediction: FoldPrediction, artifact: FoldArtifact,
) -> None:
    receipt = prediction.output_receipt
    subject = prediction.subject
    expected_base = artifact.base_model.predict_proba((dict(receipt.base_features),))[0]
    expected_state = artifact.replay_model.predict_proba((dict(receipt.state_features),))[0]
    if (
        receipt.subject_id != subject.subject_id
        or receipt.subject_hash != subject.content_hash
        or receipt.snapshot_hash != prediction.evidence_snapshot.snapshot_hash
        or receipt.fold_artifact_id != artifact.artifact_id
        or receipt.base_probabilities != expected_base
        or receipt.state_probabilities != expected_state
        or prediction.base_probabilities != receipt.base_probabilities
        or prediction.state_probabilities != receipt.state_probabilities
    ):
        raise FoldLeakageError("OOF prediction payload differs from its sealed fold output.")


def _validate_adapter_artifact(
    artifact: LinearModelArtifact, resolved: _ResolvedAdapter, *,
    fit_ids: tuple[str, ...], excluded_ids: tuple[str, ...],
    feature_names: tuple[str, ...], class_order: tuple[str, ...], analytical_run: bool,
) -> None:
    adapter = resolved.adapter
    if (
        artifact.adapter_implementation != adapter.implementation_id
        or artifact.adapter_version != adapter.implementation_version
        or artifact.feature_pipeline_hash != resolved.pipeline_hash
        or artifact.adapter_attestation_id != resolved.attestation_id
        or artifact.fit_ids != fit_ids
        or artifact.excluded_ids != excluded_ids
        or artifact.feature_names != feature_names
        or artifact.class_order != class_order
    ):
        raise LearningError("Predictor adapter returned an artifact with mismatched provenance.")
    if analytical_run and artifact.adapter_attestation_id is None:
        raise LearningError("Analytical fitting produced an unattested model artifact.")


def _fit_artifact(
    inputs: FoldInputs,
    fit: tuple[str, ...],
    validation: tuple[str, ...],
    frozen_config: FrozenFoldConfig,
    *,
    final_refit: bool,
) -> FoldArtifact:
    excluded = tuple(dict.fromkeys((*frozen_config.excluded_ids, *validation)))
    known = set(inputs.cases)
    missing = sorted((set(fit) | set(validation)) - known)
    if missing:
        raise LearningError(f"Unknown fold subject IDs: {missing}")
    selected = fit + validation
    non_development = [
        item for item in selected if inputs.cases[item].subject.partition != "development"
    ]
    if non_development:
        raise FoldLeakageError(
            f"Fold fitting and OOF validation require development subjects: {non_development}"
        )
    if frozen_config.development_manifest is None:
        raise LearningError("Fold fitting requires a sealed canonical development manifest.")
    development_ids = _validate_development_manifest(
        inputs, frozen_config.development_manifest,
    )
    if set(selected) != set(development_ids):
        raise FoldLeakageError(
            "Fold fit and validation IDs must partition the canonical development manifest."
        )
    if set(fit) & set(validation) or set(fit) & set(excluded):
        raise FoldLeakageError("Fit IDs overlap validation or excluded IDs.")
    fit_groups = tuple(dict.fromkeys(inputs.cases[item].subject.source_group_id for item in fit))
    validation_subject_groups = tuple(
        (item, inputs.cases[item].subject.source_group_id) for item in validation
    )
    validation_groups = tuple(dict.fromkeys(group for _, group in validation_subject_groups))
    if set(fit_groups) & set(validation_groups):
        raise FoldLeakageError("A validation identity group occurs in fit IDs.")
    missing_labels = [item for item in fit if item not in inputs.labels]
    if missing_labels:
        raise LearningError(f"Missing fit labels: {missing_labels}")
    if not {inputs.labels[item] for item in fit} <= set(frozen_config.class_order):
        raise LearningError("Fit labels are outside class_order.")
    if final_refit:
        if validation:
            raise LearningError("Private full-development refit cannot contain validation IDs.")
        fold_id = "full_development"
    else:
        fold_ids = {inputs.cases[item].subject.fold_id for item in validation}
        if len(fold_ids) != 1:
            raise LearningError("A fold artifact requires one validation fold ID.")
        fold_id = next(iter(fold_ids))
    base_resolved = _resolve_adapter(frozen_config, "base")
    replay_resolved = _resolve_adapter(frozen_config, "replay")
    recipe_hash = _model_recipe_hash(frozen_config, base_resolved, replay_resolved)
    base_adapter = base_resolved.adapter
    replay_adapter = replay_resolved.adapter
    fit_cases = {subject_id: inputs.cases[subject_id] for subject_id in fit}
    fit_labels = {subject_id: inputs.labels[subject_id] for subject_id in fit}
    base = base_adapter.fit(
        cases=fit_cases, labels=fit_labels, fit_ids=fit, excluded_ids=excluded,
        feature_names=frozen_config.base_feature_names,
        model_id=frozen_config.base_model_id, class_order=frozen_config.class_order,
        seed=frozen_config.seed, c=frozen_config.c, max_iter=frozen_config.max_iter,
        adapter_attestation_id=base_resolved.attestation_id,
    )
    replay = replay_adapter.fit(
        cases=fit_cases, labels=fit_labels, fit_ids=fit, excluded_ids=excluded,
        feature_names=frozen_config.state_feature_names,
        model_id=frozen_config.replay_model_id, class_order=frozen_config.class_order,
        seed=frozen_config.seed, c=frozen_config.c, max_iter=frozen_config.max_iter,
        adapter_attestation_id=replay_resolved.attestation_id,
    )
    _validate_adapter_artifact(
        base, base_resolved, fit_ids=fit, excluded_ids=excluded,
        feature_names=frozen_config.base_feature_names, class_order=frozen_config.class_order,
        analytical_run=frozen_config.analytical_run,
    )
    _validate_adapter_artifact(
        replay, replay_resolved, fit_ids=fit, excluded_ids=excluded,
        feature_names=frozen_config.state_feature_names, class_order=frozen_config.class_order,
        analytical_run=frozen_config.analytical_run,
    )
    reference_hash = _digest({
        "fit_ids": fit, "fit_group_ids": fit_groups,
        "replay_model_artifact_id": replay.artifact_id,
        "replay_feature_pipeline_hash": replay.feature_pipeline_hash,
    })
    return FoldArtifact(
        class_order=frozen_config.class_order,
        development_manifest_id=frozen_config.development_manifest.manifest_id,
        recipe_hash=recipe_hash, fold_id=fold_id, fit_ids=fit,
        validation_ids=validation, excluded_ids=excluded, fit_group_ids=fit_groups,
        validation_group_ids=validation_groups,
        validation_subject_groups=validation_subject_groups,
        base_model=base, replay_model=replay, base_fit_id=f"base_{fold_id}",
        reference_fit_id=f"reference_{fold_id}", reference_fit_hash=reference_hash,
        final_refit=final_refit,
        purpose="analytical" if frozen_config.analytical_run else "synthetic_test",
    )


def fit_fold(
    inputs: FoldInputs,
    fit_ids: Sequence[str],
    validation_ids: Sequence[str],
    frozen_config: FrozenFoldConfig,
) -> FoldArtifact:
    """Fit both branches on explicit IDs and reject subject/group leakage."""

    fit = _ordered_ids(fit_ids, "fit_ids")
    validation = _ordered_ids(validation_ids, "validation_ids")
    return _fit_artifact(inputs, fit, validation, frozen_config, final_refit=False)


def predict_fold(fold_artifact: FoldArtifact, case_inputs: FoldCaseInput) -> FoldPrediction:
    """Predict with the paired base/replay fit and bind same-fold evidence."""

    subject = case_inputs.subject
    if not fold_artifact.final_refit:
        if subject.partition != "development":
            raise FoldLeakageError("OOF prediction requires a development subject.")
        if subject.subject_id not in fold_artifact.validation_ids:
            raise LearningError("OOF prediction subject is not assigned to this validation fold.")
        if subject.subject_id in fold_artifact.fit_ids:
            raise FoldLeakageError("OOF prediction subject occurs in fit IDs.")
        if subject.source_group_id in fold_artifact.fit_group_ids:
            raise FoldLeakageError("OOF prediction identity group occurs in fit groups.")
    elif subject.partition == "development":
        raise FoldLeakageError("Final-refit models cannot generate development calibration rows.")
    snapshot = case_inputs.evidence_snapshot
    if (
        snapshot.reference_fit_id != fold_artifact.reference_fit_id
        or snapshot.reference_fit_hash != fold_artifact.reference_fit_hash
    ):
        raise LearningError("Evidence snapshot is not from the paired fold reference fit.")
    base_probability = fold_artifact.base_model.predict_proba((case_inputs.base_features,))[0]
    state_probability = fold_artifact.replay_model.predict_proba((case_inputs.state_features,))[0]
    receipt = FoldPredictionReceipt(
        subject_id=subject.subject_id, subject_hash=subject.content_hash,
        snapshot_hash=snapshot.snapshot_hash, fold_artifact_id=fold_artifact.artifact_id,
        base_features=_canonical_feature_values(
            case_inputs.base_features, fold_artifact.base_model.feature_names,
        ),
        state_features=_canonical_feature_values(
            case_inputs.state_features, fold_artifact.replay_model.feature_names,
        ),
        base_probabilities=base_probability, state_probabilities=state_probability,
        _seal=_PREDICTION_RECEIPT_SEAL,
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
        fold_artifact_id=fold_artifact.artifact_id, fold_id=fold_artifact.fold_id,
        fit_ids=fold_artifact.fit_ids, fit_group_ids=fold_artifact.fit_group_ids,
        excluded_ids=fold_artifact.excluded_ids,
        validation_id=None if fold_artifact.final_refit else subject.subject_id,
        validation_group_id=None if fold_artifact.final_refit else subject.source_group_id,
        final_refit=fold_artifact.final_refit, base_fit_id=fold_artifact.base_fit_id,
        base_fit_hash=fold_artifact.base_model.artifact_id,
        reference_fit_id=fold_artifact.reference_fit_id,
        reference_fit_hash=fold_artifact.reference_fit_hash,
        output_receipt=receipt,
    )


def _verify_oof_prediction(
    prediction: FoldPrediction, fold_manifest: FoldArtifactManifest,
) -> FoldArtifactProof:
    proof = fold_manifest.resolve(prediction.fold_artifact_id)
    artifact = fold_manifest.resolve_artifact(prediction.fold_artifact_id)
    persisted_output = fold_manifest.resolve_output(
        prediction.subject.subject_id, prediction.fold_artifact_id,
    )
    if prediction.output_receipt != persisted_output:
        raise FoldLeakageError("OOF prediction differs from its sealed fold output receipt.")
    _validate_prediction_receipt_against_artifact(prediction, artifact)
    subject = prediction.subject
    expected_group = dict(proof.validation_subject_groups).get(subject.subject_id)
    if proof.final_refit or prediction.final_refit:
        raise FoldLeakageError("Final-refit predictions cannot fit calibration or prove OOF completion.")
    if subject.partition != "development":
        raise FoldLeakageError("OOF prediction is not a development subject.")
    if (
        prediction.fold_id != proof.fold_id
        or subject.fold_id != proof.fold_id
        or prediction.fit_ids != proof.fit_ids
        or prediction.fit_group_ids != proof.fit_group_ids
        or prediction.excluded_ids != proof.excluded_ids
        or prediction.base_fit_hash != proof.base_model_artifact_hash
        or prediction.replay_context.replay_model.artifact_id != proof.replay_model_artifact_hash
        or prediction.reference_fit_id != proof.reference_fit_id
        or prediction.reference_fit_hash != proof.reference_fit_hash
        or prediction.validation_id != subject.subject_id
        or prediction.validation_group_id != subject.source_group_id
        or expected_group != subject.source_group_id
        or subject.subject_id not in proof.validation_ids
        or subject.subject_id not in proof.excluded_ids
        or subject.subject_id in proof.fit_ids
        or subject.source_group_id in proof.fit_group_ids
    ):
        raise FoldLeakageError("OOF prediction does not match its resolved sealed fold artifact.")
    return proof


def refit_full_development(
    inputs: FoldInputs,
    oof_predictions: Sequence[FoldPrediction],
    fold_manifest: FoldArtifactManifest,
    frozen_config: FrozenFoldConfig,
) -> FoldArtifact:
    """Refit only after every development subject has exactly one OOF row."""

    if frozen_config.development_manifest is None:
        raise LearningError("Full refit requires a sealed canonical development manifest.")
    expected = _validate_development_manifest(inputs, frozen_config.development_manifest)
    manifest_ids = {item.development_manifest_id for item in fold_manifest.proofs}
    if manifest_ids != {frozen_config.development_manifest.manifest_id}:
        raise FoldLeakageError("OOF artifacts differ from the canonical development manifest.")
    base_resolved = _resolve_adapter(frozen_config, "base")
    replay_resolved = _resolve_adapter(frozen_config, "replay")
    expected_recipe = _model_recipe_hash(frozen_config, base_resolved, replay_resolved)
    if {item.recipe_hash for item in fold_manifest.proofs} != {expected_recipe}:
        raise FoldLeakageError("Full-development refit changed the verified OOF model recipe.")
    for prediction in oof_predictions:
        _verify_oof_prediction(prediction, fold_manifest)
    observed = [item.subject.subject_id for item in oof_predictions]
    if len(observed) != len(set(observed)) or set(observed) != set(expected):
        raise LearningError("Full development refit requires exactly one OOF prediction per subject.")
    prediction_subjects = {item.subject.subject_id: item.subject for item in oof_predictions}
    if any(inputs.cases[item].subject != prediction_subjects[item] for item in expected):
        raise FoldLeakageError("Full-development refit subject records differ from verified OOF rows.")
    return _fit_artifact(inputs, expected, (), frozen_config, final_refit=True)


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
    fold_prediction: FoldPrediction | None = None

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
        if self.fold_prediction is not None and (
            self.fold_prediction.subject != self.subject
            or self.fold_prediction.base_probabilities != self.base_probabilities
        ):
            raise FoldLeakageError(
                "Calibration row differs from its bound fold prediction subject or base probabilities."
            )

    @classmethod
    def from_fusion(
        cls,
        fusion: FusionRow,
        *,
        state_probabilities_before: Sequence[float] | None = None,
        state_probabilities_after: Sequence[float] | None = None,
        fold_prediction: FoldPrediction | None = None,
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
            fold_prediction=fold_prediction,
        )

    @classmethod
    def from_fold_prediction(
        cls,
        prediction: FoldPrediction,
        *,
        fusion_hash: str,
        agent_v0_scores: Mapping[str, int] | None,
        agent_v1_scores: Mapping[str, int] | None,
        v1_required: bool,
        state_probabilities_after: Sequence[float] | None,
        arm_refusal_reasons: Mapping[str, str | None],
    ) -> Self:
        after = None if state_probabilities_after is None else tuple(state_probabilities_after)
        before = None if after is None else prediction.state_probabilities
        return cls(
            subject=prediction.subject, fusion_hash=fusion_hash,
            base_probabilities=prediction.base_probabilities,
            agent_v0_scores=agent_v0_scores, agent_v1_scores=agent_v1_scores,
            v1_required=v1_required, state_probabilities_before=before,
            state_probabilities_after=after, arm_refusal_reasons=arm_refusal_reasons,
            fold_prediction=prediction,
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
    fold_manifest_id: str
    oof_prediction_hashes: tuple[str, ...]
    seed: int
    artifact_id: str = ""

    def __post_init__(self) -> None:
        order = _validate_class_order(self.class_order)
        expected_heads = set(HEADS_BY_TASK[self.task])
        if set(self.arms) != set(CALIBRATED_ARMS):
            raise LearningError("Calibrator artifact must contain B and all three joint arms.")
        if set(self.matched_baselines) != {"J-A", "J-S", "J-AS"}:
            raise LearningError("Matched baseline calibrators are incomplete.")
        if not self.fold_manifest_id or not self.oof_prediction_hashes:
            raise LearningError("Calibrators must bind a fold manifest and OOF predictions.")
        if len(set(self.oof_prediction_hashes)) != len(self.oof_prediction_hashes):
            raise LearningError("OOF prediction hashes must be unique.")
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
            "fold_manifest_id": self.fold_manifest_id,
            "oof_prediction_hashes": list(self.oof_prediction_hashes),
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
            fold_manifest_id=str(data["fold_manifest_id"]),
            oof_prediction_hashes=tuple(data["oof_prediction_hashes"]),
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


def _validate_oof_rows(
    rows: Sequence[CalibrationRow],
    labels: Mapping[str, str],
    fold_manifest: FoldArtifactManifest,
) -> tuple[str, tuple[str, ...]]:
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
        if row.fold_prediction is None:
            raise FoldLeakageError("Calibration rows require a bound FoldPrediction.")
        _verify_oof_prediction(row.fold_prediction, fold_manifest)
    return next(iter(tasks)), next(iter(orders))


def fit_joint_calibrators(
    oof_rows: Sequence[CalibrationRow],
    development_labels: Mapping[str, str],
    config: CalibrationConfig,
    fold_manifest: FoldArtifactManifest,
) -> JointCalibrators:
    """Fit B and each joint arm independently using fixed OOF-only heads."""

    if not isinstance(fold_manifest, FoldArtifactManifest):
        raise LearningError("fit_joint_calibrators requires a sealed FoldArtifactManifest.")
    task, class_order = _validate_oof_rows(oof_rows, development_labels, fold_manifest)
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
        matched_baselines=matched, fold_manifest_id=fold_manifest.manifest_id,
        oof_prediction_hashes=tuple(
            row.fold_prediction.prediction_hash for row in oof_rows
            if row.fold_prediction is not None
        ),
        seed=config.seed,
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
    "FoldArtifact", "FoldArtifactManifest", "FoldArtifactProof", "FoldCaseInput",
    "FoldInputs", "FoldLeakageError", "FoldPrediction",
    "FrozenFoldConfig", "HeadCalibrator", "JointCalibrators", "LearningError",
    "LinearModelArtifact", "PredictorAdapter", "ReplayContext",
    "SyntheticLinearPredictorAdapter", "agent_contrast",
    "apply_head", "combine_head_probabilities", "fit_fold", "fit_joint_calibrators",
    "head_probability", "model_identity_manifest", "predict_fold", "predict_joint", "refit_full_development",
    "replay_log_odds_delta", "seal_fold_artifacts",
]
