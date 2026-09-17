"""Task-conditioned statistical Module A with auditable prediction packets.

This module is intentionally independent from the legacy ``train_ours`` path.
It provides the first, small-data implementation of the Module A contract: a
class-balanced regularized linear classifier plus a deterministic explanation
packet that can be handed to an evidence-review/replay layer later.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .utils import hash_values


EXPLANATION_PACKET_VERSION = "advoice.module_a.explanation_packet.v1"
MODULE_A_VERSION = "advoice.module_a.linear.v1"


def _canonical(value: Any) -> str:
    """Serialize data deterministically for public packet and snapshot hashes."""

    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
        allow_nan=False,
    )


def snapshot_hash(snapshot: Any | None) -> str:
    """Return a stable hash for an evidence/state/artifact snapshot.

    ``None`` is deliberately hashable: a missing snapshot is distinguishable
    from an empty snapshot and therefore remains auditable.
    """

    return hash_values([{"snapshot": snapshot if snapshot is not None else {"status": "missing"}}])


def _probability_dict(values: np.ndarray, labels: Sequence[str]) -> dict[str, float]:
    return {label: float(values[index]) for index, label in enumerate(labels)}


def _softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - np.max(logits)
    exponentiated = np.exp(shifted)
    return exponentiated / exponentiated.sum()


def _is_qc_feature(name: str, branch: str | None = None) -> bool:
    normalized = str(name).lower()
    branch_normalized = str(branch or "").lower()
    return (
        branch_normalized in {"qc", "qc_only", "quality", "quality_control"}
        or normalized == "qc"
        or normalized.startswith("qc_")
        or normalized.startswith("quality_")
        or normalized.endswith("_qc")
    )


@dataclass(frozen=True)
class ExplanationPacket:
    """Versioned, JSON-stable account of one Module A prediction."""

    schema_version: str
    module_version: str
    class_order: tuple[str, ...]
    predicted_label: str
    raw_probabilities: Mapping[str, float]
    calibrated_probabilities: Mapping[str, float] | None
    calibration_status: str
    logits: Mapping[str, float]
    intercepts: Mapping[str, float]
    feature_contributions: Mapping[str, Mapping[str, float]]
    branch_contributions: Mapping[str, Mapping[str, float]]
    consumed_evidence_ids: tuple[str, ...]
    uncertainty: Mapping[str, float]
    fold_disagreement: Mapping[str, float] | None
    ood: Mapping[str, float | bool | str]
    applicability_status: str
    hashes: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-native data with a fixed field and class order."""

        return {
            "schema_version": self.schema_version,
            "module_version": self.module_version,
            "class_order": list(self.class_order),
            "predicted_label": self.predicted_label,
            "raw_probabilities": dict(self.raw_probabilities),
            "calibrated_probabilities": (
                None if self.calibrated_probabilities is None else dict(self.calibrated_probabilities)
            ),
            "calibration_status": self.calibration_status,
            "logits": dict(self.logits),
            "intercepts": dict(self.intercepts),
            "feature_contributions": {
                label: dict(self.feature_contributions[label]) for label in self.class_order
            },
            "branch_contributions": {
                label: dict(self.branch_contributions[label]) for label in self.class_order
            },
            "consumed_evidence_ids": list(self.consumed_evidence_ids),
            "uncertainty": dict(self.uncertainty),
            "fold_disagreement": None if self.fold_disagreement is None else dict(self.fold_disagreement),
            "ood": dict(self.ood),
            "applicability_status": self.applicability_status,
            "hashes": dict(self.hashes),
        }

    def to_json(self) -> str:
        return _canonical(self.to_dict())


class TaskConditionedStatisticalExpert:
    """Regularized, class-balanced linear expert with frozen task adapters.

    The caller supplies clinical/state features and may name task and language
    columns.  Those categorical columns become fixed one-hot adapters.  QC
    variables are excluded from disease logits even if callers accidentally
    include them in ``feature_columns``; they may only be represented in a
    separate quality/OOD layer by a later component.
    """

    def __init__(
        self,
        labels: Sequence[str],
        *,
        c: float = 1.0,
        max_iter: int = 2000,
        task_column: str | None = None,
        language_column: str | None = None,
        random_state: int = 20260813,
        module_version: str = MODULE_A_VERSION,
    ) -> None:
        ordered = tuple(str(label) for label in labels)
        if len(ordered) < 2 or len(set(ordered)) != len(ordered):
            raise ValueError("labels must contain at least two unique labels in their fixed output order.")
        if c <= 0:
            raise ValueError("c must be positive.")
        self.labels = ordered
        self.c = float(c)
        self.max_iter = int(max_iter)
        self.task_column = task_column
        self.language_column = language_column
        self.random_state = int(random_state)
        self.module_version = str(module_version)

    def fit(
        self,
        frame: pd.DataFrame,
        y: Iterable[str],
        *,
        feature_columns: Sequence[str] | None = None,
        feature_branches: Mapping[str, str] | None = None,
        qc_features: Iterable[str] = (),
        artifact_snapshot: Any | None = None,
        calibration: Mapping[str, Any] | None = None,
    ) -> "TaskConditionedStatisticalExpert":
        """Fit only disease-permitted features and freeze artifact identity.

        ``calibration`` supports a frozen temperature contract, e.g.
        ``{"temperature": 1.15, "status": "calibrated_on_development_oof"}``.
        Passing no calibrator is valid but packets explicitly report that no
        calibrated probability exists.
        """

        target = np.asarray([str(value) for value in y], dtype=object)
        if len(frame) != len(target):
            raise ValueError("frame and y must have the same number of rows.")
        observed = set(target)
        missing = [label for label in self.labels if label not in observed]
        unknown = sorted(observed - set(self.labels))
        if missing or unknown:
            raise ValueError(f"Training labels must exactly match fixed labels; missing={missing}, unknown={unknown}.")
        branches = {str(key): str(value) for key, value in (feature_branches or {}).items()}
        declared_qc = {str(value) for value in qc_features}
        if feature_columns is None:
            candidates = [
                str(column)
                for column in frame.select_dtypes(include=[np.number]).columns
                if str(column) not in {self.task_column, self.language_column}
            ]
        else:
            candidates = [str(column) for column in feature_columns]
        absent = [column for column in candidates if column not in frame]
        if absent:
            raise ValueError(f"Unknown feature columns: {absent}")
        self.excluded_qc_features_ = tuple(
            sorted(
                column
                for column in candidates
                if column in declared_qc or _is_qc_feature(column, branches.get(column))
            )
        )
        self.numeric_features_ = tuple(
            column for column in candidates if column not in self.excluded_qc_features_
        )
        if not self.numeric_features_ and not (self.task_column or self.language_column):
            raise ValueError("At least one disease-permitted feature or adapter is required.")
        for column in (self.task_column, self.language_column):
            if column is not None and column not in frame:
                raise ValueError(f"Configured adapter column is absent: {column}")

        self.feature_branches_ = {
            column: branches.get(column, "state") for column in self.numeric_features_
        }
        self.adapter_categories_: dict[str, tuple[str, ...]] = {}
        design, design_names, design_branches = self._fit_design(frame)
        self.design_feature_names_ = tuple(design_names)
        self.design_feature_branches_ = dict(design_branches)
        self.impute_values_ = np.nanmedian(design, axis=0)
        self.impute_values_ = np.where(np.isfinite(self.impute_values_), self.impute_values_, 0.0)
        imputed = np.where(np.isfinite(design), design, self.impute_values_)
        self.scale_means_ = imputed.mean(axis=0)
        self.scale_scales_ = imputed.std(axis=0)
        self.scale_scales_ = np.where(self.scale_scales_ > 1e-12, self.scale_scales_, 1.0)
        standardized = (imputed - self.scale_means_) / self.scale_scales_
        self.classifier_ = LogisticRegression(
            C=self.c,
            max_iter=self.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state,
        ).fit(standardized, target)
        learned = tuple(str(value) for value in self.classifier_.classes_)
        if set(learned) != set(self.labels):
            raise ValueError("Classifier classes differ from the fixed class order.")
        self.calibration_ = self._validate_calibration(calibration)
        self.artifact_snapshot_hash_ = snapshot_hash(artifact_snapshot)
        self.artifact_hash_ = snapshot_hash(self._artifact_payload())
        return self

    def _fit_design(self, frame: pd.DataFrame) -> tuple[np.ndarray, list[str], dict[str, str]]:
        parts: list[np.ndarray] = []
        names: list[str] = []
        branches: dict[str, str] = {}
        if self.numeric_features_:
            numeric = frame.loc[:, list(self.numeric_features_)].apply(pd.to_numeric, errors="coerce")
            parts.append(numeric.to_numpy(dtype=float))
            names.extend(self.numeric_features_)
            branches.update(self.feature_branches_)
        for column, branch in ((self.task_column, "task_adapter"), (self.language_column, "language_adapter")):
            if column is None:
                continue
            categories = tuple(sorted(frame[column].fillna("<missing>").astype(str).unique()))
            self.adapter_categories_[column] = categories
            values = frame[column].fillna("<missing>").astype(str)
            adapter_names = [f"{column}={category}" for category in categories]
            parts.append(np.column_stack([(values == category).astype(float) for category in categories]))
            names.extend(adapter_names)
            branches.update({name: branch for name in adapter_names})
        return np.column_stack(parts), names, branches

    def _transform_design(self, frame: pd.DataFrame) -> tuple[np.ndarray, dict[str, bool]]:
        if not hasattr(self, "classifier_"):
            raise RuntimeError("fit must be called before prediction.")
        parts: list[np.ndarray] = []
        unseen: dict[str, bool] = {}
        if self.numeric_features_:
            missing = [column for column in self.numeric_features_ if column not in frame]
            if missing:
                raise ValueError(f"Prediction frame is missing trained features: {missing}")
            parts.append(
                frame.loc[:, list(self.numeric_features_)].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
            )
        for column in (self.task_column, self.language_column):
            if column is None:
                continue
            if column not in frame:
                raise ValueError(f"Prediction frame is missing adapter column: {column}")
            values = frame[column].fillna("<missing>").astype(str)
            categories = self.adapter_categories_[column]
            unseen[column] = bool((~values.isin(categories)).any())
            parts.append(np.column_stack([(values == category).astype(float) for category in categories]))
        return np.column_stack(parts), unseen

    @staticmethod
    def _validate_calibration(calibration: Mapping[str, Any] | None) -> dict[str, Any] | None:
        if calibration is None:
            return None
        if "temperature" not in calibration:
            raise ValueError("Only a frozen temperature calibration is supported by this initial Module A contract.")
        temperature = float(calibration["temperature"])
        if not np.isfinite(temperature) or temperature <= 0:
            raise ValueError("Calibration temperature must be finite and positive.")
        return {
            "temperature": temperature,
            "status": str(calibration.get("status", "calibrated_on_development_oof")),
            "artifact_hash": snapshot_hash(dict(calibration)),
        }

    def _artifact_payload(self) -> dict[str, Any]:
        return {
            "module_version": self.module_version,
            "labels": self.labels,
            "c": self.c,
            "max_iter": self.max_iter,
            "task_column": self.task_column,
            "language_column": self.language_column,
            "numeric_features": self.numeric_features_,
            "excluded_qc_features": self.excluded_qc_features_,
            "design_feature_names": self.design_feature_names_,
            "design_feature_branches": self.design_feature_branches_,
            "adapter_categories": self.adapter_categories_,
            "impute_values": self.impute_values_.tolist(),
            "scale_means": self.scale_means_.tolist(),
            "scale_scales": self.scale_scales_.tolist(),
            "classes": self.classifier_.classes_.tolist(),
            "coefficients": self.classifier_.coef_.tolist(),
            "intercepts": self.classifier_.intercept_.tolist(),
            "calibration": self.calibration_,
            "artifact_snapshot_hash": self.artifact_snapshot_hash_,
        }

    def _class_linear_terms(self, standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return softmax-compatible logits, intercepts and coefficients in fixed order."""

        class_to_row = {str(label): index for index, label in enumerate(self.classifier_.classes_)}
        feature_count = len(self.design_feature_names_)
        coefficients = np.zeros((len(self.labels), feature_count), dtype=float)
        intercepts = np.zeros(len(self.labels), dtype=float)
        if len(self.labels) == 2 and self.classifier_.coef_.shape[0] == 1:
            positive = str(self.classifier_.classes_[1])
            target_index = self.labels.index(positive)
            coefficients[target_index] = self.classifier_.coef_[0]
            intercepts[target_index] = self.classifier_.intercept_[0]
        else:
            for target_index, label in enumerate(self.labels):
                source_index = class_to_row[label]
                coefficients[target_index] = self.classifier_.coef_[source_index]
                intercepts[target_index] = self.classifier_.intercept_[source_index]
        logits = intercepts + coefficients @ standardized
        return logits, intercepts, coefficients

    def _ood(self, raw_design: np.ndarray, unseen: Mapping[str, bool], supplied: Mapping[str, Any] | None) -> dict[str, float | bool | str]:
        missing_fraction = float(np.mean(~np.isfinite(raw_design)))
        imputed = np.where(np.isfinite(raw_design), raw_design, self.impute_values_)
        standardized = np.abs((imputed - self.scale_means_) / self.scale_scales_)
        maximum = float(np.max(standardized)) if standardized.size else 0.0
        mean = float(np.mean(standardized)) if standardized.size else 0.0
        output: dict[str, float | bool | str] = {
            "missing_feature_fraction": missing_fraction,
            "max_abs_standardized_distance": maximum,
            "mean_abs_standardized_distance": mean,
            "unseen_task": bool(unseen.get(self.task_column or "", False)),
            "unseen_language": bool(unseen.get(self.language_column or "", False)),
        }
        output["score"] = float(max(missing_fraction, min(1.0, mean / 3.0), min(1.0, maximum / 6.0)))
        if supplied:
            output.update({str(key): value for key, value in supplied.items()})
        return output

    def explain_case(
        self,
        case: pd.DataFrame | Mapping[str, Any],
        *,
        consumed_evidence_ids: Iterable[str] = (),
        evidence_snapshot: Any | None = None,
        state_snapshot: Any | None = None,
        fold_disagreement: Mapping[str, float] | None = None,
        ood_components: Mapping[str, Any] | None = None,
        applicability_status: str = "applicable",
    ) -> ExplanationPacket:
        """Generate one auditable packet; batch prediction is deliberately explicit."""

        if isinstance(case, pd.Series):
            frame = case.to_frame().T
        elif isinstance(case, Mapping):
            frame = pd.DataFrame([case])
        else:
            frame = case.copy()
        if len(frame) != 1:
            raise ValueError("explain_case accepts exactly one case; call it once per subject.")
        raw_design, unseen = self._transform_design(frame)
        standardized = (np.where(np.isfinite(raw_design), raw_design, self.impute_values_) - self.scale_means_) / self.scale_scales_
        logits, intercepts, coefficients = self._class_linear_terms(standardized[0])
        raw_probability = _softmax(logits)
        if self.calibration_ is None:
            calibrated_probability: np.ndarray | None = None
            calibration_status = "not_calibrated"
        else:
            calibrated_probability = _softmax(logits / float(self.calibration_["temperature"]))
            calibration_status = str(self.calibration_["status"])
        probability_for_decision = calibrated_probability if calibrated_probability is not None else raw_probability
        per_class_features = {
            label: {
                feature: float(coefficients[class_index, feature_index] * standardized[0, feature_index])
                for feature_index, feature in enumerate(self.design_feature_names_)
            }
            for class_index, label in enumerate(self.labels)
        }
        per_class_branches = {
            label: {
                branch: float(sum(
                    contribution
                    for feature, contribution in per_class_features[label].items()
                    if self.design_feature_branches_[feature] == branch
                ))
                for branch in sorted(set(self.design_feature_branches_.values()))
            }
            for label in self.labels
        }
        ordered_probability = np.sort(probability_for_decision)[::-1]
        entropy = float(-np.sum(raw_probability * np.log(np.clip(raw_probability, 1e-12, 1.0))))
        hashes = {
            "artifact_hash": self.artifact_hash_,
            "artifact_snapshot_hash": self.artifact_snapshot_hash_,
            "evidence_snapshot_hash": snapshot_hash(evidence_snapshot),
            "state_snapshot_hash": snapshot_hash(state_snapshot),
        }
        # Short aliases keep the contract readable for consumers that bind an
        # output to one evidence/state revision without carrying terminology
        # from this implementation into the wider trace schema.
        hashes["evidence_hash"] = hashes["evidence_snapshot_hash"]
        hashes["state_hash"] = hashes["state_snapshot_hash"]
        packet = ExplanationPacket(
            schema_version=EXPLANATION_PACKET_VERSION,
            module_version=self.module_version,
            class_order=self.labels,
            predicted_label=self.labels[int(np.argmax(probability_for_decision))],
            raw_probabilities=_probability_dict(raw_probability, self.labels),
            calibrated_probabilities=(
                None if calibrated_probability is None else _probability_dict(calibrated_probability, self.labels)
            ),
            calibration_status=calibration_status,
            logits=_probability_dict(logits, self.labels),
            intercepts=_probability_dict(intercepts, self.labels),
            feature_contributions=per_class_features,
            branch_contributions=per_class_branches,
            consumed_evidence_ids=tuple(sorted({str(value) for value in consumed_evidence_ids})),
            uncertainty={"entropy": entropy, "margin": float(ordered_probability[0] - ordered_probability[1])},
            fold_disagreement=(None if fold_disagreement is None else {str(k): float(v) for k, v in fold_disagreement.items()}),
            ood=self._ood(raw_design, unseen, ood_components),
            applicability_status=str(applicability_status),
            hashes=hashes,
        )
        return packet

    def predict_packet(self, case: pd.DataFrame | Mapping[str, Any], **kwargs: Any) -> ExplanationPacket:
        """Alias retained for consumers that name a Module A output a packet."""

        return self.explain_case(case, **kwargs)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Return raw probabilities in the fixed configured class order."""

        design, _ = self._transform_design(frame)
        standardized = (np.where(np.isfinite(design), design, self.impute_values_) - self.scale_means_) / self.scale_scales_
        rows = [self._class_linear_terms(row)[0] for row in standardized]
        return np.vstack([_softmax(row) for row in rows])


# The concise name is convenient for callers while retaining the descriptive
# class name above as the primary public contract.
ModuleAExpert = TaskConditionedStatisticalExpert
