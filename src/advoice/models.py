from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from .states import build_fold_calibrated_state_frame
from .utils import json_dump


IDENTITY_COLUMNS = {
    "dataset_id",
    "subject_id",
    "label",
    "split",
    "sex",
    "recording_count",
    "total_recorded_duration_sec",
}
ACOUSTIC_TOKENS = (
    "duration_sec",
    "silence_fraction",
    "voiced_fraction",
    "long_pause_rate_min",
    "pause_",
    "speech_run_",
    "rms_db_",
    "f0_",
    "zcr_",
    "spectral_",
    "mfcc_",
)


def _labels(config: dict[str, Any]) -> list[str]:
    labels = [str(label) for label in config.get("labels", [])]
    if len(labels) < 2:
        raise ValueError("Each task must define at least two ordered labels in models_config['labels'].")
    return labels


def _pipeline(c: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=20260813,
                ),
            ),
        ]
    )


class QCOrthogonalizer(BaseEstimator, TransformerMixin):
    """Remove train-fold QC-predictable components from clinical state scores."""

    def __init__(
        self,
        state_columns: tuple[str, ...],
        qc_columns: tuple[str, ...],
        alpha: float = 1.0,
    ) -> None:
        self.state_columns = state_columns
        self.qc_columns = qc_columns
        self.alpha = alpha

    def fit(self, x: pd.DataFrame, y: np.ndarray | None = None) -> QCOrthogonalizer:
        frame = pd.DataFrame(x, columns=list(self.state_columns) + list(self.qc_columns))
        self.state_imputer_ = SimpleImputer(strategy="median")
        state = self.state_imputer_.fit_transform(frame[list(self.state_columns)])
        if self.qc_columns:
            self.qc_imputer_ = SimpleImputer(strategy="median")
            qc = self.qc_imputer_.fit_transform(frame[list(self.qc_columns)])
            self.residualizer_ = Ridge(alpha=float(self.alpha)).fit(qc, state)
        else:
            self.qc_imputer_ = None
            self.residualizer_ = None
        return self

    def transform(self, x: pd.DataFrame) -> np.ndarray:
        frame = pd.DataFrame(x, columns=list(self.state_columns) + list(self.qc_columns))
        state = self.state_imputer_.transform(frame[list(self.state_columns)])
        if self.residualizer_ is None or self.qc_imputer_ is None:
            return state
        qc = self.qc_imputer_.transform(frame[list(self.qc_columns)])
        predicted = np.asarray(self.residualizer_.predict(qc), dtype=float)
        # sklearn returns a one-dimensional vector for a single state target.
        # Keep the prediction two-dimensional so subtraction cannot broadcast
        # an (n, 1) state matrix against an (n,) prediction into (n, n).
        if predicted.ndim == 1:
            predicted = predicted.reshape(-1, 1)
        return state - predicted


def _state_pipeline(
    c: float,
    max_iter: int,
    state_columns: list[str],
    qc_columns: list[str],
    alpha: float,
) -> Pipeline:
    return Pipeline(
        [
            (
                "qc_orthogonalizer",
                QCOrthogonalizer(tuple(state_columns), tuple(qc_columns), alpha),
            ),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=c,
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=20260813,
                ),
            ),
        ]
    )


def _ordered_probability(raw: np.ndarray, classes: np.ndarray, labels: list[str]) -> np.ndarray:
    order = {str(label): index for index, label in enumerate(classes)}
    missing = [label for label in labels if label not in order]
    if missing:
        raise ValueError(f"Model did not learn configured labels: {missing}")
    return np.column_stack([raw[:, order[label]] for label in labels])


def _macro_auc(y: np.ndarray, probability: np.ndarray, labels: list[str]) -> float:
    if len(labels) == 2:
        return float(roc_auc_score((y == labels[1]).astype(int), probability[:, 1]))
    return float(
        roc_auc_score(
            label_binarize(y, classes=labels),
            probability,
            average="macro",
            multi_class="ovr",
        )
    )


def _ordered_log_loss(y: np.ndarray, probability: np.ndarray, labels: list[str]) -> float:
    encoded = np.asarray([labels.index(str(value)) for value in y], dtype=int)
    return float(log_loss(encoded, probability, labels=list(range(len(labels)))))


def _choose_c(
    x: pd.DataFrame,
    y: np.ndarray,
    config: dict[str, Any],
    labels: list[str],
) -> tuple[float, dict[str, float]]:
    folds = min(int(config["cross_validation"]["folds"]), int(pd.Series(y).value_counts().min()))
    if folds < 2:
        raise ValueError("At least two training subjects per class are required.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260813)
    scores: dict[str, float] = {}
    for c in config["b1"]["c_grid"]:
        model = _pipeline(float(c), int(config["b1"]["max_iter"]))
        raw = cross_val_predict(model, x, y, cv=splitter, method="predict_proba")
        model.fit(x, y)
        probability = _ordered_probability(raw, model.named_steps["classifier"].classes_, labels)
        scores[str(c)] = _macro_auc(y, probability, labels)
    best = max(scores, key=scores.get)
    return float(best), scores


def _prediction_frame(
    test: pd.DataFrame,
    probability: np.ndarray,
    condition: str,
    labels: list[str],
) -> pd.DataFrame:
    output = test[["dataset_id", "subject_id", "label", "split"]].copy()
    for index, label in enumerate(labels):
        output[f"prob_{label}"] = probability[:, index]
    output["predicted_label"] = np.asarray(labels)[np.argmax(probability, axis=1)]
    output["condition"] = condition
    return output


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in IDENTITY_COLUMNS
    ]


def _is_acoustic_feature(column: str) -> bool:
    base = column.split("__", 1)[-1] if column.startswith("task_") else column
    return any(base == token or base.startswith(token) for token in ACOUSTIC_TOKENS)


def _acoustic_feature_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in _feature_columns(frame) if _is_acoustic_feature(column)]


def train_b1(
    subject_features_path: Path,
    models_config: dict[str, Any],
    predictions_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> None:
    labels = _labels(models_config)
    frame = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    train, test = frame[frame["split"].eq("train")].copy(), frame[frame["split"].eq("test")].copy()
    columns = _acoustic_feature_columns(frame)
    best_c, cv_scores = _choose_c(train[columns], train["label"].to_numpy(), models_config, labels)
    model = _pipeline(best_c, int(models_config["b1"]["max_iter"]))
    model.fit(train[columns], train["label"])
    probability = _ordered_probability(model.predict_proba(test[columns]), model.classes_, labels)
    _prediction_frame(test, probability, "B1", labels).to_csv(predictions_path, index=False)
    joblib.dump({"model": model, "features": columns, "labels": labels}, model_path)
    json_dump(
        {
            "condition": "B1",
            "definition": "hand-crafted acoustic features plus regularized logistic regression",
            "selected_c": best_c,
            "cv_macro_auroc_by_c": cv_scores,
            "feature_count": len(columns),
            "features": columns,
            "labels": labels,
            "train_subjects": len(train),
            "test_subjects": len(test),
        },
        metadata_path,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _apply_caps(weights: np.ndarray, caps: np.ndarray) -> np.ndarray:
    if float(np.sum(caps)) < 1.0 - 1e-9:
        raise ValueError("Branch caps must permit weights that sum to one.")
    output = np.zeros_like(weights)
    for row_index, raw in enumerate(weights):
        remaining = 1.0
        available = list(range(len(raw)))
        while available:
            denominator = float(np.sum(raw[available]))
            proposed = {
                index: remaining * (float(raw[index]) / denominator if denominator > 0 else 1.0 / len(available))
                for index in available
            }
            saturated = [index for index in available if proposed[index] > caps[index] + 1e-12]
            if not saturated:
                for index in available:
                    output[row_index, index] = proposed[index]
                break
            for index in saturated:
                output[row_index, index] = caps[index]
                remaining -= caps[index]
                available.remove(index)
    return output


def _gate_weights(reliability: np.ndarray, params: np.ndarray, caps: np.ndarray) -> np.ndarray:
    branch_count = reliability.shape[1]
    intercept = params[:branch_count]
    beta = np.log1p(np.exp(params[branch_count:]))
    logits = intercept[None, :] + beta[None, :] * np.log(np.clip(reliability, 1e-4, 1.0))
    return _apply_caps(_softmax(logits), caps)


def _learn_gate(
    branch_probability: np.ndarray,
    reliability: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    caps: np.ndarray,
    l2: float,
    branch_quality: np.ndarray,
    quality_prior_strength: float,
) -> np.ndarray:
    branch_count = reliability.shape[1]
    quality_prior = quality_prior_strength * (branch_quality - float(np.mean(branch_quality)))
    initial = np.concatenate([quality_prior, np.full(branch_count, -0.5)])

    def objective(params: np.ndarray) -> float:
        weights = _gate_weights(reliability, params, caps)
        fused = np.sum(branch_probability * weights[:, :, None], axis=1)
        return float(
            _ordered_log_loss(y, fused, labels)
            + l2 * np.sum((params - initial) ** 2)
        )

    result = minimize(objective, initial, method="L-BFGS-B")
    return result.x if result.success else initial


def _fit_branch(
    frame: pd.DataFrame,
    y: np.ndarray,
    c_grid: list[float],
    max_iter: int,
    splitter: StratifiedKFold,
    labels: list[str],
    specification: dict[str, Any],
    qc_alpha: float,
    fold_frames: list[pd.DataFrame] | None = None,
) -> tuple[Pipeline, np.ndarray, float, dict[str, float]]:
    candidates: dict[str, tuple[float, np.ndarray]] = {}
    for c in c_grid:
        ordered = np.zeros((len(frame), len(labels)), dtype=float)
        for fold_index, (train_index, validation_index) in enumerate(splitter.split(frame, y)):
            fold_frame = fold_frames[fold_index] if fold_frames is not None else frame
            candidate = (
                _state_pipeline(
                    float(c),
                    max_iter,
                    specification.get("state_features", []),
                    specification.get("qc_features", []),
                    qc_alpha,
                )
                if specification.get("qc_orthogonalized")
                else _pipeline(float(c), max_iter)
            )
            candidate.fit(
                fold_frame.iloc[train_index][specification["features"]],
                y[train_index],
            )
            raw = candidate.predict_proba(
                fold_frame.iloc[validation_index][specification["features"]]
            )
            ordered[validation_index] = _ordered_probability(
                raw, candidate.classes_, labels
            )
        candidates[str(c)] = (_macro_auc(y, ordered, labels), ordered)
    selected_key = max(candidates, key=lambda key: candidates[key][0])
    selected_c = float(selected_key)
    model = (
        _state_pipeline(
            selected_c,
            max_iter,
            specification.get("state_features", []),
            specification.get("qc_features", []),
            qc_alpha,
        )
        if specification.get("qc_orthogonalized")
        else _pipeline(selected_c, max_iter)
    )
    model.fit(frame[specification["features"]], y)
    return model, candidates[selected_key][1], selected_c, {
        key: float(value[0]) for key, value in candidates.items()
    }


def _predict_ordered(model: Pipeline, x: pd.DataFrame, labels: list[str]) -> np.ndarray:
    return _ordered_probability(model.predict_proba(x), model.classes_, labels)


def _branch_specifications(
    frame: pd.DataFrame,
    feature_frame: pd.DataFrame,
    models_config: dict[str, Any],
) -> list[dict[str, Any]]:
    def unique(columns: list[str]) -> list[str]:
        """Preserve feature order while preventing duplicate evidence votes."""
        return list(dict.fromkeys(columns))

    state_branches = models_config.get("state_branches", {})
    qc_config = models_config.get("ours", {}).get("qc_orthogonalization", {})
    qc_columns = unique([
        column for column in qc_config.get("features", []) if column in frame.columns
    ])
    qc_enabled = bool(qc_config.get("enabled", False) and qc_columns)
    grouped: dict[str, list[str]] = {}
    for state_id, branch in state_branches.items():
        prefix = f"state_{state_id}"
        columns = [
            column
            for column in frame.columns
            if column == prefix or column.startswith(f"{prefix}__task_")
        ]
        if columns and branch not in {"auxiliary_acoustic", "task_performance"}:
            grouped.setdefault(branch, []).extend(columns)
    grouped = {branch: unique(columns) for branch, columns in grouped.items()}
    specifications = [
        {
            "name": branch,
            "features": unique(columns + qc_columns) if qc_enabled else columns,
            "state_features": columns,
            "qc_features": qc_columns if qc_enabled else [],
            "qc_orthogonalized": qc_enabled,
            "reliability": [column.replace("state_", "rel_") for column in columns],
            "kind": "clinical_state",
        }
        for branch, columns in grouped.items()
        if columns
    ]
    auxiliary = unique([
        column
        for column in _acoustic_feature_columns(feature_frame)
        if column.startswith("mfcc_")
        or column.startswith("task_") and "__mfcc_" in column
        or column.split("__", 1)[-1]
        in {
            "rms_db_mean",
            "rms_db_std",
            "f0_median_hz",
            "f0_iqr_hz",
            "zcr_mean",
            "spectral_centroid_mean",
            "spectral_bandwidth_mean",
            "spectral_rolloff_mean",
            "spectral_flatness_mean",
        }
    ])
    # A nuisance variable cannot simultaneously be a disease feature and the
    # variable used to residualize that feature in the same branch.
    auxiliary = [column for column in auxiliary if column not in qc_columns]
    if auxiliary:
        specifications.append(
            {
                "name": "auxiliary_acoustic",
                "features": unique(auxiliary + qc_columns) if qc_enabled else auxiliary,
                "state_features": auxiliary,
                "qc_features": qc_columns if qc_enabled else [],
                "qc_orthogonalized": qc_enabled,
                "reliability": ["audio_reliability"],
                "kind": "low_interpretability_auxiliary",
            }
        )
    if not specifications:
        raise ValueError("No valid clinical or auxiliary branch could be constructed.")
    return specifications


def _overall_state_specification(specification: dict[str, Any]) -> dict[str, Any] | None:
    overall_states = [
        column
        for column in specification.get("state_features", [])
        if "__task_" not in column
    ]
    if not overall_states or len(overall_states) == len(specification.get("state_features", [])):
        return None
    return {
        **specification,
        "state_features": overall_states,
        "reliability": [column.replace("state_", "rel_", 1) for column in overall_states],
        "features": overall_states + specification.get("qc_features", []),
    }


def _paired_gain_summary(gains: np.ndarray) -> dict[str, float]:
    values = np.asarray(gains, dtype=float)
    if values.size == 0:
        raise ValueError("At least one paired fold gain is required.")
    mean = float(np.mean(values))
    standard_error = (
        float(np.std(values, ddof=1) / np.sqrt(values.size))
        if values.size > 1
        else float("inf")
    )
    return {
        "mean": mean,
        "standard_error": standard_error,
        "lower_95": mean - 1.96 * standard_error,
    }


def train_ours(
    subject_features_path: Path,
    state_wide_path: Path,
    metric_evidence_path: Path,
    states_config: dict[str, Any],
    models_config: dict[str, Any],
    predictions_path: Path,
    ablations_path: Path,
    contributions_path: Path,
    interventions_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> None:
    labels = _labels(models_config)
    positive_class = str(models_config.get("positive_class", labels[-1]))
    features = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    evidence = pd.read_csv(metric_evidence_path, dtype={"subject_id": str})
    stored_states = pd.read_csv(state_wide_path, dtype={"subject_id": str})
    identity = ["dataset_id", "subject_id", "label", "split"]
    train_subject_ids = set(
        features.loc[features["split"].eq("train"), "subject_id"].astype(str)
    )
    states = build_fold_calibrated_state_frame(
        evidence,
        states_config,
        train_subject_ids,
        labels[0],
    )
    expected_subjects = set(stored_states["subject_id"].astype(str))
    if set(states["subject_id"].astype(str)) != expected_subjects:
        raise ValueError("Fold-calibrated state identities do not match persisted StateCards.")
    model_order = features[identity].copy()
    model_order["_model_row_order"] = np.arange(len(model_order))
    frame = (
        features.merge(states, on=identity, how="inner", validate="one_to_one")
        .merge(model_order, on=identity, how="inner", validate="one_to_one")
        .sort_values("_model_row_order")
        .drop(columns="_model_row_order")
        .reset_index(drop=True)
    )
    train = frame[frame["split"].eq("train")].copy().reset_index(drop=True)
    test = frame[frame["split"].eq("test")].copy().reset_index(drop=True)
    y = train["label"].to_numpy()
    folds = min(int(models_config["cross_validation"]["folds"]), int(pd.Series(y).value_counts().min()))
    if folds < 2:
        raise ValueError("At least two training subjects per class are required.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260813)
    split_indices = list(splitter.split(train, y))
    train_evidence = evidence[evidence["subject_id"].astype(str).isin(set(train["subject_id"].astype(str)))]
    fold_frames: list[pd.DataFrame] = []
    for train_index, _ in split_indices:
        reference_subject_ids = set(train.iloc[train_index]["subject_id"].astype(str))
        fold_states = build_fold_calibrated_state_frame(
            train_evidence,
            states_config,
            reference_subject_ids,
            labels[0],
        )
        ordered_features = features[features["split"].eq("train")].merge(
            model_order,
            on=identity,
            how="inner",
            validate="one_to_one",
        )
        fold_frame = ordered_features.merge(
            fold_states,
            on=identity,
            how="inner",
            validate="one_to_one",
        ).sort_values("_model_row_order")
        fold_frames.append(fold_frame.drop(columns="_model_row_order").reset_index(drop=True))
    specifications = _branch_specifications(frame, features, models_config)
    branch_names = [item["name"] for item in specifications]
    max_iter = int(models_config["ours"]["max_iter"])
    qc_alpha = float(models_config["ours"].get("qc_orthogonalization", {}).get("alpha", 1.0))

    models: dict[str, Pipeline] = {}
    branch_training: dict[str, dict[str, Any]] = {}
    branch_quality: list[float] = []
    oof_probabilities: list[np.ndarray] = []
    test_probabilities: list[np.ndarray] = []
    train_reliability: list[np.ndarray] = []
    test_reliability: list[np.ndarray] = []
    overall_branch_candidates: dict[str, dict[str, Any]] = {}
    task_min_cv_gain = float(models_config["ours"].get("task_specific_min_cv_gain", 0.0))
    for specification_index, specification in enumerate(specifications):
        model, oof, selected_c, cv_scores = _fit_branch(
            train,
            y,
            [float(value) for value in models_config["ours"].get("c_grid", [1.0])],
            max_iter,
            splitter,
            labels,
            specification,
            qc_alpha,
            fold_frames if specification["kind"] == "clinical_state" else None,
        )
        task_selection = {
            "task_specific_candidates": 0,
            "selected": "not_applicable",
            "full_task_cv_macro_auroc": None,
            "overall_only_cv_macro_auroc": None,
            "minimum_cv_gain": task_min_cv_gain,
        }
        overall_specification = (
            _overall_state_specification(specification)
            if specification["kind"] == "clinical_state"
            else None
        )
        if overall_specification is not None:
            overall_model, overall_oof, overall_c, overall_scores = _fit_branch(
                train,
                y,
                [float(value) for value in models_config["ours"].get("c_grid", [1.0])],
                max_iter,
                splitter,
                labels,
                overall_specification,
                qc_alpha,
                fold_frames,
            )
            full_score = float(cv_scores[str(selected_c)])
            overall_score = float(overall_scores[str(overall_c)])
            paired_fold_gains = np.asarray(
                [
                    _macro_auc(y[validation_index], oof[validation_index], labels)
                    - _macro_auc(
                        y[validation_index],
                        overall_oof[validation_index],
                        labels,
                    )
                    for _, validation_index in split_indices
                ],
                dtype=float,
            )
            paired_gain = _paired_gain_summary(paired_fold_gains)
            paired_gain_mean = paired_gain["mean"]
            paired_gain_lower_95 = paired_gain["lower_95"]
            overall_branch_candidates[specification["name"]] = {
                "probability": _predict_ordered(
                    overall_model,
                    test[overall_specification["features"]],
                    labels,
                ),
                "quality": overall_score,
                "reliability": test[overall_specification["reliability"]]
                .mean(axis=1)
                .fillna(0.0)
                .to_numpy(),
            }
            task_selection = {
                "task_specific_candidates": len(specification["state_features"])
                - len(overall_specification["state_features"]),
                "selected": "overall_plus_task"
                if paired_gain_lower_95 >= task_min_cv_gain
                else "overall_only",
                "full_task_cv_macro_auroc": full_score,
                "overall_only_cv_macro_auroc": overall_score,
                "paired_fold_macro_auroc_gains": paired_fold_gains.tolist(),
                "paired_fold_gain_mean": paired_gain_mean,
                "paired_fold_gain_lower_95": paired_gain_lower_95,
                "minimum_cv_gain": task_min_cv_gain,
            }
            if task_selection["selected"] == "overall_only":
                specification = overall_specification
                specifications[specification_index] = specification
                model, oof, selected_c, cv_scores = (
                    overall_model,
                    overall_oof,
                    overall_c,
                    overall_scores,
                )
        models[specification["name"]] = model
        classifier = model.named_steps["classifier"]
        coefficient_rows = np.asarray(classifier.coef_, dtype=float)
        coefficient_classes = [str(value) for value in classifier.classes_]
        if coefficient_rows.shape[0] == 1 and len(coefficient_classes) == 2:
            coefficient_classes = [coefficient_classes[1]]
        feature_coefficients = (
            {
                class_name: {
                    feature: float(value)
                    for feature, value in zip(
                        specification.get("state_features", []),
                        coefficient_rows[index],
                        strict=True,
                    )
                }
                for index, class_name in enumerate(coefficient_classes)
            }
            if specification["kind"] == "clinical_state"
            else {}
        )
        branch_training[specification["name"]] = {
            "selected_c": selected_c,
            "cv_macro_auroc_by_c": cv_scores,
            "standardized_feature_coefficients": feature_coefficients,
            "task_state_selection": task_selection,
        }
        branch_quality.append(float(cv_scores[str(selected_c)]))
        oof_probabilities.append(oof)
        test_probabilities.append(_predict_ordered(model, test[specification["features"]], labels))
        train_reliability.append(
            train[specification["reliability"]].mean(axis=1).fillna(0.0).to_numpy()
        )
        test_reliability.append(
            test[specification["reliability"]].mean(axis=1).fillna(0.0).to_numpy()
        )

    branch_oof = np.stack(oof_probabilities, axis=1)
    branch_test = np.stack(test_probabilities, axis=1)
    train_reliability_array = np.column_stack(train_reliability)
    test_reliability_array = np.column_stack(test_reliability)
    auxiliary_cap = float(models_config["ours"]["auxiliary_weight_cap"])
    min_branch_auroc = float(models_config["ours"].get("min_branch_cv_auroc", 0.52))
    base_caps = np.array([auxiliary_cap if name == "auxiliary_acoustic" else 1.0 for name in branch_names])
    branch_quality_array = np.nan_to_num(np.asarray(branch_quality, dtype=float), nan=0.5, posinf=0.5, neginf=0.5)
    caps = np.where(branch_quality_array >= min_branch_auroc, base_caps, 0.0)
    if float(np.sum(caps)) < 1.0:
        caps[int(np.nanargmax(branch_quality_array))] = 1.0
    gate_params = _learn_gate(
        branch_oof,
        train_reliability_array,
        y,
        labels,
        caps,
        float(models_config["ours"]["branch_gate_l2"]),
        branch_quality_array,
        float(models_config["ours"].get("quality_prior_strength", 4.0)),
    )
    train_weights = _gate_weights(train_reliability_array, gate_params, caps)
    test_weights = _gate_weights(test_reliability_array, gate_params, caps)
    fused_oof = np.sum(branch_oof * train_weights[:, :, None], axis=1)
    calibrator = LogisticRegression(max_iter=2000, random_state=20260813)
    calibrator.fit(np.log(np.clip(fused_oof, 1e-6, 1.0)), y)

    fused_test = np.sum(branch_test * test_weights[:, :, None], axis=1)
    calibrated = _ordered_probability(
        calibrator.predict_proba(np.log(np.clip(fused_test, 1e-6, 1.0))),
        calibrator.classes_,
        labels,
    )
    predictions = _prediction_frame(test, calibrated, "Ours", labels)
    for branch_index, branch_name in enumerate(branch_names):
        predictions[f"weight_{branch_name}"] = test_weights[:, branch_index]
    predictions["prediction_confidence"] = calibrated.max(axis=1)
    predictions.to_csv(predictions_path, index=False)

    ablation_outputs = [
        _prediction_frame(test, probability, f"Ours_{name}_only", labels)
        for name, probability in zip(branch_names, test_probabilities, strict=True)
    ]
    clinical_indices = [index for index, item in enumerate(specifications) if item["kind"] == "clinical_state"]
    if clinical_indices:
        clinical_quality = branch_quality_array[clinical_indices]
        clinical_prior = np.exp(
            float(models_config["ours"].get("quality_prior_strength", 4.0))
            * (clinical_quality - float(np.mean(clinical_quality)))
        )
        clinical_weights = test_reliability_array[:, clinical_indices] * clinical_prior[None, :]
        denominator = clinical_weights.sum(axis=1, keepdims=True)
        clinical_weights = np.divide(
            clinical_weights,
            denominator,
            out=np.full_like(clinical_weights, 1.0 / len(clinical_indices)),
            where=denominator > 0,
        )
        clinical_probability = np.sum(
            branch_test[:, clinical_indices, :] * clinical_weights[:, :, None], axis=1
        )
        ablation_outputs.append(_prediction_frame(test, clinical_probability, "Ours_concept_only", labels))

        if overall_branch_candidates:
            overall_candidates = [
                overall_branch_candidates[specifications[index]["name"]]
                for index in clinical_indices
                if specifications[index]["name"] in overall_branch_candidates
            ]
            if overall_candidates:
                overall_quality = np.asarray(
                    [candidate["quality"] for candidate in overall_candidates],
                    dtype=float,
                )
                overall_prior = np.exp(
                    float(models_config["ours"].get("quality_prior_strength", 4.0))
                    * (overall_quality - float(np.mean(overall_quality)))
                )
                overall_weights = np.column_stack(
                    [candidate["reliability"] for candidate in overall_candidates]
                ) * overall_prior[None, :]
                overall_denominator = overall_weights.sum(axis=1, keepdims=True)
                overall_weights = np.divide(
                    overall_weights,
                    overall_denominator,
                    out=np.full_like(overall_weights, 1.0 / len(overall_candidates)),
                    where=overall_denominator > 0,
                )
                overall_probability = np.sum(
                    np.stack(
                        [candidate["probability"] for candidate in overall_candidates],
                        axis=1,
                    )
                    * overall_weights[:, :, None],
                    axis=1,
                )
                ablation_outputs.append(
                    _prediction_frame(
                        test,
                        overall_probability,
                        "Ours_overall_state_only",
                        labels,
                    )
                )
    pd.concat(ablation_outputs, ignore_index=True).to_csv(ablations_path, index=False)

    state_columns = sorted(column for column in frame.columns if column.startswith("state_"))
    contributions = test[["dataset_id", "subject_id", "label"] + state_columns].copy()
    positive_index = labels.index(positive_class)
    for branch_index, branch_name in enumerate(branch_names):
        contributions[f"weight_{branch_name}"] = test_weights[:, branch_index]
        contributions[f"{branch_name}_{positive_class}_contribution"] = (
            test_weights[:, branch_index] * branch_test[:, branch_index, positive_index]
        )
    contributions.to_csv(contributions_path, index=False)

    class_state_reference = train.groupby("label")[state_columns].mean()
    state_to_branch = {
        column: branch_index
        for branch_index, specification in enumerate(specifications)
        if specification["kind"] == "clinical_state"
        for column in specification["features"]
        if column in state_columns
    }
    intervention_rows: list[dict[str, Any]] = []
    for row_index, (_, subject) in enumerate(test.iterrows()):
        predicted_label = labels[int(np.argmax(calibrated[row_index]))]
        if predicted_label == str(subject["label"]):
            continue
        eligible = list(state_to_branch)
        if not eligible:
            break
        true_reference = class_state_reference.loc[subject["label"]]
        finite_deviations = {
            column: abs(float(subject[column] - true_reference[column]))
            for column in eligible
            if pd.notna(subject[column]) and pd.notna(true_reference[column])
        }
        if not finite_deviations:
            continue
        target_state = max(finite_deviations, key=finite_deviations.get)
        branch_index = state_to_branch[target_state]
        specification = specifications[branch_index]
        changed = pd.DataFrame(
            [
                {
                    column: pd.to_numeric(subject[column], errors="coerce")
                    for column in specification["features"]
                }
            ],
            columns=specification["features"],
            dtype=float,
        )
        changed[target_state] = float(true_reference[target_state])
        changed_probability = _predict_ordered(models[specification["name"]], changed, labels)[0]
        changed_branch = branch_test[row_index].copy()
        changed_branch[branch_index] = changed_probability
        changed_fused = np.sum(changed_branch * test_weights[row_index, :, None], axis=0)
        changed_calibrated = _ordered_probability(
            calibrator.predict_proba(np.log(np.clip(changed_fused, 1e-6, 1.0)).reshape(1, -1)),
            calibrator.classes_,
            labels,
        )[0]
        true_index = labels.index(str(subject["label"]))
        before = float(calibrated[row_index, true_index])
        after = float(changed_calibrated[true_index])
        intervention_rows.append(
            {
                "subject_id": subject["subject_id"],
                "label": subject["label"],
                "prediction_before": predicted_label,
                "intervened_state": target_state,
                "branch": specification["name"],
                "original_state_z": float(subject[target_state]),
                "corrected_state_z": float(true_reference[target_state]),
                "correction_source": "training-set true-class state mean",
                "true_class_probability_before": before,
                "true_class_probability_after": after,
                "true_class_probability_change": after - before,
                "monotonic_improvement": bool(after >= before),
            }
        )
    intervention_columns = [
        "subject_id",
        "label",
        "prediction_before",
        "intervened_state",
        "branch",
        "original_state_z",
        "corrected_state_z",
        "correction_source",
        "true_class_probability_before",
        "true_class_probability_after",
        "true_class_probability_change",
        "monotonic_improvement",
    ]
    pd.DataFrame(intervention_rows, columns=intervention_columns).to_csv(interventions_path, index=False)

    branch_count = len(branch_names)
    beta = np.log1p(np.exp(gate_params[branch_count:]))
    joblib.dump(
        {
            "branch_models": models,
            "branch_specifications": specifications,
            "calibrator": calibrator,
            "gate_params": gate_params,
            "labels": labels,
        },
        model_path,
    )
    json_dump(
        {
            "condition": "Ours",
            "definition": "evidence-governed state fusion with reliability-conditioned branch gating",
            "prediction_engine": "deterministic supervised machine learning; no generative agent changes labels or probabilities",
            "report_agent_role": "post-prediction translation of frozen evidence and probability",
            "state_internal_fusion": "clinically reviewed metric weights multiplied by case-specific evidence reliability",
            "task_specific_state_policy": {
                "enabled_when": "more than one task is present in the dataset",
                "calibration": "each task-specific metric uses its own reference median and robust scale, recomputed inside every model-selection fold; held-out test data use the full training reference only",
                "model_selection": "for every clinical branch, overall-only and overall-plus-task candidates are compared on the same training folds; task-specific states are retained only when the paired fold-gain 95% lower confidence bound meets the configured minimum",
                "minimum_paired_fold_gain_lower_95": task_min_cv_gain,
                "anti_double_counting": "regularization controls correlated overall/task state inputs; the clinical report renderer must not describe them as independent mechanisms",
            },
            "qc_orthogonalization": {
                "enabled": bool(models_config["ours"].get("qc_orthogonalization", {}).get("enabled", False)),
                "method": "fold-internal ridge residualization of clinical states against QC variables",
                "features": models_config["ours"].get("qc_orthogonalization", {}).get("features", []),
                "alpha": qc_alpha,
                "note": "QC variables remove predictable nuisance components and never enter a disease classifier directly.",
            },
            "state_external_fusion": "task-available branch classifiers plus a learned reliability-conditioned softmax gate",
            "labels": labels,
            "positive_class": positive_class,
            "branches": specifications,
            "branch_training": branch_training,
            "branch_cv_quality": dict(zip(branch_names, branch_quality, strict=True)),
            "branch_effective_caps": dict(zip(branch_names, caps.tolist(), strict=True)),
            "branch_exclusion_rule": f"training-only OOF macro AUROC below {min_branch_auroc:.2f}",
            "gate_parameters": {
                name: {"intercept": float(gate_params[index]), "reliability_beta": float(beta[index])}
                for index, name in enumerate(branch_names)
            },
            "mean_train_branch_weights": {
                name: float(train_weights[:, index].mean()) for index, name in enumerate(branch_names)
            },
            "mean_test_branch_weights": {
                name: float(test_weights[:, index].mean()) for index, name in enumerate(branch_names)
            },
            "test_branch_weight_sd": {
                name: float(test_weights[:, index].std(ddof=0)) for index, name in enumerate(branch_names)
            },
            "auxiliary_weight_cap": auxiliary_cap,
            "calibration": "multinomial or binary logistic stacking fitted only on out-of-fold fused probabilities",
            "train_subjects": len(train),
            "test_subjects": len(test),
        },
        metadata_path,
    )


def train_negative_controls(
    subject_features_path: Path,
    predictions_path: Path,
    models_config: dict[str, Any],
) -> None:
    labels = _labels(models_config)
    frame = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    train, test = frame[frame["split"].eq("train")], frame[frame["split"].eq("test")]
    acoustic = _acoustic_feature_columns(frame)
    task_scopes = sorted(
        {
            column.split("__", 1)[0]
            for column in frame.columns
            if column.startswith("task_") and "__" in column
        }
    )
    task_presence: dict[str, pd.Series] = {}
    for scope in task_scopes:
        source_columns = [column for column in frame.columns if column.startswith(f"{scope}__")]
        column = f"presence__{scope}"
        task_presence[column] = frame[source_columns].notna().any(axis=1).astype(float)
    if task_presence:
        frame = pd.concat([frame, pd.DataFrame(task_presence, index=frame.index)], axis=1)
    task_presence_columns = list(task_presence)
    train, test = frame[frame["split"].eq("train")], frame[frame["split"].eq("test")]
    controls = {
        "QC_only": [
            column
            for column in ["duration_sec", "original_duration_sec", "clipping_fraction", "snr_proxy_db", "rms_db_mean"]
            if column in frame.columns
        ],
        "No_duration_no_loudness": [
            column
            for column in acoustic
            if column.split("__", 1)[-1]
            not in {"duration_sec", "original_duration_sec", "rms_db_mean", "rms_db_std"}
        ],
        "Task_presence_only": task_presence_columns,
    }
    outputs = []
    for name, columns in controls.items():
        if not columns:
            continue
        model = _pipeline(1.0, 4000).fit(train[columns], train["label"])
        probability = _predict_ordered(model, test[columns], labels)
        outputs.append(_prediction_frame(test, probability, name, labels))
    if acoustic:
        rng = np.random.default_rng(20260827)
        shuffled_labels = rng.permutation(train["label"].astype(str).to_numpy())
        permutation_model = _pipeline(1.0, 4000).fit(train[acoustic], shuffled_labels)
        permutation_probability = _predict_ordered(
            permutation_model, test[acoustic], labels
        )
        outputs.append(
            _prediction_frame(
                test,
                permutation_probability,
                "Label_permutation",
                labels,
            )
        )
    pd.concat(outputs, ignore_index=True).to_csv(predictions_path, index=False)
