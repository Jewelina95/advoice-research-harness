from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.base import clone
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, label_binarize

from .utils import json_dump


LABELS = ["HC", "MCI", "AD"]


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
                    multi_class="multinomial",
                    solver="lbfgs",
                    random_state=20260813,
                ),
            ),
        ]
    )


def _choose_c(x: pd.DataFrame, y: np.ndarray, config: dict[str, Any]) -> tuple[float, dict[str, float]]:
    folds = min(int(config["cross_validation"]["folds"]), int(pd.Series(y).value_counts().min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260813)
    scores: dict[str, float] = {}
    for c in config["b1"]["c_grid"]:
        probability = cross_val_predict(
            _pipeline(float(c), int(config["b1"]["max_iter"])),
            x,
            y,
            cv=splitter,
            method="predict_proba",
        )
        sorted_classes = sorted(LABELS)
        probability = np.column_stack([probability[:, sorted_classes.index(label)] for label in LABELS])
        score = roc_auc_score(
            label_binarize(y, classes=LABELS), probability, average="macro", multi_class="ovr"
        )
        scores[str(c)] = float(score)
    best = max(scores, key=scores.get)
    return float(best), scores


def _prediction_frame(test: pd.DataFrame, probability: np.ndarray, condition: str) -> pd.DataFrame:
    output = test[["dataset_id", "subject_id", "label", "split"]].copy()
    for index, label in enumerate(LABELS):
        output[f"prob_{label}"] = probability[:, index]
    output["predicted_label"] = np.asarray(LABELS)[np.argmax(probability, axis=1)]
    output["condition"] = condition
    return output


def _feature_columns(frame: pd.DataFrame) -> list[str]:
    excluded = {
        "dataset_id",
        "subject_id",
        "label",
        "split",
        "sex",
        "recording_count",
        "total_recorded_duration_sec",
    }
    return [
        column
        for column in frame.select_dtypes(include=[np.number]).columns
        if column not in excluded
    ]


def train_b1(
    subject_features_path: Path,
    models_config: dict[str, Any],
    predictions_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> None:
    frame = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    train, test = frame[frame["split"].eq("train")].copy(), frame[frame["split"].eq("test")].copy()
    columns = _feature_columns(frame)
    best_c, cv_scores = _choose_c(train[columns], train["label"].to_numpy(), models_config)
    model = _pipeline(best_c, int(models_config["b1"]["max_iter"]))
    model.fit(train[columns], train["label"])
    probability_raw = model.predict_proba(test[columns])
    class_index = {label: index for index, label in enumerate(model.classes_)}
    probability = np.column_stack([probability_raw[:, class_index[label]] for label in LABELS])
    _prediction_frame(test, probability, "B1").to_csv(predictions_path, index=False)
    joblib.dump({"model": model, "features": columns, "labels": LABELS}, model_path)
    json_dump(
        {
            "condition": "B1",
            "definition": "hand-crafted acoustic features + standardized multinomial logistic regression",
            "selected_c": best_c,
            "cv_macro_auroc_by_c": cv_scores,
            "feature_count": len(columns),
            "features": columns,
            "train_subjects": len(train),
            "test_subjects": len(test),
        },
        metadata_path,
    )


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def _gate_weights(reliability: np.ndarray, params: np.ndarray, auxiliary_cap: float) -> np.ndarray:
    intercept = params[:2]
    beta = max(float(params[2]), 0.0)
    logits = intercept[None, :] + beta * np.log(np.clip(reliability, 1e-4, 1.0))
    weights = _softmax(logits)
    weights[:, 1] = np.minimum(weights[:, 1], auxiliary_cap)
    weights[:, 0] = 1.0 - weights[:, 1]
    return weights


def _learn_gate(
    branch_probability: np.ndarray,
    reliability: np.ndarray,
    y: np.ndarray,
    auxiliary_cap: float,
    l2: float,
) -> np.ndarray:
    def objective(params: np.ndarray) -> float:
        weights = _gate_weights(reliability, params, auxiliary_cap)
        fused = np.sum(branch_probability * weights[:, :, None], axis=1)
        return float(log_loss(y, fused, labels=LABELS) + l2 * np.sum(params**2))

    result = minimize(objective, np.array([0.5, -0.5, 1.0]), method="L-BFGS-B")
    if not result.success:
        return np.array([0.5, -0.5, 1.0])
    return result.x


def _fit_branch(
    x: pd.DataFrame,
    y: np.ndarray,
    c: float,
    max_iter: int,
    splitter: StratifiedKFold,
) -> tuple[Pipeline, np.ndarray]:
    model = _pipeline(c, max_iter)
    oof_raw = cross_val_predict(model, x, y, cv=splitter, method="predict_proba")
    model.fit(x, y)
    order = {label: index for index, label in enumerate(model.classes_)}
    oof = np.column_stack([oof_raw[:, order[label]] for label in LABELS])
    return model, oof


def _predict_ordered(model: Pipeline, x: pd.DataFrame) -> np.ndarray:
    raw = model.predict_proba(x)
    order = {label: index for index, label in enumerate(model.classes_)}
    return np.column_stack([raw[:, order[label]] for label in LABELS])


def train_ours(
    subject_features_path: Path,
    state_wide_path: Path,
    models_config: dict[str, Any],
    predictions_path: Path,
    ablations_path: Path,
    contributions_path: Path,
    interventions_path: Path,
    model_path: Path,
    metadata_path: Path,
) -> None:
    features = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    states = pd.read_csv(state_wide_path, dtype={"subject_id": str})
    frame = features.merge(states, on=["dataset_id", "subject_id", "label", "split"], how="inner")
    train = frame[frame["split"].eq("train")].copy()
    test = frame[frame["split"].eq("test")].copy()
    y = train["label"].to_numpy()
    folds = min(int(models_config["cross_validation"]["folds"]), int(pd.Series(y).value_counts().min()))
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260813)
    behavior_cols = ["state_S01", "state_S02", "state_S03"]
    auxiliary_cols = [
        column
        for column in _feature_columns(features)
        if column.startswith("mfcc_")
        or column in {
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
    ]
    c = 1.0
    max_iter = int(models_config["ours"]["max_iter"])
    behavior_model, behavior_oof = _fit_branch(train[behavior_cols], y, c, max_iter, splitter)
    auxiliary_model, auxiliary_oof = _fit_branch(train[auxiliary_cols], y, c, max_iter, splitter)
    branch_oof = np.stack([behavior_oof, auxiliary_oof], axis=1)
    train_reliability = np.column_stack(
        [
            train[["rel_S01", "rel_S02", "rel_S03"]].mean(axis=1).to_numpy(),
            train["audio_reliability"].to_numpy(),
        ]
    )
    auxiliary_cap = float(models_config["ours"]["auxiliary_weight_cap"])
    gate_params = _learn_gate(
        branch_oof,
        train_reliability,
        y,
        auxiliary_cap,
        float(models_config["ours"]["branch_gate_l2"]),
    )
    train_weights = _gate_weights(train_reliability, gate_params, auxiliary_cap)
    fused_oof = np.sum(branch_oof * train_weights[:, :, None], axis=1)
    calibrator = LogisticRegression(max_iter=2000, multi_class="multinomial", random_state=20260813)
    calibrator.fit(np.log(np.clip(fused_oof, 1e-6, 1.0)), y)

    behavior_test = _predict_ordered(behavior_model, test[behavior_cols])
    auxiliary_test = _predict_ordered(auxiliary_model, test[auxiliary_cols])
    test_branch = np.stack([behavior_test, auxiliary_test], axis=1)
    test_reliability = np.column_stack(
        [
            test[["rel_S01", "rel_S02", "rel_S03"]].mean(axis=1).to_numpy(),
            test["audio_reliability"].to_numpy(),
        ]
    )
    test_weights = _gate_weights(test_reliability, gate_params, auxiliary_cap)
    fused_test = np.sum(test_branch * test_weights[:, :, None], axis=1)
    calibrated_raw = calibrator.predict_proba(np.log(np.clip(fused_test, 1e-6, 1.0)))
    cal_order = {label: index for index, label in enumerate(calibrator.classes_)}
    calibrated = np.column_stack([calibrated_raw[:, cal_order[label]] for label in LABELS])
    predictions = _prediction_frame(test, calibrated, "Ours")
    predictions["behavior_weight"] = test_weights[:, 0]
    predictions["auxiliary_weight"] = test_weights[:, 1]
    predictions["prediction_confidence"] = calibrated.max(axis=1)
    predictions.to_csv(predictions_path, index=False)

    behavior_output = _prediction_frame(test, behavior_test, "Ours_state_only")
    auxiliary_output = _prediction_frame(test, auxiliary_test, "Ours_auxiliary_only")
    pd.concat([behavior_output, auxiliary_output], ignore_index=True).to_csv(ablations_path, index=False)

    contributions = test[["dataset_id", "subject_id", "label"] + behavior_cols].copy()
    contributions["behavior_weight"] = test_weights[:, 0]
    contributions["auxiliary_weight"] = test_weights[:, 1]
    contributions["behavior_ad_contribution"] = test_weights[:, 0] * behavior_test[:, LABELS.index("AD")]
    contributions["auxiliary_ad_contribution"] = test_weights[:, 1] * auxiliary_test[:, LABELS.index("AD")]
    contributions.to_csv(contributions_path, index=False)

    class_state_reference = train.groupby("label")[behavior_cols].mean()
    intervention_rows: list[dict[str, Any]] = []
    for row_index, (_, subject) in enumerate(test.iterrows()):
        state_values = subject[behavior_cols].astype(float)
        true_reference = class_state_reference.loc[subject["label"]]
        target_state = (state_values - true_reference).abs().idxmax()
        changed = subject[behavior_cols].to_frame().T.copy()
        changed[target_state] = float(true_reference[target_state])
        changed_behavior = _predict_ordered(behavior_model, changed)[0]
        changed_branch = test_branch[row_index].copy()
        changed_branch[0] = changed_behavior
        changed_fused = np.sum(changed_branch * test_weights[row_index, :, None], axis=0)
        changed_raw = calibrator.predict_proba(np.log(np.clip(changed_fused, 1e-6, 1.0)).reshape(1, -1))[0]
        changed_ordered = np.array([changed_raw[cal_order[label]] for label in LABELS])
        true_index = LABELS.index(subject["label"])
        before = float(calibrated[row_index, true_index])
        after = float(changed_ordered[true_index])
        intervention_rows.append(
            {
                "subject_id": subject["subject_id"],
                "label": subject["label"],
                "intervened_state": target_state,
                "original_state_z": float(subject[target_state]),
                "corrected_state_z": float(true_reference[target_state]),
                "true_class_probability_before": before,
                "true_class_probability_after": after,
                "true_class_probability_change": after - before,
                "monotonic_improvement": bool(after >= before),
            }
        )
    pd.DataFrame(intervention_rows).to_csv(interventions_path, index=False)

    joblib.dump(
        {
            "behavior_model": behavior_model,
            "auxiliary_model": auxiliary_model,
            "calibrator": calibrator,
            "gate_params": gate_params,
            "behavior_features": behavior_cols,
            "auxiliary_features": auxiliary_cols,
            "labels": LABELS,
        },
        model_path,
    )
    json_dump(
        {
            "condition": "Ours",
            "definition": "state-guided reliability-aware hierarchical fusion",
            "state_internal_fusion": "fixed clinically reviewed metric weights multiplied by case reliability",
            "state_external_fusion": "two branch classifiers plus learned reliability-conditioned softmax gate",
            "gate_parameters": gate_params.tolist(),
            "mean_train_branch_weights": {
                "speech_behavior": float(train_weights[:, 0].mean()),
                "auxiliary_acoustic": float(train_weights[:, 1].mean()),
            },
            "mean_test_branch_weights": {
                "speech_behavior": float(test_weights[:, 0].mean()),
                "auxiliary_acoustic": float(test_weights[:, 1].mean()),
            },
            "auxiliary_weight_cap": auxiliary_cap,
            "behavior_features": behavior_cols,
            "auxiliary_feature_count": len(auxiliary_cols),
            "calibration": "multinomial logistic stacking fitted on out-of-fold fused probabilities",
            "train_subjects": len(train),
            "test_subjects": len(test),
        },
        metadata_path,
    )


def train_negative_controls(
    subject_features_path: Path,
    predictions_path: Path,
) -> None:
    frame = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    train, test = frame[frame["split"].eq("train")], frame[frame["split"].eq("test")]
    controls = {
        "QC_only": ["duration_sec", "clipping_fraction", "snr_proxy_db", "rms_db_mean"],
        "No_duration_no_loudness": [
            column
            for column in _feature_columns(frame)
            if column not in {"duration_sec", "rms_db_mean", "rms_db_std", "total_recorded_duration_sec"}
        ],
    }
    outputs = []
    for name, columns in controls.items():
        model = _pipeline(1.0, 4000).fit(train[columns], train["label"])
        probability = _predict_ordered(model, test[columns])
        outputs.append(_prediction_frame(test, probability, name))
    pd.concat(outputs, ignore_index=True).to_csv(predictions_path, index=False)
