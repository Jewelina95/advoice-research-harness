from __future__ import annotations

import json
import itertools
import math
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.impute import SimpleImputer

from .cognitive_prototypes import (
    build_case_prototype_reference,
    fit_cognitive_prototypes,
)
from .diagnostic_agent import build_case_workspace, evidence_gate, fuse_corrected_probability
from .deep_audio_embeddings import encode_multilingual_audio, load_audio_window_sequences
from .deep_embeddings import encode_multilingual_text
from .dynamic_gate import fit_dynamic_reliability_gate
from .evidence import recalibrate_metric_evidence_frame
from .models import (
    _acoustic_feature_columns,
    _branch_specifications,
    _fit_branch,
    _labels,
    _macro_auc,
    _ordered_probability,
    _predict_ordered,
    _prediction_frame,
)
from .states import build_fold_calibrated_state_frame
from .sequence_expert import fit_segment_attention_expert
from .utils import json_dump


IDENTITY = ["dataset_id", "subject_id", "label", "split"]
MORPH_TOKEN = re.compile(r"(?:[A-Za-z]+:)?[A-Za-z0-9_#-]+\|\S+")
DEPENDENCY_TOKEN = re.compile(r"\b\d+\|\d+\|[A-Z]+\b")
RECORDING_MARKER = re.compile(r"\[录音\d+\]")
CHAT_MARKUP = re.compile(r"\[[^\]]*\]|[<>+&=]\S*|\b(?:xxx|yyy|www)\b")
WHITESPACE = re.compile(r"\s+")


def _boolean_series(values: pd.Series, *, default: bool) -> pd.Series:
    """Parse CSV booleans without treating the string ``False`` as truthy."""

    normalized = values.map(
        lambda value: default
        if pd.isna(value)
        else str(value).strip().lower() in {"1", "true", "yes", "y"}
    )
    return normalized.astype(bool)


def _replace_card_metric_summaries(
    cards: pd.DataFrame,
    evidence: pd.DataFrame,
) -> pd.DataFrame:
    """Align card-level support/counterevidence with fold-local metric evidence."""

    result = cards.copy()
    payload_columns = [
        "metric_id",
        "metric_instance_id",
        "task_scope",
        "value",
        "reference_label",
        "reference_scope",
        "reference_median",
        "reference_scale",
        "cn_train_median",
        "cn_train_scale",
        "robust_z",
        "directional_z",
        "reliability",
        "report_permission",
        "confound_tags",
    ]
    for index, card in result.iterrows():
        subject_id = str(card["subject_id"])
        state_id = str(card.get("state_base_id", card["state_id"]))
        task_scope = str(card.get("task_scope", "overall"))
        rows = evidence[
            evidence["subject_id"].astype(str).eq(subject_id)
            & evidence["state_id"].astype(str).eq(state_id)
            & evidence["task_scope"].astype(str).eq(task_scope)
            & ~_boolean_series(evidence["missing"], default=True)
            & (pd.to_numeric(evidence["reliability"], errors="coerce") > 0)
        ].copy()
        rows["evidence_strength"] = (
            pd.to_numeric(rows["directional_z"], errors="coerce").abs()
            * pd.to_numeric(rows["reliability"], errors="coerce")
        )
        rows = rows.sort_values("evidence_strength", ascending=False)
        supporting = rows[pd.to_numeric(rows["directional_z"], errors="coerce") >= 0]
        counter = rows[pd.to_numeric(rows["directional_z"], errors="coerce") < 0]
        result.at[index, "supporting_metrics"] = json.dumps(
            supporting.head(3)[payload_columns].to_dict("records"),
            ensure_ascii=False,
        )
        result.at[index, "counter_evidence"] = json.dumps(
            counter.head(3)[payload_columns].to_dict("records"),
            ensure_ascii=False,
        )
    return result


def _prototype_inputs(
    frame: pd.DataFrame,
    state_columns: list[str],
) -> tuple[np.ndarray, np.ndarray]:
    values = frame[state_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
    reliability_columns = [column.replace("state_", "rel_", 1) for column in state_columns]
    reliability = np.column_stack(
        [
            pd.to_numeric(frame[column], errors="coerce").fillna(0.0).to_numpy(dtype=float)
            if column in frame
            else np.zeros(len(frame), dtype=float)
            for column in reliability_columns
        ]
    )
    return values, reliability


def clean_model_transcript(value: Any) -> str:
    """Remove CHAT/parser residue while preserving the participant's lexical content."""

    text = "" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value)
    text = RECORDING_MARKER.sub(" ", text)
    text = DEPENDENCY_TOKEN.sub(" ", text)
    text = MORPH_TOKEN.sub(" ", text)
    text = CHAT_MARKUP.sub(" ", text)
    return WHITESPACE.sub(" ", text).strip()


def _add_age_band_features(features: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    """Add outcome-independent age categories used only as model context."""
    if "age" not in features:
        return []
    features["age"] = pd.to_numeric(features["age"], errors="coerce")
    bins = [float(value) for value in config.get("age_bins", [0, 66, 81, 200])]
    if len(bins) < 3 or bins != sorted(set(bins)):
        raise ValueError("condition_c.demographic_context.age_bins must be increasing.")
    age = features["age"]
    columns: list[str] = []
    for index, (lower, upper) in enumerate(zip(bins[:-1], bins[1:], strict=True)):
        column = f"age_band_{index}"
        features[column] = ((age >= lower) & (age < upper)).astype(float)
        columns.append(column)
    features["age_band_missing"] = age.isna().astype(float)
    columns.append("age_band_missing")
    return columns


def _text_input(values: Any) -> np.ndarray:
    return np.asarray(values, dtype=object).reshape(-1)


def _text_pipeline(c: float, max_iter: int, min_df: int) -> Pipeline:
    features = FeatureUnion(
        [
            (
                "character",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    min_df=min_df,
                    max_features=30000,
                    sublinear_tf=True,
                    lowercase=True,
                ),
            ),
            (
                "word",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    min_df=min_df,
                    max_features=30000,
                    sublinear_tf=True,
                    lowercase=True,
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
        ]
    )
    return Pipeline(
        [
            ("reshape", FunctionTransformer(_text_input, validate=False)),
            ("features", features),
            (
                "classifier",
                LogisticRegression(
                    C=float(c),
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=20260821,
                ),
            ),
        ]
    )


def _numeric_pipeline(c: float, max_iter: int) -> Pipeline:
    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    C=float(c),
                    max_iter=max_iter,
                    class_weight="balanced",
                    solver="lbfgs",
                    random_state=20260821,
                ),
            ),
        ]
    )


def _ordered_from_pipeline(model: Pipeline, values: Any, labels: list[str]) -> np.ndarray:
    raw = model.predict_proba(values)
    return _ordered_probability(raw, model.named_steps["classifier"].classes_, labels)


def _selection_score(y: np.ndarray, probability: np.ndarray, labels: list[str]) -> tuple[float, float]:
    predicted = np.asarray(labels)[np.argmax(probability, axis=1)]
    macro_f1 = float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0))
    try:
        macro_auc = _macro_auc(y, probability, labels)
    except ValueError:
        macro_auc = 0.5
    return macro_f1, macro_auc


def _small_sample_correction_guard(
    y: np.ndarray,
    labels: list[str],
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    base_oof: np.ndarray,
    final_oof: np.ndarray,
    primary: str,
    config: dict[str, Any],
) -> dict[str, Any]:
    """Reject harmful corrections for every cohort and unstable corrections for small cohorts."""

    maximum_subjects = int(config.get("maximum_training_subjects", 80))
    minimum_fold_fraction = float(config.get("minimum_noninferior_fold_fraction", 0.6))
    minimum_strict_fraction = float(
        config.get("minimum_strict_improvement_fold_fraction", 0.4)
    )
    tolerance = float(config.get("noninferiority_tolerance", 1e-6))
    score_index = 1 if primary == "macro_auroc" else 0
    base_score = _selection_score(y, base_oof, labels)
    final_score = _selection_score(y, final_oof, labels)
    fold_rows: list[dict[str, float | bool | int]] = []
    for fold_index, (_, validation_index) in enumerate(split_indices):
        base_fold = _selection_score(y[validation_index], base_oof[validation_index], labels)
        final_fold = _selection_score(y[validation_index], final_oof[validation_index], labels)
        fold_rows.append(
            {
                "fold": fold_index,
                "base_macro_f1": base_fold[0],
                "final_macro_f1": final_fold[0],
                "base_macro_auroc": base_fold[1],
                "final_macro_auroc": final_fold[1],
                "primary_noninferior": bool(
                    final_fold[score_index] + tolerance >= base_fold[score_index]
                ),
                "primary_strict_improvement": bool(
                    final_fold[score_index] > base_fold[score_index] + tolerance
                ),
            }
        )
    noninferior_fraction = float(
        np.mean([bool(row["primary_noninferior"]) for row in fold_rows])
    )
    strict_improvement_fraction = float(
        np.mean([bool(row["primary_strict_improvement"]) for row in fold_rows])
    )
    small_sample = len(y) < maximum_subjects
    overall_noninferior = bool(
        final_score[score_index] + tolerance >= base_score[score_index]
    )
    triggered = bool(
        config.get("enabled", True)
        and (
            not overall_noninferior
            or (
                small_sample
                and (
                    noninferior_fraction < minimum_fold_fraction
                    or strict_improvement_fraction < minimum_strict_fraction
                )
            )
        )
    )
    return {
        "enabled": bool(config.get("enabled", True)),
        "training_subjects": int(len(y)),
        "maximum_training_subjects": maximum_subjects,
        "small_sample": small_sample,
        "selection_primary": primary,
        "minimum_noninferior_fold_fraction": minimum_fold_fraction,
        "noninferior_fold_fraction": noninferior_fraction,
        "minimum_strict_improvement_fold_fraction": minimum_strict_fraction,
        "strict_improvement_fold_fraction": strict_improvement_fraction,
        "overall_base_macro_f1": base_score[0],
        "overall_final_macro_f1": final_score[0],
        "overall_base_macro_auroc": base_score[1],
        "overall_final_macro_auroc": final_score[1],
        "overall_noninferior": overall_noninferior,
        "triggered": triggered,
        "action": "fall_back_to_base_probability" if triggered else "retain_agent_correction",
        "folds": fold_rows,
    }


def _expert_passes_cv_gate(cv_scores: dict[str, float], minimum_auroc: float) -> bool:
    """Use training-fold evidence only to reject a non-discriminative branch."""

    finite_scores = [float(score) for score in cv_scores.values() if np.isfinite(score)]
    return bool(finite_scores and max(finite_scores) >= minimum_auroc)


def _fit_text_expert(
    train_text: pd.Series,
    test_text: pd.Series,
    y: np.ndarray,
    labels: list[str],
    splitter: StratifiedKFold,
    c_grid: list[float],
    max_iter: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray, dict[str, dict[str, float]]]:
    min_df = 2 if len(train_text) >= 50 else 1
    candidates: dict[str, tuple[tuple[float, float], np.ndarray]] = {}
    for c in c_grid:
        oof = np.zeros((len(train_text), len(labels)), dtype=float)
        for train_index, validation_index in splitter.split(train_text, y):
            model = _text_pipeline(float(c), max_iter, min_df)
            model.fit(train_text.iloc[train_index], y[train_index])
            oof[validation_index] = _ordered_from_pipeline(
                model, train_text.iloc[validation_index], labels
            )
        candidates[str(c)] = (_selection_score(y, oof, labels), oof)
    selected = max(candidates, key=lambda key: candidates[key][0])
    model = _text_pipeline(float(selected), max_iter, min_df)
    model.fit(train_text, y)
    test_probability = _ordered_from_pipeline(model, test_text, labels)
    scores = {
        key: {"macro_f1": value[0][0], "macro_auroc": value[0][1]}
        for key, value in candidates.items()
    }
    return model, candidates[selected][1], test_probability, scores


def _fit_meta_model(
    train_x: np.ndarray,
    test_x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    splitter: StratifiedKFold,
    c_grid: list[float],
    max_iter: int,
) -> tuple[Pipeline, np.ndarray, np.ndarray, float, dict[str, dict[str, float]]]:
    candidates: dict[str, tuple[tuple[float, float], np.ndarray]] = {}
    for c in c_grid:
        oof = np.zeros((len(train_x), len(labels)), dtype=float)
        for train_index, validation_index in splitter.split(train_x, y):
            model = _numeric_pipeline(float(c), max_iter)
            model.fit(train_x[train_index], y[train_index])
            oof[validation_index] = _ordered_from_pipeline(
                model, train_x[validation_index], labels
            )
        candidates[str(c)] = (_selection_score(y, oof, labels), oof)
    selected = max(candidates, key=lambda key: candidates[key][0])
    model = _numeric_pipeline(float(selected), max_iter)
    model.fit(train_x, y)
    test_probability = _ordered_from_pipeline(model, test_x, labels)
    scores = {
        key: {"macro_f1": value[0][0], "macro_auroc": value[0][1]}
        for key, value in candidates.items()
    }
    return model, candidates[selected][1], test_probability, float(selected), scores


def _entropy(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-8, 1.0)
    return -np.sum(clipped * np.log(clipped), axis=1) / math.log(probability.shape[1])


def _margin(probability: np.ndarray) -> np.ndarray:
    ordered = np.sort(probability, axis=1)
    return ordered[:, -1] - ordered[:, -2]


def _fit_prototypes(
    frame: pd.DataFrame, state_columns: list[str], labels: list[str]
) -> dict[str, Any]:
    numeric = frame[state_columns].apply(pd.to_numeric, errors="coerce")
    global_median = numeric.median(axis=0)
    global_scale = (numeric.quantile(0.75) - numeric.quantile(0.25)).replace(0.0, np.nan)
    global_scale = global_scale.fillna(numeric.std(axis=0)).replace(0.0, 1.0).fillna(1.0)
    class_medians = {
        label: numeric[frame["label"].astype(str).eq(label)].median(axis=0).fillna(global_median)
        for label in labels
    }
    stacked = np.vstack([class_medians[label].to_numpy(dtype=float) for label in labels])
    discriminability = np.nanstd(stacked / global_scale.to_numpy(dtype=float), axis=0)
    discriminability = np.nan_to_num(discriminability, nan=0.0, posinf=0.0, neginf=0.0)
    if float(discriminability.sum()) <= 0.0:
        discriminability = np.ones(len(state_columns), dtype=float)
    return {
        "state_columns": state_columns,
        "global_median": global_median.to_dict(),
        "global_scale": global_scale.to_dict(),
        "class_medians": {
            label: class_medians[label].to_dict() for label in labels
        },
        "discriminability": {
            column: float(value) for column, value in zip(state_columns, discriminability, strict=True)
        },
    }


def _hierarchical_state_weights(
    columns: list[str],
    discriminability: pd.Series,
) -> np.ndarray:
    """Allocate one vote per base state, then distribute it across task views."""

    groups: dict[str, list[int]] = {}
    raw = discriminability.reindex(columns).fillna(0.0).clip(lower=0.0)
    for index, column in enumerate(columns):
        groups.setdefault(column.split("__task_", 1)[0], []).append(index)
    group_strength = {
        group: float(raw.iloc[indices].max()) for group, indices in groups.items()
    }
    group_total = max(sum(group_strength.values()), 1e-8)
    weights = np.zeros(len(columns), dtype=float)
    for group, indices in groups.items():
        within = raw.iloc[indices].to_numpy(dtype=float, copy=True)
        if float(within.sum()) <= 0.0:
            within = np.ones(len(indices), dtype=float)
        within /= float(within.sum())
        weights[indices] = (group_strength[group] / group_total) * within
    if float(weights.sum()) <= 0.0:
        return np.full(len(columns), 1.0 / max(len(columns), 1), dtype=float)
    return weights / float(weights.sum())


def _evidence_quality_by_subject(evidence: pd.DataFrame) -> dict[str, dict[str, float]]:
    clinical = evidence[
        evidence["evidence_role"].astype(str).isin(
            ["clinical", "clinical_support", "model_and_report"]
        )
        & evidence["report_permission"].fillna(False).astype(bool)
        & ~evidence["missing"].fillna(True).astype(bool)
    ].copy()
    result: dict[str, dict[str, float]] = {}
    for subject_id, group in clinical.groupby(clinical["subject_id"].astype(str)):
        confounds = group["confound_tags"].map(
            lambda value: len(json.loads(value))
            if isinstance(value, str) and value.startswith("[")
            else 0
        )
        result[str(subject_id)] = {
            "metric_coverage": float(len(group)),
            "metric_reliability": float(
                pd.to_numeric(group["reliability"], errors="coerce").fillna(0.0).mean()
            ),
            "confound_burden": float(np.clip((confounds / 3.0).mean(), 0.0, 1.0)),
        }
    return result


def _agent_feature_frame(
    frame: pd.DataFrame,
    prototypes: dict[str, Any],
    labels: list[str],
    base_probability: np.ndarray,
    evidence_quality: dict[str, dict[str, float]],
) -> tuple[pd.DataFrame, list[dict[str, float]]]:
    columns = prototypes["state_columns"]
    x = frame[columns].apply(pd.to_numeric, errors="coerce")
    observed_state_coverage = x.notna().mean(axis=1).fillna(0.0)
    global_median = pd.Series(prototypes["global_median"])
    scale = pd.Series(prototypes["global_scale"]).replace(0.0, 1.0)
    x = x.fillna(global_median)
    weight_array = _hierarchical_state_weights(
        columns,
        pd.Series(prototypes["discriminability"]),
    )

    raw_support = np.zeros((len(frame), len(labels)), dtype=float)
    for label_index, label in enumerate(labels):
        class_median = pd.Series(prototypes["class_medians"][label]).reindex(columns)
        distance = np.abs(
            (x.to_numpy(dtype=float) - class_median.to_numpy(dtype=float))
            / scale.reindex(columns).to_numpy(dtype=float)
        )
        raw_support[:, label_index] = -np.sum(distance * weight_array[None, :], axis=1)
    raw_support -= raw_support.max(axis=1, keepdims=True)
    support_probability = np.exp(np.clip(raw_support, -30.0, 0.0))
    support_probability /= support_probability.sum(axis=1, keepdims=True)

    rel_columns = [column.replace("state_", "rel_") for column in columns]
    available_rel = [column for column in rel_columns if column in frame.columns]
    state_reliability = (
        frame[available_rel].apply(pd.to_numeric, errors="coerce").mean(axis=1).fillna(0.0)
        if available_rel
        else pd.Series(np.ones(len(frame)), index=frame.index)
    )
    state_coverage = observed_state_coverage

    quality_rows = []
    for subject_id in frame["subject_id"].astype(str):
        quality_rows.append(
            evidence_quality.get(
                subject_id,
                {"metric_coverage": 0.0, "metric_reliability": 0.0, "confound_burden": 1.0},
            )
        )
    quality = pd.DataFrame(quality_rows, index=frame.index)
    max_metric_count = max(float(quality["metric_coverage"].max()), 1.0)
    coverage = np.sqrt(
        np.clip(state_coverage.to_numpy(dtype=float), 0.0, 1.0)
        * np.clip(quality["metric_coverage"].to_numpy(dtype=float) / max_metric_count, 0.0, 1.0)
    )
    reliability = np.sqrt(
        np.clip(state_reliability.to_numpy(dtype=float), 0.0, 1.0)
        * np.clip(quality["metric_reliability"].to_numpy(dtype=float), 0.0, 1.0)
    )
    confound = np.clip(quality["confound_burden"].to_numpy(dtype=float), 0.0, 1.0)

    task_groups: dict[str, list[str]] = {}
    for column in columns:
        if "__task_" in column:
            task_groups.setdefault(column.split("__task_", 1)[0], []).append(column)
    task_disagreement = np.zeros(len(frame), dtype=float)
    if task_groups:
        deviations = []
        for overall, task_columns in task_groups.items():
            if overall not in frame.columns:
                continue
            task_values = frame[task_columns].apply(pd.to_numeric, errors="coerce")
            deviations.append(
                task_values.sub(pd.to_numeric(frame[overall], errors="coerce"), axis=0)
                .abs()
                .mean(axis=1)
                .fillna(0.0)
                .to_numpy(dtype=float)
            )
        if deviations:
            task_disagreement = np.mean(np.vstack(deviations), axis=0)
    task_consistency = 1.0 / (1.0 + np.clip(task_disagreement, 0.0, None))

    output: dict[str, Any] = {
        **{
            f"agent_support_{label}": support_probability[:, index]
            for index, label in enumerate(labels)
        },
        "agent_evidence_coverage": coverage,
        "agent_evidence_reliability": reliability,
        "agent_confound_burden": confound,
        "agent_task_consistency": task_consistency,
        "agent_base_entropy": _entropy(base_probability),
        "agent_base_margin": _margin(base_probability),
        "agent_support_conflict": 1.0
        - np.max(support_probability, axis=1),
    }
    support_records = [
        {label: float(support_probability[row, index]) for index, label in enumerate(labels)}
        for row in range(len(frame))
    ]
    return pd.DataFrame(output, index=frame.index), support_records


def _temperature_scale(probability: np.ndarray, temperature: float) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-8, 1.0)) / max(float(temperature), 1e-3)
    logits -= logits.max(axis=1, keepdims=True)
    scaled = np.exp(logits)
    return scaled / scaled.sum(axis=1, keepdims=True)


def _fit_temperature(probability: np.ndarray, y: np.ndarray, labels: list[str]) -> float:
    label_index = {label: index for index, label in enumerate(labels)}
    targets = np.asarray([label_index[str(value)] for value in y], dtype=int)

    def objective(value: float) -> float:
        scaled = _temperature_scale(probability, float(value))
        return float(-np.mean(np.log(np.clip(scaled[np.arange(len(targets)), targets], 1e-8, 1.0))))

    result = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
    return float(result.x) if result.success else 1.0


def _apply_logit_offsets(probability: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    logits = np.log(np.clip(probability, 1e-8, 1.0)) + offsets[None, :]
    logits -= logits.max(axis=1, keepdims=True)
    adjusted = np.exp(logits)
    return adjusted / adjusted.sum(axis=1, keepdims=True)


def _stable_fold_offsets(
    offsets_by_fold: list[list[float]],
    *,
    minimum_direction_agreement: float = 0.80,
    minimum_supporting_folds: int = 3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Aggregate class offsets only when their direction is stable across training folds."""

    values = np.asarray(offsets_by_fold, dtype=float)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("offsets_by_fold must be a non-empty fold-by-class matrix")
    selected = np.zeros(values.shape[1], dtype=float)
    class_audit: list[dict[str, Any]] = []
    for class_index in range(values.shape[1]):
        column = values[:, class_index]
        positive = int(np.sum(column > 0.0))
        negative = int(np.sum(column < 0.0))
        supporting = max(positive, negative)
        agreement = float(supporting / len(column))
        median = float(np.median(column))
        stable = bool(
            supporting >= minimum_supporting_folds
            and agreement >= minimum_direction_agreement
            and median != 0.0
        )
        if stable:
            selected[class_index] = median
        class_audit.append(
            {
                "class_index": class_index,
                "positive_folds": positive,
                "negative_folds": negative,
                "zero_folds": int(np.sum(column == 0.0)),
                "direction_agreement": agreement,
                "fold_median": median,
                "retained": stable,
                "selected_offset": float(selected[class_index]),
            }
        )
    return selected, {
        "method": "direction_stable_fold_median",
        "minimum_direction_agreement": minimum_direction_agreement,
        "minimum_supporting_folds": minimum_supporting_folds,
        "classes": class_audit,
    }


def _fit_blend_parameters(
    base_probability: np.ndarray,
    corrected_probability: np.ndarray,
    gate: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    alpha_grid: list[float],
    offset_grid: list[float],
) -> tuple[float, np.ndarray, dict[str, dict[str, Any]]]:
    """Select bounded Agent correction using only out-of-fold predictions."""
    candidates: dict[str, dict[str, Any]] = {}
    best: tuple[tuple[float, float, float], float, np.ndarray] | None = None
    for alpha in alpha_grid:
        alpha_best: tuple[tuple[float, float, float], np.ndarray] | None = None
        for free_offsets in itertools.product(offset_grid, repeat=max(len(labels) - 1, 0)):
            offsets = np.asarray([0.0, *free_offsets], dtype=float)
            adjusted = _apply_logit_offsets(corrected_probability, offsets)
            blended = fuse_corrected_probability(
                base_probability, adjusted, gate, float(alpha)
            )
            f1, auc = _selection_score(y, blended, labels)
            score = (f1, auc, -float(np.square(offsets).sum()))
            if alpha_best is None or score > alpha_best[0]:
                alpha_best = (score, offsets)
            if best is None or score > best[0]:
                best = (score, float(alpha), offsets)
        assert alpha_best is not None
        candidates[str(alpha)] = {
            "macro_f1": alpha_best[0][0],
            "macro_auroc": alpha_best[0][1],
            "class_logit_offsets": alpha_best[1].tolist(),
        }
    assert best is not None
    return best[1], best[2], candidates


def _fit_correction_temperature(
    base_probability: np.ndarray,
    corrected_probability: np.ndarray,
    gate: np.ndarray,
    alpha: float,
    y: np.ndarray,
    labels: list[str],
) -> float:
    """Calibrate only the correction path, preserving exact base fallback at gate zero."""
    label_index = {label: index for index, label in enumerate(labels)}
    targets = np.asarray([label_index[str(value)] for value in y], dtype=int)

    def objective(value: float) -> float:
        corrected = _temperature_scale(corrected_probability, float(value))
        blended = fuse_corrected_probability(base_probability, corrected, gate, alpha)
        return float(
            -np.mean(
                np.log(np.clip(blended[np.arange(len(targets)), targets], 1e-8, 1.0))
            )
        )

    result = minimize_scalar(objective, bounds=(0.5, 5.0), method="bounded")
    return float(result.x) if result.success else 1.0


def _standardized_feature_importance(
    model: Pipeline,
    input_names: list[str],
) -> dict[str, float]:
    """Return mean absolute standardized coefficients with imputer names aligned."""
    imputer = model.named_steps["imputer"]
    classifier = model.named_steps["classifier"]
    try:
        names = [str(value) for value in imputer.get_feature_names_out(input_names)]
    except (AttributeError, ValueError):
        names = list(input_names)
    coefficients = np.abs(np.asarray(classifier.coef_, dtype=float)).mean(axis=0)
    if len(names) != len(coefficients):
        names = [f"transformed_feature_{index}" for index in range(len(coefficients))]
    return {
        name: float(value)
        for name, value in sorted(
            zip(names, coefficients, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
    }


def _case_level_expert_contributions(
    model: Pipeline,
    values: np.ndarray,
    input_names: list[str],
    expert_names: list[str],
    labels: list[str],
    probability: np.ndarray,
) -> np.ndarray:
    """Group absolute class-logit contributions into normalized expert shares."""

    imputer = model.named_steps["imputer"]
    scaler = model.named_steps["scaler"]
    classifier = model.named_steps["classifier"]
    imputed = imputer.transform(values)
    transformed = scaler.transform(imputed)
    try:
        transformed_names = [
            str(value) for value in imputer.get_feature_names_out(input_names)
        ]
    except (AttributeError, ValueError):
        transformed_names = list(input_names)
    if len(transformed_names) != transformed.shape[1]:
        transformed_names = [
            f"transformed_feature_{index}" for index in range(transformed.shape[1])
        ]

    classes = [str(value) for value in classifier.classes_]
    coefficients = np.asarray(classifier.coef_, dtype=float)
    if coefficients.shape[0] == 1 and len(classes) == 2:
        coefficient_by_class = {
            classes[0]: -coefficients[0],
            classes[1]: coefficients[0],
        }
    else:
        coefficient_by_class = {
            class_label: coefficients[index]
            for index, class_label in enumerate(classes)
        }

    feature_to_expert = np.full(len(transformed_names), -1, dtype=int)
    for feature_index, feature_name in enumerate(transformed_names):
        for expert_index, expert_name in enumerate(expert_names):
            if expert_name in feature_name:
                feature_to_expert[feature_index] = expert_index
                break

    predicted_labels = np.asarray(labels)[np.argmax(probability, axis=1)]
    contribution = np.zeros((len(transformed), len(expert_names)), dtype=float)
    for row_index, predicted_label in enumerate(predicted_labels):
        coefficient = coefficient_by_class[str(predicted_label)]
        feature_contribution = np.abs(transformed[row_index] * coefficient)
        for expert_index in range(len(expert_names)):
            contribution[row_index, expert_index] = float(
                feature_contribution[feature_to_expert == expert_index].sum()
            )

    row_total = contribution.sum(axis=1, keepdims=True)
    uniform = np.full_like(contribution, 1.0 / max(len(expert_names), 1))
    return np.divide(
        contribution,
        row_total,
        out=uniform,
        where=row_total > 0.0,
    )


def _merge_inputs(
    features: pd.DataFrame,
    states: pd.DataFrame,
    transcripts: pd.DataFrame,
) -> pd.DataFrame:
    ordered = features[IDENTITY].copy()
    ordered["_row_order"] = np.arange(len(ordered))
    text = transcripts[["subject_id", "transcript"]].copy()
    text["subject_id"] = text["subject_id"].astype(str)
    text["transcript"] = text["transcript"].map(clean_model_transcript)
    frame = (
        features.merge(states, on=IDENTITY, how="inner", validate="one_to_one")
        .merge(text, on="subject_id", how="left", validate="one_to_one")
        .merge(ordered, on=IDENTITY, how="inner", validate="one_to_one")
        .sort_values("_row_order")
        .drop(columns="_row_order")
        .reset_index(drop=True)
    )
    frame["transcript"] = frame["transcript"].fillna("[no_transcript]")
    frame.loc[frame["transcript"].str.len().eq(0), "transcript"] = "[no_transcript]"
    return frame


def _most_common_float(values: list[float], default: float) -> float:
    if not values:
        return float(default)
    counts: dict[float, int] = {}
    for value in values:
        key = float(value)
        counts[key] = counts.get(key, 0) + 1
    return max(counts, key=lambda value: (counts[value], -abs(value - default)))


def _fit_fixed_numeric_model(
    train_x: np.ndarray,
    test_x: np.ndarray,
    y: np.ndarray,
    labels: list[str],
    c: float,
    max_iter: int,
) -> tuple[Pipeline, np.ndarray]:
    model = _numeric_pipeline(c, max_iter)
    model.fit(train_x, y)
    return model, _ordered_from_pipeline(model, test_x, labels)


def _nested_fusion_selection(
    *,
    train: pd.DataFrame,
    y: np.ndarray,
    labels: list[str],
    split_indices: list[tuple[np.ndarray, np.ndarray]],
    fold_frames: list[pd.DataFrame],
    expert_sources: list[dict[str, Any]],
    c_grid: list[float],
    max_iter: int,
    qc_alpha: float,
    evidence_quality: pd.DataFrame,
    state_columns: list[str],
    dynamic_gate_config: dict[str, Any],
    alpha_grid: list[float],
    offset_grid: list[float],
) -> dict[str, Any]:
    """Select both supervised modules using outer-fold-isolated predictions."""

    candidate_names = ["logistic"]
    dynamic_enabled = bool(dynamic_gate_config.get("enabled", False)) and len(
        expert_sources
    ) >= 2
    if dynamic_enabled:
        candidate_names.append("dynamic")
    outputs: dict[str, dict[str, Any]] = {
        name: {
            "base_oof": np.zeros((len(train), len(labels)), dtype=float),
            "corrected_oof": np.zeros((len(train), len(labels)), dtype=float),
            "final_oof": np.zeros((len(train), len(labels)), dtype=float),
            "gate_oof": np.zeros(len(train), dtype=float),
            "base_c_by_fold": [],
            "correction_c_by_fold": [],
            "alpha_by_fold": [],
            "offsets_by_fold": [],
            "temperature_by_fold": [],
        }
        for name in candidate_names
    }

    for outer_fold, (outer_fit, outer_validation) in enumerate(split_indices):
        outer_frame = fold_frames[outer_fold]
        inner_train = outer_frame.iloc[outer_fit].reset_index(drop=True)
        outer_holdout = outer_frame.iloc[outer_validation].reset_index(drop=True)
        inner_y = y[outer_fit]
        inner_folds = min(
            len(split_indices), int(pd.Series(inner_y).value_counts().min())
        )
        if inner_folds < 2:
            raise ValueError(
                "Strict nested fusion requires at least two outer-training subjects per class."
            )
        inner_splitter = StratifiedKFold(
            n_splits=inner_folds,
            shuffle=True,
            random_state=20260821 + outer_fold + 1,
        )
        inner_splits = list(inner_splitter.split(inner_train, inner_y))
        inner_probability: list[np.ndarray] = []
        holdout_probability: list[np.ndarray] = []
        inner_reliability: list[np.ndarray] = []
        holdout_reliability: list[np.ndarray] = []

        for source in expert_sources:
            source_kind = str(source["kind"])
            if source_kind == "branch":
                specification = source["specification"]
                expert_model, expert_inner, _, _ = _fit_branch(
                    inner_train,
                    inner_y,
                    c_grid,
                    max_iter,
                    inner_splitter,
                    labels,
                    specification,
                    qc_alpha,
                    None,
                )
                expert_holdout = _predict_ordered(
                    expert_model,
                    outer_holdout[specification["features"]],
                    labels,
                )
            elif source_kind == "text":
                _, expert_inner, expert_holdout, _ = _fit_text_expert(
                    inner_train["transcript"],
                    outer_holdout["transcript"],
                    inner_y,
                    labels,
                    inner_splitter,
                    c_grid,
                    max_iter,
                )
            elif source_kind == "dense":
                values = np.asarray(source["values"], dtype=float)
                _, expert_inner, expert_holdout, _, _ = _fit_meta_model(
                    values[outer_fit],
                    values[outer_validation],
                    inner_y,
                    labels,
                    inner_splitter,
                    c_grid,
                    max_iter,
                )
            elif source_kind == "segment":
                sequence = np.asarray(source["sequence"], dtype=np.float32)
                mask = np.asarray(source["mask"], dtype=bool)
                _, expert_inner, expert_holdout, _ = fit_segment_attention_expert(
                    sequence[outer_fit],
                    mask[outer_fit],
                    sequence[outer_validation],
                    mask[outer_validation],
                    inner_y,
                    labels,
                    inner_splits,
                    source["config"],
                )
            else:
                raise ValueError(f"Unsupported nested expert source: {source_kind}")
            inner_probability.append(expert_inner)
            holdout_probability.append(expert_holdout)
            reliability = np.asarray(source["train_reliability"], dtype=float)
            inner_reliability.append(reliability[outer_fit])
            holdout_reliability.append(reliability[outer_validation])

        inner_stack = np.stack(inner_probability, axis=1)
        holdout_stack = np.stack(holdout_probability, axis=1)
        inner_rel = np.column_stack(inner_reliability)
        holdout_rel = np.column_stack(holdout_reliability)
        inner_base_x = np.column_stack(
            [
                np.log(
                    np.clip(inner_stack.reshape(len(inner_train), -1), 1e-8, 1.0)
                ),
                inner_rel,
            ]
        )
        holdout_base_x = np.column_stack(
            [
                np.log(
                    np.clip(
                        holdout_stack.reshape(len(outer_holdout), -1), 1e-8, 1.0
                    )
                ),
                holdout_rel,
            ]
        )
        (
            _,
            logistic_inner,
            logistic_holdout,
            logistic_c,
            _,
        ) = _fit_meta_model(
            inner_base_x,
            holdout_base_x,
            inner_y,
            labels,
            inner_splitter,
            c_grid,
            max_iter,
        )
        base_candidates: dict[str, tuple[np.ndarray, np.ndarray, float]] = {
            "logistic": (logistic_inner, logistic_holdout, logistic_c)
        }
        if dynamic_enabled:
            (
                _,
                dynamic_inner,
                dynamic_holdout,
                _,
                _,
                _,
            ) = fit_dynamic_reliability_gate(
                inner_stack,
                inner_rel,
                holdout_stack,
                holdout_rel,
                inner_y,
                labels,
                inner_splits,
                dynamic_gate_config,
            )
            base_candidates["dynamic"] = (
                dynamic_inner,
                dynamic_holdout,
                float(dynamic_gate_config.get("fixed_c_marker", 1.0)),
            )

        for candidate_name, (
            base_inner,
            base_holdout,
            base_c,
        ) in base_candidates.items():
            agent_inner = pd.DataFrame(index=np.arange(len(inner_train)))
            for inner_fit, inner_validation in inner_splits:
                inner_prototypes = _fit_prototypes(
                    inner_train.iloc[inner_fit], state_columns, labels
                )
                fold_agent, _ = _agent_feature_frame(
                    inner_train.iloc[inner_validation],
                    inner_prototypes,
                    labels,
                    base_inner[inner_validation],
                    evidence_quality,
                )
                agent_inner.loc[inner_validation, fold_agent.columns] = (
                    fold_agent.to_numpy()
                )
            agent_inner = agent_inner.astype(float)
            outer_prototypes = _fit_prototypes(inner_train, state_columns, labels)
            agent_holdout, _ = _agent_feature_frame(
                outer_holdout,
                outer_prototypes,
                labels,
                base_holdout,
                evidence_quality,
            )
            correction_inner_x = np.column_stack(
                [
                    np.log(np.clip(base_inner, 1e-8, 1.0)),
                    agent_inner.to_numpy(dtype=float),
                ]
            )
            correction_holdout_x = np.column_stack(
                [
                    np.log(np.clip(base_holdout, 1e-8, 1.0)),
                    agent_holdout.to_numpy(dtype=float),
                ]
            )
            (
                _,
                corrected_inner,
                corrected_holdout,
                correction_c,
                _,
            ) = _fit_meta_model(
                correction_inner_x,
                correction_holdout_x,
                inner_y,
                labels,
                inner_splitter,
                c_grid,
                max_iter,
            )
            inner_gate = evidence_gate(
                agent_inner["agent_evidence_coverage"].to_numpy(),
                agent_inner["agent_evidence_reliability"].to_numpy(),
                agent_inner["agent_confound_burden"].to_numpy(),
            )
            holdout_gate = evidence_gate(
                agent_holdout["agent_evidence_coverage"].to_numpy(),
                agent_holdout["agent_evidence_reliability"].to_numpy(),
                agent_holdout["agent_confound_burden"].to_numpy(),
            )
            alpha, offsets, _ = _fit_blend_parameters(
                base_inner,
                corrected_inner,
                inner_gate,
                inner_y,
                labels,
                alpha_grid,
                offset_grid,
            )
            adjusted_inner = _apply_logit_offsets(corrected_inner, offsets)
            adjusted_holdout = _apply_logit_offsets(corrected_holdout, offsets)
            temperature = _fit_correction_temperature(
                base_inner,
                adjusted_inner,
                inner_gate,
                alpha,
                inner_y,
                labels,
            )
            calibrated_holdout = _temperature_scale(
                adjusted_holdout, temperature
            )
            final_holdout = fuse_corrected_probability(
                base_holdout,
                calibrated_holdout,
                holdout_gate,
                alpha,
            )
            output = outputs[candidate_name]
            output["base_oof"][outer_validation] = base_holdout
            output["corrected_oof"][outer_validation] = calibrated_holdout
            output["final_oof"][outer_validation] = final_holdout
            output["gate_oof"][outer_validation] = holdout_gate
            output["base_c_by_fold"].append(base_c)
            output["correction_c_by_fold"].append(correction_c)
            output["alpha_by_fold"].append(alpha)
            output["offsets_by_fold"].append(offsets.tolist())
            output["temperature_by_fold"].append(temperature)

    primary = str(dynamic_gate_config.get("selection_primary", "macro_f1"))
    scores = {
        name: _selection_score(y, output["final_oof"], labels)
        for name, output in outputs.items()
    }
    if primary == "macro_auroc":
        selected = max(scores, key=lambda name: (scores[name][1], scores[name][0]))
    else:
        selected = max(scores, key=lambda name: scores[name])
    for output in outputs.values():
        output["selected_base_c"] = _most_common_float(
            output["base_c_by_fold"], 1.0
        )
        output["selected_correction_c"] = _most_common_float(
            output["correction_c_by_fold"], 1.0
        )
        output["selected_alpha"] = float(np.median(output["alpha_by_fold"]))
        output["selected_offsets"], output["offset_stability_audit"] = _stable_fold_offsets(
            output["offsets_by_fold"],
            minimum_direction_agreement=float(
                dynamic_gate_config.get("minimum_offset_direction_agreement", 0.80)
            ),
            minimum_supporting_folds=int(
                dynamic_gate_config.get("minimum_offset_supporting_folds", 3)
            ),
        )
        output["selected_temperature"] = float(
            np.median(output["temperature_by_fold"])
        )
    return {
        "selected": selected,
        "outputs": outputs,
        "scores": {
            name: {"macro_f1": score[0], "macro_auroc": score[1]}
            for name, score in scores.items()
        },
        "selection_primary": primary,
        "protocol": "strict_outer_fold_nested_refit",
    }


def train_condition_c(
    subject_features_path: Path,
    subject_transcripts_path: Path,
    state_wide_path: Path,
    metric_evidence_path: Path,
    state_cards_path: Path,
    states_config: dict[str, Any],
    models_config: dict[str, Any],
    predictions_path: Path,
    base_predictions_path: Path,
    ablations_path: Path,
    interventions_path: Path,
    workspaces_path: Path,
    contributions_path: Path,
    model_path: Path,
    metadata_path: Path,
    agent_calibration_predictions_path: Path | None = None,
    agent_calibration_workspaces_path: Path | None = None,
) -> None:
    """Train the 8.21 evidence diagnostic Agent without using held-out labels."""

    labels = _labels(models_config)
    config = models_config.get("condition_c", {})
    c_grid = [float(value) for value in config.get("c_grid", [0.01, 0.1, 1.0, 10.0])]
    max_iter = int(config.get("max_iter", 4000))
    features = pd.read_csv(subject_features_path, dtype={"subject_id": str}).copy()
    demographic_config = config.get("demographic_context", {})
    demographic_features = _add_age_band_features(features, demographic_config)
    transcripts = pd.read_csv(subject_transcripts_path, dtype={"subject_id": str})
    evidence = pd.read_csv(metric_evidence_path, dtype={"subject_id": str})
    cards = pd.read_csv(state_cards_path, dtype={"subject_id": str})
    stored_states = pd.read_csv(state_wide_path, dtype={"subject_id": str})
    missing_state_subjects = sorted(
        set(features["subject_id"].astype(str))
        - set(stored_states["subject_id"].astype(str))
    )
    if missing_state_subjects:
        raise ValueError(
            "State table is missing subjects present in the feature table: "
            f"{missing_state_subjects[:10]}"
        )
    train_subject_ids = set(
        features.loc[features["split"].eq("train"), "subject_id"].astype(str)
    )
    calibrated_states = build_fold_calibrated_state_frame(
        evidence, states_config, train_subject_ids, labels[0]
    )
    frame = _merge_inputs(features, calibrated_states, transcripts)
    train = frame[frame["split"].eq("train")].copy().reset_index(drop=True)
    test = frame[frame["split"].eq("test")].copy().reset_index(drop=True)
    y = train["label"].astype(str).to_numpy()
    folds = min(
        int(models_config["cross_validation"]["folds"]),
        int(pd.Series(y).value_counts().min()),
    )
    if folds < 2:
        raise ValueError("Condition C requires at least two training subjects per class.")
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=20260821)
    split_indices = list(splitter.split(train, y))

    train_evidence = evidence[evidence["subject_id"].astype(str).isin(train_subject_ids)]
    fold_frames: list[pd.DataFrame] = []
    train_features = features[features["split"].eq("train")].copy().reset_index(drop=True)
    train_text = train[["subject_id", "transcript"]].copy()
    for train_index, _ in split_indices:
        reference_ids = set(train.iloc[train_index]["subject_id"].astype(str))
        fold_states = build_fold_calibrated_state_frame(
            train_evidence, states_config, reference_ids, labels[0]
        )
        fold_frames.append(_merge_inputs(train_features, fold_states, train_text))

    qc_guard_config = config.get("qc_shortcut_guard", {})
    qc_guard_features = [
        column
        for column in models_config.get("ours", {})
        .get("qc_orthogonalization", {})
        .get("features", [])
        if column in train.columns
    ]
    qc_guard_metadata: dict[str, Any] = {
        "enabled": bool(qc_guard_config.get("enabled", True)),
        "features": qc_guard_features,
        "triggered": False,
    }
    if qc_guard_metadata["enabled"] and qc_guard_features:
        _, qc_guard_oof, _, qc_guard_c, _ = _fit_meta_model(
            train[qc_guard_features].to_numpy(dtype=float),
            test[qc_guard_features].to_numpy(dtype=float),
            y,
            labels,
            splitter,
            c_grid,
            max_iter,
        )
        _, qc_guard_auc = _selection_score(y, qc_guard_oof, labels)
        qc_guard_threshold = float(qc_guard_config.get("macro_auroc_threshold", 0.75))
        qc_guard_metadata.update(
            {
                "selected_c": qc_guard_c,
                "train_oof_macro_auroc": qc_guard_auc,
                "macro_auroc_threshold": qc_guard_threshold,
                "triggered": bool(qc_guard_auc >= qc_guard_threshold),
                "action": (
                    "exclude_model_only_deep_audio_experts"
                    if qc_guard_auc >= qc_guard_threshold
                    else "retain_with_case_reliability_gate"
                ),
            }
        )

    specifications = _branch_specifications(frame, features, models_config)
    traditional_acoustic_features = _acoustic_feature_columns(features)
    if traditional_acoustic_features:
        specifications.append(
            {
                "name": "traditional_acoustic_prior",
                "features": traditional_acoustic_features,
                "state_features": traditional_acoustic_features,
                "qc_features": [],
                "qc_orthogonalized": False,
                "reliability": ["audio_reliability"],
                "kind": "model_only_context",
                "report_permission": False,
            }
        )
    if (
        demographic_features
        and pd.to_numeric(train["age"], errors="coerce").notna().mean()
        >= float(demographic_config.get("minimum_age_coverage", 0.5))
    ):
        specifications.append(
            {
                "name": "demographic_context",
                "features": demographic_features,
                "reliability": [],
                "kind": "model_only_context",
                "report_permission": False,
            }
        )
    expert_names: list[str] = []
    expert_models: dict[str, Any] = {}
    expert_oof: list[np.ndarray] = []
    expert_test: list[np.ndarray] = []
    expert_train_reliability: list[np.ndarray] = []
    expert_test_reliability: list[np.ndarray] = []
    expert_sources: list[dict[str, Any]] = []
    expert_cv: dict[str, Any] = {}
    minimum_branch_cv_auroc = float(
        config.get(
            "minimum_branch_cv_auroc",
            models_config.get("ours", {}).get("min_branch_cv_auroc", 0.52),
        )
    )
    qc_alpha = float(
        models_config.get("ours", {})
        .get("qc_orthogonalization", {})
        .get("alpha", 1.0)
    )
    for specification in specifications:
        model, oof, selected_c, cv_scores = _fit_branch(
            train,
            y,
            c_grid,
            max_iter,
            splitter,
            labels,
            specification,
            qc_alpha,
            fold_frames if specification["kind"] == "clinical_state" else None,
        )
        name = str(specification["name"])
        if not _expert_passes_cv_gate(cv_scores, minimum_branch_cv_auroc):
            expert_cv[name] = {
                "selected_c": selected_c,
                "macro_auroc": cv_scores,
                "selected": False,
                "exclusion_reason": "below_train_only_cv_auroc_gate",
                "minimum_cv_auroc": minimum_branch_cv_auroc,
            }
            continue
        expert_names.append(name)
        expert_models[name] = {"model": model, "specification": specification}
        expert_oof.append(oof)
        expert_test.append(_predict_ordered(model, test[specification["features"]], labels))
        train_rel = [column for column in specification["reliability"] if column in train.columns]
        test_rel = [column for column in specification["reliability"] if column in test.columns]
        train_reliability_values = (
            train[train_rel].mean(axis=1).fillna(0.0).to_numpy()
            if train_rel
            else np.ones(len(train))
        )
        expert_train_reliability.append(train_reliability_values)
        expert_test_reliability.append(
            test[test_rel].mean(axis=1).fillna(0.0).to_numpy()
            if test_rel
            else np.ones(len(test))
        )
        expert_sources.append(
            {
                "kind": "branch",
                "specification": specification,
                "train_reliability": train_reliability_values,
            }
        )
        expert_cv[name] = {
            "selected_c": selected_c,
            "macro_auroc": cv_scores,
            "selected": True,
            "minimum_cv_auroc": minimum_branch_cv_auroc,
        }

    text_available = train["transcript"].ne("[no_transcript]").mean() >= float(
        config.get("minimum_text_coverage", 0.2)
    )
    if text_available:
        text_model, text_oof, text_test, text_scores = _fit_text_expert(
            train["transcript"],
            test["transcript"],
            y,
            labels,
            splitter,
            c_grid,
            max_iter,
        )
        expert_names.append("multilingual_text")
        expert_models["multilingual_text"] = {"model": text_model}
        expert_oof.append(text_oof)
        expert_test.append(text_test)
        text_train_reliability = (
            pd.to_numeric(train.get("text_reliability", 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        expert_train_reliability.append(text_train_reliability)
        expert_test_reliability.append(
            pd.to_numeric(test.get("text_reliability", 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        expert_sources.append(
            {
                "kind": "text",
                "train_reliability": text_train_reliability,
            }
        )
        expert_cv["multilingual_text"] = text_scores

    deep_text_metadata: dict[str, Any] = {"enabled": False}
    deep_text_config = config.get("deep_text", {})
    if (
        text_available
        and bool(deep_text_config.get("enabled", False))
        and len(train) >= int(deep_text_config.get("minimum_training_subjects", 100))
    ):
        all_embeddings, deep_text_metadata = encode_multilingual_text(
            frame["transcript"].astype(str).tolist(),
            frame["subject_id"].astype(str).tolist(),
            metadata_path.parent / "multilingual_text_embeddings.npz",
            deep_text_config,
        )
        train_mask = frame["split"].eq("train").to_numpy()
        test_mask = frame["split"].eq("test").to_numpy()
        dense_model, dense_oof, dense_test, dense_c, dense_scores = _fit_meta_model(
            all_embeddings[train_mask],
            all_embeddings[test_mask],
            y,
            labels,
            splitter,
            c_grid,
            max_iter,
        )
        expert_names.append("multilingual_dense_text")
        expert_models["multilingual_dense_text"] = {
            "model": dense_model,
            "encoder": deep_text_metadata,
        }
        expert_oof.append(dense_oof)
        expert_test.append(dense_test)
        dense_text_train_reliability = (
            pd.to_numeric(train.get("text_reliability", 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        expert_train_reliability.append(dense_text_train_reliability)
        expert_test_reliability.append(
            pd.to_numeric(test.get("text_reliability", 0.0), errors="coerce")
            .fillna(0.0)
            .to_numpy()
        )
        expert_sources.append(
            {
                "kind": "dense",
                "values": all_embeddings[train_mask],
                "train_reliability": dense_text_train_reliability,
            }
        )
        expert_cv["multilingual_dense_text"] = {
            "selected_c": dense_c,
            "macro_auroc": {
                key: value["macro_auroc"] for key, value in dense_scores.items()
            },
            "macro_f1": {
                key: value["macro_f1"] for key, value in dense_scores.items()
            },
        }
        deep_text_metadata = {**deep_text_metadata, "enabled": True}

    deep_audio_metadata: dict[str, Any] = {"enabled": False}
    deep_audio_config = config.get("deep_audio", {})
    analysis_manifest_path = subject_features_path.parent / "analysis_manifest.csv"
    if (
        bool(deep_audio_config.get("enabled", False))
        and not bool(qc_guard_metadata["triggered"])
        and analysis_manifest_path.exists()
        and len(train) >= int(deep_audio_config.get("minimum_training_subjects", 300))
    ):
        audio_embeddings, deep_audio_metadata = encode_multilingual_audio(
            analysis_manifest_path,
            frame["subject_id"].astype(str).tolist(),
            metadata_path.parent / "multilingual_audio_embeddings.npz",
            deep_audio_config,
        )
        train_mask = frame["split"].eq("train").to_numpy()
        test_mask = frame["split"].eq("test").to_numpy()
        train_audio = audio_embeddings[train_mask]
        test_audio = audio_embeddings[test_mask]
        train_audio_coverage = float(np.isfinite(train_audio).any(axis=1).mean())
        if train_audio_coverage >= float(
            deep_audio_config.get("minimum_audio_coverage", 0.8)
        ):
            audio_model, audio_oof, audio_test, audio_c, audio_scores = _fit_meta_model(
                train_audio,
                test_audio,
                y,
                labels,
                splitter,
                c_grid,
                max_iter,
            )
            expert_names.append("multilingual_dense_audio")
            expert_models["multilingual_dense_audio"] = {
                "model": audio_model,
                "encoder": deep_audio_metadata,
            }
            expert_oof.append(audio_oof)
            expert_test.append(audio_test)
            train_audio_reliability = pd.to_numeric(
                train.get("audio_reliability", pd.Series(1.0, index=train.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy()
            test_audio_reliability = pd.to_numeric(
                test.get("audio_reliability", pd.Series(1.0, index=test.index)),
                errors="coerce",
            ).fillna(0.0).to_numpy()
            train_audio_reliability = train_audio_reliability * np.isfinite(train_audio).any(axis=1)
            test_audio_reliability = test_audio_reliability * np.isfinite(test_audio).any(axis=1)
            expert_train_reliability.append(train_audio_reliability)
            expert_test_reliability.append(test_audio_reliability)
            expert_sources.append(
                {
                    "kind": "dense",
                    "values": train_audio,
                    "train_reliability": train_audio_reliability,
                }
            )
            expert_cv["multilingual_dense_audio"] = {
                "selected_c": audio_c,
                "macro_auroc": {
                    key: value["macro_auroc"] for key, value in audio_scores.items()
                },
                "macro_f1": {
                    key: value["macro_f1"] for key, value in audio_scores.items()
                },
            }
            deep_audio_metadata = {
                **deep_audio_metadata,
                "enabled": True,
                "used_as_expert": True,
                "training_subject_coverage": train_audio_coverage,
            }
            segment_config = deep_audio_config.get("segment_attention", {})
            if bool(segment_config.get("enabled", True)):
                sequence, sequence_mask = load_audio_window_sequences(
                    metadata_path.parent / "multilingual_audio_embeddings.npz",
                    frame["subject_id"].astype(str).tolist(),
                )
                segment_bundle, segment_oof, segment_test, segment_metadata = (
                    fit_segment_attention_expert(
                        sequence[train_mask],
                        sequence_mask[train_mask],
                        sequence[test_mask],
                        sequence_mask[test_mask],
                        y,
                        labels,
                        split_indices,
                        segment_config,
                    )
                )
                expert_names.append("multilingual_segment_audio")
                expert_models["multilingual_segment_audio"] = {
                    "model": segment_bundle,
                    "encoder": deep_audio_metadata,
                }
                expert_oof.append(segment_oof)
                expert_test.append(segment_test)
                expert_train_reliability.append(train_audio_reliability.copy())
                expert_test_reliability.append(test_audio_reliability.copy())
                expert_sources.append(
                    {
                        "kind": "segment",
                        "sequence": sequence[train_mask],
                        "mask": sequence_mask[train_mask],
                        "config": segment_config,
                        "train_reliability": train_audio_reliability.copy(),
                    }
                )
                segment_f1, segment_auc = _selection_score(y, segment_oof, labels)
                expert_cv["multilingual_segment_audio"] = {
                    "selected_c": "fixed",
                    "macro_f1": {"fixed": segment_f1},
                    "macro_auroc": {"fixed": segment_auc},
                    **segment_metadata,
                }
                deep_audio_metadata["segment_attention"] = segment_metadata
        else:
            deep_audio_metadata = {
                **deep_audio_metadata,
                "enabled": True,
                "used_as_expert": False,
                "training_subject_coverage": train_audio_coverage,
                "exclusion_reason": "below_minimum_audio_coverage",
            }
    elif bool(qc_guard_metadata["triggered"]):
        deep_audio_metadata = {
            "enabled": False,
            "used_as_expert": False,
            "exclusion_reason": "train_only_qc_shortcut_guard",
            "qc_shortcut_guard": qc_guard_metadata,
        }

    oof_stack = np.stack(expert_oof, axis=1)
    test_stack = np.stack(expert_test, axis=1)
    train_reliability = np.column_stack(expert_train_reliability)
    test_reliability = np.column_stack(expert_test_reliability)
    matched_encoder_test_probability: np.ndarray | None = None
    matched_encoder_metadata: dict[str, Any] = {"available": False}
    matched_names = {
        "multilingual_dense_audio",
        "multilingual_dense_text",
        "demographic_context",
    }
    matched_indices = [
        index for index, name in enumerate(expert_names) if name in matched_names
    ]
    if len(matched_indices) >= 2:
        matched_oof_stack = oof_stack[:, matched_indices, :]
        matched_test_stack = test_stack[:, matched_indices, :]
        matched_train_reliability = train_reliability[:, matched_indices]
        matched_test_reliability = test_reliability[:, matched_indices]
        (
            _,
            matched_encoder_oof,
            matched_encoder_test_probability,
            _,
            _,
            matched_gate_metadata,
        ) = fit_dynamic_reliability_gate(
            matched_oof_stack,
            matched_train_reliability,
            matched_test_stack,
            matched_test_reliability,
            y,
            labels,
            split_indices,
            config.get("dynamic_gate", {}),
        )
        matched_f1, matched_auc = _selection_score(
            y, matched_encoder_oof, labels
        )
        matched_encoder_metadata = {
            "available": True,
            "experts": [expert_names[index] for index in matched_indices],
            "train_oof_macro_f1": matched_f1,
            "train_oof_macro_auroc": matched_auc,
            "comparison_scope": "same_frozen_backbones_and_same_split",
            **matched_gate_metadata,
        }
    base_train_x = np.column_stack(
        [np.log(np.clip(oof_stack.reshape(len(train), -1), 1e-8, 1.0)), train_reliability]
    )
    base_test_x = np.column_stack(
        [np.log(np.clip(test_stack.reshape(len(test), -1), 1e-8, 1.0)), test_reliability]
    )
    base_feature_names = [
        f"log_probability__{expert_name}__{label}"
        for expert_name in expert_names
        for label in labels
    ] + [f"reliability__{expert_name}" for expert_name in expert_names]
    state_columns = sorted(
        column for column in train.columns if column.startswith("state_")
    )
    evidence_quality = _evidence_quality_by_subject(evidence)
    dynamic_gate_config = config.get("dynamic_gate", {})
    alpha_grid = [
        float(value)
        for value in config.get("alpha_grid", [0.0, 0.25, 0.5, 0.75, 1.0])
    ]
    offset_grid = [
        float(value)
        for value in config.get(
            "class_offset_grid", [-1.0, -0.5, 0.0, 0.5, 1.0]
        )
    ]
    nested_selection = _nested_fusion_selection(
        train=train,
        y=y,
        labels=labels,
        split_indices=split_indices,
        fold_frames=fold_frames,
        expert_sources=expert_sources,
        c_grid=c_grid,
        max_iter=max_iter,
        qc_alpha=qc_alpha,
        evidence_quality=evidence_quality,
        state_columns=state_columns,
        dynamic_gate_config=dynamic_gate_config,
        alpha_grid=alpha_grid,
        offset_grid=offset_grid,
    )
    logistic_nested = nested_selection["outputs"]["logistic"]
    logistic_c = float(logistic_nested["selected_base_c"])
    logistic_base_model, logistic_base_test = _fit_fixed_numeric_model(
        base_train_x,
        base_test_x,
        y,
        labels,
        logistic_c,
        max_iter,
    )
    logistic_base_oof = logistic_nested["base_oof"]
    base_c = logistic_c
    base_scores = {"strict_nested": nested_selection["scores"]["logistic"]}
    base_model: Any = logistic_base_model
    base_oof = logistic_base_oof
    base_test = logistic_base_test
    base_architecture = "multinomial_logistic_stacking"
    dynamic_gate_bundle: Any = None
    dynamic_gate_metadata: dict[str, Any] = {"enabled": False}
    dynamic_train_weights: np.ndarray | None = None
    dynamic_test_weights: np.ndarray | None = None
    if bool(dynamic_gate_config.get("enabled", False)) and len(expert_names) >= 2:
        (
            candidate_bundle,
            candidate_oof,
            candidate_test,
            candidate_train_weights,
            candidate_test_weights,
            candidate_metadata,
        ) = fit_dynamic_reliability_gate(
            oof_stack,
            train_reliability,
            test_stack,
            test_reliability,
            y,
            labels,
            split_indices,
            dynamic_gate_config,
        )
        logistic_score = nested_selection["scores"]["logistic"]
        candidate_score = nested_selection["scores"]["dynamic"]
        primary = nested_selection["selection_primary"]
        candidate_selected = nested_selection["selected"] == "dynamic"
        dynamic_gate_bundle = candidate_bundle
        dynamic_gate_metadata = {
            "enabled": True,
            "selected": bool(candidate_selected),
            "selection_primary": primary,
            "selection_protocol": nested_selection["protocol"],
            "logistic_oof_macro_f1": logistic_score["macro_f1"],
            "logistic_oof_macro_auroc": logistic_score["macro_auroc"],
            "dynamic_oof_macro_f1": candidate_score["macro_f1"],
            "dynamic_oof_macro_auroc": candidate_score["macro_auroc"],
            **candidate_metadata,
        }
        if candidate_selected:
            base_model = candidate_bundle
            base_oof = nested_selection["outputs"]["dynamic"]["base_oof"]
            base_test = candidate_test
            base_architecture = "dynamic_reliability_gate"
            dynamic_train_weights = candidate_train_weights
            dynamic_test_weights = candidate_test_weights
    selected_nested = nested_selection["outputs"][nested_selection["selected"]]
    agent_oof = pd.DataFrame(index=train.index)
    support_oof: list[dict[str, float] | None] = [None] * len(train)
    prototype_reference_oof: list[dict[str, Any] | None] = [None] * len(train)
    prototype_minimum_support = int(config.get("prototype_minimum_class_support", 8))
    for fold_index, (train_index, validation_index) in enumerate(split_indices):
        fold_frame = fold_frames[fold_index]
        prototypes = _fit_prototypes(fold_frame.iloc[train_index], state_columns, labels)
        fold_agent, fold_support = _agent_feature_frame(
            fold_frame.iloc[validation_index],
            prototypes,
            labels,
            base_oof[validation_index],
            evidence_quality,
        )
        agent_oof.loc[validation_index, fold_agent.columns] = fold_agent.to_numpy()
        for output_index, support in zip(validation_index, fold_support, strict=True):
            support_oof[int(output_index)] = support
        if labels == ["HC", "MCI", "AD"]:
            fold_values, fold_reliability = _prototype_inputs(fold_frame, state_columns)
            cognitive_prototypes = fit_cognitive_prototypes(
                fold_values[train_index],
                fold_reliability[train_index],
                fold_frame.iloc[train_index]["label"].astype(str).to_numpy(),
                state_columns,
                minimum_class_support=prototype_minimum_support,
            )
            for output_index in validation_index:
                prototype_reference_oof[int(output_index)] = build_case_prototype_reference(
                    cognitive_prototypes,
                    fold_values[int(output_index)],
                    fold_reliability[int(output_index)],
                )
    agent_oof = agent_oof.astype(float)
    prototypes = _fit_prototypes(train, state_columns, labels)
    agent_test, support_test = _agent_feature_frame(
        test, prototypes, labels, base_test, evidence_quality
    )
    prototype_reference_test: list[dict[str, Any] | None] = [None] * len(test)
    if labels == ["HC", "MCI", "AD"]:
        train_prototype_values, train_prototype_reliability = _prototype_inputs(
            train, state_columns
        )
        test_prototype_values, test_prototype_reliability = _prototype_inputs(
            test, state_columns
        )
        cognitive_prototypes = fit_cognitive_prototypes(
            train_prototype_values,
            train_prototype_reliability,
            train["label"].astype(str).to_numpy(),
            state_columns,
            minimum_class_support=prototype_minimum_support,
        )
        prototype_reference_test = [
            build_case_prototype_reference(
                cognitive_prototypes,
                test_prototype_values[index],
                test_prototype_reliability[index],
            )
            for index in range(len(test))
        ]

    correction_train_x = np.column_stack(
        [np.log(np.clip(base_oof, 1e-8, 1.0)), agent_oof.to_numpy(dtype=float)]
    )
    correction_test_x = np.column_stack(
        [np.log(np.clip(base_test, 1e-8, 1.0)), agent_test.to_numpy(dtype=float)]
    )
    correction_c = float(selected_nested["selected_correction_c"])
    correction_model, corrected_test = _fit_fixed_numeric_model(
        correction_train_x,
        correction_test_x,
        y,
        labels,
        correction_c,
        max_iter,
    )
    corrected_oof = selected_nested["corrected_oof"]
    correction_scores = {
        "strict_nested": nested_selection["scores"][nested_selection["selected"]]
    }
    correction_feature_names = [f"log_base_probability__{label}" for label in labels] + list(
        agent_oof.columns
    )
    train_gate = selected_nested["gate_oof"]
    test_gate = evidence_gate(
        agent_test["agent_evidence_coverage"].to_numpy(),
        agent_test["agent_evidence_reliability"].to_numpy(),
        agent_test["agent_confound_burden"].to_numpy(),
    )
    selected_alpha = float(selected_nested["selected_alpha"])
    class_logit_offsets = np.asarray(
        selected_nested["selected_offsets"], dtype=float
    )
    alpha_candidates = {
        "outer_fold_alpha": selected_nested["alpha_by_fold"],
        "aggregation": "median",
    }
    correction_guard = _small_sample_correction_guard(
        y,
        labels,
        split_indices,
        base_oof,
        selected_nested["final_oof"],
        nested_selection["selection_primary"],
        config.get("small_sample_correction_guard", {}),
    )
    if correction_guard["triggered"]:
        selected_alpha = 0.0
        class_logit_offsets = np.zeros(len(labels), dtype=float)
    adjusted_corrected_oof = corrected_oof
    adjusted_corrected_test = _apply_logit_offsets(corrected_test, class_logit_offsets)
    temperature = float(selected_nested["selected_temperature"])
    calibrated_corrected_oof = adjusted_corrected_oof
    calibrated_corrected_test = _temperature_scale(adjusted_corrected_test, temperature)
    blended_oof = (
        base_oof.copy()
        if correction_guard["triggered"]
        else selected_nested["final_oof"]
    )
    final_probability = fuse_corrected_probability(
        base_test, calibrated_corrected_test, test_gate, selected_alpha
    )
    final_temperature = _fit_temperature(blended_oof, y, labels)
    blended_oof = _temperature_scale(blended_oof, final_temperature)
    final_probability = _temperature_scale(final_probability, final_temperature)

    # Keep the historical internal key so the stable report/evaluation templates remain compatible.
    # User-facing reports name this condition B3.
    predictions = _prediction_frame(test, final_probability, "Ours", labels)
    predictions["base_prediction_confidence"] = base_test.max(axis=1)
    predictions["agent_correction_gate"] = test_gate
    predictions["agent_correction_alpha"] = selected_alpha
    predictions["prediction_confidence"] = final_probability.max(axis=1)
    predictions.to_csv(predictions_path, index=False)
    _prediction_frame(test, base_test, "B3_base_supervised", labels).to_csv(
        base_predictions_path, index=False
    )

    # Reserve one deterministic outer-fold validation partition for calibrating
    # the downstream Agent correction. Its prior is strictly out-of-fold; labels
    # are retained in a separate table and are never serialized into workspaces.
    if (
        agent_calibration_predictions_path is not None
        and agent_calibration_workspaces_path is not None
    ):
        calibration_index = np.asarray(split_indices[0][1], dtype=int)
        calibration_frame = train.iloc[calibration_index].copy().reset_index(drop=True)
        calibration_probability = blended_oof[calibration_index]
        _prediction_frame(
            calibration_frame,
            calibration_probability,
            "B3_agent_calibration_prior",
            labels,
        ).to_csv(agent_calibration_predictions_path, index=False)

        fold_frame = fold_frames[0].iloc[calibration_index].copy().reset_index(drop=True)
        calibration_ids = set(calibration_frame["subject_id"].astype(str))
        calibration_cards = cards[
            cards["subject_id"].astype(str).isin(calibration_ids)
        ].copy()
        outer_fit_index = np.asarray(split_indices[0][0], dtype=int)
        outer_fit = train.iloc[outer_fit_index]
        outer_fit_hc_ids = set(
            outer_fit.loc[
                outer_fit["label"].astype(str).eq("HC"), "subject_id"
            ].astype(str)
        )
        calibration_evidence = recalibrate_metric_evidence_frame(
            evidence,
            reference_subject_ids=outer_fit_hc_ids,
            target_subject_ids=calibration_ids,
        )
        state_lookup = fold_frame.set_index("subject_id")
        for row_index, card in calibration_cards.iterrows():
            subject_id = str(card["subject_id"])
            state_id = str(card["state_id"])
            state_column = f"state_{state_id}"
            reliability_column = f"rel_{state_id}"
            if subject_id not in state_lookup.index or state_column not in state_lookup:
                continue
            state_z = float(state_lookup.at[subject_id, state_column])
            reliability = float(state_lookup.at[subject_id, reliability_column])
            calibration_cards.at[row_index, "state_z"] = state_z
            calibration_cards.at[row_index, "raw_state_z"] = state_z
            calibration_cards.at[row_index, "report_state_z"] = state_z
            calibration_cards.at[row_index, "confidence"] = reliability
            calibration_cards.at[row_index, "report_confidence"] = min(
                reliability,
                float(card.get("report_confidence", reliability)),
            )
            calibration_cards.at[row_index, "severity"] = float(
                1.0 / (1.0 + np.exp(-state_z))
            )
            calibration_cards.at[row_index, "category"] = (
                "unreliable"
                if reliability < 0.45
                else "impaired"
                if state_z >= 2.0
                else "borderline"
                if state_z >= 1.0
                else "normal"
            )
        calibration_cards = _replace_card_metric_summaries(
            calibration_cards,
            calibration_evidence,
        )

        calibration_workspaces: list[dict[str, Any]] = []
        for local_index, subject in calibration_frame.iterrows():
            original_index = int(calibration_index[local_index])
            support = support_oof[original_index]
            if support is None:
                support = {label: 1.0 / len(labels) for label in labels}
            workspace = build_case_workspace(
                subject_id=str(subject["subject_id"]),
                base_probabilities={
                    label: float(calibration_probability[local_index, label_index])
                    for label_index, label in enumerate(labels)
                },
                state_cards=calibration_cards,
                metric_evidence=calibration_evidence,
                class_support=support,
                max_supporting_evidence=int(config.get("max_agent_evidence", 8)),
            )
            workspace["workspace_role"] = "agent_correction_calibration"
            workspace["state_calibration"] = "outer_fold_training_reference"
            if prototype_reference_oof[original_index] is not None:
                workspace["cognitive_state_reference"] = prototype_reference_oof[
                    original_index
                ]
            calibration_workspaces.append(workspace)
        with agent_calibration_workspaces_path.open("w", encoding="utf-8") as handle:
            for workspace in calibration_workspaces:
                handle.write(json.dumps(workspace, ensure_ascii=False) + "\n")

    concept_probability = np.asarray(
        [[support[label] for label in labels] for support in support_test], dtype=float
    )
    overall_state_columns = [column for column in state_columns if "__task_" not in column]
    overall_prototypes = _fit_prototypes(train, overall_state_columns, labels)
    overall_agent_test, overall_support_test = _agent_feature_frame(
        test,
        overall_prototypes,
        labels,
        base_test,
        evidence_quality,
    )
    overall_probability = np.asarray(
        [[support[label] for label in labels] for support in overall_support_test],
        dtype=float,
    )
    ablation_frames = [
            _prediction_frame(test, base_test, "Ours_base_supervised", labels),
            _prediction_frame(test, concept_probability, "Ours_concept_only", labels),
            _prediction_frame(
                test,
                overall_probability,
                "Ours_overall_state_only",
                labels,
            ),
            _prediction_frame(
                test,
                calibrated_corrected_test,
                "Ours_unbounded_agent_correction",
                labels,
            ),
        ]
    if matched_encoder_test_probability is not None:
        ablation_frames.append(
            _prediction_frame(
                test,
                matched_encoder_test_probability,
                "Matched_encoder_modal_gate",
                labels,
            )
        )
    pd.concat(ablation_frames, ignore_index=True).to_csv(ablations_path, index=False)

    if dynamic_test_weights is not None:
        expert_contribution = np.asarray(dynamic_test_weights, dtype=float)
        expert_contribution = np.divide(
            expert_contribution,
            expert_contribution.sum(axis=1, keepdims=True),
            out=np.full_like(
                expert_contribution, 1.0 / max(len(expert_names), 1)
            ),
            where=expert_contribution.sum(axis=1, keepdims=True) > 0.0,
        )
    else:
        expert_contribution = _case_level_expert_contributions(
            logistic_base_model,
            base_test_x,
            base_feature_names,
            expert_names,
            labels,
            logistic_base_test,
        )

    contribution = test[IDENTITY].copy()
    for expert_index, name in enumerate(expert_names):
        contribution[f"expert_reliability_{name}"] = test_reliability[:, expert_index]
        contribution[f"expert_{name}_top_probability"] = test_stack[:, expert_index, :].max(axis=1)
        contribution[f"expert_{name}_contribution"] = expert_contribution[:, expert_index]
        if dynamic_test_weights is not None:
            contribution[f"base_gate_weight_{name}"] = dynamic_test_weights[:, expert_index]
    contribution = pd.concat([contribution, agent_test.reset_index(drop=True)], axis=1)
    contribution["agent_correction_gate"] = test_gate
    contribution["agent_correction_alpha"] = selected_alpha
    contribution.to_csv(contributions_path, index=False)

    intervention_rows: list[dict[str, Any]] = []
    discriminability = pd.Series(prototypes["discriminability"])
    class_medians = {
        label: pd.Series(prototypes["class_medians"][label]) for label in labels
    }
    for row_index, subject in test.iterrows():
        predicted_label = labels[int(np.argmax(final_probability[row_index]))]
        true_label = str(subject["label"])
        if predicted_label == true_label:
            continue
        true_reference = class_medians[true_label].reindex(state_columns)
        current = pd.to_numeric(subject[state_columns], errors="coerce")
        priority = (
            (current - true_reference).abs()
            * discriminability.reindex(state_columns).fillna(0.0)
        ).dropna()
        if priority.empty:
            continue
        target_state = str(priority.idxmax())
        changed = test.iloc[[row_index]].copy()
        changed.loc[changed.index[0], target_state] = float(true_reference[target_state])
        changed_agent, _ = _agent_feature_frame(
            changed,
            prototypes,
            labels,
            base_test[row_index : row_index + 1],
            evidence_quality,
        )
        changed_x = np.column_stack(
            [
                np.log(np.clip(base_test[row_index : row_index + 1], 1e-8, 1.0)),
                changed_agent.to_numpy(dtype=float),
            ]
        )
        changed_corrected = _ordered_from_pipeline(
            correction_model, changed_x, labels
        )
        changed_corrected = _temperature_scale(
            _apply_logit_offsets(changed_corrected, class_logit_offsets), temperature
        )
        changed_gate = evidence_gate(
            changed_agent["agent_evidence_coverage"].to_numpy(),
            changed_agent["agent_evidence_reliability"].to_numpy(),
            changed_agent["agent_confound_burden"].to_numpy(),
        )
        changed_probability = fuse_corrected_probability(
            base_test[row_index : row_index + 1],
            changed_corrected,
            changed_gate,
            selected_alpha,
        )
        changed_probability = _temperature_scale(
            changed_probability, final_temperature
        )[0]
        true_index = labels.index(true_label)
        before = float(final_probability[row_index, true_index])
        after = float(changed_probability[true_index])
        intervention_rows.append(
            {
                "subject_id": str(subject["subject_id"]),
                "label": true_label,
                "prediction_before": predicted_label,
                "intervened_state": target_state,
                "original_state_z": float(current[target_state]),
                "corrected_state_z": float(true_reference[target_state]),
                "correction_source": "training-fold true-class state prototype",
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
        "original_state_z",
        "corrected_state_z",
        "correction_source",
        "true_class_probability_before",
        "true_class_probability_after",
        "true_class_probability_change",
        "monotonic_improvement",
    ]
    pd.DataFrame(intervention_rows, columns=intervention_columns).to_csv(
        interventions_path, index=False
    )

    workspace_rows: list[dict[str, Any]] = []
    for row_index, subject in test.iterrows():
        base_probabilities = {
            label: float(base_test[row_index, label_index])
            for label_index, label in enumerate(labels)
        }
        workspace = build_case_workspace(
            subject_id=str(subject["subject_id"]),
            base_probabilities=base_probabilities,
            state_cards=cards,
            metric_evidence=evidence,
            class_support=support_test[row_index],
            max_supporting_evidence=int(config.get("max_agent_evidence", 8)),
        )
        if prototype_reference_test[row_index] is not None:
            workspace["cognitive_state_reference"] = prototype_reference_test[row_index]
        workspace["corrected_probabilities"] = {
            label: float(calibrated_corrected_test[row_index, label_index])
            for label_index, label in enumerate(labels)
        }
        workspace["final_probabilities"] = {
            label: float(final_probability[row_index, label_index])
            for label_index, label in enumerate(labels)
        }
        workspace["final_prediction"] = labels[int(np.argmax(final_probability[row_index]))]
        workspace["correction_alpha"] = selected_alpha
        workspace_rows.append(workspace)
    with workspaces_path.open("w", encoding="utf-8") as handle:
        for workspace in workspace_rows:
            handle.write(json.dumps(workspace, ensure_ascii=False) + "\n")

    model_bundle = {
        "labels": labels,
        "expert_names": expert_names,
        "expert_models": expert_models,
        "base_model": base_model,
        "logistic_base_model": logistic_base_model,
        "dynamic_gate_model": dynamic_gate_bundle,
        "base_architecture": base_architecture,
        "prototypes": prototypes,
        "correction_model": correction_model,
        "selected_alpha": selected_alpha,
        "class_logit_offsets": class_logit_offsets,
        "temperature": temperature,
        "final_probability_temperature": final_temperature,
        "agent_feature_columns": list(agent_test.columns),
    }
    joblib.dump(model_bundle, model_path)
    base_oof_score = _selection_score(y, base_oof, labels)
    final_oof_score = _selection_score(y, blended_oof, labels)
    json_dump(
        {
            "schema_version": "condition-c-evidence-agent-v3-nested",
            "condition": "B3",
            "agent_type": "single_stateful_evidence_diagnostic_agent",
            "supervised_modules": [
                "multibranch_base_predictor",
                "bounded_risk_correction_and_temperature_calibration",
            ],
            "labels": labels,
            "folds": folds,
            "expert_names": expert_names,
            "model_only_experts": [
                name
                for name in expert_names
                if name
                in {
                    "auxiliary_acoustic",
                    "traditional_acoustic_prior",
                    "demographic_context",
                    "multilingual_dense_text",
                    "multilingual_dense_audio",
                    "multilingual_segment_audio",
                }
            ],
            "expert_cv": expert_cv,
            "base_architecture": base_architecture,
            "dynamic_gate": dynamic_gate_metadata,
            "qc_shortcut_guard": qc_guard_metadata,
            "selection_protocol": nested_selection["protocol"],
            "outer_fold_selection": {
                "base_c": selected_nested["base_c_by_fold"],
                "correction_c": selected_nested["correction_c_by_fold"],
                "correction_alpha": selected_nested["alpha_by_fold"],
                "class_logit_offsets": selected_nested["offsets_by_fold"],
                "correction_temperature": selected_nested[
                    "temperature_by_fold"
                ],
            },
            "offset_stability_audit": selected_nested[
                "offset_stability_audit"
            ],
            "deep_text_encoder": deep_text_metadata,
            "deep_audio_encoder": deep_audio_metadata,
            "matched_encoder_isolation": matched_encoder_metadata,
            "base_standardized_feature_importance": (
                _standardized_feature_importance(logistic_base_model, base_feature_names)
                if base_architecture == "multinomial_logistic_stacking"
                else {}
            ),
            "base_selected_c": base_c,
            "base_cv": base_scores,
            "correction_selected_c": correction_c,
            "correction_cv": correction_scores,
            "correction_standardized_feature_importance": _standardized_feature_importance(
                correction_model, correction_feature_names
            ),
            "top_state_discriminability": dict(
                sorted(
                    prototypes["discriminability"].items(),
                    key=lambda item: float(item[1]),
                    reverse=True,
                )[:20]
            ),
            "alpha_cv": alpha_candidates,
            "small_sample_correction_guard": correction_guard,
            "correction_stability_guard": correction_guard,
            "selected_alpha": selected_alpha,
            "class_logit_offsets": {
                label: float(value)
                for label, value in zip(labels, class_logit_offsets, strict=True)
            },
            "temperature": temperature,
            "final_probability_temperature": final_temperature,
            "base_oof_macro_f1": base_oof_score[0],
            "base_oof_macro_auroc": base_oof_score[1],
            "final_oof_macro_f1": final_oof_score[0],
            "final_oof_macro_auroc": final_oof_score[1],
            "test_labels_used_by_agent": False,
            "agent_workspace_count": len(workspace_rows),
        },
        metadata_path,
    )
