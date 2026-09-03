from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
)

from .utils import json_dump
from .agent_runtime import case_pseudonym


EVALUATION_SCHEMA_VERSION = "2026-08-21.4"

REPORTABLE_EVIDENCE_ROLES = {
    "clinical",
    "clinical_support",
    "cautious_support",
    "model_and_report",
}


def _ordered_log_loss(
    y_true: np.ndarray | pd.Series,
    probability: np.ndarray,
    labels: list[str],
) -> float:
    """Compute log loss without allowing sklearn to reorder probability columns."""
    encoded = np.asarray([labels.index(str(value)) for value in y_true], dtype=int)
    return float(log_loss(encoded, probability, labels=list(range(len(labels)))))


def _available_state_cards(cards: pd.DataFrame) -> pd.DataFrame:
    """Exclude states that were declared unavailable for the source task."""
    if "missing_fraction" not in cards:
        return cards.copy()
    missing_fraction = pd.to_numeric(cards["missing_fraction"], errors="coerce")
    return cards[missing_fraction.fillna(1.0).lt(1.0)].copy()


def _has_trace_items(value: Any) -> bool:
    if pd.isna(value):
        return False
    try:
        return len(json.loads(str(value))) > 0
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def _trace_presence_rate(cards: pd.DataFrame) -> float:
    if cards.empty:
        return np.nan
    return float(cards["evidence_segments"].map(_has_trace_items).mean())


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def _report_subject_mask(frame: pd.DataFrame, report: pd.Series) -> pd.Series:
    """Match current pseudonymous reports and legacy internal test fixtures."""
    if "case_id" in report.index and str(report.get("case_id", "")):
        return frame["subject_id"].astype(str).map(case_pseudonym).eq(str(report["case_id"]))
    return frame["subject_id"].astype(str).eq(str(report.get("subject_id", "")))


def _report_permission_rate(
    reports: pd.DataFrame,
    evidence: pd.DataFrame,
    cards: pd.DataFrame | None = None,
) -> float:
    """Verify every report citation against the subject-level evidence registry."""

    if reports.empty:
        return np.nan
    valid = 0
    total = 0
    for _, report in reports.iterrows():
        try:
            payload = json.loads(str(report.get("evidence", "{}")))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        identifiers = [
            str(identifier)
            for key in ("used_evidence_ids", "counterevidence_ids")
            for identifier in payload.get(key, [])
        ]
        subject = evidence[_report_subject_mask(evidence, report)]
        subject_cards = (
            cards[_report_subject_mask(cards, report)]
            if cards is not None and not cards.empty
            else pd.DataFrame()
        )
        for identifier in identifiers:
            kind, raw_identifier = (
                identifier.split(":", 1)
                if ":" in identifier
                and identifier.split(":", 1)[0] in {"state", "metric", "segment", "qc"}
                else ("", identifier)
            )
            if kind == "qc":
                total += 1
                continue
            if kind == "segment" or raw_identifier.startswith("SEG-"):
                # Segment aliases are audited against the source span registry separately.
                continue
            total += 1
            if kind in {"", "state"} and not subject_cards.empty:
                state_matches = subject_cards[
                    subject_cards["state_id"].astype(str).eq(raw_identifier)
                ]
                if not state_matches.empty:
                    state_allowed = (
                        state_matches["report_permission"].map(_truthy).all()
                        and pd.to_numeric(
                            state_matches["report_confidence"], errors="coerce"
                        ).fillna(0.0).gt(0.0).all()
                        and pd.to_numeric(
                            state_matches["missing_fraction"], errors="coerce"
                        ).fillna(1.0).lt(1.0).all()
                    )
                    valid += int(state_allowed)
                    continue
            matches = subject[
                subject["metric_instance_id"].astype(str).eq(raw_identifier)
            ]
            if matches.empty:
                fallback = subject[
                    subject["metric_id"].astype(str).eq(raw_identifier)
                ]
                if fallback["task_scope"].astype(str).nunique() != 1:
                    continue
                matches = fallback
            allowed = (
                not matches.empty
                and matches["report_permission"].map(_truthy).all()
                and matches["evidence_role"].astype(str).isin(
                    REPORTABLE_EVIDENCE_ROLES
                ).all()
                and ~matches["missing"].map(_truthy).any()
            )
            valid += int(allowed)
    return float(valid / total) if total else 0.0


def _quality_reference_rate(
    reports: pd.DataFrame,
    evidence: pd.DataFrame,
) -> float:
    """Ensure QC evidence is isolated in the report's quality-only citation field."""

    if reports.empty:
        return np.nan
    valid = 0
    total = 0
    for _, report in reports.iterrows():
        try:
            payload = json.loads(str(report.get("evidence", "{}")))
        except (TypeError, ValueError, json.JSONDecodeError):
            payload = {}
        subject = evidence[_report_subject_mask(evidence, report)]
        for identifier in payload.get("quality_evidence_ids", []):
            total += 1
            identifier = str(identifier)
            kind, raw_identifier = (
                identifier.split(":", 1)
                if ":" in identifier
                and identifier.split(":", 1)[0] in {"state", "metric", "segment", "qc"}
                else ("", identifier)
            )
            matches = subject[
                subject["metric_instance_id"].astype(str).eq(raw_identifier)
            ]
            allowed = (
                kind in {"", "qc"}
                and not matches.empty
                and matches["evidence_role"].astype(str).isin(
                    {"qc", "qc_only", "quality_control"}
                ).all()
                and ~matches["report_permission"].map(_truthy).any()
            )
            valid += int(allowed)
    return float(valid / total) if total else 1.0


def _segment_faithfulness_rate(
    cards: pd.DataFrame,
    segments: pd.DataFrame,
) -> float:
    """Check that every reported local span exists and matches the extracted segment."""

    if segments.empty:
        return np.nan
    traceable = _available_state_cards(
        cards[
            cards["state_base_id"].isin({"S01", "S02", "S03"})
            & cards["trace_resolution"].astype(str).str.contains("segment")
        ]
    )
    if traceable.empty:
        return np.nan
    segment_index = segments.copy()
    segment_index["segment_id"] = segment_index["segment_id"].astype(str)
    expected_basis = {
        "S01": "highest_silence_fraction",
        "S02": "lowest_voiced_fraction",
        "S03": "high_activity_switching_and_short_voiced_runs",
    }
    valid = 0
    total = 0
    for _, card in traceable.iterrows():
        for item in _json_list(card["evidence_segments"]):
            total += 1
            if not isinstance(item, dict):
                continue
            source = segment_index[
                segment_index["segment_id"].eq(str(item.get("segment_id", "")))
            ]
            if len(source) != 1:
                continue
            source_row = source.iloc[0]
            spans = _json_list(item.get("source_spans"))
            if not spans:
                spans = _json_list(source_row.get("source_spans"))
            same_bounds = np.isclose(
                float(item.get("start_sec", np.nan)),
                float(source_row["start_sec"]),
            ) and np.isclose(
                float(item.get("end_sec", np.nan)),
                float(source_row["end_sec"]),
            )
            same_case = str(item.get("case_id", "")) == str(source_row["case_id"])
            basis_ok = item.get("selection_basis") == expected_basis.get(
                str(card["state_base_id"])
            )
            valid += int(bool(same_bounds and same_case and spans and basis_ok))
    return float(valid / total) if total else 0.0


def _labels(config: dict[str, Any]) -> list[str]:
    labels = [str(label) for label in config.get("labels", [])]
    if len(labels) < 2:
        raise ValueError("Evaluation requires at least two configured labels.")
    return labels


def _one_hot(y: np.ndarray, labels: list[str]) -> np.ndarray:
    index = {label: position for position, label in enumerate(labels)}
    output = np.zeros((len(y), len(labels)), dtype=float)
    for row, value in enumerate(y):
        output[row, index[str(value)]] = 1.0
    return output


def expected_calibration_error(
    y: np.ndarray,
    probability: np.ndarray,
    bins: int,
    labels: list[str],
) -> float:
    predicted = np.argmax(probability, axis=1)
    truth = np.array([labels.index(str(value)) for value in y])
    confidence = probability.max(axis=1)
    correct = predicted == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            score += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return score


def _screening_operating_point(
    y: np.ndarray,
    probability: np.ndarray,
    labels: list[str],
    positive_class: str,
    threshold: float = 0.5,
) -> dict[str, float]:
    if len(labels) != 2 or positive_class not in labels:
        return {}
    positive = (y == positive_class).astype(int)
    score = probability[:, labels.index(positive_class)]
    threshold = float(np.clip(threshold, 0.0, 1.0))
    predicted = score >= threshold
    tp = int(((predicted == 1) & (positive == 1)).sum())
    fn = int(((predicted == 0) & (positive == 1)).sum())
    fp = int(((predicted == 1) & (positive == 0)).sum())
    tn = int(((predicted == 0) & (positive == 0)).sum())
    return {
        "screening_threshold": threshold,
        "sensitivity_at_locked_threshold": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity_at_locked_threshold": float(tn / (tn + fp)) if tn + fp else np.nan,
        "ppv_at_locked_threshold": float(tp / (tp + fp)) if tp + fp else np.nan,
        "npv_at_locked_threshold": float(tn / (tn + fn)) if tn + fn else np.nan,
    }


def _binary_calibration(
    y: np.ndarray,
    probability: np.ndarray,
    labels: list[str],
    positive_class: str,
) -> dict[str, float]:
    if len(labels) != 2:
        return {}
    target = (y == positive_class).astype(int)
    score = np.clip(probability[:, labels.index(positive_class)], 1e-5, 1 - 1e-5)
    logit = np.log(score / (1 - score)).reshape(-1, 1)
    model = LogisticRegression(C=1e6, solver="lbfgs").fit(logit, target)
    return {
        "calibration_intercept": float(model.intercept_[0]),
        "calibration_slope": float(model.coef_[0, 0]),
    }


def _multiclass_referral_metrics(
    y: np.ndarray,
    probability: np.ndarray,
    labels: list[str],
    reference_class: str,
) -> dict[str, float]:
    """Collapse multiclass severity labels into reference vs clinical referral."""

    if len(labels) <= 2 or reference_class not in labels:
        return {}
    referral_truth = (y != reference_class).astype(int)
    referral_score = 1.0 - probability[:, labels.index(reference_class)]
    if np.unique(referral_truth).size < 2:
        return {}
    binary_labels = ["no_referral", "referral"]
    binary_probability = np.column_stack([1.0 - referral_score, referral_score])
    operating_point = _screening_operating_point(
        np.where(referral_truth == 1, "referral", "no_referral"),
        binary_probability,
        binary_labels,
        "referral",
    )
    renamed = {
        f"referral_{key}": value for key, value in operating_point.items()
    }
    renamed.update(
        {
            "referral_auroc": float(roc_auc_score(referral_truth, referral_score)),
            "referral_auprc": float(
                average_precision_score(referral_truth, referral_score)
            ),
            "referral_brier": float(
                np.mean((referral_score - referral_truth) ** 2)
            ),
        }
    )
    return renamed


def evaluate_predictions(
    frame: pd.DataFrame,
    bins: int,
    labels: list[str],
    positive_class: str,
) -> dict[str, Any]:
    y = frame["label"].astype(str).to_numpy()
    predicted = frame["predicted_label"].astype(str).to_numpy()
    probability = frame[[f"prob_{label}" for label in labels]].to_numpy(dtype=float)
    binary = _one_hot(y, labels)
    matrix = confusion_matrix(y, predicted, labels=labels)
    confidence = probability.max(axis=1)
    sorted_probability = np.sort(probability, axis=1)
    probability_margin = sorted_probability[:, -1] - sorted_probability[:, -2]
    error = predicted != y
    per_class = {}
    class_auroc = []
    class_auprc = []
    for index, label in enumerate(labels):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        target = binary[:, index]
        auroc = float(roc_auc_score(target, probability[:, index])) if np.unique(target).size == 2 else np.nan
        auprc = float(average_precision_score(target, probability[:, index])) if np.unique(target).size == 2 else np.nan
        class_auroc.append(auroc)
        class_auprc.append(auprc)
        per_class[label] = {
            "precision": float(tp / (tp + fp)) if tp + fp else np.nan,
            "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "f1": float(2 * tp / (2 * tp + fp + fn)) if 2 * tp + fp + fn else np.nan,
            "auroc": auroc,
            "auprc": auprc,
            "support": int(tp + fn),
        }
    supports = binary.sum(axis=0)
    result = {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, labels=labels, average="macro", zero_division=0)),
        "micro_f1": float(f1_score(y, predicted, labels=labels, average="micro", zero_division=0)),
        "weighted_f1": float(f1_score(y, predicted, labels=labels, average="weighted", zero_division=0)),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "macro_auroc_ovr": float(np.nanmean(class_auroc)),
        "micro_auroc_ovr": float(roc_auc_score(binary, probability, average="micro")),
        "weighted_auroc_ovr": float(np.nansum(np.asarray(class_auroc) * supports) / max(supports.sum(), 1)),
        "macro_auprc": float(np.nanmean(class_auprc)),
        "micro_auprc": float(average_precision_score(binary, probability, average="micro")),
        "weighted_auprc": float(average_precision_score(binary, probability, average="weighted")),
        "multiclass_brier": float(np.mean(np.sum((probability - binary) ** 2, axis=1))),
        "log_loss": _ordered_log_loss(y, probability, labels),
        "ece": expected_calibration_error(y, probability, bins, labels),
        "error_rate": float(error.mean()),
        "near_threshold_uncertainty_rate": float((probability_margin <= 0.10).mean()),
        "high_confidence_error_rate": float((error & (confidence >= 0.80)).mean()),
        "macro_fnr": float(np.nanmean([1.0 - values["sensitivity"] for values in per_class.values()])),
        "macro_fpr": float(np.nanmean([1.0 - values["specificity"] for values in per_class.values()])),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }
    if len(labels) == 2 and positive_class in labels:
        positive_index = labels.index(positive_class)
        tp = int(matrix[positive_index, positive_index])
        fn = int(matrix[positive_index, :].sum() - tp)
        fp = int(matrix[:, positive_index].sum() - tp)
        tn = int(matrix.sum() - tp - fn - fp)
        result.update(
            {
                "tp": tp,
                "tn": tn,
                "fp": fp,
                "fn": fn,
                "positive_prevalence": float((y == positive_class).mean()),
            }
        )
    locked_threshold = (
        float(
            pd.to_numeric(frame["screening_threshold"], errors="coerce")
            .dropna()
            .median()
        )
        if "screening_threshold" in frame
        and frame["screening_threshold"].notna().any()
        else 0.5
    )
    result.update(
        _screening_operating_point(
            y, probability, labels, positive_class, locked_threshold
        )
    )
    result.update(_binary_calibration(y, probability, labels, positive_class))
    result.update(_multiclass_referral_metrics(y, probability, labels, labels[0]))
    return result


def bootstrap_intervals(
    frame: pd.DataFrame,
    bins: int,
    iterations: int,
    labels: list[str],
    positive_class: str,
    seed: int = 20260813,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "micro_f1",
        "weighted_f1",
        "macro_auroc_ovr",
        "micro_auroc_ovr",
        "weighted_auroc_ovr",
        "macro_auprc",
        "micro_auprc",
        "weighted_auprc",
        "mcc",
        "multiclass_brier",
        "log_loss",
        "ece",
        "calibration_intercept",
        "calibration_slope",
        "high_confidence_error_rate",
        "referral_auroc",
        "referral_auprc",
        "referral_specificity_at_locked_threshold",
    ]
    values = {metric: [] for metric in metrics}
    attempts = 0
    while len(values[metrics[0]]) < iterations and attempts < iterations * 8:
        attempts += 1
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        if sample["label"].nunique() < len(labels):
            continue
        y = sample["label"].astype(str).to_numpy()
        predicted = sample["predicted_label"].astype(str).to_numpy()
        probability = sample[
            [f"prob_{label}" for label in labels]
        ].to_numpy(dtype=float)
        binary = _one_hot(y, labels)
        class_auroc = [
            float(roc_auc_score(binary[:, index], probability[:, index]))
            for index in range(len(labels))
        ]
        class_auprc = [
            float(average_precision_score(binary[:, index], probability[:, index]))
            for index in range(len(labels))
        ]
        supports = binary.sum(axis=0)
        result = {
            "accuracy": float(accuracy_score(y, predicted)),
            "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
            "macro_f1": float(
                f1_score(
                    y,
                    predicted,
                    labels=labels,
                    average="macro",
                    zero_division=0,
                )
            ),
            "micro_f1": float(
                f1_score(y, predicted, labels=labels, average="micro", zero_division=0)
            ),
            "weighted_f1": float(
                f1_score(y, predicted, labels=labels, average="weighted", zero_division=0)
            ),
            "macro_auroc_ovr": float(np.mean(class_auroc)),
            "micro_auroc_ovr": float(
                roc_auc_score(binary, probability, average="micro")
            ),
            "weighted_auroc_ovr": float(
                np.sum(np.asarray(class_auroc) * supports) / max(supports.sum(), 1)
            ),
            "macro_auprc": float(np.mean(class_auprc)),
            "micro_auprc": float(
                average_precision_score(binary, probability, average="micro")
            ),
            "weighted_auprc": float(
                average_precision_score(binary, probability, average="weighted")
            ),
        }
        # Reuse the exact primary evaluator so discrimination, calibration and
        # operating-point confidence intervals cannot drift from point estimates.
        result = evaluate_predictions(sample, bins, labels, positive_class)
        for metric in metrics:
            if metric in result and np.isfinite(result[metric]):
                values[metric].append(result[metric])
    intervals = {
        metric: [float(np.percentile(series, 2.5)), float(np.percentile(series, 97.5))]
        for metric, series in values.items()
        if series
    }
    effective = len(values[metrics[0]])
    intervals["__effective_resamples__"] = [float(effective), float(effective)]
    intervals["__attempted_resamples__"] = [float(attempts), float(attempts)]
    return intervals


def paired_prediction_comparison(
    ours: pd.DataFrame,
    baseline: pd.DataFrame,
    bins: int,
    iterations: int,
    labels: list[str],
    positive_class: str,
    seed: int = 20260827,
) -> dict[str, Any]:
    """Compare two conditions on identical subjects using paired resampling."""
    required = {"subject_id", "label", "predicted_label"} | {
        f"prob_{label}" for label in labels
    }
    if not required.issubset(ours) or not required.issubset(baseline):
        return {"status": "not_available", "reason": "required prediction columns missing"}
    left = ours[list(required)].copy()
    right = baseline[list(required)].copy()
    if left["subject_id"].duplicated().any() or right["subject_id"].duplicated().any():
        return {"status": "not_available", "reason": "subject_id is not unique"}
    ours_subjects = set(left["subject_id"].astype(str))
    baseline_subjects = set(right["subject_id"].astype(str))
    if ours_subjects != baseline_subjects:
        return {
            "status": "invalid",
            "reason": "condition subject cohorts differ; paired analysis cannot use a silent intersection",
            "ours_only_subjects": sorted(ours_subjects - baseline_subjects),
            "baseline_only_subjects": sorted(baseline_subjects - ours_subjects),
        }
    matched = left.merge(
        right,
        on="subject_id",
        suffixes=("_ours", "_baseline"),
        validate="one_to_one",
    )
    if matched.empty:
        return {"status": "not_available", "reason": "no matched subjects"}
    if not matched["label_ours"].astype(str).equals(
        matched["label_baseline"].astype(str)
    ):
        return {"status": "invalid", "reason": "ground-truth labels differ between conditions"}

    def condition_frame(sample: pd.DataFrame, suffix: str) -> pd.DataFrame:
        columns: dict[str, Any] = {
            "subject_id": sample["subject_id"].to_numpy(),
            "label": sample["label_ours"].to_numpy(),
            "predicted_label": sample[f"predicted_label_{suffix}"].to_numpy(),
        }
        for label in labels:
            columns[f"prob_{label}"] = sample[f"prob_{label}_{suffix}"].to_numpy()
        return pd.DataFrame(columns)

    ours_frame = condition_frame(matched, "ours")
    baseline_frame = condition_frame(matched, "baseline")
    ours_result = evaluate_predictions(ours_frame, bins, labels, positive_class)
    baseline_result = evaluate_predictions(baseline_frame, bins, labels, positive_class)
    metrics = [
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "micro_f1",
        "macro_auroc_ovr",
        "micro_auroc_ovr",
        "macro_auprc",
        "micro_auprc",
    ]
    deltas = {
        metric: float(ours_result[metric] - baseline_result[metric])
        for metric in metrics
    }
    rng = np.random.default_rng(seed)
    bootstrap = {metric: [] for metric in metrics}
    attempts = 0
    while len(bootstrap[metrics[0]]) < iterations and attempts < iterations * 8:
        attempts += 1
        sample = matched.iloc[rng.integers(0, len(matched), len(matched))]
        if sample["label_ours"].nunique() < len(labels):
            continue
        sample_ours = evaluate_predictions(
            condition_frame(sample, "ours"), bins, labels, positive_class
        )
        sample_baseline = evaluate_predictions(
            condition_frame(sample, "baseline"), bins, labels, positive_class
        )
        for metric in metrics:
            delta = float(sample_ours[metric] - sample_baseline[metric])
            if np.isfinite(delta):
                bootstrap[metric].append(delta)
    intervals = {
        metric: [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]
        for metric, values in bootstrap.items()
        if values
    }
    truth = matched["label_ours"].astype(str)
    ours_correct = matched["predicted_label_ours"].astype(str).eq(truth)
    baseline_correct = matched["predicted_label_baseline"].astype(str).eq(truth)
    ours_only = int((ours_correct & ~baseline_correct).sum())
    baseline_only = int((~ours_correct & baseline_correct).sum())
    discordant = ours_only + baseline_only
    mcnemar_p = (
        float(binomtest(min(ours_only, baseline_only), discordant, 0.5).pvalue)
        if discordant
        else 1.0
    )
    return {
        "status": "completed",
        "n_matched": int(len(matched)),
        "delta_ours_minus_baseline": deltas,
        "paired_bootstrap_95_ci": intervals,
        "bootstrap_effective_resamples": int(len(bootstrap[metrics[0]])),
        "bootstrap_attempted_resamples": int(attempts),
        "mcnemar_exact": {
            "ours_correct_baseline_wrong": ours_only,
            "ours_wrong_baseline_correct": baseline_only,
            "discordant_pairs": discordant,
            "p_value": mcnemar_p,
        },
    }


def _report_rubric(reports: pd.DataFrame, condition: str) -> dict[str, float]:
    names = [
        "evidence_completeness",
        "clinical_interpretability",
        "safety_calibration",
        "diagnostic_usefulness",
        "traceability",
    ]
    if reports.empty:
        return {**{name: np.nan for name in names}, "total_25": np.nan}
    scores = []
    for row in reports.fillna("").to_dict("records"):
        text = row.get("report_zh", "")
        uncertainty = row.get("uncertainty_zh", "")
        evidence = row.get("evidence", "")
        evidence_score = min(5.0, 2.0 + (1.5 if evidence and evidence != "[]" else 0.0) + (1.5 if len(text) >= 120 else 0.0))
        interpretability = min(5.0, 2.0 + sum(token in text for token in ["停顿", "输出", "连续", "词", "语义"]))
        safety = min(5.0, 1.5 + 1.5 * bool(uncertainty) + sum(token in text for token in ["筛查", "不能", "不确定", "限制"]))
        usefulness = min(5.0, 1.5 + sum(token in text for token in ["复核", "量表", "评估", "随访", "转诊"]))
        traceability = min(
            5.0,
            1.0
            + (1.5 if evidence and evidence != "[]" else 0.0)
            + sum(token in text for token in ["片段", "指标", "状态", "证据"]),
        )
        scores.append([evidence_score, interpretability, safety, usefulness, traceability])
    mean = np.mean(scores, axis=0)
    return {**{name: float(mean[index]) for index, name in enumerate(names)}, "total_25": float(mean.sum())}


def build_layer_b(
    evidence_path: Path,
    state_cards_path: Path,
    segments_path: Path,
    contributions_path: Path,
    interventions_path: Path,
    ablations_path: Path,
    ours_predictions_path: Path,
    b2_reports_path: Path,
    ours_reports_path: Path,
    report_scores_path: Path,
    cognitive_agent_audit_path: Path,
    cognitive_agent_status_path: Path,
    bins: int,
    labels: list[str],
    positive_class: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence = pd.read_csv(evidence_path, dtype={"subject_id": str})
    cards = pd.read_csv(state_cards_path, dtype={"subject_id": str})
    segments = pd.read_csv(segments_path, dtype={"case_id": str, "segment_id": str})
    contributions = pd.read_csv(contributions_path, dtype={"subject_id": str})
    interventions = pd.read_csv(interventions_path, dtype={"subject_id": str})
    ablations = pd.read_csv(ablations_path, dtype={"subject_id": str})
    ours = pd.read_csv(ours_predictions_path, dtype={"subject_id": str})
    agent_off_path = ours_predictions_path.with_name("b3_supervised_predictions.csv")
    agent_off = (
        pd.read_csv(agent_off_path, dtype={"subject_id": str})
        if agent_off_path.exists()
        else pd.DataFrame()
    )
    b2_reports = pd.read_csv(b2_reports_path, dtype={"case_id": str})
    ours_reports = pd.read_csv(ours_reports_path, dtype={"case_id": str})
    report_scores = pd.read_csv(report_scores_path, dtype={"case_id": str})
    agent_audits = [
        json.loads(line)
        for line in cognitive_agent_audit_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ] if cognitive_agent_audit_path.exists() else []
    agent_status = (
        json.loads(cognitive_agent_status_path.read_text(encoding="utf-8"))
        if cognitive_agent_status_path.exists()
        else {}
    )
    calibration_path = cognitive_agent_status_path.with_name(
        "agent_correction_calibration.json"
    )
    agent_calibration = (
        json.loads(calibration_path.read_text(encoding="utf-8"))
        if calibration_path.exists()
        else {}
    )

    required_evidence = [
        "metric_id",
        "metric_instance_id",
        "task_scope",
        "reference_label",
        "reference_median",
        "reference_scale",
        "direction",
        "evidence_role",
        "reliability",
        "missing",
        "confound_tags",
        "report_permission",
    ]
    evidence_field_complete = evidence[required_evidence].notna()
    evidence_field_complete["value_or_declared_missing"] = (
        evidence["value"].notna() | evidence["missing"].astype(bool)
    )
    metric_completeness = float(evidence_field_complete.mean().mean())
    required_state = [
        "state_id",
        "state_base_id",
        "task_scope",
        "state_z",
        "category",
        "confidence",
        "supporting_metrics",
        "counter_evidence",
        "evidence_segments",
        "trace_resolution",
        "report_state_z",
        "report_confidence",
        "report_permission",
    ]
    state_completeness = float(cards[required_state].notna().mean().mean())
    task_cards = _available_state_cards(
        cards[~cards["task_scope"].fillna("overall").eq("overall")]
    )
    if task_cards.empty:
        task_metric_trace = np.nan
        task_segment_trace = np.nan
    else:
        task_metric_trace = float(
            task_cards.apply(
                lambda row: all(
                    item.get("task_scope") == row["task_scope"]
                    for item in json.loads(row["supporting_metrics"])
                ),
                axis=1,
            ).mean()
        )
        segment_capable = task_cards[
            task_cards["trace_resolution"].eq("task_and_segment")
        ]
        task_segment_trace = _trace_presence_rate(segment_capable)
    contribution_columns = [
        column for column in contributions if column.endswith("_contribution")
    ]
    if contribution_columns:
        normalized_contribution = contributions[contribution_columns].apply(
            pd.to_numeric, errors="coerce"
        )
        valid_contribution = (
            normalized_contribution.notna().all(axis=1)
            & normalized_contribution.ge(0.0).all(axis=1)
            & np.isclose(
                normalized_contribution.sum(axis=1).to_numpy(dtype=float),
                1.0,
                atol=1e-6,
            )
        )
        branch_trace = float(valid_contribution.mean())
    else:
        branch_trace = 0.0
    permission = _report_permission_rate(ours_reports, evidence, cards)
    quality_separation = _quality_reference_rate(ours_reports, evidence)
    span_faithfulness = _segment_faithfulness_rate(cards, segments)
    intervention_rate = float(interventions["monotonic_improvement"].astype(bool).mean()) if len(interventions) else np.nan
    concept = ablations[ablations["condition"].eq("Ours_concept_only")]
    overall_state_only = ablations[
        ablations["condition"].eq("Ours_overall_state_only")
    ]
    ours_auc = evaluate_predictions(ours, bins, labels, positive_class)["macro_auroc_ovr"]
    concept_auc = evaluate_predictions(concept, bins, labels, positive_class)["macro_auroc_ovr"] if len(concept) else np.nan
    overall_state_auc = (
        evaluate_predictions(overall_state_only, bins, labels, positive_class)["macro_auroc_ovr"]
        if len(overall_state_only)
        else np.nan
    )
    ablation_gain = float(ours_auc - concept_auc) if np.isfinite(concept_auc) else np.nan
    task_state_gain = (
        float(concept_auc - overall_state_auc)
        if np.isfinite(concept_auc) and np.isfinite(overall_state_auc)
        else np.nan
    )
    matched_report_cases = set(b2_reports.get("case_id", pd.Series(dtype=str)).astype(str)) & set(
        ours_reports.get("case_id", pd.Series(dtype=str)).astype(str)
    )
    if matched_report_cases:
        b2_scored = b2_reports[b2_reports["case_id"].astype(str).isin(matched_report_cases)]
        ours_scored = ours_reports[ours_reports["case_id"].astype(str).isin(matched_report_cases)]
    else:
        b2_scored = b2_reports.iloc[0:0].copy()
        ours_scored = ours_reports.iloc[0:0].copy()
    paired_report_scores = report_scores
    if matched_report_cases and "case_id" in paired_report_scores:
        paired_report_scores = paired_report_scores[
            paired_report_scores["case_id"].astype(str).isin(matched_report_cases)
        ]
    elif not matched_report_cases:
        paired_report_scores = report_scores.iloc[0:0].copy()
    if not paired_report_scores.empty:
        grouped_scores = paired_report_scores.groupby("condition")[[
            "evidence_completeness",
            "clinical_interpretability",
            "safety_calibration",
            "diagnostic_usefulness",
            "traceability",
            "total_25",
        ]].mean()
        b2_rubric = grouped_scores.loc["B2"].to_dict() if "B2" in grouped_scores.index else _report_rubric(b2_scored, "B2")
        ours_rubric = grouped_scores.loc["Ours"].to_dict() if "Ours" in grouped_scores.index else _report_rubric(ours_scored, "Ours")
        report_rater = "paired blinded agent rater"
    else:
        b2_rubric = _report_rubric(b2_scored, "B2")
        ours_rubric = _report_rubric(ours_scored, "Ours")
        report_rater = (
            "paired deterministic structural completeness audit"
            if matched_report_cases
            else "not comparable: no matched B2/Ours report cases"
        )
    empty_card_value = pd.Series(np.nan, index=cards.index, dtype=float)
    raw_state_z = pd.to_numeric(cards.get("raw_state_z", empty_card_value), errors="coerce")
    finite_raw_state_z = raw_state_z[np.isfinite(raw_state_z)]
    extreme_state_rate = (
        float(finite_raw_state_z.abs().gt(20.0).mean())
        if len(finite_raw_state_z)
        else np.nan
    )
    state_clip = pd.to_numeric(
        cards.get("metric_contribution_clip_z", empty_card_value), errors="coerce"
    )
    state_z = pd.to_numeric(cards.get("state_z", empty_card_value), errors="coerce")
    saturation_mask = state_z.notna() & state_clip.notna() & np.isclose(
        state_z.abs().to_numpy(dtype=float), state_clip.to_numpy(dtype=float), atol=1e-8
    )
    state_saturation_rate = float(saturation_mask.mean()) if len(cards) else np.nan
    fallback_report_rate = (
        float(ours_reports.get("validation_status", pd.Series(dtype=str)).astype(str).eq("fallback_replaced").mean())
        if not ours_reports.empty and "validation_status" in ours_reports
        else np.nan
    )
    source_identifier_clean = (
        float((~ours_reports["report_zh"].fillna("").str.contains(r"(?:AD|MCI|HC)_[FM]_", regex=True)).mean())
        if not ours_reports.empty
        else np.nan
    )
    agent_returned = int(agent_status.get("agent_returned_cases", 0))
    agent_requested = int(agent_status.get("agent_requested_cases", 0))
    agent_coverage = agent_returned / agent_requested if agent_requested else np.nan
    agent_trace_validity = (
        float(np.mean([bool(item.get("valid")) for item in agent_audits]))
        if agent_audits
        else np.nan
    )
    invalid_cases = {
        str(item.get("case_id")) for item in agent_audits if not bool(item.get("valid"))
    }
    rollback_count = int(agent_status.get("rolled_back_candidates", 0))
    rollback_enforcement = (
        min(rollback_count / len(invalid_cases), 1.0) if invalid_cases else 1.0
    )
    accepted_rate = (
        int(agent_status.get("accepted_bounded_corrections", 0)) / agent_returned
        if agent_returned
        else np.nan
    )
    agent_probability_change_rate = np.nan
    agent_decision_change_rate = np.nan
    agent_mean_probability_l1 = np.nan
    agent_auroc_delta = np.nan
    agent_accuracy_delta = np.nan
    agent_net_corrected_cases = np.nan
    matched_encoder_macro_auroc_gain = np.nan
    matched_encoder_accuracy_gain = np.nan
    if not agent_off.empty:
        paired_agent = agent_off.merge(
            ours,
            on="subject_id",
            how="inner",
            suffixes=("_off", "_on"),
        )
        probability_columns = [f"prob_{label}" for label in labels]
        if not paired_agent.empty and all(
            f"{column}_{suffix}" in paired_agent
            for column in probability_columns
            for suffix in ["off", "on"]
        ):
            off_probability = paired_agent[
                [f"{column}_off" for column in probability_columns]
            ].to_numpy(dtype=float)
            on_probability = paired_agent[
                [f"{column}_on" for column in probability_columns]
            ].to_numpy(dtype=float)
            probability_l1 = np.abs(on_probability - off_probability).sum(axis=1)
            agent_probability_change_rate = float((probability_l1 > 1e-9).mean())
            agent_mean_probability_l1 = float(probability_l1.mean())
            agent_decision_change_rate = float(
                paired_agent["predicted_label_off"].astype(str).ne(
                    paired_agent["predicted_label_on"].astype(str)
                ).mean()
            )
            off_result = evaluate_predictions(agent_off, bins, labels, positive_class)
            on_result = evaluate_predictions(ours, bins, labels, positive_class)
            agent_auroc_delta = float(
                on_result["macro_auroc_ovr"] - off_result["macro_auroc_ovr"]
            )
            agent_accuracy_delta = float(on_result["accuracy"] - off_result["accuracy"])
            off_correct = paired_agent["predicted_label_off"].astype(str).eq(
                paired_agent["label_off"].astype(str)
            )
            on_correct = paired_agent["predicted_label_on"].astype(str).eq(
                paired_agent["label_on"].astype(str)
            )
            agent_net_corrected_cases = float(
                ((~off_correct) & on_correct).sum() - (off_correct & (~on_correct)).sum()
            )
        matched_encoder = ablations[
            ablations["condition"].eq("Matched_encoder_modal_gate")
        ]
        if not matched_encoder.empty:
            matched_result = evaluate_predictions(
                matched_encoder, bins, labels, positive_class
            )
            off_result = evaluate_predictions(agent_off, bins, labels, positive_class)
            matched_encoder_macro_auroc_gain = float(
                off_result["macro_auroc_ovr"] - matched_result["macro_auroc_ovr"]
            )
            matched_encoder_accuracy_gain = float(
                off_result["accuracy"] - matched_result["accuracy"]
            )
    prior_blindness = bool(
        agent_status.get("test_labels_exposed_to_agent") is False
        and agent_status.get("supervised_prior_exposed_to_agent") is False
    )
    correction_strength = float(agent_status.get("correction_strength", np.nan))
    selection_status = str(
        agent_status.get(
            "correction_selection_status",
            agent_calibration.get("selection_status", "not_available"),
        )
    )
    correction_selection_valid = bool(
        (correction_strength == 0.0 and selection_status == "failed_closed_no_joint_gain")
        or (correction_strength > 0.0 and selection_status == "validated_joint_gain")
    )
    rows = [
        ["Ours", "MetricEvidence completeness", metric_completeness, metric_completeness >= 0.95, "all evidence-object fields populated"],
        ["Ours", "StateCard completeness", state_completeness, state_completeness >= 0.95, "all state-estimate fields populated"],
        [
            "Ours",
            "task-specific metric trace",
            task_metric_trace,
            bool(not np.isfinite(task_metric_trace) or task_metric_trace >= 0.95),
            "task-specific states retain supporting metrics from the same source task; not applicable to single-task datasets",
        ],
        [
            "Ours",
            "task-specific segment trace",
            task_segment_trace,
            bool(not np.isfinite(task_segment_trace) or task_segment_trace >= 0.90),
            "locally measurable task states link to source-time segments from that task; not applicable when local segment evidence is unavailable",
        ],
        [
            "Ours",
            "branch contribution trace",
            branch_trace,
            branch_trace >= 0.95,
            "case-level expert reliance is finite, non-negative and normalized to one",
        ],
        ["Ours", "report-permission audit", permission, bool(np.isfinite(permission) and permission == 1.0), "every disease-supporting metric or state citation resolves to a non-missing report-permitted subject-level evidence object"],
        ["Ours", "quality-evidence separation audit", quality_separation, bool(np.isfinite(quality_separation) and quality_separation == 1.0), "recording and transcript quality references are isolated from disease support and used only to communicate limitations"],
        ["Ours", "source identifier privacy audit", source_identifier_clean, bool(np.isfinite(source_identifier_clean) and source_identifier_clean == 1.0), "clinical text does not expose source diagnosis identifiers"],
        ["Ours", "diagnostic Agent execution coverage", agent_coverage, bool(np.isfinite(agent_coverage) and agent_coverage == 1.0), "fraction of the preregistered held-out Agent cohort with a returned structured candidate"],
        ["Ours", "diagnostic Agent evidence validity", agent_trace_validity, bool(np.isfinite(agent_trace_validity) and agent_trace_validity >= 0.95), "candidate decisions cite only valid clinical, counter and quality evidence in their permitted roles"],
        ["Ours", "diagnostic Agent prior-blind audit", float(prior_blindness), prior_blindness, "the Agent input excludes test labels, supervised probabilities, predicted labels and class-support summaries"],
        ["Ours", "validation-only correction selection", correction_strength, correction_selection_valid, "one shared correction strength is selected on development cases under joint macro-F1 gain and macro-AUROC non-inferiority; zero is the fail-closed result"],
        ["Ours", "cognitive rollback enforcement", rollback_enforcement, bool(rollback_enforcement == 1.0), "every invalid Agent candidate is prevented from changing the supervised prior"],
        ["Ours", "accepted bounded Agent correction rate", accepted_rate, bool(np.isfinite(accepted_rate)), "descriptive fraction of valid classify candidates that changed probability within the train-selected bound; not a pass/fail performance target"],
        ["Ours", "Agent-on probability application rate", agent_probability_change_rate, bool(np.isfinite(agent_probability_change_rate)), "paired held-out fraction whose frozen supervised prior probability was actually changed by the validated Agent correction"],
        ["Ours", "Agent-on decision change rate", agent_decision_change_rate, bool(np.isfinite(agent_decision_change_rate)), "paired held-out fraction whose predicted class changed after enabling the Agent; descriptive, not a required target"],
        ["Ours", "Agent-on mean probability L1 change", agent_mean_probability_l1, bool(np.isfinite(agent_mean_probability_l1)), "mean paired total probability displacement from the same frozen supervised prior"],
        ["Ours", "Agent-on macro AUROC delta", agent_auroc_delta, bool(np.isfinite(agent_auroc_delta) and agent_auroc_delta >= 0.0), "macro AUROC difference between Agent-on and Agent-off using the same frozen supervised prior; a negative value fails the Agent-gain claim"],
        ["Ours", "Agent-on accuracy delta", agent_accuracy_delta, bool(np.isfinite(agent_accuracy_delta) and agent_accuracy_delta >= 0.0), "accuracy difference between Agent-on and Agent-off using the same frozen supervised prior; a negative value fails the Agent-gain claim"],
        ["Ours", "Agent net corrected classifications", agent_net_corrected_cases, bool(np.isfinite(agent_net_corrected_cases) and agent_net_corrected_cases >= 0.0), "number of Agent-corrected errors minus previously correct cases changed to errors"],
        ["Ours", "same-encoder cognition macro AUROC gain", matched_encoder_macro_auroc_gain, bool(np.isfinite(matched_encoder_macro_auroc_gain) and matched_encoder_macro_auroc_gain >= 0.0), "Agent-off cognition framework minus a matched frozen text/audio encoder gate on the identical held-out split"],
        ["Ours", "same-encoder cognition accuracy gain", matched_encoder_accuracy_gain, bool(np.isfinite(matched_encoder_accuracy_gain) and matched_encoder_accuracy_gain >= 0.0), "Agent-off cognition framework minus a matched frozen text/audio encoder gate on the identical held-out split"],
        ["Ours", "evidence-span faithfulness", span_faithfulness, bool(not np.isfinite(span_faithfulness) or span_faithfulness >= 0.90), "each locally traceable speech-behaviour citation resolves to the same extracted case, time bounds, source span and state-specific selection rule; not applicable without local audio spans"],
        ["Ours", "reference-state intervention on errors", intervention_rate, bool(np.isfinite(intervention_rate) and intervention_rate >= 0.60), "proportion of misclassified cases whose true-class probability does not decrease after replacing one discrepant state with its training-set true-class reference"],
        ["Ours", "concept-only vs full fusion", ablation_gain, bool(np.isfinite(ablation_gain) and ablation_gain >= 0.0), "macro AUROC difference between full fusion and concept-only"],
        ["Ours", "task-specific state ablation", task_state_gain, bool(not np.isfinite(task_state_gain) or task_state_gain >= 0.0), "macro AUROC difference between the train-selected task-aware concept model and the forced overall-state-only model; zero means the training rule retained no task states; not applicable to single-task datasets"],
        ["Ours", "state raw-deviation extreme rate", extreme_state_rate, bool(np.isfinite(extreme_state_rate) and extreme_state_rate == 0.0), "fraction of StateCards with |raw_state_z| > 20; near-constant reference metrics must be marked unavailable rather than clipped"],
        ["Ours", "state contribution saturation rate", state_saturation_rate, bool(np.isfinite(state_saturation_rate) and state_saturation_rate <= 0.20), "fraction of StateCards at the configured contribution clip; high rates indicate loss of case-level resolution"],
        ["Ours", "clinical report safe-fallback replacement rate", fallback_report_rate, bool(np.isfinite(fallback_report_rate) and fallback_report_rate <= 0.25), "fraction of generated reports rejected by deterministic validation and replaced with a safe template"],
        ["Ours", "clinical report rubric /25", ours_rubric["total_25"], bool(np.isfinite(ours_rubric["total_25"]) and ours_rubric["total_25"] >= 20), "automated structural audit; physician review remains separate"],
        ["B2", "clinical report rubric /25", b2_rubric["total_25"], bool(np.isfinite(b2_rubric["total_25"]) and b2_rubric["total_25"] >= 20), "direct-agent report from transcript only"],
    ]
    layer_b = pd.DataFrame(rows, columns=["condition", "check", "value", "passed", "interpretation"])
    detail = {
        "ours_report_rubric": ours_rubric,
        "b2_report_rubric": b2_rubric,
        "matched_report_cases": len(matched_report_cases),
        "reference_state_intervention_cases": int(len(interventions)),
        "concept_only_macro_auroc": concept_auc,
        "overall_state_only_macro_auroc": overall_state_auc,
        "task_specific_state_macro_auroc_gain": task_state_gain,
        "report_rater": report_rater,
        "diagnostic_agent": {
            "execution_coverage": agent_coverage,
            "candidate_evidence_validity": agent_trace_validity,
            "rollback_enforcement": rollback_enforcement,
            "accepted_bounded_correction_rate": accepted_rate,
            "prior_blindness": prior_blindness,
            "correction_strength": correction_strength,
            "correction_selection_status": selection_status,
            "correction_calibration": agent_calibration,
            "agent_off_vs_on": {
                "probability_application_rate": agent_probability_change_rate,
                "decision_change_rate": agent_decision_change_rate,
                "mean_probability_l1_change": agent_mean_probability_l1,
                "macro_auroc_delta": agent_auroc_delta,
                "accuracy_delta": agent_accuracy_delta,
                "net_corrected_classifications": agent_net_corrected_cases,
            },
            "matched_encoder_isolation": {
                "macro_auroc_gain": matched_encoder_macro_auroc_gain,
                "accuracy_gain": matched_encoder_accuracy_gain,
            },
            "status": agent_status,
        },
        "limitations": [
            "The /25 score is an automated structural audit, not a completed clinician study.",
            "Semantic coherence and task-content states remain disabled unless a validated task scorer exists.",
            "The diagnostic Agent contributes bounded evidence-derived risk correction; the separate communication renderer only converts the locked decision trace into report text.",
            "The reference-state intervention is a model mechanism stress test on misclassified cases; its reference is a training-set class mean, not a clinician-annotated counterfactual state and not causal proof.",
        ],
    }
    return layer_b, detail


def _append_metric_rows(
    rows: list[list[Any]],
    condition: str,
    scope: str,
    result: dict[str, Any],
    intervals: dict[str, list[float]],
) -> None:
    for metric, value in result.items():
        if isinstance(value, (int, float)):
            ci = intervals.get(metric, [np.nan, np.nan])
            rows.append([condition, scope, metric, value, ci[0], ci[1]])
    for label, values in result["per_class"].items():
        for metric in ["precision", "sensitivity", "specificity", "f1", "auroc", "auprc"]:
            rows.append([condition, scope, f"{label}_{metric}", values[metric], np.nan, np.nan])


def run_evaluation(
    predictions: dict[str, Path],
    controls_path: Path,
    evidence_path: Path,
    state_cards_path: Path,
    segments_path: Path,
    contributions_path: Path,
    interventions_path: Path,
    ablations_path: Path,
    b2_reports_path: Path,
    ours_reports_path: Path,
    report_scores_path: Path,
    cognitive_agent_audit_path: Path,
    cognitive_agent_status_path: Path,
    config: dict[str, Any],
    layer_a_path: Path,
    layer_b_path: Path,
    summary_path: Path,
) -> None:
    labels = _labels(config)
    positive_class = str(config.get("positive_class", labels[-1]))
    bins = int(config["ece_bins"])
    iterations = int(config["bootstrap_iterations"])
    frames = {condition: pd.read_csv(path, dtype={"subject_id": str}) for condition, path in predictions.items()}
    layer_a_rows: list[list[Any]] = []
    summary: dict[str, Any] = {"labels": labels, "positive_class": positive_class, "layer_a": {}, "layer_b": {}}
    for condition, frame in frames.items():
        if frame.empty or "predicted_label" not in frame:
            summary["layer_a"][condition] = {"status": "not_run"}
            continue
        result = evaluate_predictions(frame, bins, labels, positive_class)
        intervals = bootstrap_intervals(frame, bins, iterations, labels, positive_class)
        result["bootstrap_effective_resamples"] = int(
            intervals.get("__effective_resamples__", [0])[0]
        )
        result["bootstrap_attempted_resamples"] = int(
            intervals.get("__attempted_resamples__", [0])[0]
        )
        result["bootstrap_95_ci"] = intervals
        summary["layer_a"][condition] = {"full_available_cohort": result}
        _append_metric_rows(layer_a_rows, condition, "full_available_cohort", result, intervals)

    completed = [frame for frame in frames.values() if not frame.empty and "predicted_label" in frame]
    if len(completed) == len(frames):
        cohorts = [set(frame["subject_id"].astype(str)) for frame in completed]
        common = set.intersection(*cohorts)
        matched_differs_from_full = any(cohort != cohorts[0] for cohort in cohorts[1:])
        if matched_differs_from_full:
            cohort_sizes = {
                condition: int(len(set(frame["subject_id"].astype(str))))
                for condition, frame in frames.items()
            }
            raise ValueError(
                "B1, B2 and Ours must contain identical subject cohorts; "
                f"observed sizes={cohort_sizes}"
            )
        for condition, frame in frames.items():
            if matched_differs_from_full:
                matched = frame[frame["subject_id"].astype(str).isin(common)].copy()
                result = evaluate_predictions(matched, bins, labels, positive_class)
                intervals = bootstrap_intervals(
                    matched,
                    bins,
                    iterations,
                    labels,
                    positive_class,
                )
            else:
                result = dict(summary["layer_a"][condition]["full_available_cohort"])
                intervals = result.get("bootstrap_95_ci", {})
            summary["layer_a"][condition]["matched_three_arm"] = result
            if matched_differs_from_full:
                _append_metric_rows(
                    layer_a_rows,
                    condition,
                    "matched_three_arm",
                    result,
                    intervals,
                )
        summary["matched_three_arm_subjects"] = len(common)
        summary["matched_three_arm_equals_full_cohort"] = not matched_differs_from_full
    summary["paired_comparisons"] = {}
    if "Ours" in frames and not frames["Ours"].empty:
        for baseline_name in ["B1", "B2"]:
            if baseline_name not in frames or frames[baseline_name].empty:
                continue
            comparison = paired_prediction_comparison(
                frames["Ours"],
                frames[baseline_name],
                bins,
                iterations,
                labels,
                positive_class,
                seed=20260827 + (1 if baseline_name == "B1" else 2),
            )
            summary["paired_comparisons"][f"Ours_minus_{baseline_name}"] = comparison
            if comparison.get("status") != "completed":
                continue
            for metric, value in comparison["delta_ours_minus_baseline"].items():
                ci = comparison.get("paired_bootstrap_95_ci", {}).get(
                    metric, [np.nan, np.nan]
                )
                layer_a_rows.append(
                    [
                        f"Ours-{baseline_name}",
                        "paired_difference",
                        f"delta_{metric}",
                        value,
                        ci[0],
                        ci[1],
                    ]
                )
            layer_a_rows.append(
                [
                    f"Ours-{baseline_name}",
                    "paired_difference",
                    "mcnemar_exact_p_value",
                    comparison["mcnemar_exact"]["p_value"],
                    np.nan,
                    np.nan,
                ]
            )

    controls = pd.read_csv(controls_path, dtype={"subject_id": str})
    for condition, frame in controls.groupby("condition"):
        result = evaluate_predictions(frame, bins, labels, positive_class)
        intervals = bootstrap_intervals(
            frame, bins, iterations, labels, positive_class, seed=20260831
        )
        result["bootstrap_effective_resamples"] = int(
            intervals.get("__effective_resamples__", [0])[0]
        )
        result["bootstrap_attempted_resamples"] = int(
            intervals.get("__attempted_resamples__", [0])[0]
        )
        result["bootstrap_95_ci"] = intervals
        summary["layer_a"][condition] = {"full_available_cohort": result}
        _append_metric_rows(
            layer_a_rows, condition, "full_available_cohort", result, intervals
        )
    pd.DataFrame(
        layer_a_rows,
        columns=["condition", "analysis_scope", "metric", "value", "ci_low", "ci_high"],
    ).to_csv(layer_a_path, index=False)

    layer_b, detail = build_layer_b(
        evidence_path,
        state_cards_path,
        segments_path,
        contributions_path,
        interventions_path,
        ablations_path,
        predictions["Ours"],
        b2_reports_path,
        ours_reports_path,
        report_scores_path,
        cognitive_agent_audit_path,
        cognitive_agent_status_path,
        bins,
        labels,
        positive_class,
    )
    layer_b.to_csv(layer_b_path, index=False)
    summary["layer_b"] = detail
    json_dump(summary, summary_path)
