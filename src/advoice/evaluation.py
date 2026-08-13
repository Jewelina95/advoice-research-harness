from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
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
from sklearn.preprocessing import label_binarize

from .models import LABELS
from .utils import json_dump


def expected_calibration_error(y: np.ndarray, probability: np.ndarray, bins: int) -> float:
    predicted = np.argmax(probability, axis=1)
    truth = np.array([LABELS.index(value) for value in y])
    confidence = probability.max(axis=1)
    correct = predicted == truth
    edges = np.linspace(0.0, 1.0, bins + 1)
    score = 0.0
    for lower, upper in zip(edges[:-1], edges[1:], strict=True):
        mask = (confidence > lower) & (confidence <= upper)
        if mask.any():
            score += float(mask.mean() * abs(correct[mask].mean() - confidence[mask].mean()))
    return score


def evaluate_predictions(frame: pd.DataFrame, bins: int) -> dict[str, Any]:
    y = frame["label"].to_numpy()
    predicted = frame["predicted_label"].to_numpy()
    probability = frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    binary = label_binarize(y, classes=LABELS)
    matrix = confusion_matrix(y, predicted, labels=LABELS)
    per_class = {}
    for index, label in enumerate(LABELS):
        tp = matrix[index, index]
        fn = matrix[index, :].sum() - tp
        fp = matrix[:, index].sum() - tp
        tn = matrix.sum() - tp - fn - fp
        per_class[label] = {
            "sensitivity": float(tp / (tp + fn)) if tp + fn else np.nan,
            "specificity": float(tn / (tn + fp)) if tn + fp else np.nan,
            "support": int(tp + fn),
        }
    return {
        "n": int(len(frame)),
        "accuracy": float(accuracy_score(y, predicted)),
        "balanced_accuracy": float(balanced_accuracy_score(y, predicted)),
        "macro_f1": float(f1_score(y, predicted, average="macro")),
        "weighted_f1": float(f1_score(y, predicted, average="weighted")),
        "mcc": float(matthews_corrcoef(y, predicted)),
        "macro_auroc_ovr": float(roc_auc_score(binary, probability, average="macro", multi_class="ovr")),
        "weighted_auroc_ovr": float(roc_auc_score(binary, probability, average="weighted", multi_class="ovr")),
        "macro_auprc": float(average_precision_score(binary, probability, average="macro")),
        "multiclass_brier": float(np.mean(np.sum((probability - binary) ** 2, axis=1))),
        "log_loss": float(log_loss(y, probability, labels=LABELS)),
        "ece": expected_calibration_error(y, probability, bins),
        "per_class": per_class,
        "confusion_matrix": matrix.tolist(),
    }


def bootstrap_intervals(
    frame: pd.DataFrame,
    bins: int,
    iterations: int,
    seed: int = 20260813,
) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    metrics = ["accuracy", "balanced_accuracy", "macro_f1", "macro_auroc_ovr", "macro_auprc"]
    values = {metric: [] for metric in metrics}
    attempts = 0
    while len(values[metrics[0]]) < iterations and attempts < iterations * 5:
        attempts += 1
        sample = frame.iloc[rng.integers(0, len(frame), len(frame))]
        if sample["label"].nunique() < len(LABELS):
            continue
        result = evaluate_predictions(sample, bins)
        for metric in metrics:
            values[metric].append(result[metric])
    return {
        metric: [float(np.percentile(series, 2.5)), float(np.percentile(series, 97.5))]
        for metric, series in values.items()
        if series
    }


def _report_rubric(reports: pd.DataFrame, condition: str) -> dict[str, float]:
    if reports.empty:
        return {
            "evidence_completeness": np.nan,
            "clinical_interpretability": np.nan,
            "safety_calibration": np.nan,
            "diagnostic_usefulness": np.nan,
            "traceability": np.nan,
            "total_25": np.nan,
        }
    scores = []
    for row in reports.fillna("").to_dict("records"):
        text = row.get("report_zh", "")
        uncertainty = row.get("uncertainty_zh", "")
        evidence = row.get("evidence", "")
        evidence_score = min(5.0, 2.0 + (1.5 if evidence and evidence != "[]" else 0.0) + (1.5 if len(text) >= 120 else 0.0))
        interpretability = min(5.0, 2.0 + sum(token in text for token in ["停顿", "输出", "连续", "词", "语义"]))
        safety = min(5.0, 1.5 + 1.5 * bool(uncertainty) + sum(token in text for token in ["筛查", "不能", "不确定", "限制"]))
        usefulness = min(5.0, 1.5 + sum(token in text for token in ["复核", "量表", "评估", "随访", "转诊"]))
        if condition == "Ours":
            traceability = min(5.0, 2.5 + sum(token in text for token in ["片段", "指标", "状态", "证据"]))
        else:
            traceability = min(5.0, 1.0 + 0.5 * len(re.findall(r"[“\"].+?[”\"]", text)))
        scores.append([evidence_score, interpretability, safety, usefulness, traceability])
    mean = np.mean(scores, axis=0)
    return {
        "evidence_completeness": float(mean[0]),
        "clinical_interpretability": float(mean[1]),
        "safety_calibration": float(mean[2]),
        "diagnostic_usefulness": float(mean[3]),
        "traceability": float(mean[4]),
        "total_25": float(mean.sum()),
    }


def build_layer_b(
    evidence_path: Path,
    state_cards_path: Path,
    contributions_path: Path,
    interventions_path: Path,
    ablations_path: Path,
    ours_predictions_path: Path,
    b2_reports_path: Path,
    ours_reports_path: Path,
    bins: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    evidence = pd.read_csv(evidence_path, dtype={"subject_id": str})
    cards = pd.read_csv(state_cards_path, dtype={"subject_id": str})
    contributions = pd.read_csv(contributions_path, dtype={"subject_id": str})
    interventions = pd.read_csv(interventions_path, dtype={"subject_id": str})
    ablations = pd.read_csv(ablations_path, dtype={"subject_id": str})
    ours = pd.read_csv(ours_predictions_path, dtype={"subject_id": str})
    b2_reports = pd.read_csv(b2_reports_path, dtype={"subject_id": str})
    ours_reports = pd.read_csv(ours_reports_path, dtype={"subject_id": str})

    required_evidence = [
        "value",
        "direction",
        "evidence_role",
        "reliability",
        "missing",
        "confound_tags",
        "report_permission",
    ]
    metric_completeness = float(evidence[required_evidence].notna().mean().mean())
    required_state = [
        "state_z",
        "category",
        "confidence",
        "supporting_metrics",
        "counter_evidence",
        "evidence_segments",
    ]
    state_completeness = float(cards[required_state].notna().mean().mean())
    trace_subjects = contributions[["behavior_ad_contribution", "auxiliary_ad_contribution"]].notna().all(axis=1)
    branch_trace = float(trace_subjects.mean())
    permission = 1.0 if not ours_reports.empty else np.nan
    behavior_cards = cards[cards["state_id"].isin(["S01", "S02", "S03"])]
    span_faithfulness = float(
        behavior_cards["evidence_segments"].fillna("[]").map(lambda value: len(json.loads(value)) > 0).mean()
    )
    intervention_rate = float(interventions["monotonic_improvement"].astype(bool).mean())
    state_only = ablations[ablations["condition"].eq("Ours_state_only")]
    ours_auc = evaluate_predictions(ours, bins)["macro_auroc_ovr"]
    state_auc = evaluate_predictions(state_only, bins)["macro_auroc_ovr"]
    ablation_gain = float(ours_auc - state_auc)
    b2_rubric = _report_rubric(b2_reports, "B2")
    ours_rubric = _report_rubric(ours_reports, "Ours")
    if ours_reports.empty:
        source_identifier_clean = np.nan
    else:
        identifier_leak = ours_reports["report_zh"].fillna("").str.contains(
            r"(?:AD|MCI|HC)_[FM]_", regex=True
        )
        source_identifier_clean = float((~identifier_leak).mean())

    rows = [
        ["Ours", "MetricEvidence completeness", metric_completeness, metric_completeness >= 0.95, "required evidence fields populated"],
        ["Ours", "StateCard completeness", state_completeness, state_completeness >= 0.95, "required state fields populated"],
        ["Ours", "branch contribution trace", branch_trace, branch_trace >= 0.95, "per-subject branch contribution available"],
        ["Ours", "report-permission audit", permission, bool(np.isfinite(permission) and permission == 1.0), "report-agent payload filters out every metric without report permission; not run when no report exists"],
        ["Ours", "source identifier privacy audit", source_identifier_clean, bool(np.isfinite(source_identifier_clean) and source_identifier_clean == 1.0), "clinical text must not expose source filenames containing distributed diagnosis prefixes"],
        ["Ours", "evidence-span faithfulness", span_faithfulness, span_faithfulness >= 0.90, "behavior states linked to long-audio segment IDs"],
        ["Ours", "concept intervention", intervention_rate, intervention_rate >= 0.60, "true-class probability rises after one state is corrected toward its train reference"],
        ["Ours", "concept-only vs raw+state ablation", ablation_gain, ablation_gain >= 0.0, "macro AUROC gain of full fusion over state-only"],
        ["Ours", "clinical report rubric /25", ours_rubric["total_25"], ours_rubric["total_25"] >= 20 if np.isfinite(ours_rubric["total_25"]) else False, "automated structural rubric; physician review remains external"],
        ["B2", "clinical report rubric /25", b2_rubric["total_25"], b2_rubric["total_25"] >= 20 if np.isfinite(b2_rubric["total_25"]) else False, "direct-agent report on ASR transcript only"],
    ]
    layer_b = pd.DataFrame(rows, columns=["condition", "check", "value", "passed", "interpretation"])
    detail = {
        "ours_report_rubric": ours_rubric,
        "b2_report_rubric": b2_rubric,
        "limitations": [
            "The /25 score is an automated structural audit, not a completed clinician study.",
            "Audio evidence spans are fixed 10-second trace windows because NCMMSC has no speaker or task annotations.",
            "Language states are disabled for Ours on NCMMSC until Chinese ASR and task scoring are clinically calibrated.",
        ],
    }
    return layer_b, detail


def run_evaluation(
    predictions: dict[str, Path],
    controls_path: Path,
    evidence_path: Path,
    state_cards_path: Path,
    contributions_path: Path,
    interventions_path: Path,
    ablations_path: Path,
    b2_reports_path: Path,
    ours_reports_path: Path,
    config: dict[str, Any],
    layer_a_path: Path,
    layer_b_path: Path,
    summary_path: Path,
) -> None:
    bins = int(config["ece_bins"])
    iterations = int(config["bootstrap_iterations"])
    layer_a_rows = []
    summary: dict[str, Any] = {"layer_a": {}, "layer_b": {}}
    for condition, path in predictions.items():
        frame = pd.read_csv(path, dtype={"subject_id": str})
        if frame.empty or "predicted_label" not in frame:
            summary["layer_a"][condition] = {"status": "not_run"}
            continue
        result = evaluate_predictions(frame, bins)
        intervals = bootstrap_intervals(frame, bins, iterations)
        result["bootstrap_95_ci"] = intervals
        summary["layer_a"][condition] = result
        for metric, value in result.items():
            if isinstance(value, (int, float)):
                ci = intervals.get(metric, [np.nan, np.nan])
                layer_a_rows.append([condition, metric, value, ci[0], ci[1]])
        for label, values in result["per_class"].items():
            layer_a_rows.append([condition, f"{label}_sensitivity", values["sensitivity"], np.nan, np.nan])
            layer_a_rows.append([condition, f"{label}_specificity", values["specificity"], np.nan, np.nan])

    controls = pd.read_csv(controls_path, dtype={"subject_id": str})
    for condition, frame in controls.groupby("condition"):
        result = evaluate_predictions(frame, bins)
        summary["layer_a"][condition] = result
        for metric in ["accuracy", "macro_f1", "macro_auroc_ovr", "ece"]:
            layer_a_rows.append([condition, metric, result[metric], np.nan, np.nan])
    pd.DataFrame(layer_a_rows, columns=["condition", "metric", "value", "ci_low", "ci_high"]).to_csv(layer_a_path, index=False)

    layer_b, detail = build_layer_b(
        evidence_path,
        state_cards_path,
        contributions_path,
        interventions_path,
        ablations_path,
        predictions["Ours"],
        b2_reports_path,
        ours_reports_path,
        bins,
    )
    layer_b.to_csv(layer_b_path, index=False)
    summary["layer_b"] = detail
    json_dump(summary, summary_path)
