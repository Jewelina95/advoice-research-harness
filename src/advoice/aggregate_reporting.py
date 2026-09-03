from __future__ import annotations

import json
from pathlib import Path
import re
import textwrap
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
import yaml

from .config import ProjectPaths, load_all
from .evaluation import evaluate_predictions
from .failure_analysis import build_failure_analysis
from .utils import json_load, now_utc


COLORS = {"B1": "#8A8F98", "B2": "#E68178", "Ours": "#278C82"}
BRANCH_COLORS = {
    "speech_behavior": "#4D84C4",
    "language": "#7864C7",
    "interaction": "#48A5A1",
    "auxiliary_acoustic": "#E0A13B",
}

STATE_NAMES_EN = {
    "S01": "Pause and fluency burden",
    "S02": "Output efficiency",
    "S03": "Speech continuity",
    "S04": "Vocal-intensity stability",
    "S05": "Prosodic variability",
    "S06": "Low-level spectral pattern",
    "S07": "Lexical retrieval and specificity",
    "S08": "Lexical diversity",
    "S09": "Semantic coherence",
    "S10": "Information density",
    "S11": "Syntactic complexity",
    "S12": "Disfluency, repetition and repair",
    "S13": "Interaction and pragmatic burden",
    "S14": "Task performance",
}


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 12.5,
            "axes.titlesize": 14,
            "axes.titleweight": "bold",
            "axes.labelsize": 12.5,
            "axes.labelweight": "semibold",
            "xtick.labelsize": 11.5,
            "ytick.labelsize": 11.5,
            "legend.fontsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _scope(layer_a: pd.DataFrame) -> str:
    return "matched_three_arm" if layer_a["analysis_scope"].eq("matched_three_arm").any() else "full_available_cohort"


def _short_name(dataset_id: str) -> str:
    mapping = {
        "ADReSS_2020": "ADReSS 2020",
        "ADReSSo_2021_diagnosis": "ADReSSo diagnosis",
        "ADReSSo_2021_progression": "ADReSSo progression",
        "PREPARE_DrivenData": "PREPARE",
        "PROCESS_2": "PROCESS-2",
        "TAUKADIAL": "TAUKADIAL",
        "DementiaBank_Pitt": "Pitt",
        "DementiaNet_PublicFigures": "DementiaNet",
        "NCMMSC2021_AD": "NCMMSC 2021",
    }
    return mapping.get(dataset_id, dataset_id.replace("_", " "))


def collect_results(paths: ProjectPaths, dataset_ids: list[str]) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    datasets: list[dict[str, Any]] = []
    layer_a_frames: list[pd.DataFrame] = []
    layer_b_frames: list[pd.DataFrame] = []
    for dataset_id in dataset_ids:
        artifact = paths.artifacts / dataset_id
        required = [artifact / "manifest.csv", artifact / "layer_a_metrics.csv", artifact / "layer_b_checks.csv"]
        if not all(path.exists() for path in required):
            datasets.append({"dataset_id": dataset_id, "display_name": _short_name(dataset_id), "status": "not_run"})
            continue
        config = load_all(dataset_id)
        manifest = pd.read_csv(artifact / "manifest.csv", dtype={"subject_id": str})
        audit = json_load(artifact / "dataset_audit.json", {})
        b2_status = json_load(artifact / "b2_status.json", {})
        ours_report_status = json_load(artifact / "ours_report_status.json", {})
        ours_meta = json_load(artifact / "ours_model.json", {})
        contributions_path = artifact / "branch_contributions.csv"
        contributions = (
            pd.read_csv(contributions_path)
            if contributions_path.exists()
            else pd.DataFrame()
        )
        correction_gate = pd.to_numeric(
            contributions.get("agent_correction_gate", pd.Series(dtype=float)),
            errors="coerce",
        )
        base_gate_shares = {
            column.removeprefix("base_gate_weight_"): float(
                pd.to_numeric(contributions[column], errors="coerce").mean()
            )
            for column in contributions.columns
            if column.startswith("base_gate_weight_")
            and pd.to_numeric(contributions[column], errors="coerce").notna().any()
        }
        if not base_gate_shares:
            base_gate_shares = {
                column.removeprefix("expert_").removesuffix("_contribution"): float(
                    pd.to_numeric(contributions[column], errors="coerce").mean()
                )
                for column in contributions.columns
                if column.startswith("expert_")
                and column.endswith("_contribution")
                and pd.to_numeric(contributions[column], errors="coerce")
                .notna()
                .any()
            }
        summary = json_load(artifact / "evaluation_summary.json", {})
        datasets.append(
            {
                "dataset_id": dataset_id,
                "display_name": config["dataset"]["display_name"],
                "short_name": _short_name(dataset_id),
                "channel": config["dataset"]["channel"],
                "channel_profile": config["channel_profile"]["id"],
                "task_type": config["dataset"]["task_type"],
                "labels": config["dataset"]["labels"],
                "recordings": audit.get("recordings", len(manifest)),
                "subjects": audit.get("subjects", manifest["subject_id"].nunique()),
                "test_subjects": audit.get("test_subjects", manifest.loc[manifest["split"].eq("test"), "subject_id"].nunique()),
                "b2_status": b2_status.get("status", "not_run"),
                "ours_report_status": ours_report_status.get("status", "not_run"),
                "ours_meta": ours_meta,
                "base_gate_shares": base_gate_shares,
                "agent_correction_gate_mean": (
                    float(correction_gate.mean()) if correction_gate.notna().any() else None
                ),
                "agent_correction_gate_std": (
                    float(correction_gate.std(ddof=0)) if correction_gate.notna().any() else None
                ),
                "summary": summary,
                "task_counts": audit.get("task_counts", {}),
                "language_counts": audit.get("language_counts", {}),
                "capture_label_confounding_flag": bool(
                    audit.get("capture_label_confounding_flag", False)
                ),
                "acquisition_group_max_label_purity": audit.get(
                    "acquisition_group_max_label_purity"
                ),
                "long_recording_only_passed": audit.get("long_recording_only_passed"),
                "six_second_files_in_analysis": audit.get("six_second_files_in_analysis"),
                "status": "completed",
            }
        )
        layer_a = pd.read_csv(artifact / "layer_a_metrics.csv")
        required_layer_a = {"condition", "analysis_scope", "metric", "value", "ci_low", "ci_high"}
        layer_b = pd.read_csv(artifact / "layer_b_checks.csv")
        required_layer_b = {"condition", "check", "value", "passed", "interpretation"}
        if not required_layer_a.issubset(layer_a.columns) or not required_layer_b.issubset(layer_b.columns):
            datasets[-1]["status"] = "stale_schema"
            continue
        layer_a["dataset_id"] = dataset_id
        layer_a["dataset_name"] = _short_name(dataset_id)
        layer_a["preferred_scope"] = _scope(layer_a)
        layer_a_frames.append(layer_a)
        layer_b["dataset_id"] = dataset_id
        layer_b["dataset_name"] = _short_name(dataset_id)
        layer_b_frames.append(layer_b)
    return datasets, pd.concat(layer_a_frames, ignore_index=True) if layer_a_frames else pd.DataFrame(), pd.concat(layer_b_frames, ignore_index=True) if layer_b_frames else pd.DataFrame()


def _preferred_layer_a(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["analysis_scope"].eq(frame["preferred_scope"])].copy()


def plot_legacy_c_comparison(
    paths: ProjectPaths,
    dataset_ids: list[str],
    path: Path,
) -> pd.DataFrame:
    """Compare the locked current B3 with the archived pre-Agent condition C."""

    rows: list[dict[str, Any]] = []
    for dataset_id in dataset_ids:
        artifact = paths.artifacts / dataset_id
        legacy_path = artifact / "legacy_c_predictions.csv"
        current_path = artifact / "ours_predictions.csv"
        if not legacy_path.exists() or not current_path.exists():
            continue
        legacy = pd.read_csv(legacy_path, dtype={"subject_id": str})
        current = pd.read_csv(current_path, dtype={"subject_id": str})
        matched_ids = sorted(
            set(legacy["subject_id"].astype(str))
            & set(current["subject_id"].astype(str))
        )
        if not matched_ids:
            continue
        legacy = legacy[legacy["subject_id"].astype(str).isin(matched_ids)].copy()
        current = current[current["subject_id"].astype(str).isin(matched_ids)].copy()
        config = load_all(dataset_id)["dataset"]
        labels = [str(value) for value in config["labels"]]
        positive_class = str(config.get("positive_class", labels[-1]))
        legacy_metrics = evaluate_predictions(
            legacy, 10, labels, positive_class
        )
        current_metrics = evaluate_predictions(
            current, 10, labels, positive_class
        )
        for metric in ["accuracy", "macro_f1", "macro_auroc_ovr"]:
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset_name": _short_name(dataset_id),
                    "metric": metric,
                    "legacy_c": float(legacy_metrics[metric]),
                    "current_b3": float(current_metrics[metric]),
                    "delta": float(current_metrics[metric] - legacy_metrics[metric]),
                    "matched_subjects": len(matched_ids),
                }
            )
    frame = pd.DataFrame(rows)
    _style()
    metric_order = ["accuracy", "macro_f1", "macro_auroc_ovr"]
    metric_titles = ["Accuracy", "Macro F1", "Macro AUROC"]
    dataset_order = [
        _short_name(dataset_id)
        for dataset_id in dataset_ids
        if _short_name(dataset_id) in set(frame.get("dataset_name", []))
    ]
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(14.2, max(5.8, 0.45 * max(len(dataset_order), 1) + 1.5)),
        sharey=True,
    )
    for axis, metric, title in zip(axes, metric_order, metric_titles, strict=True):
        subset = frame[frame["metric"].eq(metric)].set_index("dataset_name")
        values = subset.reindex(dataset_order)["delta"].to_numpy(dtype=float)
        positions = np.arange(len(dataset_order))
        colors = np.where(values > 1e-9, COLORS["Ours"], np.where(values < -1e-9, COLORS["B2"], COLORS["B1"]))
        axis.axvline(0.0, color="#4B5563", linewidth=1.0)
        axis.scatter(values, positions, s=58, color=colors, edgecolor="white", linewidth=0.8, zorder=3)
        for value, position in zip(values, positions, strict=True):
            axis.text(
                value + (0.008 if value >= 0 else -0.008),
                position,
                f"{value:+.3f}",
                ha="left" if value >= 0 else "right",
                va="center",
                fontsize=8.5,
            )
        axis.set_title(title, fontweight="bold")
        axis.set_xlabel("Current B3 minus legacy C")
        axis.set_yticks(positions, dataset_order)
        axis.invert_yaxis()
        finite = np.abs(values[np.isfinite(values)])
        limit = max(0.12, float(finite.max()) + 0.08) if len(finite) else 0.12
        axis.set_xlim(-limit, limit)
    fig.suptitle(
        "Current evidence-diagnostic Agent versus archived condition C",
        fontsize=15,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(
        0.02,
        0.01,
        "Positive values favour the current B3. This is a locked post-hoc audit, not a model-selection criterion.",
        fontsize=9,
        color="#5B6470",
    )
    fig.tight_layout(rect=[0.0, 0.04, 1.0, 0.94])
    _save(fig, path)
    return frame


def plot_speechcare_aligned_comparison(
    paths: ProjectPaths,
    layer_a: pd.DataFrame,
    output: Path,
) -> list[dict[str, Any]]:
    """Plot only PREPARE official-test metrics with explicit metric alignment."""
    config_path = paths.configs / "benchmarks" / "speechcare.yaml"
    benchmark = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = _preferred_layer_a(layer_a)
    prepare = data[data["dataset_id"].eq("PREPARE_DrivenData")]
    specifications = [
        ("micro_auroc_ovr", "Micro AUROC", "micro_auroc_ovr"),
        ("micro_f1", "F1", "reported_f1"),
        ("weighted_auroc_ovr", "Weighted AUROC", "weighted_auroc_ovr"),
        ("micro_auprc", "Micro AUPRC", "micro_auprc"),
        ("weighted_auprc", "Weighted AUPRC", "weighted_auprc"),
    ]
    conditions = ["B1", "B2", "Ours", "SpeechCARE"]
    colors = [COLORS["B1"], COLORS["B2"], COLORS["Ours"], "#5B6F9B"]
    rows: list[dict[str, Any]] = []
    values = np.full((len(conditions), len(specifications)), np.nan)
    errors = np.zeros_like(values)
    for metric_index, (ours_metric, display, benchmark_metric) in enumerate(specifications):
        for condition_index, condition in enumerate(conditions[:-1]):
            selected = prepare[
                prepare["condition"].eq(condition) & prepare["metric"].eq(ours_metric)
            ]
            if selected.empty:
                continue
            record = selected.iloc[0]
            values[condition_index, metric_index] = float(record["value"])
            if np.isfinite(record.get("ci_low", np.nan)) and np.isfinite(record.get("ci_high", np.nan)):
                errors[condition_index, metric_index] = float(record["ci_high"] - record["ci_low"]) / 2.0
            rows.append(
                {
                    "dataset_id": "PREPARE_DrivenData",
                    "condition": condition,
                    "metric": display,
                    "metric_key": ours_metric,
                    "value": values[condition_index, metric_index],
                    "protocol": "retrospective_official_test_benchmark",
                }
            )
        reference = benchmark["prepare_official_test"]["metrics"][benchmark_metric]
        values[-1, metric_index] = float(reference["value"])
        errors[-1, metric_index] = (
            float(reference["ci_half_width"])
            if reference.get("ci_half_width") is not None
            else 0.0
        )
        rows.append(
            {
                "dataset_id": "PREPARE_DrivenData",
                "condition": "SpeechCARE",
                "metric": display,
                "metric_key": benchmark_metric,
                "value": values[-1, metric_index],
                "protocol": (
                    "official_test_mean_of_10_runs"
                    if reference.get("ci_half_width") is not None
                    else "official_test_published_figure_point_estimate"
                ),
            }
        )
    _style()
    fig, axes = plt.subplots(
        1, 2, figsize=(16.5, 7.8), gridspec_kw={"width_ratios": [1.55, 1]},
        constrained_layout=True,
    )
    labels = [item[1] for item in specifications]
    y = np.arange(len(labels), dtype=float)
    offsets = {"B1": 0.27, "B2": 0.09, "Ours": -0.09, "SpeechCARE": -0.27}
    markers = {"B1": "o", "B2": "s", "Ours": "D", "SpeechCARE": "^"}
    for condition_index, condition in enumerate(conditions):
        current = values[condition_index]
        finite = np.isfinite(current)
        current_y = y + offsets[condition]
        xerr = errors[condition_index]
        axes[0].errorbar(
            current[finite], current_y[finite], xerr=xerr[finite],
            fmt=markers[condition], color=colors[condition_index],
            ecolor=colors[condition_index], markersize=8, capsize=3,
            elinewidth=1.4, label="B3 ADvoice" if condition == "Ours" else condition,
            zorder=3,
        )
        for value, row_y in zip(current[finite], current_y[finite], strict=True):
            axes[0].text(value + 0.008, row_y, f"{value:.3f}", va="center", fontsize=10, fontweight="semibold")
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.35, 0.94)
    axes[0].set_xlabel("Official-test score")
    axes[0].set_title("a  Absolute performance under aligned PREPARE endpoints", loc="left")
    axes[0].grid(axis="y", visible=False)
    axes[0].legend(ncol=2, loc="lower right")

    gaps = values[2] - values[3]
    gap_colors = [COLORS["Ours"] if gap >= 0 else "#C65D57" for gap in gaps]
    bars = axes[1].barh(y, gaps, color=gap_colors, height=0.58)
    axes[1].axvline(0, color="#4E585D", linewidth=1.2)
    axes[1].set_yticks(y, labels)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("B3 ADvoice minus SpeechCARE")
    axes[1].set_title("b  Remaining benchmark gap", loc="left")
    axes[1].grid(axis="y", visible=False)
    for bar, gap in zip(bars, gaps, strict=True):
        axes[1].text(
            gap + (0.003 if gap >= 0 else -0.003),
            bar.get_y() + bar.get_height() / 2,
            f"{gap:+.3f}",
            ha="left" if gap >= 0 else "right", va="center",
            fontsize=10.5, fontweight="bold",
        )
    finite_gap = np.abs(gaps[np.isfinite(gaps)])
    gap_limit = max(0.055, float(finite_gap.max()) + 0.015) if len(finite_gap) else 0.055
    axes[1].set_xlim(-gap_limit, gap_limit)
    fig.suptitle(
        "PREPARE official test | ADvoice versus SpeechCARE",
        fontsize=20, fontweight="bold", x=0.01, ha="left",
    )
    fig.text(
        0.01, 0.005,
        "SpeechCARE reports the mean of 10 training runs. ADvoice is a retrospective official-test benchmark; the right panel makes every unresolved gap explicit.",
        fontsize=11, color="#5F6B70",
    )
    _save(fig, output)
    pd.DataFrame(rows).to_csv(output.parent / "speechcare_protocol_aligned_metrics.csv", index=False)
    return rows


def _metric_forest(
    ax: plt.Axes,
    data: pd.DataFrame,
    metric: str,
    title: str,
    xlabel: str,
    xlim: tuple[float, float],
    show_ci: bool = False,
) -> None:
    subset = data[data["metric"].eq(metric)]
    datasets = list(dict.fromkeys(subset["dataset_name"]))
    y = np.arange(len(datasets), dtype=float)
    offsets = {"B1": 0.22, "B2": 0.0, "Ours": -0.22}
    markers = {"B1": "o", "B2": "s", "Ours": "D"}
    for condition in ["B1", "B2", "Ours"]:
        indexed = subset[subset["condition"].eq(condition)].set_index("dataset_name")
        values = np.array([indexed.loc[name, "value"] if name in indexed.index else np.nan for name in datasets], dtype=float)
        finite = np.isfinite(values)
        current_y = y + offsets[condition]
        if show_ci:
            lows = np.array([indexed.loc[name, "ci_low"] if name in indexed.index else np.nan for name in datasets], dtype=float)
            highs = np.array([indexed.loc[name, "ci_high"] if name in indexed.index else np.nan for name in datasets], dtype=float)
            xerr = np.vstack([np.maximum(values - lows, 0), np.maximum(highs - values, 0)])
            ax.errorbar(
                values[finite], current_y[finite], xerr=xerr[:, finite], fmt=markers[condition],
                color=COLORS[condition], ecolor=COLORS[condition], markersize=7.2,
                elinewidth=1.15, capsize=2.2, label=condition, zorder=3,
            )
        else:
            ax.scatter(values[finite], current_y[finite], s=54, marker=markers[condition], color=COLORS[condition], label=condition, zorder=3)
        for value, row_y in zip(values[finite], current_y[finite], strict=True):
            ax.text(min(value + 0.018, xlim[1] - 0.01), row_y, f"{value:.2f}", va="center", fontsize=8.8, fontweight="semibold", color="#293238")
    ax.set_yticks(y, datasets)
    ax.invert_yaxis()
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    ax.set_title(title, loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)


def plot_layer_a_overview(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    data = _preferred_layer_a(layer_a)
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 13.0), constrained_layout=True)
    _metric_forest(axes[0, 0], data, "macro_auroc_ovr", "a  Discrimination with 95% CI", "Macro AUROC", (0.35, 1.04), True)
    axes[0, 0].axvline(0.5, color="#A7ADB2", linestyle=(0, (3, 3)), linewidth=1)
    axes[0, 0].text(
        0.99, 0.01,
        "IAEAV is capture-confounded and is not treated as clinical evidence.",
        transform=axes[0, 0].transAxes, ha="right", va="bottom",
        fontsize=10.5, color="#9B5A22", fontweight="bold",
    )
    _metric_forest(axes[0, 1], data, "accuracy", "b  Fixed-threshold classification", "Accuracy", (0, 1.04))
    _metric_forest(axes[1, 0], data, "macro_f1", "c  Class-balanced performance", "Macro F1", (0, 1.04))
    _metric_forest(axes[1, 1], data, "macro_auprc", "d  Precision-recall performance", "Macro AUPRC", (0, 1.04))
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[name]) for name in ["B1", "B2", "Ours"]]
    fig.legend(handles, ["B1 traditional acoustic ML", "B2 direct transcript agent", "Ours evidence-governed fusion"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.018))
    fig.suptitle("Layer A | Medical prediction performance across independent datasets", fontsize=17, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def plot_paired_auroc_differences(layer_a: pd.DataFrame, path: Path) -> None:
    """Plot paired B3-minus-baseline AUROC effects with subject bootstrap CIs."""
    _style()
    paired = layer_a[
        layer_a["analysis_scope"].eq("paired_difference")
        & layer_a["metric"].eq("delta_macro_auroc_ovr")
        & layer_a["condition"].isin(["Ours-B1", "Ours-B2"])
    ].copy()
    if paired.empty:
        fig, ax = plt.subplots(figsize=(14, 4.5), constrained_layout=True)
        ax.axis("off")
        ax.text(0.5, 0.5, "Paired comparison not available", ha="center", va="center", fontsize=16)
        _save(fig, path)
        return
    names = list(dict.fromkeys(paired["dataset_name"]))
    fig, ax = plt.subplots(figsize=(15.5, max(6.5, len(names) * 0.72)), constrained_layout=True)
    y = np.arange(len(names), dtype=float)
    styles = [
        ("Ours-B1", 0.16, "#278C82", "B3 − B1"),
        ("Ours-B2", -0.16, "#7258B7", "B3 − B2"),
    ]
    for condition, offset, color, label in styles:
        subset = paired[paired["condition"].eq(condition)].set_index("dataset_name")
        values = np.array([pd.to_numeric(subset.loc[name, "value"], errors="coerce") if name in subset.index else np.nan for name in names], dtype=float)
        lows = np.array([pd.to_numeric(subset.loc[name, "ci_low"], errors="coerce") if name in subset.index else np.nan for name in names], dtype=float)
        highs = np.array([pd.to_numeric(subset.loc[name, "ci_high"], errors="coerce") if name in subset.index else np.nan for name in names], dtype=float)
        valid = np.isfinite(values) & np.isfinite(lows) & np.isfinite(highs)
        xerr = np.vstack([values[valid] - lows[valid], highs[valid] - values[valid]])
        ax.errorbar(values[valid], y[valid] + offset, xerr=xerr, fmt="o", color=color, ecolor=color, elinewidth=2.2, capsize=4, markersize=7.5, label=label)
        for value, ypos in zip(values[valid], y[valid] + offset, strict=True):
            ax.text(value + (0.008 if value >= 0 else -0.008), ypos, f"{value:+.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=10, fontweight="bold", color=color)
    ax.axvline(0, color="#303A43", linestyle=(0, (4, 3)), linewidth=1.3)
    ax.set_yticks(y, names, fontsize=11.5, fontweight="semibold")
    ax.invert_yaxis()
    ax.set_xlabel("Paired macro AUROC difference (B3 minus baseline); 95% subject-bootstrap CI", fontsize=12.5, fontweight="semibold")
    ax.set_title("Paired evidence for improvement on identical held-out subjects", loc="left", fontsize=18, fontweight="bold")
    ax.legend(loc="lower right", frameon=False, fontsize=11.5)
    ax.grid(axis="y", visible=False)
    _save(fig, path)


def plot_iaeav_capture_confounding(paths: ProjectPaths, path: Path) -> None:
    manifest_path = paths.artifacts / "IAEAV" / "manifest.csv"
    if not manifest_path.exists():
        return
    frame = pd.read_csv(manifest_path, dtype={"subject_id": str})
    frame = frame.drop_duplicates("subject_id", keep="last")
    if "acquisition_group" not in frame:
        frame["acquisition_group"] = frame["subject_id"].str.extract(
            r"^(inv\d+)", flags=re.IGNORECASE, expand=False
        ).str.lower().fillna("unknown")
    counts = pd.crosstab(frame["acquisition_group"], frame["label"]).sort_index()
    for label in ["HC", "AD"]:
        if label not in counts:
            counts[label] = 0
    counts = counts[["HC", "AD"]]
    _style()
    fig, ax = plt.subplots(figsize=(12.5, 6.8), constrained_layout=True)
    y = np.arange(len(counts))
    hc = counts["HC"].to_numpy(dtype=float)
    ad = counts["AD"].to_numpy(dtype=float)
    ax.barh(y, hc, color="#7FA8C9", label="Healthy control")
    ax.barh(y, ad, left=hc, color="#D7776E", label="AD")
    for row_y, healthy, disease in zip(y, hc, ad, strict=True):
        total = healthy + disease
        purity = max(healthy, disease) / total if total else np.nan
        ax.text(total + 1.5, row_y, f"n={int(total)} | label purity={purity:.0%}", va="center", fontsize=11, fontweight="semibold")
    ax.set_yticks(y, counts.index)
    ax.invert_yaxis()
    ax.set_xlabel("Number of unique participants")
    ax.set_title("IAEAV acquisition group is strongly associated with diagnosis", loc="left")
    ax.legend(ncol=2, loc="lower right")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max((hc + ad).max() * 1.42, 10))
    fig.text(
        0.01, 0.005,
        "Most acquisition groups contain one diagnosis. The current audit holds out one complete label-balanced group so interviewer identity does not overlap training and test.",
        fontsize=11, color="#5F6B70",
    )
    _save(fig, path)


def plot_safety_overview(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    data = _preferred_layer_a(layer_a)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10), constrained_layout=True)
    _metric_forest(axes[0, 0], data, "ece", "a  Calibration error", "ECE (lower is better)", (0, 0.55))
    _metric_forest(axes[0, 1], data, "multiclass_brier", "b  Probabilistic error", "Brier score (lower is better)", (0, 1.05))
    _metric_forest(axes[1, 0], data, "mcc", "c  Error balance", "Matthews correlation", (-0.18, 1.05))
    controls = layer_a[
        layer_a["analysis_scope"].eq("full_available_cohort")
        & layer_a["metric"].eq("macro_auroc_ovr")
        & layer_a["condition"].isin(["QC_only", "No_duration_no_loudness", "Ours"])
    ]
    names = list(dict.fromkeys(controls["dataset_name"]))
    y = np.arange(len(names))
    for index, condition in enumerate(["QC_only", "No_duration_no_loudness", "Ours"]):
        values = controls[controls["condition"].eq(condition)].set_index("dataset_name")["value"]
        axes[1, 1].barh(y + (index - 1) * 0.24, [values.get(name, np.nan) for name in names], height=0.22, color={"QC_only": "#C7CCD1", "No_duration_no_loudness": "#D9BD78", "Ours": COLORS["Ours"]}[condition], label=condition.replace("_", " "))
        for row_y, value in zip(y + (index - 1) * 0.24, [values.get(name, np.nan) for name in names], strict=True):
            if np.isfinite(value):
                axes[1, 1].text(min(value + 0.012, 1.02), row_y, f"{value:.2f}", va="center", fontsize=6.8)
    axes[1, 1].set_yticks(y, names)
    axes[1, 1].set_xlim(0, 1.05)
    axes[1, 1].set_xlabel("Macro AUROC")
    axes[1, 1].set_title("d  Shortcut and confounding controls", loc="left", fontweight="bold")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Layer A | Calibration, imbalance and robustness audit", fontsize=17, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def plot_screening_operating_points(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    data = _preferred_layer_a(layer_a)
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)
    panels = [
        ("specificity_at_locked_threshold", "a  Specificity at locked threshold", "Specificity"),
        ("ppv_at_locked_threshold", "b  Positive predictive value at locked threshold", "PPV"),
        ("npv_at_locked_threshold", "c  Negative predictive value at locked threshold", "NPV"),
    ]
    for ax, (metric, title, xlabel) in zip(axes.flat[:3], panels, strict=True):
        _metric_forest(ax, data, metric, title, xlabel, (0, 1.04))

    auc = data[data["metric"].eq("macro_auroc_ovr")]
    datasets = list(dict.fromkeys(auc["dataset_name"]))
    indexed = {
        condition: auc[auc["condition"].eq(condition)].set_index("dataset_name")["value"]
        for condition in ["B1", "B2", "Ours"]
    }
    y = np.arange(len(datasets), dtype=float)
    for offset, baseline in [(0.14, "B1"), (-0.14, "B2")]:
        values = np.array(
            [indexed["Ours"].get(name, np.nan) - indexed[baseline].get(name, np.nan) for name in datasets],
            dtype=float,
        )
        colors = [COLORS["Ours"] if np.isfinite(value) and value >= 0 else "#C6CBC9" for value in values]
        axes[1, 1].barh(y + offset, values, height=0.25, color=colors, alpha=1.0 if baseline == "B1" else 0.72, label=f"Ours − {baseline}")
        for row_y, value in zip(y + offset, values, strict=True):
            if np.isfinite(value):
                axes[1, 1].text(value + (0.008 if value >= 0 else -0.008), row_y, f"{value:+.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=7)
    axes[1, 1].axvline(0, color="#59656B", linestyle=(0, (3, 3)), linewidth=1)
    axes[1, 1].set_yticks(y, datasets)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(-0.45, 0.45)
    axes[1, 1].set_xlabel("Macro AUROC difference")
    axes[1, 1].set_title("d  Increment over each baseline", loc="left", fontweight="bold")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[name]) for name in ["B1", "B2", "Ours"]]
    fig.legend(handles, ["B1 traditional acoustic ML", "B2 direct transcript agent", "Ours evidence-governed fusion"], loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.018))
    fig.suptitle("Layer A | Screening operating points and incremental value", fontsize=17, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def _metric_series(data: pd.DataFrame, metric: str, condition: str) -> pd.Series:
    subset = data[data["metric"].eq(metric) & data["condition"].eq(condition)]
    return subset.drop_duplicates("dataset_name", keep="last").set_index("dataset_name")["value"]


def plot_framework_revision(path: Path) -> None:
    """Show the methodological changes without mixing in engineering/run status."""
    _style()
    fig, ax = plt.subplots(figsize=(16, 7.4), constrained_layout=True)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    def box(x: float, y: float, w: float, h: float, title: str, body: str, fill: str, edge: str) -> None:
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.025,rounding_size=0.08", facecolor=fill, edgecolor=edge, linewidth=1.5))
        ax.text(x + 0.18, y + h - 0.33, title, fontsize=11.5, fontweight="bold", va="top")
        ax.text(x + 0.18, y + h - 0.78, body, fontsize=8.8, va="top", color="#445158", linespacing=1.35)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=13, linewidth=1.35, color="#667177"))

    ax.text(0.2, 7.62, "Framework revision implemented before the full rerun", fontsize=19, fontweight="bold")
    ax.text(0.2, 7.20, "The revision changes evidence validity and evaluation integrity; it does not alter labels or manufacture performance.", fontsize=10.5, color="#647178")
    titles = [
        ("1  Identity-safe data units", "Unique case IDs across tasks\nSubject-grouped train/test split\nLong recordings retained for NCMMSC"),
        ("2  Task-specific reference", "Diagnosis: configured control class\nProgression: no-decline reference\nFold-internal robust calibration"),
        ("3  Channel-specific evidence", "Value + direction + reliability\nMissingness + confound tags\nReport permission by channel"),
        ("4  Reliability-aware fusion", "Within-state evidence aggregation\nFold-internal QC residualization\nLearned branch allocation"),
        ("5  Evidence-governed Agent", "Agent audits support and counter-evidence\nBounded train-only risk correction\nLocked trace reused for the report"),
    ]
    fills = ["#EDF3FA", "#F0ECFA", "#E9F5F2", "#FFF3E2", "#FCEBE9"]
    edges = ["#6F98C9", "#8D77C9", "#58A498", "#D7A047", "#D77A70"]
    for index, (title, body) in enumerate(titles):
        x = 0.2 + index * 3.14
        box(x, 3.75, 2.68, 2.55, title, body, fills[index], edges[index])
        if index < len(titles) - 1:
            arrow(x + 2.70, 5.02, x + 3.10, 5.02)
    ax.text(0.2, 3.12, "What these changes prevent", fontsize=12.5, fontweight="bold")
    safeguards = [
        "task collisions\nand subject leakage",
        "wrong normal reference\nin progression tasks",
        "device or ASR artifacts\nentering clinical claims",
        "QC becoming a direct\ndisease-risk shortcut",
        "free-form Agent reasoning\ndetached from prediction",
    ]
    for index, text_value in enumerate(safeguards):
        x = 0.2 + index * 3.14
        box(x, 1.05, 2.68, 1.55, "", text_value, "#F7F8F7", "#C8CFCC")
        arrow(x + 1.34, 3.68, x + 1.34, 2.66)
    _save(fig, path)


def plot_medical_standards_summary(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    data = _preferred_layer_a(layer_a)
    fig = plt.figure(figsize=(18, 11.7), constrained_layout=True)
    grid = fig.add_gridspec(2, 3, width_ratios=[1.15, 1.0, 1.0])
    axes = [fig.add_subplot(grid[row, col]) for row in range(2) for col in range(3)]
    _metric_forest(axes[0], data, "macro_auroc_ovr", "a  Discrimination with 95% CI", "Macro AUROC", (0.35, 1.04), True)
    axes[0].axvline(0.5, color="#9DA6A9", linestyle=(0, (3, 3)), linewidth=1)
    _metric_forest(axes[1], data, "specificity_at_locked_threshold", "b  Locked screening operating point", "Specificity at locked threshold", (0, 1.04))

    auc = data[data["metric"].eq("macro_auroc_ovr")]
    names = list(dict.fromkeys(auc["dataset_name"]))
    ours = _metric_series(data, "macro_auroc_ovr", "Ours")
    y = np.arange(len(names), dtype=float)
    for offset, baseline in [(0.15, "B1"), (-0.15, "B2")]:
        base = _metric_series(data, "macro_auroc_ovr", baseline)
        values = np.array([ours.get(name, np.nan) - base.get(name, np.nan) for name in names], dtype=float)
        axes[2].barh(y + offset, values, height=0.27, color=COLORS["Ours"], alpha=1.0 if baseline == "B1" else 0.62, label=f"Ours − {baseline}")
        for yy, value in zip(y + offset, values, strict=True):
            if np.isfinite(value):
                axes[2].text(value + (0.008 if value >= 0 else -0.008), yy, f"{value:+.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=7)
    axes[2].axvline(0, color="#727C80", linestyle=(0, (3, 3)), linewidth=1)
    axes[2].set_yticks(y, names)
    axes[2].invert_yaxis()
    axes[2].set_xlim(-0.48, 0.48)
    axes[2].set_xlabel("Macro AUROC difference")
    axes[2].set_title("c  Increment over each baseline", loc="left", fontweight="bold")
    axes[2].legend(fontsize=8, loc="lower right")

    metrics = [("accuracy", "Accuracy"), ("mcc", "MCC"), ("macro_auprc", "AUPRC")]
    x = np.arange(len(metrics))
    for index, condition in enumerate(["B1", "B2", "Ours"]):
        values = [float(_metric_series(data, metric, condition).mean()) for metric, _ in metrics]
        bars = axes[3].bar(x + (index - 1) * 0.24, values, width=0.22, color=COLORS[condition], label=condition)
        axes[3].bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    axes[3].set_xticks(x, [label for _, label in metrics])
    axes[3].set_ylim(-0.15, 1.05)
    axes[3].set_ylabel("Cross-dataset mean")
    axes[3].set_title("d  Fixed-threshold and imbalance-aware performance", loc="left", fontweight="bold")
    axes[3].legend(fontsize=8)

    controls = layer_a[layer_a["analysis_scope"].eq("full_available_cohort") & layer_a["metric"].eq("macro_auroc_ovr")]
    control_names = list(dict.fromkeys(controls["dataset_name"]))
    full_ours = _metric_series(controls, "macro_auroc_ovr", "Ours")
    qc = _metric_series(controls, "macro_auroc_ovr", "QC_only")
    reduced = _metric_series(controls, "macro_auroc_ovr", "No_duration_no_loudness")
    y_control = np.arange(len(control_names), dtype=float)
    margins = [("Ours − QC-only", [full_ours.get(name, np.nan) - qc.get(name, np.nan) for name in control_names], "#638C88"), ("Ours − no duration/loudness", [full_ours.get(name, np.nan) - reduced.get(name, np.nan) for name in control_names], "#D7A047")]
    for offset, (label, values, color) in zip([0.15, -0.15], margins, strict=True):
        axes[4].barh(y_control + offset, values, height=0.27, color=color, label=label)
    axes[4].axvline(0, color="#727C80", linestyle=(0, (3, 3)), linewidth=1)
    axes[4].set_yticks(y_control, control_names)
    axes[4].invert_yaxis()
    axes[4].set_xlabel("AUROC margin")
    axes[4].set_title("e  Shortcut-control margin", loc="left", fontweight="bold")
    axes[4].legend(fontsize=7.5)

    completed = data[data["condition"].isin(["B1", "B2", "Ours"])].pivot_table(index="dataset_name", columns="condition", values="value", aggfunc="count")
    completed = completed.reindex(index=names, columns=["B1", "B2", "Ours"]).fillna(0).gt(0).astype(int)
    axes[5].imshow(completed.to_numpy(), aspect="auto", vmin=0, vmax=1, cmap=matplotlib.colors.ListedColormap(["#ECEFED", COLORS["Ours"]]))
    axes[5].set_xticks(np.arange(3), ["B1", "B2", "Ours"])
    axes[5].set_yticks(np.arange(len(names)), names)
    axes[5].set_title("f  Three-condition evaluation coverage", loc="left", fontweight="bold")
    for row in range(len(names)):
        for col in range(3):
            axes[5].text(col, row, "RUN" if completed.iloc[row, col] else "—", ha="center", va="center", color="white" if completed.iloc[row, col] else "#7B8589", fontsize=8, fontweight="bold")
    fig.suptitle("Layer A | Medical prediction and screening standards", fontsize=19, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def plot_error_imbalance(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    data = _preferred_layer_a(layer_a)
    fig, axes = plt.subplots(2, 2, figsize=(16, 10.5), constrained_layout=True)
    panels = [
        ("error_rate", "a  Overall error rate", "Error rate", (0, 0.75)),
        ("macro_fnr", "b  False-negative rate", "Macro FNR", (0, 0.85)),
        ("macro_fpr", "c  False-positive rate", "Macro FPR", (0, 0.85)),
        ("positive_prevalence", "d  Positive-class prevalence", "Prevalence", (0, 1.0)),
    ]
    for ax, (metric, title, xlabel, limits) in zip(axes.flat, panels, strict=True):
        _metric_forest(ax, data, metric, title, xlabel, limits)
    fig.suptitle("Layer A-4 | Error structure and class imbalance", fontsize=18, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def plot_robustness_controls(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    controls = layer_a[layer_a["analysis_scope"].eq("full_available_cohort")]
    fig, axes = plt.subplots(1, 2, figsize=(16, 6.7), constrained_layout=True)
    names = list(dict.fromkeys(controls[controls["metric"].eq("macro_auroc_ovr")]["dataset_name"]))
    ours = _metric_series(controls, "macro_auroc_ovr", "Ours")
    y = np.arange(len(names), dtype=float)
    for offset, condition, color, marker in [
        (-0.20, "Ours", COLORS["Ours"], "D"),
        (0.00, "QC_only", "#AEB5B6", "o"),
        (0.20, "No_duration_no_loudness", "#D7A047", "s"),
    ]:
        indexed = _metric_series(controls, "macro_auroc_ovr", condition)
        values = np.array([indexed.get(name, np.nan) for name in names], dtype=float)
        finite = np.isfinite(values)
        axes[0].scatter(values[finite], y[finite] + offset, color=color, marker=marker, s=38, label=condition.replace("_", " "), zorder=3)
        for value, yy in zip(values[finite], y[finite] + offset, strict=True):
            axes[0].text(min(value + 0.014, 1.02), yy, f"{value:.2f}", va="center", fontsize=7)
    axes[0].axvline(0.5, color="#9DA6A9", linestyle=(0, (3, 3)), linewidth=1)
    axes[0].set_yticks(y, names)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0.30, 1.04)
    axes[0].set_xlabel("Macro AUROC")
    axes[0].set_title("a  Full model and negative controls", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8, loc="lower right")
    for offset, condition, color in [(0.15, "QC_only", "#AEB5B6"), (-0.15, "No_duration_no_loudness", "#D7A047")]:
        baseline = _metric_series(controls, "macro_auroc_ovr", condition)
        values = np.array([ours.get(name, np.nan) - baseline.get(name, np.nan) for name in names], dtype=float)
        axes[1].barh(y + offset, values, height=0.27, color=color, label=f"Ours − {condition.replace('_', ' ')}")
        for yy, value in zip(y + offset, values, strict=True):
            if np.isfinite(value):
                axes[1].text(value + (0.007 if value >= 0 else -0.007), yy, f"{value:+.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=7)
    axes[1].axvline(0, color="#727C80", linestyle=(0, (3, 3)), linewidth=1)
    axes[1].set_yticks(y, names)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("AUROC margin")
    axes[1].set_title("b  Signal retained beyond shortcut controls", loc="left", fontweight="bold")
    axes[1].legend(fontsize=8)
    fig.suptitle("Layer A-5 | Robustness and shortcut-control audit", fontsize=18, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def plot_layer_b_overview(layer_b: pd.DataFrame, path: Path) -> None:
    _style()
    ours = layer_b[layer_b["condition"].eq("Ours")].copy()
    datasets = list(dict.fromkeys(ours["dataset_name"]))
    fig, axes = plt.subplots(2, 2, figsize=(15.5, 10.5), constrained_layout=True)

    structural = [
        "MetricEvidence completeness", "StateCard completeness", "branch contribution trace",
        "report-permission audit", "source identifier privacy audit", "evidence-span faithfulness",
    ]
    short = ["Metric\nevidence", "State\ncards", "Branch\ntrace", "Report\npermission", "Privacy", "Source\nspan"]
    x = np.arange(len(structural))
    per_dataset = []
    for dataset in datasets:
        values = []
        for check in structural:
            row = ours[ours["dataset_name"].eq(dataset) & ours["check"].eq(check)]
            values.append(float(row.iloc[0]["value"]) if not row.empty else np.nan)
        per_dataset.append(values)
        axes[0, 0].plot(x, values, color="#A9B2B0", alpha=0.42, linewidth=0.9, marker="o", markersize=2.8)
    mean_values = np.nanmean(np.asarray(per_dataset, dtype=float), axis=0)
    axes[0, 0].plot(x, mean_values, color=COLORS["Ours"], linewidth=2.5, marker="D", markersize=5.5, label="Cross-dataset mean")
    axes[0, 0].set_xticks(x, short)
    axes[0, 0].set_ylim(0, 1.06)
    axes[0, 0].set_ylabel("Audit completeness")
    axes[0, 0].set_title("a  Evidence and traceability contract", loc="left", fontweight="bold")
    axes[0, 0].legend(loc="lower left", fontsize=8)

    def dataset_bars(ax: plt.Axes, check: str, title: str, xlabel: str, xlim: tuple[float, float], center: float | None = None) -> None:
        values = []
        passed = []
        for dataset in datasets:
            row = ours[ours["dataset_name"].eq(dataset) & ours["check"].eq(check)]
            values.append(float(row.iloc[0]["value"]) if not row.empty else np.nan)
            passed.append(bool(row.iloc[0]["passed"]) if not row.empty else False)
        y = np.arange(len(datasets))
        colors = [COLORS["Ours"] if flag else "#C9CECC" for flag in passed]
        ax.barh(y, values, height=0.58, color=colors)
        if center is not None:
            ax.axvline(center, color="#7E888D", linestyle=(0, (3, 3)), linewidth=1)
        ax.set_yticks(y, datasets)
        ax.invert_yaxis()
        ax.set_xlim(*xlim)
        ax.set_xlabel(xlabel)
        ax.set_title(title, loc="left", fontweight="bold")
        for row_y, value in zip(y, values, strict=True):
            if np.isfinite(value):
                ax.text(value + 0.012 * (xlim[1] - xlim[0]), row_y, f"{value:.2f}", va="center", fontsize=7)

    dataset_bars(
        axes[0, 1],
        "reference-state intervention on errors",
        "b  Reference-state intervention on errors",
        "Non-decreasing true-class probability",
        (0, 1.06),
    )
    dataset_bars(axes[1, 0], "concept-only vs full fusion", "c  Added value beyond clinical states", "Full minus concept-only macro AUROC", (-0.25, 0.45), center=0.0)

    report_rows = layer_b[layer_b["check"].eq("clinical report rubric /25")]
    y = np.arange(len(datasets))
    for offset, condition in [(0.17, "B2"), (-0.17, "Ours")]:
        indexed = report_rows[report_rows["condition"].eq(condition)].set_index("dataset_name")["value"]
        values = [indexed.get(dataset, np.nan) for dataset in datasets]
        axes[1, 1].barh(y + offset, values, height=0.31, color=COLORS[condition], label=condition)
        for row_y, value in zip(y + offset, values, strict=True):
            if np.isfinite(value):
                axes[1, 1].text(min(value + 0.3, 25.2), row_y, f"{value:.1f}", va="center", fontsize=7)
    axes[1, 1].set_yticks(y, datasets)
    axes[1, 1].invert_yaxis()
    axes[1, 1].set_xlim(0, 25.8)
    axes[1, 1].set_xlabel("Automated report audit (/25)")
    axes[1, 1].set_title("d  Clinical-report structure", loc="left", fontweight="bold")
    axes[1, 1].legend(loc="lower right", fontsize=8)
    fig.suptitle("Layer B | Framework-specific traceability and clinical-communication checks", fontsize=17, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def _report_dimension_means(datasets: list[dict[str, Any]], key: str) -> list[float]:
    dimensions = ["evidence_completeness", "clinical_interpretability", "safety_calibration", "diagnostic_usefulness", "traceability"]
    output = []
    for dimension in dimensions:
        values = [row.get("summary", {}).get("layer_b", {}).get(key, {}).get(dimension, np.nan) for row in datasets]
        finite = [float(value) for value in values if value is not None and np.isfinite(value)]
        output.append(float(np.mean(finite)) if finite else np.nan)
    return output


def plot_layer_b_comprehensive(layer_a: pd.DataFrame, layer_b: pd.DataFrame, datasets: list[dict[str, Any]], path: Path) -> None:
    _style()
    data_a = _preferred_layer_a(layer_a)
    ours_b = layer_b[layer_b["condition"].eq("Ours")]
    names = list(dict.fromkeys(ours_b["dataset_name"]))
    fig, axes = plt.subplots(2, 2, figsize=(18, 14.5), constrained_layout=True)

    structural = ["MetricEvidence completeness", "StateCard completeness", "branch contribution trace", "report-permission audit", "evidence-span faithfulness"]
    structural_labels = ["Metric evidence", "State cards", "Branch trace", "Report permission", "Source span"]
    structural_matrix = np.full((len(names), len(structural)), np.nan)
    for row_index, name in enumerate(names):
        dataset_checks = ours_b[ours_b["dataset_name"].eq(name)].drop_duplicates("check", keep="last")
        indexed = dataset_checks.set_index("check")["value"]
        structural_matrix[row_index] = [pd.to_numeric(indexed.get(check, np.nan), errors="coerce") for check in structural]
    image = axes[0, 0].imshow(structural_matrix, aspect="auto", vmin=0, vmax=1, cmap="YlGn")
    for row_index in range(len(names)):
        for column_index in range(len(structural)):
            value = structural_matrix[row_index, column_index]
            if np.isfinite(value):
                axes[0, 0].text(column_index, row_index, f"{value:.2f}", ha="center", va="center", fontsize=9.5, fontweight="bold")
    axes[0, 0].set_xticks(np.arange(len(structural_labels)), structural_labels, rotation=22, ha="right")
    axes[0, 0].set_yticks(np.arange(len(names)), names)
    axes[0, 0].set_title("a  Evidence contract by dataset", loc="left")
    axes[0, 0].grid(False)
    fig.colorbar(image, ax=axes[0, 0], fraction=0.035, pad=0.02, label="Completeness / faithfulness")

    report_rows = layer_b[layer_b["check"].eq("clinical report rubric /25")]
    y = np.arange(len(names), dtype=float)
    report_series = {
        condition: report_rows[report_rows["condition"].eq(condition)]
        .drop_duplicates("dataset_name", keep="last")
        .set_index("dataset_name")["value"]
        for condition in ["B2", "Ours"]
    }
    for row_index, name in enumerate(names):
        b2_value = float(report_series["B2"].get(name, np.nan))
        ours_value = float(report_series["Ours"].get(name, np.nan))
        if np.isfinite(b2_value) and np.isfinite(ours_value):
            axes[0, 1].plot([b2_value, ours_value], [row_index, row_index], color="#B8C0BD", linewidth=2, zorder=1)
        if np.isfinite(b2_value):
            axes[0, 1].scatter(b2_value, row_index, s=65, color=COLORS["B2"], marker="s", zorder=3)
            axes[0, 1].text(b2_value - 0.35, row_index, f"{b2_value:.1f}", ha="right", va="center", fontsize=9.5)
        if np.isfinite(ours_value):
            axes[0, 1].scatter(ours_value, row_index, s=70, color=COLORS["Ours"], marker="D", zorder=3)
            axes[0, 1].text(ours_value + 0.35, row_index, f"{ours_value:.1f}", ha="left", va="center", fontsize=9.5, fontweight="bold")
    axes[0, 1].set_yticks(y, names)
    axes[0, 1].invert_yaxis()
    axes[0, 1].set_xlim(0, 26)
    axes[0, 1].set_xlabel("Automated report-structure audit (/25)")
    axes[0, 1].set_title("b  Direct agent versus evidence-governed Agent", loc="left")
    axes[0, 1].scatter([], [], color=COLORS["B2"], marker="s", label="B2 direct agent")
    axes[0, 1].scatter([], [], color=COLORS["Ours"], marker="D", label="B3 ADvoice")
    axes[0, 1].legend(loc="lower right")
    axes[0, 1].grid(axis="y", visible=False)

    auc = data_a[data_a["metric"].eq("macro_auroc_ovr")]
    auc_names = list(dict.fromkeys(auc["dataset_name"]))
    ours_auc = _metric_series(data_a, "macro_auroc_ovr", "Ours")
    y_auc = np.arange(len(auc_names), dtype=float)
    for offset, baseline in [(0.15, "B1"), (-0.15, "B2")]:
        base = _metric_series(data_a, "macro_auroc_ovr", baseline)
        values = np.array([ours_auc.get(name, np.nan) - base.get(name, np.nan) for name in auc_names], dtype=float)
        bars = axes[1, 0].barh(y_auc + offset, values, height=0.27, color=COLORS["Ours"] if baseline == "B1" else "#67B8AF", label=f"B3 − {baseline}")
        for bar, value in zip(bars, values, strict=True):
            if np.isfinite(value):
                axes[1, 0].text(value + (0.008 if value >= 0 else -0.008), bar.get_y() + bar.get_height()/2, f"{value:+.2f}", ha="left" if value >= 0 else "right", va="center", fontsize=9.2, fontweight="semibold")
    axes[1, 0].axvline(0, color="#727C80", linestyle=(0, (3, 3)), linewidth=1)
    axes[1, 0].set_yticks(y_auc, auc_names)
    axes[1, 0].invert_yaxis()
    axes[1, 0].set_xlabel("Macro AUROC difference")
    axes[1, 0].set_title("c  Predictive increment over baselines", loc="left")
    axes[1, 0].legend(loc="lower right")
    axes[1, 0].grid(axis="y", visible=False)

    mechanism_checks = [
        "reference-state intervention on errors",
        "concept-only vs full fusion",
        "task-specific state ablation",
        "evidence-span faithfulness",
    ]
    mechanism_labels = ["State correction", "Full − state-only", "Task routing gain", "Source-span trace"]
    value_matrix = np.full((len(names), len(mechanism_checks)), np.nan)
    pass_matrix = np.full_like(value_matrix, np.nan)
    for row_index, name in enumerate(names):
        dataset_checks = ours_b[ours_b["dataset_name"].eq(name)].drop_duplicates("check", keep="last").set_index("check")
        for column_index, check in enumerate(mechanism_checks):
            if check not in dataset_checks.index:
                continue
            record = dataset_checks.loc[check]
            value_matrix[row_index, column_index] = pd.to_numeric(record["value"], errors="coerce")
            passed = record["passed"]
            pass_matrix[row_index, column_index] = float(passed) if pd.notna(passed) else np.nan
    masked = np.ma.masked_invalid(pass_matrix)
    pass_image = axes[1, 1].imshow(masked, aspect="auto", vmin=0, vmax=1, cmap="RdYlGn")
    for row_index in range(len(names)):
        for column_index in range(len(mechanism_checks)):
            value = value_matrix[row_index, column_index]
            if np.isfinite(value):
                axes[1, 1].text(column_index, row_index, f"{value:+.2f}" if column_index in {1, 2} else f"{value:.2f}", ha="center", va="center", fontsize=9.2, fontweight="bold")
    axes[1, 1].set_xticks(np.arange(len(mechanism_labels)), mechanism_labels, rotation=20, ha="right")
    axes[1, 1].set_yticks(np.arange(len(names)), names)
    axes[1, 1].set_title("d  Mechanism and trace checks by dataset", loc="left")
    axes[1, 1].grid(False)
    fig.colorbar(pass_image, ax=axes[1, 1], fraction=0.035, pad=0.02, ticks=[0, 1], label="Pre-specified check")
    fig.suptitle("Layer B | Does the framework add auditable clinical evidence?", fontsize=21, fontweight="bold", x=0.01, ha="left")
    _save(fig, path)


def plot_failure_mode_audit(layer_a: pd.DataFrame, layer_b: pd.DataFrame, path: Path) -> None:
    _style()
    data = _preferred_layer_a(layer_a)
    names = list(dict.fromkeys(data[data["condition"].eq("Ours")]["dataset_name"]))
    ours_ece = _metric_series(data, "ece", "Ours")
    ours_spec = _metric_series(data, "specificity_at_locked_threshold", "Ours")
    ours_hce = _metric_series(data, "high_confidence_error_rate", "Ours")
    full = layer_a[layer_a["analysis_scope"].eq("full_available_cohort")]
    ours_auc = _metric_series(full, "macro_auroc_ovr", "Ours")
    qc_auc = _metric_series(full, "macro_auroc_ovr", "QC_only")
    span = layer_b[layer_b["condition"].eq("Ours") & layer_b["check"].eq("evidence-span faithfulness")].drop_duplicates("dataset_name", keep="last").set_index("dataset_name")["value"]
    values = []
    for name in names:
        shortcut = ours_auc.get(name, np.nan) - qc_auc.get(name, np.nan)
        values.append([
            1.0 if np.isfinite(shortcut) and shortcut >= 0.05 else (0.0 if np.isfinite(shortcut) else np.nan),
            1.0 if np.isfinite(ours_ece.get(name, np.nan)) and ours_ece.get(name) <= 0.10 else (0.0 if np.isfinite(ours_ece.get(name, np.nan)) else np.nan),
            1.0 if np.isfinite(ours_spec.get(name, np.nan)) and ours_spec.get(name) >= 0.70 else (0.0 if np.isfinite(ours_spec.get(name, np.nan)) else np.nan),
            1.0 if np.isfinite(ours_hce.get(name, np.nan)) and ours_hce.get(name) <= 0.05 else (0.0 if np.isfinite(ours_hce.get(name, np.nan)) else np.nan),
            1.0 if np.isfinite(span.get(name, np.nan)) and span.get(name) >= 0.90 else (0.0 if np.isfinite(span.get(name, np.nan)) else np.nan),
        ])
    matrix = np.asarray(values, dtype=float)
    cmap = matplotlib.colors.ListedColormap(["#E7B06B", "#3C9585"])
    cmap.set_bad("#E8ECEA")
    fig, ax = plt.subplots(figsize=(13.5, max(5.5, len(names) * 0.55 + 2.2)), constrained_layout=True)
    ax.imshow(np.ma.masked_invalid(matrix), aspect="auto", vmin=0, vmax=1, cmap=cmap)
    labels = ["Shortcut gap\n≥ 0.05", "ECE\n≤ 0.10", "Specificity@S85\n≥ 0.70", "High-conf. error\n≤ 0.05", "Evidence span\n≥ 0.90"]
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_yticks(np.arange(len(names)), names)
    ax.set_title("Failure-mode audit after the current framework revision", loc="left", fontsize=16, fontweight="bold")
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax.text(col, row, "PASS" if value == 1 else ("WATCH" if value == 0 else "NA"), ha="center", va="center", color="white" if np.isfinite(value) else "#69757A", fontsize=8, fontweight="bold")
    ax.grid(False)
    _save(fig, path)


def plot_report_rubric(datasets: list[dict[str, Any]], path: Path) -> None:
    _style()
    dimensions = ["evidence_completeness", "clinical_interpretability", "safety_calibration", "diagnostic_usefulness", "traceability"]
    labels = ["Evidence", "Interpretability", "Safety / calibration", "Clinical usefulness", "Traceability"]
    values = {"B2": [], "Ours": []}
    for condition, key in [("B2", "b2_report_rubric"), ("Ours", "ours_report_rubric")]:
        for dimension in dimensions:
            scores = [row.get("summary", {}).get("layer_b", {}).get(key, {}).get(dimension, np.nan) for row in datasets]
            finite = [float(value) for value in scores if value is not None and np.isfinite(value)]
            values[condition].append(float(np.mean(finite)) if finite else np.nan)
    fig, ax = plt.subplots(figsize=(10.5, 5.5), constrained_layout=True)
    y = np.arange(len(labels))
    ax.barh(y + 0.18, values["B2"], height=0.34, color=COLORS["B2"], label="B2 direct agent")
    ax.barh(y - 0.18, values["Ours"], height=0.34, color=COLORS["Ours"], label="B3 evidence diagnostic Agent")
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 5.2)
    ax.axvline(4.0, color="#6D7780", linestyle="--", linewidth=1)
    ax.set_xlabel("Automated structural audit score (0–5)")
    ax.set_title("Clinical-report structure audit (not physician validation)", loc="left", fontweight="bold")
    ax.legend(loc="lower right")
    for condition, offset in [("B2", 0.18), ("Ours", -0.18)]:
        for index, value in enumerate(values[condition]):
            if np.isfinite(value):
                ax.text(value + 0.06, index + offset, f"{value:.2f}", va="center", fontsize=8)
    _save(fig, path)


def plot_dataset_landscape(datasets: list[dict[str, Any]], path: Path) -> None:
    _style()
    completed = [row for row in datasets if row.get("status") == "completed"]
    completed.sort(key=lambda row: row["subjects"])
    fig, ax = plt.subplots(figsize=(11.5, 6.2), constrained_layout=True)
    y = np.arange(len(completed))
    channel_names = list(dict.fromkeys(row["channel"] for row in completed))
    palette = ["#6E9ED4", "#D98076", "#74A883", "#D6A343", "#8875C9", "#59A4A0"]
    channel_color = {name: palette[index % len(palette)] for index, name in enumerate(channel_names)}
    bars = ax.barh(y, [row["subjects"] for row in completed], color=[channel_color[row["channel"]] for row in completed], height=0.66)
    ax.set_yticks(y, [row["short_name"] for row in completed])
    ax.set_xscale("log")
    ax.set_xlabel("Independent subjects (log scale)")
    ax.set_title("Dataset landscape by channel and analysis unit", loc="left", fontsize=16, fontweight="bold")
    for bar, row in zip(bars, completed, strict=True):
        ax.text(bar.get_width() * 1.06, bar.get_y() + bar.get_height() / 2, f"n={row['subjects']}; test={row['test_subjects']}", va="center", fontsize=8)
    handles = [plt.Rectangle((0, 0), 1, 1, color=channel_color[name]) for name in channel_names]
    ax.legend(handles, [name.replace("_", " ") for name in channel_names], loc="lower right", fontsize=8)
    _save(fig, path)


def plot_channel_state_matrix(paths: ProjectPaths, datasets: list[dict[str, Any]], path: Path) -> None:
    _style()
    completed = [row for row in datasets if row.get("status") == "completed"]
    profiles = list(dict.fromkeys(row["channel_profile"] for row in completed))
    all_states = load_all(completed[0]["dataset_id"])["states"]["states"] if completed else []
    state_catalog = json_load(paths.root / "configs" / "states" / "audio_states.json", {})
    if not state_catalog:
        import yaml
        state_catalog = yaml.safe_load((paths.root / "configs" / "states" / "audio_states.yaml").read_text(encoding="utf-8"))
    state_ids = [item["id"] for item in state_catalog.get("states", all_states)]
    matrix = np.zeros((len(profiles), len(state_ids)))
    for row, profile in enumerate(profiles):
        dataset_id = next(item["dataset_id"] for item in completed if item["channel_profile"] == profile)
        enabled = {item["id"] for item in load_all(dataset_id)["states"]["states"]}
        matrix[row] = [1.0 if state in enabled else 0.0 for state in state_ids]
    fig, ax = plt.subplots(figsize=(14.5, max(4.5, 0.65 * len(profiles))), constrained_layout=True)
    ax.imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#ECEFED", "#278C82"]), vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(state_ids)), state_ids)
    ax.set_yticks(range(len(profiles)), [name.replace("_", " ") for name in profiles])
    for i in range(len(profiles)):
        for j in range(len(state_ids)):
            ax.text(j, i, "●" if matrix[i, j] else "–", ha="center", va="center", color="white" if matrix[i, j] else "#8A9398", fontsize=9)
    ax.set_title("Channel-specific clinical-state availability", loc="left", fontsize=16, fontweight="bold")
    ax.set_xlabel("Clinical state identifier")
    _save(fig, path)


def plot_metric_state_map(paths: ProjectPaths, path: Path) -> None:
    _style()
    import yaml

    catalog = yaml.safe_load(
        (paths.root / "configs" / "states" / "audio_states.yaml").read_text(encoding="utf-8")
    )["states"]
    metric_labels = {
        "silence_fraction": "silence fraction",
        "long_pause_rate_min": "long pauses/min",
        "pause_mean_sec": "mean pause",
        "pause_p90_sec": "pause P90",
        "voiced_fraction": "voiced fraction",
        "speech_run_mean_sec": "mean speech run",
        "speech_run_rate_min": "speech runs/min",
        "speech_rate_wpm": "speech rate",
        "speech_run_cv": "speech-run variability",
        "rms_db_mean": "mean intensity",
        "rms_db_std": "intensity variability",
        "f0_median_hz": "median F0",
        "f0_iqr_hz": "F0 IQR",
        "zcr_mean": "zero-crossing rate",
        "spectral_centroid_mean": "spectral centroid",
        "spectral_bandwidth_mean": "spectral bandwidth",
        "spectral_rolloff_mean": "spectral roll-off",
        "spectral_flatness_mean": "spectral flatness",
        "pronoun_ratio": "pronoun ratio",
        "content_word_ratio": "content-word ratio",
        "lexical_mattr50": "MATTR-50",
        "lexical_ttr": "type-token ratio",
        "mean_utterance_words": "mean utterance length",
        "filler_rate_100w": "fillers/100 words",
        "repair_rate_100w": "repairs/100 words",
        "patient_turn_share": "patient turn share",
    }
    branch_labels = {
        "speech_behavior": "Speech behaviour",
        "language": "Language",
        "interaction": "Interaction",
        "auxiliary_acoustic": "Auxiliary acoustic",
        "task_performance": "Task performance",
    }
    fig, ax = plt.subplots(figsize=(17.2, 12.3), constrained_layout=True)
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.15, len(catalog) + 1.45)
    ax.axis("off")
    ax.text(0.0, len(catalog) + 1.20, "Clinical-state dictionary and within-state evidence fusion", fontsize=17, fontweight="bold", va="top")
    ax.text(0.0, len(catalog) + 0.72, "State", fontsize=10, fontweight="bold", color="#39454B")
    ax.text(0.29, len(catalog) + 0.72, "Clinical branch", fontsize=10, fontweight="bold", color="#39454B")
    ax.text(0.48, len(catalog) + 0.72, "Metric composition (pre-specified clinical weight)", fontsize=10, fontweight="bold", color="#39454B")
    ax.text(0.92, len(catalog) + 0.72, "Status", fontsize=10, fontweight="bold", color="#39454B")
    ax.plot([0, 1], [len(catalog) + 0.52] * 2, color="#8D979B", linewidth=1.1)

    for index, definition in enumerate(catalog):
        y = len(catalog) - index - 0.05
        if index % 2:
            ax.axhspan(y - 0.45, y + 0.45, color="#F6F8F7", zorder=0)
        state_id = definition["id"]
        branch = definition["branch"]
        color = BRANCH_COLORS.get(branch, "#8A927B")
        ax.add_patch(plt.Rectangle((0.0, y - 0.27), 0.008, 0.54, color=color, linewidth=0))
        ax.text(0.018, y + 0.11, state_id, fontsize=8.8, fontweight="bold", va="center", color="#263238")
        ax.text(0.062, y + 0.11, STATE_NAMES_EN.get(state_id, state_id), fontsize=8.8, va="center", color="#263238")
        ax.text(0.29, y + 0.11, branch_labels.get(branch, branch.replace("_", " ")), fontsize=8.2, va="center", color=color, fontweight="bold")
        metrics = definition.get("metrics", [])
        weights = definition.get("weights", [])
        if metrics:
            composition = "  ·  ".join(
                f"{metric_labels.get(metric, metric.replace('_', ' '))} {weight:.2f}"
                for metric, weight in zip(metrics, weights, strict=True)
            )
            status = "Specified"
            status_color = "#217C73"
        else:
            composition = "No validated task scorer or cross-language algorithm in the executable pipeline"
            status = "Planned"
            status_color = "#A46E20"
        ax.text(0.48, y + 0.11, textwrap.fill(composition, width=76), fontsize=7.6, va="center", color="#46535A", linespacing=1.25)
        ax.text(0.92, y + 0.11, status, fontsize=8.2, va="center", color=status_color, fontweight="bold")
        ax.plot([0, 1], [y - 0.45] * 2, color="#E1E5E3", linewidth=0.65)

    ax.text(
        0.0,
        -0.75,
        r"Within-state estimate:  $s_k = \sum_m w_{km} r_m z_m \; / \; \sum_m w_{km} r_m$;  "
        r"missing evidence receives zero effective weight. Metric weights are reviewed priors; case reliability is dynamic.",
        fontsize=9.2,
        color="#39454B",
    )
    _save(fig, path)


def plot_gate_weights(datasets: list[dict[str, Any]], path: Path) -> None:
    _style()
    completed = [
        row
        for row in datasets
        if row.get("base_gate_shares")
        or row.get("ours_meta", {}).get("base_standardized_feature_importance")
    ]
    expert_shares: list[dict[str, float]] = []
    for row in completed:
        if row.get("base_gate_shares"):
            expert_shares.append(dict(row["base_gate_shares"]))
            continue
        grouped: dict[str, float] = {}
        for feature, value in row["ours_meta"]["base_standardized_feature_importance"].items():
            parts = str(feature).split("__")
            if len(parts) < 2:
                continue
            expert = parts[1]
            grouped[expert] = grouped.get(expert, 0.0) + float(value)
        total = sum(grouped.values())
        expert_shares.append(
            {name: value / max(total, 1e-12) for name, value in grouped.items()}
        )
    branches = list(
        dict.fromkeys(branch for grouped in expert_shares for branch in grouped)
    )
    fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), constrained_layout=True)
    ax = axes[0]
    y = np.arange(len(completed))
    left = np.zeros(len(completed))
    for branch in branches:
        values = np.array([grouped.get(branch, 0.0) for grouped in expert_shares])
        bars = ax.barh(y, values, left=left, color=BRANCH_COLORS.get(branch, "#A7ADB2"), height=0.62, label=branch.replace("_", " "))
        for index, value in enumerate(values):
            if value >= 0.09:
                ax.text(left[index] + value / 2, index, f"{value:.2f}", ha="center", va="center", fontsize=8, color="white")
        left += values
    ax.set_yticks(y, [row["short_name"] for row in completed])
    ax.set_xlim(0, 1.02)
    ax.set_xlabel("Mean case-level gate weight or normalized coefficient share")
    ax.set_title("a  Base module: OOF-selected expert reliance", loc="left", fontsize=14, fontweight="bold")
    if branches:
        ax.legend(loc="lower center", bbox_to_anchor=(0.5, -0.24), ncol=min(4, len(branches)))

    gate_means = [row.get("agent_correction_gate_mean") for row in completed]
    gate_stds = [row.get("agent_correction_gate_std") for row in completed]
    valid = [index for index, value in enumerate(gate_means) if value is not None]
    ax = axes[1]
    if valid:
        x = np.arange(len(valid))
        values = np.asarray([gate_means[index] for index in valid], dtype=float)
        errors = np.asarray([gate_stds[index] or 0.0 for index in valid], dtype=float)
        bars = ax.bar(
            x,
            values,
            yerr=errors,
            color="#278C82",
            alpha=0.82,
            capsize=3,
            width=0.66,
        )
        ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=3, fontsize=8)
        ax.set_xticks(
            x,
            [completed[index]["short_name"] for index in valid],
            rotation=35,
            ha="right",
        )
        ax.set_ylim(0, 1.02)
        ax.set_ylabel("Mean evidence correction gate ± SD")
    else:
        ax.text(0.5, 0.5, "No case-level correction gate", ha="center", va="center")
        ax.axis("off")
    ax.set_title("b  Diagnostic Agent: permitted case-level correction", loc="left", fontsize=14, fontweight="bold")
    fig.suptitle("Global training parameters and case-specific evidence authority", fontsize=17, x=0.01, ha="left")
    _save(fig, path)


def plot_task_state_learning(datasets: list[dict[str, Any]], path: Path) -> None:
    _style()
    rows = []
    for row in datasets:
        if row.get("status") != "completed":
            continue
        features: dict[str, list[float]] = {}
        candidate_count = 0
        for training in row.get("ours_meta", {}).get("branch_training", {}).values():
            candidate_count += int(
                training.get("task_state_selection", {}).get(
                    "task_specific_candidates", 0
                )
            )
            for class_map in training.get("standardized_feature_coefficients", {}).values():
                if not isinstance(class_map, dict):
                    continue
                for feature, coefficient in class_map.items():
                    if "__task_" in feature:
                        features.setdefault(feature, []).append(abs(float(coefficient)))
        rows.append(
            {
                "dataset": row["short_name"],
                "candidate_count": candidate_count,
                "retained_count": len(features),
                "mean_abs": float(np.mean([np.mean(values) for values in features.values()])) if features else np.nan,
            }
        )

    frame = pd.DataFrame(rows)
    applicable = frame[frame["candidate_count"].gt(0)].copy().reset_index(drop=True)
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13, max(5.2, 0.65 * max(len(applicable), 1) + 2.0)),
        constrained_layout=True,
    )
    if applicable.empty:
        for ax in axes:
            ax.text(
                0.5,
                0.5,
                "No multi-task candidate states are available in current completed runs",
                ha="center",
                va="center",
                fontsize=12,
                color="#647178",
            )
            ax.axis("off")
        _save(fig, path)
        return
    y = np.arange(len(applicable))
    candidate_bars = axes[0].barh(
        y + 0.14,
        applicable["candidate_count"],
        color="#B9C8D8",
        height=0.24,
        label="Candidate task states",
    )
    retained_bars = axes[0].barh(
        y - 0.14,
        applicable["retained_count"],
        color="#4D84C4",
        height=0.24,
        label="Retained after train-fold audit",
    )
    axes[0].bar_label(
        candidate_bars,
        labels=[str(int(value)) if value else "" for value in applicable["candidate_count"]],
        padding=3,
        fontsize=8,
    )
    axes[0].bar_label(
        retained_bars,
        labels=[str(int(value)) if value else "" for value in applicable["retained_count"]],
        padding=3,
        fontsize=8,
    )
    axes[0].set_yticks(y, applicable["dataset"])
    axes[0].set_xlabel("Number of task-specific states")
    axes[0].set_title("Candidate versus retained task states", loc="left", fontweight="bold")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].set_ylim(-0.65, max(len(applicable) - 0.35, 0.65))
    values = applicable["mean_abs"].fillna(0.0)
    if values.gt(0).any():
        bars = axes[1].barh(y, values, color="#7864C7", height=0.62)
        axes[1].bar_label(bars, fmt="%.3f", padding=3, fontsize=8)
        axes[1].set_yticks(y, [""] * len(applicable))
        axes[1].set_xlabel("Mean absolute standardized coefficient")
        axes[1].set_title("Learned task-state use", loc="left", fontweight="bold")
    else:
        axes[1].text(
            0.5,
            0.55,
            "No task-specific state passed\nthe paired train-fold stability audit",
            ha="center",
            va="center",
            fontsize=13,
            color="#647178",
            linespacing=1.4,
        )
        axes[1].text(
            0.5,
            0.36,
            "Task evidence remains available for trace review,\nbut does not enter the current risk model.",
            ha="center",
            va="center",
            fontsize=9.5,
            color="#7B858B",
            linespacing=1.35,
        )
        axes[1].axis("off")
    fig.suptitle(
        "Task states enter prediction only after a paired train-fold stability audit",
        fontsize=15,
        fontweight="bold",
        x=0.01,
        ha="left",
    )
    _save(fig, path)


def plot_system_architecture(path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(16, 10), constrained_layout=True)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(x: float, y: float, width: float, height: float, title: str, body: str, fill: str, edge: str) -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), width, height,
                boxstyle="round,pad=0.025,rounding_size=0.11",
                facecolor=fill, edgecolor=edge, linewidth=1.8,
            )
        )
        title_y = y + height - (0.18 if height < 0.9 else 0.26)
        body_y = y + (0.14 if height < 0.9 else height - 0.72)
        ax.text(x + 0.16, title_y, title, fontsize=10.8, fontweight="bold", color="#182126", va="top")
        ax.text(x + 0.16, body_y, body, fontsize=7.9 if height < 0.9 else 8.5, color="#46535A", va="bottom" if height < 0.9 else "top", linespacing=1.45)

    def arrow(x1: float, y1: float, x2: float, y2: float) -> None:
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, linewidth=1.5, color="#59656B"))

    ax.text(0.25, 9.72, "ADvoice | Evidence-governed, task-aware speech screening system", fontsize=20, fontweight="bold", va="top")

    # a. Data are routed before feature interpretation, so task and speaker structure are not mixed.
    ax.text(0.25, 9.12, "a", fontsize=15, fontweight="bold")
    ax.text(0.62, 9.12, "Task-aware input and evidence construction", fontsize=14, fontweight="bold")
    channels = [
        (0.45, "Clinical interview", "audio + transcript", "#EAF1FA", "#4D84C4"),
        (2.62, "Picture description", "standardized task", "#EAF5EE", "#62A579"),
        (4.79, "Structured tasks", "multi-task / multilingual", "#F3F1EC", "#8A927B"),
        (6.96, "Public speech", "free speech / video", "#FCECEC", "#D97A78"),
    ]
    for x, title, body, fill, edge in channels:
        box(x, 7.62, 1.9, 1.05, title, body, fill, edge)
        ax.plot([x + 0.95, x + 0.95], [8.67, 8.82], color="#8A9499", linewidth=1.1)
    ax.plot([1.40, 8.84], [8.82, 8.82], color="#8A9499", linewidth=1.1)
    arrow(8.84, 8.82, 9.06, 8.82)
    box(9.08, 7.38, 1.75, 1.52, "Data routing", "task · language\nspeaker role · source", "#EAF1FA", "#4D84C4")
    arrow(10.85, 8.14, 11.18, 8.14)
    box(11.22, 7.38, 2.02, 1.52, "MetricEvidence", "value · direction\nreliability · confounds", "#F0ECFA", "#7864C7")
    arrow(13.27, 8.14, 13.60, 8.14)
    box(13.64, 7.38, 1.92, 1.52, "State cards", "severity · confidence\nsupport · counter-evidence", "#EAF5EE", "#62A579")

    # b. Within-state fusion and segment-level trace.
    ax.text(0.25, 6.82, "b", fontsize=15, fontweight="bold")
    ax.text(0.62, 6.82, "Local evidence and within-state fusion", fontsize=14, fontweight="bold")
    box(0.45, 4.72, 2.05, 1.55, "Clinical metrics", "pause burden\nspeech rate\nlexical retrieval", "#F0ECFA", "#7864C7")
    box(2.88, 4.72, 2.18, 1.55, "Evidence weights", "reviewed direction ×\ncase reliability ×\navailability", "#FFF3E4", "#D89A32")
    box(5.45, 4.72, 2.18, 1.55, "Clinical state", "normal · borderline\nimpaired · unreliable", "#EAF5EE", "#62A579")
    arrow(2.52, 5.50, 2.84, 5.50)
    arrow(5.08, 5.50, 5.41, 5.50)
    ax.plot([0.55, 1.33, 2.11, 2.89, 3.67, 4.45, 5.23, 6.01, 6.79, 7.48], [4.24, 4.39, 4.18, 4.50, 4.60, 4.27, 4.12, 4.48, 4.78, 4.57], color="#D97A78", linewidth=2.5)
    ax.axhline(4.40, xmin=0.035, xmax=0.47, color="#778187", linestyle=(0, (4, 3)), linewidth=1.2)
    ax.text(0.48, 3.83, "Segment trajectory: local peaks remain linked to transcript and source-time audio; dashed line = task-appropriate reference group.", fontsize=8.7, color="#536068")

    # c. The first supervised module, one evidence Agent, and the bounded correction module.
    ax.text(8.05, 6.82, "c", fontsize=15, fontweight="bold")
    ax.text(8.42, 6.82, "One diagnostic Agent with two supervised modules", fontsize=14, fontweight="bold")
    branch_boxes = [
        (8.30, 5.75, "Clinical states", "language · behaviour · interaction", "#F0ECFA", "#7864C7"),
        (8.30, 4.78, "Text representation", "frozen multilingual E5; model-only", "#EAF1FA", "#4D84C4"),
        (8.30, 3.81, "Audio representation", "mHuBERT windows + attention", "#EAF6F4", "#278C82"),
        (8.30, 2.84, "Age context", "fixed categories; model-only", "#FFF3E4", "#D89A32"),
    ]
    for x, y, title, body, fill, edge in branch_boxes:
        box(x, y, 2.18, 0.76, title, body, fill, edge)
        arrow(x + 2.20, y + 0.38, 11.12, 4.70)
    box(11.16, 4.05, 1.72, 1.42, "Module 1", "cross-fitted base\nrisk prediction", "#EAF6F4", "#278C82")
    arrow(12.91, 4.76, 13.19, 4.76)
    box(13.23, 4.05, 2.12, 1.42, "Diagnostic Agent", "support + counter-evidence\nconfound and trace audit", "#FCECEC", "#D97A78")
    box(11.16, 2.72, 1.72, 0.94, "QC / reliability", "controls permission;\nnever direct disease risk", "#F2F3F4", "#8B9397")
    arrow(12.90, 3.18, 13.20, 3.65)
    box(13.23, 2.58, 2.12, 1.05, "Module 2", "bounded correction\n+ train-only calibration", "#FFF3E4", "#D89A32")
    arrow(14.29, 4.02, 14.29, 3.66)

    # d. The locked Agent decision is reused for communication and clinician review.
    ax.text(8.05, 2.24, "d", fontsize=15, fontweight="bold")
    ax.text(8.42, 2.24, "Locked decision trace and clinical communication", fontsize=14, fontweight="bold")
    box(8.30, 0.55, 2.15, 1.24, "Locked Agent result", "calibrated probability\nuncertainty · decision trace", "#EAF5EE", "#62A579")
    arrow(10.49, 1.17, 10.85, 1.17)
    box(10.89, 0.55, 2.18, 1.24, "Communication phase", "same Agent, same trace\nno second diagnosis", "#FCECEC", "#D97A78")
    arrow(13.11, 1.17, 13.47, 1.17)
    box(13.51, 0.55, 2.05, 1.24, "Clinician review", "verify evidence\nact / defer / request retest", "#FFF3E4", "#D89A32")
    ax.text(0.45, 0.62, "B1  traditional acoustic ML", fontsize=9.5, fontweight="bold", color="#666D73")
    ax.text(3.02, 0.62, "B2  direct diagnostic agent", fontsize=9.5, fontweight="bold", color="#C36563")
    ax.text(5.56, 0.62, "Ours  one evidence diagnostic Agent + two supervised modules", fontsize=9.5, fontweight="bold", color="#217C73")
    _save(fig, path)


def _channel_payload(paths: ProjectPaths, datasets: list[dict[str, Any]]) -> list[dict[str, Any]]:
    payload = []
    seen = set()
    for row in datasets:
        if row.get("status") != "completed" or row["channel_profile"] in seen:
            continue
        seen.add(row["channel_profile"])
        config = load_all(row["dataset_id"])
        profile_datasets = [
            item for item in datasets
            if item.get("status") == "completed" and item.get("channel_profile") == row["channel_profile"]
        ]
        evidence_frames = []
        for item in profile_datasets:
            evidence_path = paths.artifacts / item["dataset_id"] / "metric_evidence.csv"
            if evidence_path.exists():
                evidence_frames.append(pd.read_csv(evidence_path))
        observed = pd.concat(evidence_frames, ignore_index=True) if evidence_frames else pd.DataFrame()
        metric_rows = []
        for definition in config["metrics"]["metrics"]:
            item = dict(definition)
            item["design_reliability"] = float(definition["reliability"])
            metric_observed = observed[observed["metric_id"].eq(definition["id"])] if not observed.empty else pd.DataFrame()
            item["observed_reliability"] = (
                float(metric_observed["reliability"].mean()) if not metric_observed.empty else float("nan")
            )
            item["missing_rate"] = (
                float(metric_observed["missing"].astype(float).mean()) if not metric_observed.empty else float("nan")
            )
            item["source_modality"] = (
                str(metric_observed["source_modality"].dropna().iloc[0])
                if not metric_observed.empty and "source_modality" in metric_observed and not metric_observed["source_modality"].dropna().empty
                else "derived from branch contract"
            )
            metric_rows.append(item)
        payload.append(
            {
                "id": row["channel_profile"],
                "datasets": [item["short_name"] for item in profile_datasets],
                "metrics": metric_rows,
                "states": config["states"]["states"],
                "unavailable_states": config["states"].get("unavailable_states", []),
            }
        )
    return payload


def _evaluation_payload(
    layer_a: pd.DataFrame,
    layer_b: pd.DataFrame,
    datasets: list[dict[str, Any]] | None = None,
    legacy_comparison: pd.DataFrame | None = None,
) -> dict[str, Any]:
    data = _preferred_layer_a(layer_a)
    auc = data[data["metric"].eq("macro_auroc_ovr")]
    names = list(dict.fromkeys(auc["dataset_name"]))
    series = {condition: _metric_series(data, "macro_auroc_ovr", condition) for condition in ["B1", "B2", "Ours"]}
    control_series = {
        condition: _metric_series(data, "macro_auroc_ovr", condition)
        for condition in ["QC_only", "Task_presence_only", "Label_permutation"]
    }
    paired = layer_a[
        layer_a["analysis_scope"].eq("paired_difference")
        & layer_a["metric"].eq("delta_macro_auroc_ovr")
    ].copy()
    rows = []
    dataset_audits = {row.get("short_name"): row for row in (datasets or [])}
    for name in names:
        values = {condition: float(series[condition].get(name, np.nan)) for condition in series}
        available = {key: value for key, value in values.items() if np.isfinite(value)}
        best_value = max(available.values()) if available else np.nan
        winners = [
            key
            for key, value in available.items()
            if np.isclose(value, best_value, rtol=0.0, atol=1e-12)
        ]
        winner = "/".join(winners) if winners else "NA"
        ours_value = values["Ours"]
        paired_support: dict[str, bool] = {}
        paired_intervals: dict[str, list[float]] = {}
        for baseline in ["B1", "B2"]:
            paired_row = paired[
                paired["dataset_name"].eq(name)
                & paired["condition"].eq(f"Ours-{baseline}")
            ]
            if paired_row.empty:
                paired_support[baseline] = False
                paired_intervals[baseline] = [np.nan, np.nan]
            else:
                low = float(pd.to_numeric(paired_row.iloc[0]["ci_low"], errors="coerce"))
                high = float(pd.to_numeric(paired_row.iloc[0]["ci_high"], errors="coerce"))
                paired_support[baseline] = bool(np.isfinite(low) and low > 0.0)
                paired_intervals[baseline] = [low, high]
        strength = "区分能力较强" if np.isfinite(ours_value) and ours_value >= 0.80 else ("区分能力中等" if np.isfinite(ours_value) and ours_value >= 0.70 else "区分能力不足")
        confounded = bool(dataset_audits.get(name, {}).get("capture_label_confounding_flag", False))
        control_values = {
            key: float(values_by_dataset.get(name, np.nan))
            for key, values_by_dataset in control_series.items()
        }
        shortcut_controls = {
            key: value
            for key, value in control_values.items()
            if key != "Label_permutation" and np.isfinite(value) and value >= 0.75
        }
        if confounded:
            conclusion = "采集批次/采访者与标签高度共线；该结果只用于方法审计，不作为临床区分能力证据。"
        elif shortcut_controls:
            controls_text = "、".join(
                f"{key}={value:.3f}" for key, value in shortcut_controls.items()
            )
            conclusion = (
                f"负控出现高区分度（{controls_text}），提示模型可利用采集质量或任务存在性捷径；"
                "本数据集结果降级为偏差审计，不能作为临床有效性或优于基线的证据。"
            )
        elif winners == ["Ours"] and all(
            paired_support[baseline]
            for baseline in ["B1", "B2"]
            if np.isfinite(values[baseline])
        ):
            baseline_text = "两个基线" if np.isfinite(values["B1"]) and np.isfinite(values["B2"]) else "当前可用基线"
            conclusion = f"{strength}；配对置信区间支持本轮融合优于{baseline_text}。"
        elif winners == ["Ours"]:
            conclusion = f"{strength}；本轮融合数值最高，但配对置信区间尚不能证明对所有基线的优势。"
        elif "Ours" in winners:
            tied = "、".join(key for key in winners if key != "Ours")
            conclusion = f"{strength}；本轮融合与{tied}并列最高，不能写成单独胜出。"
        elif winner == "NA":
            conclusion = "没有形成可比较的 held-out 结果。"
        else:
            conclusion = f"{strength}；{winner} 在本任务更高，需保留为失败模式。"
        rows.append({
            "dataset": name,
            **values,
            "winner": winner,
            "delta_b1": ours_value - values["B1"] if np.isfinite(ours_value) and np.isfinite(values["B1"]) else np.nan,
            "delta_b2": ours_value - values["B2"] if np.isfinite(ours_value) and np.isfinite(values["B2"]) else np.nan,
            "control_values": control_values,
            "shortcut_invalidated": bool(shortcut_controls),
            "paired_ci_b1": paired_intervals["B1"],
            "paired_ci_b2": paired_intervals["B2"],
            "paired_supported_b1": paired_support["B1"],
            "paired_supported_b2": paired_support["B2"],
            "conclusion": conclusion,
            "capture_confounded": confounded,
        })
    ours_vs_b1 = sum(np.isfinite(row["Ours"]) and np.isfinite(row["B1"]) and row["Ours"] > row["B1"] for row in rows)
    ours_vs_b2 = sum(np.isfinite(row["Ours"]) and np.isfinite(row["B2"]) and row["Ours"] > row["B2"] for row in rows)
    ours_supported_vs_b1 = sum(row["paired_supported_b1"] for row in rows if not row["capture_confounded"])
    ours_supported_vs_b2 = sum(row["paired_supported_b2"] for row in rows if not row["capture_confounded"])
    completed_b2 = sum(np.isfinite(row["B2"]) for row in rows)
    ours_rows = layer_b[layer_b["condition"].eq("Ours")]
    pass_rate = float(pd.to_numeric(ours_rows["passed"], errors="coerce").mean()) if not ours_rows.empty else np.nan
    prepare_micro = data[
        data["dataset_name"].eq("PREPARE")
        & data["condition"].eq("Ours")
        & data["metric"].eq("micro_auroc_ovr")
    ]
    prepare_micro_f1 = data[
        data["dataset_name"].eq("PREPARE")
        & data["condition"].eq("Ours")
        & data["metric"].eq("micro_f1")
    ]
    legacy_auc = legacy_comparison[
        legacy_comparison["metric"].eq("macro_auroc_ovr")
        & ~legacy_comparison["dataset_name"].eq("IAEAV")
    ] if legacy_comparison is not None and not legacy_comparison.empty else pd.DataFrame()
    return {
        "rows": rows,
        "dataset_count": len(rows),
        "ours_vs_b1": ours_vs_b1,
        "ours_vs_b2": ours_vs_b2,
        "ours_supported_vs_b1": ours_supported_vs_b1,
        "ours_supported_vs_b2": ours_supported_vs_b2,
        "completed_b2": completed_b2,
        "framework_pass_rate": pass_rate,
        "prepare_micro_auroc": float(prepare_micro.iloc[0]["value"]) if not prepare_micro.empty else np.nan,
        "prepare_micro_f1": float(prepare_micro_f1.iloc[0]["value"]) if not prepare_micro_f1.empty else np.nan,
        "valid_dataset_count": sum(not row["capture_confounded"] for row in rows),
        "median_auc_gain_vs_legacy": float(legacy_auc["delta"].median()) if not legacy_auc.empty else np.nan,
    }


def _write_oral_script(report_dir: Path, payload: dict[str, Any]) -> None:
    target = report_dir / "evaluation_oral_presentation_zh.md"
    dataset_lines = []
    for row in payload["rows"]:
        values = []
        for condition in ["B1", "B2", "Ours"]:
            value = row[condition]
            values.append(f"{condition} 为 {value:.3f}" if np.isfinite(value) else f"{condition} 暂无可比结果")
        dataset_lines.append(
            f"{row['dataset']}：" + "，".join(values) + f"。{row['conclusion']}"
        )
    speechcare = {
        str(row["metric_key"]): float(row["value"])
        for row in payload.get("speechcare_rows", [])
        if row.get("condition") == "SpeechCARE"
    }
    prepare_auc = payload.get("prepare_micro_auroc", np.nan)
    prepare_f1 = payload.get("prepare_micro_f1", np.nan)
    prepare_auc_text = f"{prepare_auc:.3f}" if np.isfinite(prepare_auc) else "暂无"
    prepare_f1_text = f"{prepare_f1:.3f}" if np.isfinite(prepare_f1) else "暂无"
    speechcare_auc = speechcare.get("micro_auroc_ovr", np.nan)
    speechcare_f1 = speechcare.get("reported_f1", np.nan)
    speechcare_auc_text = f"{speechcare_auc:.3f}" if np.isfinite(speechcare_auc) else "暂无"
    speechcare_f1_text = f"{speechcare_f1:.3f}" if np.isfinite(speechcare_f1) else "暂无"
    content = f"""# ADvoice 全数据集评估口语汇报

生成时间：{payload.get('generated_at', '未记录')}

这次更新把原来只负责写报告的组件改成了单一证据诊断 Agent。两个小型监督模块先在训练折内形成跨分支基础概率，并学习修正上限、证据门控与概率校准；这些边界在测试前冻结。推理时，Agent 再读取指标证据、临床状态、任务轨迹和质量标记，只能在冻结边界内提出修正。证据不足时严格退回基础预测。最后的医生报告只是把同一条已锁定决策轨迹翻译成临床文字，不会进行第二次自由诊断。

本轮按数据集记录各阶段运行状态。是否从原始音频重建、是否复用经过来源核验的转录缓存，以每个数据集的运行审计为准；汇总报告不再用一段固定文字替代实际运行记录。

实验仍然比较三个条件。B1 是传统声学机器学习，只使用低层声学特征；B2 是直接诊断代理，直接根据转录或音频摘要给出判断和报告；B3 是单一证据诊断 Agent 加两个小型监督模块，并保留从概率到状态、指标和片段的完整回溯。

    医学评估，也就是 A 层，不只看 AUROC。我们同时检查区分能力、准确率、均衡准确率、宏平均 F1、MCC、AUPRC、高敏感度筛查操作点、概率校准、漏诊和误报结构，以及质量控制负控。目前共有 {payload['dataset_count']} 个独立任务进入汇总，其中 IAEAV 因采访者与标签高度共线而降级为采集偏倚审计，不进入有效临床性能概括。

与 SpeechCARE 的比较只保留 PREPARE 官方测试集上定义可对应的终点。本轮 B3 的微平均 AUROC 为 {prepare_auc_text}，SpeechCARE 报告值为 {speechcare_auc_text}；本轮微平均 F1 为 {prepare_f1_text}，SpeechCARE 为 {speechcare_f1_text}。SpeechCARE 是十次训练均值，本项目当前是一次锁定训练加受试者 bootstrap，因此这是协议知情的描述性比较，不是等价的重复实验，也不能用可解释性优势替代预测性能胜出。

框架验证，也就是 B 层，检查的不是分类性能，而是系统主张是否成立。它包括指标证据完整性、状态卡完整性、分支贡献回溯、报告权限、原始片段忠实度、误分类病例的参考状态压力测试、完整融合对仅状态模型的消融，以及医生报告五维结构。这样可以区分两件事：模型是否预测正确，以及它给出的理由是否真正参与决策并且能够被核查。

逐数据集的 AUROC 结果如下：
""" + "\n".join(dataset_lines) + """

最后看失败模式图。绿色表示达到预设门槛，黄色表示需要谨慎解释，灰色表示该任务不适用。这里重点检查模型是否依赖录音质量捷径、概率是否校准、高敏感度时特异度是否足够、高置信错误是否受控，以及状态结论是否有原始片段支持。因此最终结论不是“所有数据集都最好”，而是明确哪些通道可以支持当前筛查主张，哪些仍需要重新校准或外部验证。
"""
    target.write_text(content, encoding="utf-8")


def build_aggregate_report(paths: ProjectPaths, dataset_ids: list[str]) -> Path:
    report_dir = paths.reports / "latest"
    assets = report_dir / "assets"
    report_dir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)
    datasets, layer_a, layer_b = collect_results(paths, dataset_ids)
    completed = [row for row in datasets if row.get("status") == "completed"]
    if not completed:
        raise RuntimeError("No completed dataset artifacts are available for aggregate reporting.")
    layer_a.to_csv(report_dir / "all_dataset_layer_a_metrics.csv", index=False)
    layer_b.to_csv(report_dir / "all_dataset_layer_b_checks.csv", index=False)
    core_metric_names = {
        "n": "n",
        "accuracy": "acc",
        "balanced_accuracy": "bal_acc",
        "macro_f1": "f1",
        "macro_auroc_ovr": "auc",
        "micro_auroc_ovr": "micro_auc",
        "ece": "ece",
    }
    core = _preferred_layer_a(layer_a)
    core = core[
        core["condition"].isin(["B1", "B2", "Ours"])
        & core["metric"].isin(core_metric_names)
    ]
    core = (
        core.pivot_table(
            index=["dataset_id", "condition"],
            columns="metric",
            values="value",
            aggfunc="first",
        )
        .rename(columns=core_metric_names)
        .reset_index()
        .rename(columns={"dataset_id": "dataset"})
    )
    ordered_core_columns = [
        "dataset",
        "condition",
        "n",
        "acc",
        "bal_acc",
        "f1",
        "auc",
        "micro_auc",
        "ece",
    ]
    core.reindex(columns=ordered_core_columns).to_csv(
        report_dir / "all_dataset_three_condition_core_metrics.csv", index=False
    )
    plot_system_architecture(assets / "figure_1_system_architecture.png")
    plot_dataset_landscape(datasets, assets / "figure_2_dataset_landscape.png")
    plot_channel_state_matrix(paths, datasets, assets / "figure_3_channel_state_matrix.png")
    plot_metric_state_map(paths, assets / "figure_10_metric_to_state_map.png")
    plot_framework_revision(assets / "figure_0_framework_revision.png")
    speechcare_rows = plot_speechcare_aligned_comparison(
        paths, layer_a, assets / "figure_speechcare_protocol_aligned.png"
    )
    plot_medical_standards_summary(layer_a, assets / "figure_A_layer_medical_standards_summary.png")
    plot_layer_a_overview(layer_a, assets / "figure_4_layer_a_performance.png")
    plot_paired_auroc_differences(layer_a, assets / "figure_A1b_paired_auroc_differences.png")
    plot_iaeav_capture_confounding(paths, assets / "figure_4b_iaeav_capture_confounding.png")
    plot_safety_overview(layer_a, assets / "figure_5_layer_a_safety.png")
    plot_error_imbalance(layer_a, assets / "figure_A4_error_imbalance.png")
    plot_robustness_controls(layer_a, assets / "figure_A5_robustness_controls.png")
    plot_screening_operating_points(layer_a, assets / "figure_9_screening_operating_points.png")
    plot_layer_b_overview(layer_b, assets / "figure_6_layer_b_validation.png")
    plot_layer_b_comprehensive(layer_a, layer_b, datasets, assets / "figure_B_layer_framework_validation_summary.png")
    legacy_comparison = plot_legacy_c_comparison(
        paths,
        dataset_ids,
        assets / "figure_B5_current_vs_legacy_c.png",
    )
    legacy_comparison.to_csv(
        report_dir / "current_b3_vs_legacy_c_metrics.csv", index=False
    )
    plot_failure_mode_audit(layer_a, layer_b, assets / "figure_failure_mode_audit.png")
    plot_report_rubric(datasets, assets / "figure_7_report_rubric.png")
    plot_gate_weights(datasets, assets / "figure_8_gate_weights.png")
    plot_task_state_learning(datasets, assets / "figure_11_task_state_learning.png")
    env = Environment(loader=FileSystemLoader(paths.root / "templates"), autoescape=select_autoescape(["html"]))
    generated_at = now_utc()
    channel_payload = _channel_payload(paths, datasets)
    evaluation_payload = _evaluation_payload(layer_a, layer_b, datasets, legacy_comparison)
    shared = {
        "datasets": datasets,
        "completed": completed,
        "layer_a": layer_a,
        "layer_b": layer_b,
        "channel_payload": channel_payload,
        "evaluation": evaluation_payload,
        "speechcare_rows": speechcare_rows,
        "generated_at": generated_at,
        "assets": "assets",
    }
    output = report_dir / "aggregate_evaluation_report.html"
    evaluation_html = env.get_template("aggregate_report.html").render(**shared)
    output.write_text(evaluation_html, encoding="utf-8")
    (report_dir / "evaluation_report.html").write_text(evaluation_html, encoding="utf-8")
    _write_oral_script(
        report_dir,
        {
            **evaluation_payload,
            "speechcare_rows": speechcare_rows,
            "generated_at": generated_at,
        },
    )
    system_html = env.get_template("aggregate_system_report.html").render(**shared)
    (report_dir / "aggregate_system_report.html").write_text(system_html, encoding="utf-8")
    (report_dir / "system_report.html").write_text(system_html, encoding="utf-8")
    index_html = env.get_template("aggregate_index.html").render(**shared)
    (report_dir / "index.html").write_text(index_html, encoding="utf-8")
    audit_rows = []
    for row in datasets:
        failure_path = paths.artifacts / row["dataset_id"] / "feature_extraction_failures.csv"
        failures = 0
        if failure_path.exists():
            failures = len(pd.read_csv(failure_path))
        audit_rows.append(
            {
                **row,
                "feature_failures": failures,
                "audit_path": str(paths.artifacts / row["dataset_id"] / "dataset_audit.json"),
            }
        )
    batch_status_path = paths.reports / "batch_run_status.json"
    batch_status = json.dumps(json_load(batch_status_path), ensure_ascii=False, indent=2) if batch_status_path.exists() else "not available"
    run_html = env.get_template("aggregate_run_report.html").render(
        datasets=audit_rows,
        generated_at=generated_at,
        batch_status=batch_status,
    )
    (report_dir / "aggregate_run_audit.html").write_text(run_html, encoding="utf-8")
    (report_dir / "run_report.html").write_text(run_html, encoding="utf-8")
    build_failure_analysis(paths)
    return output
