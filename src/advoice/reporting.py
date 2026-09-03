from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sklearn.calibration import calibration_curve

from .agent_runtime import case_pseudonym
from .config import ProjectPaths
from .utils import json_load, source_inventory


COLORS = {"B1": "#8A8F98", "B2": "#E68178", "Ours": "#278C82"}


def _expert_cv_values(training: dict[str, Any]) -> tuple[dict[str, float], Any]:
    """Normalize grid-searched and fixed expert metadata for report rendering."""
    macro_auroc = training.get("macro_auroc")
    if isinstance(macro_auroc, dict):
        return {
            str(key): float(value)
            for key, value in macro_auroc.items()
            if value is not None
        }, training.get("selected_c")
    if macro_auroc is not None:
        return {"fixed": float(macro_auroc)}, training.get("selected_c", "fixed")
    cv_values = {
        str(key): float(value["macro_auroc"])
        for key, value in training.items()
        if isinstance(value, dict) and value.get("macro_auroc") is not None
    }
    selected_c = (
        max(
            training,
            key=lambda key: (
                float(training[key].get("macro_f1", float("-inf"))),
                float(training[key].get("macro_auroc", float("-inf"))),
            ),
        )
        if cv_values
        else None
    )
    return cv_values, selected_c


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["PingFang SC", "Arial Unicode MS", "Noto Sans CJK SC", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.grid": True,
            "grid.alpha": 0.22,
            "grid.linewidth": 0.7,
        }
    )


def _annotate_bars(ax: plt.Axes, bars: Any, fmt: str = ".3f") -> None:
    for bar in bars:
        value = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.018, format(value, fmt), ha="center", va="bottom", fontsize=10)


def _save_figure(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_dataset_inventory(manifest: pd.DataFrame, path: Path, labels: list[str]) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    recording = manifest.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
    subject = manifest.drop_duplicates(["split", "subject_id"]).groupby(["split", "label"]).size().unstack(fill_value=0).reindex(columns=labels, fill_value=0)
    palette = ["#76A6D8", "#E4B45D", "#D97972"]
    recording.plot(kind="bar", ax=axes[0], color=palette, width=0.72)
    subject.plot(kind="bar", ax=axes[1], color=palette, width=0.72)
    axes[0].set_title("Recordings")
    axes[1].set_title("Independent analysis units")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="Target label", frameon=False, ncol=len(labels), loc="upper left")
        for container in ax.containers:
            ax.bar_label(container, padding=2, fontsize=9)
    fig.suptitle("Dataset inclusion and unit of analysis", fontsize=16, x=0.01, ha="left")
    _save_figure(fig, path)


def plot_layer_a(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    main = layer_a[layer_a["condition"].isin(["B1", "B2", "Ours"])]
    if "analysis_scope" in main and main["analysis_scope"].eq("matched_three_arm").any():
        main = main[main["analysis_scope"].eq("matched_three_arm")]
    metrics = [
        ("macro_auroc_ovr", "Macro AUROC", True),
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro F1", True),
        ("ece", "Expected calibration error", False),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), constrained_layout=True)
    for ax, (metric, title, higher) in zip(axes.flat, metrics, strict=True):
        subset = main[main["metric"].eq(metric)].set_index("condition")
        conditions = [name for name in ["B1", "B2", "Ours"] if name in subset.index]
        values = [float(subset.loc[name, "value"]) for name in conditions]
        bars = ax.bar(conditions, values, color=[COLORS[name] for name in conditions], width=0.58)
        _annotate_bars(ax, bars)
        ax.set_ylim(0, max(1.02, max(values, default=0) + 0.12))
        ax.set_title(title)
        ax.set_ylabel("Higher is better" if higher else "Lower is better")
        if "Ours" in conditions:
            bars[conditions.index("Ours")].set_edgecolor("#0F4C47")
            bars[conditions.index("Ours")].set_linewidth(1.6)
    fig.suptitle("Layer A | Medical prediction and screening performance", fontsize=17, x=0.01, ha="left")
    _save_figure(fig, path)


def plot_confusions(prediction_paths: dict[str, Path], path: Path, labels: list[str]) -> None:
    _style()
    available = []
    for condition, source in prediction_paths.items():
        frame = pd.read_csv(source, dtype={"subject_id": str})
        if not frame.empty and "predicted_label" in frame:
            available.append((condition, frame))
    fig, axes = plt.subplots(1, max(len(available), 1), figsize=(5.2 * max(len(available), 1), 4.5), constrained_layout=True)
    axes = np.atleast_1d(axes)
    if not available:
        axes[0].text(0.5, 0.5, "无可用预测", ha="center", va="center")
        axes[0].axis("off")
    for ax, (condition, frame) in zip(axes, available, strict=False):
        matrix = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(index=labels, columns=labels, fill_value=0).to_numpy()
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * 0.55 else "#24323A", fontsize=12)
        ax.set_xticks(range(len(labels)), labels)
        ax.set_yticks(range(len(labels)), labels)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(condition)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.suptitle("Held-out confusion matrices", fontsize=16, x=0.01, ha="left")
    _save_figure(fig, path)


def plot_negative_controls(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    conditions = ["QC_only", "No_duration_no_loudness", "B1", "Ours"]
    subset = layer_a[layer_a["metric"].eq("macro_auroc_ovr")]
    if "analysis_scope" in subset:
        subset = subset[subset["analysis_scope"].eq("full_available_cohort")]
    subset = subset.set_index("condition")
    present = [name for name in conditions if name in subset.index]
    values = [float(subset.loc[name, "value"]) for name in present]
    palette = ["#C8CDD3" if name == "QC_only" else "#8CB6B0" if name == "No_duration_no_loudness" else COLORS[name] for name in present]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    bars = ax.bar(present, values, color=palette, width=0.62)
    _annotate_bars(ax, bars)
    ax.axhline(0.5, color="#4F5963", linestyle="--", linewidth=1, label="Binary chance reference")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Macro AUROC")
    ax.set_title("Negative controls and shortcut audit")
    ax.legend(frameon=False, fontsize=9)
    _save_figure(fig, path)


def plot_layer_b(layer_b: pd.DataFrame, path: Path) -> None:
    _style()
    ours = layer_b[layer_b["condition"].eq("Ours")].copy()
    ours["normalized"] = ours.apply(
        lambda row: row["value"] / 25.0 if row["check"] == "clinical report rubric /25" else row["value"], axis=1
    )
    ours = ours[np.isfinite(ours["normalized"])].copy()
    ours["normalized"] = ours["normalized"].clip(lower=-0.2, upper=1.0)
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    y = np.arange(len(ours))
    colors = ["#278C82" if bool(value) else "#D9857C" for value in ours["passed"]]
    bars = ax.barh(y, ours["normalized"], color=colors, height=0.62)
    ax.set_yticks(y, [value.replace("-", " ") for value in ours["check"]])
    ax.invert_yaxis()
    ax.set_xlim(-0.22, 1.08)
    ax.axvline(0.8, color="#303A43", linestyle="--", linewidth=1)
    ax.set_xlabel("Normalized completion or effect")
    ax.set_title("Layer B | Traceability, auditability and failure-mode tests")
    for bar, raw in zip(bars, ours["value"], strict=True):
        label = "未运行" if not np.isfinite(raw) else f"{raw:.3f}"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
    _save_figure(fig, path)


def plot_states(state_wide: pd.DataFrame, path: Path, labels: list[str]) -> None:
    _style()
    state_cols = [column for column in state_wide if column.startswith("state_")][:6]
    columns = min(3, max(1, len(state_cols)))
    rows = int(np.ceil(len(state_cols) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(4.4 * columns, 4.1 * rows), constrained_layout=True)
    palette_values = ["#76A6D8", "#E4B45D", "#D97972", "#7BAE8B"]
    palette = {label: palette_values[index % len(palette_values)] for index, label in enumerate(labels)}
    axes = np.atleast_1d(axes).flat
    for ax, state in zip(axes, state_cols, strict=False):
        values = [state_wide[state_wide["label"].eq(label)][state].dropna() for label in labels]
        boxes = ax.boxplot(values, tick_labels=labels, patch_artist=True, widths=0.58, showfliers=False)
        for patch, label in zip(boxes["boxes"], labels, strict=True):
            patch.set_facecolor(palette[label])
            patch.set_alpha(0.8)
        ax.axhline(0, color="#3F4952", linestyle="--", linewidth=1)
        ax.axhline(1, color="#8C5B54", linestyle=":", linewidth=1)
        ax.set_title(state.replace("state_", ""))
        ax.set_ylabel("Directional robust deviation from train-HC")
    fig.suptitle("Clinical-state distributions", fontsize=16, x=0.01, ha="left")
    _save_figure(fig, path)


def plot_branch_weights(
    model_meta: dict[str, Any],
    contributions: pd.DataFrame,
    path: Path,
    labels: list[str],
) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    importance = model_meta.get("base_standardized_feature_importance", {})
    grouped_importance: dict[str, float] = {}
    for feature, value in importance.items():
        if "__" not in feature:
            continue
        parts = str(feature).split("__")
        expert = parts[1] if len(parts) > 1 else str(feature)
        grouped_importance[expert] = grouped_importance.get(expert, 0.0) + float(value)
    if grouped_importance:
        ordered = sorted(grouped_importance.items(), key=lambda item: item[1], reverse=True)
        names = [item[0].replace("_", " ") for item in ordered]
        values = np.asarray([item[1] for item in ordered], dtype=float)
        values = values / max(values.sum(), 1e-12)
        bars = axes[0].bar(names, values, color=["#4D84C4", "#7864C7", "#D89A32", "#278C82"][: len(names)])
        axes[0].bar_label(bars, labels=[f"{value:.1%}" for value in values], padding=3, fontsize=9)
        axes[0].set_ylim(0, max(0.55, float(values.max()) + 0.12))
        axes[0].set_ylabel("Normalized absolute coefficient")
        axes[0].tick_params(axis="x", rotation=18)
    else:
        axes[0].text(0.5, 0.5, "No fitted expert coefficients", ha="center", va="center")
        axes[0].axis("off")
    axes[0].set_title("Base supervised module: learned expert reliance")

    if "agent_correction_gate" in contributions and not contributions.empty:
        grouped = [
            contributions.loc[contributions["label"].astype(str).eq(label), "agent_correction_gate"].dropna().to_numpy()
            for label in labels
        ]
        grouped = [values for values in grouped if len(values)]
        used_labels = [
            label
            for label in labels
            if contributions["label"].astype(str).eq(label).any()
        ]
        boxes = axes[1].boxplot(
            grouped,
            tick_labels=used_labels,
            patch_artist=True,
            widths=0.5,
        )
        for patch, color in zip(boxes["boxes"], ["#76A6D8", "#E68178", "#D89A32"], strict=False):
            patch.set_facecolor(color)
            patch.set_alpha(0.75)
        means = [float(np.mean(values)) for values in grouped]
        axes[1].scatter(range(1, len(means) + 1), means, color="#172126", marker="D", s=28, label="Mean")
        axes[1].legend(frameon=False)
        axes[1].set_ylim(0, 1.02)
        axes[1].set_ylabel("Evidence correction gate")
    else:
        axes[1].text(0.5, 0.5, "No case-level correction gate", ha="center", va="center")
        axes[1].axis("off")
    axes[1].set_title("Diagnostic Agent: case-specific correction authority")
    fig.suptitle("What is learned globally and what changes by case", fontsize=16, x=0.01, ha="left")
    _save_figure(fig, path)


def plot_task_state_coefficients(model_meta: dict[str, Any], path: Path) -> None:
    """Show whether task-specific states entered the fitted clinical branches."""
    _style()
    rows: list[dict[str, Any]] = []
    for branch, training in model_meta.get("branch_training", {}).items():
        coefficients = training.get("standardized_feature_coefficients", {})
        if not isinstance(coefficients, dict):
            continue
        for class_label, feature_map in coefficients.items():
            if not isinstance(feature_map, dict):
                continue
            for feature, coefficient in feature_map.items():
                if "__task_" not in feature:
                    continue
                rows.append(
                    {
                        "branch": branch,
                        "class_label": class_label,
                        "feature": feature.replace("state_", "", 1),
                        "coefficient": float(coefficient),
                    }
                )

    fig, ax = plt.subplots(figsize=(11, 6.2), constrained_layout=True)
    if not rows and model_meta.get("top_state_discriminability"):
        summary = pd.Series(model_meta["top_state_discriminability"], dtype=float).head(16).sort_values()
        bars = ax.barh(
            [str(value).replace("state_", "", 1).replace("__task_", " | task ") for value in summary.index],
            summary.to_numpy(),
            color="#7864C7",
            height=0.66,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlabel("Train-only class separation score")
        ax.set_ylabel("Clinical state")
        ax.set_title("States available to the diagnostic Agent")
        ax.set_xlim(0, max(float(summary.max()) * 1.2, 0.05))
        _save_figure(fig, path)
        return
    if not rows:
        ax.text(
            0.5,
            0.55,
            "Single-task dataset or no fitted task-specific state",
            ha="center",
            va="center",
            fontsize=14,
            color="#647178",
        )
        ax.text(
            0.5,
            0.43,
            "Overall states remain active; task-specific coefficients are not applicable.",
            ha="center",
            va="center",
            fontsize=10,
            color="#7B858B",
        )
        ax.axis("off")
    else:
        frame = pd.DataFrame(rows)
        summary = (
            frame.assign(abs_coefficient=frame["coefficient"].abs())
            .groupby(["branch", "feature"], as_index=False)["abs_coefficient"]
            .mean()
            .sort_values("abs_coefficient", ascending=False)
            .head(16)
            .sort_values("abs_coefficient")
        )
        palette = {
            "speech_behavior": "#4D84C4",
            "language": "#7864C7",
            "interaction": "#278C82",
            "auxiliary_acoustic": "#D89A32",
        }
        bars = ax.barh(
            summary["feature"],
            summary["abs_coefficient"],
            color=[palette.get(branch, "#8A8F98") for branch in summary["branch"]],
            height=0.66,
        )
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)
        ax.set_xlabel("Mean absolute standardized coefficient across target classes")
        ax.set_ylabel("Task-specific clinical state")
        ax.set_title("Learned use of task-specific states")
        ax.set_xlim(0, max(summary["abs_coefficient"].max() * 1.2, 0.05))
    _save_figure(fig, path)


def _task_trace_examples(cards: pd.DataFrame, limit: int = 8) -> list[dict[str, Any]]:
    task_cards = cards[cards.get("task_scope", "overall").astype(str).ne("overall")].copy()
    if task_cards.empty:
        return []
    task_cards["rank"] = task_cards["state_z"].abs() * task_cards["confidence"].fillna(0.0)
    task_cards["has_segment_trace"] = task_cards["evidence_segments"].fillna("[]").map(
        lambda value: len(json.loads(value)) > 0
    )
    segment_limit = min(limit // 2, int(task_cards["has_segment_trace"].sum()))
    selected = pd.concat(
        [
            task_cards[task_cards["has_segment_trace"]].nlargest(segment_limit, "rank"),
            task_cards[~task_cards["has_segment_trace"]].nlargest(limit - segment_limit, "rank"),
        ],
        ignore_index=True,
    ).sort_values("rank", ascending=False)
    examples: list[dict[str, Any]] = []
    for row in selected.itertuples(index=False):
        try:
            metrics = json.loads(row.supporting_metrics)
        except (TypeError, json.JSONDecodeError):
            metrics = []
        try:
            segments = json.loads(row.evidence_segments)
        except (TypeError, json.JSONDecodeError):
            segments = []
        examples.append(
            {
                "case_id": case_pseudonym(str(row.subject_id)),
                "task_scope": row.task_scope,
                "state_id": row.state_id,
                "state_name": row.state_name_zh,
                "state_z": float(row.state_z),
                "raw_state_z": float(getattr(row, "raw_state_z", row.state_z)),
                "confidence": float(row.confidence),
                "metrics": ", ".join(str(item.get("metric_id", "")) for item in metrics[:3]),
                "segments": "; ".join(
                    f"{item.get('segment_id', '')} [{float(item.get('start_sec', 0)):.1f}-{float(item.get('end_sec', 0)):.1f}s]"
                    for item in segments[:2]
                ),
            }
        )
    return examples


def _environment(paths: ProjectPaths) -> Environment:
    return Environment(
        loader=FileSystemLoader(paths.root / "templates"),
        autoescape=select_autoescape(["html"]),
    )


def _render(template: Any, output: Path, **context: Any) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(template.render(**context), encoding="utf-8")


def build_reports(
    paths: ProjectPaths,
    configs: dict[str, Any],
    run_manifest: dict[str, Any],
    artifact_dir: Path,
    report_dir: Path,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    assets = report_dir / "assets"
    assets.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(artifact_dir / "manifest.csv", dtype={"subject_id": str})
    evidence = pd.read_csv(artifact_dir / "metric_evidence.csv", dtype={"subject_id": str})
    cards = pd.read_csv(artifact_dir / "state_cards.csv", dtype={"subject_id": str})
    state_wide = pd.read_csv(artifact_dir / "state_wide.csv", dtype={"subject_id": str})
    layer_a = pd.read_csv(artifact_dir / "layer_a_metrics.csv")
    layer_b = pd.read_csv(artifact_dir / "layer_b_checks.csv")
    ours_predictions = pd.read_csv(artifact_dir / "ours_predictions.csv", dtype={"subject_id": str})
    contributions = pd.read_csv(
        artifact_dir / "branch_contributions.csv", dtype={"subject_id": str}
    )
    b2_status = json_load(artifact_dir / "b2_status.json", {})
    ours_report_status = json_load(artifact_dir / "ours_report_status.json", {})
    summary = json_load(artifact_dir / "evaluation_summary.json", {})
    dataset_audit = json_load(artifact_dir / "dataset_audit.json", {})
    b1_meta = json_load(artifact_dir / "b1_model.json", {})
    ours_meta = json_load(artifact_dir / "ours_model.json", {})
    labels = [str(label) for label in configs["dataset"]["labels"]]
    preferred_scope = "matched_three_arm" if layer_a["analysis_scope"].eq("matched_three_arm").any() else "full_available_cohort"
    core_metrics = [
        "macro_auroc_ovr",
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "macro_auprc",
        "mcc",
        "multiclass_brier",
        "ece",
    ]
    condition_rows = []
    for condition in ["B1", "B2", "Ours"]:
        subset = layer_a[
            layer_a["condition"].eq(condition)
            & layer_a["analysis_scope"].eq(preferred_scope)
            & layer_a["metric"].isin(core_metrics)
        ]
        values = {row.metric: float(row.value) for row in subset.itertuples(index=False)}
        condition_rows.append({"condition": condition, "status": "completed" if values else "not_run", **values})

    gate_parameters = ours_meta.get("gate_parameters", {})
    reliability_beta = max((float(value.get("reliability_beta", 0.0)) for value in gate_parameters.values()), default=float("nan"))
    weight_columns = [column for column in ours_predictions if column.startswith("weight_")]
    behavior_weight_std = max((float(ours_predictions[column].std(ddof=0)) for column in weight_columns), default=float("nan"))
    correction_gate = pd.to_numeric(
        contributions.get("agent_correction_gate", pd.Series(dtype=float)),
        errors="coerce",
    )
    gate_audit = {
        "reliability_beta": reliability_beta,
        "behavior_weight_std": behavior_weight_std,
        "correction_gate_mean": float(correction_gate.mean()) if correction_gate.notna().any() else None,
        "correction_gate_std": float(correction_gate.std(ddof=0)) if correction_gate.notna().any() else None,
        "selected_alpha": ours_meta.get("selected_alpha"),
        "case_dynamic": bool(
            correction_gate.notna().any() and float(correction_gate.std(ddof=0)) >= 0.01
        )
        if ours_meta.get("schema_version") == "condition-c-evidence-agent-v1"
        else bool(abs(reliability_beta) >= 0.01 and behavior_weight_std >= 0.01),
        "test_labels_used_by_agent": ours_meta.get("test_labels_used_by_agent"),
    }

    plot_dataset_inventory(manifest, assets / "dataset_inventory.png", labels)
    plot_layer_a(layer_a, assets / "layer_a_summary.png")
    plot_confusions(
        {
            "B1": artifact_dir / "b1_predictions.csv",
            "B2": artifact_dir / "b2_predictions.csv",
            "Ours": artifact_dir / "ours_predictions.csv",
        },
        assets / "confusion_matrices.png",
        labels,
    )
    plot_negative_controls(layer_a, assets / "negative_controls.png")
    plot_layer_b(layer_b, assets / "layer_b_summary.png")
    plot_states(state_wide[state_wide["split"].eq("test")], assets / "state_distributions.png", labels)
    plot_branch_weights(ours_meta, contributions, assets / "branch_weights.png", labels)
    plot_task_state_coefficients(ours_meta, assets / "task_state_coefficients.png")

    task_cards = cards[cards["task_scope"].astype(str).ne("overall")] if "task_scope" in cards else cards.iloc[0:0]
    segment_capable_task_cards = (
        task_cards[task_cards["trace_resolution"].eq("task_and_segment")]
        if "trace_resolution" in task_cards
        else task_cards.iloc[0:0]
    )
    task_state_summary = {
        "scopes": sorted(task_cards["task_scope"].astype(str).unique().tolist()) if not task_cards.empty else [],
        "card_count": int(len(task_cards)),
        "state_count": int(
            len(task_cards[["task_scope", "state_id"]].drop_duplicates())
        )
        if not task_cards.empty
        else 0,
        "segment_trace_rate": float(
            segment_capable_task_cards["evidence_segments"].astype(str).ne("[]").mean()
        )
        if not segment_capable_task_cards.empty
        else None,
    }
    task_trace_examples = _task_trace_examples(cards)
    branch_training_summary = []
    for branch, training in ours_meta.get("branch_training", {}).items():
        coefficients = training.get("standardized_feature_coefficients", {})
        task_features = {
            feature
            for feature_map in coefficients.values()
            if isinstance(feature_map, dict)
            for feature in feature_map
            if "__task_" in feature
        }
        branch_training_summary.append(
            {
                "branch": branch,
                "selected_c": training.get("selected_c"),
                "best_cv_macro_auroc": max(training.get("cv_macro_auroc_by_c", {}).values(), default=None),
                "task_feature_count": len(task_features),
                "task_state_selection": training.get("task_state_selection", {}).get(
                    "selected", "not_applicable"
                ),
                "task_state_cv_gain": (
                    float(training["task_state_selection"]["full_task_cv_macro_auroc"])
                    - float(training["task_state_selection"]["overall_only_cv_macro_auroc"])
                    if training.get("task_state_selection", {}).get("full_task_cv_macro_auroc")
                    is not None
                    and training.get("task_state_selection", {}).get(
                        "overall_only_cv_macro_auroc"
                    )
                    is not None
                    else None
                ),
                "task_state_gain_lower_95": training.get(
                    "task_state_selection", {}
                ).get("paired_fold_gain_lower_95"),
            }
        )
    if not branch_training_summary:
        for expert, training in ours_meta.get("expert_cv", {}).items():
            cv_values, selected_c = _expert_cv_values(training)
            branch_training_summary.append(
                {
                    "branch": expert,
                    "selected_c": selected_c,
                    "best_cv_macro_auroc": max(
                        (float(value) for value in cv_values.values() if value is not None),
                        default=None,
                    ),
                    "task_feature_count": 0,
                    "task_state_selection": "fold-internal evidence expert",
                    "task_state_cv_gain": None,
                    "task_state_gain_lower_95": None,
                }
            )

    env = _environment(paths)
    common = {
        "project": configs["project"],
        "dataset": configs["dataset"],
        "run": run_manifest,
        "assets": "assets",
    }
    _render(
        env.get_template("system_report.html"),
        report_dir / "system_report.html",
        **common,
        dataset_audit=dataset_audit,
        channel_profile=configs["channel_profile"],
        manifest=manifest,
        metric_definitions=configs["metrics"]["metrics"],
        state_definitions=configs["states"]["states"],
        unavailable_states=configs["states"].get("unavailable_states", []),
        evidence=evidence,
        cards=cards,
        b1_meta=b1_meta,
        ours_meta=ours_meta,
        task_state_summary=task_state_summary,
        task_trace_examples=task_trace_examples,
        branch_training_summary=branch_training_summary,
        gate_audit=gate_audit,
        b2_status=b2_status,
        ours_report_status=ours_report_status,
        source_files=source_inventory(paths.root),
        root_uri=paths.root.as_uri(),
    )
    b2_reports = pd.read_csv(artifact_dir / "b2_reports.csv", dtype={"subject_id": str})
    ours_reports = pd.read_csv(artifact_dir / "ours_reports.csv", dtype={"subject_id": str})
    matched_report_cases = sorted(
        set(b2_reports.get("case_id", pd.Series(dtype=str)).astype(str))
        & set(ours_reports.get("case_id", pd.Series(dtype=str)).astype(str))
    )[:3]
    b2_report_examples = b2_reports[
        b2_reports.get("case_id", pd.Series(index=b2_reports.index, dtype=str))
        .astype(str)
        .isin(matched_report_cases)
    ]
    ours_report_examples = ours_reports[
        ours_reports.get("case_id", pd.Series(index=ours_reports.index, dtype=str))
        .astype(str)
        .isin(matched_report_cases)
    ]
    _render(
        env.get_template("evaluation_report.html"),
        report_dir / "evaluation_report.html",
        **common,
        layer_a=layer_a,
        layer_b=layer_b,
        summary=summary,
        gate_audit=gate_audit,
        b2_status=b2_status,
        b2_reports=b2_report_examples.to_dict("records"),
        ours_reports=ours_report_examples.to_dict("records"),
        labels=labels,
        preferred_scope=preferred_scope,
        condition_rows=condition_rows,
    )
    _render(
        env.get_template("run_report.html"),
        report_dir / "run_report.html",
        **common,
        dataset_audit=dataset_audit,
        b2_status=b2_status,
        ours_report_status=ours_report_status,
        source_files=source_inventory(paths.root),
        root_uri=paths.root.as_uri(),
    )
    _render(env.get_template("index.html"), report_dir / "index.html", **common)


def publish_latest(paths: ProjectPaths, run_report_dir: Path, dataset_id: str) -> Path:
    latest = paths.reports / "datasets" / dataset_id / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_report_dir, latest)
    return latest
