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

from .config import ProjectPaths
from .models import LABELS
from .utils import json_load, source_inventory


COLORS = {"B1": "#8A8F98", "B2": "#E68178", "Ours": "#278C82"}


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


def plot_dataset_inventory(manifest: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    recording = manifest.groupby(["split", "label"]).size().unstack(fill_value=0).reindex(columns=LABELS)
    subject = manifest.drop_duplicates(["split", "subject_id"]).groupby(["split", "label"]).size().unstack(fill_value=0).reindex(columns=LABELS)
    palette = ["#76A6D8", "#E4B45D", "#D97972"]
    recording.plot(kind="bar", ax=axes[0], color=palette, width=0.72)
    subject.plot(kind="bar", ax=axes[1], color=palette, width=0.72)
    axes[0].set_title("长音频记录数")
    axes[1].set_title("受试者数（模型分析单位）")
    for ax in axes:
        ax.set_xlabel("")
        ax.set_ylabel("数量")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(title="诊断标签", frameon=False, ncol=3, loc="upper left")
        for container in ax.containers:
            ax.bar_label(container, padding=2, fontsize=9)
    fig.suptitle("NCMMSC2021_AD 数据纳入与分析单位", fontsize=16, x=0.01, ha="left")
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_layer_a(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    main = layer_a[layer_a["condition"].isin(["B1", "B2", "Ours"])]
    metrics = [
        ("macro_auroc_ovr", "宏平均 AUROC", True),
        ("accuracy", "准确率", True),
        ("macro_f1", "宏平均 F1", True),
        ("ece", "校准误差 ECE", False),
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
        ax.set_ylabel("越高越好" if higher else "越低越好")
        if "Ours" in conditions:
            bars[conditions.index("Ours")].set_edgecolor("#0F4C47")
            bars[conditions.index("Ours")].set_linewidth(1.6)
    fig.suptitle("A 层｜医学预测与筛查核心结果", fontsize=17, x=0.01, ha="left")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_confusions(prediction_paths: dict[str, Path], path: Path) -> None:
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
        matrix = pd.crosstab(frame["label"], frame["predicted_label"]).reindex(index=LABELS, columns=LABELS, fill_value=0).to_numpy()
        image = ax.imshow(matrix, cmap="Blues", vmin=0)
        for i in range(len(LABELS)):
            for j in range(len(LABELS)):
                ax.text(j, i, str(matrix[i, j]), ha="center", va="center", color="white" if matrix[i, j] > matrix.max() * 0.55 else "#24323A", fontsize=12)
        ax.set_xticks(range(len(LABELS)), LABELS)
        ax.set_yticks(range(len(LABELS)), LABELS)
        ax.set_xlabel("预测")
        ax.set_ylabel("真实")
        ax.set_title(condition)
        fig.colorbar(image, ax=ax, fraction=0.045, pad=0.04)
    fig.suptitle("官方测试集混淆矩阵", fontsize=16, x=0.01, ha="left")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_negative_controls(layer_a: pd.DataFrame, path: Path) -> None:
    _style()
    conditions = ["QC_only", "No_duration_no_loudness", "B1", "Ours"]
    subset = layer_a[layer_a["metric"].eq("macro_auroc_ovr")].set_index("condition")
    present = [name for name in conditions if name in subset.index]
    values = [float(subset.loc[name, "value"]) for name in present]
    palette = ["#C8CDD3" if name == "QC_only" else "#8CB6B0" if name == "No_duration_no_loudness" else COLORS[name] for name in present]
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    bars = ax.bar(present, values, color=palette, width=0.62)
    _annotate_bars(ax, bars)
    ax.axhline(0.5, color="#4F5963", linestyle="--", linewidth=1, label="三分类随机参考不等于 0.5；该线仅作二分类直觉参照")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("宏平均 AUROC")
    ax.set_title("负控与去捷径审计")
    ax.legend(frameon=False, fontsize=9)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_layer_b(layer_b: pd.DataFrame, path: Path) -> None:
    _style()
    ours = layer_b[layer_b["condition"].eq("Ours")].copy()
    ours["normalized"] = ours.apply(
        lambda row: row["value"] / 25.0 if row["check"] == "clinical report rubric /25" else row["value"], axis=1
    )
    ours["normalized"] = ours["normalized"].clip(lower=-0.2, upper=1.0)
    fig, ax = plt.subplots(figsize=(11, 6.4), constrained_layout=True)
    y = np.arange(len(ours))
    colors = ["#278C82" if bool(value) else "#D9857C" for value in ours["passed"]]
    bars = ax.barh(y, ours["normalized"], color=colors, height=0.62)
    ax.set_yticks(y, [value.replace("-", " ") for value in ours["check"]])
    ax.invert_yaxis()
    ax.set_xlim(-0.22, 1.08)
    ax.axvline(0.8, color="#303A43", linestyle="--", linewidth=1)
    ax.set_xlabel("归一化完成度 / 效果（报告 /25 已除以 25）")
    ax.set_title("B 层｜框架是否解决可追溯、可审查和捷径问题")
    for bar, raw in zip(bars, ours["value"], strict=True):
        label = "未运行" if not np.isfinite(raw) else f"{raw:.3f}"
        ax.text(bar.get_width() + 0.02, bar.get_y() + bar.get_height() / 2, label, va="center", fontsize=9)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_states(state_wide: pd.DataFrame, path: Path) -> None:
    _style()
    state_cols = ["state_S01", "state_S02", "state_S03"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), constrained_layout=True)
    palette = {"HC": "#76A6D8", "MCI": "#E4B45D", "AD": "#D97972"}
    for ax, state in zip(axes, state_cols, strict=True):
        values = [state_wide[state_wide["label"].eq(label)][state].dropna() for label in LABELS]
        boxes = ax.boxplot(values, tick_labels=LABELS, patch_artist=True, widths=0.58, showfliers=False)
        for patch, label in zip(boxes["boxes"], LABELS, strict=True):
            patch.set_facecolor(palette[label])
            patch.set_alpha(0.8)
        ax.axhline(0, color="#3F4952", linestyle="--", linewidth=1)
        ax.axhline(1, color="#8C5B54", linestyle=":", linewidth=1)
        ax.set_title({"state_S01": "S01 停顿负担", "state_S02": "S02 输出效率", "state_S03": "S03 连续性"}[state])
        ax.set_ylabel("相对训练集 HC 的方向性稳健偏离")
    fig.suptitle("临床状态分布：0 为训练集健康参考中位附近", fontsize=16, x=0.01, ha="left")
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


def plot_branch_weights(predictions: pd.DataFrame, path: Path) -> None:
    _style()
    grouped = predictions.groupby("label")[["behavior_weight", "auxiliary_weight"]].mean().reindex(LABELS)
    fig, ax = plt.subplots(figsize=(9.5, 4.8), constrained_layout=True)
    bottom = np.zeros(len(grouped))
    for column, color, label in [
        ("behavior_weight", "#4D84C4", "语音行为状态分支"),
        ("auxiliary_weight", "#E4A33B", "低层声学辅助分支"),
    ]:
        bars = ax.bar(grouped.index, grouped[column], bottom=bottom, color=color, width=0.58, label=label)
        for index, bar in enumerate(bars):
            value = grouped.iloc[index][column]
            if value > 0.08:
                ax.text(bar.get_x() + bar.get_width() / 2, bottom[index] + value / 2, f"{value:.2f}", ha="center", va="center", color="white", fontsize=10)
        bottom += grouped[column].to_numpy()
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("平均病例级门控权重")
    ax.set_title("可靠度条件化分支权重（官方测试集）")
    ax.legend(frameon=False, ncol=2)
    fig.savefig(path, dpi=190, bbox_inches="tight")
    plt.close(fig)


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
    b2_status = json_load(artifact_dir / "b2_status.json", {})
    ours_report_status = json_load(artifact_dir / "ours_report_status.json", {})
    summary = json_load(artifact_dir / "evaluation_summary.json", {})
    dataset_audit = json_load(artifact_dir / "dataset_audit.json", {})
    b1_meta = json_load(artifact_dir / "b1_model.json", {})
    ours_meta = json_load(artifact_dir / "ours_model.json", {})

    gate_parameters = ours_meta.get("gate_parameters", [])
    reliability_beta = float(gate_parameters[2]) if len(gate_parameters) >= 3 else float("nan")
    behavior_weight_std = float(ours_predictions["behavior_weight"].std(ddof=0))
    gate_audit = {
        "reliability_beta": reliability_beta,
        "behavior_weight_std": behavior_weight_std,
        "case_dynamic": bool(abs(reliability_beta) >= 0.01 and behavior_weight_std >= 0.01),
    }

    plot_dataset_inventory(manifest, assets / "dataset_inventory.png")
    plot_layer_a(layer_a, assets / "layer_a_summary.png")
    plot_confusions(
        {
            "B1": artifact_dir / "b1_predictions.csv",
            "B2": artifact_dir / "b2_predictions.csv",
            "Ours": artifact_dir / "ours_predictions.csv",
        },
        assets / "confusion_matrices.png",
    )
    plot_negative_controls(layer_a, assets / "negative_controls.png")
    plot_layer_b(layer_b, assets / "layer_b_summary.png")
    plot_states(state_wide[state_wide["split"].eq("test")], assets / "state_distributions.png")
    plot_branch_weights(ours_predictions, assets / "branch_weights.png")

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
        manifest=manifest,
        metric_definitions=configs["metrics"]["metrics"],
        state_definitions=configs["states"]["states"],
        unavailable_states=configs["states"].get("unavailable_states", []),
        evidence=evidence,
        cards=cards,
        b1_meta=b1_meta,
        ours_meta=ours_meta,
        gate_audit=gate_audit,
        b2_status=b2_status,
        ours_report_status=ours_report_status,
        source_files=source_inventory(paths.root),
        root_uri=paths.root.as_uri(),
    )
    b2_reports = pd.read_csv(artifact_dir / "b2_reports.csv", dtype={"subject_id": str})
    ours_reports = pd.read_csv(artifact_dir / "ours_reports.csv", dtype={"subject_id": str})
    _render(
        env.get_template("evaluation_report.html"),
        report_dir / "evaluation_report.html",
        **common,
        layer_a=layer_a,
        layer_b=layer_b,
        summary=summary,
        gate_audit=gate_audit,
        b2_status=b2_status,
        b2_reports=b2_reports.head(3).to_dict("records"),
        ours_reports=ours_reports.head(3).to_dict("records"),
    )
    _render(env.get_template("index.html"), report_dir / "index.html", **common)


def publish_latest(paths: ProjectPaths, run_report_dir: Path) -> Path:
    latest = paths.reports / "latest"
    if latest.exists():
        shutil.rmtree(latest)
    shutil.copytree(run_report_dir, latest)
    return latest
