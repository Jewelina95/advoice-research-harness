from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "latest"
ASSETS = REPORT / "assets"

COLORS = {
    "ink": "#17212B",
    "muted": "#63717E",
    "line": "#CED5DC",
    "blue": "#4E79A7",
    "blue_light": "#E8F0F8",
    "green": "#4E8B68",
    "green_light": "#E8F3EC",
    "orange": "#D28C36",
    "orange_light": "#FFF2DE",
    "purple": "#7562A8",
    "purple_light": "#F0ECF8",
    "red": "#C76660",
    "red_light": "#FBEAE8",
    "gray": "#A5ADB5",
}

mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial Unicode MS", "Songti SC", "Arial", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.8,
        "legend.frameon": False,
    }
)


def save(fig: plt.Figure, stem: str) -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    for suffix, kwargs in {
        "png": {"dpi": 260},
        "svg": {},
        "pdf": {},
    }.items():
        fig.savefig(ASSETS / f"{stem}.{suffix}", bbox_inches="tight", facecolor="white", **kwargs)
    plt.close(fig)


def box(ax, x, y, w, h, title, detail, color, fill, tag=None):
    patch = FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.018",
        linewidth=1.5, edgecolor=color, facecolor=fill, zorder=2
    )
    ax.add_patch(patch)
    if tag:
        ax.text(x + w - 0.018, y + h - 0.022, tag, color=color, fontsize=9, weight="bold", va="top", ha="right")
    ax.text(x + 0.02, y + h - 0.035, title, color=COLORS["ink"], fontsize=11.7, weight="bold", va="top")
    ax.text(x + 0.02, y + h - 0.085, detail, color=COLORS["muted"], fontsize=8.6, va="top", linespacing=1.3)
    return patch


def arrow(ax, start, end, color="#77838D", rad=0.0, lw=1.7, style="-"):
    ax.add_patch(
        FancyArrowPatch(
            start, end, arrowstyle="-|>", mutation_scale=13,
            linewidth=lw, color=color, linestyle=style,
            connectionstyle=f"arc3,rad={rad}", zorder=3
        )
    )


def framework_figure() -> None:
    fig, ax = plt.subplots(figsize=(17, 10.5))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.text(0.02, 0.965, "ADvoice 新版：证据治理的单一诊断 Agent", fontsize=24, weight="bold", color=COLORS["ink"])
    ax.text(0.02, 0.925, "从任务路由、证据对象和临床状态，到受限风险更新与同源医生报告", fontsize=12.5, color=COLORS["muted"])

    lane_y = [0.66, 0.36, 0.08]
    lane_h = 0.22
    lane_titles = ["A  数据与临床证据", "B  学习、融合与诊断 Agent", "C  临床输出与治理"]
    lane_fills = ["#F7F9FB", "#F7FAF8", "#FBF9F5"]
    for y, title, fill in zip(lane_y, lane_titles, lane_fills):
        ax.add_patch(FancyBboxPatch((0.015, y), 0.97, lane_h, boxstyle="round,pad=0.01,rounding_size=.018", facecolor=fill, edgecolor="#DFE4E9", linewidth=1.0))
        ax.text(0.03, y + lane_h - 0.035, title, fontsize=12.5, weight="bold", color=COLORS["muted"], va="top")

    # Lane A
    w, h, y = 0.16, 0.135, 0.695
    xs = [0.035, 0.225, 0.415, 0.605, 0.795]
    boxes_a = [
        box(ax, xs[0], y, w, h, "多通道输入", "音频 · 转录 · 任务\n语言 · 说话人角色", COLORS["blue"], COLORS["blue_light"], "01"),
        box(ax, xs[1], y, w, h, "角色与任务路由", "患者/采访者分离\n图片描述/访谈/回忆", COLORS["blue"], COLORS["blue_light"], "02"),
        box(ax, xs[2], y, w, h, "证据原子提取", "声学 · 语言 · 对话\n片段事件 · QC", COLORS["purple"], COLORS["purple_light"], "03"),
        box(ax, xs[3], y, w, h, "MetricEvidence", "方向 · 可靠度 · 混杂\n权限 · 原始片段", COLORS["purple"], COLORS["purple_light"], "04"),
        box(ax, xs[4], y, w, h, "StateCard 与轨迹", "严重度 · 置信度\n支持/反证 · 任务变化", COLORS["green"], COLORS["green_light"], "05"),
    ]
    for x1, x2 in zip(xs[:-1], xs[1:]):
        arrow(ax, (x1 + w, y + h / 2), (x2 - 0.008, y + h / 2))

    # U-turn from the evidence row into the learning row.
    ax.plot([xs[4] + w, 0.975, 0.975, 0.035], [y + h / 2, y + h / 2, 0.59, 0.59], color="#77838D", lw=1.7)
    arrow(ax, (0.035, 0.59), (0.035, 0.54), color="#77838D")

    # Lane B
    yb, wb, hb = 0.405, 0.18, 0.135
    xsb = [0.04, 0.25, 0.46, 0.67]
    box(ax, xsb[0], yb, wb, hb, "监督模块 A", "分支专家学习基础风险\n训练折内筛除弱分支", COLORS["blue"], COLORS["blue_light"], "06")
    box(ax, xsb[1], yb, wb, hb, "可靠性感知融合", "逻辑堆叠 vs 动态门控\n仅按外层折结果选择", COLORS["green"], COLORS["green_light"], "07")
    box(ax, xsb[2], yb, wb, hb, "单一诊断 Agent", "检索证据 → 形成假设\n检查反证 → 记录理由", COLORS["purple"], COLORS["purple_light"], "08")
    box(ax, xsb[3], yb, wb, hb, "监督模块 B", "学习有限风险修正\n不稳定时修正系数归零", COLORS["orange"], COLORS["orange_light"], "09")
    for x1, x2 in zip(xsb[:-1], xsb[1:]):
        arrow(ax, (x1 + wb, yb + hb / 2), (x2 - 0.008, yb + hb / 2))
    ax.text(0.49, 0.597, "同一证据工作区同时服务基础预测与 Agent 审查", ha="center", va="bottom", color=COLORS["muted"], fontsize=9.2)

    # Lane C
    yc, wc, hc = 0.125, 0.205, 0.125
    xsc = [0.07, 0.315, 0.56, 0.805]
    box(ax, xsc[0], yc, wc, hc, "校准风险与不确定性", "概率 · 置信区间\n阈值附近/高置信错误", COLORS["green"], COLORS["green_light"], "10")
    box(ax, xsc[1], yc, wc, hc, "共享决策轨迹", "采用/拒绝证据\n状态与分支贡献", COLORS["purple"], COLORS["purple_light"])
    box(ax, xsc[2], yc, wc, hc, "医生报告", "初筛结论 · 原始证据\n解释限制 · 复核建议", COLORS["orange"], COLORS["orange_light"])
    box(ax, xsc[3], yc, 0.14, hc, "安全边界", "非确诊\n医生复核", COLORS["red"], COLORS["red_light"])
    ax.plot([xsb[3] + wb, 0.94, 0.94, 0.07], [yb + hb / 2, yb + hb / 2, 0.305, 0.305], color=COLORS["orange"], lw=1.7)
    arrow(ax, (0.07, 0.305), (0.07, yc + hc), color=COLORS["orange"])
    for x1, x2, width2 in [(xsc[0], xsc[1], wc), (xsc[1], xsc[2], wc), (xsc[2], xsc[3], 0.14)]:
        arrow(ax, (x1 + wc, yc + hc / 2), (x2 - 0.008, yc + hc / 2))

    # Training-only boundary and clinician points.
    ax.plot([0.025, 0.975], [0.62, 0.62], color="#AEB7C0", lw=1, ls=(0, (4, 4)))
    ax.text(0.975, 0.625, "上：病例证据构建  |  下：训练与决策", ha="right", va="bottom", color=COLORS["muted"], fontsize=9.5)
    ax.text(0.97, 0.322, "医生介入：证据定义审查 · 状态方向审查 · 报告盲评", ha="right", color=COLORS["red"], fontsize=9.7, weight="bold")
    ax.text(0.02, 0.025, "绿色路径：可进入疾病风险；紫色路径：可审查临床证据；橙色路径：受限更新；QC 仅改变可靠度，不直接增加疾病风险。", color=COLORS["muted"], fontsize=10)
    save(fig, "figure_framework_new_complete")


def layer_a_figure() -> None:
    df = pd.read_csv(REPORT / "all_dataset_three_condition_core_metrics.csv")
    name_map = {
        "ADReSS_2020": "ADReSS 2020",
        "ADReSSo_2021_diagnosis": "ADReSSo diagnosis",
        "ADReSSo_2021_progression": "ADReSSo progression",
        "PROCESS_2": "PROCESS-2",
        "PREPARE_DrivenData": "PREPARE",
        "DementiaBank_Pitt": "Pitt",
        "DementiaNet_PublicFigures": "DementiaNet",
        "NCMMSC2021_AD": "NCMMSC 2021",
    }
    df["display_name"] = df["dataset"].replace(name_map)
    order = [
        "ADReSS 2020", "ADReSSo diagnosis", "PROCESS-2", "PREPARE",
        "Pitt", "NCMMSC 2021", "DementiaNet", "TAUKADIAL",
        "ADReSSo progression", "IAEAV",
    ]
    colors = {"B1": "#A6ADB4", "B2": "#D18B86", "Ours": COLORS["green"]}
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={"width_ratios": [1.42, 1]})
    ax = axes[0]
    y = np.arange(len(order))
    offsets = {"B1": -0.22, "B2": 0, "Ours": 0.22}
    for condition in ["B1", "B2", "Ours"]:
        vals = []
        for name in order:
            row = df[(df.display_name == name) & (df.condition == condition)]
            vals.append(float(row.auc.iloc[0]) if len(row) else np.nan)
        ax.scatter(vals, y + offsets[condition], s=76 if condition == "Ours" else 58, color=colors[condition], label=condition, zorder=3)
        for x, yy in zip(vals, y + offsets[condition]):
            if np.isfinite(x):
                ax.text(x + 0.012, yy, f"{x:.3f}", va="center", fontsize=8.8, color=colors[condition], weight="bold" if condition == "Ours" else "normal")
    ax.axvline(0.5, color="#AEB6BE", ls="--", lw=1)
    ax.set_yticks(y, order, fontsize=10.5)
    ax.set_xlim(0.38, 1.08)
    ax.invert_yaxis()
    ax.set_xlabel("Macro AUROC  (higher is better)", weight="bold")
    ax.set_title("a  Medical discrimination across datasets", loc="left", fontsize=15, weight="bold")
    ax.grid(axis="x", color="#E6EAEE", lw=0.8)
    ax.legend(ncol=3, loc="lower right")

    ax = axes[1]
    ours = df[df.condition == "Ours"].set_index("display_name").reindex(order)
    b1 = df[df.condition == "B1"].set_index("display_name").reindex(order)
    b2 = df[df.condition == "B2"].set_index("display_name").reindex(order)
    gain1 = ours.auc - b1.auc
    gain2 = ours.auc - b2.auc
    h = 0.33
    ax.barh(y - h / 2, gain1, height=h, color="#79A98A", label="Ours − B1")
    ax.barh(y + h / 2, gain2, height=h, color="#907EBE", label="Ours − B2")
    ax.axvline(0, color=COLORS["ink"], lw=1)
    for i, (g1, g2) in enumerate(zip(gain1, gain2)):
        ax.text(g1 + (0.009 if g1 >= 0 else -0.009), i - h / 2, f"{g1:+.3f}", va="center", ha="left" if g1 >= 0 else "right", fontsize=8.5)
        ax.text(g2 + (0.009 if g2 >= 0 else -0.009), i + h / 2, f"{g2:+.3f}", va="center", ha="left" if g2 >= 0 else "right", fontsize=8.5)
    ax.set_yticks(y, [])
    ax.set_xlim(-0.27, 0.42)
    ax.invert_yaxis()
    ax.set_xlabel("AUROC difference", weight="bold")
    ax.set_title("b  Increment over the two baselines", loc="left", fontsize=15, weight="bold")
    ax.grid(axis="x", color="#E6EAEE", lw=0.8)
    ax.legend(ncol=2, loc="lower right")
    fig.suptitle("Layer A | Medical prediction and screening performance", x=0.06, y=1.01, ha="left", fontsize=20, weight="bold")
    fig.text(0.06, -0.01, "IAEAV is retained as a capture-confounding audit; DementiaNet has only six test subjects. Values are not pooled across datasets.", fontsize=9.5, color=COLORS["muted"])
    fig.tight_layout()
    save(fig, "figure_framework_layer_a_clear")


def layer_b_figure() -> None:
    df = pd.read_csv(REPORT / "all_dataset_layer_b_checks.csv")
    dataset_order = ["ADReSS 2020", "ADReSSo diagnosis", "PROCESS-2", "PREPARE", "Pitt", "NCMMSC 2021", "TAUKADIAL", "ADReSSo progression", "IAEAV"]
    checks = [
        "MetricEvidence completeness", "StateCard completeness", "branch contribution trace",
        "report-permission audit", "evidence-span faithfulness", "reference-state intervention on errors",
    ]
    labels = ["MetricEvidence", "StateCard", "Branch trace", "Report permission", "Span faithfulness", "State intervention"]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7.6), gridspec_kw={"width_ratios": [1.25, 1, 1]})
    ax = axes[0]
    mat = np.full((len(dataset_order), len(checks)), np.nan)
    for i, ds in enumerate(dataset_order):
        for j, check in enumerate(checks):
            row = df[(df.dataset_name == ds) & (df.condition == "Ours") & (df.check == check)]
            if len(row):
                value = row.iloc[0]["value"]
                mat[i, j] = float(value) if pd.notna(value) else float(bool(row.iloc[0]["passed"]))
    cmap = mpl.colors.ListedColormap(["#F4D9D6", "#F7E6BC", "#DDEEE3", "#4E8B68"])
    norm = mpl.colors.BoundaryNorm([-0.01, 0.5, 0.8, 0.999, 1.01], cmap.N)
    ax.imshow(mat, aspect="auto", cmap=cmap, norm=norm)
    ax.set_yticks(np.arange(len(dataset_order)), dataset_order, fontsize=9.6)
    ax.set_xticks(np.arange(len(labels)), labels, rotation=32, ha="right", fontsize=9.2)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            if np.isfinite(mat[i, j]):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8.2, color="white" if mat[i,j] > .99 else COLORS["ink"], weight="bold")
    ax.set_title("a  Evidence-contract audit", loc="left", fontsize=14.5, weight="bold")
    for spine in ax.spines.values(): spine.set_visible(False)

    ax = axes[1]
    score = df[df.check == "clinical report rubric /25"].pivot(index="dataset_name", columns="condition", values="value").reindex(dataset_order)
    yy = np.arange(len(score))
    for i, (_, row) in enumerate(score.iterrows()):
        if pd.notna(row.get("B2")) and pd.notna(row.get("Ours")):
            ax.plot([row["B2"], row["Ours"]], [i, i], color="#CBD1D7", lw=2.2, zorder=1)
    ax.scatter(score.get("B2"), yy, color="#D18B86", s=58, label="B2", zorder=3)
    ax.scatter(score.get("Ours"), yy, color=COLORS["green"], s=68, label="Ours", zorder=3)
    ax.set_yticks(yy, [])
    ax.invert_yaxis(); ax.set_xlim(8, 26)
    ax.set_xlabel("Automated report-structure score /25", weight="bold")
    ax.set_title("b  Clinical-report structure", loc="left", fontsize=14.5, weight="bold")
    ax.grid(axis="x", color="#E6EAEE"); ax.legend(ncol=2, loc="lower right")

    ax = axes[2]
    concept = df[(df.condition == "Ours") & (df.check == "concept-only vs full fusion")].set_index("dataset_name").reindex(dataset_order)
    vals = pd.to_numeric(concept.value, errors="coerce")
    bar_colors = [COLORS["green"] if x >= 0 else COLORS["red"] for x in vals.fillna(0)]
    ax.barh(yy, vals, color=bar_colors, height=.62)
    ax.axvline(0, color=COLORS["ink"], lw=1)
    for i, x in enumerate(vals):
        if pd.notna(x): ax.text(x + (.008 if x >= 0 else -.008), i, f"{x:+.3f}", va="center", ha="left" if x >= 0 else "right", fontsize=8.7)
    ax.set_yticks(yy, [])
    ax.invert_yaxis(); ax.set_xlim(min(-.12, np.nanmin(vals)-.04), max(.34, np.nanmax(vals)+.04))
    ax.set_xlabel("Full fusion − concept-only AUROC", weight="bold")
    ax.set_title("c  Increment beyond StateCards", loc="left", fontsize=14.5, weight="bold")
    ax.grid(axis="x", color="#E6EAEE")
    fig.suptitle("Layer B | Does the evidence-governed framework operate as designed?", x=.055, y=1.015, ha="left", fontsize=20, weight="bold")
    fig.text(.055, -.015, "Layer B audits traceability and mechanism execution; it does not replace physician review or Layer A medical performance.", fontsize=9.5, color=COLORS["muted"])
    fig.tight_layout()
    save(fig, "figure_framework_layer_b_clear")


def improvement_figure() -> None:
    df = pd.read_csv(REPORT / "current_b3_vs_legacy_c_metrics.csv")
    order = ["ADReSS 2020", "ADReSSo diagnosis", "PREPARE", "Pitt", "DementiaNet", "NCMMSC 2021", "TAUKADIAL", "PROCESS-2", "ADReSSo progression", "IAEAV"]
    metrics = [("macro_auroc_ovr", "Macro AUROC"), ("accuracy", "Accuracy"), ("macro_f1", "Macro F1")]
    fig, axes = plt.subplots(1, 3, figsize=(17, 8.2), sharey=False)
    y = np.arange(len(order))
    for ax, (metric, title) in zip(axes, metrics):
        sub = df[df.metric == metric].set_index("dataset_name").reindex(order)
        old, new = sub.legacy_c, sub.current_b3
        for i, (a, b) in enumerate(zip(old, new)):
            color = COLORS["green"] if b >= a else COLORS["red"]
            ax.plot([a, b], [i, i], color=color, lw=2.3, alpha=.8)
            ax.text(b + .012, i, f"{b:.3f}", va="center", fontsize=8.8, color=color, weight="bold")
        ax.scatter(old, y, s=50, color=COLORS["gray"], label="Legacy C", zorder=3)
        ax.scatter(new, y, s=70, color=[COLORS["green"] if b >= a else COLORS["red"] for a, b in zip(old, new)], label="Current B3", zorder=4)
        ax.set_xlim(.30, 1.08); ax.invert_yaxis(); ax.grid(axis="x", color="#E6EAEE")
        ax.set_xlabel(title, weight="bold"); ax.set_title(title, fontsize=14, weight="bold")
    axes[0].set_yticks(y, order, fontsize=10.2)
    axes[1].set_yticks(y, []); axes[2].set_yticks(y, [])
    axes[0].legend(ncol=2, loc="lower right")
    fig.suptitle("Current evidence-diagnostic Agent versus the legacy Condition C", x=.07, y=1.01, ha="left", fontsize=20, weight="bold")
    fig.text(.07, -.01, "Connected points use the same held-out subjects. Green indicates improvement; red indicates regression. IAEAV remains a capture-confounding audit.", fontsize=9.5, color=COLORS["muted"])
    fig.tight_layout()
    save(fig, "figure_framework_current_vs_legacy_clear")


if __name__ == "__main__":
    framework_figure()
    layer_a_figure()
    layer_b_figure()
    improvement_figure()
