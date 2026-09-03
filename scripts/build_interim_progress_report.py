from __future__ import annotations

import html
import json
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
OUTPUT = ROOT / "reports" / "interim_2026-08-13"
ASSETS = OUTPUT / "assets"

COLORS = {"B1": "#7A8085", "B2": "#E07A6A", "Ours": "#2A8C82"}
DATASETS = [
    "IAEAV",
    "ADReSS_2020",
    "ADReSSo_2021_diagnosis",
    "ADReSSo_2021_progression",
    "PROCESS_2",
    "PREPARE_DrivenData",
    "TAUKADIAL",
    "DementiaBank_Pitt",
    "DementiaNet_PublicFigures",
    "NCMMSC2021_AD",
]
DISPLAY = {
    "IAEAV": "IAEAV",
    "ADReSS_2020": "ADReSS 2020",
    "ADReSSo_2021_diagnosis": "ADReSSo diagnosis",
    "ADReSSo_2021_progression": "ADReSSo progression",
    "PROCESS_2": "PROCESS-2",
    "PREPARE_DrivenData": "PREPARE",
    "TAUKADIAL": "TAUKADIAL",
    "DementiaBank_Pitt": "DementiaBank Pitt",
    "DementiaNet_PublicFigures": "DementiaNet",
    "NCMMSC2021_AD": "NCMMSC 2021",
}


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#263238",
            "xtick.color": "#45545C",
            "ytick.color": "#45545C",
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )


def save(fig: plt.Figure, name: str) -> None:
    for suffix in ("png", "svg", "pdf"):
        fig.savefig(ASSETS / f"{name}.{suffix}", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def read_metric(dataset: str, metric: str, condition: str) -> float:
    path = ARTIFACTS / dataset / "layer_a_metrics.csv"
    if not path.exists():
        return np.nan
    frame = pd.read_csv(path)
    preferred = frame[
        frame["analysis_scope"].eq("matched_three_arm")
        & frame["condition"].eq(condition)
        & frame["metric"].eq(metric)
    ]
    if preferred.empty:
        preferred = frame[
            frame["analysis_scope"].eq("full_available_cohort")
            & frame["condition"].eq(condition)
            & frame["metric"].eq(metric)
        ]
    return float(preferred.iloc[0]["value"]) if not preferred.empty else np.nan


def status(dataset: str, name: str) -> str:
    path = ARTIFACTS / dataset / name
    if not path.exists():
        return "not_run"
    try:
        return str(json.loads(path.read_text(encoding="utf-8")).get("status", "unknown"))
    except json.JSONDecodeError:
        return "invalid"


def dataset_rows() -> list[dict[str, object]]:
    rows = []
    for dataset in DATASETS:
        audit_path = ARTIFACTS / dataset / "dataset_audit.json"
        audit = json.loads(audit_path.read_text(encoding="utf-8")) if audit_path.exists() else {}
        rows.append(
            {
                "dataset": dataset,
                "display": DISPLAY[dataset],
                "subjects": audit.get("subjects", 0),
                "test": audit.get("test_subjects", 0),
                "b2": status(dataset, "b2_status.json"),
                "ours_report": status(dataset, "ours_report_status.json"),
                "b1_auc": read_metric(dataset, "macro_auroc_ovr", "B1"),
                "b2_auc": read_metric(dataset, "macro_auroc_ovr", "B2"),
                "ours_auc": read_metric(dataset, "macro_auroc_ovr", "Ours"),
                "b1_accuracy": read_metric(dataset, "accuracy", "B1"),
                "b2_accuracy": read_metric(dataset, "accuracy", "B2"),
                "ours_accuracy": read_metric(dataset, "accuracy", "Ours"),
            }
        )
    return rows


def plot_all_dataset_auc(rows: list[dict[str, object]]) -> None:
    names = [str(row["display"]) for row in rows]
    y = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10.8, 6.8))
    offsets = {"B1": -0.22, "B2": 0.0, "Ours": 0.22}
    for condition in ("B1", "B2", "Ours"):
        values = np.array([float(row[f"{condition.lower()}_auc"]) for row in rows])
        mask = np.isfinite(values)
        ax.scatter(values[mask], y[mask] + offsets[condition], s=68, color=COLORS[condition], label=condition, zorder=3)
        for value, yy in zip(values[mask], y[mask] + offsets[condition], strict=True):
            ax.text(value + 0.012, yy, f"{value:.3f}", va="center", fontsize=8.2, color=COLORS[condition])
    ax.axvline(0.5, color="#B7BEC2", linestyle="--", linewidth=1)
    ax.set_xlim(0.35, 1.06)
    ax.set_yticks(y, names)
    ax.invert_yaxis()
    ax.set_xlabel("Macro AUROC (higher is better)")
    fig.subplots_adjust(top=0.86, bottom=0.12)
    fig.suptitle("Current held-out numerical snapshot across registered datasets", x=0.12, y=0.97, ha="left", fontsize=16, weight="bold")
    fig.text(0.12, 0.92, "Missing B2 markers mean the direct-agent arm has not completed; mixed snapshots are not a final harmonized study.", fontsize=9, color="#5C6870")
    ax.grid(axis="x", color="#E6EAEC", linewidth=0.8)
    fig.legend(frameon=False, ncol=3, loc="upper right", bbox_to_anchor=(0.93, 0.965))
    save(fig, "interim_all_dataset_auroc")


def plot_ncmmsc() -> None:
    metrics = [
        ("accuracy", "Accuracy", True),
        ("macro_f1", "Macro F1", True),
        ("macro_auroc_ovr", "Macro AUROC", True),
        ("macro_auprc", "Macro AUPRC", True),
        ("ece", "ECE", False),
        ("multiclass_brier", "Brier", False),
    ]
    x = np.arange(len(metrics))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10.5, 5.3))
    for offset, condition in [(-width / 2, "B1"), (width / 2, "Ours")]:
        values = [read_metric("NCMMSC2021_AD", metric, condition) for metric, _, _ in metrics]
        bars = ax.bar(x + offset, values, width, color=COLORS[condition], label=condition)
        ax.bar_label(bars, labels=[f"{value:.3f}" for value in values], padding=3, fontsize=8.4)
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x, [label for _, label, _ in metrics])
    ax.set_ylabel("Score")
    fig.subplots_adjust(top=0.82)
    fig.suptitle("NCMMSC 2021 long-recording snapshot", x=0.11, y=0.97, ha="left", fontsize=16, weight="bold")
    fig.text(0.11, 0.90, "Discrimination improves slightly; fixed-threshold classification does not. ECE and Brier are lower-is-better.", fontsize=9, color="#5C6870")
    ax.grid(axis="y", color="#E6EAEC", linewidth=0.8)
    ax.legend(frameon=False, ncol=2)
    save(fig, "interim_ncmmsc_results")


def plot_three_arm(rows: list[dict[str, object]]) -> None:
    complete = [row for row in rows if row["b2"] == "completed"]
    names = [str(row["display"]) for row in complete]
    x = np.arange(len(names))
    width = 0.24
    fig, axes = plt.subplots(2, 1, figsize=(10.8, 8.2), sharex=True)
    for ax, key, title in [
        (axes[0], "auc", "Macro AUROC"),
        (axes[1], "accuracy", "Accuracy"),
    ]:
        for index, condition in enumerate(("B1", "B2", "Ours")):
            values = [float(row[f"{condition.lower()}_{key}"]) for row in complete]
            bars = ax.bar(x + (index - 1) * width, values, width, color=COLORS[condition], label=condition)
            ax.bar_label(bars, labels=[f"{value:.2f}" for value in values], padding=2, fontsize=7.8, rotation=90)
        ax.set_ylim(0, 1.08)
        ax.set_ylabel(title)
        ax.grid(axis="y", color="#E6EAEC", linewidth=0.8)
    axes[0].set_title("Completed real three-condition comparisons", loc="left", fontsize=16, weight="bold")
    axes[0].legend(frameon=False, ncol=3, loc="lower right")
    axes[1].set_xticks(x, names)
    axes[1].set_xlabel("Dataset / task")
    save(fig, "interim_completed_three_arm")


def copy_core_figures() -> None:
    source = ROOT / "reports" / "latest" / "assets"
    for stem in [
        "figure_1_system_architecture",
        "figure_2_dataset_landscape",
        "figure_3_channel_state_matrix",
        "figure_10_metric_to_state_map",
    ]:
        for suffix in ("png", "svg", "pdf"):
            path = source / f"{stem}.{suffix}"
            if path.exists():
                shutil.copy2(path, ASSETS / path.name)


def metric_table(rows: list[dict[str, object]]) -> str:
    body = []
    for row in rows:
        complete = row["b2"] == "completed" and row["ours_report"] == "completed"
        state = "Three-arm complete" if complete else "Numerical snapshot; agent arm pending"
        fmt = lambda value: "—" if not np.isfinite(float(value)) else f"{float(value):.3f}"
        body.append(
            "<tr>"
            f"<td>{html.escape(str(row['display']))}</td><td>{row['subjects']}</td><td>{row['test']}</td>"
            f"<td>{fmt(row['b1_auc'])}</td><td>{fmt(row['b2_auc'])}</td><td>{fmt(row['ours_auc'])}</td>"
            f"<td><span class='pill {'done' if complete else 'pending'}'>{state}</span></td></tr>"
        )
    return "".join(body)


def write_oral_script() -> str:
    script = """# 阶段性口语汇报稿

这次工作不是只把一个新数据集接进旧脚本，而是把原来按日期分散的处理、训练、评估和报告过程，整理成一套可以追溯、可以增量更新的工程系统。当前系统已经登记十个独立任务，每个任务分别建立受试者级训练集和测试集，不把不同数据集混在一起训练，也不让同一位受试者跨越训练集和测试集。

系统的三个比较条件已经固定。第一种是传统声学机器学习，使用人工设计的声学特征和正则化逻辑回归。第二种是直接大模型，只读取去标识化转录文本，自行给出分类和报告。第三种是我们的系统。这里需要特别说明，我们的数值预测本身不是由大模型代理完成，而是由可复现的监督学习完成；大模型只在概率冻结以后，把指标证据、临床状态和音频片段翻译成医生报告，不能修改风险概率。因此更准确的定位是“带报告代理的临床筛查系统”，而不是“由代理直接决策的模型”。

这次框架有几项实质修改。第一，指标不再只是一个数，而是带有参照组、异常方向、可靠度、缺失情况、混杂标签和报告权限的证据对象。第二，同类指标先在状态内部融合，再由语言、语音行为、互动和辅助声学分支分别建模；录音质量只用于降低可靠度和去除技术成分，不直接增加疾病风险。第三，Pitt 多任务数据原来会出现跨任务病例编号重复，现在已经改为录音级唯一编号，同时仍按受试者分组切分。第四，进展预测任务不能用健康对照作为参照，因此已经改为使用训练集中的未下降组建立参照。第五，新增逐录音缓存，以后加入新数据时只重算新增或发生变化的录音。

目前已经完成真实三条件比较的任务有 IAEAV、ADReSS 2020、ADReSSo 诊断、ADReSSo 进展和 PROCESS-2。结果不是所有数据集都支持我们的系统领先。PROCESS-2 上，我们的宏平均受试者工作特征曲线下面积达到 0.834，高于传统声学的 0.733 和直接大模型的 0.723；ADReSSo 诊断上，我们的准确率为 0.595，是三种方法中最高，校准误差也最低，但受试者工作特征曲线下面积没有超过传统声学。ADReSS 2020 和进展预测上，我们的当前结果较弱，其中进展任务还受到刚发现的参照组设置错误影响，所以修正后必须重新训练，旧数值不能作为最终结论。IAEAV 上传统声学达到近乎完美的区分能力，这更像数据集特异性或采集捷径信号，不能直接解释为临床泛化能力。

新加入的 NCMMSC 2021 数据已经纳入 399 条长录音、175 名受试者，并明确排除了六秒短片段。现有数值快照中，我们的方法宏平均受试者工作特征曲线下面积为 0.904，略高于传统声学的 0.894；校准误差从 0.122 降至 0.100，概率误差也略有下降。但固定阈值准确率为 0.755，低于传统声学的 0.774，宏平均 F1 也略低。因此目前能说的是排序和校准略有改善，不能说整体分类性能已经全面超过基线。暂停时中文转录已经缓存 134 条，直接大模型条件尚未完成，所以 NCMMSC 还不是完整三条件结果。

接下来的工作很明确：继续完成 NCMMSC、PREPARE 和 TAUKADIAL 的转录缓存；用修正后的任务参照、Pitt 唯一病例键和逐录音缓存统一重跑十个任务；完成所有测试对象的直接大模型分类；重新生成医学预测指标、校准与负控、框架可追溯性、报告质量和干预实验。最终报告必须把“已经完成”“等待重跑”和“仍需真实医生验证”分开，不能把自动评分写成医生验证。
"""
    (OUTPUT / "oral_presentation_zh.md").write_text(script, encoding="utf-8")
    return script


def main() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    style()
    rows = dataset_rows()
    plot_all_dataset_auc(rows)
    plot_ncmmsc()
    plot_three_arm(rows)
    copy_core_figures()
    oral = write_oral_script()
    oral_body = html.escape(oral.removeprefix("# 阶段性口语汇报稿\n\n"))
    asr = {"NCMMSC 2021": (134, 399), "PREPARE": (882, 2058), "TAUKADIAL": (158, 507)}
    asr_html = "".join(
        f"<div class='progress-row'><div><strong>{name}</strong><span>{done}/{total} ({done/total:.1%})</span></div><progress value='{done}' max='{total}'></progress></div>"
        for name, (done, total) in asr.items()
    )
    report = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADvoice 阶段性进展与评估汇报</title><style>
:root{{--ink:#18252b;--muted:#637078;--line:#d9dfdc;--paper:#fff;--wash:#f5f7f6;--green:#2a8c82;--coral:#e07a6a;--amber:#d99b35}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"PingFang SC","Microsoft YaHei",sans-serif;line-height:1.68}}header{{background:#fff;border-bottom:1px solid var(--line)}}header>div,main{{max-width:1220px;margin:auto;padding:34px 34px}}h1{{font-size:38px;line-height:1.18;margin:4px 0 10px;letter-spacing:0}}h2{{font-size:25px;margin:0 0 18px}}h3{{font-size:18px;margin:0 0 8px}}p{{margin:8px 0}}.eyebrow{{font-size:12px;letter-spacing:.08em;text-transform:uppercase;color:var(--green);font-weight:700}}.meta{{color:var(--muted)}}section{{background:#fff;border-top:1px solid var(--line);padding:30px;margin:0 0 22px}}.notice{{border-left:4px solid var(--amber);background:#fff9ec;padding:15px 18px;margin:18px 0}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:16px}}.stat{{border:1px solid var(--line);padding:18px;min-height:120px}}.stat b{{display:block;font-size:30px;line-height:1.1;color:var(--green)}}.stat span{{color:var(--muted);font-size:13px}}.figure{{width:100%;height:auto;display:block;margin:12px auto}}.caption{{font-size:13px;color:var(--muted);border-top:1px solid var(--line);padding-top:9px}}table{{width:100%;border-collapse:collapse;font-size:13px}}th,td{{padding:10px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}}th{{background:#f2f5f4}}.pill{{display:inline-block;padding:3px 7px;border:1px solid;border-radius:3px;font-size:11px}}.done{{color:#1b746b;background:#eef8f5}}.pending{{color:#96651d;background:#fff8e8}}.progress-row{{margin:14px 0}}.progress-row>div{{display:flex;justify-content:space-between}}progress{{width:100%;height:12px;accent-color:var(--green)}}.change{{display:grid;grid-template-columns:180px 1fr 190px;gap:16px;border-bottom:1px solid var(--line);padding:15px 0}}.change .result{{color:#315f59}}.oral{{white-space:pre-wrap;background:#f7f9f8;border:1px solid var(--line);padding:22px;font-size:15px}}code{{font-size:12px}}@media(max-width:800px){{header>div,main{{padding:24px 16px}}.grid{{grid-template-columns:1fr}}.change{{grid-template-columns:1fr}}table{{display:block;overflow:auto}}h1{{font-size:30px}}}}
</style></head><body><header><div><div class="eyebrow">Interim progress report · 13 August 2026</div><h1>ADvoice 阶段性进展与评估汇报</h1><p class="meta">本页是暂停点快照，不是最终统一重跑结果。所有未完成项和混合版本均显式标注。</p></div></header><main>
<section><h2>1. 当前结论边界</h2><div class="grid"><div class="stat"><b>10</b><span>已登记并按数据集独立切分的任务</span></div><div class="stat"><b>5</b><span>已完成真实 B1 / B2 / Ours 三条件的任务</span></div><div class="stat"><b>19/19</b><span>当前代码自动化测试通过</span></div></div><div class="notice"><strong>Ours 是否是 agent：</strong>数值预测不是代理，而是可靠度感知的分层监督学习；概率冻结以后，受约束的临床报告代理才生成医生文字报告，且不能改变类别或概率。因此系统定位是 <em>agent-assisted clinical screening system</em>。</div></section>
<section><h2>2. 数据与系统架构</h2><img class="figure" src="assets/figure_1_system_architecture.png"><p class="caption">Figure 1. 预测链、证据链与报告代理边界。该图来自当前固定系统模板。</p><img class="figure" src="assets/figure_2_dataset_landscape.png"><p class="caption">Figure 2. 当前数据通道、语言和诊断构成。</p></section>
<section><h2>3. 本轮框架修改及其结果</h2>
<div class="change"><strong>全数据工程化</strong><span>十个任务统一进入配置驱动的 manifest → transcript → feature → MetricEvidence → StateCard → fusion → evaluation → report 链路，仍按数据集独立训练。</span><span class="result">已实现；最终统一重跑待完成</span></div>
<div class="change"><strong>代理边界</strong><span>B2 是直接文本诊断代理；Ours 的风险由监督学习产生，报告代理只翻译冻结证据。</span><span class="result">已实现并写入模型元数据</span></div>
<div class="change"><strong>通道特异证据</strong><span>访谈、图片描述、多任务、公开视频、自由讲话和进展语音使用不同状态集合、可靠度折扣、混杂标签和报告权限。</span><span class="result">已实现；见动态图和通道矩阵</span></div>
<div class="change"><strong>质量控制正交化</strong><span>在每个训练折内，用时长、噪声、增益和角色覆盖解释状态中的技术成分；质量控制不直接进入疾病风险。</span><span class="result">已实现并通过单元测试</span></div>
<div class="change"><strong>Pitt 病例键</strong><span>修正同一受试者在 Cookie、流畅性、回忆等任务中出现重复病例编号的问题；保持受试者级分组切分。</span><span class="result">已修正；Pitt 必须重跑</span></div>
<div class="change"><strong>任务参照组</strong><span>诊断任务使用训练集健康对照；进展任务使用训练集未下降组，不再硬编码健康对照。</span><span class="result">已修正；进展结果必须重跑</span></div>
<div class="change"><strong>增量特征缓存</strong><span>缓存键包含音频内容、分析区间、转录内容、语言、转录可靠度和提取器版本；新增录音不再触发全库特征重算。</span><span class="result">已实现并通过身份重绑定测试</span></div>
<div class="change"><strong>干预实验定义</strong><span>只对测试集误分类病例替换最偏离的状态，并明确这是机制压力测试，不是医生标注的因果反事实。</span><span class="result">代码已修正；旧 Layer B 数值停用</span></div>
</section>
<section><h2>4. Metric 到 State 的现行设计</h2><img class="figure" src="assets/figure_3_channel_state_matrix.png"><p class="caption">Figure 3. 不同数据通道实际启用和暂缓的临床状态。</p><img class="figure" src="assets/figure_10_metric_to_state_map.png"><p class="caption">Figure 4. 33 个现行指标到 14 个状态的映射。已规划但没有有效算法或评分键的状态不会进入模型。</p></section>
<section><h2>5. 当前评估快照</h2><img class="figure" src="assets/interim_completed_three_arm.png"><p class="caption">Figure 5. 已完成真实三条件的五个任务。ADReSSo progression 的旧结果受参照组错误影响，修正后必须重跑。</p><img class="figure" src="assets/interim_all_dataset_auroc.png"><p class="caption">Figure 6. 十个任务当前可读取的最近数值快照。缺失 B2 的任务不能用于三条件胜负判断。</p><table><thead><tr><th>Dataset</th><th>Subjects</th><th>Test</th><th>B1 AUROC</th><th>B2 AUROC</th><th>Ours AUROC</th><th>Status</th></tr></thead><tbody>{metric_table(rows)}</tbody></table></section>
<section><h2>6. 新数据：NCMMSC 2021 长录音</h2><p>当前 manifest 包含 <strong>399 条长录音、175 名受试者</strong>，标签为健康对照、轻度认知障碍和阿尔茨海默病；六秒短音和无标签测试音频均未进入本轮建模。</p><img class="figure" src="assets/interim_ncmmsc_results.png"><p class="caption">Figure 7. 当前既有数值快照：Ours 的宏平均 AUROC 为 0.904，B1 为 0.894；Ours 的 ECE 为 0.100，B1 为 0.122。但 Ours 准确率 0.755 低于 B1 的 0.774，不能表述为全面超过基线。</p><div class="notice">该图来自暂停前已经完成的数值模型快照，不包含本轮尚未完成的全部中文 ASR 和 B2。它可用于阶段汇报，不能作为最终三条件实验结论。</div></section>
<section><h2>7. 暂停位置</h2>{asr_html}<p>DementiaBank/Pitt 绝大多数病例已有人工 CHAT 转录；本轮仅对一个空转录病例生成 ASR。所有已完成缓存均已保留，恢复后不会从头转写。</p></section>
<section><h2>8. 当前结果如何解释</h2><ul><li>PROCESS-2 是目前最明确支持新融合框架的数据：Ours 宏平均 AUROC 0.834，高于 B1 的 0.733 和 B2 的 0.723。</li><li>ADReSSo diagnosis 中 Ours 准确率 0.595、ECE 0.036，但 AUROC 0.586 未超过 B1 的 0.614；分类阈值表现和排序能力必须分开解释。</li><li>ADReSS 2020 中 Ours 当前低于两个基线，说明现有状态和门控不能自动保证跨任务有效。</li><li>IAEAV 的 B1 AUROC 1.000 且去掉时长和响度后仍接近完美，需要继续做站点、录音链路和采访协议捷径审计，不能当作临床泛化证据。</li><li>ADReSSo progression 的旧 Ours AUROC 0.444 不能继续使用，因为旧状态参照错误；修正后的结果尚未产生。</li></ul></section>
<section><h2>9. 接下来必须完成</h2><ol><li>完成 NCMMSC、PREPARE、TAUKADIAL 的逐录音 ASR；保留人工转录优先规则。</li><li>用任务特异参照、Pitt 唯一病例键和最新缓存代码统一重跑十个任务。</li><li>把 PREPARE 和 PROCESS-2 的 B2 扩展到完整测试集，形成真正同一测试对象的三条件比较。</li><li>重新计算 Layer A：AUROC 与 95% CI、准确率、平衡准确率、F1、AUPRC、敏感度/特异度、PPV/NPV、MCC、Brier、ECE、校准斜率/截距、混淆矩阵和负控。</li><li>重新计算 Layer B：证据与状态完整性、分支贡献、报告权限、片段回溯、修正后的错误病例状态干预、消融和盲评报告量表。</li><li>最终报告把自动代理评分与真实医生验证分开；后者仍需医生参与，当前没有完成。</li></ol></section>
<section><h2>10. 中文口语汇报</h2><div class="oral">{oral_body}</div><p class="caption">独立文本文件：<a href="oral_presentation_zh.md">oral_presentation_zh.md</a></p></section>
</main></body></html>"""
    (OUTPUT / "current_progress_evaluation_report.html").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
