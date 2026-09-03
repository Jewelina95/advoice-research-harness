from __future__ import annotations

import argparse
import html
import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize


LABELS = ["HC", "MCI", "AD"]
SPEECHCARE = {
    "micro_f1": 0.7211,
    "micro_auroc_ovr": 0.8683,
    "weighted_auroc_ovr": 0.8067,
    "micro_auprc": 0.7473,
    "weighted_auprc": 0.7350,
    "log_loss": 0.6460,
}


def metric_bundle(frame: pd.DataFrame) -> dict[str, float]:
    y = frame["label"].astype(str).to_numpy()
    pred = frame["predicted_label"].astype(str).to_numpy()
    probability = frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    binary = label_binarize(y, classes=LABELS)
    return {
        "n": float(len(frame)),
        "accuracy": accuracy_score(y, pred),
        "micro_f1": f1_score(y, pred, labels=LABELS, average="micro", zero_division=0),
        "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
        "micro_auroc_ovr": roc_auc_score(binary, probability, average="micro", multi_class="ovr"),
        "weighted_auroc_ovr": roc_auc_score(binary, probability, average="weighted", multi_class="ovr"),
        "macro_auroc_ovr": roc_auc_score(binary, probability, average="macro", multi_class="ovr"),
        "micro_auprc": average_precision_score(binary, probability, average="micro"),
        "weighted_auprc": average_precision_score(binary, probability, average="weighted"),
        "log_loss": log_loss(y, probability, labels=LABELS),
    }


def subgroup_metrics(predictions: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    columns = ["subject_id", "language", "corpus"]
    merged = predictions.merge(manifest[columns].drop_duplicates("subject_id"), on="subject_id", how="left")
    rows: list[dict[str, object]] = []
    for language, group in merged.groupby("language", dropna=False):
        y = group["label"].astype(str)
        pred = group["predicted_label"].astype(str)
        row: dict[str, object] = {
            "language": str(language),
            "n": len(group),
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, labels=LABELS, average="macro", zero_division=0),
            "macro_auroc_ovr": np.nan,
        }
        if y.nunique() == len(LABELS):
            probability = group[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
            row["macro_auroc_ovr"] = roc_auc_score(
                label_binarize(y, classes=LABELS), probability, average="macro", multi_class="ovr"
            )
        rows.append(row)
    return pd.DataFrame(rows)


def class_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    precision, recall, f1, support = precision_recall_fscore_support(
        predictions["label"], predictions["predicted_label"], labels=LABELS, zero_division=0
    )
    return pd.DataFrame(
        {
            "class": LABELS,
            "support": support,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
    )


def latest_dataset_summary(root: Path) -> pd.DataFrame:
    latest = json.loads((root / "reports/latest_runs.json").read_text(encoding="utf-8"))
    rows: list[dict[str, object]] = []
    for dataset, record in latest.items():
        metrics_path = Path(record["run_dir"]) / "artifacts/layer_a_metrics.csv"
        if not metrics_path.exists():
            continue
        metrics = pd.read_csv(metrics_path)
        metrics = metrics[metrics["analysis_scope"].eq("full_available_cohort")]
        for condition in ["B1", "B2", "Ours"]:
            subset = metrics[metrics["condition"].eq(condition)].set_index("metric")["value"]
            rows.append(
                {
                    "dataset": dataset,
                    "condition": condition,
                    "accuracy": subset.get("accuracy", np.nan),
                    "macro_f1": subset.get("macro_f1", np.nan),
                    "macro_auroc_ovr": subset.get("macro_auroc_ovr", np.nan),
                    "log_loss": subset.get("log_loss", np.nan),
                    "ece": subset.get("ece", np.nan),
                }
            )
    return pd.DataFrame(rows)


def pct(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{100 * value:.2f}%"


def num(value: float) -> str:
    return "—" if not np.isfinite(value) else f"{value:.3f}"


def table(frame: pd.DataFrame, percent_columns: set[str] | None = None) -> str:
    percent_columns = percent_columns or set()
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    body = []
    for record in frame.to_dict("records"):
        cells = []
        for column in frame.columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                rendered = pct(float(value)) if column in percent_columns else num(float(value))
            else:
                rendered = html.escape(str(value))
            cells.append(f"<td>{rendered}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{header}</tr></thead><tbody>{''.join(body)}</tbody></table></div>"


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for record in frame.to_dict("records"):
        values = []
        for column in frame.columns:
            value = record[column]
            if isinstance(value, (float, np.floating)):
                values.append(num(float(value)))
            else:
                values.append(str(value).replace("|", "\\|"))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def comparison_rows(ours: dict[str, float]) -> pd.DataFrame:
    labels = {
        "micro_f1": "Micro F1 / accuracy",
        "micro_auroc_ovr": "Micro AUROC",
        "weighted_auroc_ovr": "Weighted AUROC",
        "micro_auprc": "Micro average precision",
        "weighted_auprc": "Weighted average precision",
        "log_loss": "Multiclass log loss",
    }
    rows = []
    for key, label in labels.items():
        direction = "lower" if key == "log_loss" else "higher"
        ours_value = ours[key]
        paper_value = SPEECHCARE[key]
        delta = ours_value - paper_value
        better = delta < 0 if direction == "lower" else delta > 0
        rows.append(
            {
                "metric": label,
                "Ours_current": ours_value,
                "SpeechCARE_published": paper_value,
                "delta": delta,
                "Ours_better": "yes" if better else "no",
            }
        )
    return pd.DataFrame(rows)


def render_report(
    root: Path,
    ours: dict[str, float],
    comparison: pd.DataFrame,
    classes: pd.DataFrame,
    subgroup: pd.DataFrame,
    confusion: np.ndarray,
    model: dict,
    layer_b: pd.DataFrame,
    dataset_summary: pd.DataFrame,
) -> str:
    ours_best = (
        dataset_summary.pivot(index="dataset", columns="condition", values="macro_auroc_ovr")
        .assign(best=lambda x: x.max(axis=1))
    )
    wins = int(np.isclose(ours_best["Ours"], ours_best["best"], equal_nan=False).sum())
    dataset_count = int(len(ours_best))
    confusion_frame = pd.DataFrame(confusion, index=[f"true {x}" for x in LABELS], columns=[f"pred {x}" for x in LABELS]).reset_index(names="")
    gate = pd.DataFrame(
        [
            {
                "branch": branch,
                "mean_weight": model["mean_test_branch_weights"].get(branch, np.nan),
                "weight_sd": model["test_branch_weight_sd"].get(branch, np.nan),
                "cap": model["branch_effective_caps"].get(branch, np.nan),
            }
            for branch in model["mean_test_branch_weights"]
        ]
    )
    layer_b_view = layer_b[["condition", "check", "value", "passed"]].copy()
    layer_b_view["passed"] = layer_b_view["passed"].map({True: "passed", False: "failed"}).fillna("not run")
    dataset_view = dataset_summary[["dataset", "condition", "accuracy", "macro_f1", "macro_auroc_ovr"]]
    findings = [
        ("Critical", "筛查阈值使用测试标签选择", "evaluation.py 会枚举当前评估集的真实标签和概率，选择 sensitivity≥0.85 时 specificity 最大的阈值。该阈值不是可部署操作点，相关 sensitivity、specificity、PPV、NPV 均受测试集污染。"),
        ("Critical", "同协议预测未超过 SpeechCARE", "PREPARE 官方测试集 n=412 上，Ours 的 Micro AUROC、Weighted AUROC、Micro F1、AP 和 log loss 全部落后。当前不能写成性能优于 SpeechCARE。"),
        ("Critical", "AD 与 MCI 召回失败", f"当前 AD recall={classes.loc[classes['class'].eq('AD'), 'recall'].iloc[0]:.3f}，MCI recall={classes.loc[classes['class'].eq('MCI'), 'recall'].iloc[0]:.3f}；模型主要预测 HC，作为筛查器不可接受。"),
        ("Critical", "Pitt 任务与诊断标签共线", "Cookie 描述同时包含 AD/HC，但 fluency、recall、sentence 等任务几乎只有 AD。任务特异状态可能通过任务是否存在猜标签，而不是测量同任务下的认知差异；必须增加 task-only 负控和 task-matched 分析。"),
        ("High", "实际门控不读取任务或内容", "门控只读取三个分支的平均可靠度，不读取任务、语言、状态值、音频/文本表示或分支冲突；辅助声学分支又频繁触及 0.35 上限，因此接近固定加权。"),
        ("High", "PREPARE 任务路由在真实运行中失效", "全部样本被写成同一个 prepare_cognitive_task_audio；所有状态卡均为 overall。图片描述、朗读、故事回忆和语义流畅性差异在融合前被抹平。"),
        ("High", "多语言文本规则对中文不成立", "中文转录按单字切分，但 filler 配置使用多字短语；代词与停用词主要覆盖英语/西班牙语。这会把语言差异写入 content_word_ratio、pronoun_ratio 等状态，而非认知差异。"),
        ("High", "未知说话人被当作完整患者覆盖", "没有 diarization 时，role_coverage_fraction 仍可为 1.0；这表示整段被使用，不表示患者语音覆盖完整。采访者提示与轮次声学可能进入疾病分支。"),
        ("High", "负控暴露数据捷径", "Pitt 的 QC-only AUROC 高于 Ours；ADReSSo diagnosis 也出现相同方向，IAEAV 的去时长/去响度负控仍接近完美。高分不能自动解释为认知语言信号。"),
        ("High", "Layer B 多项检查只是字段存在性", "report-permission 只要存在任意 Ours 报告就记 1；evidence-span faithfulness 只检查列表非空；不适用的 task trace 被标为 passed。这些不能证明证据正确。"),
        ("High", "Agent 不是预测器", "Ours Agent 接收冻结概率和允许报告的状态证据，只负责医生报告翻译。它可以改善边界和可追溯性，但不会提高 AUROC，不能与 SpeechCARE 预测网络作为同一对象比较。"),
        ("Medium", "临床状态阈值尚未验证", "normal/borderline/impaired 使用固定 z=1/2 和 reliability=0.45 阈值；这些是工程规则，不是经过量表、专家标注或外部队列验证的临床界值。"),
        ("Medium", "片段轨迹信息不足", "当前 10 秒不重叠窗口只保存静音比例和 RMS，尚未区分话内停顿、轮次停顿、采访者诱发停顿，也没有把 ASR 文本时间戳完整对齐到状态证据。"),
        ("Medium", "校准与内部选择仍偏乐观", "分支超参数选择、门控学习和堆叠校准共享 OOF 预测；测试集未泄漏，但内部性能和模型选择需要嵌套交叉验证或独立元校准集。"),
        ("Medium", "复现链缺少发表级锁定", "运行清单记录哈希是优点，但依赖没有 lockfile，上游 full-run 引用未全部使用不可变标识；单元测试不等于端到端、冻结外部模型或报告事实忠实性测试。"),
    ]
    finding_html = "".join(
        f"<article class='finding'><span class='severity {severity.lower()}'>{severity}</span><h3>{title}</h3><p>{body}</p></article>"
        for severity, title, body in findings
    )
    return f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADvoice 独立架构与 Agent 审查</title>
<style>
:root{{--ink:#17212b;--muted:#607080;--line:#d7dde2;--paper:#fff;--blue:#3b73b9;--green:#478967;--amber:#c28a2e;--red:#b94b52;--soft:#f5f7f8}}
*{{box-sizing:border-box}} body{{margin:0;background:#eef1f3;color:var(--ink);font:16px/1.65 Arial,"Noto Sans SC",sans-serif;letter-spacing:0}}
main{{max-width:1240px;margin:0 auto;background:var(--paper);padding:54px 68px 80px}} h1{{font-size:38px;line-height:1.18;margin:0 0 10px}} h2{{font-size:25px;margin:48px 0 16px;border-top:1px solid var(--line);padding-top:26px}} h3{{font-size:17px;margin:8px 0 4px}} p{{margin:8px 0}} .lead{{font-size:19px;max-width:980px}} .meta{{color:var(--muted)}}
.verdict{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:28px 0}} .verdict div{{border-top:5px solid var(--red);background:var(--soft);padding:18px}} .verdict b{{display:block;font-size:22px}}
.pipeline{{display:grid;grid-template-columns:repeat(8,minmax(110px,1fr));gap:8px;overflow:auto;padding:10px 0}} .step{{border:1px solid var(--line);border-top:4px solid var(--blue);padding:12px;min-height:126px;background:white}} .step strong{{display:block;margin-bottom:7px}} .step span{{font-size:13px;color:var(--muted)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);margin:14px 0 24px}} table{{border-collapse:collapse;width:100%;font-size:14px}} th{{background:#edf2f5;text-align:left}} th,td{{padding:9px 11px;border-bottom:1px solid var(--line);white-space:nowrap}} tr:last-child td{{border-bottom:0}}
.finding{{position:relative;border-left:5px solid var(--line);padding:8px 15px 12px;margin:10px 0;background:#fafbfb}} .severity{{font-size:12px;font-weight:bold;text-transform:uppercase}} .critical{{color:var(--red)}} .high{{color:#a96823}} .medium{{color:var(--blue)}}
.compare{{display:grid;grid-template-columns:1fr 1fr;gap:20px}} .box{{border:1px solid var(--line);padding:18px}} .box h3{{margin-top:0}} .good{{border-top:5px solid var(--green)}} .bad{{border-top:5px solid var(--red)}}
.code{{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13px;background:#f2f4f5;padding:2px 5px}} .note{{border-left:5px solid var(--amber);padding:10px 15px;background:#fff8e8}}
ol li{{margin:9px 0}} footer{{margin-top:50px;padding-top:20px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
@media(max-width:850px){{main{{padding:30px 20px}}.verdict,.compare{{grid-template-columns:1fr}}.pipeline{{grid-template-columns:repeat(8,150px)}}h1{{font-size:30px}}}}
</style></head><body><main>
<p class="meta">Independent reviewer-agent audit · {date.today().isoformat()} · frozen artifacts only</p>
<h1>ADvoice 系统架构、融合与 Agent 独立审查</h1>
<p class="lead">结论先行：当前系统在“证据治理与报告可追溯”方向上比纯黑箱多模态模型多了一层有价值的临床约束，但预测端尚未达到 SpeechCARE 的表示学习和自适应融合水平。PREPARE 官方测试协议下，当前结果不能支持“优于 SpeechCARE”；筛查操作点还存在测试集阈值污染，必须先修复再讨论临床性能。</p>
<div class="verdict"><div><b>预测优越性：未成立</b>同一 PREPARE 测试集的全部公开主指标落后。</div><div><b>临床可追溯：有结构优势</b>状态卡、报告权限和冻结概率边界是实质升级。</div><div><b>临床有效性：尚未证明</b>自动字段审查不能替代证据正确性与医生实验。</div></div>

<h2>1. 代码实际运行的架构</h2>
<div class="pipeline">
<div class="step"><strong>1 数据路由</strong><span>audio / transcript / task / speaker role</span></div>
<div class="step"><strong>2 指标抽取</strong><span>手工声学、词汇、句法、停顿和 QC</span></div>
<div class="step"><strong>3 证据对象</strong><span>value + direction + reliability + confound + report permission</span></div>
<div class="step"><strong>4 折内参照</strong><span>仅训练折 HC 的 median / MAD robust z</span></div>
<div class="step"><strong>5 状态卡</strong><span>可靠度加权的状态内静态融合</span></div>
<div class="step"><strong>6 QC 正交化</strong><span>Ridge 去除 QC 可预测的状态分量</span></div>
<div class="step"><strong>7 分支门控</strong><span>状态逻辑回归 + 辅助声学 + reliability softmax gate</span></div>
<div class="step"><strong>8 冻结后报告</strong><span>Agent 不改概率，只翻译允许报告的证据</span></div>
</div>
<p class="note">当前 Python 预测链没有 mHuBERT、mGTE、DistilBERT 或其他 Transformer encoder。它是可解释手工指标系统，不应在汇报中称为已训练的多模态 Transformer Agent。</p>

<h2>2. 与 SpeechCARE 的同协议结果</h2>
<p>两者都使用 PREPARE 官方训练集 1,646 人和测试集 412 人。SpeechCARE 的数字来自论文最终模型；Ours 由冻结的 <span class="code">ours_predictions.csv</span> 现场重算。Micro F1 在单标签多分类中等于 accuracy。</p>
{table(comparison)}
<p><strong>直接回答：</strong>当前 Ours 的预测效果没有超过 SpeechCARE。尤其是 AD recall 和 MCI recall 过低，不能用总体 ECE 看似较低来掩盖少数类别失败。</p>
<h3>Ours 类别级表现</h3>{table(classes, {"precision", "recall", "f1"})}
<h3>Ours 混淆矩阵</h3>{table(confusion_frame)}
<h3>语言亚组</h3>{table(subgroup, {"accuracy", "macro_f1", "macro_auroc_ovr"})}

<h2>3. 当前门控实际学到了什么</h2>
<p>门控输入只有每个分支的平均可靠度。公式是 <span class="code">softmax(intercept + positive_beta × log(reliability))</span>，之后对辅助声学设置权重上限。它没有根据任务、语言、状态模式或内容表示动态选模态。</p>
{table(gate, {"mean_weight", "weight_sd", "cap"})}
<p>辅助声学平均权重 {pct(model['mean_test_branch_weights'].get('auxiliary_acoustic', np.nan))}，接近 {pct(model.get('auxiliary_weight_cap', np.nan))} 上限；三个分支权重波动很小。因此当前“动态”主要反映可靠度变化，而不是 SpeechCARE 式的样本内容自适应。</p>

<h2>4. 主要审查发现</h2>{finding_html}

<h2>5. Layer B 现在证明了什么</h2>
<p>下面是系统原始 Layer B 输出。它能证明对象存在和链路完整，但不能自动证明医学含义正确。特别是 report-permission 与 evidence-span 两项需要重写验证逻辑。</p>
{table(layer_b_view, {"value"})}

<h2>6. 跨数据集稳定性</h2>
<p>在当前 {dataset_count} 个数据集/任务中，Ours 的 macro AUROC 并非稳定最优，最多在 {wins} 个任务上达到三臂最高值。这个结果支持“部分数据通道有效”，不支持“普遍超过传统 ML 与直接 LLM”。</p>
{table(dataset_view, {"accuracy", "macro_f1", "macro_auroc_ovr"})}

<h2>7. 与论文相比，真正的升级与尚缺部分</h2>
<div class="compare">
<div class="box good"><h3>ADvoice 的实质升级</h3><p>临床构念层：原始特征先经过证据角色、混杂、可靠度和报告权限治理，再形成状态卡。</p><p>可追溯报告层：医生文字只能引用允许报告的状态和片段，且不能让生成模型改写冻结预测。</p><p>质量控制层：QC 用于削弱污染和控制解释，而不是直接作为疾病机制。</p></div>
<div class="box bad"><h3>SpeechCARE 仍明显更强的部分</h3><p>表示学习：mHuBERT 与 mGTE 从音频/文本内容学习高维表征。</p><p>融合：AGF 对样本内容、任务和三个模态的隐藏表示进行联合训练。</p><p>协议：十个随机种子、官方固定测试集、类别阈值优化、语言与人口学公平审计、两个外部数据集。</p></div>
</div>
<p>因此最准确的定位是：ADvoice 增加了 SpeechCARE 缺少的“证据治理和医生报告边界”，但尚未证明预测更强。两者不是简单替代关系。</p>

<h2>8. 收敛后的升级方案</h2>
<ol>
<li><strong>先修评估污染：</strong>所有筛查阈值只能在训练折或独立验证集选择，再冻结到测试集；不再从测试标签搜索操作点。</li>
<li><strong>再修任务路由：</strong>恢复 PREPARE 的真实任务标签；task 与 corpus 分开；在每个训练折内建立任务特异 HC 参照。Pitt 仅做 task-matched 主分析，并增加 task-only 负控。</li>
<li><strong>修复输入真实性：</strong>中文采用词级分词和中文代词/填充词规则；未知角色与患者覆盖率分开；未完成 diarization 的访谈音频不得作患者声学机制解释。</li>
<li><strong>保留临床状态，增加表示分支：</strong>固定窗口只服务 mHuBERT 编码；事件/轮次切分服务状态证据。文本使用多语种表示，低层 embedding 不获准直接写进医生报告。</li>
<li><strong>门控升级为状态约束的自适应融合：</strong>输入 task、language、branch reliability、state vector、audio/text embeddings 和 branch disagreement；保留辅助声学上限、QC 禁止直接增加疾病风险。</li>
<li><strong>训练协议重做：</strong>官方 PREPARE split + 10 seeds；嵌套选择门控和校准；验证集学习类别阈值；报告 micro/weighted AUC、AP、micro F1、log loss 及 HC/MCI/AD precision/recall。</li>
<li><strong>修正 Layer B：</strong>逐字段检查报告引用是否全部有 permission；用时间重叠和语义一致性验证 span；N/A 单独标记；报告评分 Agent 必须看到去标识化的原始证据，而非只看最终文字。</li>
<li><strong>真正临床验证：</strong>医生盲评 B2/Ours 报告，测证据核查正确率、纠错后决策变化、审阅时间和评分者一致性；在完成前只称“筛查研究原型”。</li>
</ol>

<h2>9. 可核查代码位置</h2>
<ul>
<li><span class="code">src/advoice/models.py:272-300</span>：可靠度门控及其训练目标。</li>
<li><span class="code">src/advoice/models.py:464+</span>：Ours 分支训练、融合和校准。</li>
<li><span class="code">src/advoice/states.py:13-20</span>：工程性状态类别阈值。</li>
<li><span class="code">src/advoice/features.py:109-138</span>：10 秒固定窗口及 silence/RMS 轨迹。</li>
<li><span class="code">src/advoice/report_agent.py:39-113</span>：冻结概率后的报告 Agent 边界。</li>
<li><span class="code">src/advoice/evaluation.py:298-453</span>：Layer B 检查逻辑及当前存在性代理指标。</li>
<li><span class="code">src/advoice/evaluation.py:85-123</span>：当前在评估集标签上搜索筛查阈值的污染点。</li>
<li><span class="code">src/advoice/transcripts.py:20-75</span>：多语言 token、filler、pronoun 与 content-word 规则。</li>
<li><span class="code">src/advoice/features.py:57+</span>：角色覆盖与未切分整段音频的处理。</li>
</ul>
<footer>数据来源：本地冻结 run artifacts；<a href="https://www.nature.com/articles/s41746-025-02026-x">SpeechCARE, npj Digital Medicine (2025)</a>；<a href="https://github.com/SpeechCARE/SpeechCARE-NIA-Phase2">作者代码</a>。审查脚本不重新训练模型，不修改预测，只复算与核查。</footer>
</main></body></html>"""


def render_markdown(ours: dict[str, float], comparison: pd.DataFrame, classes: pd.DataFrame) -> str:
    return f"""# ADvoice 独立架构与 Agent 审查

日期：{date.today().isoformat()}

## 结论

当前系统在证据治理、状态卡、QC 边界和冻结后医生报告方面有结构创新，但 PREPARE 官方测试集上没有超过 SpeechCARE。当前 Ours micro AUROC={ours['micro_auroc_ovr']:.4f}、micro F1={ours['micro_f1']:.4f}、weighted AUROC={ours['weighted_auroc_ovr']:.4f}、log loss={ours['log_loss']:.4f}；SpeechCARE 论文对应结果为 0.8683、0.7211、0.8067 和 0.6460。

Ours Agent 不做诊断预测。它接收冻结概率、状态卡与获准报告的证据，仅生成医生文字。因此不能把 Agent 的报告可追溯性当成 AUROC 提升。

## 同协议指标

{markdown_table(comparison)}

## 类别表现

{markdown_table(classes)}

## 必须修复

1. 恢复 PREPARE 真实任务标签和任务特异折内参照。
2. 删除测试集阈值搜索，改为训练/验证集确定并冻结操作点。
3. 修复 Pitt 任务标签共线、中文规则和未知角色覆盖。
4. 增加音频/文本表示分支，但保留状态层和报告权限作为临床约束。
5. 将可靠度标量门控升级为读取 task、language、state、embedding、reliability 与 branch disagreement 的状态约束自适应门控。
6. 用官方 split、10 seeds、嵌套校准和类别阈值优化重新评估。
7. 把 Layer B 从字段存在性改成 permission、时间跨度和语义支持的真实验证。
8. 通过医生盲评证明报告是否改善核查与决策，而不是使用自动结构评分替代临床实验。
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--prepare-run",
        type=Path,
        default=None,
        help="PREPARE evaluation run; defaults to reports/latest_runs.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output directory; defaults to reports/independent_architecture_audit_YYYY-MM-DD",
    )
    args = parser.parse_args()
    root = args.root.resolve()
    latest = json.loads((root / "reports/latest_runs.json").read_text(encoding="utf-8"))
    prepare_run = args.prepare_run or Path(latest["PREPARE_DrivenData"]["run_dir"])
    output = args.output or root / "reports" / f"independent_architecture_audit_{date.today().isoformat()}"
    output.mkdir(parents=True, exist_ok=True)
    artifacts = prepare_run / "artifacts"
    predictions = pd.read_csv(artifacts / "ours_predictions.csv", dtype={"subject_id": str})
    manifest = pd.read_csv(artifacts / "manifest.csv", dtype={"subject_id": str})
    model = json.loads((artifacts / "ours_model.json").read_text(encoding="utf-8"))
    layer_b = pd.read_csv(artifacts / "layer_b_checks.csv")
    ours = metric_bundle(predictions)
    comparison = comparison_rows(ours)
    classes = class_metrics(predictions)
    subgroup = subgroup_metrics(predictions, manifest)
    confusion = confusion_matrix(predictions["label"], predictions["predicted_label"], labels=LABELS)
    dataset_summary = latest_dataset_summary(root)
    comparison.to_csv(output / "speechcare_same_protocol_comparison.csv", index=False)
    classes.to_csv(output / "prepare_class_metrics.csv", index=False)
    subgroup.to_csv(output / "prepare_language_subgroups.csv", index=False)
    dataset_summary.to_csv(output / "cross_dataset_three_condition_summary.csv", index=False)
    (output / "architecture_agent_audit_zh.html").write_text(
        render_report(root, ours, comparison, classes, subgroup, confusion, model, layer_b, dataset_summary),
        encoding="utf-8",
    )
    (output / "architecture_agent_audit_zh.md").write_text(
        render_markdown(ours, comparison, classes), encoding="utf-8"
    )
    print(output / "architecture_agent_audit_zh.html")


if __name__ == "__main__":
    main()
