from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

from advoice.cognitive_extension import LABELS, benchmark_metrics


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "latest" / "speechcare_gap_resolution"
ASSETS = OUT / "assets"


def style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 13,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def metric_figure(result: dict[str, object]) -> None:
    published = result["speechcare_published_mean"]
    released = result["speechcare_released_checkpoint"]
    extended = result["speechcare_plus_cognition"]
    metrics = [
        ("Micro AUROC", "micro_auroc_ovr"),
        ("Micro F1", "micro_f1"),
        ("Weighted AUROC", "weighted_auroc_ovr"),
        ("Micro AUPRC", "micro_auprc"),
        ("Weighted AUPRC", "weighted_auprc"),
    ]
    x = np.arange(len(metrics))
    fig, ax = plt.subplots(figsize=(13, 6.2))
    series = [
        ("SpeechCARE published mean", published, "#9AA0A6"),
        ("SpeechCARE released checkpoint", released, "#4C78A8"),
        ("SpeechCARE + ADvoice cognition", extended, "#D95F59"),
    ]
    for index, (name, values, color) in enumerate(series):
        positions = x + (index - 1) * 0.24
        bars = ax.bar(
            positions,
            [float(values[key]) for _, key in metrics],
            width=0.22,
            color=color,
            label=name,
        )
        ax.bar_label(bars, fmt="%.3f", fontsize=10, padding=3)
    ax.set_ylim(0.68, 0.91)
    ax.set_ylabel("Score")
    ax.set_xticks(x, [label for label, _ in metrics])
    ax.set_title("PREPARE official test: same-cohort retrospective comparison", loc="left", weight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    fig.tight_layout()
    fig.savefig(ASSETS / "protocol_aligned_metrics.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def progression_figure(result: dict[str, object]) -> None:
    original = pd.read_csv(ROOT / "artifacts" / "PREPARE_DrivenData" / "ours_predictions.csv")
    label_index = {label: index for index, label in enumerate(LABELS)}
    original_metrics = benchmark_metrics(
        original["label"].map(label_index).to_numpy(dtype=int),
        original[["prob_HC", "prob_MCI", "prob_AD"]].to_numpy(),
    )
    stages = [
        ("8.27 original", original_metrics),
        ("Cognition model", result["advoice_cognition_model"]),
        ("Strong backbone\n+ cognition", result["speechcare_plus_cognition"]),
    ]
    x = np.arange(len(stages))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    auc = [float(values["micro_auroc_ovr"]) for _, values in stages]
    f1 = [float(values["micro_f1"]) for _, values in stages]
    ax.plot(x, auc, marker="o", markersize=9, linewidth=2.5, color="#4C78A8", label="Micro AUROC")
    ax.plot(x, f1, marker="s", markersize=9, linewidth=2.5, color="#D95F59", label="Micro F1")
    for values in (auc, f1):
        for position, value in zip(x, values, strict=True):
            ax.text(position, value + 0.009, f"{value:.3f}", ha="center", fontsize=11, weight="bold")
    ax.set_xticks(x, [name for name, _ in stages])
    ax.set_ylim(0.62, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("Same test cohort; model training is not protocol-equivalent", loc="left", weight="bold")
    ax.grid(axis="y", color="#E5E7EB", linewidth=0.8)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(ASSETS / "revision_progression.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def confusion_figure() -> None:
    extension = pd.read_csv(
        ROOT / "artifacts" / "PREPARE_DrivenData" / "speechcare_cognitive_extension" / "official_test_predictions.csv"
    )
    speechcare = pd.read_csv(
        ROOT / "references" / "speechcare" / "released_outputs" / "mhubert_test_predictions_after_bias_mitigation.csv",
        dtype={"uid": str},
    )
    merged = extension.merge(speechcare[["uid", "C", "MCI", "ADRD"]], left_on="subject_id", right_on="uid")
    speechcare_prediction = np.asarray(LABELS)[merged[["C", "MCI", "ADRD"]].to_numpy().argmax(axis=1)]
    matrices = [
        ("SpeechCARE released checkpoint", confusion_matrix(merged.true_label, speechcare_prediction, labels=LABELS)),
        ("SpeechCARE + ADvoice cognition", confusion_matrix(merged.true_label, merged.predicted_label, labels=LABELS)),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.8))
    for ax, (title, matrix) in zip(axes, matrices, strict=True):
        image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(item.max() for _, item in matrices))
        for row in range(3):
            for column in range(3):
                ax.text(column, row, str(matrix[row, column]), ha="center", va="center", fontsize=14, weight="bold")
        ax.set_xticks(range(3), LABELS)
        ax.set_yticks(range(3), LABELS)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Observed")
        ax.set_title(title, weight="bold")
    fig.colorbar(image, ax=axes, fraction=0.035, pad=0.04)
    fig.savefig(ASSETS / "confusion_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def build_html(result: dict[str, object]) -> None:
    published = result["speechcare_published_mean"]
    extended = result["speechcare_plus_cognition"]
    candidate = result["advoice_cognition_model"]
    protocol = pd.read_csv(
        ROOT / "references" / "speechcare" / "prepare_protocol_inputs.csv",
        dtype={"uid": str},
    )
    candidate_prediction = pd.read_csv(
        ROOT / "artifacts" / "PREPARE_DrivenData" / "cognitive_fusion_protocol" / "official_test_predictions.csv",
        dtype={"subject_id": str},
    )
    released_prediction = pd.read_csv(
        ROOT / "references" / "speechcare" / "released_outputs" / "mhubert_test_predictions_after_bias_mitigation.csv",
        dtype={"uid": str},
    )
    same_test_subjects = (
        set(candidate_prediction["subject_id"])
        == set(released_prediction["uid"])
        == set(protocol.loc[protocol["reference_partition"].eq("test"), "uid"])
    )
    current_run = json.loads(
        (
            ROOT
            / "artifacts"
            / "PREPARE_DrivenData"
            / "cognitive_fusion_protocol"
            / "result.json"
        ).read_text(encoding="utf-8")
    )
    provenance_complete = bool(
        current_run.get("official_test") and current_run.get("prediction_sha256")
    )
    merged = candidate_prediction.merge(
        protocol[["uid", "task", "language"]],
        left_on="subject_id",
        right_on="uid",
        validate="one_to_one",
    )
    released_labels = np.asarray(LABELS)[
        released_prediction[["C", "MCI", "ADRD"]].to_numpy(dtype=float).argmax(axis=1)
    ]
    released_label_frame = pd.DataFrame(
        {"uid": released_prediction["uid"], "speechcare_prediction": released_labels}
    )
    merged = merged.merge(released_label_frame, on="uid", validate="one_to_one")
    merged["advoice_correct"] = merged["true_label"].eq(merged["predicted_label"])
    merged["speechcare_correct"] = merged["true_label"].eq(
        merged["speechcare_prediction"]
    )
    subgroup = merged.groupby("language").agg(
        n=("uid", "size"),
        advoice=("advoice_correct", "mean"),
        speechcare=("speechcare_correct", "mean"),
    )
    subgroup_rows = "".join(
        f"<tr><td>{language}</td><td>{int(row['n'])}</td>"
        f"<td>{row['advoice']:.3f}</td><td>{row['speechcare']:.3f}</td>"
        f"<td>{row['advoice'] - row['speechcare']:+.3f}</td></tr>"
        for language, row in subgroup.iterrows()
    )
    html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>SpeechCARE 差距审计与认知扩展结果</title>
<style>body{{margin:0;color:#202124;background:#f7f8fa;font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC",sans-serif;line-height:1.65}}main{{max-width:1160px;margin:auto;background:white;padding:48px 64px}}h1{{font-size:34px;line-height:1.25}}h2{{margin-top:42px;border-top:1px solid #dfe3e8;padding-top:28px}}.lead{{font-size:19px;color:#3c4043}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.card{{border:1px solid #dfe3e8;border-radius:6px;padding:18px}}.value{{font-size:30px;font-weight:750;color:#b6423c}}.good{{border-left:5px solid #4f8a63;background:#f3faf5;padding:14px 18px}}.warn{{border-left:5px solid #d49b35;background:#fff9ed;padding:14px 18px}}.bad{{border-left:5px solid #c75852;background:#fff5f4;padding:14px 18px}}img{{width:100%;margin:14px 0 4px}}table{{width:100%;border-collapse:collapse}}th,td{{padding:11px;border-bottom:1px solid #dfe3e8;text-align:left}}th{{background:#eef2f5}}code{{background:#eef2f5;padding:2px 5px;border-radius:3px}}@media(max-width:760px){{main{{padding:24px 18px}}.grid{{grid-template-columns:1fr}}}}</style></head><body><main>
<h1>为什么独立 ADvoice 仍低于 SpeechCARE</h1>
<p class="lead">双方使用的是同一批 PREPARE 官方测试对象，但不是同一套端到端训练协议。412 个测试 ID 完全一致：<b>{str(same_test_subjects).lower()}</b>。独立 ADvoice 的微平均 AUROC 为 {candidate['micro_auroc_ovr']:.3f}，说明排序能力已经接近，但 Micro F1 只有 {candidate['micro_f1']:.3f}；主要差距来自冻结声学编码器、过早压缩时序、文本主干仅局部训练，以及多类别操作点不稳定。这里的分类概率来自监督学习模型，诊断 Agent 没有参与该数值。</p>
<div class="grid"><div class="card"><div>Micro AUROC</div><div class="value">{extended['micro_auroc_ovr']:.3f}</div><div>论文均值 {published['micro_auroc_ovr']:.3f}</div></div><div class="card"><div>Micro F1 / accuracy</div><div class="value">{extended['micro_f1']:.3f}</div><div>论文均值 {published['micro_f1']:.3f}</div></div><div class="card"><div>Micro AUPRC</div><div class="value">{extended['micro_auprc']:.3f}</div><div>论文均值 {published['micro_auprc']:.3f}</div></div></div>
<h2>1. 是否真的是同一数据集</h2><table><tr><th>核对项</th><th>结论</th><th>含义</th></tr><tr><td>官方测试对象</td><td>相同：412 人，ID 集合和顺序完全一致</td><td>测试队列可直接逐病例比较</td></tr><tr><td>训练/验证 ID</td><td>相同快照：1295 / 327</td><td>开发集划分对齐作者公开文件</td></tr><tr><td>转录、任务、语言、年龄</td><td>使用作者公开快照</td><td>结构化输入字段对齐</td></tr><tr><td>声学表示</td><td><b>不相同</b></td><td>SpeechCARE 端到端微调 mHuBERT；ADvoice 使用冻结缓存</td></tr><tr><td>文本表示</td><td><b>不相同</b></td><td>SpeechCARE 全量训练 mGTE；ADvoice 仅训练 E5 最后一层</td></tr><tr><td>重复训练</td><td><b>不相同</b></td><td>论文报告 10 个随机种子；当前 ADvoice 正式预测只有一个种子</td></tr></table>
<h2>2. 根因审计</h2><table><tr><th>差距</th><th>SpeechCARE</th><th>独立 ADvoice</th><th>影响</th></tr><tr><td>声学时序</td><td>每个 5 秒片段保留约 250 个帧级表示，再跨片段注意力整合</td><td>每个片段先压缩成一个均值向量</td><td>停顿边界、局部韵律和发声变化在分类前已经丢失</td></tr><tr><td>编码器训练</td><td>mHuBERT 与 mGTE 进入端到端优化</td><td>mHuBERT 完全冻结，E5 只训练最后一层</td><td>预训练表示不能适配 PREPARE 的语言、任务与录音分布</td></tr><tr><td>认知状态利用</td><td>无状态层</td><td>状态分支平均权重 0.073，中位数仅 0.019</td><td>当前模型实际仍主要依赖文本/音频，认知层尚未成为稳定预测依据</td></tr><tr><td>类别边界</td><td>公开 checkpoint 更保守地预测 HC</td><td>提高 MCI 召回，但增加 HC→AD/MCI 和 AD→HC</td><td>AUROC 接近，固定阈值 F1/accuracy 仍明显偏低</td></tr><tr><td>样本规模</td><td>10 次训练并报告均值与区间</td><td>单种子且 MCI 测试样本只有 51</td><td>少量边界病例即可明显改变 F1</td></tr></table>
<h2>3. Agent 到底参与了什么</h2><div class="warn"><b>当前 PREPARE 分类指标不是 Agent 的诊断准确率。</b>概率由监督学习网络产生；Agent 读取概率、MetricEvidence、认知状态与片段证据后生成医学报告。因而“Agent 比 SpeechCARE 低”并不准确，真正低的是独立监督预测主干。若要检验 Agent 是否改善诊断，必须单列 Agent 可修改预测的实验，并使用锁定规则和独立测试集。</div>
<h2>4. 类别和语言层面的失分</h2><table><tr><th>语言</th><th>n</th><th>ADvoice accuracy</th><th>SpeechCARE accuracy</th><th>差值</th></tr>{subgroup_rows}</table><p>独立 ADvoice 对 MCI 的召回为 0.431，高于公开 SpeechCARE checkpoint 的 0.333；但 HC 召回从 0.934 降至 0.856，AD 召回从 0.553 降至 0.455。也就是说，它不是单纯“不会识别 MCI”，而是在追求 MCI 敏感性时破坏了 HC 和 AD 的边界。西班牙语准确率差距为 10.1 个百分点，是最清楚的语言失败模式。</p>
<h2>5. 架构应如何收敛</h2><div class="good"><b>正确方向：</b>先训练一个可复现的 SpeechCARE 级多语言声学—文本主干，再把 ADvoice 的 MetricEvidence、任务条件化认知状态和片段轨迹作为受限认知残差，而不是让状态分支替代深层表示。Agent 最后读取同一证据图，负责证据校验、反证检查和报告，不应在未验证条件下自由改写分类。</div>
<img src="assets/revision_progression.png" alt="revision progression"><p>图中三阶段共享同一测试队列，但训练协议不同，因此只能用于工程诊断，不能当作严格消融。</p>
<h2>6. PREPARE 回顾性结果</h2><img src="assets/protocol_aligned_metrics.png" alt="same cohort retrospective metrics"><p>论文均值来自十次训练；released checkpoint 是作者公开概率文件；扩展模型使用固定 0.20 认知权重。它说明认知概率具有互补性，不等于本项目已独立复现 SpeechCARE 主干。</p>
<img src="assets/confusion_comparison.png" alt="confusion matrices"><p>20 个病例在加入认知残差后改变类别：修正 9 个原错误病例，破坏 10 个原正确病例，另有 1 个仍然错误。MCI 正确数从 17 增至 20，但 HC 正确数从 214 降至 211、AD 从 73 降至 72，所以宏平均 F1 上升而 accuracy 下降 1/412。原先“主要来自 MCI 边界病例重新分配”的准确含义仅此，不是独立模型落后的根因。</p>
<h2>7. 结果可追溯性</h2><div class="{('good' if provenance_complete else 'bad')}"><b>当前正式预测元数据完整：</b>{str(provenance_complete).lower()}。现有目录中的预测文件与后来运行的 <code>window_stats</code> 验证结果发生过元数据错位，因此旧预测可以复算指标，但不能完整恢复其训练 checkpoint。本轮已修改运行器：验证实验隔离保存；正式预测必须记录输入哈希、配置、预测哈希和可恢复参数，扩展脚本默认拒绝无 provenance 的文件。</div>
<h2>8. 能说与不能说</h2><div class="good"><b>可以说：</b>在作者公开 SpeechCARE 主干输出上，增加固定上限的 ADvoice 认知证据残差后，同一 PREPARE 测试队列的回顾性指标高于论文十次均值，说明认知状态提供了互补预测信息。</div><div class="bad"><b>不能说：</b>ADvoice 已独立复现并全面超过 SpeechCARE。当前使用了作者公开 test 概率，且该 test 已被检查；确认性优效结论仍需重新训练可部署的强主干，并在未查看标签的新外部队列锁定验证。</div>
<h2>9. 下一项工程任务</h2><p>将公开概率替换为本项目自己训练的强主干：保留 mHuBERT 帧级时序，端到端或参数高效微调声学与文本编码器；仅用 validation 选择 checkpoint、认知残差上限和多类别操作点；完成 10 种子重复；最后在新的锁定外部队列做确认性比较。没有这一轮，不能把回顾性 0.882 写成独立优效结果。</p>
<p><a href="https://www.nature.com/articles/s41746-025-02026-x">SpeechCARE 论文</a> · <a href="https://github.com/SpeechCARE/SpeechCARE-NIA-Phase2">作者代码</a></p>
</main></body></html>"""
    (OUT / "speechcare_gap_resolution_report_zh.html").write_text(html, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ASSETS.mkdir(parents=True, exist_ok=True)
    result = json.loads(
        (ROOT / "artifacts" / "PREPARE_DrivenData" / "speechcare_cognitive_extension" / "result.json").read_text(encoding="utf-8")
    )
    style()
    metric_figure(result)
    progression_figure(result)
    confusion_figure()
    build_html(result)
    print(OUT / "speechcare_gap_resolution_report_zh.html")


if __name__ == "__main__":
    main()
