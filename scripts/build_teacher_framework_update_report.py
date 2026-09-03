#!/usr/bin/env python3
"""Build the standalone teacher-facing 8.27 framework update report.

The report reads only versioned aggregate CSV outputs. It does not retrain models,
alter predictions, or overwrite the full ten-dataset evaluation report.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime, timezone
from html import escape
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
LATEST = ROOT / "reports" / "latest"
ASSETS = LATEST / "assets"
HTML_OUT = LATEST / "teacher_framework_update_presentation_zh.html"
ORAL_OUT = LATEST / "teacher_framework_update_oral_presentation_zh.md"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: str | float) -> float:
    return float(value)


def fmt(value: float, digits: int = 3) -> str:
    return f"{value:.{digits}f}"


def dataset_label(dataset_id: str) -> str:
    return {
        "ADReSS_2020": "ADReSS 2020",
        "ADReSSo_2021_diagnosis": "ADReSSo 诊断",
        "ADReSSo_2021_progression": "ADReSSo 进展",
        "DementiaBank_Pitt": "Pitt",
        "DementiaNet_PublicFigures": "DementiaNet",
        "IAEAV": "IAEAV",
        "NCMMSC2021_AD": "NCMMSC 2021",
        "PREPARE_DrivenData": "PREPARE",
        "PROCESS_2": "PROCESS-2",
        "TAUKADIAL": "TAUKADIAL",
    }.get(dataset_id, dataset_id)


def collect() -> dict[str, object]:
    core = read_csv(LATEST / "all_dataset_three_condition_core_metrics.csv")
    layer_b = read_csv(LATEST / "all_dataset_layer_b_checks.csv")
    legacy = read_csv(LATEST / "current_b3_vs_legacy_c_metrics.csv")
    speechcare = read_csv(ASSETS / "speechcare_protocol_aligned_metrics.csv")

    by_dataset: dict[str, dict[str, dict[str, float]]] = defaultdict(dict)
    for row in core:
        by_dataset[row["dataset"]][row["condition"]] = {
            key: f(row[key]) for key in ("acc", "bal_acc", "f1", "auc", "micro_auc", "ece")
        }

    stronger_than_both = []
    clean_stronger = []
    downgraded = {"IAEAV", "DementiaBank_Pitt", "NCMMSC2021_AD"}
    for dataset, arms in by_dataset.items():
        if not {"B1", "B2", "Ours"}.issubset(arms):
            continue
        if arms["Ours"]["auc"] > max(arms["B1"]["auc"], arms["B2"]["auc"]):
            stronger_than_both.append(dataset)
            if dataset not in downgraded:
                clean_stronger.append(dataset)

    legacy_delta: dict[str, list[float]] = defaultdict(list)
    for row in legacy:
        legacy_delta[row["metric"]].append(f(row["delta"]))

    layer_b_values: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in layer_b:
        if row["value"]:
            layer_b_values[(row["condition"], row["check"])].append(f(row["value"]))

    def bmean(condition: str, check: str) -> float:
        values = layer_b_values[(condition, check)]
        return mean(values) if values else float("nan")

    sc = {(row["condition"], row["metric_key"]): f(row["value"]) for row in speechcare}

    return {
        "by_dataset": by_dataset,
        "dataset_count": len(by_dataset),
        "stronger_than_both": stronger_than_both,
        "clean_stronger": clean_stronger,
        "legacy_acc_delta": mean(legacy_delta["accuracy"]),
        "legacy_f1_delta": mean(legacy_delta["macro_f1"]),
        "legacy_auc_delta": mean(legacy_delta["macro_auroc_ovr"]),
        "ours_report": bmean("Ours", "clinical report rubric /25"),
        "b2_report": bmean("B2", "clinical report rubric /25"),
        "rollback": bmean("Ours", "cognitive rollback enforcement"),
        "agent_execution": bmean("Ours", "diagnostic Agent execution coverage"),
        "agent_validity": bmean("Ours", "diagnostic Agent evidence validity"),
        "agent_accept": bmean("Ours", "accepted bounded Agent correction rate"),
        "agent_auc_delta": bmean("Ours", "Agent-on macro AUROC delta"),
        "safe_fallback": bmean("Ours", "clinical report safe-fallback replacement rate"),
        "span_faithfulness": bmean("Ours", "evidence-span faithfulness"),
        "metric_complete": bmean("Ours", "MetricEvidence completeness"),
        "state_complete": bmean("Ours", "StateCard completeness"),
        "permission": bmean("Ours", "report-permission audit"),
        "qc_separation": bmean("Ours", "quality-evidence separation audit"),
        "task_state_gain": bmean("Ours", "task-specific state ablation"),
        "prepare_ours_auc": sc[("Ours", "micro_auroc_ovr")],
        "prepare_sc_auc": sc[("SpeechCARE", "micro_auroc_ovr")],
        "prepare_ours_f1": sc[("Ours", "micro_f1")],
        "prepare_sc_f1": sc[("SpeechCARE", "reported_f1")],
    }


def result_rows(data: dict[str, object]) -> str:
    by_dataset = data["by_dataset"]
    assert isinstance(by_dataset, dict)
    rows = []
    for dataset, arms in by_dataset.items():
        best_baseline = max(arms["B1"]["auc"], arms["B2"]["auc"])
        delta = arms["Ours"]["auc"] - best_baseline
        cls = "gain" if delta > 0 else "loss" if delta < 0 else "neutral"
        caveat = {
            "IAEAV": "采访者/批次共线，降级为偏倚审计",
            "DementiaBank_Pitt": "QC-only 负控过高，不能作为临床有效性证据",
            "NCMMSC2021_AD": "QC-only 负控过高，仍需设备/来源去混杂",
            "ADReSSo_2021_progression": "纵向终点与横断面状态融合不匹配",
            "TAUKADIAL": "语言/任务迁移不足",
            "DementiaNet_PublicFigures": "仅 6 个测试病例，估计极不稳定",
        }.get(dataset, "当前未触发主要降级条件")
        rows.append(
            "<tr>"
            f"<td>{escape(dataset_label(dataset))}</td>"
            f"<td>{fmt(arms['B1']['auc'])}</td>"
            f"<td>{fmt(arms['B2']['auc'])}</td>"
            f"<td><b>{fmt(arms['Ours']['auc'])}</b></td>"
            f"<td class='{cls}'>{delta:+.3f}</td>"
            f"<td>{escape(caveat)}</td>"
            "</tr>"
        )
    return "\n".join(rows)


def build_html(data: dict[str, object]) -> str:
    generated = datetime.now(timezone.utc).astimezone().strftime("%Y-%m-%d %H:%M %Z")
    report_gain = f(data["ours_report"]) - f(data["b2_report"])
    clean = ", ".join(dataset_label(x) for x in data["clean_stronger"])
    result_table = result_rows(data)
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADvoice 8.27 方法更新与评估映射｜老师汇报版</title>
<style>
:root{{--ink:#17222c;--muted:#61707c;--line:#d9e0e4;--paper:#fff;--soft:#f5f7f7;--teal:#248f87;--teal2:#dff1ee;--coral:#df746b;--coral2:#fbe9e7;--blue:#527fb7;--blue2:#e8eff8;--gold:#c48a2b;--gold2:#fbf1dc;--purple:#7561b5;--purple2:#eeeafd;}}
*{{box-sizing:border-box}} body{{margin:0;background:#edf1f1;color:var(--ink);font:16px/1.68 -apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;letter-spacing:0}}
main{{max-width:1280px;margin:auto;background:var(--paper);min-height:100vh}} header{{padding:64px 72px 48px;border-bottom:1px solid var(--line)}}
.eyebrow{{font-size:13px;font-weight:800;color:var(--teal);text-transform:uppercase}} h1{{font-size:42px;line-height:1.2;margin:10px 0 18px;letter-spacing:0}} h2{{font-size:28px;margin:0 0 18px}} h3{{font-size:20px;margin:0 0 10px}} p{{margin:8px 0 14px}} .lead{{font-size:20px;max-width:980px;color:#35434e}}
.meta{{font-size:13px;color:var(--muted)}} section{{padding:48px 72px;border-bottom:1px solid var(--line)}}
.kpis{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:1px;background:var(--line);border:1px solid var(--line);margin-top:28px}} .kpi{{background:#fff;padding:22px}} .kpi b{{display:block;font-size:30px;line-height:1.1;color:var(--teal)}} .kpi span{{font-size:13px;color:var(--muted)}}
.notice{{border-left:5px solid var(--gold);background:var(--gold2);padding:18px 22px;margin:22px 0}} .danger{{border-left-color:var(--coral);background:var(--coral2)}}
table{{width:100%;border-collapse:collapse;margin:18px 0;font-size:14px}} th{{background:#edf3f3;text-align:left;font-weight:800}} th,td{{border:1px solid var(--line);padding:12px 13px;vertical-align:top}} td.gain{{color:#08776d;font-weight:800}} td.loss{{color:#b53f38;font-weight:800}} td.neutral{{color:var(--muted);font-weight:800}}
    .matrix-wrap{{overflow-x:auto;border:1px solid var(--line);margin:20px 0}} .matrix{{min-width:1120px;margin:0;font-size:13px}} .matrix th{{text-align:center;line-height:1.35}} .matrix th:first-child,.matrix td:first-child{{position:sticky;left:0;z-index:2;background:#fff;min-width:250px;text-align:left}} .matrix thead th:first-child{{background:#edf3f3;z-index:3}} .matrix td:not(:first-child){{text-align:center;vertical-align:middle;min-width:108px}} .mark{{display:inline-grid;place-items:center;width:28px;height:28px;border:1px solid var(--line);font-size:17px;font-weight:900}} .yes{{background:var(--teal2);color:#08756c;border-color:#8dc9c3}} .partial{{background:var(--gold2);color:#8a5e15;border-color:#dfbd7b}} .no{{background:#f2f4f4;color:#89949c}} .legend{{font-size:13px;color:var(--muted)}}
.pipeline{{display:grid;grid-template-columns:repeat(6,minmax(0,1fr));gap:10px;align-items:stretch;margin:24px 0}} .step{{position:relative;border:1px solid var(--line);padding:16px 14px;background:#fff;min-height:174px}} .step:not(:last-child):after{{content:"→";position:absolute;right:-16px;top:68px;z-index:2;font-size:22px;font-weight:800;color:#71808a}} .step .n{{font-size:12px;font-weight:800;color:var(--muted)}} .step strong{{display:block;margin:7px 0 8px;font-size:17px}} .step p{{font-size:13px;color:#52616c;line-height:1.55}}
.cot{{display:grid;grid-template-columns:1fr 48px 1fr 48px 1fr;align-items:center;margin:20px 0}} .cotbox{{padding:22px;border:1px solid var(--line);background:var(--purple2);min-height:150px}} .arrow{{text-align:center;font-size:24px;color:var(--purple)}}
    .method-grid{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:16px;margin:22px 0}} .method-card{{border:1px solid var(--line);border-top:5px solid var(--blue);padding:20px;background:#fff}} .method-card:nth-child(2){{border-top-color:var(--purple)}} .method-card:nth-child(3){{border-top-color:var(--teal)}} .method-card h3{{font-size:19px}} .method-card p{{font-size:14px;color:#465660}} .method-card .route{{display:block;background:var(--soft);padding:10px 12px;font-size:13px;font-weight:700;line-height:1.55;margin:12px 0}}
    .update-grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:22px 0}} .update{{border:1px solid var(--line);padding:20px;background:#fff}} .update .u{{font-size:12px;font-weight:900;color:var(--teal)}} .update strong{{display:block;font-size:18px;margin:4px 0 8px}} .update p{{font-size:14px;color:#465660}} .example{{border-left:4px solid var(--gold);background:var(--gold2);padding:10px 12px;margin-top:12px;font-size:13px;line-height:1.6}} .benefit{{color:#075f58;font-weight:800}}
    .map-wrap{{overflow-x:auto;margin:20px 0}} .map-table{{min-width:1450px;margin:0;font-size:13px}} .map-table th{{text-align:left}} .map-table td:first-child{{min-width:170px}} .map-table td:nth-child(2){{min-width:190px}} .map-table td:nth-child(3){{min-width:250px}} .map-table td:nth-child(4){{min-width:210px}} .map-table td:nth-child(5){{min-width:230px}} .map-table td:nth-child(6){{min-width:240px}} .tag{{display:inline-block;padding:2px 8px;border-radius:3px;font-size:12px;font-weight:800}} .supported{{background:var(--teal2);color:#086d65}} .boundary{{background:var(--gold2);color:#805614}} .unproven{{background:var(--coral2);color:#9b3934}}
    .figure{{display:block;width:100%;height:auto;margin:18px 0 8px;border:1px solid #edf0f1}} .caption{{font-size:13px;color:var(--muted);margin-bottom:28px}} .twocol{{display:grid;grid-template-columns:1fr 1fr;gap:28px}}
    .figure-block{{padding:22px 0 32px;border-top:1px solid var(--line)}} .figure-block:first-of-type{{border-top:0}} .figure-kicker{{font-size:12px;font-weight:800;color:var(--teal);text-transform:uppercase}} .figure-block h3{{margin-top:5px}} .reading{{background:var(--soft);padding:14px 18px;margin:10px 0 0}} .reading b{{color:#263a45}} .mechanism-table td:nth-child(1){{width:18%}} .mechanism-table td:nth-child(2){{width:24%}} .mechanism-table td:nth-child(3){{width:25%}} .mechanism-table td:nth-child(4){{width:33%}}
    .advantages{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px;margin:22px 0 30px}} .advantage{{border:1px solid var(--line);border-top:5px solid var(--teal);padding:20px;background:#fff}} .advantage:nth-child(2){{border-top-color:var(--purple)}} .advantage:nth-child(3){{border-top-color:var(--gold)}} .advantage:nth-child(4){{border-top-color:var(--blue)}} .advantage .index{{font-size:12px;font-weight:900;color:var(--muted)}} .advantage strong{{display:block;font-size:19px;margin:4px 0 8px}} .advantage .number{{font-size:28px;font-weight:900;color:var(--teal);line-height:1.2}} .advantage p{{font-size:14px;color:#4b5a64}}
.claim{{display:grid;grid-template-columns:180px 1fr;gap:18px;border-top:1px solid var(--line);padding:18px 0}} .claim b{{font-size:18px}} ul{{margin:8px 0 0;padding-left:22px}} footer{{padding:28px 72px;color:var(--muted);font-size:13px}} a{{color:#166f9c}} code{{background:#eef1f2;padding:2px 5px}}
    @media(max-width:900px){{header,section,footer{{padding-left:24px;padding-right:24px}} .kpis,.twocol,.advantages,.method-grid,.update-grid{{grid-template-columns:1fr}} .pipeline{{grid-template-columns:1fr}} .step:not(:last-child):after{{content:"↓";right:50%;top:auto;bottom:-24px}} .cot{{grid-template-columns:1fr}} .arrow{{transform:rotate(90deg)}} h1{{font-size:32px}}}}
@media print{{body{{background:#fff}} section{{break-inside:avoid}} a{{color:inherit;text-decoration:none}}}}
</style>
</head>
<body><main>
<header>
  <div class="eyebrow">ADvoice 8.27 · Teacher briefing</div>
  <h1>从“多模态分类”升级为“可维护的认知状态与证据链”</h1>
  <p class="lead">本轮更新不是再增加一组特征，也不是让大模型自由解释。核心变化是把患者在不同语音任务中表现出的认知状态变成一个可更新、可反驳、可回退的显式工作区，并让预测、Agent 审查和医生报告共享同一条证据链。</p>
  <p class="meta">生成时间：{generated}｜本页是老师汇报版；完整十数据集报告、逐数据集结果和原始 CSV 均保持不变。</p>
  <div class="kpis">
    <div class="kpi"><b>{int(data['dataset_count'])}</b><span>个独立数据任务完整保留</span></div>
    <div class="kpi"><b>+{fmt(f(data['legacy_auc_delta']))}</b><span>新版相对历史 C 的平均宏 AUROC 变化</span></div>
    <div class="kpi"><b>+{fmt(report_gain,2)}</b><span>结构化医生报告相对直接 Agent 的 /25 提升</span></div>
    <div class="kpi"><b>{fmt(f(data['agent_auc_delta']))}</b><span>Agent-on 宏 AUROC 增量：尚未证明预测增益</span></div>
  </div>
</header>

<section>
  <h2>1. 相关工作做到什么，本项目增加什么</h2>
  <p>这里比较的是两项直接相关工作与 ADvoice，而不是把所有提供过方法启发的论文并列。<b>SpeechCARE</b>代表语音认知筛查中的多语言、多任务表征与动态融合；<b>MEDRF</b>代表认知障碍决策支持中“监督分类器 + 检索增强大模型”的证据校正；<b>ADvoice</b>进一步把语音指标、认知状态、任务片段、Agent 审查和医生报告锁在同一条可回退证据链上。</p>
  <div class="method-grid">
    <div class="method-card"><h3><a href="https://www.nature.com/articles/s41746-025-02026-x">SpeechCARE</a></h3><p><b>研究目标：</b>从英语、西班牙语和普通话的多种短语音任务中区分健康、MCI 与 ADRD。</p><span class="route">低通与切片 / Whisper 转录 → mHuBERT 声学表示 + mGTE 文本表示 + 人口学表示 → 定制自注意编码器 → 自适应门控融合 → 分类</span><p><b>大模型边界：</b>大模型用于音频异常过滤和任务类型识别；核心诊断不是大模型，也没有诊断 Agent。最终预测由声学/文本编码器和门控网络完成。</p><p><b>仍缺什么：</b>门控权重说明模型依赖哪个模态，但不能回答风险来自哪个临床认知状态、哪个原始指标和哪段语音，也没有候选判断的医学回退。</p></div>
    <div class="method-card"><h3><a href="https://www.nature.com/articles/s41746-026-03048-9">MEDRF</a></h3><p><b>研究目标：</b>利用常规临床资料和结构 MRI 识别认知障碍阶段与病因，并处理资料缺失和诊断线索冲突。</p><span class="route">临床资料 + 结构 MRI → 多模态层次级联分类器 mHC → 检索相似病例与文本证据 → RAG-LLM 多步校正 → 诊断与报告</span><p><b>大模型边界：</b>RAG-LLM 真正参与决策校正，不只是写报告。论文在异质外部队列中报告，加入 RAG-LLM 后总体准确率由 0.706±0.038 提升至 0.753±0.032。</p><p><b>仍缺什么：</b>它的证据单位是临床资料、MRI、相似病例和文本知识，不处理语音任务中的角色、停顿、词汇检索、片段轨迹及录音质量混杂。</p></div>
    <div class="method-card"><h3>ADvoice 8.27</h3><p><b>研究目标：</b>从异质语音任务中估计可审查的认知状态，并把筛查结论追溯到任务、指标和原始片段。</p><span class="route">任务/角色路由 → 指标证据 → 任务条件化认知状态 → 两个监督模块 → 证据约束诊断 Agent → 确定性验证与回退 → 同轨医生报告</span><p><b>Agent 边界：</b>Agent 在决策锁定前审查支持证据、反证和替代解释；它只能提出有界修正，验证失败即恢复监督先验。</p><p><b>相对优势：</b>继承 SpeechCARE 的任务条件化思想和 MEDRF 的证据校正思想，但把决策单位细化为“指标—状态—片段”，并增加 QC 权限、反证、违规码和可回退轨迹。</p></div>
  </div>
  <div class="matrix-wrap"><table class="matrix">
    <thead><tr><th>论文 / 系统</th><th>任务与语言条件化</th><th>学习式多模态融合</th><th>大模型参与诊断决策</th><th>临床命名认知状态</th><th>指标—状态—片段追溯</th><th>支持证据与反证</th><th>确定性违规回退</th><th>医生报告证据权限</th></tr></thead>
    <tbody>
      <tr><td><b>SpeechCARE</b><br>语音多任务预测</td><td><span class="mark yes">✓</span></td><td><span class="mark yes">✓</span></td><td><span class="mark no">—</span></td><td><span class="mark no">—</span></td><td><span class="mark no">—</span></td><td><span class="mark no">—</span></td><td><span class="mark no">—</span></td><td><span class="mark no">—</span></td></tr>
      <tr><td><b>MEDRF</b><br>多模态证据决策支持</td><td><span class="mark no">—</span></td><td><span class="mark yes">✓</span></td><td><span class="mark yes">✓</span></td><td><span class="mark partial">◐</span></td><td><span class="mark partial">◐</span></td><td><span class="mark partial">◐</span></td><td><span class="mark no">—</span></td><td><span class="mark partial">◐</span></td></tr>
      <tr><td><b>ADvoice 8.27</b><br>证据约束语音认知筛查</td><td><span class="mark yes">✓</span></td><td><span class="mark yes">✓</span></td><td><span class="mark partial">◐</span></td><td><span class="mark yes">✓</span></td><td><span class="mark yes">✓</span></td><td><span class="mark yes">✓</span></td><td><span class="mark yes">✓</span></td><td><span class="mark yes">✓</span></td></tr>
    </tbody>
  </table></div>
  <p class="legend"><span class="mark yes">✓</span> 已实现或论文明确报告　<span class="mark partial">◐</span> 部分实现、证据粒度不同或尚未证明有效　<span class="mark no">—</span> 未报告、非研究目标。ADvoice 的“大模型参与诊断决策”标为部分实现，因为调用、验证和回退链已运行，但目前所有候选修正均被拒绝，尚未产生 Agent-on 预测增益。</p>
  <div class="notice"><b>三者不是同一条技术路线：</b>SpeechCARE 的强项是端到端语音表征和动态模态权重；MEDRF 的强项是层次分类器与 RAG-LLM 在缺失、模糊病例中的联合决策；ADvoice 的研究增量是把语音特有的可解释证据组织成可维护、可反驳、可回退的认知状态，再让监督模型、Agent 和医生报告共享这一对象。</div>
  <div class="notice"><b>学术边界：</b>本项目在 PREPARE 的同终点描述性比较中仍低于 SpeechCARE，因此优势目前成立在“证据治理与临床可审查性”，不是“已经全面超过 SpeechCARE 的预测性能”。</div>
</section>

<section>
  <h2>2. 8.27 到底更新了什么</h2>
  <div class="twocol">
    <div><h3>本轮保留的既有基础</h3><p><code>角色/任务路由 → MetricEvidence → StateCard → 片段轨迹 → 可靠性感知融合</code></p><p><b>MetricEvidence、StateCard 和片段追溯不是 8.27 新发明。</b>它们是前一版本已经建立的证据表示层。本轮只对它们的字段契约、任务索引和调用方式做工程固化。</p></div>
    <div><h3>8.27 新增的决策链</h3><p><code>监督先验 → 结构化认知工作区 → 单一诊断 Agent → 确定性验证/回退 → 校准决策锁定 → 同轨报告</code></p><p>真正变化是证据不再只供报告引用，而是在决策锁定前成为 Agent 可调用、验证器可否决、监督模块可校准的同一个对象。</p></div>
  </div>
  <div class="pipeline">
    <div class="step"><span class="n">更新 01</span><strong>三类权重解耦</strong><p>可观察性、可靠度、预测贡献分别计算。</p></div>
    <div class="step"><span class="n">更新 02</span><strong>任务条件化工作区</strong><p>总体状态与任务状态同时保留。</p></div>
    <div class="step"><span class="n">更新 03</span><strong>两个监督模块</strong><p>基础风险与修正资格分开学习。</p></div>
    <div class="step"><span class="n">更新 04</span><strong>诊断 Agent 前移</strong><p>在决策锁定前审查证据和反证。</p></div>
    <div class="step"><span class="n">更新 05</span><strong>确定性认知回退</strong><p>违规候选定位后重做或恢复先验。</p></div>
    <div class="step"><span class="n">更新 06</span><strong>同轨报告</strong><p>预测和报告读取同一锁定轨迹。</p></div>
  </div>
  <div class="update-grid">
    <div class="update"><span class="u">更新 01</span><strong>三类权重解耦</strong><p><b>解决：</b>旧系统把“能不能测到、测得准不准、对分类有没有用”混成一个权重。新版先用任务规则决定可观察性，再按覆盖、角色、对齐、稳定性和 QC 计算可靠度，训练折只学习剩余预测贡献。</p><div class="example"><b>例子：</b>一段录音的 RMS 响度很低，但同时存在远场麦克风和背景噪声。该指标可计算，却可靠度低，只能进入 QC，不能因为与标签相关就写成疾病证据。<br><span class="benefit">好处：把预测相关性和医学解释资格分开。</span></div></div>
    <div class="update"><span class="u">更新 02</span><strong>任务条件化认知工作区</strong><p><b>解决：</b>旧系统把 Cookie 图片描述、词语流畅性和故事回忆先平均，可能抹掉“只在某项认知任务中出现”的异常。新版同时维护病例总体状态和每项任务的状态、指标与片段。</p><div class="example"><b>例子：</b>患者句子朗读基本正常，但故事回忆遗漏关键事件、出现多次检索停顿。新版保留“回忆任务受损、朗读任务相对保留”，不会把两者平均成轻度异常。<br><span class="benefit">好处：区分局部任务缺陷与跨任务持续异常。</span></div></div>
    <div class="update"><span class="u">更新 03</span><strong>两个监督模块</strong><p><b>解决：</b>基础分类、Agent 修正资格和概率校准原来容易混在一起。模块 A 学基础风险和状态贡献；模块 B 只用训练折外轨迹学习是否允许 Agent 修正、最大幅度和校准映射。</p><div class="example"><b>例子：</b>模块 A 给出 0.68 的筛查风险，但当前病例 QC 较差、证据覆盖不足。模块 B 将可修正幅度限制为 0，Agent 只能解释不确定性，不能把概率推到 0.90。<br><span class="benefit">好处：保留学习能力，同时阻止自由语言绕过统计边界。</span></div></div>
    <div class="update"><span class="u">更新 04</span><strong>单一诊断 Agent 前移</strong><p><b>解决：</b>旧 Agent 在分类完成后写报告，理由不能进入判断。新版 Agent 在锁定前读取医学 Skill、监督先验和结构化证据包，必须同时提交支持、反证、替代解释和证据 ID。</p><div class="example"><b>例子：</b>词汇提取状态异常，但语义连贯性仍在正常范围。Agent 必须同时引用检索停顿片段和连贯表达片段，才能提出“存在词汇检索困难，但尚无广泛语义崩解”的候选判断。<br><span class="benefit">好处：Agent 参与诊断，但不能只挑支持结论的证据。</span></div></div>
    <div class="update"><span class="u">更新 05</span><strong>确定性认知回退</strong><p><b>解决：</b>Agent 可能引用无效片段、把设备/噪声当疾病机制、遗漏反证或无依据分期。验证器按固定医学优先级检查，并返回违规码和最早回退点。</p><div class="example"><b>例子：</b>Agent 把“音量偏低”直接解释为 AD。系统触发“QC 敏感证据越权”，删除该证据并重做；再次失败则恢复模块 A 的校准先验。<br><span class="benefit">好处：语言表达再流畅，也不能越过证据权限。</span></div></div>
    <div class="update"><span class="u">更新 06</span><strong>决策轨迹锁定与同轨报告</strong><p><b>解决：</b>旧报告可能在预测结束后重新组织甚至重新编理由。新版先锁定风险、状态贡献、支持、反证、限制和片段 ID，报告生成器只能读取该只读对象。</p><div class="example"><b>例子：</b>报告写“输出效率下降”时，医生可沿 ID 回到每分钟有效内容、发声时间占比及对应时间片段；若这些对象不在锁定轨迹中，该句不能生成。<br><span class="benefit">好处：预测理由与报告理由不再是两套叙事。</span></div></div>
  </div>
</section>

<section>
  <h2>3. Cognition-of-Thought 只作为更新机制的思想来源</h2>
  <p><b>它不是本项目的直接相关医疗工作，也不进入前面的三方方法比较。</b>本项目仅借鉴“维护外显状态—按原则监测—定位最早违规步骤—带约束回退”的框架思想。系统不读取或展示隐藏思维链，而是把这一思想改造成结构化、可审计的病例认知工作区。</p>
  <div class="cot">
    <div class="cotbox"><h3>生成器 → 诊断 Agent</h3><p>不直接给最终标签，而是提交结构化候选：当前状态假设、引用的证据 ID、反证、替代解释和有限概率修正。</p></div><div class="arrow">→</div>
    <div class="cotbox"><h3>认知监测器 → 确定性验证器</h3><p>依次检查泄漏、QC 冒充疾病、任务不可观察、无效引用、遗漏反证、无依据分期和修正越界。</p></div><div class="arrow">→</div>
    <div class="cotbox"><h3>回退 → 医学安全回退</h3><p>返回明确违规码与需撤销对象；重做仍失败则拒绝 Agent 修正，恢复模块 A 的校准监督先验。</p></div>
  </div>
  <div class="notice"><b>例子：</b>如果 Agent 把 RMS 响度降低作为 AD 证据，验证器触发“QC/设备敏感证据不得直接解释疾病”的违规码，删除该引用并要求仅用允许报告的停顿、词汇检索和任务内容证据重做。若再次违规，最终概率保持模块 A 的监督先验。</div>
  <table class="mechanism-table"><thead><tr><th>Cognition 机制</th><th>理论上应改变的系统行为</th><th>当前对应评估</th><th>本轮观察与正确解释</th></tr></thead><tbody>
    <tr><td><b>C1 显式认知工作区</b></td><td>Agent 的判断应绑定状态、支持证据、反证和来源片段，而不是只读取自由文本摘要。</td><td>证据对象完整率、状态卡完整率、分支贡献、片段忠实度、报告权限。</td><td>结构检查均达到 1.000。但 MetricEvidence 与 StateCard 原本已存在，所以这是“新版成功调用既有证据层”，不是 Cognition 机制独立造成的新增提升。</td></tr>
    <tr><td><b>C2 违规监测与定位</b></td><td>无效引用、QC 冒充疾病、遗漏反证和越界修正应在最终决策前被识别。</td><td>Agent 候选证据有效率、违规码覆盖、认知回退执行率。</td><td>候选有效率仅 {fmt(f(data['agent_validity']))}，回退执行率 {fmt(f(data['rollback']))}。这说明监测器确实发现并拦截大量不合格候选，也同时暴露 Agent 原始候选质量不足。</td></tr>
    <tr><td><b>C3 回退与有界采纳</b></td><td>Agent 不可靠时应保持监督先验，避免因自由修正造成性能下降或产生不可追溯报告。</td><td>采纳率、Agent-on AUROC 增量、安全模板替换率、报告结构评分。</td><td>采纳率 {fmt(f(data['agent_accept']))}、Agent-on AUROC 增量 {fmt(f(data['agent_auc_delta']))}，说明当前系统采取保守回退；报告结构提高 {report_gain:+.2f}/25。安全性与报告改善有证据，预测增益没有证据。</td></tr>
  </tbody></table>
  <p class="caption">这一设计直接验证的是“错误候选能否被发现和阻止”，不是“Agent 是否因此更准确”。当前回退执行率为 {fmt(f(data['rollback']))}，但 Agent-on 宏 AUROC 增量仍为 {fmt(f(data['agent_auc_delta']))}。</p>
</section>

<section>
  <h2>4. 本轮变化带来的优势与历史结果</h2>
  <p>先看整套系统更新后实际发生了什么，再看各机制分别对应什么评估。下面四点是当前结果能够支持的优势，其中性能变化属于整套升级的前后比较，安全与追溯则有各自直接检查。</p>
  <div class="advantages">
    <div class="advantage"><span class="index">优势 01</span><strong>整体预测表现较历史系统提高</strong><div class="number">AUROC {f(data['legacy_auc_delta']):+.3f}</div><p>十个任务平均准确率 {f(data['legacy_acc_delta']):+.3f}、宏 F1 {f(data['legacy_f1_delta']):+.3f}、宏 AUROC {f(data['legacy_auc_delta']):+.3f}。这是当前 B3 相对历史条件 C 的锁定结果。</p></div>
    <div class="advantage"><span class="index">优势 02</span><strong>报告从自由生成变成同轨证据报告</strong><div class="number">{report_gain:+.2f} / 25</div><p>自动结构评分由 {fmt(f(data['b2_report']),2)} 提升至 {fmt(f(data['ours_report']),2)}；报告权限与证据片段忠实度均为 1.000。</p></div>
    <div class="advantage"><span class="index">优势 03</span><strong>不合格 Agent 判断能够被拦截</strong><div class="number">回退 {fmt(f(data['rollback']))}</div><p>Agent 候选证据有效率只有 {fmt(f(data['agent_validity']))}，但违规候选未进入有界概率修正。优势是防止错误扩散，不是 Agent 已提高准确率。</p></div>
    <div class="advantage"><span class="index">优势 04</span><strong>同一框架覆盖多任务并保留审计边界</strong><div class="number">{int(data['dataset_count'])} 个任务</div><p>所有任务均完成 B1、B2、B3 同病例比较；系统同时标出 IAEAV、Pitt、NCMMSC 等混杂风险，不用高分掩盖数据问题。</p></div>
  </div>
  <h3>历史系统变化总览</h3>
  <img class="figure" src="assets/figure_B5_current_vs_legacy_c.png" alt="Current B3 versus legacy condition C">
  <p class="caption">横轴为当前 B3 减去历史条件 C。正值表示新版在同一终点上提高，负值表示下降。图中变化来自整套 8.27 更新，不能拆分归因给某一个机制。</p>
  <h3>综合框架—评估对应表</h3>
  <p><b>MetricEvidence 和 StateCard 是既有基础，不作为 8.27 新增贡献。</b>下表说明每项本轮更新的工程优势、直接观测和当前证据边界。</p>
  <div class="map-wrap"><table class="map-table">
    <thead><tr><th>8.27 新增机制</th><th>此前的具体缺口</th><th>本轮具体变化</th><th>带来的工程优势</th><th>直接评估与观察值</th><th>证据边界</th></tr></thead>
    <tbody>
      <tr><td><b>U1 三类权重解耦</b><br>可观察性、可靠度、预测贡献</td><td>“任务能不能测到”和“模型是否应该依赖”混成一个权重，QC 敏感指标可能取得疾病权重。</td><td>规则先生成可观察性掩码；可靠度由覆盖、角色、对齐、稳定性、参考支持与 QC 惩罚组成；训练折只学习剩余预测贡献。</td><td>不可测状态不会进入当前任务；低可靠证据自动降权；模型权重不再等同于医学解释权限。</td><td>质量证据分离审计 {fmt(f(data['qc_separation']))}，报告权限审计 {fmt(f(data['permission']))}。但 Pitt 与 NCMMSC 的 QC-only 负控仍过高。</td><td><span class="tag boundary">边界实现得到直接支持</span><br>不能说采集捷径已经消除。</td></tr>
      <tr><td><b>U2 任务条件化认知工作区</b></td><td>多任务数据在进入状态融合前被平均，Cookie 描述、流畅性与回忆任务差异丢失。</td><td>同时保留病例总体状态和折内校准的任务状态；每个任务状态绑定同任务指标与片段。</td><td>可以区分“跨任务持续异常”和“只在单一任务出现的异常”，同时把判断回到对应任务片段。</td><td>任务指标与片段追溯在适用数据上通过；任务特异状态消融均值为 {f(data['task_state_gain']):+.3f}。</td><td><span class="tag unproven">可追溯性成立，预测收益未成立</span><br>均值略为负，不能宣称该模块提高 AUROC。</td></tr>
      <tr><td><b>U3 两个监督模块</b></td><td>基础风险、Agent 修正资格和概率校准混在一个模型里，容易发生训练/测试泄漏和无边界修正。</td><td>模块 A 在外层训练折生成可分解先验；模块 B 只用内层折外轨迹学习修正资格、最大幅度和校准，测试集只使用冻结参数。</td><td>稳定预测与 Agent 修正分工清楚；即使 Agent 不可靠，系统仍有冻结先验和校准边界。</td><td>整套新版相对历史 C：准确率 {f(data['legacy_acc_delta']):+.3f}，宏 F1 {f(data['legacy_f1_delta']):+.3f}，宏 AUROC {f(data['legacy_auc_delta']):+.3f}。</td><td><span class="tag boundary">整套升级相关提升</span><br>没有单独 U3 消融，不能把全部提升归因于两模块。</td></tr>
      <tr><td><b>U4 单一诊断 Agent 前移</b></td><td>旧 Agent 主要在预测后写报告，证据理由不能影响候选判断。</td><td>Agent 在决策锁定前读取 Skill、监督先验和结构化证据包，输出引用 ID、反证、替代解释及有界修正。</td><td>Agent 的临床审查发生在决策完成前；每项建议都必须绑定证据与反证，而不是事后补写理由。</td><td>执行覆盖率 {fmt(f(data['agent_execution']))}；候选证据有效率 {fmt(f(data['agent_validity']))}；采纳率 {fmt(f(data['agent_accept']))}；Agent-on 宏 AUROC 增量 {fmt(f(data['agent_auc_delta']))}。</td><td><span class="tag unproven">通路已运行，预测价值未证</span><br>当前候选质量不足，不能说 Agent 提高分类性能。</td></tr>
      <tr><td><b>U5 确定性认知回退</b></td><td>Agent 可能把 QC 当病因、引用无效片段、遗漏反证或越过概率边界。</td><td>按固定医学优先级验证，记录违规码和回退点；重做失败则拒绝修正并恢复监督先验。</td><td>错误不会因为语言流畅而进入最终概率；失败位置可定位，系统可以恢复到最近一个有效决策状态。</td><td>回退执行率 {fmt(f(data['rollback']))}；无效候选未进入最终修正；安全模板替换率均值 {fmt(f(data['safe_fallback']))}。</td><td><span class="tag supported">安全拦截得到直接支持</span><br>它证明错误被拦截，不证明 Agent 本身更准确。</td></tr>
      <tr><td><b>U6 决策轨迹锁定与同轨报告</b></td><td>报告可能在预测完成后重新编理由，无法证明正文证据与最终判断一致。</td><td>先锁定概率、状态贡献、证据 ID、反证和限制；报告生成器只能读取该只读对象。</td><td>医生报告与最终判断共享同一证据来源；任何结论都可回到状态、指标和片段。</td><td>片段忠实度 {fmt(f(data['span_faithfulness']))}；结构化报告由 {fmt(f(data['b2_report']),2)}/25 提升至 {fmt(f(data['ours_report']),2)}/25，增量 {report_gain:+.2f}。</td><td><span class="tag supported">自动结构审计支持</span><br>尚无真实医生读者研究，不能称为临床效用验证。</td></tr>
    </tbody>
  </table></div>
  <div class="notice"><b>迁移连续性检查：</b>既有 MetricEvidence 完整率 {fmt(f(data['metric_complete']))}、StateCard 完整率 {fmt(f(data['state_complete']))}、证据片段忠实度 {fmt(f(data['span_faithfulness']))}。这些数值说明迁移没有破坏旧证据层，不是本轮新增贡献。</div>
</section>

<section>
  <h2>5. Layer A：医学预测与筛查评估</h2>
  <p>Layer A 回答“模型作为认知筛查研究是否具备基本医学预测有效性”。它与框架可解释性分开报告，包含区分能力、固定阈值分类、高敏感度操作点、概率校准、错误结构和捷径负控。</p>
  <div class="figure-block"><div class="figure-kicker">Layer A overview</div><h3>A. 医学评估总图</h3><img class="figure" src="assets/figure_A_layer_medical_standards_summary.png" alt="Layer A medical standards summary"><div class="reading"><b>怎么看：</b>总图同时呈现 AUROC 与 95% CI、高敏感度筛查点、B3 相对 B1/B2 的增量、固定阈值指标、负控差距和三条件覆盖。它防止只用一张 AUROC 图宣称临床可用。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer A-1</div><h3>区分能力与固定阈值分类</h3><img class="figure" src="assets/figure_4_layer_a_performance.png" alt="Discrimination and classification metrics"><div class="reading"><b>怎么看：</b>AUROC、Accuracy、Macro F1 和 AUPRC 越高越好。该图直接比较 B1、B2、B3，但不能单独回答校准和筛查操作点。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer A-1b</div><h3>同一测试病例上的配对 AUROC 差值</h3><img class="figure" src="assets/figure_A1b_paired_auroc_differences.png" alt="Paired AUROC differences"><div class="reading"><b>怎么看：</b>点在零右侧表示 B3 数值更高；95% CI 跨零时不能声明统计学优于基线。这比只比较两个独立 AUROC 数字更严格。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer A-2</div><h3>高敏感度筛查操作点</h3><img class="figure" src="assets/figure_9_screening_operating_points.png" alt="Screening operating points"><div class="reading"><b>怎么看：</b>固定 sensitivity ≥ 0.85 后查看 specificity、PPV 与 NPV，回答实际筛查中“漏多少、误报多少”。非筛查或多分类任务不强行填充值。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer A-3</div><h3>概率安全与校准</h3><img class="figure" src="assets/figure_5_layer_a_safety.png" alt="Calibration and probability safety"><div class="reading"><b>怎么看：</b>ECE 与 Brier 越低越好，MCC 越高越好。认知回退理论上应防止不可靠 Agent 扰动校准；当前 Agent 修正未被采纳，因此校准结果主要来自监督模块，而不是 Agent 增益。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer A-4</div><h3>错误结构与类别不平衡</h3><img class="figure" src="assets/figure_A4_error_imbalance.png" alt="Error structure and imbalance"><div class="reading"><b>怎么看：</b>错误率、漏诊率和误报率越低越好；阳性比例只描述数据构成。三类权重解耦若真正改善疾病证据选择，理论上应降低跨通道错误，但当前没有独立 U1 消融，不能直接归因。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer A-5</div><h3>鲁棒性与捷径负控</h3><img class="figure" src="assets/figure_A5_robustness_controls.png" alt="Shortcut and robustness controls"><div class="reading"><b>怎么看：</b>比较完整模型、QC-only 和去时长/响度模型。U1 三类权重解耦最直接对应这张图：理论预期是完整模型明显高于 QC-only，且删除时长/响度后仍保留信号；Pitt 与 NCMMSC 尚未满足这一要求。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer A bias audit</div><h3>IAEAV 采访者与采集批次混杂</h3><img class="figure" src="assets/figure_4b_iaeav_capture_confounding.png" alt="IAEAV capture confounding"><div class="reading"><b>怎么看：</b>这张图解释为什么 IAEAV 的 1.000 不能当作模型满分表现。身份安全切分发现了采集组与标签共线，因此该数据集被降级为偏倚审计。</div></div>
</section>

<section>
  <h2>6. Layer B：框架是否真正解决旧系统问题</h2>
  <p>Layer B 不重复问分类是否准确，而是检查预测是否经过临床状态、是否保留反证与原始片段、Agent 是否被约束、错误是否能够回退，以及医生报告是否来自同一条锁定决策轨迹。</p>
  <div class="figure-block"><div class="figure-kicker">Layer B overview</div><h3>B. 框架验证总图</h3><img class="figure" src="assets/figure_B_layer_framework_validation_summary.png" alt="Layer B framework validation summary"><div class="reading"><b>怎么看：</b>总图覆盖逐数据集证据契约、B2 与 B3 报告结构、相对基线 AUROC，以及状态干预、完整融合、任务路由和片段回溯。这是老师汇报中连接“机制设计”和“框架结果”的主图。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer B-0</div><h3>证据契约、状态干预与融合增量</h3><img class="figure" src="assets/figure_6_layer_b_validation.png" alt="Layer B validation details"><div class="reading"><b>怎么看：</b>C1 显式工作区主要对应证据与追溯完整性；状态干预检查命名状态是否进入决策；完整融合减去仅状态模型检查低层表示是否仍提供额外信息。任务状态增益目前并不稳定。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer B-1</div><h3>B2 与 B3 的医生报告结构</h3><img class="figure" src="assets/figure_7_report_rubric.png" alt="Clinical report structural audit"><div class="reading"><b>怎么看：</b>B3 在证据完整性、临床可解释性、安全/校准、诊断用途和可追溯性上均高于直接 Agent，平均总分提高 {report_gain:+.2f}/25。它直接对应 C1 工作区和 U6 同轨报告，但仍属于自动结构审计。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer B-2</div><h3>不同数据任务下学习到的分支依赖</h3><img class="figure" src="assets/figure_8_gate_weights.png" alt="Learned branch weights"><div class="reading"><b>怎么看：</b>不同数据任务对语言、语音行为和辅助声学分支的依赖不同。权重表示模型依赖，不代表疾病机制；三类权重解耦要求这些预测权重必须在可观察性和可靠度门控之后才生效。</div></div>
  <div class="figure-block"><div class="figure-kicker">Layer B failure audit</div><h3>未通过项目与解释降级</h3><img class="figure" src="assets/figure_failure_mode_audit.png" alt="Failure mode audit"><div class="reading"><b>怎么看：</b>绿色、黄色与灰色分别表示通过、需谨慎和不适用。C2/C3 的价值首先体现在把不合格 Agent 候选和不可靠数据通道暴露出来，而不是把所有结果调成更高。</div></div>
</section>

<section>
  <h2>7. 三条件结果与历史变化的总体判断</h2>
  <div class="matrix-wrap"><table><thead><tr><th>数据任务</th><th>B1 AUROC</th><th>B2 AUROC</th><th>新版 AUROC</th><th>新版−较强基线</th><th>当前解释边界</th></tr></thead><tbody>{result_table}</tbody></table></div>
  <div class="claim"><b>可以支持</b><div>整套 8.27 系统相对历史 C 在平均准确率、宏 F1 和宏 AUROC 上提高；B3 在多个清洁数据任务上高于 B1/B2；证据追溯、回退执行与报告结构达到预设检查。</div></div>
  <div class="claim"><b>不能支持</b><div>不能根据总体提升推断三类权重解耦、Cognition 监测器或 Agent 单独提高了 AUROC。现有直接结果反而显示 Agent-on AUROC 增量为 0.000，任务特异状态消融均值为 {f(data['task_state_gain']):+.3f}。</div></div>
</section>

<section>
  <h2>8. 与 SpeechCARE 的结果边界</h2>
  <img class="figure" src="assets/figure_speechcare_protocol_aligned.png" alt="PREPARE comparison with SpeechCARE">
  <p class="caption">PREPARE 可对应终点的描述性比较。本项目微平均 AUROC 为 {fmt(f(data['prepare_ours_auc']))}，SpeechCARE 为 {fmt(f(data['prepare_sc_auc']))}；F1 分别为 {fmt(f(data['prepare_ours_f1']))} 与 {fmt(f(data['prepare_sc_f1']))}。SpeechCARE 报告十次训练均值，本项目当前为一次锁定训练与 bootstrap，两者仍不是完整复现实验。</p>
  <div class="notice danger"><b>不能说：</b>“新版已经超过 SpeechCARE。”<br><b>可以说：</b>“新版在同一 PREPARE 任务上数值高于 B1/B2，并新增了 SpeechCARE 未提供的指标—状态—片段追溯、报告权限和认知回退；但端到端多语言表征和预测性能仍有差距。”</div>
</section>

<section>
  <h2>9. 给老师的最终结论</h2>
  <div class="claim"><b>已经完成</b><div>十个任务的三条件评估、任务/角色路由、证据对象、认知状态、可靠性融合、受约束 Agent、回退审计和医生报告使用同一版本化链路。</div></div>
  <div class="claim"><b>已经支持</b><div>新版整体优于历史 C；在多个清洁数据任务上优于 B1/B2；报告结构和证据追溯明显提升；违规 Agent 候选能够被确定性回退。</div></div>
  <div class="claim"><b>尚未支持</b><div>Agent 本身带来预测增益、跨任务状态一定有效、在 PREPARE 超过 SpeechCARE、仅凭语音可靠完成 AD 病理确诊或精确分期。</div></div>
  <div class="claim"><b>下一轮关键实验</b><div>固定同协议 10-seed 训练；完成 PREPARE 多语言编码器训练折内微调；对任务特异状态做稳定性筛选；在独立临床数据上做外部验证与医生读者研究。</div></div>
</section>
<footer>完整材料：<a href="evaluation_report.html">十数据集评估报告</a> · <a href="system_report.html">系统报告</a> · <a href="aggregate_run_audit.html">运行审计</a> · <a href="teacher_framework_update_oral_presentation_zh.md">中文口语稿</a></footer>
</main></body></html>"""


def build_oral(data: dict[str, object]) -> str:
    report_gain = f(data["ours_report"]) - f(data["b2_report"])
    clean = "、".join(dataset_label(x) for x in data["clean_stronger"])
    return f"""# ADvoice 8.27 老师汇报中文口语稿

这次汇报先把三条技术路线分清楚。第一条是 SpeechCARE。它解决的是多语言、多任务语音筛查：原始录音经过低通处理、切片和 Whisper 转录，声学端用 mHuBERT，文本端用 mGTE，再加入人口学表示，通过定制自注意编码器和自适应门控网络完成分类。这里需要准确说明，大模型确实被用于音频异常过滤和任务类型识别，但核心诊断不是大模型，也没有诊断 Agent。它的优势是表征和动态融合强，缺点是门控权重最多说明模型依赖声学、文本还是人口学，不能继续追溯到哪个临床认知状态、哪个原始指标和哪段语音。

第二条是 2026 年的 MEDRF，也就是多模态证据驱动推理框架。它先用临床资料和结构 MRI 训练多模态层次级联分类器，再让检索增强大模型查找相似病例和文本证据，对资料缺失或线索冲突的病例进行多步校正。它与本项目真正相关的地方，是大模型已经进入决策校正，而不是只负责写报告。论文在异质外部队列中报告，加入检索增强大模型后，总体准确率由 0.706 提升至 0.753。不过它的证据单位是临床资料、MRI、相似病例和文本知识，不处理语音任务中的角色、停顿、词汇检索、片段轨迹和录音质量混杂。

第三条是 ADvoice。我们的目标不是复制 SpeechCARE 的模态门控，也不是把 MEDRF 的 RAG 直接搬到语音上，而是把语音中可解释的指标组织成可维护的认知状态，再让监督模型、诊断 Agent 和医生报告共同读取这一对象。SpeechCARE 提供了任务条件化和动态融合的参照，MEDRF 提供了“分类器加证据推理校正”的参照；ADvoice 增加的是指标到状态到片段的追溯、支持与反证、证据权限以及确定性回退。

Cognition-of-Thought 不属于前面两项直接相关医疗工作。我们只借鉴它的一个框架思想：系统不让大模型自由思考，而是维护一个外显状态；每次候选判断之后按规则检查；发现最早违规步骤后回退重做。这个思想在 8.27 被拆成六项实际更新。

第一项更新是三类权重解耦。可观察性回答当前任务能不能测这个状态，可靠度回答这次测量是否可信，预测贡献才由训练折学习。比如一段录音的 RMS 响度偏低，但同时存在远场麦克风和背景噪声。这个数值可以计算，却只能进入质量控制，不能因为与标签相关就写成 AD 证据。这样做的好处，是把预测相关性和医学解释资格分开。

第二项是任务条件化认知工作区。旧版把图片描述、词语流畅性和故事回忆先做平均，可能抹掉只在某一任务出现的异常。新版同时保留病例总体状态和任务状态。比如患者朗读正常，但故事回忆遗漏关键事件并出现反复检索停顿，新版会保留“回忆任务受损、朗读相对保留”，而不是平均成一个不清楚的轻度异常。它让系统区分局部任务缺陷与跨任务持续异常。当前任务特异状态消融均值为 {f(data['task_state_gain']):+.3f}，所以追溯结构已经成立，但预测收益还没有成立。

第三项是两个监督模块。模块 A 学基础风险和状态贡献；模块 B 只使用训练折外轨迹，学习当前病例是否允许 Agent 修正、最大修正幅度和最终校准。比如模块 A 给出 0.68 的风险，但病例的质量控制较差、证据覆盖不足，模块 B 会把可修正幅度限制为零，Agent 只能解释不确定性，不能把概率自由推到 0.90。这样保留了统计学习能力，也阻止自由语言绕过概率边界。

第四项是把单一诊断 Agent 移到决策锁定之前。Agent 读取 AD 医学 Skill、监督先验和结构化证据包，必须提交支持证据、反证、替代解释和证据 ID。比如词汇提取异常，但语义连贯仍正常，Agent 必须同时引用检索停顿片段和连贯表达片段，才能提出“存在词汇检索困难，但尚无广泛语义崩解”的候选判断。这样 Agent 真正参与审查，又不能只挑支持结论的内容。当前执行覆盖率为 {f(data['agent_execution']):.3f}，候选证据有效率为 {f(data['agent_validity']):.3f}，采纳率为 {f(data['agent_accept']):.3f}，说明链路已经运行，但 Agent 的预测增益还未证明。

第五项是确定性认知回退。验证器按固定医学优先级检查标签泄漏、质量证据越权、任务不可观察、无效引用、遗漏反证、无依据分期和概率越界。比如 Agent 把音量偏低直接解释为 AD，系统会触发质量证据越权，删除该引用并要求重做；再次失败就恢复模块 A 的校准先验。当前回退执行率为 {f(data['rollback']):.3f}，Agent-on 宏 AUROC 增量为 {f(data['agent_auc_delta']):.3f}。它证明的是错误候选被阻止，不是 Agent 已经提高准确率。

第六项是决策轨迹锁定与同轨报告。最终风险、状态贡献、支持、反证、限制和片段 ID 先被锁定，报告生成器只能读取这一对象。比如报告写“输出效率下降”，医生可以沿 ID 回到每分钟有效内容、发声时间占比和对应片段；如果这些证据不在锁定轨迹中，这句话不能生成。直接 Agent 的自动结构评分为 {f(data['b2_report']):.2f}/25，新版为 {f(data['ours_report']):.2f}/25，提高 {report_gain:.2f} 分，说明报告结构和追溯链改善，但仍需真实医生读者研究。

把这些更新放在一起看，目前可以总结为四项优势。第一，整套新版相对历史条件 C，在十个任务上的平均准确率提高 {f(data['legacy_acc_delta']):.3f}，宏 F1 提高 {f(data['legacy_f1_delta']):.3f}，宏 AUROC 提高 {f(data['legacy_auc_delta']):.3f}。第二，直接 Agent 到同轨证据报告的自动结构评分提高 {report_gain:.2f}/25。第三，Agent 候选虽然只有 {f(data['agent_validity']):.3f} 符合证据要求，但回退执行率达到 {f(data['rollback']):.3f}，因此不合格候选没有扩散到最终概率。第四，同一套流程已经覆盖十个独立数据任务，同时保留对 IAEAV、Pitt 和 NCMMSC 等混杂通道的降级说明。

这四项优势的证据等级并不相同。性能变化是一项整套系统前后比较；报告提升来自结构审计；回退机制有直接执行记录；跨数据集覆盖则属于工程完整性。清除已知强混杂后，较干净支持优于两个基线的任务包括 {clean}。没有逐项消融，因此不能把全部性能变化归因于 Cognition-of-Thought、Agent 或某一个监督模块。

与 SpeechCARE 的边界必须保持清楚。在 PREPARE 可对应终点上，我们的微平均 AUROC 是 {f(data['prepare_ours_auc']):.3f}，SpeechCARE 是 {f(data['prepare_sc_auc']):.3f}；F1 分别为 {f(data['prepare_ours_f1']):.3f} 和 {f(data['prepare_sc_f1']):.3f}。当前还没有超过 SpeechCARE。与 MEDRF 也不能直接比准确率，因为输入、标签和临床任务不同。我们的已验证增量主要是语音特有的临床命名状态、指标与片段追溯、反证、报告权限和可回退决策；端到端多语言预测和真实医生验证仍需补齐。

当前未解决的问题包括 PREPARE 多语言表征不足、ADReSSo progression 的纵向终点不匹配、TAUKADIAL 的语言迁移不足，以及 IAEAV、Pitt、NCMMSC 的采集或质量捷径。下一轮需要用同协议多随机种子实验、训练折内编码器适配、任务状态稳定性筛选、外部临床验证和真实医生读者研究分别解决，不能继续用增加指标代替这些验证。
"""


def main() -> None:
    data = collect()
    HTML_OUT.write_text(build_html(data), encoding="utf-8")
    ORAL_OUT.write_text(build_oral(data), encoding="utf-8")
    print(HTML_OUT)
    print(ORAL_OUT)


if __name__ == "__main__":
    main()
