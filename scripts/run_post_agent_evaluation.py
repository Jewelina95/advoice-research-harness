#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd

from advoice.post_agent_evaluation import (
    audit_correction_propagation,
    audit_permission_compliance,
    audit_trace_integrity,
    evaluate_agent_gain,
    evaluate_perturbation_stability,
    portable_source_descriptor,
    summarize_representation_stage,
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = _load_json(path)
    cases = payload.get("cases", payload) if isinstance(payload, dict) else payload
    if not isinstance(cases, list):
        raise TypeError("case audit must contain a list under 'cases'")
    return cases


def _load_truth(path: Path) -> dict[str, str]:
    frame = pd.read_csv(path)
    required = {"subject_id", "label"}
    if not required.issubset(frame.columns):
        raise ValueError(f"truth CSV must contain {sorted(required)}")
    return frame.assign(subject_id=frame["subject_id"].astype(str)).set_index("subject_id")["label"].astype(str).to_dict()


def _fmt(value: Any, digits: int = 3) -> str:
    if value is None:
        return "NA"
    if isinstance(value, (float, int)):
        return f"{float(value):.{digits}f}"
    return html.escape(str(value))


def _percent(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def _historical_context(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    payload = _load_json(path)
    summary = {row["arm"]: row for row in payload.get("summary", [])}
    return {
        "dataset": payload.get("dataset"),
        "n": payload.get("n"),
        "repeats": payload.get("repeats"),
        "selected_median_weights": payload.get("selected_median_weights", {}),
        "supervised": summary.get("supervised"),
        "supervised_agent": summary.get("supervised_agent"),
        "full_joint": summary.get("full_joint"),
        "limitations": payload.get("limitations", []),
    }


def _bar(label: str, value: float, color: str) -> str:
    width = max(0.0, min(100.0, value * 100.0))
    return (
        "<div class='bar-row'><span>" + html.escape(label) + "</span>"
        f"<div class='track'><i style='width:{width:.1f}%;background:{color}'></i></div>"
        f"<b>{value:.3f}</b></div>"
    )


def _render_report(result: dict[str, Any]) -> str:
    gain = result["agent_gain"]
    permission = result["permission"]
    trace = result["trace"]
    propagation = result["propagation"]
    perturbation = result["perturbation"]
    representation = result["representation"]
    historical = result.get("historical_context")
    frozen = gain["frozen"]
    fused = gain["fused"]
    paired = gain["paired_counts"]

    metric_bars = "".join(
        [
            _bar("监督模型 Accuracy", frozen["accuracy"], "#77848b"),
            _bar("联合模型 Accuracy", fused["accuracy"], "#147a63"),
            _bar("监督模型观测类别 Macro-F1", frozen["observed_class_macro_f1"], "#77848b"),
            _bar("联合模型观测类别 Macro-F1", fused["observed_class_macro_f1"], "#147a63"),
        ]
    )
    question_rows = [
        (
            "Agent 是否提高准确率",
            "小样本实证",
            f"本批 {gain['case_count']} 例：Accuracy {_fmt(frozen['accuracy'])} → {_fmt(fused['accuracy'])}；"
            f"修正 {paired['helped']} 例、伤害 {paired['harmed']} 例；配对精确检验 p={_fmt(gain['paired_exact_p_value'])}。",
            "不能外推为总体增益。",
        ),
        (
            "Agent 是否比监督模型更稳定",
            "未完成",
            "当前每例只有一次新 Agent 输出，没有同一病例的重复采样。",
            "需固定输入后重复运行，并与监督模型的零运行方差分开比较。",
        ),
        (
            "Agent 是否遵守证据权限",
            "接口验证 + 结构审计",
            f"共 {permission['citation_count']} 条引用，病例/状态越界 {permission['violation_count']} 条。"
            "运行时 schema 只暴露 inference-permitted evidence。",
            "医生报告尚未生成，因此 report permission 尚未实证。",
        ),
        (
            "是否受语言改写或证据顺序影响",
            "未运行",
            perturbation.get("reason", "尚无配对扰动输出。"),
            "必须从原始结构化证据包重新调用 Agent。",
        ),
        (
            "Trace Map 是否忠实",
            "结构链已审计",
            f"{trace['complete_case_count']}/{trace['case_count']} 例通过 packet、transaction、revision、fusion 哈希链接。",
            "哈希完整不等于因果忠实；还需删除/替换证据后的反事实测试。",
        ),
        (
            "证据修正是否一致传播",
            "状态与预测可审计",
            f"{propagation['transaction_to_state_change']} 例 transaction 进入状态变化；"
            f"{propagation['state_change_to_prediction_change']} 例继续改变最终类别；"
            f"{propagation['state_change_blocked_or_subthreshold']} 例被门控或未越过分类边界。",
            "报告一致性尚不可检验，因为当前研究路径明确禁止生成医生报告。",
        ),
    ]
    question_html = "".join(
        "<tr>"
        f"<td>{html.escape(question)}</td><td><span class='status'>{html.escape(status)}</span></td>"
        f"<td>{html.escape(evidence)}</td><td>{html.escape(boundary)}</td></tr>"
        for question, status, evidence, boundary in question_rows
    )
    trace_rows = "".join(
        f"<tr><td>{html.escape(row['case_id'])}</td>"
        f"<td>{'通过' if row['complete'] else '失败'}</td>"
        f"<td>{sum(row['checks'].values())}/{len(row['checks'])}</td></tr>"
        for row in trace["cases"]
    )
    propagation_rows = "".join(
        f"<tr><td>{html.escape(row['case_id'])}</td><td>{'是' if row['has_transaction'] else '否'}</td>"
        f"<td>{'是' if row['state_changed'] else '否'}</td><td>{row['state_probability_l1']:.3f}</td>"
        f"<td>{'是' if row['prediction_changed'] else '否'}</td>"
        f"<td>{row['state_gate']:.2f} / {row['agent_gate']:.2f} / {row['staging_gate']:.2f}</td></tr>"
        for row in propagation["cases"]
    )
    effect_rows = "".join(
        f"<tr><td>{html.escape(row['case_id'])}</td><td>{html.escape(row['truth'])}</td>"
        f"<td>{html.escape(row['before'])} → {html.escape(row['after'])}</td>"
        f"<td>{row['before_true_probability']:.3f} → {row['after_true_probability']:.3f}</td>"
        f"<td>{row['true_probability_delta']:+.3f}</td></tr>"
        for row in gain["case_effects"]
    )
    historical_html = ""
    if historical and historical.get("supervised") and historical.get("supervised_agent"):
        base = historical["supervised"]
        agent = historical["supervised_agent"]
        joint = historical["full_joint"]
        historical_html = f"""
        <section><h2>不能回避的历史反证</h2>
        <p>ADReSS 2020 的旧缓存 Agent 做过 100 次重复分层留出。监督模型 Accuracy 为 {_fmt(base['accuracy'])}±{_fmt(base['accuracy_sd'])}，
        监督+Agent 为 {_fmt(agent['accuracy'])}±{_fmt(agent['accuracy_sd'])}，完整联合为 {_fmt(joint['accuracy'])}±{_fmt(joint['accuracy_sd'])}。
        验证集选择的 Agent 中位权重为 {_fmt(historical['selected_median_weights'].get('agent_weight'))}。
        这说明旧 Agent 没有形成可重复的净增益；PREPARE 9 例的正向结果只能作为新版机制的先导信号。</p></section>
        """

    return f"""<!doctype html><html lang='zh-CN'><head><meta charset='utf-8'>
<meta name='viewport' content='width=device-width,initial-scale=1'><title>Agent 后证据治理验证</title>
<style>
:root{{--ink:#172229;--muted:#59666d;--line:#d8dfe1;--paper:#fff;--bg:#f4f6f5;--green:#147a63;--green-soft:#e7f2ee;--amber:#a46613;--amber-soft:#fff3df;--red:#a23f37}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--ink);font-family:Arial,"PingFang SC",sans-serif;letter-spacing:0}}
main{{max-width:1180px;margin:auto;padding:48px 28px 80px}}h1{{font-size:38px;line-height:1.18;margin:0 0 12px}}h2{{font-size:25px;margin:42px 0 14px}}h3{{font-size:18px;margin:0 0 8px}}
p,li,td{{line-height:1.68}}.lede{{font-size:18px;color:var(--muted);max-width:950px}}.kpis{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:28px 0}}
.kpi{{background:var(--paper);border:1px solid var(--line);padding:18px}}.kpi b{{display:block;font-size:30px}}.kpi span{{color:var(--muted)}}
.callout{{background:var(--amber-soft);border-left:6px solid var(--amber);padding:18px 20px;line-height:1.7}}
.bars{{background:white;border:1px solid var(--line);padding:20px}}.bar-row{{display:grid;grid-template-columns:210px 1fr 60px;gap:12px;align-items:center;margin:13px 0}}
.track{{height:22px;background:#edf0f1;position:relative}}.track i{{display:block;height:100%}}.bar-row b{{font-variant-numeric:tabular-nums}}
table{{width:100%;border-collapse:collapse;background:white;font-size:14px}}th,td{{padding:12px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#e9edef}}
.status{{display:inline-block;background:var(--green-soft);color:#0c5d49;padding:4px 7px;font-weight:700}}.grid{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}
.panel{{background:white;border:1px solid var(--line);padding:20px}}.panel p{{color:var(--muted)}}code{{font-family:ui-monospace,SFMono-Regular,monospace}}
@media(max-width:800px){{.kpis,.grid{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:1fr}}main{{padding:28px 16px}}.table-wrap{{overflow:auto}}}}
</style></head><body><main>
<h1>Agent 后证据治理验证：当前能证明什么，不能证明什么</h1>
<p class='lede'>本报告只分析已经完成的真实 Agent 审计结果，并把预训练表示、监督预测、状态重放和 Agent 修正分开。结论不使用测试标签调门控，也不把接口约束当作效果证据。</p>
<div class='kpis'><div class='kpi'><b>{gain['case_count']}</b><span>PREPARE 先导病例</span></div>
<div class='kpi'><b>{paired['helped']}</b><span>纠正的监督错误</span></div><div class='kpi'><b>{paired['harmed']}</b><span>破坏的正确病例</span></div>
<div class='kpi'><b>{_percent(trace['trace_complete_rate'])}</b><span>结构 Trace 完整率</span></div></div>
<div class='callout'><b>当前结论：</b>新版 Agent 在 9 例 PREPARE pilot 中出现正向病例级变化，但样本量不足，配对检验也不足以支持总体准确率提升。权限和 Trace 目前主要证明“系统按规则运行”，还没有证明“Agent 的医学推理稳定且因果忠实”。</div>
<section><h2>六个问题的证据状态</h2><div class='table-wrap'><table><thead><tr><th>问题</th><th>状态</th><th>已有证据</th><th>结论边界</th></tr></thead><tbody>{question_html}</tbody></table></div></section>
<section><h2>本批病例的预测变化</h2><div class='bars'>{metric_bars}</div>
<p>Accuracy 变化 {_fmt(gain['delta']['accuracy'])}，观测类别 Macro-F1 变化 {_fmt(gain['delta']['observed_class_macro_f1'])}，Log loss 变化 {_fmt(gain['delta']['log_loss'])}。本批没有 MCI，三分类 protocol Macro-F1 只作为审计值，不作为效果结论；Log loss 越低越好。</p></section>
<section><h2>改判来自哪条路径</h2><div class='callout'><b>直接 Agent 门控启动 {propagation['agent_gate_active_count']}/{propagation['case_count']} 例；状态门控启动 {propagation['state_gate_active_count']}/{propagation['case_count']} 例。</b> 因此这批变化来自 Agent 提交证据修正后重算 StateCard，再由状态门控改变概率；不是把 Agent 的类别意见直接加到监督概率上。</div>
<div class='table-wrap'><table><thead><tr><th>病例</th><th>真实标签</th><th>类别</th><th>真实类别概率</th><th>概率变化</th></tr></thead><tbody>{effect_rows}</tbody></table></div></section>
{historical_html}
<section><h2>自监督表示与 Agent 增益必须拆开</h2><div class='grid'><div class='panel'><h3>表示阶段</h3><p>音频骨干：<code>{html.escape(str(representation['audio_backbone']['model']))}</code><br>文本骨干：<code>{html.escape(str(representation['text_backbone']['model']))}</code></p><p>这些是预训练表示骨干；进入本项目后，融合器和校正器仍由监督标签拟合。</p></div>
<div class='panel'><h3>监督训练内部变化</h3><p>OOF Macro-F1：{_fmt(representation['base_oof_macro_f1'])} → {_fmt(representation['final_oof_macro_f1'])}<br>OOF Macro-AUROC：{_fmt(representation['base_oof_macro_auroc'])} → {_fmt(representation['final_oof_macro_auroc'])}</p><p>这是监督训练链内部变化，不是 Agent 的因果增益。当前还缺“相同融合器、不使用预训练表示”的匹配消融。</p></div></div></section>
<section><h2>Trace 链逐例检查</h2><div class='table-wrap'><table><thead><tr><th>病例</th><th>链路状态</th><th>通过检查</th></tr></thead><tbody>{trace_rows}</tbody></table></div></section>
<section><h2>证据修正传播</h2><div class='table-wrap'><table><thead><tr><th>病例</th><th>有事务</th><th>状态改变</th><th>状态概率 L1</th><th>最终类别改变</th><th>状态 / Agent / 分期门控</th></tr></thead><tbody>{propagation_rows}</tbody></table></div></section>
<section><h2>下一轮正式实验</h2><ol><li>每个病例生成固定的 baseline、中文同义改写、英文同义改写、证据倒序、状态块倒序五个版本。</li><li>每个版本重复运行，报告类别一致率、概率 Jensen-Shannon 散度、引用集合 Jaccard 和状态修正一致率。</li><li>做证据删除与替换：删除被 Trace 标记为关键的证据应改变对应状态；删除无关证据不应改变结论。</li><li>在相同表示骨干下比较监督、监督+状态、监督+Agent、完整联合；另设无预训练表示臂，隔离自监督表示贡献。</li><li>接入医生报告后，再验证 inference permission 与 report permission 是否分别生效，以及状态、概率和报告是否同步更新。</li></ol></section>
</main></body></html>"""


def _render_oral(result: dict[str, Any]) -> str:
    gain = result["agent_gain"]
    paired = gain["paired_counts"]
    permission = result["permission"]
    trace = result["trace"]
    propagation = result["propagation"]
    representation = result["representation"]
    return f"""# Agent 后验证口语汇报

这次研究回答的不是“系统能不能跑”，而是 Agent 加入以后，是否真的改变了预测，而且这种改变是否稳定、受约束、能够回溯。

先看准确率。当前已经完成的是 PREPARE 的 {gain['case_count']} 例先导实验。冻结监督模型的准确率是 {gain['frozen']['accuracy']:.3f}，加入状态重放和 Agent 联合修正后是 {gain['fused']['accuracy']:.3f}。病例层面，系统纠正了 {paired['helped']} 例，没有破坏原来判对的病例。这个方向是正的，但样本只有九例，配对精确检验的 p 值是 {gain['paired_exact_p_value']:.3f}，所以现在只能说发现了先导信号，不能说 Agent 已经稳定提高总体准确率。

第二，当前不能证明 Agent 比监督模型更稳定。监督模型对固定输入是确定的，新 Agent 每例目前只有一次输出。要回答稳定性，必须对同一个证据包重复运行，并加入语言改写和证据顺序扰动，比较类别一致率、概率变化和引用集合变化。现有结果文件没有保存可以重新排序的完整输入包，所以这部分需要重新调用 Agent，不能从单次输出反推。

第三，证据权限目前有两层结论。代码层面，Agent 的 schema 只允许引用具有推理权限的证据；结果层面，本批共有 {permission['citation_count']} 条引用，病例或状态越界为 {permission['violation_count']} 条。但是医生报告尚未在这条研究链中生成，因此我们还没有验证“只能推理、不能写入报告”的证据是否真的被报告层阻断。

第四，Trace Map 的哈希链在 {trace['complete_case_count']}/{trace['case_count']} 个病例上完整连接了原始预测、修正事务、重放状态和最终融合。这证明记录没有断链，但不等于 Trace 就是实际因果路径。下一步要删除 Trace 所指向的关键证据，观察对应状态和判断是否随之改变；否则它可能只是事后记录。

第五，证据修正进入状态层的病例有 {propagation['transaction_to_state_change']} 例，其中 {propagation['state_change_to_prediction_change']} 例继续改变最终类别，{propagation['state_change_blocked_or_subthreshold']} 例没有跨越分类边界。直接 Agent 门控在 {propagation['agent_gate_active_count']} 例启动，状态门控在 {propagation['state_gate_active_count']} 例启动。因此本批改善来自 Agent 审查证据之后的状态重放，而不是 Agent 类别意见的直接加权。报告层还没有接入，所以状态、预测、报告三者一致更新目前只能验证前两项。

最后要澄清自监督学习。当前音频使用 {representation['audio_backbone']['model']}，文本使用 {representation['text_backbone']['model']}。它们提供预训练表示，后面的融合器和校正器仍然是监督训练。监督训练内部 Macro-F1 从 {representation['base_oof_macro_f1']:.3f} 提高到 {representation['final_oof_macro_f1']:.3f}，这不能算作 Agent 增益。正式实验必须在相同编码器下比较有无 Agent，同时再增加无预训练表示的匹配消融，分别回答表示学习和 Agent 各自贡献了多少。
"""


def _perturbation_manifest(cases: list[dict[str, Any]]) -> dict[str, Any]:
    variants = [
        {"id": "baseline", "operation": "unchanged structured evidence package"},
        {"id": "paraphrase_zh", "operation": "meaning-preserving Chinese paraphrase of narrative fields"},
        {"id": "paraphrase_en", "operation": "meaning-preserving English paraphrase of narrative fields"},
        {"id": "reverse_evidence_order", "operation": "reverse evidence order within every state"},
        {"id": "reverse_state_order", "operation": "reverse state block order without changing identifiers"},
    ]
    return {
        "schema_version": "advoice.post_agent_perturbation_manifest.v1",
        "case_ids": [str(case["case_id"]) for case in cases],
        "variants": variants,
        "required_repeats_per_variant": 5,
        "locked_fields": [
            "model",
            "model_revision",
            "temperature",
            "skill_version",
            "evidence_values",
            "evidence_permissions",
            "state_ids",
        ],
        "primary_endpoints": [
            "label_agreement",
            "probability_js_divergence",
            "citation_jaccard",
            "state_revision_agreement",
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate evidence-governed Agent behavior after inference.")
    parser.add_argument("--case-audit", type=Path, required=True)
    parser.add_argument("--truth-csv", type=Path, required=True)
    parser.add_argument("--model-metadata", type=Path, required=True)
    parser.add_argument("--historical-five-arm", type=Path)
    parser.add_argument("--perturbation-results", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    cases = _load_cases(args.case_audit)
    truth = _load_truth(args.truth_csv)
    class_order = tuple(cases[0]["prepared"]["class_order"])
    perturbation_rows: list[dict[str, Any]] = []
    if args.perturbation_results and args.perturbation_results.exists():
        payload = _load_json(args.perturbation_results)
        perturbation_rows = payload.get("rows", payload) if isinstance(payload, dict) else payload
    result = {
        "schema_version": "advoice.post_agent_evaluation.v1",
        "source": {
            "case_audit": portable_source_descriptor(args.case_audit),
            "truth_csv": portable_source_descriptor(args.truth_csv),
            "model_metadata": portable_source_descriptor(args.model_metadata),
        },
        "agent_gain": evaluate_agent_gain(cases, truth, class_order=class_order),
        "permission": audit_permission_compliance(cases),
        "trace": audit_trace_integrity(cases),
        "propagation": audit_correction_propagation(cases),
        "perturbation": evaluate_perturbation_stability(perturbation_rows),
        "representation": summarize_representation_stage(_load_json(args.model_metadata)),
        "historical_context": _historical_context(args.historical_five_arm),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "report.html").write_text(_render_report(result), encoding="utf-8")
    (args.output_dir / "oral_zh.md").write_text(_render_oral(result), encoding="utf-8")
    (args.output_dir / "perturbation_manifest.json").write_text(
        json.dumps(_perturbation_manifest(cases), ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
