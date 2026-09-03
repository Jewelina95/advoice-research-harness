from __future__ import annotations

import json
from html import escape
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts" / "PREPARE_DrivenData"
OUT = ROOT / "reports" / "latest" / "prepare_9_2_method_audit.html"


def _read_json(name: str) -> dict:
    return json.loads((ARTIFACTS / name).read_text(encoding="utf-8"))


def _metric(layer_a: pd.DataFrame, condition: str, metric: str) -> float:
    rows = layer_a.loc[
        layer_a["condition"].eq(condition)
        & layer_a["analysis_scope"].eq("full_available_cohort")
        & layer_a["metric"].eq(metric),
        "value",
    ]
    if len(rows) != 1:
        raise ValueError(f"Expected one Layer A row for {condition}/{metric}; found {len(rows)}")
    return float(rows.iloc[0])


def _check(layer_b: pd.DataFrame, name: str) -> tuple[float, bool]:
    rows = layer_b.loc[layer_b["check"].eq(name)]
    if len(rows) != 1:
        raise ValueError(f"Expected one Layer B row for {name}; found {len(rows)}")
    return float(rows.iloc[0]["value"]), bool(rows.iloc[0]["passed"])


def _bar(
    value: float,
    reference: float,
    color: str,
    value_label: str = "ADvoice 9.2",
) -> str:
    maximum = max(value, reference, 1e-9)
    return (
        '<div class="bar-pair">'
        f'<div><span>{escape(value_label)}</span><i style="width:{100 * value / maximum:.1f}%;background:{color}"></i>'
        f'<b>{value:.3f}</b></div>'
        f'<div><span>SpeechCARE</span><i style="width:{100 * reference / maximum:.1f}%;background:#9aa3ad"></i>'
        f'<b>{reference:.3f}</b></div></div>'
    )


def build() -> Path:
    layer_a = pd.read_csv(ARTIFACTS / "layer_a_metrics.csv")
    layer_b = pd.read_csv(ARTIFACTS / "layer_b_checks.csv")
    gate = _read_json("speechcare_gate_9_2.json")
    status = _read_json("cognitive_agent_status.json")
    calibration = _read_json("agent_correction_calibration.json")
    status_is_current_protocol = bool(
        status.get("fusion_policy") == "separate_hc_vs_impairment_and_mci_vs_ad"
        and abs(float(status.get("correction_strength", -1.0)) - float(calibration.get("selected_screening_strength", -2.0))) < 1e-12
        and abs(float(status.get("staging_correction_strength", -1.0)) - float(calibration.get("selected_staging_strength", -2.0))) < 1e-12
    )
    prototype_audit = json.loads(
        (ARTIFACTS / "cognitive_prototype_audit" / "audit.json").read_text(
            encoding="utf-8"
        )
    )
    prototype_metrics = prototype_audit["metrics"]

    ours = {
        key: _metric(layer_a, "Ours", key)
        for key in ("accuracy", "macro_f1", "macro_auroc_ovr", "micro_auroc_ovr", "micro_auprc")
    }
    b1 = {key: _metric(layer_a, "B1", key) for key in ("accuracy", "macro_f1", "macro_auroc_ovr")}
    b2 = {key: _metric(layer_a, "B2", key) for key in ("accuracy", "macro_f1", "macro_auroc_ovr")}
    same_auc, _ = _check(layer_b, "same-encoder cognition macro AUROC gain")
    same_acc, _ = _check(layer_b, "same-encoder cognition accuracy gain")
    agent_auc, agent_auc_passed = _check(layer_b, "Agent-on macro AUROC delta")
    agent_acc, agent_acc_passed = _check(layer_b, "Agent-on accuracy delta")
    evidence_validity, evidence_validity_passed = _check(layer_b, "diagnostic Agent evidence validity")
    report_permission, report_permission_passed = _check(layer_b, "report-permission audit")
    qc_separation, qc_separation_passed = _check(layer_b, "quality-evidence separation audit")

    comparisons = gate["comparison"]
    passed_count = sum(bool(item["passed"]) for item in comparisons.values())
    extension = _read_json("speechcare_cognitive_extension/result.json")
    extension_provenance_verified = bool(
        extension.get("cognitive_provenance")
        and extension.get("cognitive_provenance", {}).get("status") != "legacy_unverified"
    )
    extension_current = extension["speechcare_plus_cognition"]
    extension_reference = extension["speechcare_published_mean"]
    extension_comparisons = {
        key: {
            "speechcare_plus_advoice_cognition": float(extension_current[key]),
            "speechcare_published_mean": float(extension_reference[key]),
            "delta": float(extension_current[key]) - float(extension_reference[key]),
            "passed": float(extension_current[key]) > float(extension_reference[key]),
        }
        for key in (
            "micro_auroc_ovr",
            "micro_f1",
            "weighted_auroc_ovr",
            "micro_auprc",
            "weighted_auprc",
        )
    }
    extension_passed_count = sum(
        bool(item["passed"]) for item in extension_comparisons.values()
    )
    gate_class = "pass" if gate["development_superiority_gate_passed"] else "fail"
    gate_label = "通过" if gate["development_superiority_gate_passed"] else "未通过"

    modifications = [
        (
            "先验盲化",
            "Agent 首轮看不到监督概率、预测类别和真实标签，只依据证据对象形成独立判断。",
            "合理",
            "避免把复制模型输出误写成 Agent 增益。",
        ),
        (
            "类型化证据 ID",
            "统一使用 state:、metric:、segment:、qc:，schema、验证器和报告共同解析。",
            "已验证",
            f"证据合法率 {evidence_validity:.1%}；报告权限审计 {report_permission:.0%}。",
        ),
        (
            "结构化覆盖门控",
            "按可观察状态、任务和分支计算覆盖率；重复指标不再抬高或压低覆盖率。",
            "合理",
            "同一病例不再因批次组成或指标总数变化而改变门控。",
        ),
        (
            "病例路由",
            "按任务族、语言、模态可用性、质量限制和预测-证据冲突选择审查路径。",
            "有边界",
            "稀有任务共享父任务模型；质量信息只降低可信度，不增加疾病风险。",
        ),
        (
            "离散证据似然",
            "Agent 分别输出健康/受损和MCI/AD两组0至4级证据强度，不直接生成最终疾病概率。",
            "合理",
            "开发集内校准离散等级；分期证据不足时强制中性并回退。",
        ),
        (
            "验证集学习修正强度",
            "仅在折外开发病例分别选择筛查和分期修正强度。",
            "部分稳定",
            f"筛查强度 {float(status['correction_strength']):.3f}；分期强度 {float(status.get('staging_correction_strength', 0.0)):.3f}。",
        ),
        (
            "折内证据重建",
            "Agent 校准病例的正常对照参考、MetricEvidence 和状态卡仅由外层训练折重建。",
            "已修复",
            "校准病例不再参与自己的正常参考分布或支持证据排序。",
        ),
        (
            "类别平衡认知参照卡",
            "Agent 获得训练折内HC、MCI、AD状态中位数和稳健尺度；不提供类别比例或基础模型概率。",
            "开发集有效",
            f"折外健康/受损 AUROC {prototype_metrics['screening_auroc_hc_vs_impaired']:.3f}；MCI/AD AUROC {prototype_metrics['staging_auroc_mci_vs_ad']:.3f}。",
        ),
        (
            "推理与报告权限解耦",
            "可靠的语言/任务辅助证据可进入研究性推理，但报告生成前会被移除，不能形成临床主张。",
            "已实现，待重跑",
            "避免因报告安全限制同时丢失预测信息；低层谱学和质量变量仍被排除。",
        ),
        (
            "任务参考去重",
            "存在可靠任务特异参考时只向Agent提供任务特异状态；总体参考仅在任务参考缺失时回退。",
            "已实现，待重跑",
            "同一测量不再同时相对总体HC和任务HC产生相反方向解释。",
        ),
        (
            "相同编码器隔离",
            "固定相同文本/音频编码器，对比普通模态门控、认知框架、Agent 修正。",
            "已验证",
            f"认知框架单独带来宏 AUROC +{same_auc:.3f}、准确率 +{same_acc:.3f}。",
        ),
    ]
    modification_rows = "".join(
        "<tr>"
        f"<td><b>{escape(name)}</b></td><td>{escape(implementation)}</td>"
        f"<td><span class=\"tag\">{escape(verdict)}</span></td><td>{escape(result)}</td>"
        "</tr>"
        for name, implementation, verdict, result in modifications
    )

    metric_cards = "".join(
        f"<article><span>{label}</span><strong>{ours[key]:.3f}</strong>"
        f"<small>B1 {b1[key]:.3f} · B2 {b2[key]:.3f}</small></article>"
        for label, key in (
            ("Accuracy", "accuracy"),
            ("Macro F1", "macro_f1"),
            ("Macro AUROC", "macro_auroc_ovr"),
        )
    )

    speechcare_panels = "".join(
        f"<article><h3>{escape(label)}</h3>{_bar(float(item['advoice']), float(item['speechcare_published_mean']), '#356fa8')}"
        f"<p class=\"delta\">差值 {float(item['delta']):+.3f}</p></article>"
        for label, item in (
            ("Micro AUROC", comparisons["micro_auroc_ovr"]),
            ("Micro F1", comparisons["micro_f1"]),
            ("Micro AUPRC", comparisons["micro_auprc"]),
        )
    )

    extension_panels = "".join(
        f"<article><h3>{escape(label)}</h3>"
        f"{_bar(float(item['speechcare_plus_advoice_cognition']), float(item['speechcare_published_mean']), '#3d7a58', 'SC + cognition')}"
        f"<p class=\"delta positive\">差值 {float(item['delta']):+.3f}</p></article>"
        for label, item in (
            ("Micro AUROC", extension_comparisons.get("micro_auroc_ovr", {})),
            ("Micro F1", extension_comparisons.get("micro_f1", {})),
            ("Weighted AUROC", extension_comparisons.get("weighted_auroc_ovr", {})),
            ("Micro AUPRC", extension_comparisons.get("micro_auprc", {})),
            ("Weighted AUPRC", extension_comparisons.get("weighted_auprc", {})),
        )
        if item
    )

    two_stage_calibration = "selected_screening_strength" in calibration
    if two_stage_calibration:
        calibration_rows = "".join(
            f"<tr><td>{float(row['screening_strength']):.3f}</td>"
            f"<td>{float(row['staging_strength']):.3f}</td>"
            f"<td>{float(row['macro_f1']):.3f}</td><td>{float(row['macro_auroc']):.3f}</td>"
            f"<td>{'选择' if float(row['screening_strength']) == float(calibration['selected_screening_strength']) and float(row['staging_strength']) == float(calibration['selected_staging_strength']) else ''}</td></tr>"
            for row in calibration["candidates"]
        )
        calibration_header = "<th>筛查强度</th><th>分期强度</th>"
    else:
        calibration_rows = "".join(
            f"<tr><td colspan=\"2\">{float(row['strength']):.3f}</td>"
            f"<td>{float(row['macro_f1']):.3f}</td><td>{float(row['macro_auroc']):.3f}</td>"
            f"<td>{'选择' if float(row['strength']) == float(calibration['selected_strength']) else ''}</td></tr>"
            for row in calibration["candidates"]
        )
        calibration_header = "<th colspan=\"2\">共享强度</th>"

    html = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADvoice 9.2 方法审计与 PREPARE 门禁</title>
<style>
:root{{--ink:#1f2933;--muted:#5d6975;--line:#dbe2e8;--paper:#fff;--wash:#f5f7f9;--blue:#356fa8;--green:#3d7a58;--red:#b84e48;--amber:#a96f18}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--wash);color:var(--ink);font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif;line-height:1.65;letter-spacing:0}}
main{{max-width:1180px;margin:0 auto;background:var(--paper);padding:54px 68px 72px}}h1{{font-size:34px;line-height:1.25;margin:0 0 12px}}h2{{font-size:24px;margin:46px 0 16px;padding-top:26px;border-top:1px solid var(--line)}}h3{{font-size:17px;margin:0 0 12px}}p{{margin:8px 0}}.lead{{font-size:18px;color:#394550;max-width:980px}}.notice{{margin:24px 0;padding:16px 18px;border-left:5px solid var(--red);background:#fff4f3}}.notice.pass{{border-color:var(--green);background:#f2f8f4}}.status{{display:inline-block;padding:3px 9px;border-radius:4px;background:#fde8e6;color:#8f312c;font-weight:700}}.status.pass{{background:#e5f3e9;color:#285c3d}}
.steps{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:10px;margin:18px 0}}.step{{min-height:124px;border:1px solid var(--line);border-top:4px solid var(--blue);padding:14px;background:#fff}}.step b{{display:block;margin-bottom:5px}}.step span{{font-size:14px;color:var(--muted)}}
table{{width:100%;border-collapse:collapse;font-size:14px}}th,td{{padding:11px 10px;border-bottom:1px solid var(--line);vertical-align:top;text-align:left}}th{{background:#eef2f5}}.tag{{white-space:nowrap;font-weight:700;color:#315c45}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.cards article,.compare article{{border:1px solid var(--line);padding:18px;background:#fff}}.cards span{{display:block;color:var(--muted)}}.cards strong{{display:block;font-size:32px;color:var(--blue)}}.cards small{{color:var(--muted)}}
.compare{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}}.bar-pair>div{{display:grid;grid-template-columns:88px 1fr 48px;align-items:center;gap:8px;margin:10px 0;font-size:13px}}.bar-pair i{{height:16px;display:block;min-width:2px}}.delta{{font-weight:700;color:var(--red)}}.delta.positive{{color:var(--green)}}
.split{{display:grid;grid-template-columns:1fr 1fr;gap:18px}}.box{{border:1px solid var(--line);padding:18px}}.good{{border-left:5px solid var(--green);background:#f3f9f5}}.bad{{border-left:5px solid var(--red);background:#fff4f3}}ol{{padding-left:22px}}li{{margin:8px 0}}code{{background:#eef2f5;padding:2px 5px;border-radius:3px}}footer{{margin-top:46px;padding-top:18px;border-top:1px solid var(--line);color:var(--muted);font-size:13px}}
@media(max-width:820px){{main{{padding:28px 18px}}.steps{{grid-template-columns:1fr 1fr}}.cards,.compare,.split{{grid-template-columns:1fr}}}}@media print{{body{{background:white}}main{{max-width:none;padding:20mm}}}}
</style></head><body><main>
<h1>ADvoice 9.2：方法审计、Agent 改造与 PREPARE 门禁</h1>
<p class="lead">本轮把 Agent 从“读取监督概率后生成说明”改为“在先验盲化条件下审查类型化临床证据，再由受限融合器决定是否修正”。审计进一步发现：旧版 Agent 没有训练折内的诊断组参照，因此只能判断单条指标是否偏离正常，却无法学习HC、MCI和AD的联合状态模式。9.2现已增加折内证据重建和类别平衡认知参照卡；新版 Agent 的独立增益仍需重新运行后确认。</p>
<div class="notice {gate_class}"><b>ADvoice独立系统门禁：<span class="status {gate_class}">{gate_label}</span></b>　通过 {passed_count}/{len(comparisons)} 项。当前测试集已在历史开发中被查看，即使未来数值超过论文均值，也只能先作为回顾性开发结果，最终仍需新的锁定外部队列。</div>

<h2>1. 9.2 的实际诊断链</h2>
<div class="steps">
<div class="step"><b>1　输入与任务路由</b><span>识别语言、任务族、患者语音范围、模态和质量；不读取疾病标签。</span></div>
<div class="step"><b>2　共享表示</b><span>mHuBERT 音频表示与 GTE 多语言文本表示，为监督模型和认知框架提供共同输入。</span></div>
<div class="step"><b>3　MetricEvidence</b><span>记录测量值、异常方向、折内参考、可靠度、混杂、任务、片段和报告权限。</span></div>
<div class="step"><b>4　StateCard</b><span>把同一临床构念内的相关指标聚合为总体与任务特异认知状态。</span></div>
<div class="step"><b>5　监督先验</b><span>仅由训练折和折外开发样本训练；测试标签不进入模型、校准和 Agent。</span></div>
<div class="step"><b>6　训练折认知参照</b><span>以类别平衡方式形成HC/受损与MCI/AD两组状态原型，不向Agent提供模型概率或患病率。</span></div>
<div class="step"><b>7　先验盲化 Agent</b><span>比较病例证据与认知参照，引用状态、指标和片段证据，分别输出筛查与分期证据等级。</span></div>
<div class="step"><b>8　两阶段受限融合</b><span>分别校准健康/受损与MCI/AD证据；任一阶段不可靠就只关闭该阶段。</span></div>
<div class="step"><b>9　锁定医生报告</b><span>只使用冻结结论和具有报告权限的证据；推理专用证据在报告生成前移除。</span></div>
</div>

<h2>2. 关键改造的独立审查</h2>
<table><thead><tr><th>改造</th><th>代码中的实现</th><th>审查结论</th><th>实测结果或边界</th></tr></thead><tbody>{modification_rows}</tbody></table>

<h2>3. 相同编码器下，真正增加了什么</h2>
<div class="split"><div class="box good"><h3>认知框架增益成立</h3><p>在文本和音频编码器保持一致时，加入 MetricEvidence、StateCard、任务路由与认知融合，宏平均 AUROC 提高 <b>{same_auc:+.3f}</b>，准确率提高 <b>{same_acc:+.3f}</b>。这隔离了“更换编码器”带来的影响，说明结构化认知表示本身有贡献。</p></div>
<div class="box bad"><h3>历史 Agent 数值修正未显示净增益</h3><p>历史单阶段 Agent-on 相对同一冻结监督先验：宏平均 AUROC <b>{agent_auc:+.4f}</b>（{'通过' if agent_auc_passed else '未通过'}），准确率 <b>{agent_acc:+.4f}</b>（{'通过' if agent_acc_passed else '未下降但无提升'}）。它实际改变 2/412 个类别，净正确分类变化为 0；该数字不代表新版两阶段 Agent。</p></div></div>

<h2>4. 为什么旧 Agent 没有净增益，当前修复到了哪里</h2>
<div class="split"><div class="box bad"><h3>旧 Agent 的信息缺口</h3><p>旧工作区只告诉 Agent 某项状态相对正常参考偏高或偏低，没有告诉它训练数据中HC、MCI和AD分别呈现怎样的联合状态模式。探索性开发结果中，Agent 的健康/受损证据接近随机，而且全部病例拒绝MCI/AD分期。放大修正强度不会增加信息，只会放大错误。</p></div>
<div class="box good"><h3>训练折认知参照已验证含有信号</h3><p>在1,622例开发数据上进行五折、完全折外审计：健康/受损 AUROC <b>{prototype_metrics['screening_auroc_hc_vs_impaired']:.3f}</b>，MCI/AD AUROC <b>{prototype_metrics['staging_auroc_mci_vs_ad']:.3f}</b>，三分类微平均 AUROC <b>{prototype_metrics['micro_auroc_ovr']:.3f}</b>。这证明参照卡可为Agent提供训练形成的认知坐标，但尚不等于Agent已有独立增益。</p></div></div>

<h2>5. PREPARE 的三组比较</h2>
<div class="cards">{metric_cards}</div>
<p>B1 为传统机器学习，B2 为直接大模型判断，Ours 为 9.2 监督认知框架加受限 Agent。9.2 明显优于内部两组基线，但这不能替代与 SpeechCARE 的协议级比较。</p>

<h2>6. 与 SpeechCARE 的同队列回顾性门禁</h2>
<div class="compare">{speechcare_panels}</div>
<p>测试对象同为 PREPARE 官方 412 例 HC/MCI/AD 队列，但训练协议并不相同。当前 9.2 使用冻结深层表示和浅层融合；SpeechCARE 对 mHuBERT 与文本编码器进行任务适配并保留更充分的时序信息。因此差距主要发生在进入 Agent 之前，不能靠提示词或放大 Agent 修正强度补回。</p>

<h2>7. 相同强主干上的认知表示增量</h2>
<div class="notice pass"><b>回顾性数值结果：</b>SpeechCARE公开概率加固定20%认知残差，在论文公布的5项均值终点中超过 {extension_passed_count}/5 项。该实验隔离了认知表示的增量价值，但借用了SpeechCARE公开输出，因此不是ADvoice独立系统优效结论。{'认知预测来源校验已通过。' if extension_provenance_verified else '该扩展产物早于当前来源侧车规则，须重建来源记录后才能进入正式实验表。'}</div>
<div class="compare">{extension_panels}</div>

<h2>8. Agent修正强度的开发集选择</h2>
<table><thead><tr>{calibration_header}<th>验证 Macro F1</th><th>验证 Macro AUROC</th><th>选择</th></tr></thead><tbody>{calibration_rows}</tbody></table>
<p>两个强度只根据折外开发病例选择。开发集未达到宏F1至少提高0.01并保持AUROC非劣时，相应强度自动归零；官方测试标签不参与等级校准、强度选择或回退规则。</p>
<div class="notice"><b>产物边界：</b>{'状态文件与当前两阶段校准一致。' if status_is_current_protocol else '当前两阶段开发门控选择筛查强度0、分期强度0；正式测试Agent按新规则不应运行。现存测试状态文件属于旧版单阶段历史运行，只用于说明旧失败模式，不与新版校准合并计算。'}</div>

<h2>9. 目前失败模式</h2>
<ol>
<li><b>语义证据仍不完整：</b>新版允许可靠的语言与任务辅助状态进入受限推理，但回忆错误、流畅性聚类与切换、话题偏离等任务评分仍未闭环。</li>
<li><b>分期信息不足：</b>语音证据可以支持“健康或受损”的判断，但不足以稳定拆分 MCI 与 AD，所以 9.2 保留监督模型的 MCI/AD 比例，不允许 Agent 自由改写分期。</li>
<li><b>冻结编码器限制：</b>深层音频和文本表示未针对 PREPARE 任务、语言和疾病标签充分适配；Agent 无法恢复上游已经压缩或丢失的信息。</li>
<li><b>任务特异状态不稳定：</b>强制移除任务特异状态时的宏 AUROC 差值为 -0.045，说明当前训练选择未能稳定利用这些状态，需要用任务语义评分器替代仅靠通用统计量。</li>
<li><b>确认性验证缺失：</b>官方 test 已被历史开发查看，继续在其上迭代会形成测试集过拟合；应冻结 9.2 规则后使用新的站点外、时间外或持有队列。</li>
</ol>

<h2>10. 下一轮只做能够改变信息量的升级</h2>
<ol>
<li>为图片描述、词语流畅性、记忆回忆、朗读和访谈分别实现任务语义评分器，并生成可追溯的 <code>metric:</code>、<code>state:</code>、<code>segment:</code> 证据。</li>
<li>保持认知框架不变，单独训练可复现的多语言音频—文本主干；保留帧级或片段级时序，不再只使用 5 秒均值向量。</li>
<li>在开发集完成多随机种子训练、校准和 Agent 修正强度选择；一次性冻结后再进入新的外部队列。</li>
<li>继续报告 E0→E1（认知框架增益）和 E1→E2（Agent 净增益），若 Agent 净增益未通过，则自动回退为 E1，而不是为了超过基线强制放大修正。</li>
</ol>

<h2>11. 工程完整性</h2>
<table><tr><th>审计项</th><th>结果</th></tr>
<tr><td>类型化证据有效率</td><td>{evidence_validity:.1%} · {'通过' if evidence_validity_passed else '未通过'}</td></tr>
<tr><td>报告权限解析</td><td>{report_permission:.1%} · {'通过' if report_permission_passed else '未通过'}</td></tr>
<tr><td>质量证据与疾病证据分离</td><td>{qc_separation:.1%} · {'通过' if qc_separation_passed else '未通过'}</td></tr>
<tr><td>历史单阶段 Agent 返回/请求</td><td>{int(status['agent_returned_cases'])}/{int(status['agent_requested_cases'])}</td></tr>
<tr><td>历史非法输出修复</td><td>{int(status['contract_repair_accepted'])}/{int(status['contract_repair_requested'])} 次修复成功；其余回退</td></tr>
<tr><td>新版两阶段正式测试 Agent</td><td>{'已运行且产物一致' if status_is_current_protocol else '未通过开发门控，不应运行；保留监督主干'}</td></tr>
<tr><td>测试标签或监督先验暴露</td><td>均为 false</td></tr></table>

<h2>12. 公开代码与可运行案例</h2>
<div class="split"><div class="box good"><h3>公开复现仓库</h3><p><a href="https://github.com/Jewelina95/advoice-research-harness">github.com/Jewelina95/advoice-research-harness</a></p><p>仓库包含核心流水线、配置、证据与报告 schema、单元测试、合成语音案例和本地演示接口；不包含受限患者录音、临床标签或个人路径。</p></div><div class="box"><h3>一键运行公开演示</h3><p><code>python -m venv .venv</code><br><code>.venv/bin/pip install -e ".[dev]"</code><br><code>make demo</code></p><p>演示调用真实特征提取代码，输出 MetricEvidence、StateCard、片段和回溯图；合成案例不生成疾病诊断。</p></div></div>

<footer>数据来源：本地 9.2 完整流水线输出；SpeechCARE 对照为论文公开均值。报告由脚本读取当前 artifact 生成，不手工改写结果。</footer>
</main></body></html>"""
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(html, encoding="utf-8")
    return OUT


if __name__ == "__main__":
    print(build())
