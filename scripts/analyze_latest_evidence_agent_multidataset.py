#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, log_loss

from advoice.authority_joint_fusion import AuthorityJointFusionConfig, fuse_authority_joint

HOME = Path("/Users/wenshaoyue")
VOICE = HOME / "Desktop/research/ad general/AD voice"
AUTH = HOME / ".config/superpowers/worktrees/advoice-research-harness/agent-led-multichannel/.local"
SPECS = {
    "PREPARE": ([AUTH / "authority_expanded_pilot_9/PREPARE_DrivenData/case_audit.json"], VOICE / "9.2/artifacts/PREPARE_DrivenData/condition_c_base_predictions.csv"),
    "ADReSS 2020": ([AUTH / "authority_joint_pilot_v1/ADReSS_2020/case_audit.json", AUTH / "authority_joint_holdout_v1/ADReSS_2020/case_audit.json"], VOICE / "8.27/artifacts/ADReSS_2020/condition_c_base_predictions.csv"),
    "ADReSSo diagnosis": ([AUTH / "authority_expanded_pilot/ADReSSo_2021_diagnosis/case_audit.json"], VOICE / "8.27/artifacts/ADReSSo_2021_diagnosis/condition_c_base_predictions.csv"),
    "IAEAV": ([AUTH / "authority_alternative_pilot/IAEAV/case_audit.json"], VOICE / "8.27/artifacts/IAEAV/condition_c_base_predictions.csv"),
    "PROCESS-2": ([AUTH / "authority_alternative_pilot/PROCESS_2/case_audit.json"], VOICE / "8.27/artifacts/PROCESS_2/condition_c_base_predictions.csv"),
    "DementiaNet": ([AUTH / "authority_alternative_pilot/DementiaNet_PublicFigures/case_audit.json"], VOICE / "8.27/artifacts/DementiaNet_PublicFigures/condition_c_base_predictions.csv"),
    "NCMMSC2021-AD": ([AUTH / "authority_expanded_pilot_v3/NCMMSC2021_AD/case_audit.json"], VOICE / "8.27/artifacts/NCMMSC2021_AD/condition_c_base_predictions.csv"),
}
ARMS = {"supervised":"仅监督", "agent_only":"仅 Agent", "supervised_state":"监督+状态", "supervised_agent":"监督+Agent", "full_joint":"完整联合"}

def config(state: float, agent: float, staging: float, gated: bool = True):
    return AuthorityJointFusionConfig(state_strength=state, agent_strength=agent, staging_strength=staging, max_abs_state_delta=.75, ordinal_temperature=1.0, conflict_aware_gating=gated, min_state_uncertainty=.65, min_frozen_uncertainty=.75, min_counterevidence_margin=.75)

def evaluate(name: str, audits: list[Path], label_path: Path) -> dict:
    cases = []
    for path in audits: cases.extend(json.loads(path.read_text())["cases"])
    cases = list({case["case_id"]: case for case in cases}.values())
    labels = pd.read_csv(label_path).set_index("subject_id")["label"].astype(str).to_dict()
    order = list(cases[0]["prepared"]["class_order"]); index = {label:i for i,label in enumerate(order)}
    y=[]; probability={key:[] for key in ARMS}; gates=[]
    case_rows=[]
    for case in cases:
        y.append(index[labels[case["case_id"]]])
        frozen=case["frozen"]["probabilities"]; pre=case["pre_state"]["probabilities"]; post=case["post_state"]["probabilities"]; ordinal=case["fusion"]["blind_ordinal_scores"]
        probability["supervised"].append([frozen[x] for x in order])
        uniform={x:1/len(order) for x in order}
        variants={
            "agent_only": fuse_authority_joint(uniform,uniform,uniform,ordinal,class_order=order,config=config(0,1,1,False)),
            "supervised_state": fuse_authority_joint(frozen,pre,post,ordinal,class_order=order,config=config(1,0,0)),
            "supervised_agent": fuse_authority_joint(frozen,pre,post,ordinal,class_order=order,config=config(0,1,1)),
            "full_joint": fuse_authority_joint(frozen,pre,post,ordinal,class_order=order,config=config(1,1,1)),
        }
        for arm,result in variants.items(): probability[arm].append([result.fused_probabilities[x] for x in order])
        full=variants["full_joint"]; gates.append((full.state_authority_gate,full.agent_authority_gate,full.staging_authority_gate,full.state_agent_conflict))
        case_rows.append({"case_id":case["case_id"],"truth":labels[case["case_id"]],"supervised":max(frozen,key=frozen.get),"full_joint":full.predicted_label})
    y=np.asarray(y); summary=[]
    for arm,values in probability.items():
        p=np.asarray(values); pred=p.argmax(1)
        summary.append({"arm":arm,"label":ARMS[arm],"accuracy":float(accuracy_score(y,pred)),"balanced_accuracy":float(balanced_accuracy_score(y,pred)),"macro_f1":float(f1_score(y,pred,average="macro",zero_division=0)),"log_loss":float(log_loss(y,p,labels=list(range(len(order)))))})
    helped=harmed=changed=0
    for row in case_rows:
        if row["supervised"] != row["full_joint"]:
            changed += 1
            helped += row["supervised"] != row["truth"] and row["full_joint"] == row["truth"]
            harmed += row["supervised"] == row["truth"] and row["full_joint"] != row["truth"]
    g=np.asarray(gates,dtype=float)
    return {"dataset":name,"n":len(y),"classes":order,"summary":summary,"changed":changed,"helped":int(helped),"harmed":int(harmed),"state_gate_rate":float(np.mean(g[:,0]>0)),"agent_gate_rate":float(np.mean(g[:,1]>0)),"staging_gate_rate":float(np.mean(g[:,2]>0)),"conflict_rate":float(np.mean(g[:,3]>0)),"cases":case_rows}

def main():
    output=Path("reports/latest_evidence_agent_multidataset_2026-09-17"); output.mkdir(parents=True,exist_ok=True)
    results=[evaluate(name,*spec) for name,spec in SPECS.items()]
    (output/"results.json").write_text(json.dumps(results,ensure_ascii=False,indent=2))
    rows=""; mechanisms=""; deltas=""; case_sections=""
    interpretations = {
        "PREPARE": "联合模型修正了 1 个 HC 假阳性和 1 个 AD 假阴性，未破坏原有正确病例；这是当前唯一出现净病例级增益的数据集。",
        "ADReSS 2020": "监督模型仍漏判 S081，但状态、Agent 和分期门控均未启动。这里不是联合模型效果等同于监督模型的证据，而是修正召回率不足。",
        "ADReSSo diagnosis": "状态门控在 1/3 病例启动并改善概率校准，但没有越过分类边界；Agent 本身没有获得决策权限。",
        "IAEAV": "三个病例的状态变化与 Agent 判断全部冲突，冲突门控将修正归零。它避免了不受支持的覆盖，但也保留了 iaeav-inv01-002 的监督错误。",
        "PROCESS-2": "PROCESS-2_rec__372 被监督模型判为 MCI、真实为 AD；Agent 没有形成足够一致的状态证据来纠正该边界错误。",
        "DementiaNet": "三个抽样病例真实标签均为 AD，类别覆盖不足，不能解释 balanced accuracy 或泛化能力；Jonathan Miller 的错误未被修正。",
        "NCMMSC2021-AD": "唯一一次类别改变把 MCI 错误改成 HC，仍未命中真实 AD。该结果保留为中文通道失败模式，不用于模型选择。",
    }
    for result in results:
        by_arm={row["arm"]:row for row in result["summary"]}
        baseline=by_arm["supervised"]; full=by_arm["full_joint"]
        best_accuracy=max(row["accuracy"] for row in result["summary"])
        best_f1=max(row["macro_f1"] for row in result["summary"])
        for i,row in enumerate(result["summary"]):
            winner=row["accuracy"]==best_accuracy and row["macro_f1"]==best_f1
            rows += f"<tr class={'best' if winner else ''}><td>{result['dataset'] if i==0 else ''}</td><td>{row['label']}</td><td>{row['accuracy']:.3f}</td><td>{row['balanced_accuracy']:.3f}</td><td>{row['macro_f1']:.3f}</td><td>{row['log_loss']:.3f}</td></tr>"
        mechanisms += f"<tr><td>{result['dataset']}</td><td>{result['n']}</td><td>{result['changed']}</td><td>{result['helped']}</td><td>{result['harmed']}</td><td>{result['state_gate_rate']:.0%}</td><td>{result['agent_gate_rate']:.0%}</td><td>{result['staging_gate_rate']:.0%}</td><td>{result['conflict_rate']:.0%}</td></tr>"
        da=full["accuracy"]-baseline["accuracy"]
        df=full["macro_f1"]-baseline["macro_f1"]
        dll=full["log_loss"]-baseline["log_loss"]
        status="净改善" if result["helped"]>result["harmed"] else "净伤害" if result["harmed"]>result["helped"] else "无类别净变化"
        deltas += f"<tr><td>{result['dataset']}</td><td>{result['n']}</td><td class={'up' if da>0 else 'down' if da<0 else ''}>{da:+.3f}</td><td class={'up' if df>0 else 'down' if df<0 else ''}>{df:+.3f}</td><td class={'up' if dll<0 else 'down' if dll>0 else ''}>{dll:+.3f}</td><td>{status}</td></tr>"
        changed=[row for row in result["cases"] if row["supervised"] != row["full_joint"]]
        unchanged_errors=[row for row in result["cases"] if row["supervised"] == row["full_joint"] and row["supervised"] != row["truth"]]
        changed_text="；".join(f"{row['case_id']}: {row['supervised']} → {row['full_joint']}（真实 {row['truth']}）" for row in changed) or "无类别改变"
        missed_text="；".join(f"{row['case_id']}: 保持 {row['supervised']}（真实 {row['truth']}）" for row in unchanged_errors) or "本批次没有未修正错误"
        case_sections += f"<article><div class='article-head'><h3>{result['dataset']}</h3><span>n={result['n']}</span></div><p>{interpretations[result['dataset']]}</p><dl><dt>实际改变</dt><dd>{changed_text}</dd><dt>仍未修正</dt><dd>{missed_text}</dd></dl></article>"
    total=sum(result["n"] for result in results)
    improved=sum(result["helped"]>result["harmed"] for result in results)
    harmed=sum(result["harmed"] for result in results)
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Evidence Agent 多数据集完整分析</title><style>:root{{--ink:#172129;--muted:#52616b;--line:#d8e0e3;--green:#17664d;--green-bg:#e5f1eb;--amber:#9b5c0b;--amber-bg:#fff2db;--red:#a13d35;--red-bg:#fae9e7}}*{{box-sizing:border-box}}body{{margin:0;background:#f4f6f5;color:var(--ink);font-family:Arial,"PingFang SC",sans-serif}}main{{max-width:1220px;margin:auto;padding:44px 28px 80px}}h1{{font-size:36px;line-height:1.2;margin:0 0 12px}}h2{{margin-top:38px;font-size:24px}}h3{{margin:0;font-size:20px}}p,dd{{line-height:1.7;color:var(--muted)}}.lede{{max-width:980px;font-size:17px}}.stats{{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:26px 0}}.stat{{background:#fff;border:1px solid var(--line);padding:18px}}.stat b{{display:block;font-size:29px;margin-bottom:5px}}.stat span{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;background:white;margin:14px 0 28px;font-variant-numeric:tabular-nums}}th,td{{padding:11px 9px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}thead{{background:#e7ecee}}.best{{background:var(--green-bg);font-weight:700}}.up{{color:var(--green);font-weight:700}}.down{{color:var(--red);font-weight:700}}.call{{padding:18px 20px;background:var(--green-bg);border-left:6px solid var(--green);font-size:16px;line-height:1.65}}.warn{{background:var(--amber-bg);border-left-color:var(--amber)}}.danger{{background:var(--red-bg);border-left-color:var(--red)}}code{{font-family:ui-monospace,monospace}}.cases{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:14px}}article{{background:#fff;border:1px solid var(--line);padding:20px}}.article-head{{display:flex;align-items:center;justify-content:space-between;gap:20px}}.article-head span{{color:var(--muted);font-weight:700}}dl{{display:grid;grid-template-columns:92px 1fr;gap:8px 12px;margin-bottom:0}}dt{{font-weight:700}}dd{{margin:0}}@media(max-width:800px){{.stats,.cases{{grid-template-columns:1fr}}main{{padding:28px 16px}}.table-wrap{{overflow:auto}}}}</style></head><body><main><h1>Evidence Agent 多数据集五组消融与病例机制审计</h1><p class="lede">结果由当前仓库的 <code>fuse_authority_joint()</code> 从真实 case audit 重建。每个数据集内部五组方法共享相同的监督概率、pre/post StateCard、盲态 Agent ordinal evidence 与类别顺序，因此组内差异来自融合机制，而不是更换输入。</p><div class="stats"><div class="stat"><b>7</b><span>数据集</span></div><div class="stat"><b>{total}</b><span>真实缓存病例</span></div><div class="stat"><b>{improved}</b><span>出现净类别改善的数据集</span></div><div class="stat"><b>{harmed}</b><span>被联合模型破坏的正确病例</span></div></div><div class="call warn"><b>范围限制：</b>这是多通道机制 pilot，不是总体性能试验。PREPARE 为 9 例、ADReSS 为 6 例，其余各 3 例。三例数据集中一个病例对应 33.3 个百分点；DementiaNet 抽样还只有 AD 标签，不能据此估计跨类别泛化。</div><h2>联合模型相对监督模型的净变化</h2><div class="table-wrap"><table><thead><tr><th>数据集</th><th>n</th><th>Δ Accuracy</th><th>Δ Macro-F1</th><th>Δ Log loss ↓</th><th>病例结论</th></tr></thead><tbody>{deltas}</tbody></table></div><h2>五组性能</h2><div class="table-wrap"><table><thead><tr><th>数据集</th><th>方法</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro-F1</th><th>Log loss ↓</th></tr></thead><tbody>{rows}</tbody></table></div><h2>门控是否实际工作</h2><div class="table-wrap"><table><thead><tr><th>数据集</th><th>n</th><th>改变</th><th>修正</th><th>伤害</th><th>状态门控</th><th>Agent门控</th><th>分期门控</th><th>状态-Agent冲突</th></tr></thead><tbody>{mechanisms}</tbody></table></div><div class="call"><b>核心判断：</b>当前完整联合模型在 PREPARE pilot 中有净增益，并在其余新通道中没有破坏原有正确病例；但 IAEAV、PROCESS-2、DementiaNet 和 ADReSS 的错误病例也没有被修正。当前瓶颈已经从“Agent 权重过大”转为“门控召回率不足”。这还不能支持跨数据集优于监督模型的结论。</div><h2>逐数据集病例解释</h2><section class="cases">{case_sections}</section><h2>为什么 NCMMSC 看起来特别差</h2><div class="call danger">NCMMSC 的三例分别覆盖 HC、MCI、AD。监督模型只判对 HC；联合模型把一例错误从 MCI 改为 HC，但真实标签是 AD，因此属于“错误类别迁移”，不是修正。与此同时，中文通道状态-Agent 冲突率为 66.7%，Agent 门控为 0%。结果提示中文证据方向与监督先验未对齐，不能通过换数据集或提高固定 Agent 权重解决。</div><h2>下一步应验证什么</h2><p>扩大验证不应继续随机抽三例，而应按“监督正确/错误、低/高不确定性、状态-Agent一致/冲突、不同任务与语言”分层抽样。只有在锁定阈值后，联合模型的 helped 数稳定高于 harmed 数，且 log loss 与校准不恶化，才能进入全量评估。NCMMSC 必须保留为失败审计，但不应主导架构选择。</p></main></body></html>'''
    (output/"report.html").write_text(html)
    oral = f"""# 多数据集快速验证口语稿

这次没有再用 NCMMSC 的三例结果代表整个框架，而是加入 IAEAV、PROCESS-2 和 DementiaNet 三种不同数据通道。全部合计 {total} 个真实缓存病例。

结果分成两部分。PREPARE 的九例中，完整联合模型修正了两个监督错误，没有破坏正确病例，Accuracy 从 0.778 提高到 1.000。这个结果说明状态修正和 Agent 判断在特定病例上可能互补，但样本仍然很小。

另一方面，ADReSS、IAEAV、PROCESS-2 和 DementiaNet 中，联合模型都没有改变监督模型的类别。它避免了新增错误，但也没有纠正已经存在的错误。尤其 IAEAV 的状态与 Agent 判断冲突率达到 100%，因此所有修正被门控拦截。PROCESS-2 仍保留一例 AD 被判为 MCI；DementiaNet 仍保留一例 AD 被判为 HC。

NCMMSC 继续作为失败模式保留。它只有三例，联合模型发生一次类别改变，但只是从一个错误类别转到另一个错误类别。这里暴露的是中文证据方向、状态变化和 Agent 判断没有完成校准，而不是简单的模型权重不足。

因此当前最准确的结论不是“完整联合已经跨数据集更优”，而是：它在 PREPARE 小样本中显示净改善，在其他通道中表现为保守回退。下一轮需要按监督错误、模型不确定性、状态和 Agent 是否冲突来分层抽样，再锁定门控阈值。只有修正数稳定高于伤害数，才能进入全量比较。
"""
    (output/"oral_zh.md").write_text(oral)

if __name__ == "__main__": main()
