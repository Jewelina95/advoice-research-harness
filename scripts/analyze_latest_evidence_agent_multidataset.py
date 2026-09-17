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
    rows=""; mechanisms=""
    for result in results:
        best_accuracy=max(row["accuracy"] for row in result["summary"])
        best_f1=max(row["macro_f1"] for row in result["summary"])
        for i,row in enumerate(result["summary"]):
            winner=row["accuracy"]==best_accuracy and row["macro_f1"]==best_f1
            rows += f"<tr class={'best' if winner else ''}><td>{result['dataset'] if i==0 else ''}</td><td>{row['label']}</td><td>{row['accuracy']:.3f}</td><td>{row['balanced_accuracy']:.3f}</td><td>{row['macro_f1']:.3f}</td><td>{row['log_loss']:.3f}</td></tr>"
        mechanisms += f"<tr><td>{result['dataset']}</td><td>{result['n']}</td><td>{result['changed']}</td><td>{result['helped']}</td><td>{result['harmed']}</td><td>{result['state_gate_rate']:.0%}</td><td>{result['agent_gate_rate']:.0%}</td><td>{result['staging_gate_rate']:.0%}</td><td>{result['conflict_rate']:.0%}</td></tr>"
    html=f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Evidence Agent 多数据集完整分析</title><style>body{{margin:0;background:#f5f7f6;color:#172129;font-family:Arial,"PingFang SC",sans-serif}}main{{max-width:1160px;margin:auto;padding:42px 26px 70px}}h1{{font-size:34px}}p{{line-height:1.72;color:#40505a}}table{{width:100%;border-collapse:collapse;background:white;margin:14px 0 28px}}th,td{{padding:11px 9px;border-bottom:1px solid #d9e0e4;text-align:right}}th:first-child,td:first-child,th:nth-child(2),td:nth-child(2){{text-align:left}}thead{{background:#e6edf0}}.best{{background:#e0f0e7;font-weight:700}}.call{{padding:18px;background:#e0f0e7;border-left:6px solid #277357;font-size:17px}}.warn{{background:#fff1d9;border-left-color:#bd791c}}code{{font-family:ui-monospace,monospace}}</style></head><body><main><h1>最新版 Evidence Agent：多数据集五组消融与机制审计</h1><p>全部结果由 GitHub main 的 <code>fuse_authority_joint()</code> 从真实 case audit 重建。每个数据集内部共享完全相同的监督概率、pre/post StateCard、盲态 Agent ordinal evidence 与类别顺序。</p><div class="call warn">这是快速机制 pilot：PREPARE 9例、ADReSS 6例、另外两组各3例。它适合检查实现和病例级作用，不足以估计总体性能或宣称超过 SpeechCARE。</div><h2>五组性能</h2><table><thead><tr><th>数据集</th><th>方法</th><th>Accuracy</th><th>Balanced Acc.</th><th>Macro-F1</th><th>Log loss ↓</th></tr></thead><tbody>{rows}</tbody></table><h2>完整联合改变了哪些病例</h2><table><thead><tr><th>数据集</th><th>n</th><th>改变</th><th>修正</th><th>伤害</th><th>状态门控</th><th>Agent门控</th><th>分期门控</th><th>状态-Agent冲突</th></tr></thead><tbody>{mechanisms}</tbody></table><div class="call"><b>选择结论：</b>完整联合是下一阶段候选架构，因为它保留监督先验，并把状态变化、Agent screening 与 staging 分开门控。PREPARE pilot 显示三者互补；其他数据集用于暴露门控失效和语言任务迁移风险。是否正式选用仍必须由扩大后的锁定验证决定，而不是把绿色行预设为赢家。</div><h2>为什么某些数据集不会提升</h2><p>完整联合只有在 Agent 修正监督错误多于破坏正确病例时才会提高 Accuracy。状态门控低说明 pre/post StateCard 没形成足够强的反证；Agent 门控低说明盲态判断没有跨过不确定性和反证阈值；冲突率高说明状态变化与 Agent ordinal evidence 指向不同类别。小样本中一个病例就会改变 11%–33% 的 Accuracy，因此必须同时看 helped/harmed，而不能只看最终百分比。</p><h2>方法优势与劣势</h2><p><b>仅监督：</b>稳定、便宜，但缺少病例级认知修订。<b>仅 Agent：</b>可读取复杂证据，但缺少数据集监督先验。<b>监督+状态：</b>可解释且可回溯，但受状态构建质量约束。<b>监督+Agent：</b>能补充边界判断，但容易受语言和任务迁移影响。<b>完整联合：</b>表达能力最完整，也最依赖严格门控、独立校准和证据合法性。</p></main></body></html>'''
    (output/"report.html").write_text(html)

if __name__ == "__main__": main()
