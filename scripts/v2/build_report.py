#!/usr/bin/env python3
"""Build the v2 progress report (HTML, self-contained) from result JSONs and figures."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

V2, FIG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
mc = json.loads((V2 / "model_compare/model_compare.json").read_text()) if (V2 / "model_compare/model_compare.json").exists() else {"models": {}}
th = json.loads((V2 / "theory.json").read_text())
ct = json.loads((V2 / "contamination/summary.json").read_text())
p3 = json.loads((V2 / "prepare_attn.json").read_text())


def img(path: Path) -> str:
    if not path.exists():
        return '<p class="muted">（图待生成）</p>'
    return f'<img src="data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}" alt="{path.stem}">'


def f(x, d=3):
    return "—" if x is None else f"{x:.{d}f}"


rows = [("A 声学", p3["A_acoustic"]), ("A + 文本指标", p3["A_plus_text"]), ("冻结 embedding + 人口学", p3["E_plus_demo"]),
        ("S 框架 + 分层常模", p3["S_framework_z"]), ("S + Whisper", p3["S_framework_z+w32"]),
        ("S + Whisper（集成 + OOF 偏置）", p3["S_framework_z+w32__ens_acc_offset"])]
arm_rows = "".join(f"<tr><td>{n}</td><td>{f(m['acc'])}</td><td>{f(m['macro_f1'])}</td><td>{f(m['micro_auc'])}</td><td>{f(m['log_loss'])}</td></tr>" for n, m in rows)
arm_rows += "<tr class='ref'><td>SpeechCARE 论文均值</td><td>0.721</td><td>—</td><td>0.868</td><td>—</td></tr>"

mrows = ""
for name, e in mc.get("models", {}).items():
    if e == "pending":
        mrows += f"<tr><td>{name}</td><td colspan=6 class='muted'>运行中</td></tr>"
        continue
    b, c = e["B_plain_agent"], e["C_framework_agent"]
    mrows += (f"<tr><td>{name}</td><td>{f(b['acc'])}</td><td>{f(b['micro_auc'])}</td><td>{f(c['acc'])}</td><td>{f(c['micro_auc'])}</td>"
              f"<td>{f(100 * e.get('B_flip_rate', float('nan')), 1)}%</td><td>{f(100 * e.get('C_flip_rate', float('nan')), 1)}%</td></tr>")
s0 = mc.get("S_framework_no_agent")
s_line = f"不含 Agent 的框架（S）：Accuracy {f(s0['acc'])}，micro-AUC {f(s0['micro_auc'])}。" if s0 else ""

dr, rd = th["drift"], th["redundancy"]
lc = {}
for r in th["learning_curve"]:
    lc.setdefault(r["n_train"], {})[r["arm"]] = r["macro_auc"]
lc_rows = "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{f(v.get(a))}</td>" for a in ("A_acoustic", "E_embeddings", "S_framework_states", "S_full")) + "</tr>" for n, v in sorted(lc.items()))
par_rows = "".join(f"<tr><td>{p['n_groups']}</td><td>{p['added']}</td><td>{f(p['oof_logloss'], 4)}</td><td>{f(p['test_macro_auc'])}</td></tr>" for p in th["pareto"])
a_, p_ = ct["ADReSS_public_CHAT"], ct["PREPARE_ASR_control"]

html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADvoice v2 进展汇报</title><style>
:root{{--bg:#fbfbfa;--ink:#1d2327;--muted:#5f6b72;--line:#dde2e5;--accent:#8a2f2f;--card:#fff}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16191b;--ink:#e6e9eb;--muted:#9aa5ab;--line:#2c3236;--accent:#e08a8a;--card:#1d2124}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.65 -apple-system,"PingFang SC",Arial,sans-serif}}
main{{max-width:980px;margin:auto;padding:32px 16px 80px}}h1{{font-size:28px;margin:0 0 6px}}h2{{margin-top:40px;font-size:20px;border-bottom:1px solid var(--line);padding-bottom:6px}}
.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;margin:12px 0 20px;font-variant-numeric:tabular-nums;background:var(--card)}}
th,td{{padding:7px 8px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}
tr.ref td{{font-style:italic;color:var(--muted)}}.wrap{{overflow-x:auto}}img{{max-width:100%;background:#fff;border:1px solid var(--line);margin:8px 0}}
.k{{display:grid;grid-template-columns:repeat(auto-fit,minmax(200px,1fr));gap:10px;margin:16px 0}}.k div{{background:var(--card);border:1px solid var(--line);padding:12px}}.k b{{display:block;font-size:22px}}
.note{{border-left:4px solid var(--accent);padding:8px 12px;background:var(--card)}}</style></head><body><main>
<h1>ADvoice v2：三家模型对比与证据治理验证</h1>
<p class="muted">2026-10-08 · PREPARE 官方 412 例测试集（回顾性：该测试集历史上被多次查看）· 所有选择只在训练集 OOF 上完成</p>
<div class="k"><div><b>0.869</b>micro-AUC，SpeechCARE 0.868</div><div><b>0.70</b>Accuracy，SpeechCARE 0.721</div>
<div><b>{f(100*dr['B_plain_agent_flip_rate'],1)}% → {f(100*dr['C_framework_flip_rate'],1)}%</b>换措辞后的结论翻转（GPT，纯 Agent → 框架）</div><div><b>{rd['effective_rank']}</b>55 个指标的有效维数</div></div>

<h2>1. 三个条件：A 传统声学、B 纯 Agent、C 框架</h2>
<div class="wrap"><table><tr><th>方案</th><th>Accuracy</th><th>macro-F1</th><th>micro-AUC</th><th>log-loss</th></tr>{arm_rows}</table></div>
<p>结论：C 相对 A 提高约 11 个点；AUC 与 SpeechCARE 持平，Accuracy 仍低约 2 个点，差距来自编码器未微调。</p>

<h2>2. GPT / Claude / DeepSeek 在同一框架下</h2>
<p>{s_line}三家使用同一提示、同一 schema、同一推理强度；B 只看转录直接给概率，C 为框架 + Agent 测量 + Agent 判断堆叠。</p>
<div class="wrap"><table><tr><th>模型</th><th>B Acc</th><th>B AUC</th><th>C Acc</th><th>C AUC</th><th>B 翻转</th><th>C 翻转</th></tr>{mrows}</table></div>
{img(V2 / "model_compare/Fig_model_compare.png")}

<h2>3. 老师的问题</h2>
<h3>Q1 稳定性（语言漂移）</h3><p>同一 120 例、只改提示措辞：纯 Agent 结论翻转 {f(100*dr['B_plain_agent_flip_rate'],1)}%，概率平均变化 {f(dr['B_mean_abs_prob_change'])}；框架 {f(100*dr['C_framework_flip_rate'],1)}%，{f(dr['C_mean_abs_prob_change'],4)}。测量重测一致性（Spearman）：{", ".join(f"{k} {v}" for k, v in dr['measure_test_retest_spearman'].items())}。</p>
<h3>Q2/Q3 证据冗余与“为什么这么多 feature”</h3><p>{rd['n_metrics']} 个手工指标中 {rd['n_constant']} 个在训练集为常数（{", ".join(rd['constant'])}），{rd['n_pairs_rho_ge_0.9']} 对 |ρ|≥0.9，有效维数 {rd['effective_rank']}。按训练 OOF log-loss 逐组加入证据：</p>
<div class="wrap"><table><tr><th>组数</th><th>加入</th><th>OOF log-loss</th><th>测试 macro-AUC</th></tr>{par_rows}</table></div>
<p>OOF 最优在第 4 组；之后加入的组不再改善训练内指标，即“最小充分证据集”约 4 组。</p>
<h3>Q4 底层模型是否见过数据</h3><p>续写探针（GPT-5.5）：公开 ADReSS 转录与真实后文的 4-gram 重合 {a_['mean_4gram_overlap_true']}，与其他病例 {a_['mean_4gram_overlap_other_case']}，无记忆迹象。PREPARE 对照组较高（{p_['mean_4gram_overlap_true']} vs {p_['mean_4gram_overlap_other_case']}）来自朗读任务的固定文本。阴性结果不能证明模型从未见过数据。</p>
<h3>Q5 少样本</h3><div class="wrap"><table><tr><th>训练例数</th><th>A 声学</th><th>E embedding</th><th>S 状态</th><th>S 全部</th></tr>{lc_rows}</table></div>
<p>32 例时状态模型略优且方差更小；64 例以上 embedding 反超；完整系统在各样本量下不低于 embedding。</p>
<h3>Agent 增益</h3><p class="note">在 PREPARE（ASR 转录、约 30 秒）上，LLM 测量对准确率无显著增益，信息已被文本 embedding 覆盖；Agent 的可测价值在稳定性和可核验性。增益检验应转到人工转录、较长录音的数据集（ADReSS、Pitt、PROCESS-2）。</p>

<h2>4. 论文图</h2>{img(FIG / "Fig1_architecture.png")}{img(FIG / "Fig3_results.png")}{img(FIG / "Fig4_governance.png")}
</main></body></html>"""
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "index.html").write_text(html, encoding="utf-8")
print("wrote", OUT / "index.html")
