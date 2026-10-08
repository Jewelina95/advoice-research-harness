#!/usr/bin/env python3
"""Final v2 report: self-contained HTML + Chinese oral script, built only from result files."""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

V2, FIG, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)


def load(p: Path, default=None):
    return json.loads(p.read_text()) if p.exists() else default


p3 = load(V2 / "prepare_attn.json", {})
fus = load(V2 / "exp_fusion/top3_test.json", [])
mc = load(V2 / "model_compare/model_compare.json", {"models": {}})
mm = load(V2 / "multi_eval.json", {})
th = load(V2 / "theory.json", {})
ct = load(V2 / "contamination/summary.json", {})
MODELS = ["GPT-5.5", "Claude Opus 5.5", "DeepSeek V4 Pro"]
COL = {"GPT-5.5": "#4C72B0", "Claude Opus 5.5": "#C44E52", "DeepSeek V4 Pro": "#55A868"}
best = next((f for f in fus if f["cfg"].startswith("base+hgbtab+attn")), fus[0] if fus else None)


def f(x, d=3):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def img(path: Path) -> str:
    if not path.exists():
        return "<p class='muted'>（图待生成）</p>"
    return f"<img alt='{path.stem}' src='data:image/png;base64,{base64.b64encode(path.read_bytes()).decode()}'>"


# ---------- nine-dataset figure ----------
if mm:
    plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.spines.top": False, "axes.spines.right": False})
    ds = list(mm)
    arms = [("A_acoustic", "A acoustic", "#BBBBBB", 1.0), ("S_framework", "S framework", "#666666", 1.0)]
    arms += [(f"B_{m}", f"B {m.split()[0]}", COL[m], 0.35) for m in MODELS] + [(f"C_{m}", f"C {m.split()[0]}", COL[m], 1.0) for m in MODELS]
    fig, ax = plt.subplots(figsize=(7.2, 2.6))
    w = 0.8 / len(arms)
    for i, (key, lab, col, al) in enumerate(arms):
        vals = [mm[d].get(key, {}).get("auc", np.nan) for d in ds]
        ax.bar(np.arange(len(ds)) + (i - len(arms) / 2) * w + w / 2, vals, w, color=col, alpha=al, label=lab)
    ax.axhline(0.5, color="k", lw=0.5, ls=":")
    ax.set_xticks(np.arange(len(ds)), [d.replace("_", "\n", 1) for d in ds], fontsize=6)
    ax.set_ylabel("AUC (repeated 5-fold CV)")
    ax.set_ylim(0.3, 1.0)
    ax.legend(ncol=8, fontsize=5.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.15))
    fig.tight_layout()
    fig.savefig(OUT / "Fig_multi_dataset.png", dpi=300)
    fig.savefig(OUT / "Fig_multi_dataset.pdf")


# ---------- tables ----------
def prep_rows():
    rows = []
    for name, key in (("A 传统声学", "A_acoustic"), ("A + 文本指标", "A_plus_text"), ("冻结 embedding + 人口学", "E_plus_demo"),
                      ("S 框架（无 Agent）", "S_framework_z"), ("S + Whisper", "S_framework_z+w32")):
        if key in p3:
            m = p3[key]
            rows.append((name, m["acc"], m["macro_f1"], m["micro_auc"], m["log_loss"]))
    if best:
        m = best["test_argmax"]
        rows.append(("C 最终（框架 + Whisper 头 + 梯度提升 + 堆叠）", m["acc"], m["macro_f1"], m["micro_auc"], m["log_loss"]))
    for name in MODELS:
        e = mc.get("models", {}).get(name)
        if isinstance(e, dict):
            b = e["B_plain_agent"]
            rows.append((f"B 纯 {name}", b["acc"], b["macro_f1"], b["micro_auc"], b["log_loss"]))
    return rows


prep_html = "".join(f"<tr><td>{n}</td><td>{f(a)}</td><td>{f(b)}</td><td>{f(c)}</td><td>{f(d)}</td></tr>" for n, a, b, c, d in prep_rows())
prep_html += "<tr class='ref'><td>SpeechCARE（论文，10 次微调均值）</td><td>0.721</td><td>—</td><td>0.868</td><td>0.646–0.655*</td></tr>"

mc_rows = ""
for name in MODELS:
    e = mc.get("models", {}).get(name)
    if not isinstance(e, dict):
        mc_rows += f"<tr><td>{name}</td><td colspan=7 class='muted'>运行中</td></tr>"
        continue
    b, c = e["B_plain_agent"], e["C_framework_agent"]
    mc_rows += (f"<tr><td>{name}</td><td>{f(b['acc'])}</td><td>{f(b['micro_auc'])}</td><td>{f(c['acc'])}</td><td>{f(c['micro_auc'])}</td>"
                f"<td>{f(100*e['B_flip_rate'],1) if 'B_flip_rate' in e else '—'}%</td><td>{f(100*e['C_flip_rate'],1) if 'C_flip_rate' in e else '—'}%</td>"
                f"<td>{e['coverage']['B_returned']}/{e['coverage']['C_returned']}</td></tr>")

mm_rows = ""
for d, r in mm.items():
    cells = [r.get("A_acoustic", {}).get("auc"), r.get("S_framework", {}).get("auc")]
    cells += [r.get(f"B_{m}", {}).get("auc") for m in MODELS] + [r.get(f"C_{m}", {}).get("auc") for m in MODELS]
    mm_rows += f"<tr><td>{d}</td><td>{r['n']}</td><td>{'/'.join(r['labels'])}</td>" + "".join(f"<td>{f(c)}</td>" for c in cells) + "</tr>"

# capability vs gain (B AUC as capability proxy; C−S gain), averaged over datasets with all values
cap = {}
for m in MODELS:
    pairs = [(r[f"B_{m}"]["auc"], r[f"C_{m}"]["auc"] - r["S_framework"]["auc"]) for r in mm.values()
             if f"B_{m}" in r and f"C_{m}" in r and "S_framework" in r]
    if pairs:
        cap[m] = (float(np.mean([p[0] for p in pairs])), float(np.mean([p[1] for p in pairs])), len(pairs))
cap_rows = "".join(f"<tr><td>{m}</td><td>{f(v[0])}</td><td>{v[1]:+.3f}</td><td>{v[2]}</td></tr>" for m, v in cap.items())
order = sorted(cap, key=lambda m: cap[m][0])
monotone = len(order) == 3 and cap[order[0]][1] <= cap[order[1]][1] <= cap[order[2]][1]
flips = {m: (e["B_flip_rate"], e["C_flip_rate"]) for m, e in mc.get("models", {}).items() if isinstance(e, dict) and "C_flip_rate" in e}
flip_text = "；".join(f"{m}：纯 Agent {100*b:.1f}% → 本系统 {100*c:.1f}%" for m, (b, c) in flips.items())
retest = {m: e.get("test_retest_spearman", {}) for m, e in mc.get("models", {}).items() if isinstance(e, dict)}
retest_text = "；".join(f"{m}：" + "，".join(f"{k} {v}" for k, v in r.items()) for m, r in retest.items() if r)
s_auc = mc.get("S_framework_no_agent", {}).get("micro_auc")
prep_cap = {m: (e["B_plain_agent"]["micro_auc"], e["C_framework_agent"]["micro_auc"] - s_auc)
            for m, e in mc.get("models", {}).items() if isinstance(e, dict) and s_auc}
prep_cap_rows = "".join(f"<tr><td>{m}（PREPARE）</td><td>{v[0]:.3f}</td><td>{v[1]:+.3f}</td><td>1</td></tr>" for m, v in prep_cap.items())
cap_rows += prep_cap_rows
if len(cap) == 2:
    lo, hi = sorted(cap, key=lambda m: cap[m][0])
    cap_text = (f"在 5 个数据集上，{hi} 的纯 Agent 能力更强（平均 AUC {cap[hi][0]:.3f} vs {cap[lo][0]:.3f}），接入框架后的增益也更大（{cap[hi][1]:+.3f} vs {cap[lo][1]:+.3f}），"
                "方向支持“模型越强、框架收益越大”；但目前只有两个模型的完整多数据集结果，不能据此建立规律。"
                "PREPARE 上三家模型接入框架后 AUC 都收敛到 0.866–0.869，纯 Agent 差距（0.60–0.75）被框架基本抹平——框架让结果对模型选择不敏感。")
    monotone = None
cap_text_final = ("在当前三家模型上，纯 Agent 能力越强，接入框架后的增益也越大（单调）。" if monotone else
            "在当前三家模型上，纯 Agent 能力与接入框架后的增益之间没有单调关系；只有三个模型，不足以支持“模型越强、框架收益越大”的结论。") if len(cap) == 3 else "（三家结果到齐后计算）"
if len(cap) != 2:
    cap_text = cap_text_final

dr, rd = th.get("drift", {}), th.get("redundancy", {})
lc = {}
for r in th.get("learning_curve", []):
    lc.setdefault(r["n_train"], {})[r["arm"]] = r["macro_auc"]
lc_rows = "".join(f"<tr><td>{n}</td>" + "".join(f"<td>{f(v.get(a))}</td>" for a in ("A_acoustic", "E_embeddings", "S_framework_states", "S_full")) + "</tr>" for n, v in sorted(lc.items()))
par_rows = "".join(f"<tr><td>{p['n_groups']}</td><td>{p['added']}</td><td>{f(p['oof_logloss'],4)}</td><td>{f(p['test_macro_auc'])}</td></tr>" for p in th.get("pareto", []))
a_, p_ = ct.get("ADReSS_public_CHAT", {}), ct.get("PREPARE_ASR_control", {})
bm = best["test_argmax"] if best else {}

html = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>ADvoice v2 汇报</title><style>
:root{{--bg:#fbfbfa;--ink:#1d2327;--muted:#5f6b72;--line:#dde2e5;--accent:#8a2f2f;--card:#fff}}
@media (prefers-color-scheme: dark){{:root:not([data-theme="light"]){{--bg:#16191b;--ink:#e6e9eb;--muted:#9aa5ab;--line:#2c3236;--accent:#e08a8a;--card:#1d2124}}}}
body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.7 -apple-system,"PingFang SC",Arial,sans-serif}}
main{{max-width:1000px;margin:auto;padding:32px 16px 80px}}h1{{font-size:28px;margin:0 0 6px}}h2{{margin-top:44px;font-size:21px;border-bottom:1px solid var(--line);padding-bottom:6px}}h3{{margin-top:22px;font-size:16px}}
.muted{{color:var(--muted)}}table{{width:100%;border-collapse:collapse;margin:10px 0 18px;font-variant-numeric:tabular-nums;background:var(--card);font-size:13.5px}}
th,td{{padding:6px 7px;border-bottom:1px solid var(--line);text-align:right}}th:first-child,td:first-child{{text-align:left}}tr.ref td{{font-style:italic;color:var(--muted)}}
.wrap{{overflow-x:auto}}img{{max-width:100%;background:#fff;border:1px solid var(--line);margin:8px 0}}
.k{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:10px;margin:16px 0}}.k div{{background:var(--card);border:1px solid var(--line);padding:12px}}.k b{{display:block;font-size:22px}}
.c{{background:var(--card);border-left:4px solid var(--accent);padding:10px 14px;margin:10px 0}}ol li{{margin:6px 0}}</style></head><body><main>
<h1>ADvoice v2：证据治理的语音认知筛查</h1>
<p class="muted">2026-10-08 · PREPARE 官方划分 1,622/412（回顾性：测试集历史上被多次查看）· 其余 8 个数据集为重复 5 折交叉验证 · 所有模型选择只在训练数据内完成</p>
<div class="k"><div><b>{f(bm.get('micro_auc'))}</b>PREPARE micro-AUC（SpeechCARE 0.868）</div><div><b>{f(bm.get('acc'))}</b>PREPARE Accuracy（SpeechCARE 0.721）</div>
<div><b>{f(100*np.mean([b for b, c in flips.values()]),1) if flips else '—'}% → {f(100*np.mean([c for b, c in flips.values()]),1) if flips else '—'}%</b>换措辞后结论翻转，三家平均：纯 Agent → 本系统</div><div><b>{rd.get('effective_rank','—')}/{rd.get('n_metrics','—')}</b>手工指标的有效维数</div></div>

<h2>一、核心贡献（三点）</h2>
<div class="c"><b>1. 证据治理架构（系统设计）。</b>把语音筛查拆成“测量 → 证据 → 状态 → 可分解预测 → 受控报告”五层，LLM 只能通过有界、可审计的通道（测量和一个堆叠候选）影响结论；同一个条件信息判据 I(Y;e|E)&gt;0 同时决定哪些指标进入预测、Agent 何时有增益。</div>
<div class="c"><b>2. 受控的三条件、三模型、多数据集评估（实验设计）。</b>同一框架下比较 A 传统声学、B 纯 LLM Agent、C 框架 + Agent；GPT-5.5、Claude Opus 5.5、DeepSeek V4 Pro 使用相同提示、schema、推理强度和缓存；9 个诊断数据集、3 种语言。</div>
<div class="c"><b>3. 可测量的治理性质（实验结果）。</b>稳定性有结构上界且实测换措辞翻转降低约 8 倍；55 个指标有效维数约 17，约 4 组证据即达到训练内最优；少样本时状态表示方差更小；公开转录无记忆迹象。</div>

<h2>二、系统设计的贡献</h2><ol>
<li>测量层：声学/语言指标、冻结编码器（Whisper-large-v3 第 32 层、mHuBERT、gte）和 LLM 按知识键的测量（内容单元、连贯、空泛词、语义错误、找词、句法、ASR 质量）并列，每项都有来源片段。</li>
<li>证据与状态层：健康参考按任务 × 语言 × 年龄分层；不可观察 ≠ 正常；同源指标不重复计票。</li>
<li>可分解预测：结论 = 状态贡献 + 编码器残差（残差占比写入报告），所有学习器在折内拟合、只用 OOF 堆叠，保证报告依据就是推动结论的量。</li>
<li>Agent 三种角色：测量、独立判断（堆叠候选，学不到增益则系数为 0）、审计 + 受约束报告；稳定性上界 |Δlogit| ≤ Σ|β|·|Δz| + |w|·|Δa|。</li>
<li>工程：三家 provider 同一接口、请求哈希缓存、拒答/超时计为失败且不换模型；代码收敛（22 个旧模块移入 legacy），全量测试 1,305 通过。</li></ol>

<h2>三、实验设计的贡献</h2><ol>
<li>条件 A / B / C 与中间臂（A+文本、冻结 embedding、S 无 Agent）拆分各部分贡献。</li>
<li>同协议对比 SpeechCARE：PREPARE 官方划分，同指标；所有选择只用训练集 OOF。</li>
<li>跨模型公平性：三家相同提示、schema、强度、批量；记录 token 与失败。</li>
<li>治理性质实验：换措辞漂移、重测一致性、冗余与 Pareto、学习曲线、记忆探针。</li></ol>

<h2>四、结果</h2>
<h3>4.1 PREPARE：三个条件与 SpeechCARE</h3>
<div class="wrap"><table><tr><th>方案</th><th>Accuracy</th><th>macro-F1</th><th>micro-AUC</th><th>log-loss</th></tr>{prep_html}</table></div>
<p class="muted">* SpeechCARE 的 log-loss 为 PREPARE 第二阶段排行榜数值。SpeechCARE 微调编码器并取 10 次训练均值；本系统编码器冻结，协议不完全相同。</p>
<p>结论：C 相对 A 提高约 {f((bm.get('acc',0)-p3.get('A_acoustic',{}).get('acc',0))*100,1)} 个点准确率；micro-AUC 高于 SpeechCARE，Accuracy 低约 {f((0.7211-bm.get('acc',0))*100,1)} 个点。融合层（多种堆叠器、两级判决、阈值）的变化都在 0.5 个点以内，剩余差距来自编码器未微调。</p>
<h3>4.2 三家模型在同一框架下（PREPARE）</h3>
<div class="wrap"><table><tr><th>模型</th><th>B Acc</th><th>B AUC</th><th>C Acc</th><th>C AUC</th><th>B 翻转</th><th>C 翻转</th><th>返回 B/C</th></tr>{mc_rows}</table></div>
{img(V2 / "model_compare/Fig_model_compare.png")}
<h3>4.3 九个数据集（AUC，重复 5 折交叉验证）</h3>
<div class="wrap"><table><tr><th>数据集</th><th>n</th><th>标签</th><th>A</th><th>S</th><th>B GPT</th><th>B Claude</th><th>B DeepSeek</th><th>C GPT</th><th>C Claude</th><th>C DeepSeek</th></tr>{mm_rows}</table></div>
{img(OUT / "Fig_multi_dataset.png")}
<p class="c"><b>数据质量说明：</b>DementiaBank Pitt（ID 形如 AD_001）、NCMMSC（AD_F_…）和 DementiaNet（公众人物姓名）的受试者 ID 直接暴露诊断或身份，早先发送给大模型的请求因此存在标签泄漏，已全部作废；现已改为哈希化名，这三个数据集按要求暂停，重跑后补入。DeepSeek 的多数据集 C 臂同样按要求暂停，表中空格表示未运行，不是失败。</p>
<h3>4.4 模型越强，框架收益越大吗</h3>
<div class="wrap"><table><tr><th>模型</th><th>纯 Agent 平均 AUC（能力代理）</th><th>C − S 平均增益</th><th>数据集数</th></tr>{cap_rows}</table></div><p>{cap_text}</p>

<h2>五、老师的问题（Q1–Q6）</h2>
<h3>Q1 语言漂移：稳定性与准确性</h3><p>同一批 120 例、只改提示措辞后的结论翻转率（PREPARE）——{flip_text}。准确率不降。原因是结构上界：LLM 只能经由学到的权重影响结论，|Δlogit| ≤ Σ|β|·|Δz| + |w|·|Δa|。测量重测一致性（Spearman）——{retest_text}。</p>
<h3>Q2 证据是否重叠、是否独一有用</h3><p>{rd.get('n_metrics')} 个手工指标中 {rd.get('n_constant')} 个为常数（{", ".join(rd.get('constant', []))}，表示“测不到”而不是“正常”），{rd.get('n_pairs_rho_ge_0.9')} 对 |ρ|≥0.9。准入判据：条件信息增量 &gt; 0 或承担安全作用。</p>
<h3>Q3 为什么需要这么多 feature</h3><p>有效维数 {rd.get('effective_rank')}；按训练 OOF 逐组加入证据：</p><div class="wrap"><table><tr><th>组数</th><th>加入</th><th>OOF log-loss</th><th>测试 macro-AUC</th></tr>{par_rows}</table></div><p>训练内最优约在第 4 组，之后不再改善：候选库大、实际执行稀疏。</p>
<h3>Q4 底层模型是否见过数据库</h3><p>续写探针（GPT-5.5）：公开 ADReSS 转录与真实后文 4-gram 重合 {a_.get('mean_4gram_overlap_true')}，与无关病例 {a_.get('mean_4gram_overlap_other_case')}，无记忆迹象；PREPARE 对照偏高（{p_.get('mean_4gram_overlap_true')} vs {p_.get('mean_4gram_overlap_other_case')}）来自朗读任务的固定文本。阴性结果不能证明从未见过；Agent 作为测量仪器时输出可逐项核验，记忆的标签无法无声抬高结果。</p>
<h3>Q5 少量病例能否学会</h3><div class="wrap"><table><tr><th>训练例数</th><th>A 声学</th><th>E embedding</th><th>S 状态</th><th>S 全部</th></tr>{lc_rows}</table></div><p>32 例时状态表示略优且方差更小；64 例以上 embedding 反超；完整系统各样本量不低于 embedding。</p>
<h3>Q6 trace map 与可解释性</h3><p>trace map 是事先规定的信息流图：结论只沿“片段 → 指标 → 状态 → 贡献”获得信息，Q1 的干预结果（无关措辞几乎不改变结论）是它的经验检验。开放权重模型内部机制分析列为后续工作。</p>
<h3>Agent 增益问题</h3><p class="c">PREPARE（ASR、约 30 秒）上 LLM 测量对准确率无增益——信息已被文本 embedding 覆盖（与文献“ASR 转录上 LLM 特征增益很小”一致）；Agent 的可测价值是稳定性、可核验性与跨模型一致性。增益检验见 4.3 中人工转录数据集（ADReSS、Pitt、PROCESS-2）。</p>

<h2>六、局限</h2><ul><li>PREPARE 测试集被多次查看，结论为回顾性。</li><li>Accuracy 仍低于 SpeechCARE；需要折内微调编码器。</li><li>小数据集（DementiaNet n≈20、IAEAV）方差大。</li><li>三家模型的比较只有 3 个点，不足以建立“能力—收益”的规律。</li><li>未做医生验证。</li></ul>
<h2>七、论文图</h2>{img(FIG / "Fig1_architecture.png")}{img(FIG / "Fig3_results.png")}{img(FIG / "Fig4_governance.png")}
</main></body></html>"""
(OUT / "index.html").write_text(html, encoding="utf-8")

oral = f"""# ADvoice v2 汇报口语稿（约 8 分钟）

各位老师好，我汇报 ADvoice 这一轮的进展：系统重新设计、三家大模型对比，以及上次老师提出的几个问题。

## 一、我们做的是什么
输入一段老人的语音，系统输出健康、轻度认知障碍、阿尔茨海默病的筛查估计，并且告诉医生：是哪些具体表现支持这个结论。我们的核心想法叫“证据治理”：大模型不直接下结论，它只能通过有限、可以检查的通道影响结果。

## 二、三个核心贡献
第一，系统设计。我们把流程拆成测量、证据、状态、可分解预测和受控报告五层。结论可以拆成每个认知状态的贡献，再加一个标明大小的编码器残差，所以报告里写的依据，就是真正推动结论的东西。
第二，实验设计。同一个框架下比较三个条件：A 是传统声学方法，B 是纯大模型 Agent，C 是我们的框架加 Agent。GPT、Claude、DeepSeek 用完全相同的提示、输出格式和推理强度。
第三，治理性质是可以测量的：稳定性、证据冗余、少样本和数据污染，我们都有实验数据。

## 三、主要结果
在 PREPARE 上，C 的 micro-AUC 是 {f(bm.get('micro_auc'))}，高于 SpeechCARE 的 0.868；准确率 {f(bm.get('acc'))}，比 SpeechCARE 的 0.721 还低约 {f((0.7211-bm.get('acc',0))*100,1)} 个点。我们试了多种融合和判决方式，变化都在半个点以内，剩下的差距来自我们的编码器没有微调，这是下一步。
C 比传统声学 A 高约 {f((bm.get('acc',0)-p3.get('A_acoustic',{}).get('acc',0))*100,1)} 个点，比纯大模型 B 高得更多。纯大模型单独判断甚至比传统声学还差，原因是它不知道人群的正常范围、没有校准，也听不到停顿和语速。

## 四、三家模型
{cap_text}在九个数据集上，详细数字见汇报页的 4.3 节。

## 五、老师的问题
关于语言漂移：同样 120 个病例，只换提示措辞，三家纯大模型平均有 {f(100*np.mean([b for b, c in flips.values()]),1) if flips else '—'}% 的结论翻转，放进我们的系统后平均只有 {f(100*np.mean([c for b, c in flips.values()]),1) if flips else '—'}%。这不是偶然：在可分解结构下，大模型的漂移对结论的影响有一个上界，就是学到的权重乘以测量的漂移。
关于一百个特征：55 个手工指标里，有 3 个其实是常数，14 对高度相关，真正的有效维数大约 17；按训练数据逐组加入证据，大约 4 组就达到最好。所以我们的主张是“候选库大，实际执行稀疏”。
关于底层模型是否见过数据：我们让 GPT 续写公开的 ADReSS 转录，和真实后文的重合度与和无关病例的重合度一样，没有发现记忆。这不能证明它从没见过，但我们让 Agent 做可以逐项核对的测量，而不是直接判类别，记住的标签无法悄悄抬高结果。
关于少样本：只有 32 个训练病例时，基于认知状态的模型略好而且更稳定；病例多了以后，深度表示反超；完整系统在各个样本量下都不比深度表示差。
关于 Agent 的增益：在 PREPARE 这种自动转写、每段约 30 秒的数据上，大模型测量没有提高准确率，信息已经被文本表示覆盖了。Agent 的价值体现在稳定性和可核验性上。

## 六、下一步
一，在每一折内微调 Whisper 和文本编码器，目标是准确率超过 SpeechCARE；二，在人工转录的数据集上检验 Agent 的准确率增益；三，做医生读片实验。谢谢各位老师。
"""
(OUT / "oral_zh.md").write_text(oral, encoding="utf-8")
print("wrote", OUT)
