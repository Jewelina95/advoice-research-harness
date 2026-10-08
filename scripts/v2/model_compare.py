#!/usr/bin/env python3
"""Compare GPT / Claude / DeepSeek under the same harness on PREPARE: B (plain agent), C (framework + agent),
drift under prompt rewording, and measurement test-retest. Writes JSON + a Nature-style figure."""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pv2", HERE / "prepare_v2.py")
pv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv2)
ART, V2, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
MODELS = {"GPT-5.5": "openai_api_gpt-5.5-2026-04-23", "Claude Opus 5.5": "claude_cli_claude-opus-5-5",
          "DeepSeek V4 Pro": "deepseek_api_deepseek-v4-pro"}
AG = V2 / "agent"

feats = pd.read_csv(ART / "subject_features.csv")
man = pd.read_csv(ART / "manifest.csv")[["subject_id", "task_type", "language"]]
df = feats.drop(columns=[c for c in ("task_type", "language") if c in feats]).merge(man, on="subject_id")
st = pd.read_csv(ART / "state_wide.csv")
df = df.merge(st[["subject_id"] + [c for c in st.columns if c.startswith("state_S")]], on="subject_id")
y_all = df["label"].map({k: i for i, k in enumerate(pv2.LABELS)}).to_numpy()
tr = (df["split"] == "train").to_numpy()
te = ~tr
y = y_all[tr]
df["age_bin"] = pd.cut(df["age"], [0, 65, 75, 200], labels=False)
demo = pd.get_dummies(df[["task_type", "language", "sex"]].astype(str)).astype(float)
demo["age"] = df["age"]


def emb(path: Path) -> np.ndarray:
    d = np.load(path, allow_pickle=True)
    ids = d["subject_ids"].astype(str) if "subject_ids" in d.files else np.array(sorted(feats["subject_id"].astype(str)))
    return d["embeddings"][pd.Series(range(len(ids)), index=ids).loc[df["subject_id"].astype(str)].to_numpy()]


z = pv2.healthy_z(df, pv2.ACOUSTIC + pv2.TEXT, tr, y_all, ["task_type", "language", "age_bin"])
base = {"acoustic": df[pv2.ACOUSTIC].to_numpy(float), "text_metrics": df[pv2.TEXT].to_numpy(float),
        "states": df[[c for c in df.columns if c.startswith("state_S")]].to_numpy(float),
        "z_stratified": z.to_numpy(float), "demo": demo.to_numpy(float),
        "audio_emb": emb(ART / "multilingual_audio_embeddings.npz"),
        "text_emb": emb(ART / "multilingual_text_embeddings.npz"), "whisper32": emb(V2 / "whisper/whisper_l32.npz")}
OUT.mkdir(parents=True, exist_ok=True)
cache_file = OUT / "base_branch_cache.npz"
if cache_file.exists():
    d = np.load(cache_file)
    cache = {k: (d[k + "__oof"], d[k + "__test"]) for k in base}
else:
    cache = {k: pv2.oof_branch(x[tr], y, x[te], 0, cs=(0.05,)) for k, x in base.items()}
    np.savez(cache_file, **{f"{k}__oof": v[0] for k, v in cache.items()}, **{f"{k}__test": v[1] for k, v in cache.items()})


def stacked(extra: dict):
    c = dict(cache)
    for k, x in extra.items():
        c[k] = pv2.oof_branch(x[tr], y, x[te], 0)
    keys = list(c)
    xtr = np.hstack([np.log(np.clip(c[k][0], 1e-6, 1)) for k in keys])
    xte = np.hstack([np.log(np.clip(c[k][1], 1e-6, 1)) for k in keys])
    return pv2.oof_branch(xtr, y, xte, 100, cs=(0.01, 0.1, 1.0)), c


(_, p_s), _ = stacked({})
res = {"S_framework_no_agent": pv2.metrics(y_all[te], p_s), "models": {}}
ids = (V2 / "drift_ids.txt").read_text().split()
pos = pd.Series(range(len(df)), index=df["subject_id"]).loc[ids].to_numpy()
tpos = np.searchsorted(np.where(te)[0], pos)
pc = ["p_HC", "p_MCI", "p_AD"]
meas = ["info_units", "coherence", "empty_word_ratio", "word_finding", "impairment_support"]
for label, tag in MODELS.items():
    cpath, bpath = AG / f"C_{tag}/results.csv", AG / f"B_{tag}/results.csv"
    if not (cpath.exists() and bpath.exists()):
        res["models"][label] = "pending"
        continue
    cm = pd.read_csv(cpath).set_index("subject_id").apply(pd.to_numeric, errors="coerce").reindex(df["subject_id"])
    bm = pd.read_csv(bpath).set_index("subject_id")[pc].reindex(df["subject_id"]).to_numpy(float)
    cover = {"C_returned": int(cm.notna().any(axis=1).sum()), "B_returned": int((~np.isnan(bm).any(1)).sum())}
    bm[np.isnan(bm).any(1)] = 1 / 3
    bm = np.clip(bm, 1e-6, None)
    bm /= bm.sum(1, keepdims=True)
    (_, p_c), c_all = stacked({"agent_measure": cm.to_numpy(float), "agent_judgment": np.log(bm)})
    entry = {"coverage": cover, "B_plain_agent": pv2.metrics(y_all[te], bm[te]), "C_framework_agent": pv2.metrics(y_all[te], p_c)}
    c1p, b1p = AG / f"C_{tag}_v1/results.csv", AG / f"B_{tag}_v1/results.csv"
    if c1p.exists() and b1p.exists():
        b1 = pd.read_csv(b1p).set_index("subject_id")[pc].reindex(ids).to_numpy(float)
        ok = ~np.isnan(b1).any(1)
        entry["B_flip_rate"] = float((bm[pos][ok].argmax(1) != b1[ok].argmax(1)).mean())
        c1 = pd.read_csv(c1p).set_index("subject_id").apply(pd.to_numeric, errors="coerce").reindex(ids)
        keys = list(c_all)
        stk = pv2.lr(0.1).fit(np.hstack([np.log(np.clip(c_all[k][0], 1e-6, 1)) for k in keys]), y)
        gm = pv2.lr(0.1).fit(cm.to_numpy(float)[tr], y)

        def pred(cx):
            parts = [np.log(np.clip(gm.predict_proba(cx) if k == "agent_measure" else c_all[k][1][tpos], 1e-6, 1)) for k in keys]
            return stk.predict_proba(np.hstack(parts))

        p0, p1 = pred(cm.to_numpy(float)[pos]), pred(c1.to_numpy(float))
        entry["C_flip_rate"] = float((p0.argmax(1) != p1.argmax(1)).mean())
        entry["test_retest_spearman"] = {k: round(float(cm.iloc[pos][k].corr(c1[k], method="spearman")), 3) for k in meas}
    res["models"][label] = entry
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "model_compare.json").write_text(json.dumps(res, indent=1, default=float))

# ---------------- figure ----------------
plt.rcParams.update({"font.family": "Arial", "font.size": 7, "axes.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "xtick.major.width": 0.6, "ytick.major.width": 0.6})
done = {k: v for k, v in res["models"].items() if v != "pending"}
colors = {"GPT-5.5": "#4C72B0", "Claude Opus 5.5": "#C44E52", "DeepSeek V4 Pro": "#55A868"}
fig, axs = plt.subplots(1, 4, figsize=(7.2, 2.1))
names = list(done)
x = np.arange(len(names))
for ax, metric, title in ((axs[0], "acc", "Accuracy"), (axs[1], "micro_auc", "Micro-AUC")):
    ax.bar(x - 0.18, [done[n]["B_plain_agent"][metric] for n in names], 0.34, color=[colors[n] for n in names], alpha=0.35, label="B plain agent")
    ax.bar(x + 0.18, [done[n]["C_framework_agent"][metric] for n in names], 0.34, color=[colors[n] for n in names], label="C framework + agent")
    ax.axhline(res["S_framework_no_agent"][metric], color="0.4", lw=0.7, ls=":", label="S framework, no agent")
    ax.axhline({"acc": 0.7211, "micro_auc": 0.8683}[metric], color="k", lw=0.7, ls="--", label="SpeechCARE (paper)")
    ax.set_xticks(x, [n.split()[0] for n in names])
    ax.set_ylim(0.4, 0.9)
    ax.set_title(title, fontsize=7.5)
fig.legend(*axs[0].get_legend_handles_labels(), frameon=False, fontsize=6, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02))
ax = axs[2]
fl = [(n, done[n].get("B_flip_rate"), done[n].get("C_flip_rate")) for n in names if done[n].get("B_flip_rate") is not None]
if fl:
    xx = np.arange(len(fl))
    ax.bar(xx - 0.18, [100 * f[1] for f in fl], 0.34, color=[colors[f[0]] for f in fl], alpha=0.35)
    ax.bar(xx + 0.18, [100 * f[2] for f in fl], 0.34, color=[colors[f[0]] for f in fl])
    ax.set_xticks(xx, [f[0].split()[0] for f in fl])
ax.set_title("Decision flips after rewording (%)", fontsize=7.5)
ax = axs[3]
for i, n in enumerate(names):
    tr_ = done[n].get("test_retest_spearman", {})
    ax.scatter(np.arange(len(tr_)) + (i - 1) * 0.15, list(tr_.values()), s=12, color=colors[n], label=n)
ax.set_xticks(np.arange(len(meas)), ["info units", "coherence", "empty words", "word finding", "impairment"], rotation=40, ha="right")
ax.set_ylim(0, 1)
ax.set_title("Measurement test–retest (ρ)", fontsize=7.5)
for a, l in zip(axs, "abcd"):
    a.text(-0.18, 1.08, l, transform=a.transAxes, fontweight="bold", fontsize=9)
fig.tight_layout(rect=(0, 0.08, 1, 1))
fig.savefig(OUT / "Fig_model_compare.pdf")
fig.savefig(OUT / "Fig_model_compare.png", dpi=300)
print(json.dumps({k: (v if isinstance(v, str) else {kk: (vv if not isinstance(vv, dict) or kk == "coverage" else {m: round(vv[m], 4) for m in ("acc", "macro_f1", "micro_auc")} if "acc" in vv else vv) for kk, vv in v.items()}) for k, v in res["models"].items()}, indent=1, default=float))
