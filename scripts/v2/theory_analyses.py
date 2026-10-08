#!/usr/bin/env python3
"""Q1 drift, Q2/Q3 redundancy + Pareto, Q5 learning curve on PREPARE cached artifacts. Writes one JSON."""
from __future__ import annotations

import importlib.util
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, roc_auc_score

warnings.filterwarnings("ignore")
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("pv2", HERE / "prepare_v2.py")
pv2 = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pv2)
ART = Path(sys.argv[1])
V2 = Path(sys.argv[2])
OUT = Path(sys.argv[3])
LABELS = pv2.LABELS

feats = pd.read_csv(ART / "subject_features.csv")
man = pd.read_csv(ART / "manifest.csv")[["subject_id", "task_type", "language"]]
df = feats.drop(columns=[c for c in ("task_type", "language") if c in feats]).merge(man, on="subject_id")
st = pd.read_csv(ART / "state_wide.csv")
df = df.merge(st[["subject_id"] + [c for c in st.columns if c.startswith("state_S")]], on="subject_id")
y_all = df["label"].map({k: i for i, k in enumerate(LABELS)}).to_numpy()
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
gptc = pd.read_csv(V2 / "agent/C_openai_api_gpt-5.5-2026-04-23/results.csv").set_index("subject_id")
gptc = gptc.apply(pd.to_numeric, errors="coerce")
B = {
    "acoustic": df[pv2.ACOUSTIC].to_numpy(float),
    "text_metrics": df[pv2.TEXT].to_numpy(float),
    "states": df[[c for c in df.columns if c.startswith("state_S")]].to_numpy(float),
    "z_stratified": z.to_numpy(float),
    "demo": demo.to_numpy(float),
    "audio_emb": emb(ART / "multilingual_audio_embeddings.npz"),
    "text_emb": emb(ART / "multilingual_text_embeddings.npz"),
    "whisper32": emb(V2 / "whisper/whisper_l32.npz"),
    "gpt_measure": gptc.reindex(df["subject_id"]).to_numpy(float),
}
res: dict = {}

# ---------- Q2/Q3: redundancy among hand-crafted metrics ----------
metric_cols = pv2.ACOUSTIC + pv2.TEXT + [c for c in feats.columns if c.startswith(("mfcc_", "repair", "mean_utt", "patient_turn", "filler"))]
metric_cols = list(dict.fromkeys(c for c in metric_cols if c in df))
M = df.loc[tr, metric_cols].astype(float)
const = [c for c in metric_cols if M[c].nunique(dropna=True) <= 1]
Mv = M.drop(columns=const)
rho = Mv.corr(method="spearman").abs()
pairs = [(a, b, round(float(rho.loc[a, b]), 3)) for i, a in enumerate(rho.columns) for b in rho.columns[i + 1:] if rho.loc[a, b] >= 0.9]
Zs = ((Mv - Mv.median()) / (Mv.std() + 1e-9)).fillna(0).to_numpy()
ev = np.clip(np.linalg.eigvalsh(np.corrcoef(Zs, rowvar=False)), 1e-12, None)
pnorm = ev / ev.sum()
res["redundancy"] = {"n_metrics": len(metric_cols), "n_constant": len(const), "constant": const,
                     "n_pairs_rho_ge_0.9": len(pairs), "pairs": pairs[:25],
                     "effective_rank": round(float(np.exp(-(pnorm * np.log(pnorm)).sum())), 2),
                     "n_variable": Mv.shape[1]}

# ---------- Q2/Q3: greedy forward selection of evidence groups (selection on train OOF only) ----------
cache = {k: pv2.oof_branch(x[tr], y, x[te], 0) for k, x in B.items()}


def stack(members: list[str]):
    if len(members) == 1:
        return cache[members[0]]
    xtr = np.hstack([np.log(np.clip(cache[m][0], 1e-6, 1)) for m in members])
    xte = np.hstack([np.log(np.clip(cache[m][1], 1e-6, 1)) for m in members])
    return pv2.oof_branch(xtr, y, xte, 100, cs=(0.01, 0.1, 1.0))


chosen: list[str] = []
pareto = []
left = list(B)
while left:
    best = None
    for k in left:
        o, t = stack(chosen + [k])
        l = log_loss(y, o)
        if best is None or l < best[0]:
            best = (l, k, t)
    chosen.append(best[1])
    left.remove(best[1])
    onehot = np.eye(3)[y_all[te]]
    pareto.append({"n_groups": len(chosen), "added": best[1], "oof_logloss": round(best[0], 4),
                   "test_macro_auc": round(float(roc_auc_score(onehot, best[2], average="macro", multi_class="ovr")), 4),
                   "test_acc": round(float((best[2].argmax(1) == y_all[te]).mean()), 4)})
res["pareto"] = pareto

# ---------- Q5: learning curve ----------
arms = {"A_acoustic": ["acoustic"], "E_embeddings": ["audio_emb", "text_emb", "whisper32"],
        "S_framework_states": ["states", "z_stratified", "demo"], "S_full": ["states", "z_stratified", "demo", "audio_emb", "text_emb", "whisper32"]}
rng = np.random.default_rng(0)
idx_tr = np.where(tr)[0]
curve = []
for n in [32, 64, 128, 256, 512, len(idx_tr)]:
    for arm, members in arms.items():
        aucs = []
        for rep in range(5 if n < len(idx_tr) else 1):
            sel = np.concatenate([rng.choice(idx_tr[y == k], max(2, int(round(n * (y == k).mean()))), replace=False) for k in range(3)])
            ps = []
            for m in members:
                model = pv2.lr(0.1 if m in ("states", "z_stratified", "demo", "acoustic") else 0.01).fit(B[m][sel], y_all[sel])
                ps.append(np.log(np.clip(model.predict_proba(B[m][te]), 1e-6, 1)))
            p = np.exp(np.mean(ps, 0))
            p /= p.sum(1, keepdims=True)
            aucs.append(roc_auc_score(np.eye(3)[y_all[te]], p, average="macro", multi_class="ovr"))
        curve.append({"n_train": int(n), "arm": arm, "macro_auc": round(float(np.mean(aucs)), 4), "sd": round(float(np.std(aucs)), 4)})
res["learning_curve"] = curve

# ---------- Q1: drift under prompt rewording (GPT-5.5), 120 test cases ----------
ids = Path(V2 / "drift_ids.txt").read_text().split()
b0 = pd.read_csv(V2 / "agent/B_openai_api_gpt-5.5-2026-04-23/results.csv").set_index("subject_id").loc[ids]
b1 = pd.read_csv(V2 / "agent/B_openai_api_gpt-5.5-2026-04-23_v1/results.csv").set_index("subject_id").reindex(ids)
pc = ["p_HC", "p_MCI", "p_AD"]
flip_b = float((b0[pc].to_numpy().argmax(1) != b1[pc].to_numpy().argmax(1)).mean())
c1 = pd.read_csv(V2 / "agent/C_openai_api_gpt-5.5-2026-04-23_v1/results.csv").set_index("subject_id").apply(pd.to_numeric, errors="coerce")
full = ["acoustic", "text_metrics", "states", "z_stratified", "demo", "audio_emb", "text_emb", "whisper32", "gpt_measure"]
xtr = np.hstack([np.log(np.clip(cache[m][0], 1e-6, 1)) for m in full])
stk = pv2.lr(0.1).fit(xtr, y)
g_model = pv2.lr(0.1).fit(B["gpt_measure"][tr], y)
pos = pd.Series(range(len(df)), index=df["subject_id"]).loc[ids].to_numpy()
tpos = np.searchsorted(np.where(te)[0], pos)


def c_pred(gpt_x: np.ndarray) -> np.ndarray:
    parts = []
    for m in full:
        parts.append(np.log(np.clip(g_model.predict_proba(gpt_x) if m == "gpt_measure" else cache[m][1][tpos], 1e-6, 1)))
    return stk.predict_proba(np.hstack(parts))


p0 = c_pred(B["gpt_measure"][pos])
p1 = c_pred(c1.reindex(ids).to_numpy(float))
meas = ["info_units", "coherence", "empty_word_ratio", "word_finding", "impairment_support"]
g0 = gptc.loc[ids, meas]
g1 = c1.reindex(ids)[meas]
res["drift"] = {"n_cases": len(ids), "B_plain_agent_flip_rate": round(flip_b, 4),
                "B_mean_abs_prob_change": round(float(np.abs(b0[pc].to_numpy() - b1[pc].to_numpy()).mean()), 4),
                "C_framework_flip_rate": round(float((p0.argmax(1) != p1.argmax(1)).mean()), 4),
                "C_mean_abs_prob_change": round(float(np.abs(p0 - p1).mean()), 4),
                "measure_test_retest_spearman": {k: round(float(g0[k].corr(g1[k], method="spearman")), 3) for k in meas}}
OUT.write_text(json.dumps(res, indent=1, ensure_ascii=False))
print(json.dumps({k: (v if k != "pareto" and k != "learning_curve" else v) for k, v in res.items()}, ensure_ascii=False)[:6000])
