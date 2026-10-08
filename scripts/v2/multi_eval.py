#!/usr/bin/env python3
"""Nine-dataset A / S / B / C comparison for GPT, Claude and DeepSeek (repeated 5-fold CV over subjects).

A = acoustic metrics; S = framework without agent (acoustic + linguistic metrics + StateCards + demographics);
B_<m> = plain LLM probabilities; C_<m> = S + LLM measurement + LLM judgement (stacked, OOF only).
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT, MULTI, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
DATASETS = ["ADReSS_2020", "ADReSSo_2021_diagnosis", "DementiaBank_Pitt", "DementiaNet_PublicFigures", "IAEAV",
            "NCMMSC2021_AD", "PROCESS_2", "TAUKADIAL"]
MODELS = {"GPT-5.5": "openai_api_gpt-5.5-2026-04-23", "Claude Opus 5.5": "claude_cli_claude-opus-5-5",
          "DeepSeek V4 Pro": "deepseek_api_deepseek-v4-pro"}
ACOUSTIC = ["silence_fraction", "long_pause_rate_min", "pause_mean_sec", "pause_p90_sec", "speech_run_mean_sec",
            "speech_run_rate_min", "speech_run_cv", "rms_db_mean", "rms_db_std", "f0_median_hz", "f0_iqr_hz",
            "zcr_mean", "spectral_centroid_mean", "spectral_bandwidth_mean", "spectral_rolloff_mean",
            "spectral_flatness_mean", "duration_sec"]
TEXT = ["word_count", "speech_rate_wpm", "lexical_ttr", "lexical_mattr50", "filler_rate_100w", "pronoun_ratio",
        "content_word_ratio"]


def lr(c):
    return make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True), StandardScaler(),
                         LogisticRegression(C=c, max_iter=3000))


def oof(xtr, ytr, xte, seed, cs=(0.01, 0.1, 1.0)):
    k = min(5, int(np.bincount(ytr).min()))
    skf = StratifiedKFold(max(k, 2), shuffle=True, random_state=seed)
    best = None
    for c in cs:
        o = np.zeros((len(ytr), len(np.unique(ytr))))
        for a, b in skf.split(xtr, ytr):
            o[b] = lr(c).fit(xtr[a], ytr[a]).predict_proba(xtr[b])
        loss = -np.mean(np.log(np.clip(o[np.arange(len(ytr)), ytr], 1e-6, 1)))
        if best is None or loss < best[0]:
            best = (loss, c, o)
    return best[2], lr(best[1]).fit(xtr, ytr).predict_proba(xte)


def score(y, p, k):
    pred = p.argmax(1)
    auc = roc_auc_score(y, p[:, 1]) if k == 2 else roc_auc_score(np.eye(k)[y], p, average="macro", multi_class="ovr")
    return {"auc": auc, "bacc": balanced_accuracy_score(y, pred), "acc": accuracy_score(y, pred)}


def collapse(pb: np.ndarray, labels: list[str]) -> np.ndarray:
    """Map plain-agent HC/MCI/AD probabilities onto the dataset's label set."""
    hc, mci, ad = pb.T
    if labels == ["HC", "AD"]:
        q = np.stack([hc, mci + ad], 1)
    elif labels == ["HC", "MCI"]:
        q = np.stack([hc, mci + ad], 1)
    else:
        q = pb
    return q / q.sum(1, keepdims=True)


results = {}
for ds in DATASETS:
    art = ROOT / ds
    f = pd.read_csv(art / "subject_features.csv").drop_duplicates("subject_id")
    st = pd.read_csv(art / "state_wide.csv").drop_duplicates("subject_id")
    df = f.merge(st[["subject_id"] + [c for c in st.columns if c.startswith("state_S")]], on="subject_id", how="left")
    labels = [l for l in ["HC", "MCI", "AD"] if l in set(df["label"])]
    df = df[df["label"].isin(labels)].reset_index(drop=True)
    y = df["label"].map({l: i for i, l in enumerate(labels)}).to_numpy()
    k = len(labels)
    demo = pd.DataFrame({"age": pd.to_numeric(df["age"], errors="coerce") if "age" in df else np.nan}, index=df.index)
    if "sex" in df:
        demo["male"] = (df["sex"].astype(str).str.lower() == "male").astype(float)
    branches = {"acoustic": df[[c for c in ACOUSTIC if c in df]].to_numpy(float),
                "text": df[[c for c in TEXT if c in df]].to_numpy(float),
                "states": df[[c for c in df.columns if c.startswith("state_S")]].to_numpy(float),
                "demo": demo.to_numpy(float)}
    llm = {}
    for name, tag in MODELS.items():
        cp, bp = MULTI / ds / f"C_{tag}/results.csv", MULTI / ds / f"B_{tag}/results.csv"
        if cp.exists() and bp.exists():
            c = pd.read_csv(cp).drop_duplicates("subject_id").set_index("subject_id").apply(pd.to_numeric, errors="coerce")
            b = pd.read_csv(bp).drop_duplicates("subject_id").set_index("subject_id")[["p_HC", "p_MCI", "p_AD"]]
            bm = b.reindex(df["subject_id"]).to_numpy(float)
            cov = float((~np.isnan(bm).any(1)).mean())
            bm[np.isnan(bm).any(1)] = 1 / 3
            bm = collapse(np.clip(bm, 1e-6, None), labels)
            llm[name] = {"measure": c.reindex(df["subject_id"]).to_numpy(float), "judge": np.log(bm), "direct": bm, "coverage": cov}
    arms = {"A_acoustic": ["acoustic"], "S_framework": ["acoustic", "text", "states", "demo"]}
    for name in llm:
        arms[f"C_{name}"] = ["acoustic", "text", "states", "demo", f"m_{name}", f"j_{name}"]
        branches[f"m_{name}"] = llm[name]["measure"]
        branches[f"j_{name}"] = llm[name]["judge"]
    scores: dict = {a: [] for a in arms}
    for rep in range(3):
        skf = StratifiedKFold(min(5, int(np.bincount(y).min())), shuffle=True, random_state=rep)
        for tr, te in skf.split(np.zeros(len(y)), y):
            cache = {b: oof(x[tr], y[tr], x[te], rep) for b, x in branches.items()}
            for arm, mem in arms.items():
                if len(mem) == 1:
                    p = cache[mem[0]][1]
                else:
                    xs = np.hstack([np.log(np.clip(cache[m][0], 1e-6, 1)) for m in mem])
                    xt = np.hstack([np.log(np.clip(cache[m][1], 1e-6, 1)) for m in mem])
                    p = oof(xs, y[tr], xt, rep + 50)[1]
                scores[arm].append((te, p))
    res = {"n": int(len(y)), "labels": labels}
    for arm, parts in scores.items():
        reps = [parts[i * (len(parts) // 3):(i + 1) * (len(parts) // 3)] for i in range(3)]
        vals = []
        for r in reps:
            p = np.zeros((len(y), k))
            for te, pp in r:
                p[te] = pp
            vals.append(score(y, p, k))
        res[arm] = {m: round(float(np.mean([v[m] for v in vals])), 4) for m in vals[0]}
    for name, d in llm.items():
        res[f"B_{name}"] = {m: round(float(v), 4) for m, v in score(y, d["direct"], k).items()}
        res[f"B_{name}"]["coverage"] = round(d["coverage"], 3)
    results[ds] = res
    print(ds, json.dumps(res), flush=True)
OUT.write_text(json.dumps(results, indent=1))
