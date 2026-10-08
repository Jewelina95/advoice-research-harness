#!/usr/bin/env python3
"""v2 PREPARE experiment on cached 9.2 artifacts (retrospective; official test used once per arm).

Branches are fitted with 5-fold stratified OOF on train; a multinomial stacker
combines branch OOF probabilities. Outputs a compact JSON of test metrics.
"""
from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

warnings.filterwarnings("ignore")
LABELS = ["HC", "MCI", "AD"]
ACOUSTIC = ["silence_fraction", "long_pause_rate_min", "pause_mean_sec", "pause_p90_sec",
            "speech_run_mean_sec", "speech_run_rate_min", "speech_run_cv", "rms_db_mean", "rms_db_std",
            "f0_median_hz", "f0_iqr_hz", "f0_valid_fraction", "zcr_mean", "spectral_centroid_mean",
            "spectral_bandwidth_mean", "spectral_rolloff_mean", "spectral_flatness_mean", "duration_sec"]
TEXT = ["word_count", "speech_rate_wpm", "lexical_ttr", "lexical_mattr50", "filler_rate_100w",
        "pronoun_ratio", "content_word_ratio"]


def metrics(y: np.ndarray, p: np.ndarray) -> dict:
    onehot = np.eye(3)[y]
    pred = p.argmax(1)
    recall = [float(((pred == k) & (y == k)).sum() / max((y == k).sum(), 1)) for k in range(3)]
    return {"acc": accuracy_score(y, pred), "macro_f1": f1_score(y, pred, average="macro"),
            "micro_auc": roc_auc_score(onehot.ravel(), p.ravel()),
            "macro_auc": roc_auc_score(onehot, p, average="macro", multi_class="ovr"),
            "log_loss": log_loss(y, np.clip(p, 1e-6, 1), labels=[0, 1, 2]),
            "recall_HC_MCI_AD": recall}


def lr(c: float):
    return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                         LogisticRegression(C=c, max_iter=3000))


def oof_branch(xtr, y, xte, seed, cs=(0.001, 0.01, 0.1, 1.0)):
    """Choose C by inner OOF log loss, return train OOF and test probabilities."""
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    best = None
    for c in cs:
        oof = np.zeros((len(y), 3))
        for tr, va in skf.split(xtr, y):
            oof[va] = lr(c).fit(xtr[tr], y[tr]).predict_proba(xtr[va])
        loss = log_loss(y, oof)
        if best is None or loss < best[0]:
            best = (loss, c, oof)
    test = lr(best[1]).fit(xtr, y).predict_proba(xte)
    return best[2], test


def healthy_z(df: pd.DataFrame, cols: list[str], train_mask: np.ndarray, y: np.ndarray, keys: list[str]) -> pd.DataFrame:
    """Robust z of each metric against train HC sharing the same strata (fallback to coarser strata)."""
    out = pd.DataFrame(index=df.index)
    hc = df[train_mask & (y == 0)]
    for col in cols:
        z = pd.Series(np.nan, index=df.index)
        for level in range(len(keys), -1, -1):
            k = keys[:level]
            if k:
                stats = hc.groupby(k)[col].agg(["median", lambda s: (s.quantile(.75) - s.quantile(.25)) / 1.349, "count"])
                stats.columns = ["med", "scale", "n"]
                stats = stats[stats["n"] >= 15]
                merged = df[k].merge(stats, left_on=k, right_index=True, how="left")
                med, scale = merged["med"].to_numpy(), merged["scale"].to_numpy()
            else:
                med = np.full(len(df), hc[col].median())
                scale = np.full(len(df), (hc[col].quantile(.75) - hc[col].quantile(.25)) / 1.349)
            vals = (df[col].to_numpy() - med) / np.where(scale > 1e-9, scale, np.nan)
            fill = z.isna().to_numpy() & ~np.isnan(vals)
            z[fill] = vals[fill]
        out["z_" + col] = z.clip(-6, 6)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifacts", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--extra", nargs="*", default=[], help="name=path.npz extra embedding branches")
    a = ap.parse_args()
    art = Path(a.artifacts)
    feats = pd.read_csv(art / "subject_features.csv")
    man = pd.read_csv(art / "manifest.csv")[["subject_id", "task_type", "language"]]
    df = feats.drop(columns=[c for c in ("task_type", "language") if c in feats]).merge(man, on="subject_id", how="left")
    states = pd.read_csv(art / "state_wide.csv")
    states = states[["subject_id"] + [c for c in states.columns if c.startswith("state_S")]]
    df = df.merge(states, on="subject_id", how="left")
    y_all = df["label"].map({k: i for i, k in enumerate(LABELS)}).to_numpy()
    tr = (df["split"] == "train").to_numpy()
    te = ~tr
    df["age_bin"] = pd.cut(df["age"], [0, 65, 75, 200], labels=False)
    demo = pd.get_dummies(df[["task_type", "language", "sex"]].astype(str)).astype(float)
    demo["age"] = df["age"]

    def emb(name: str) -> np.ndarray:
        d = np.load(art / name, allow_pickle=True)
        # Text embeddings carry no IDs; all 9.2 artifacts share sorted subject order.
        ids = d["subject_ids"].astype(str) if "subject_ids" in d.files else np.array(sorted(feats["subject_id"].astype(str)))
        idx = pd.Series(range(len(ids)), index=ids)
        return d["embeddings"][idx.loc[df["subject_id"].astype(str)].to_numpy()]

    zcols = ACOUSTIC + TEXT
    z_strat = healthy_z(df, zcols, tr, y_all, ["task_type", "language", "age_bin"])
    state_cols = [c for c in df.columns if c.startswith("state_S")]
    branches = {
        "acoustic": df[ACOUSTIC].to_numpy(float),
        "text_metrics": df[TEXT].to_numpy(float),
        "states": df[state_cols].to_numpy(float),
        "z_stratified": z_strat.to_numpy(float),
        "audio_emb": emb("multilingual_audio_embeddings.npz"),
        "audio_winstats": emb("multilingual_audio_window_stats_embeddings.npz"),
        "text_emb": emb("multilingual_text_embeddings.npz"),
        "demo": demo.to_numpy(float),
    }
    for spec in a.extra:
        name, path = spec.split("=", 1)
        d = np.load(path, allow_pickle=True)
        idx = pd.Series(range(len(d["subject_ids"])), index=d["subject_ids"].astype(str))
        branches[name] = d["embeddings"][idx.loc[df["subject_id"].astype(str)].to_numpy()]
    arms = {
        "A_acoustic": ["acoustic"],
        "A_plus_text": ["acoustic", "text_metrics"],
        "E_embeddings": ["audio_emb", "audio_winstats", "text_emb"],
        "E_plus_demo": ["audio_emb", "audio_winstats", "text_emb", "demo"],
        "S_framework": ["acoustic", "text_metrics", "states", "audio_emb", "audio_winstats", "text_emb", "demo"],
        "S_framework_z": ["acoustic", "text_metrics", "states", "z_stratified", "audio_emb", "audio_winstats", "text_emb", "demo"],
    }
    for name in a.extra:
        key = name.split("=")[0]
        arms[f"S_framework_z+{key}"] = arms["S_framework_z"] + [key]
    y = y_all[tr]
    results: dict = {}
    for seed in range(a.seeds):
        cache = {b: oof_branch(x[tr], y, x[te], seed) for b, x in branches.items()}
        for arm, members in arms.items():
            if len(members) == 1:
                p = cache[members[0]][1]
            else:
                xs_tr = np.hstack([np.log(np.clip(cache[m][0], 1e-6, 1)) for m in members])
                xs_te = np.hstack([np.log(np.clip(cache[m][1], 1e-6, 1)) for m in members])
                _, p = oof_branch(xs_tr, y, xs_te, seed + 100, cs=(0.01, 0.1, 1.0))
            results.setdefault(arm, []).append(metrics(y_all[te], p))
    summary = {arm: {k: (np.mean([r[k] for r in rs], 0).round(4).tolist()) for k in rs[0]} for arm, rs in results.items()}
    summary["_speechcare_paper_mean"] = {"acc": 0.7211, "micro_auc": 0.8683}
    Path(a.out).write_text(json.dumps(summary, indent=1))
    for arm, m in summary.items():
        print(arm, m)


if __name__ == "__main__":
    main()
