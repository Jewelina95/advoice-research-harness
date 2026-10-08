#!/usr/bin/env python3
"""Fusion/stacking/decision search on PREPARE. Selection on train OOF only; test reported for top-3."""
import importlib.util, json, sys, time, warnings
from pathlib import Path
import numpy as np, pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier as HGB
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parents[2]
spec = importlib.util.spec_from_file_location("prepare_v2", ROOT / "scripts/v2/prepare_v2.py")
P = importlib.util.module_from_spec(spec); spec.loader.exec_module(P)
ART = Path("/Users/wenshaoyue/Desktop/research/ad general/AD voice/9.2/artifacts/PREPARE_DrivenData")
OUT = ROOT / ".local/v2/exp_fusion"; OUT.mkdir(exist_ok=True, parents=True)
NS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
V2 = ROOT / ".local/v2"

feats = pd.read_csv(ART / "subject_features.csv")
man = pd.read_csv(ART / "manifest.csv")[["subject_id", "task_type", "language"]]
df = feats.drop(columns=[c for c in ("task_type", "language") if c in feats]).merge(man, on="subject_id", how="left")
st = pd.read_csv(ART / "state_wide.csv"); st = st[["subject_id"] + [c for c in st.columns if c.startswith("state_S")]]
df = df.merge(st, on="subject_id", how="left")
yall = df["label"].map({k: i for i, k in enumerate(P.LABELS)}).to_numpy()
trm = (df["split"] == "train").to_numpy(); tem = ~trm; y = yall[trm]; yte = yall[tem]
df["age_bin"] = pd.cut(df["age"], [0, 65, 75, 200], labels=False)
demo = pd.get_dummies(df[["task_type", "language", "sex"]].astype(str)).astype(float); demo["age"] = df["age"]
sid = df["subject_id"].astype(str)
def emb(path):
    d = np.load(path, allow_pickle=True)
    ids = d["subject_ids"].astype(str) if "subject_ids" in d.files else np.array(sorted(feats["subject_id"].astype(str)))
    return d["embeddings"][pd.Series(range(len(ids)), index=ids).loc[sid].to_numpy()]
zs = P.healthy_z(df, P.ACOUSTIC + P.TEXT, trm, yall, ["task_type", "language", "age_bin"])
state_cols = [c for c in df.columns if c.startswith("state_S")]
B = {"acoustic": df[P.ACOUSTIC].to_numpy(float), "text_metrics": df[P.TEXT].to_numpy(float),
     "states": df[state_cols].to_numpy(float), "z_stratified": zs.to_numpy(float),
     "audio_emb": emb(ART / "multilingual_audio_embeddings.npz"),
     "audio_winstats": emb(ART / "multilingual_audio_window_stats_embeddings.npz"),
     "text_emb": emb(ART / "multilingual_text_embeddings.npz"), "demo": demo.to_numpy(float)}
for l in (8, 16, 24, 32): B[f"w{l}"] = emb(V2 / f"whisper/whisper_l{l}.npz")
c = pd.read_csv(V2 / "agent/C_openai_api_gpt-5.5-2026-04-23/results.csv").set_index("subject_id").apply(pd.to_numeric, errors="coerce")
B["gptC"] = c.reindex(df["subject_id"]).to_numpy(float)
b = pd.read_csv(V2 / "agent/B_openai_api_gpt-5.5-2026-04-23/results.csv").set_index("subject_id")[["p_HC", "p_MCI", "p_AD"]].reindex(df["subject_id"]).to_numpy(float)
b[np.isnan(b).any(1)] = 1 / 3; b = np.clip(b, 1e-6, None); b /= b.sum(1, keepdims=True); B["gptB"] = np.log(b)
d = np.load(V2 / "attn_l32_s3.npz"); pr = pd.DataFrame(d["probs"], index=d["subject_ids"].astype(str)).loc[sid].to_numpy()
ATTN = (np.clip(pr[trm], 1e-6, 1), np.clip(pr[tem], 1e-6, 1))
TAB = np.hstack([B["acoustic"], B["text_metrics"], B["states"], B["z_stratified"], B["demo"]])
EMB = ["audio_emb", "audio_winstats", "text_emb", "w16", "w32"]
PCAK = ["audio_emb", "text_emb", "w32"]

def mk(kind, c):
    if kind == "lr": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    if kind == "pca": return make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), PCA(256, svd_solver="randomized", random_state=0), StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    if kind == "hgb": return HGB(max_depth=3, learning_rate=0.04, max_iter=int(c), min_samples_leaf=30, l2_regularization=1.0, random_state=0)
GRID = {"lr": (0.001, 0.01, 0.1), "pca": (0.003, 0.03), "hgb": (80, 160)}
def branch(kind, x, seed):
    skf = StratifiedKFold(5, shuffle=True, random_state=seed); xt = x[trm]; xe = x[tem]; best = None
    for cc in GRID[kind]:
        oof = np.zeros((len(y), 3))
        for a, v in skf.split(xt, y): oof[v] = mk(kind, cc).fit(xt[a], y[a]).predict_proba(xt[v])
        ll = log_loss(y, oof)
        if best is None or ll < best[0]: best = (ll, cc, oof)
    return best[2], mk(kind, best[1]).fit(xt, y).predict_proba(xe)

def run_seed(seed):
    jobs = {}
    for k in ["acoustic", "text_metrics", "states", "z_stratified", "demo", "gptC"] + EMB: jobs[k] = ("lr", B[k])
    for k in PCAK: jobs[k + "_pca"] = ("pca", B[k])
    jobs["tab_hgb"] = ("hgb", TAB)
    out = {k: branch(kd, x, seed) for k, (kd, x) in jobs.items()}
    out["gptB"] = (np.exp(B["gptB"][trm]), np.exp(B["gptB"][tem])); out["attn"] = ATTN
    return out
import pickle
def seed_cached(seed):
    f = OUT / f"branch_seed{seed}.pkl"
    if f.exists(): return pickle.load(open(f, "rb"))
    r = run_seed(seed); pickle.dump(r, open(f, "wb")); return r
t0 = time.time()
if len(sys.argv) > 2 and sys.argv[2] == "branches":
    Parallel(n_jobs=NS)(delayed(seed_cached)(s) for s in range(NS)); sys.exit()
R = {i: pickle.load(open(f, "rb")) for i, f in enumerate(sorted(OUT.glob("branch_seed*.pkl")))}
NS = len(R)
print("branches done", time.time() - t0, flush=True)

# ---- stackers ----
LANG = pd.get_dummies(df["language"].astype(str)).to_numpy(float); TASK = pd.get_dummies(df["task_type"].astype(str)).to_numpy(float)
CTX = np.hstack([LANG, TASK]); CTXtr, CTXte = CTX[trm], CTX[tem]
def feat(seed, members, split):
    i = 0 if split == "tr" else 1
    return np.hstack([np.log(np.clip(R[seed][m][i], 1e-6, 1)) for m in members])
def inter(X, ctx): return np.hstack([X] + [X * ctx[:, [j]] for j in range(ctx.shape[1])])
def smk(kind, c):
    if kind in ("lr", "lr_int"): return make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    if kind == "mlp": return make_pipeline(StandardScaler(), MLPClassifier((16,), alpha=c, max_iter=600, early_stopping=False, random_state=0))
    if kind == "hgb": return HGB(max_depth=2, learning_rate=0.04, max_iter=int(c), min_samples_leaf=30, l2_regularization=1.0, random_state=0)
SG = {"lr": (0.01, 0.1, 1.0), "lr_int": (0.003, 0.01, 0.03, 0.1), "mlp": (1.0, 3.0, 10.0), "hgb": (50, 100)}
def stack_fit(kind, Xtr, ytr, Xte, seed, ctx_tr=None, ctx_te=None, binary=None):
    """Stacker with C chosen by inner OOF logloss; returns (oof, test) probs. binary: ('hc'|'ad') for hierarchical."""
    if kind == "lr_int": Xtr, Xte = inter(Xtr, ctx_tr), inter(Xte, ctx_te)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed + 100); best = None
    for cc in SG[kind]:
        oof = np.zeros((len(ytr), len(np.unique(ytr))))
        for a, v in skf.split(Xtr, ytr): oof[v] = smk(kind, cc).fit(Xtr[a], ytr[a]).predict_proba(Xtr[v])
        ll = log_loss(ytr, np.clip(oof, 1e-6, 1))
        if best is None or ll < best[0]: best = (ll, cc, oof)
    m = smk(kind, best[1]).fit(Xtr, ytr)
    return best[2], m.predict_proba(Xte), best[1], skf, Xtr
def flat(kind, members, seed):
    Xtr, Xte = feat(seed, members, "tr"), feat(seed, members, "te")
    o, t, *_ = stack_fit(kind, Xtr, y, Xte, seed, CTXtr, CTXte); return o, t
def hier(kind, members, seed):
    Xtr, Xte = feat(seed, members, "tr"), feat(seed, members, "te")
    kk = "lr" if kind == "lr_int" else kind
    o1, t1, *_ = stack_fit(kk, Xtr, (y > 0).astype(int), Xte, seed)   # HC vs impaired
    imp = y > 0
    # MCI vs AD: OOF over impaired rows; for non-impaired rows use model trained on all impaired
    skf = StratifiedKFold(5, shuffle=True, random_state=seed + 7)
    o2 = np.zeros(len(y)); 
    _, _, cc, _, _ = stack_fit(kk, Xtr[imp], (y[imp] == 2).astype(int), Xte, seed)
    full = smk(kk, cc).fit(Xtr[imp], (y[imp] == 2).astype(int))
    o2[:] = full.predict_proba(Xtr)[:, 1]  # placeholder for non-impaired rows
    ii = np.where(imp)[0]
    for a, v in skf.split(Xtr[ii], y[ii]):
        m = smk(kk, cc).fit(Xtr[ii[a]], (y[ii[a]] == 2).astype(int)); o2[ii[v]] = m.predict_proba(Xtr[ii[v]])[:, 1]
    # non-impaired OOF rows: model fit on impaired folds only is already out-of-sample w.r.t. them
    t2 = full.predict_proba(Xte)[:, 1]
    comb = lambda p1, q: np.c_[p1[:, 0], p1[:, 1] * (1 - q), p1[:, 1] * q]
    return comb(o1, o2), comb(t1, t2)

BASE = ["acoustic", "text_metrics", "states", "z_stratified", "audio_emb", "audio_winstats", "text_emb", "demo", "w32"]
SETS = {
 "base(zw32)": BASE,
 "base+attn": BASE + ["attn"],
 "base+attn+gptC": BASE + ["attn", "gptC"],
 "base+attn+gptC+gptB": BASE + ["attn", "gptC", "gptB"],
 "base+w16": BASE + ["w16"],
 "pca_emb+attn": ["acoustic", "text_metrics", "states", "z_stratified", "demo", "audio_emb_pca", "audio_winstats", "text_emb_pca", "w32_pca", "attn"],
 "base+pca32+attn": BASE + ["w32_pca", "attn"],
 "base+hgbtab+attn": BASE + ["tab_hgb", "attn"],
 "base+hgbtab+pca+attn+gptC": BASE + ["tab_hgb", "w32_pca", "text_emb_pca", "attn", "gptC"],
 "all_lr+attn+gptC+gptB": BASE + ["w16", "attn", "gptC", "gptB"],
}
CFG = []
for sname in SETS:
    for kind in ("lr", "lr_int", "mlp", "hgb"):
        if kind != "lr" and sname not in ("base+attn", "base+attn+gptC", "base+hgbtab+pca+attn+gptC"): continue
        CFG.append((sname, kind, "flat"))
    if sname in ("base+attn", "base+attn+gptC", "base+hgbtab+pca+attn+gptC"): CFG.append((sname, "lr", "hier"))
def one(cfg, seed):
    s, k, h = cfg; f = hier if h == "hier" else flat
    return f(k, SETS[s], seed)
print(len(CFG), "configs", flush=True)
res = Parallel(n_jobs=10)(delayed(one)(cfg, s) for cfg in CFG for s in range(NS))
G = np.arange(-1.0, 1.01, 0.1)
def best_off(po, yy):
    lp = np.log(np.clip(po, 1e-6, 1)); best = (-1, (0, 0))
    for b1 in G:
        for b2 in G:
            a = ((lp + [0, b1, b2]).argmax(1) == yy).mean()
            if a > best[0] + 1e-9: best = (a, (b1, b2))
    return best[1]
def apply_off(p, off):
    q = np.exp(np.log(np.clip(p, 1e-6, 1)) + [0, *off]); return q / q.sum(1, keepdims=True)
rows = []; TEST = {}
for i, cfg in enumerate(CFG):
    rr = res[i * NS:(i + 1) * NS]; po = np.mean([r[0] for r in rr], 0); pt = np.mean([r[1] for r in rr], 0)
    off = best_off(po, y)
    # nested (honest) offset estimate: tune on 4/5 of OOF, apply to held-out 1/5
    pred = np.zeros(len(y), int)
    for a, v in StratifiedKFold(5, shuffle=True, random_state=11).split(po, y):
        pred[v] = apply_off(po[v], best_off(po[a], y[a])).argmax(1)
    mo = P.metrics(y, po) if False else None
    from sklearn.metrics import f1_score
    rows.append({"set": cfg[0], "stacker": cfg[1], "mode": cfg[2], "oof_acc": (po.argmax(1) == y).mean(),
                 "oof_ll": log_loss(y, np.clip(po, 1e-6, 1)), "oof_f1": f1_score(y, po.argmax(1), average="macro"),
                 "oof_acc_off_inbag": (apply_off(po, off).argmax(1) == y).mean(), "oof_acc_off_nested": (pred == y).mean(),
                 "off": off, "idx": i})
    TEST[i] = (po, pt, off)
T = pd.DataFrame(rows).sort_values("oof_acc_off_nested", ascending=False)
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
print(T.drop(columns=["idx"]).round(4).to_string(index=False))
T.to_csv(OUT / "oof_table.csv", index=False)
top = []
for _, r in T.head(3).iterrows():
    po, pt, off = TEST[r.idx]
    top.append({"cfg": f"{r.set}|{r.stacker}|{r['mode']}", "test_argmax": P.metrics(yte, pt), "test_offset": P.metrics(yte, apply_off(pt, off))})
json.dump(top, open(OUT / "top3_test.json", "w"), indent=1, default=float)
for t in top: print(json.dumps(t, default=float))
