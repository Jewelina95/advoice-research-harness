import numpy as np
import matplotlib.pyplot as plt
from style import PAL, setup, save, panel, jload

setup()
T = jload("theory.json"); C = jload("contamination/summary.json")
dr = T["drift"]; red = T["redundancy"]

fig = plt.figure(figsize=(7.3, 5.4))
gs = fig.add_gridspec(2, 3, hspace=0.75, wspace=0.5, left=0, right=1)

# a drift
ax = fig.add_subplot(gs[0, 0]); panel(ax, "a", x=-0.28, y=1.1)
flip = [dr["B_plain_agent_flip_rate"] * 100, dr["C_framework_flip_rate"] * 100]
b = ax.bar([0, 1], flip, color=[PAL["B"], PAL["S"]], width=0.55)
for x, v in zip([0, 1], flip):
    ax.text(x, v + 0.4, f"{v:.1f}%", ha="center", fontsize=6.5)
ax.set_xticks([0, 1]); ax.set_xticklabels(["Plain agent", "Framework"])
ax.set_ylabel("Class flips under rewording (%)"); ax.set_ylim(0, 18)
ax.set_title(f"Prompt-rewording drift (n={dr['n_cases']})", fontsize=6.8)
ax.text(0.5, 0.82, f"mean |$\\Delta$p|\n{dr['B_mean_abs_prob_change']:.3f} vs {dr['C_mean_abs_prob_change']:.4f}",
        transform=ax.transAxes, ha="center", fontsize=6)

# b test-retest
ax = fig.add_subplot(gs[0, 1]); panel(ax, "b", x=-0.35, y=1.1)
tr = dr["measure_test_retest_spearman"]
names = {"info_units": "Information units", "coherence": "Coherence", "empty_word_ratio": "Empty-word ratio",
         "word_finding": "Word finding", "impairment_support": "Impairment support"}
ks = list(tr)[::-1]
ax.barh(range(len(ks)), [tr[k] for k in ks], color=PAL["C"], height=0.6)
for i, k in enumerate(ks):
    ax.text(tr[k] + 0.01, i, f"{tr[k]:.2f}", va="center", fontsize=6)
ax.set_yticks(range(len(ks))); ax.set_yticklabels([names[k] for k in ks])
ax.set_xlim(0, 1); ax.set_xlabel("Test-retest Spearman $\\rho$")
ax.set_title("LLM measurement stability", fontsize=6.8)

# c redundancy
ax = fig.add_subplot(gs[0, 2]); panel(ax, "c", x=-0.3, y=1.1)
parts = [("Variable,\nnot in\nhigh-corr pair", red["n_variable"] - 2 * 0, None)]
nm = red["n_metrics"]; nc = red["n_constant"]; npair = red["n_pairs_rho_ge_0.9"]
vals = [nm, nc, npair, red["effective_rank"]]
labs = ["Metrics", "Constant", "Pairs\n|$\\rho$|$\\geq$0.9", "Effective\nrank"]
cols = [PAL["A"], PAL["B"], PAL["C"], PAL["S"]]
ax.bar(range(4), vals, color=cols, width=0.6)
for i, v in enumerate(vals):
    ax.text(i, v + 1, f"{v:g}", ha="center", fontsize=6.5)
ax.set_xticks(range(4)); ax.set_xticklabels(labs); ax.set_ylim(0, 65)
ax.set_title("Metric redundancy", fontsize=6.8)

# d pareto
ax = fig.add_subplot(gs[1, 0]); panel(ax, "d", x=-0.28, y=1.1)
P = T["pareto"]
ax.plot([p["n_groups"] for p in P], [p["oof_logloss"] for p in P], "-o", color=PAL["S"], ms=3.5, lw=1)
i = int(np.argmin([p["oof_logloss"] for p in P]))
ax.plot(P[i]["n_groups"], P[i]["oof_logloss"], "o", ms=7, mfc="none", mec=PAL["C"], mew=1)
for p in P:
    ax.text(p["n_groups"], p["oof_logloss"] + 0.0016, p["added"].replace("_", " "), rotation=60, fontsize=5, ha="left", va="bottom")
ax.set_xlabel("Evidence groups added (greedy)"); ax.set_ylabel("OOF log-loss")
ax.set_ylim(0.705, 0.727); ax.set_title("Pareto: evidence groups vs fit", fontsize=6.8)

# e learning curve
ax = fig.add_subplot(gs[1, 1]); panel(ax, "e", x=-0.3, y=1.1)
lab = {"A_acoustic": ("A acoustic", PAL["A"]), "E_embeddings": ("Embeddings", PAL["E"]),
       "S_framework_states": ("States only", PAL["S"]), "S_full": ("Full framework", PAL["S2"])}
for arm, (l, c) in lab.items():
    rows = [r for r in T["learning_curve"] if r["arm"] == arm]
    n = np.array([r["n_train"] for r in rows]); m = np.array([r["macro_auc"] for r in rows]); sd = np.array([r["sd"] for r in rows])
    ax.plot(n, m, "-o", color=c, ms=3, lw=1, label=l)
    ax.fill_between(n, m - sd, m + sd, color=c, alpha=0.15, lw=0)
ax.set_xscale("log", base=2); ax.set_xticks([32, 128, 512, 1622]); ax.set_xticklabels(["32", "128", "512", "1622"])
ax.set_xlabel("Training participants"); ax.set_ylabel("Test macro-AUC")
ax.legend(loc="lower right", fontsize=5.8); ax.set_title("Learning curve", fontsize=6.8)

# f contamination
ax = fig.add_subplot(gs[1, 2]); panel(ax, "f", x=-0.3, y=1.1)
ks = list(C); x = np.arange(len(ks)); w = 0.34
ax.bar(x - w / 2, [C[k]["mean_4gram_overlap_true"] for k in ks], w, color=PAL["C"], label="True case")
ax.bar(x + w / 2, [C[k]["mean_4gram_overlap_other_case"] for k in ks], w, color=PAL["A"], label="Other case")
for i, k in enumerate(ks):
    ax.text(i - w / 2, C[k]["mean_4gram_overlap_true"] + 0.003, f"{C[k]['mean_4gram_overlap_true']:.3f}", ha="center", fontsize=5.8)
    ax.text(i + w / 2, C[k]["mean_4gram_overlap_other_case"] + 0.003, f"{C[k]['mean_4gram_overlap_other_case']:.3f}", ha="center", fontsize=5.8)
ax.set_xticks(x); ax.set_xticklabels([f"{k.replace('_', ' ')}\n(n={C[k]['n']})" for k in ks], fontsize=5.8)
ax.set_ylabel("Mean 4-gram overlap"); ax.legend(fontsize=5.8); ax.set_ylim(0, 0.12)
ax.set_title("Contamination probe", fontsize=6.8)
save(fig, "Fig4_governance")
