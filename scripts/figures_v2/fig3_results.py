import numpy as np
import matplotlib.pyplot as plt
from style import PAL, CLASS_COL, setup, save, panel, jload

setup()
R = jload("prepare_v2_3seeds.json")
arms = [("A_acoustic", "A  acoustic only", PAL["A"]),
        ("A_plus_text", "A + text metrics", PAL["A2"]),
        ("E_plus_demo", "Frozen embeddings + demographics", PAL["E"]),
        ("S_framework_z", "S  evidence-state framework", PAL["S"]),
        ("S_framework_z+w32", "S + Whisper encoder", PAL["S2"]),
        ("C_gptC", "C  S + GPT measurement", PAL["C"]),
        ("B_gptB_direct", "B  plain GPT (transcript)", PAL["B"])]
SC = R["_speechcare_paper_mean"]

fig = plt.figure(figsize=(7.3, 5.6))
gs = fig.add_gridspec(2, 6, height_ratios=[1.15, 1], hspace=0.55, wspace=0.35,
                      left=0.0, right=1.0)
ys = np.arange(len(arms))[::-1]
metrics = [("acc", "Accuracy", SC["acc"]), ("macro_f1", "Macro-F1", None), ("micro_auc", "Micro-AUC", SC["micro_auc"])]
axes = []
for j, (key, lab, ref) in enumerate(metrics):
    ax = fig.add_subplot(gs[0, j * 1:(j + 1) * 1 + (1 if j == 0 else 0)] if False else gs[0, [slice(0, 2), slice(2, 3), slice(3, 4)][j]])
    axes.append(ax)
    vals = [R[a][key] for a, _, _ in arms]
    lo = min(vals) - 0.05
    ax.hlines(ys, lo, vals, color="#D9D9D9", lw=1.0, zorder=1)
    ax.scatter(vals, ys, c=[c for _, _, c in arms], s=26, zorder=3, edgecolor="white", lw=0.4)
    for v, y in zip(vals, ys):
        ax.text(v + 0.006, y + 0.28, f"{v:.3f}", fontsize=5.6, va="bottom", ha="left", color="#333")
    if ref:
        ax.axvline(ref, color=PAL["ref"], ls="--", lw=0.7)
        ax.text(ref, len(arms) - 0.1, f"SpeechCARE\n{ref:.4f}", fontsize=5.4, ha="center", va="bottom", bbox=dict(fc="white", ec="none", pad=1))
    ax.set_xlim(lo, max(max(vals), ref or 0) + 0.06)
    ax.set_ylim(-0.6, len(arms) + 0.6)
    ax.set_xlabel(lab); ax.grid(axis="x", color=PAL["grid"], lw=0.4); ax.set_axisbelow(True)
    if j == 0:
        ax.set_yticks(ys); ax.set_yticklabels([l for _, l, _ in arms])
        panel(ax, "a", x=-0.62, y=1.12)
        ax.set_title("PREPARE official test (n=412, mean of 3 seeds)", fontsize=6.8)
    else:
        ax.set_yticks([])
        ax.spines["left"].set_visible(False)

# panel b recall
bx = fig.add_subplot(gs[0, 4:6]); panel(bx, "b", x=-0.2, y=1.12)
sel = arms
w = 0.26
x = np.arange(len(sel))
for k, cl in enumerate(["HC", "MCI", "AD"]):
    bx.bar(x + (k - 1) * w, [R[a]["recall_HC_MCI_AD"][k] for a, _, _ in sel], w,
           color=CLASS_COL[cl], label=cl, lw=0)
bx.set_xticks(x); bx.set_xticklabels(["A", "A+t", "E+d", "S", "S+W", "C", "B"])
bx.set_ylabel("Recall"); bx.set_ylim(0, 1); bx.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), handlelength=1)
bx.set_title("Per-class recall", fontsize=6.8, pad=10)
bx.grid(axis="y", color=PAL["grid"], lw=0.4); bx.set_axisbelow(True)

# panel c historical per-dataset (locked 2026-09 evaluation; macro-AUROC; source: manuscript Table 1)
H = [("ADReSS 2020", 0.643, 0.530, 0.978), ("ADReSSo diag.", 0.614, 0.634, 0.857),
     ("ADReSSo prog.", 0.741, 0.759, 0.556), ("Pitt (bias audit)", 0.453, 0.582, 0.897),
     ("DementiaNet (n=6)", 0.500, 0.625, 0.500), ("IAEAV (bias audit)", 0.755, 0.704, 0.939),
     ("NCMMSC (bias audit)", 0.898, 0.588, 0.910), ("PREPARE", 0.728, 0.584, 0.811),
     ("PROCESS-2", 0.737, 0.756, 0.862), ("TAUKADIAL", 0.459, 0.625, 0.516)]
cx = fig.add_subplot(gs[1, 0:4]); panel(cx, "c", x=-0.07, y=1.1)
xs = np.arange(len(H)); w = 0.26
for k, (lab, col, idx) in enumerate([("B1 acoustic ML", PAL["A"], 1), ("B2 plain LLM", PAL["B"], 2), ("Ours (framework)", PAL["S"], 3)]):
    cx.bar(xs + (k - 1) * w, [h[idx] for h in H], w, color=col, label=lab, lw=0)
cx.axhline(0.5, color="#999", lw=0.5, ls=":")
cx.set_xticks(xs); cx.set_xticklabels([h[0] for h in H], rotation=30, ha="right")
cx.set_ylabel("Macro-AUROC"); cx.set_ylim(0.4, 1.02)
cx.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12))
cx.set_title("Historical pilot (locked 2026-09 evaluation, 745 held-out units; not the v2 pipeline)",
             fontsize=6.4, pad=14)

# panel d placeholder
dx = fig.add_subplot(gs[1, 4:6]); panel(dx, "d", x=-0.1, y=1.1)
dx.set_xticks([]); dx.set_yticks([])
for s in dx.spines.values():
    s.set_linestyle((0, (3, 3))); s.set_color("#999")
dx.spines["top"].set_visible(True); dx.spines["right"].set_visible(True)
dx.text(0.5, 0.58, "GPT / Claude / DeepSeek\nmeasurement comparison", ha="center", va="center", fontsize=7)
dx.text(0.5, 0.36, "pending (only GPT-5.5 completed)", ha="center", va="center", fontsize=6.2, color=PAL["C"])
dx.set_title("Model comparison", fontsize=6.8, pad=14)
save(fig, "Fig3_results")
