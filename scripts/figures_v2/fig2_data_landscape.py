import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from style import PAL, CLASS_COL, setup, save, panel

setup()
# source: configs/datasets/*.yaml; held-out n from the locked 2026-09 evaluation (manuscript Table 1)
D = [
 # name, lang, task/channel, labels, transcript, roles, heldout_n
 ("ADReSS 2020",        "en",    "Picture description",        "HC/AD",     "Human (CHAT)", "Yes",         27),
 ("ADReSSo 2021 diag.", "en",    "Picture description (audio)", "HC/AD",    "ASR only",     "Segmented",   42),
 ("ADReSSo 2021 prog.", "en",    "Longitudinal pict. descr.",  "stable/decl.", "ASR only",  "Segmented",   12),
 ("DementiaBank Pitt",  "en",    "Neuropsych. multitask",      "HC/AD",     "Human (CHAT)", "Yes",         59),
 ("DementiaNet",        "en",    "Public-figure speech",       "HC/AD",     "Local text",   "No",           6),
 ("IAEAV",              "es",    "Clinical interview",         "HC/AD",     "Human (patient)", "Segmented", 14),
 ("NCMMSC 2021",        "zh",    "Long picture description",   "HC/MCI/AD", "ASR only",     "No",          53),
 ("PREPARE",            "multi", "7 structured tasks",         "HC/MCI/AD", "ASR only",     "No",         412),
 ("PROCESS-2",          "en",    "Structured multitask",       "HC/MCI/AD", "Provided text","Per task",    80),
 ("TAUKADIAL",          "zh-en", "Spontaneous description",    "HC/MCI",    "ASR only",     "No",          40),
]
cols = ["Language", "Task / channel", "Labels", "Transcript", "Speaker roles"]
lang_c = {"en": "#3B6FA0", "es": "#D08C2F", "zh": "#B5403A", "multi": "#6B5B95", "zh-en": "#8C5A7A"}
tr_c = {"Human (CHAT)": "#4E9A8E", "Human (patient)": "#4E9A8E", "Provided text": "#8CC0B7",
        "Local text": "#BFD9D4", "ASR only": "#D6A04B"}
role_c = {"Yes": "#4E9A8E", "Segmented": "#8CC0B7", "Per task": "#BFD9D4", "No": "#E0E0E0"}
lab_c = {"HC/AD": "#DCE6F0", "HC/MCI": "#EBE4C9", "HC/MCI/AD": "#E6D3D1", "stable/decl.": "#E4E4E4"}

fig = plt.figure(figsize=(7.3, 4.3))
gs = fig.add_gridspec(1, 3, width_ratios=[3.0, 1.15, 0.9], wspace=0.06, left=0.0, right=1.0)
ax = fig.add_subplot(gs[0]); ax.set_xlim(0, 10); ax.set_ylim(len(D), -1.1); ax.axis("off")
panel(ax, "a", x=0.0, y=1.0)
xs = [0, 1.55, 3.05, 5.35, 6.95]; ws = [1.5, 1.45, 2.25, 1.55, 1.5, 1.5]
# row labels
for i, r in enumerate(D):
    ax.text(-0.15, i + 0.5, r[0], ha="right", va="center", fontsize=6.5)
# header
xs = [0.0, 1.0, 3.3, 5.0, 6.7, 8.4]  # language, task, labels, transcript, roles
cw = [0.95, 2.25, 1.65, 1.65, 1.65]
xs = [0.0, 1.0, 3.3, 5.0, 6.7]
for x, w, h in zip(xs, cw, cols):
    ax.text(x + w / 2, -0.55, h, ha="center", va="center", fontsize=6.5, fontweight="bold")
for i, r in enumerate(D):
    vals = [r[1], r[2], r[3], r[4], r[5]]
    cm = [lang_c[r[1]], "#F2F2F2", lab_c[r[3]], tr_c[r[4]], role_c[r[5]]]
    tc = ["white", "black", "black", "black", "black"]
    for x, w, v, c, t in zip(xs, cw, vals, cm, tc):
        ax.add_patch(Rectangle((x + 0.02, i + 0.06), w - 0.04, 0.88, fc=c, ec="white", lw=0.5))
        ax.text(x + w / 2, i + 0.5, v, ha="center", va="center", fontsize=5.6, color=t)
ax.set_xlim(-0.1, 8.4)
# panel b: held-out n (log) with PREPARE train
bx = fig.add_subplot(gs[1]); panel(bx, "b", x=-0.08, y=1.08)
names = [r[0] for r in D]; n = [r[6] for r in D]
bx.barh(range(len(D)), n, color=[lang_c[r[1]] for r in D], height=0.7)
bx.set_xscale("log"); bx.set_xlim(3, 1500); bx.set_ylim(len(D) - 0.5, -1.1 + 0.0)
bx.set_yticks([]); bx.set_xlabel("Held-out participants (log)")
for i, v in enumerate(n):
    bx.text(v * 1.12, i, str(v), va="center", fontsize=6)
bx.text(0.02, 0.0, "", transform=bx.transAxes)
bx.set_title("Held-out size", fontsize=6.8, loc="left", pad=3)
bx.spines["left"].set_visible(False)
bx.text(1.0, -0.17, "Train n: n/a except PREPARE (1,622)", transform=bx.transAxes,
        fontsize=5.4, ha="right", color="#555555")
# panel c: PREPARE composition
cx = fig.add_subplot(gs[2]); panel(cx, "c", x=-0.3, y=1.08)
tr = [899, 216, 507]; te = [229, 51, 132]
for k, (lab, v) in enumerate(zip(["Train\n(n=1,622)", "Test\n(n=412)"], [tr, te])):
    tot = sum(v); left = 0
    for cl, x in zip(["HC", "MCI", "AD"], v):
        cx.bar(k, x / tot, bottom=left / tot, color=CLASS_COL[cl], width=0.62, ec="white", lw=0.5)
        cx.text(k, (left + x / 2) / tot, f"{cl}\n{x}", ha="center", va="center", fontsize=5.8, color="white")
        left += x
cx.set_xticks([0, 1]); cx.set_xticklabels(["Train\nn=1,622", "Test\nn=412"])
cx.set_ylabel("Class fraction"); cx.set_ylim(0, 1)
cx.set_title("PREPARE labels", fontsize=6.8, loc="left", pad=3)
save(fig, "Fig2_data_landscape")
