import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from style import PAL, setup, save

setup()
fig, ax = plt.subplots(figsize=(7.3, 4.3))
ax.set_xlim(0, 100); ax.set_ylim(0, 62); ax.axis("off")

def box(x, y, w, h, text, fc="#FFFFFF", ec="#555555", fs=6.3, lw=0.7, bold=False, tc="#111111"):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.25,rounding_size=0.8",
                       fc=fc, ec=ec, lw=lw)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal", color=tc, linespacing=1.25)

def arrow(x0, y0, x1, y1, col="#444444", lw=0.8, style="-|>", ls="-", rad=0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=7,
                                 lw=lw, color=col, ls=ls, connectionstyle=f"arc3,rad={rad}"))

# governance layer (background band)
ax.add_patch(FancyBboxPatch((0.5, 2.2), 99, 5.6, boxstyle="round,pad=0.2,rounding_size=1",
                            fc="#EFEBF6", ec=PAL["gov"], lw=0.8))
ax.text(2, 6.4, "Governance layer", fontsize=6.8, fontweight="bold", color=PAL["gov"], va="center")
gov = ["Provenance\n(source span IDs)", "Observability\n(per-step trace)",
       "Permissions\n(no QC-as-disease)", "Revision closure\n(snapshot + re-score)"]
for i, g in enumerate(gov):
    box(22 + i * 19, 2.9, 17.5, 3.9, g, fc="#FFFFFF", ec=PAL["gov"], fs=5.8, lw=0.5)

# main pipeline
y0, h = 14, 34
cols = [(1.0, 11.5), (15, 15), (33.5, 13), (50, 13), (66.5, 14.5), (84.5, 14.5)]
heads = ["Input", "Measurement", "MetricEvidence", "StateCards",
         "Prediction", "Output"]
for (x, w), t in zip(cols, heads):
    ax.text(x + w / 2, y0 + h + 0.8, t, ha="center", fontsize=7, fontweight="bold")

# input
box(*cols[0][:1], y0 + 20, cols[0][1], 13, "Audio\n(task, language)", fc="#F4F4F4")
box(cols[0][0], y0 + 3, cols[0][1], 13, "Transcript\nhuman / ASR\n+ speaker roles", fc="#F4F4F4")

# measurement
mx, mw = cols[1]
box(mx, y0 + 25, mw, 8.5, "Acoustic\n(pause, rate, F0 ...)", fc="#EDEDED")
box(mx, y0 + 15.5, mw, 8.5, "Linguistic\n(lexical, syntactic)", fc="#EDEDED")
box(mx, y0 + 6.5, mw, 8, "Encoders: Whisper,\nmHuBERT, text", fc="#E3ECF5", ec=PAL["S"])
box(mx, y0 - 1.5, mw, 7, "LLM agent measure\n(content units, drift)", fc="#F7E6E4", ec=PAL["C"])

# evidence
ex, ew = cols[2]
box(ex, y0 + 6, ew, 27, "MetricEvidence\n\nvalue, unit,\nsource span,\nquality flag,\nmeasurer ID", fc="#FFFFFF", bold=False)

# state cards
sx, sw = cols[3]
box(sx, y0 + 6, sw, 27, "Task-conditioned\nStateCards\n\nz vs healthy\nreference\n(task x language\nx age)", fc="#E3ECF5", ec=PAL["S"])

# prediction
px, pw = cols[4]
box(px, y0 + 20.5, pw, 12.5, "Head 1\nstate contributions\n(decomposable)", fc="#E3ECF5", ec=PAL["S"])
box(px, y0 + 4.5, pw, 12.5, "Head 2\nencoder residual", fc="#EDEDED")
box(px - 0.0, y0 - 5, pw, 6.5, "Calibration + lock", fc="#FFFFFF", bold=True, fs=6)
ax.text(px + pw / 2, y0 + 18.8, "+", ha="center", va="center", fontsize=9)

# output
ox, ow = cols[5]
box(ox, y0 + 6, ow, 27, "Clinician report\n\nclass probabilities\ncontribution table\ntrace map to\nsource spans\nlimitations", fc="#FFFFFF", bold=False)

# arrows
for (xa, wa), (xb, _) in zip(cols[:-1], cols[1:]):
    arrow(xa + wa + 0.5, y0 + 19.5, xb - 0.5, y0 + 19.5)
arrow(px + pw / 2, y0 + 4.5, px + pw / 2, y0 + 1.5, lw=0.7)
arrow(px + pw + 0.4, y0 - 1.8, ox + ow / 2, y0 + 5.7, rad=0.0, lw=0.7)
# governance links
for xx in [6.7, 22.5, 40, 56.5, 91.7]:
    arrow(xx, 7.9, xx, y0 - 1.5 if xx > 14 and xx < 90 else y0 + 2.5, col=PAL["gov"], lw=0.5, style="-", ls=(0, (2, 2)))

# comparison conditions (brackets above)
def cond(x0, x1, y, label, col):
    ax.plot([x0, x1], [y, y], color=col, lw=2.2, solid_capstyle="round")
    ax.text((x0 + x1) / 2, y + 1.2, label, ha="center", fontsize=6.2, color=col, fontweight="bold")
cond(1.0, 12.5, 56.5, "B  plain LLM\n(transcript only)", PAL["B"])
cond(15, 30, 56.5, "A  acoustic ML\n(handcrafted features)", PAL["A"])
cond(33.5, 98.5, 56.5, "C  evidence-governed pipeline (framework, with or without LLM measurement)", PAL["C"])
save(fig, "Fig1_architecture")
