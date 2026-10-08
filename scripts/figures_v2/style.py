"""Shared Nature-style settings and palette for ADvoice v2 figures."""
import json
import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "paper" / "overleaf_v2_2026-10-08" / "figures" / "v2"
V2 = ROOT / ".local" / "v2"

PAL = {
    "A": "#7F7F7F",       # conventional acoustic / handcrafted
    "A2": "#B3B3B3",      # acoustic + text
    "E": "#4E9A8E",       # frozen embeddings (+demo)
    "S": "#3B6FA0",       # evidence-state framework
    "S2": "#1F3F66",      # framework + Whisper encoder
    "C": "#B5403A",       # framework + LLM measurement
    "B": "#D08C2F",       # plain LLM
    "ref": "#222222",
    "gov": "#6B5B95",
    "light": "#E9EEF4",
    "grid": "#E6E6E6",
}
CLASS_COL = {"HC": "#4E9A8E", "MCI": "#D6A04B", "AD": "#B5403A"}

def setup():
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 7.5, "axes.labelsize": 7,
        "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6.5,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "xtick.major.size": 2.5, "ytick.major.size": 2.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False, "pdf.fonttype": 42, "ps.fonttype": 42,
        "savefig.dpi": 300, "figure.dpi": 150, "axes.titlelocation": "left",
        "axes.titlepad": 4,
    })

def panel(ax, letter, x=-0.12, y=1.06):
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=10, fontweight="bold",
            va="bottom", ha="left")

def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(OUT / f"{name}.png", bbox_inches="tight", dpi=300)
    plt.close(fig)

def jload(rel):
    return json.load(open(V2 / rel))
