#!/usr/bin/env python3
"""Generate the compact figures used by the Evidence Governance meeting report."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "evidence_governance_meeting_response_2026-09-17"

INK = "#17212b"
BLUE = "#2468a9"
GREEN = "#34785b"
AMBER = "#ad6a12"
RED = "#a94444"
GREY = "#9aa8b4"


def _style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.labelcolor": INK,
            "axes.edgecolor": "#9aa8b4",
            "axes.titlecolor": INK,
            "xtick.color": INK,
            "ytick.color": INK,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def plot_redundancy() -> None:
    data = pd.read_csv(REPORT / "cross_stratum_redundancy_summary.csv")
    data["label"] = data.metric_a.str.replace("_", " ") + "  /  " + data.metric_b.str.replace("_", " ")
    data = data.sort_values("median_absolute_spearman", ascending=True)
    strict = data.exact_on_observed_rows | data.complement_formula
    colors = np.where(strict, RED, BLUE)

    fig, axis = plt.subplots(figsize=(11.6, 6.4))
    bars = axis.barh(data.label, data.median_absolute_spearman, color=colors, height=0.64)
    for bar, rho, strata in zip(
        bars, data.median_absolute_spearman, data.n_dataset_language_strata, strict=True
    ):
        axis.text(
            rho - 0.008,
            bar.get_y() + bar.get_height() / 2,
            f"|rho|={rho:.3f}  ({int(strata)} strata)",
            ha="right",
            va="center",
            color="white",
            fontsize=10,
            fontweight="bold",
        )
    axis.axvline(0.90, color=AMBER, linestyle="--", linewidth=1.5, label="review threshold = 0.90")
    axis.set_xlim(0.88, 1.005)
    axis.set_xlabel("Median absolute Spearman correlation across eligible dataset-language strata")
    axis.set_title("Repeated redundancy families are reproducible across data strata", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.16)
    axis.legend(frameon=False, loc="lower right")
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(REPORT / "cross_dataset_redundancy_families.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_selection_instability() -> None:
    data = pd.read_csv(REPORT / "selection_instability_summary.csv")
    names = {
        "ADReSS_2020": "ADReSS 2020",
        "ADReSSo_2021_progression": "ADReSSo progression",
        "NCMMSC2021_AD": "NCMMSC2021 long",
        "DementiaNet_PublicFigures": "DementiaNet (test n=6)",
    }
    data["label"] = data.dataset_id.map(names)
    data = data.iloc[::-1].reset_index(drop=True)
    y = np.arange(len(data))

    fig, axis = plt.subplots(figsize=(11.6, 6.4))
    for row, ypos in zip(data.itertuples(index=False), y, strict=True):
        axis.plot(
            [row.cv_balanced_accuracy, row.selected_test_balanced_accuracy],
            [ypos, ypos],
            color=GREY,
            linewidth=3,
            zorder=1,
        )
    axis.scatter(data.cv_balanced_accuracy, y, s=105, color=BLUE, label="Training CV at selected K", zorder=3)
    axis.scatter(data.selected_test_balanced_accuracy, y, s=105, color=RED, marker="s", label="Locked test at same K", zorder=3)
    for row, ypos in zip(data.itertuples(index=False), y, strict=True):
        cv_ha = "left" if row.cv_balanced_accuracy <= row.selected_test_balanced_accuracy else "right"
        cv_dx = 0.012 if cv_ha == "left" else -0.012
        test_ha = "right" if row.selected_test_balanced_accuracy <= row.cv_balanced_accuracy else "left"
        test_dx = -0.012 if test_ha == "right" else 0.012
        axis.text(row.cv_balanced_accuracy + cv_dx, ypos + 0.13, f"CV {row.cv_balanced_accuracy:.3f}", ha=cv_ha, fontsize=9, color=BLUE)
        axis.text(row.selected_test_balanced_accuracy + test_dx, ypos - 0.18, f"test {row.selected_test_balanced_accuracy:.3f}", ha=test_ha, fontsize=9, color=RED)
        axis.text(1.005, ypos, f"K={int(row.cv_selected_count)}", va="center", fontsize=10, fontweight="bold")
    axis.set_yticks(y, data.label)
    axis.set_xlim(0.30, 1.06)
    axis.set_xlabel("Balanced accuracy")
    axis.set_title("Feature-count choices do not transfer consistently from CV to locked tests", loc="left", fontweight="bold")
    axis.grid(axis="x", alpha=0.16)
    axis.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    axis.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(REPORT / "feature_selection_instability.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    _style()
    REPORT.mkdir(parents=True, exist_ok=True)
    plot_redundancy()
    plot_selection_instability()


if __name__ == "__main__":
    main()
