from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from jinja2 import Environment, FileSystemLoader, select_autoescape
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from .config import ProjectPaths
from .utils import now_utc


FOCUS_DATASETS = {
    "ADReSSo_2021_progression": "ADReSSo progression",
    "TAUKADIAL": "TAUKADIAL",
    "PREPARE_DrivenData": "PREPARE",
}
CONDITION_FILES = {
    "B1": "b1_predictions.csv",
    "B2": "b2_predictions.csv",
    "B3 base": "condition_c_base_predictions.csv",
    "B3": "ours_predictions.csv",
}
COLORS = {"B1": "#8A8F98", "B2": "#E68178", "B3 base": "#86A7A1", "B3": "#278C82"}


def _style() -> None:
    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#738087",
            "axes.labelcolor": "#263238",
            "xtick.color": "#45545B",
            "ytick.color": "#45545B",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def _save(fig: plt.Figure, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    if frame.empty:
        return {
            "n": 0.0,
            "macro_auroc": float("nan"),
            "accuracy": float("nan"),
            "macro_f1": float("nan"),
        }
    labels = sorted(frame["label"].astype(str).unique())
    if len(labels) < 2:
        return {
            "n": float(len(frame)),
            "macro_auroc": float("nan"),
            "accuracy": float(
                accuracy_score(frame["label"], frame["predicted_label"])
            ),
            "macro_f1": float(
                f1_score(
                    frame["label"],
                    frame["predicted_label"],
                    average="macro",
                    zero_division=0,
                )
            ),
        }
    probabilities = frame[[f"prob_{label}" for label in labels]].to_numpy(float)
    truth = frame["label"].astype(str).to_numpy()
    if len(labels) == 2:
        auc = roc_auc_score((truth == labels[1]).astype(int), probabilities[:, 1])
    else:
        indicator = np.column_stack([(truth == label).astype(int) for label in labels])
        auc = roc_auc_score(indicator, probabilities, average="macro", multi_class="ovr")
    return {
        "n": float(len(frame)),
        "macro_auroc": float(auc),
        "accuracy": float(accuracy_score(truth, frame["predicted_label"].astype(str))),
        "macro_f1": float(f1_score(truth, frame["predicted_label"].astype(str), average="macro")),
    }


def _micro_metrics(frame: pd.DataFrame) -> dict[str, float]:
    labels = sorted(frame["label"].astype(str).unique())
    truth = frame["label"].astype(str).to_numpy()
    indicator = np.column_stack([(truth == label).astype(int) for label in labels])
    probabilities = frame[[f"prob_{label}" for label in labels]].to_numpy(float)
    return {
        "micro_auroc": float(roc_auc_score(indicator, probabilities, average="micro", multi_class="ovr")),
        "micro_f1": float(f1_score(truth, frame["predicted_label"].astype(str), average="micro")),
    }


def _normalized_language(value: Any) -> str:
    key = str(value).strip().lower()
    return {
        "english": "en",
        "spanish": "es",
        "mandarin": "zh",
        "chinese": "zh",
    }.get(key, key)


def _focus_metrics(paths: ProjectPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id, dataset_name in FOCUS_DATASETS.items():
        artifact_dir = paths.artifacts / dataset_id
        for condition, filename in CONDITION_FILES.items():
            frame = pd.read_csv(artifact_dir / filename, dtype={"subject_id": str})
            rows.append(
                {
                    "dataset_id": dataset_id,
                    "dataset": dataset_name,
                    "condition": condition,
                    **_prediction_metrics(frame),
                }
            )
    return pd.DataFrame(rows)


def _agent_audit(paths: ProjectPaths) -> pd.DataFrame:
    rows = []
    for dataset_id, dataset_name in FOCUS_DATASETS.items():
        artifact_dir = paths.artifacts / dataset_id
        base = pd.read_csv(artifact_dir / "condition_c_base_predictions.csv", dtype={"subject_id": str})
        final = pd.read_csv(artifact_dir / "ours_predictions.csv", dtype={"subject_id": str})
        paired = base[["subject_id", "label", "predicted_label"]].merge(
            final[["subject_id", "predicted_label"]],
            on="subject_id",
            suffixes=("_base", "_agent"),
            validate="one_to_one",
        )
        changed = paired[paired["predicted_label_base"] != paired["predicted_label_agent"]]
        helped = int(
            (
                (changed["predicted_label_agent"] == changed["label"])
                & (changed["predicted_label_base"] != changed["label"])
            ).sum()
        )
        hurt = int(
            (
                (changed["predicted_label_agent"] != changed["label"])
                & (changed["predicted_label_base"] == changed["label"])
            ).sum()
        )
        base_metrics = _prediction_metrics(base)
        final_metrics = _prediction_metrics(final)
        rows.append(
            {
                "dataset_id": dataset_id,
                "dataset": dataset_name,
                "n": len(paired),
                "changed": len(changed),
                "helped": helped,
                "hurt": hurt,
                "changed_but_still_wrong": int(len(changed) - helped - hurt),
                "delta_auroc": final_metrics["macro_auroc"] - base_metrics["macro_auroc"],
                "delta_accuracy": final_metrics["accuracy"] - base_metrics["accuracy"],
                "delta_macro_f1": final_metrics["macro_f1"] - base_metrics["macro_f1"],
            }
        )
    return pd.DataFrame(rows)


def _state_stability(paths: ProjectPaths) -> pd.DataFrame:
    frame = pd.read_csv(paths.artifacts / "ADReSSo_2021_progression" / "state_wide.csv")
    rows = []
    for state in [column for column in frame.columns if column.startswith("state_")]:
        values: dict[str, float] = {}
        for split in ["train", "test"]:
            subset = frame[frame["split"].eq(split)].dropna(subset=[state])
            truth = subset["label"].eq("decline").astype(int)
            values[split] = float(roc_auc_score(truth, subset[state]))
        rows.append(
            {
                "state": state.replace("state_", ""),
                "train_auroc_same_direction": values["train"],
                "test_auroc_same_direction": values["test"],
                "direction_reversed": bool((values["train"] - 0.5) * (values["test"] - 0.5) < 0),
            }
        )
    return pd.DataFrame(rows)


def _language_subgroups(paths: ProjectPaths) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for dataset_id in ["TAUKADIAL", "PREPARE_DrivenData"]:
        artifact_dir = paths.artifacts / dataset_id
        analysis = pd.read_csv(artifact_dir / "analysis_manifest.csv", dtype={"subject_id": str})
        metadata = (
            analysis.assign(language=analysis["language"].map(_normalized_language))
            .groupby("subject_id", as_index=False)
            .agg(language=("language", lambda values: values.mode().iloc[0]))
        )
        for condition, filename in CONDITION_FILES.items():
            frame = pd.read_csv(artifact_dir / filename, dtype={"subject_id": str}).merge(
                metadata, on="subject_id", validate="one_to_one"
            )
            for language, group in frame.groupby("language"):
                if len(group) < 5:
                    continue
                if group["label"].nunique() > 1:
                    metrics = _prediction_metrics(group)
                else:
                    metrics = {
                        "macro_auroc": np.nan,
                        "accuracy": float(accuracy_score(group["label"], group["predicted_label"])),
                        "macro_f1": float(f1_score(group["label"], group["predicted_label"], average="macro")),
                    }
                rows.append(
                    {
                        "dataset_id": dataset_id,
                        "dataset": FOCUS_DATASETS[dataset_id],
                        "condition": condition,
                        "language": language,
                        "n": len(group),
                        "label_count": int(group["label"].nunique()),
                        **metrics,
                    }
                )
    return pd.DataFrame(rows)


def _data_quality_audit(paths: ProjectPaths) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for dataset_id in FOCUS_DATASETS:
        artifact_dir = paths.artifacts / dataset_id
        manifest = pd.read_csv(artifact_dir / "manifest.csv", dtype={"subject_id": str})
        analysis = pd.read_csv(artifact_dir / "analysis_manifest.csv", dtype={"subject_id": str})
        joined = manifest[["case_id", "language"]].merge(
            analysis[["case_id", "language"]], on="case_id", suffixes=("_manifest", "_analysis")
        )
        trusted = joined["language_manifest"].map(_normalized_language)
        detected = joined["language_analysis"].map(_normalized_language)
        multilingual = trusted.isin(["zh-en", "multilingual", "unknown", "unspecified", ""])
        features = pd.read_csv(artifact_dir / "recording_features.csv")
        output[dataset_id] = {
            "subjects": int(manifest["subject_id"].nunique()),
            "recordings": int(len(manifest)),
            "task_types": int(manifest["task_type"].nunique()),
            "task_type_names": sorted(manifest["task_type"].astype(str).unique()),
            "language_overrides": int(((trusted != detected) & ~multilingual).sum()),
            "fixed_text_reliability_rate": float(features["text_reliability"].eq(0.75).mean()),
            "zero_pause_rate": float(features["long_pause_rate_min"].eq(0.0).mean()),
            "zero_pronoun_rate": float(features["pronoun_ratio"].eq(0.0).mean()),
            "content_ratio_one_rate": float(features["content_word_ratio"].eq(1.0).mean()),
        }
    return output


def _plot_focus_performance(metrics: pd.DataFrame, path: Path) -> None:
    _style()
    datasets = list(FOCUS_DATASETS.values())
    conditions = list(CONDITION_FILES)
    fig, ax = plt.subplots(figsize=(12.8, 5.8))
    x = np.arange(len(datasets))
    width = 0.19
    for index, condition in enumerate(conditions):
        values = [
            float(metrics[metrics["dataset"].eq(dataset) & metrics["condition"].eq(condition)]["macro_auroc"].iloc[0])
            for dataset in datasets
        ]
        bars = ax.bar(x + (index - 1.5) * width, values, width, label=condition, color=COLORS[condition])
        for bar, value in zip(bars, values, strict=True):
            ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.3f}", ha="center", fontsize=9)
    ax.axhline(0.5, color="#AAB3B1", linestyle="--", linewidth=1)
    ax.set_xticks(x, datasets)
    ax.set_ylim(0.35, 0.95)
    ax.set_ylabel("Macro AUROC")
    ax.set_title("Where B3 still fails: ranking performance on the three focus tasks", loc="left", fontweight="bold")
    ax.legend(ncol=4, frameon=False, loc="upper center")
    _save(fig, path)


def _plot_agent_audit(audit: pd.DataFrame, path: Path) -> None:
    _style()
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 5.2), gridspec_kw={"width_ratios": [1.2, 1]})
    x = np.arange(len(audit))
    axes[0].bar(x, audit["helped"], color="#278C82", label="Helped")
    axes[0].bar(x, audit["hurt"], bottom=audit["helped"], color="#D96C61", label="Hurt")
    axes[0].bar(
        x,
        audit["changed_but_still_wrong"],
        bottom=audit["helped"] + audit["hurt"],
        color="#B8C0BD",
        label="Changed, still wrong",
    )
    axes[0].set_xticks(x, audit["dataset"])
    axes[0].set_ylabel("Number of test subjects")
    axes[0].set_title("Label changes caused by Agent correction", loc="left", fontweight="bold")
    axes[0].legend(frameon=False)
    deltas = audit[["delta_auroc", "delta_accuracy", "delta_macro_f1"]].to_numpy()
    image = axes[1].imshow(deltas, aspect="auto", cmap="RdYlGn", vmin=-0.08, vmax=0.08)
    axes[1].set_xticks(np.arange(3), ["AUROC", "Accuracy", "Macro F1"])
    axes[1].set_yticks(np.arange(len(audit)), audit["dataset"])
    axes[1].set_title("B3 final minus B3 base", loc="left", fontweight="bold")
    for row in range(deltas.shape[0]):
        for column in range(deltas.shape[1]):
            axes[1].text(column, row, f"{deltas[row, column]:+.3f}", ha="center", va="center", fontsize=9)
    fig.colorbar(image, ax=axes[1], fraction=0.046, pad=0.04)
    _save(fig, path)


def _plot_state_stability(stability: pd.DataFrame, path: Path) -> None:
    _style()
    fig, ax = plt.subplots(figsize=(10.8, 5.2))
    x = np.arange(len(stability))
    ax.plot(x, stability["train_auroc_same_direction"], marker="o", color="#527AA3", label="Train")
    ax.plot(x, stability["test_auroc_same_direction"], marker="s", color="#D96C61", label="Test")
    ax.axhline(0.5, color="#8D9894", linestyle="--", linewidth=1)
    for index, reversed_direction in enumerate(stability["direction_reversed"]):
        if reversed_direction:
            ax.text(index, 0.94, "REV", ha="center", va="top", color="#B44C43", fontsize=8, fontweight="bold")
    ax.set_xticks(x, stability["state"])
    ax.set_ylim(0.15, 0.95)
    ax.set_ylabel("Univariate AUROC, same direction")
    ax.set_title("ADReSSo progression: state direction stability", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, path)


def _plot_prepare_benchmark(paths: ProjectPaths, path: Path) -> dict[str, float]:
    benchmark = yaml.safe_load((paths.configs / "benchmarks" / "speechcare.yaml").read_text(encoding="utf-8"))
    speechcare = benchmark["prepare_official_test"]["metrics"]
    ours = pd.read_csv(paths.artifacts / "PREPARE_DrivenData" / "ours_predictions.csv")
    ours_micro = _micro_metrics(ours)
    metrics = {
        "B3 micro AUROC": ours_micro["micro_auroc"],
        "SpeechCARE micro AUROC": float(speechcare["micro_auroc_ovr"]["value"]),
        "B3 F1": ours_micro["micro_f1"],
        "SpeechCARE F1": float(speechcare["reported_f1"]["value"]),
    }
    _style()
    fig, ax = plt.subplots(figsize=(9.6, 4.8))
    labels = ["Micro AUROC", "F1"]
    b3 = [metrics["B3 micro AUROC"], metrics["B3 F1"]]
    reference = [metrics["SpeechCARE micro AUROC"], metrics["SpeechCARE F1"]]
    x = np.arange(2)
    bars1 = ax.bar(x - 0.18, b3, 0.36, color=COLORS["B3"], label="B3 ADvoice")
    bars2 = ax.bar(x + 0.18, reference, 0.36, color="#5B6F9B", label="SpeechCARE")
    for bars in [bars1, bars2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.012, f"{bar.get_height():.3f}", ha="center")
    ax.set_xticks(x, labels)
    ax.set_ylim(0.55, 0.92)
    ax.set_ylabel("Score")
    ax.set_title("PREPARE official test: corrected primary-source benchmark", loc="left", fontweight="bold")
    ax.legend(frameon=False)
    _save(fig, path)
    return metrics


def build_failure_analysis(paths: ProjectPaths) -> Path:
    output_dir = paths.reports / "latest" / "failure_mode_analysis"
    assets = output_dir / "assets"
    output_dir.mkdir(parents=True, exist_ok=True)
    assets.mkdir(parents=True, exist_ok=True)

    focus = _focus_metrics(paths)
    agent = _agent_audit(paths)
    stability = _state_stability(paths)
    language = _language_subgroups(paths)
    quality = _data_quality_audit(paths)
    benchmark = _plot_prepare_benchmark(paths, assets / "prepare_speechcare_gap.png")
    _plot_focus_performance(focus, assets / "focus_task_performance.png")
    _plot_agent_audit(agent, assets / "agent_correction_audit.png")
    _plot_state_stability(stability, assets / "progression_state_stability.png")

    focus.to_csv(output_dir / "focus_condition_metrics.csv", index=False)
    agent.to_csv(output_dir / "agent_correction_audit.csv", index=False)
    stability.to_csv(output_dir / "progression_state_direction_stability.csv", index=False)
    language.to_csv(output_dir / "language_subgroup_metrics.csv", index=False)
    (output_dir / "data_quality_audit.json").write_text(
        json.dumps(quality, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    get_agent = lambda dataset_id: agent[agent["dataset_id"].eq(dataset_id)].iloc[0].to_dict()
    env = Environment(loader=FileSystemLoader(paths.root / "templates"), autoescape=select_autoescape(["html"]))
    output = output_dir / "failure_mode_root_cause_report.html"
    output.write_text(
        env.get_template("failure_mode_report.html").render(
            generated_at=now_utc(),
            progression_agent=get_agent("ADReSSo_2021_progression"),
            tau_agent=get_agent("TAUKADIAL"),
            prepare_agent=get_agent("PREPARE_DrivenData"),
            reversed_states=stability[stability["direction_reversed"]]["state"].tolist(),
            tau_subgroups=language[
                language["dataset_id"].eq("TAUKADIAL") & language["condition"].isin(["B2", "B3"])
            ].to_dict("records"),
            prepare_subgroups=language[
                language["dataset_id"].eq("PREPARE_DrivenData") & language["condition"].eq("B3")
            ].to_dict("records"),
            quality=quality,
            benchmark=benchmark,
            assets="assets",
        ),
        encoding="utf-8",
    )
    return output
