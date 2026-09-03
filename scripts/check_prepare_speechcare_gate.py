from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

from advoice.cognitive_extension import LABELS, benchmark_metrics, normalize_probability


PUBLISHED_SPEECHCARE = {
    "micro_auroc_ovr": 0.8683,
    "micro_f1": 0.7211,
    "weighted_auroc_ovr": 0.8067,
    "micro_auprc": 0.7473,
    "weighted_auprc": 0.7350,
}


def evaluate_prediction_file(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path, dtype={"subject_id": str})
    truth_column = "true_label" if "true_label" in frame else "label"
    label_index = {label: index for index, label in enumerate(LABELS)}
    truth = frame[truth_column].astype(str).map(label_index)
    if truth.isna().any():
        raise ValueError(f"Unexpected labels in {path}.")
    probability = normalize_probability(
        frame[[f"prob_{label}" for label in LABELS]].to_numpy(dtype=float)
    )
    return benchmark_metrics(truth.to_numpy(dtype=int), probability)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--prediction",
        type=Path,
        help="Defaults to the 9.2 Agent-on PREPARE prediction artifact.",
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--enforce",
        action="store_true",
        help="Exit non-zero when any retrospective PREPARE benchmark gate fails.",
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    prediction = arguments.prediction or (
        root / "artifacts" / "PREPARE_DrivenData" / "ours_predictions.csv"
    )
    metrics = evaluate_prediction_file(prediction)
    comparisons = {
        metric: {
            "advoice": float(metrics[metric]),
            "speechcare_published_mean": float(reference),
            "delta": float(metrics[metric] - reference),
            "passed": bool(metrics[metric] > reference),
        }
        for metric, reference in PUBLISHED_SPEECHCARE.items()
    }
    gate_passed = all(item["passed"] for item in comparisons.values())
    extension_result_path = (
        root
        / "artifacts"
        / "PREPARE_DrivenData"
        / "speechcare_cognitive_extension"
        / "result.json"
    )
    extension_comparisons: dict[str, dict[str, float | bool]] = {}
    if extension_result_path.exists():
        extension_result = json.loads(extension_result_path.read_text(encoding="utf-8"))
        extension_metrics = extension_result["speechcare_plus_cognition"]
        extension_comparisons = {
            metric: {
                "speechcare_plus_advoice_cognition": float(extension_metrics[metric]),
                "speechcare_published_mean": float(reference),
                "delta": float(extension_metrics[metric] - reference),
                "passed": bool(extension_metrics[metric] > reference),
            }
            for metric, reference in PUBLISHED_SPEECHCARE.items()
        }
    extension_passed = bool(extension_comparisons) and all(
        bool(item["passed"]) for item in extension_comparisons.values()
    )
    result = {
        "protocol": "PREPARE official 412-case test; HC/MCI/AD",
        "prediction_path": str(prediction),
        "development_superiority_gate_passed": gate_passed,
        "confirmatory_superiority_claim_allowed": False,
        "other_dataset_expansion_allowed_by_user_gate": gate_passed,
        "comparison": comparisons,
        "retrospective_same_backbone_extension": {
            "available": bool(extension_comparisons),
            "all_published_endpoints_exceeded": extension_passed,
            "comparison": extension_comparisons,
            "claim_boundary": (
                "This isolates the incremental value of the cognitive representation "
                "on top of released SpeechCARE probabilities. It is not a standalone "
                "ADvoice result and is not a confirmatory superiority test."
            ),
        },
        "warning": (
            "This official test set was inspected during historical development. "
            "Passing this gate would remain retrospective and requires confirmation "
            "on a newly locked external, temporal, or site-held-out cohort."
        ),
    }
    output = arguments.output or (
        root / "artifacts" / "PREPARE_DrivenData" / "speechcare_gate_9_2.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if arguments.enforce and not gate_passed:
        sys.exit(2)


if __name__ == "__main__":
    main()
