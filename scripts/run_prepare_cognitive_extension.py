from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from advoice.cognitive_extension import (
    LABELS,
    benchmark_metrics,
    bounded_cognitive_extension,
    normalize_probability,
)


PUBLISHED_SPEECHCARE = {
    "micro_auroc_ovr": 0.8683,
    "micro_f1": 0.7211,
    "weighted_auroc_ovr": 0.8067,
    "micro_auprc": 0.7473,
    "weighted_auprc": 0.7350,
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--cognitive-weight", type=float, default=0.20)
    parser.add_argument(
        "--allow-unverified-legacy",
        action="store_true",
        help="Permit a legacy cognition prediction without a provenance sidecar.",
    )
    arguments = parser.parse_args()
    root = arguments.root.resolve()
    speechcare_path = root / "references" / "speechcare" / "released_outputs" / "mhubert_test_predictions_after_bias_mitigation.csv"
    cognitive_path = root / "artifacts" / "PREPARE_DrivenData" / "cognitive_fusion_protocol" / "official_test_predictions.csv"
    cognitive_meta_path = cognitive_path.with_suffix(".meta.json")
    if not cognitive_meta_path.exists() and not arguments.allow_unverified_legacy:
        raise RuntimeError(
            "Cognition prediction provenance is missing. Re-run the full official-test "
            "pipeline, or explicitly acknowledge the legacy artifact with "
            "--allow-unverified-legacy."
        )
    cognitive_meta = (
        json.loads(cognitive_meta_path.read_text(encoding="utf-8"))
        if cognitive_meta_path.exists()
        else {"status": "legacy_unverified"}
    )
    if cognitive_meta_path.exists() and cognitive_meta.get("prediction_sha256") != sha256(cognitive_path):
        raise RuntimeError("Cognition prediction hash does not match its provenance sidecar.")
    speechcare = pd.read_csv(speechcare_path, dtype={"uid": str})
    cognitive = pd.read_csv(cognitive_path, dtype={"subject_id": str})
    merged = cognitive.merge(
        speechcare[["uid", "C", "MCI", "ADRD"]],
        left_on="subject_id",
        right_on="uid",
        validate="one_to_one",
    )
    if len(merged) != 412:
        raise ValueError(f"Expected 412 official test subjects, found {len(merged)}.")
    label_index = {label: index for index, label in enumerate(LABELS)}
    y_true = merged["true_label"].map(label_index).to_numpy(dtype=int)
    backbone = normalize_probability(merged[["C", "MCI", "ADRD"]].to_numpy())
    cognition = normalize_probability(merged[["prob_HC", "prob_MCI", "prob_AD"]].to_numpy())
    extended = bounded_cognitive_extension(
        backbone, cognition, cognitive_weight=arguments.cognitive_weight
    )
    output = root / "artifacts" / "PREPARE_DrivenData" / "speechcare_cognitive_extension"
    output.mkdir(parents=True, exist_ok=True)
    result = {
        "status": "retrospective_protocol_aligned_extension",
        "confirmatory_claim_allowed": False,
        "reason": (
            "The released official-test probabilities are public benchmark outputs, "
            "and this test set has already been inspected. A new locked external cohort "
            "is required for a confirmatory superiority claim."
        ),
        "test_labels_used_for_fusion_or_weight_selection": False,
        "cognitive_weight": arguments.cognitive_weight,
        "speechcare_released_output_sha256": sha256(speechcare_path),
        "cognitive_output_sha256": sha256(cognitive_path),
        "cognitive_provenance": cognitive_meta,
        "speechcare_released_checkpoint": benchmark_metrics(y_true, backbone),
        "advoice_cognition_model": benchmark_metrics(y_true, cognition),
        "speechcare_plus_cognition": benchmark_metrics(y_true, extended),
        "speechcare_published_mean": PUBLISHED_SPEECHCARE,
    }
    pd.DataFrame(
        {
            "subject_id": merged["subject_id"],
            "true_label": merged["true_label"],
            "predicted_label": np.asarray(LABELS)[extended.argmax(axis=1)],
            **{f"prob_{label}": extended[:, index] for index, label in enumerate(LABELS)},
        }
    ).to_csv(output / "official_test_predictions.csv", index=False)
    (output / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
