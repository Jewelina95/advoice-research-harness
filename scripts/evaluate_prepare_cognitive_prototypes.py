from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from advoice.cognitive_prototypes import (
    LABELS,
    fit_cognitive_prototypes,
    predict_cognitive_prototypes,
)
from advoice.cognitive_representation_fusion import active_task_state_matrix
from advoice.config import load_all
from advoice.states import build_fold_calibrated_state_frame


SEED = 20260827


def _metrics(truth: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    truth_index = pd.Categorical(truth, categories=list(LABELS)).codes
    one_hot = np.eye(len(LABELS))[truth_index]
    prediction = np.asarray(LABELS)[probability.argmax(axis=1)]
    impaired = truth != "HC"
    staged = impaired
    return {
        "accuracy": float(accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro")),
        "micro_auroc_ovr": float(roc_auc_score(one_hot, probability, average="micro")),
        "macro_auroc_ovr": float(roc_auc_score(one_hot, probability, average="macro")),
        "micro_auprc": float(average_precision_score(one_hot, probability, average="micro")),
        "screening_auroc_hc_vs_impaired": float(
            roc_auc_score(impaired.astype(int), probability[:, 1:].sum(axis=1))
        ),
        "staging_auroc_mci_vs_ad": float(
            roc_auc_score(
                (truth[staged] == "AD").astype(int),
                probability[staged, 2] / probability[staged, 1:].sum(axis=1),
            )
        ),
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    artifact = root / "artifacts" / "PREPARE_DrivenData"
    output = artifact / "cognitive_prototype_audit"
    output.mkdir(parents=True, exist_ok=True)

    subjects = pd.read_csv(artifact / "subject_features.csv", dtype={"subject_id": str})
    development = subjects[subjects["split"].eq("train")].reset_index(drop=True)
    evidence = pd.read_csv(artifact / "metric_evidence.csv", dtype={"subject_id": str})
    manifest = pd.read_csv(artifact / "analysis_manifest.csv", dtype={"subject_id": str})
    task_by_subject = (
        manifest.groupby("subject_id", sort=False)["task_type"].first().astype(str).to_dict()
    )
    state_config = load_all("PREPARE_DrivenData")["states"]
    identifiers = development["subject_id"].astype(str).to_numpy()
    truth = development["label"].astype(str).to_numpy()
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    probability = np.zeros((len(development), len(LABELS)), dtype=float)
    detail_rows: list[dict[str, object]] = []
    expected_names: list[str] | None = None

    for fold, (fit_index, evaluation_index) in enumerate(
        splitter.split(identifiers, truth), start=1
    ):
        reference_ids = set(identifiers[fit_index][truth[fit_index] == "HC"])
        fold_states = build_fold_calibrated_state_frame(
            evidence,
            state_config,
            reference_ids,
            "HC",
        )
        fold_states = development[["subject_id"]].merge(
            fold_states, on="subject_id", validate="one_to_one"
        )
        values, reliability, names = active_task_state_matrix(
            fold_states, task_by_subject
        )
        if expected_names is None:
            expected_names = names
        elif expected_names != names:
            raise RuntimeError("Fold-local state features are not stable across folds")
        model = fit_cognitive_prototypes(
            values[fit_index],
            reliability[fit_index],
            truth[fit_index],
            names,
        )
        fold_probability, details = predict_cognitive_prototypes(
            model,
            values[evaluation_index],
            reliability[evaluation_index],
        )
        probability[evaluation_index] = fold_probability
        detail_rows.extend(
            {
                "subject_id": str(identifiers[index]),
                "fold": fold,
                **detail,
            }
            for index, detail in zip(evaluation_index, details, strict=True)
        )

    result = {
        "protocol": "five-fold development-only out-of-fold prototype audit",
        "official_test_labels_used": False,
        "selection_or_tuning_on_official_test": False,
        "seed": SEED,
        "development_cases": int(len(development)),
        "state_features": expected_names or [],
        "metrics": _metrics(truth, probability),
        "interpretation": (
            "This audit measures information in class-balanced cognitive state references. "
            "It is not evidence of an independent Agent gain and is not an official test result."
        ),
    }
    prediction = pd.DataFrame(
        {
            "subject_id": identifiers,
            "label": truth,
            "prob_HC": probability[:, 0],
            "prob_MCI": probability[:, 1],
            "prob_AD": probability[:, 2],
            "predicted_label": np.asarray(LABELS)[probability.argmax(axis=1)],
        }
    ).merge(pd.DataFrame(detail_rows), on="subject_id", validate="one_to_one")
    prediction.to_csv(output / "development_oof_predictions.csv", index=False)
    (output / "audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
