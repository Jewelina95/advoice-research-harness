from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, f1_score, log_loss, roc_auc_score


LABELS = ("HC", "MCI", "AD")


def normalize_probability(probability: np.ndarray) -> np.ndarray:
    values = np.asarray(probability, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != len(LABELS):
        raise ValueError("Probability must have shape (n_subjects, 3).")
    if not np.isfinite(values).all() or (values < 0.0).any():
        raise ValueError("Probability contains invalid values.")
    denominator = values.sum(axis=1, keepdims=True)
    if (denominator <= 0.0).any():
        raise ValueError("Each probability row must have positive mass.")
    return values / denominator


def bounded_cognitive_extension(
    backbone_probability: np.ndarray,
    cognitive_probability: np.ndarray,
    cognitive_weight: float = 0.20,
) -> np.ndarray:
    """Add a bounded cognition residual without allowing it to replace the backbone."""
    if not 0.0 <= cognitive_weight <= 0.20:
        raise ValueError("cognitive_weight must stay within the prespecified [0, 0.20] bound.")
    backbone = normalize_probability(backbone_probability)
    cognitive = normalize_probability(cognitive_probability)
    if backbone.shape != cognitive.shape:
        raise ValueError("Backbone and cognition probabilities must align by subject.")
    return normalize_probability(
        (1.0 - cognitive_weight) * backbone + cognitive_weight * cognitive
    )


def benchmark_metrics(y_true: np.ndarray, probability: np.ndarray) -> dict[str, float]:
    probability = normalize_probability(probability)
    y_true = np.asarray(y_true, dtype=int)
    predicted = probability.argmax(axis=1)
    one_hot = np.eye(len(LABELS), dtype=int)[y_true]
    return {
        "accuracy": float(accuracy_score(y_true, predicted)),
        "micro_f1": float(f1_score(y_true, predicted, average="micro")),
        "macro_f1": float(f1_score(y_true, predicted, average="macro")),
        "weighted_f1": float(f1_score(y_true, predicted, average="weighted")),
        "micro_auroc_ovr": float(
            roc_auc_score(one_hot, probability, average="micro", multi_class="ovr")
        ),
        "weighted_auroc_ovr": float(
            roc_auc_score(y_true, probability, average="weighted", multi_class="ovr")
        ),
        "micro_auprc": float(average_precision_score(one_hot, probability, average="micro")),
        "weighted_auprc": float(
            average_precision_score(one_hot, probability, average="weighted")
        ),
        "log_loss": float(log_loss(y_true, probability, labels=np.arange(len(LABELS)))),
    }
