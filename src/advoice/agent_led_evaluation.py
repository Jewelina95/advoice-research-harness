"""Opt-in research evaluation of final Agent decisions, not clinical validation.

No providers, datasets, training routines, or implicit calibration are called.
Scalar temperature fitting is available only through its explicit API.
"""

from __future__ import annotations

from collections import Counter
from numbers import Real

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.metrics import roc_auc_score

__all__ = [
    "evaluate_agent_decisions",
    "fit_agent_temperature",
    "apply_agent_temperature",
]


def _strings(values, name: str, *, empty: bool = False) -> list[str]:
    if not isinstance(values, (list, tuple)):
        raise ValueError(f"{name} must be a sequence of unique nonempty strings")
    if (not values and not empty) or any(
        not isinstance(value, str) or not value.strip() for value in values
    ):
        raise ValueError(f"{name} must contain nonempty strings")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} must be unique")
    return list(values)


def _fingerprint(value) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("fingerprint must be a nonempty string")
    return value


def _number(value, name: str, upper: float) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be numeric, not boolean or text")
    number = float(value)
    if not np.isfinite(number) or not 0 <= number <= upper:
        raise ValueError(f"{name} must be finite and in [0, {upper}]")
    return number


def _vector(value, labels: list[str], name: str) -> list[float] | None:
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) != set(labels):
        raise ValueError(f"{name} must contain exactly the configured labels")
    vector = [_number(value[label], name, 4 if name == "scores" else 1) for label in labels]
    if name == "scores" and any(not item.is_integer() for item in vector):
        raise ValueError("scores must be ordinal integers in [0, 4]")
    if name == "probabilities" and not np.isclose(sum(vector), 1, rtol=0, atol=1e-8):
        raise ValueError("probabilities must be supplied normalized; no renormalization is done")
    return vector


def _ratio(numerator: int, denominator: int) -> float | None:
    return numerator / denominator if denominator else None


def _structural_count(row: dict, field: str) -> int | None:
    value = row.get(field)
    if field == "revisions" and value is None:
        value = row.get("state_revisions")
    if value is None:
        return None
    if field == "trace":
        if isinstance(value, dict):
            value = value.get("steps")
        if not isinstance(value, list):
            raise ValueError("trace must be a list or an object with a steps list")
        return len(value)
    if isinstance(value, list):
        return len(value)
    if type(value) is int and value >= 0:
        return value
    raise ValueError(f"{field} must be a list or a nonnegative integer count")


def evaluate_agent_decisions(
    results: list[dict], truth: dict[str, str], labels: list[str]
) -> dict:
    """Evaluate exactly one result per truth ID; malformed inputs raise ValueError.

    Each result requires case_id, status, predicted_label, scores, probabilities,
    and either a nonempty schema_fingerprint or the runtime's nonempty
    (schema_version, model_fingerprint) pair, shared by this cohort. Optional
    model fingerprints must also match; absent and present identities cannot be
    mixed. The runtime's state_revisions count is accepted as revisions.
    A decided result
    requires a known predicted_label; every other nonempty status is counted as
    a nondecision and requires a null prediction. Scores/probabilities may be
    null, but supplied vectors must be complete and valid. Predictions are never
    replaced by score argmax or residual priors. probability_status is optional
    descriptive metadata, not proof of calibration.

    Ranking and probability metrics are conditional on decided cases with the
    respective complete vector. Their case IDs and denominators are explicit.
    Macro AUROC is null unless every configured class has positives and negatives
    in that subset. Missing structural fields are unobserved, not zero counts.
    """
    labels = _strings(labels, "labels")
    if not isinstance(results, list) or not isinstance(truth, dict):
        raise ValueError("results must be a list and truth must be an ID-to-label dict")
    _strings(list(truth), "truth IDs", empty=True)
    if any(not isinstance(value, str) or value not in labels for value in truth.values()):
        raise ValueError("truth contains unknown true labels")
    required = {"case_id", "status", "predicted_label", "scores", "probabilities"}
    indexed = {}
    for row in results:
        if not isinstance(row, dict) or not required.issubset(row):
            raise ValueError("result is missing required fields")
        case_id = row["case_id"]
        _strings([case_id], "case_id")
        if case_id in indexed:
            raise ValueError("duplicate result case_id")
        indexed[case_id] = row
    if set(indexed) != set(truth):
        raise ValueError("result and truth cohort IDs must match exactly")

    rows = [indexed[case_id] for case_id in truth]
    identities = set()
    for row in rows:
        schema_fingerprint = (
            _fingerprint(row["schema_fingerprint"]) if "schema_fingerprint" in row else None
        )
        schema_version = _fingerprint(row["schema_version"]) if "schema_version" in row else None
        model_fingerprint = (
            _fingerprint(row["model_fingerprint"]) if "model_fingerprint" in row else None
        )
        if schema_fingerprint is None and (schema_version is None or model_fingerprint is None):
            raise ValueError("a schema fingerprint or runtime schema/model identity is required")
        identities.add((schema_fingerprint, schema_version, model_fingerprint))
    if len(identities) > 1:
        raise ValueError("mixed schema/model fingerprints in evaluation cohort")
    schema_fingerprint, schema_version, model_fingerprint = next(iter(identities), (None, None, None))
    status_counts = Counter()
    probability_status_counts = Counter()
    score_rows, probability_rows = [], []
    structural = {"trace_length": [], "revisions": [], "clinical_claims": []}
    predictions, decided = [], []
    for row in rows:
        status = row["status"]
        if not isinstance(status, str) or not status.strip():
            raise ValueError("status must be a nonempty string")
        prediction = row["predicted_label"]
        is_decided = status == "decided"
        if is_decided:
            if not isinstance(prediction, str) or prediction not in labels:
                raise ValueError("decided output requires a known predicted_label")
        elif prediction is not None:
            raise ValueError("nondecided output must have a null predicted_label")
        scores = _vector(row["scores"], labels, "scores")
        probabilities = _vector(row["probabilities"], labels, "probabilities")
        probability_status = row.get("probability_status")
        if probability_status is not None:
            if not isinstance(probability_status, str) or not probability_status.strip():
                raise ValueError("probability_status must be null or a nonempty string")
            probability_status_counts[probability_status] += 1
        status_counts[status] += 1
        predictions.append(prediction)
        decided.append(is_decided)
        if is_decided and scores is not None:
            score_rows.append((row["case_id"], scores))
        if is_decided and probabilities is not None:
            probability_rows.append((row["case_id"], probabilities))
        for source, output in (("trace", "trace_length"), ("revisions", "revisions"),
                               ("clinical_claims", "clinical_claims")):
            count = _structural_count(row, source)
            if count is not None:
                structural[output].append(count)

    actual = list(truth.values())
    n, n_decided = len(rows), sum(decided)
    correct = sum(a == p for a, p in zip(actual, predictions))
    per_class = {}
    confusion = [[0] * (len(labels) + 1) for _ in labels]
    for a, p in zip(actual, predictions):
        confusion[labels.index(a)][labels.index(p) if p is not None else len(labels)] += 1
    for label in labels:
        support = actual.count(label)
        predicted = predictions.count(label)
        tp = sum(a == p == label for a, p in zip(actual, predictions))
        fp, fn = predicted - tp, support - tp
        tn = n - tp - fp - fn
        abstained = sum(a == label and p is None for a, p in zip(actual, predictions))
        per_class[label] = {
            "support": support, "predicted": predicted, "decided": support - abstained,
            "abstained": abstained, "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": _ratio(tp, predicted), "recall": _ratio(tp, support),
            "specificity": _ratio(tn, n - support),
            "f1": _ratio(2 * tp, 2 * tp + fp + fn),
            "coverage": _ratio(support - abstained, support),
            "conditional_accuracy": _ratio(tp, support - abstained),
            "denominators": {
                "precision": predicted, "recall": support, "specificity": n - support,
                "f1": 2 * tp + fp + fn, "coverage": support,
                "conditional_accuracy": support - abstained,
            },
        }

    ranking_ids = [case_id for case_id, _ in score_rows]
    per_class_auc = {}
    for index, label in enumerate(labels):
        binary = [truth[case_id] == label for case_id in ranking_ids]
        per_class_auc[label] = (
            float(roc_auc_score(binary, [vector[index] for _, vector in score_rows]))
            if binary and 0 < sum(binary) < len(binary) else None
        )
    all_auc_defined = all(value is not None for value in per_class_auc.values())
    probability_metrics = {
        "source": "supplied_probabilities", "scope": "decided_with_complete_probabilities",
        "denominator": len(probability_rows),
        "case_ids": [case_id for case_id, _ in probability_rows],
        "log_loss": None, "log_loss_status": "unavailable",
        "multiclass_brier": None,
        "calibration_claim": "normalization_validated_not_calibration_proven",
    }
    if probability_rows:
        p = np.asarray([vector for _, vector in probability_rows])
        y = np.asarray([labels.index(truth[case_id]) for case_id, _ in probability_rows])
        true_p = p[np.arange(len(y)), y]
        if np.any(true_p == 0):
            probability_metrics["log_loss_status"] = "infinite_zero_true_probability"
        else:
            probability_metrics["log_loss"] = float(-np.log(true_p).mean())
            probability_metrics["log_loss_status"] = "finite"
        probability_metrics["multiclass_brier"] = float(
            np.square(p - np.eye(len(labels))[y]).sum(axis=1).mean()
        )

    return {
        "research_only": True, "clinical_performance_proven": False,
        "schema_fingerprint": schema_fingerprint, "schema_version": schema_version,
        "model_fingerprint": model_fingerprint, "labels": labels,
        "n_cases": n, "n_decided": n_decided, "n_nondecided": n - n_decided,
        "n_correct": correct, "accuracy": _ratio(correct, n),
        "coverage": _ratio(n_decided, n), "conditional_accuracy": _ratio(correct, n_decided),
        "denominators": {"accuracy": n, "coverage": n, "conditional_accuracy": n_decided},
        "status_counts": dict(status_counts), "probability_status_counts": dict(probability_status_counts),
        "per_class": per_class,
        "confusion_matrix": {"row_labels": labels, "column_labels": labels + [None],
                             "counts": confusion, "null_column": "nondecision"},
        "ranking_metrics": {
            "source": "ranking_scores", "scope": "decided_with_complete_scores",
            "denominator": len(score_rows), "case_ids": ranking_ids,
            "per_class_auroc": per_class_auc,
            "macro_auroc_ovr": float(np.mean(list(per_class_auc.values()))) if all_auc_defined else None,
            "macro_class_denominator": len(labels) if all_auc_defined else 0,
            "undefined_classes": [label for label, value in per_class_auc.items() if value is None],
        },
        "probability_metrics": probability_metrics,
        "layer_b": {
            "interpretation": "structural_only_not_clinical_efficacy",
            **{name: {"total": sum(values), "denominator": len(values),
                      "mean": _ratio(sum(values), len(values))}
               for name, values in structural.items()},
        },
    }


def _score_matrix(scores: np.ndarray, labels: list[str], case_ids: list[str]) -> np.ndarray:
    array = np.asarray(scores)
    if array.dtype.kind not in "iuf" or array.shape != (len(case_ids), len(labels)):
        raise ValueError("scores must be a numeric (case_ids, labels) matrix")
    array = array.astype(float)
    if (not np.isfinite(array).all() or (array < 0).any() or (array > 4).any()
            or not np.equal(array, np.floor(array)).all()):
        raise ValueError("scores must be finite ordinal integers in [0, 4]")
    return array


def fit_agent_temperature(
    scores: np.ndarray,
    y: list[str],
    case_ids: list[str],
    model_fingerprint: str,
    partition: str = "development",
    *,
    labels: list[str],
    training_case_ids: list[str] | tuple[str, ...] = (),
) -> dict:
    """Explicitly fit one positive temperature on caller-declared independent data.

    labels is the required column order (never inferred from y). The caller must
    ensure calibration subjects were not used for model/prompt selection or
    training and that IDs consistently identify independent subjects. Provided
    training_case_ids are checked for overlap; independence cannot be inferred
    from arrays or an empty exclusion list. Only development/calibration
    partitions are accepted. Every class must occur in y. No data is loaded.

    Log loss is minimized over log(T) in [-6, 6]; bounds and baseline are also
    evaluated. This bounded research fit is not evidence of clinical calibration.
    """
    labels = _strings(labels, "labels")
    if len(labels) < 2:
        raise ValueError("temperature calibration requires at least two labels")
    case_ids = _strings(case_ids, "fit case IDs")
    training_ids = _strings(training_case_ids, "training case IDs", empty=True)
    _fingerprint(model_fingerprint)
    if partition not in ("development", "calibration"):
        raise ValueError("partition must be development or calibration, never training/test")
    if set(case_ids) & set(training_ids):
        raise ValueError("fit case IDs overlap training case IDs")
    matrix = _score_matrix(scores, labels, case_ids)
    if (len(y) != len(case_ids)
            or any(not isinstance(value, str) or value not in labels for value in y)
            or set(y) != set(labels)):
        raise ValueError("y must match fit case IDs and include every configured class")
    encoded = np.asarray([labels.index(value) for value in y])

    def objective(log_temperature: float) -> float:
        logits = matrix / np.exp(log_temperature)
        return float((logsumexp(logits, axis=1) - logits[np.arange(len(encoded)), encoded]).mean())

    fit = minimize_scalar(objective, bounds=(-6.0, 6.0), method="bounded")
    if not fit.success or not np.isfinite(fit.fun):
        raise ValueError("temperature optimization failed")
    best = min((0.0, -6.0, 6.0, float(fit.x)), key=objective)
    return {
        "artifact_version": 1, "method": "scalar_temperature_on_ordinal_scores",
        "temperature": float(np.exp(best)), "fit_case_ids": case_ids,
        "class_order": labels, "model_fingerprint": model_fingerprint,
        "partition": partition, "training_case_ids": training_ids,
        "independence": "caller_contract_not_inferred", "research_only": True,
        "log_temperature_bounds": [-6.0, 6.0], "fit_log_loss": objective(best),
        "uncalibrated_log_loss": objective(0.0),
    }


def apply_agent_temperature(
    scores: np.ndarray,
    artifact: dict,
    case_ids: list[str],
    model_fingerprint: str,
    *,
    labels: list[str],
) -> np.ndarray:
    """Apply a frozen scalar T, rejecting calibration overlap and changed contracts.

    Returns supplied-class-order probability columns; score argmax (including
    ties) is preserved. This does not replace the Agent's final predicted_label.
    """
    labels = _strings(labels, "labels")
    case_ids = _strings(case_ids, "application case IDs", empty=True)
    _fingerprint(model_fingerprint)
    if not isinstance(artifact, dict):
        raise ValueError("temperature artifact must be a dict")
    if (artifact.get("artifact_version") != 1
            or artifact.get("method") != "scalar_temperature_on_ordinal_scores"
            or artifact.get("partition") not in ("development", "calibration")):
        raise ValueError("invalid temperature artifact or partition")
    if artifact.get("model_fingerprint") != model_fingerprint:
        raise ValueError("model fingerprint mismatch")
    if artifact.get("class_order") != labels:
        raise ValueError("class order mismatch")
    fit_ids = _strings(artifact.get("fit_case_ids"), "artifact fit case IDs")
    if set(case_ids) & set(fit_ids):
        raise ValueError("application case IDs overlap calibration fit case IDs")
    temperature = _number(artifact.get("temperature"), "temperature", float(np.exp(6)))
    if temperature < np.exp(-6):
        raise ValueError("temperature must be positive and within the fit bounds")
    matrix = _score_matrix(scores, labels, case_ids)
    return softmax(matrix / temperature, axis=1)
