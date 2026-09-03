from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .agent_runtime import (
    case_pseudonym,
    normalize_probabilities,
    run_structured_batch,
    select_agent_cohort,
)
from .diagnostic_agent import (
    fuse_evidence_likelihood,
    fuse_two_stage_evidence,
    route_case,
)
from .utils import hash_values, json_dump, json_load, now_utc


SKILL_FILES = [
    "SKILL.md",
    "MEDICAL_SCOPE.md",
    "TASK_OBSERVABILITY.md",
    "STATE_KNOWLEDGE.md",
    "CONFOUND_AND_DIFFERENTIAL.md",
    "EVIDENCE_HIERARCHY.md",
    "COGNITIVE_ROLLBACK_PROTOCOL.md",
    "REPORT_CONTRACT.md",
    "REPORT_PERMISSION_POLICY.md",
    "LABEL_LEAKAGE_POLICY.md",
    "EVIDENCE_REGISTRY.csv",
    "TOOLS.json",
    "REFERENCES.md",
]


_BLINDED_WORKSPACE_KEYS = {
    "base_probabilities",
    "corrected_probabilities",
    "final_probabilities",
    "final_prediction",
    "class_support",
    "correction_alpha",
}


def blind_workspace(workspace: dict[str, Any]) -> dict[str, Any]:
    """Remove model predictions before the Agent forms its evidence hypothesis."""

    return {
        key: value
        for key, value in workspace.items()
        if key not in _BLINDED_WORKSPACE_KEYS
    }


def _batch_schema(path: Path, labels: list[str]) -> None:
    hierarchical = labels == ["HC", "MCI", "AD"]
    evidence_scores = {
        "type": "object",
        "additionalProperties": False,
        "required": labels,
        "properties": {
            label: {"type": "integer", "minimum": 0, "maximum": 4}
            for label in labels
        },
    }
    state_update = {
        "type": "object",
        "additionalProperties": False,
        "required": ["state_id", "task_scope", "action", "reason", "evidence_ids"],
        "properties": {
            "state_id": {"type": "string"},
            "task_scope": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["keep", "downweight", "invalidate", "mark_unavailable"],
            },
            "reason": {"type": "string"},
            "evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
        },
    }
    case = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "case_id",
            "action",
            "evidence_class",
            "evidence_scores",
            "used_evidence_ids",
            "counterevidence_ids",
            "quality_evidence_ids",
            "state_updates",
            "counterevidence_checked",
            "screening_rationale",
            "uncertainty_reason",
        ],
        "properties": {
            "case_id": {"type": "string"},
            "action": {
                "type": "string",
                "enum": ["classify", "abstain", "retest"],
            },
            "evidence_class": {"type": "string", "enum": labels},
            "evidence_scores": evidence_scores,
            "used_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "counterevidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "quality_evidence_ids": {
                "type": "array",
                "items": {"type": "string"},
                "uniqueItems": True,
            },
            "state_updates": {"type": "array", "items": state_update},
            "counterevidence_checked": {"type": "boolean"},
            "screening_rationale": {"type": "string"},
            "uncertainty_reason": {"type": "string"},
        },
    }
    if hierarchical:
        for field in ["evidence_class", "evidence_scores"]:
            case["required"].remove(field)
            case["properties"].pop(field)
        case["required"].extend(
            ["screening_class", "screening_scores", "staging_action", "staging_class", "staging_scores"]
        )
        case["properties"].update(
            {
                "screening_class": {"type": "string", "enum": ["HC", "impaired"]},
                "screening_scores": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["HC", "impaired"],
                    "properties": {
                        "HC": {"type": "integer", "minimum": 0, "maximum": 4},
                        "impaired": {"type": "integer", "minimum": 0, "maximum": 4},
                    },
                },
                "staging_action": {
                    "type": "string",
                    "enum": ["classify", "insufficient"],
                },
                "staging_class": {
                    "type": "string",
                    "enum": ["MCI", "AD", "undetermined"],
                },
                "staging_scores": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["MCI", "AD"],
                    "properties": {
                        "MCI": {"type": "integer", "minimum": 0, "maximum": 4},
                        "AD": {"type": "integer", "minimum": 0, "maximum": 4},
                    },
                },
            }
        )
    json_dump(
        {
            "type": "object",
            "additionalProperties": False,
            "required": ["cases"],
            "properties": {"cases": {"type": "array", "items": case}},
        },
        path,
    )


def _read_workspaces(path: Path) -> dict[str, dict[str, Any]]:
    return {
        str(item["case_id"]): item
        for item in (
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }


def _skill_text(root: Path) -> str:
    skill_root = root / "skills" / "ad_evidence_diagnostic"
    return "\n\n".join(
        (skill_root / name).read_text(encoding="utf-8") for name in SKILL_FILES
    )


def _cached_response(
    output_path: Path,
    metadata_path: Path,
    fingerprint: str,
    case_ids: list[str],
) -> dict[str, Any] | None:
    response = json_load(output_path, {})
    metadata = json_load(metadata_path, {})
    returned = [str(item.get("case_id", "")) for item in response.get("cases", [])]
    returned_set = set(returned)
    requested_set = set(case_ids)
    if (
        metadata.get("fingerprint") == fingerprint
        and len(returned) == len(returned_set)
        and returned_set.issubset(requested_set)
    ):
        return response
    return None


def _index_expected_candidates(
    candidates: list[dict[str, Any]], expected_case_ids: set[str]
) -> tuple[dict[str, dict[str, Any]], list[str]]:
    indexed: dict[str, dict[str, Any]] = {}
    unexpected: list[str] = []
    for candidate in candidates:
        case_id = str(candidate.get("case_id", ""))
        if case_id not in expected_case_ids:
            unexpected.append(case_id)
            continue
        indexed[case_id] = candidate
    return indexed, unexpected


def _run_batch(
    *,
    root: Path,
    provider: str,
    model: str,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    metadata_path: Path,
    case_ids: list[str],
) -> dict[str, Any]:
    fingerprint = hash_values([provider, model, prompt, schema_path, case_ids])
    cached = _cached_response(output_path, metadata_path, fingerprint, case_ids)
    if cached is not None:
        return cached
    response = run_structured_batch(
        root, prompt, schema_path, output_path, model, provider
    )
    json_dump(
        {
            "fingerprint": fingerprint,
            "case_ids": case_ids,
            "provider": provider,
            "model": model,
        },
        metadata_path,
    )
    return response


def _allowed_ids(workspace: dict[str, Any]) -> tuple[set[str], set[str], set[str]]:
    clinical: set[str] = set()
    counter: set[str] = set()
    quality: set[str] = set()
    for key in [
        "state_observations",
        "selected_supporting_evidence",
        "inference_only_metric_observations",
    ]:
        clinical.update(str(item["evidence_id"]) for item in workspace.get(key, []))
    counter.update(
        str(item["evidence_id"])
        for item in workspace.get("selected_counterevidence", [])
    )
    counter.update(
        str(item["evidence_id"])
        for item in workspace.get("inference_only_metric_observations", [])
        if float(item.get("directional_z", 0.0)) < 0.0
    )
    clinical.update(counter)
    for state in workspace.get("state_observations", []):
        clinical.update(str(value) for value in state.get("metric_evidence_ids", []))
        clinical.update(
            str(segment["segment_id"])
            for segment in state.get("evidence_segments", [])
            if segment.get("segment_id")
        )
    quality.update(
        str(item["evidence_id"]) for item in workspace.get("quality_observations", [])
    )
    registry = {
        str(item.get("evidence_id", "")): str(item.get("evidence_type", ""))
        for item in workspace.get("evidence_registry", [])
        if item.get("evidence_id")
    }
    if registry:
        clinical &= {
            evidence_id
            for evidence_id, evidence_type in registry.items()
            if evidence_type in {"state", "metric", "segment"}
        }
        counter &= clinical
        quality &= {
            evidence_id
            for evidence_id, evidence_type in registry.items()
            if evidence_type in {"quality", "qc"}
        }
    return clinical, counter, quality


def fit_agent_correction_strength(
    labels_true: np.ndarray,
    labels: list[str],
    base_probability: np.ndarray,
    evidence_likelihood: np.ndarray,
    gate: np.ndarray,
    route_multiplier: np.ndarray,
    strength_grid: list[float],
    minimum_macro_f1_gain: float = 0.0,
    auroc_noninferiority_margin: float = 0.001,
) -> dict[str, Any]:
    """Choose one correction strength on development labels only.

    A non-zero Agent correction must improve macro F1 without materially
    degrading macro AUROC. Otherwise the prediction path fails closed to the
    supervised prior. This keeps the official test set out of model selection.
    """

    from sklearn.metrics import f1_score, roc_auc_score

    rows: list[dict[str, float]] = []
    for strength in strength_grid:
        probability = fuse_evidence_likelihood(
            base_probability,
            evidence_likelihood,
            gate,
            float(strength),
            route_multiplier,
            hierarchical_reference_index=0 if len(labels) > 2 else None,
        )
        predicted = np.asarray(labels)[probability.argmax(axis=1)]
        macro_f1 = float(f1_score(labels_true, predicted, labels=labels, average="macro"))
        one_hot = np.column_stack(
            [(np.asarray(labels_true, dtype=str) == label).astype(int) for label in labels]
        )
        try:
            macro_auroc = float(
                roc_auc_score(one_hot, probability, average="macro")
            )
        except ValueError:
            macro_auroc = float("nan")
        row = {"strength": float(strength), "macro_f1": macro_f1, "macro_auroc": macro_auroc}
        rows.append(row)
    selected = _select_correction_candidate(
        rows,
        minimum_macro_f1_gain=minimum_macro_f1_gain,
        auroc_noninferiority_margin=auroc_noninferiority_margin,
    )
    return {
        "selected_strength": selected["strength"],
        "selection_status": selected["selection_status"],
        "baseline_macro_f1": selected["baseline_macro_f1"],
        "baseline_macro_auroc": selected["baseline_macro_auroc"],
        "minimum_macro_f1_gain": float(minimum_macro_f1_gain),
        "auroc_noninferiority_margin": float(auroc_noninferiority_margin),
        "candidates": rows,
    }


def fit_agent_two_stage_strengths(
    labels_true: np.ndarray,
    base_probability: np.ndarray,
    screening_likelihood: np.ndarray,
    staging_likelihood: np.ndarray,
    screening_gate: np.ndarray,
    staging_gate: np.ndarray,
    screening_route_multiplier: np.ndarray,
    staging_route_multiplier: np.ndarray,
    strength_grid: list[float],
    minimum_macro_f1_gain: float = 0.0,
    auroc_noninferiority_margin: float = 0.001,
) -> dict[str, Any]:
    """Select independent screening/staging strengths on development labels only."""

    from sklearn.metrics import f1_score, roc_auc_score

    labels = np.asarray(["HC", "MCI", "AD"])
    truth = np.asarray(labels_true, dtype=str)
    one_hot = np.column_stack([(truth == label).astype(int) for label in labels])
    rows: list[dict[str, float]] = []
    for screening_strength in strength_grid:
        for staging_strength in strength_grid:
            probability = fuse_two_stage_evidence(
                base_probability,
                screening_likelihood,
                staging_likelihood,
                screening_gate,
                staging_gate,
                float(screening_strength),
                float(staging_strength),
                screening_route_multiplier,
                staging_route_multiplier,
            )
            predicted = labels[probability.argmax(axis=1)]
            rows.append(
                {
                    "screening_strength": float(screening_strength),
                    "staging_strength": float(staging_strength),
                    "macro_f1": float(
                        f1_score(truth, predicted, labels=labels, average="macro")
                    ),
                    "macro_auroc": float(
                        roc_auc_score(one_hot, probability, average="macro")
                    ),
                }
            )
    baseline = next(
        row
        for row in rows
        if row["screening_strength"] == 0.0 and row["staging_strength"] == 0.0
    )
    eligible = [
        row
        for row in rows
        if row is not baseline
        and row["macro_f1"] >= baseline["macro_f1"] + minimum_macro_f1_gain
        and row["macro_auroc"]
        >= baseline["macro_auroc"] - auroc_noninferiority_margin
    ]
    if eligible:
        selected = max(
            eligible,
            key=lambda row: (
                row["macro_f1"],
                row["macro_auroc"],
                -(row["screening_strength"] + row["staging_strength"]),
            ),
        )
        status = "validated_joint_gain"
    else:
        selected = baseline
        status = "failed_closed_no_joint_gain"
    return {
        "selected_screening_strength": selected["screening_strength"],
        "selected_staging_strength": selected["staging_strength"],
        "selection_status": status,
        "baseline_macro_f1": baseline["macro_f1"],
        "baseline_macro_auroc": baseline["macro_auroc"],
        "minimum_macro_f1_gain": float(minimum_macro_f1_gain),
        "auroc_noninferiority_margin": float(auroc_noninferiority_margin),
        "candidates": rows,
    }


def fit_binary_evidence_calibrator(
    labels_true: np.ndarray,
    evidence_likelihood: np.ndarray,
    usable: np.ndarray,
    *,
    random_state: int = 20260903,
    maximum_folds: int = 5,
) -> dict[str, Any]:
    """Calibrate a two-class Agent score margin without using evaluation labels.

    Balanced logistic calibration converts the Agent's ordinal score difference
    into an equal-prior evidence likelihood. Out-of-fold likelihoods are used
    to select the downstream correction strength; the full development fit is
    retained only for applying the frozen transform to evaluation cases.
    """

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold

    truth = np.asarray(labels_true, dtype=int)
    likelihood = np.asarray(evidence_likelihood, dtype=float)
    mask = np.asarray(usable, dtype=bool)
    if likelihood.ndim != 2 or likelihood.shape[1] != 2:
        raise ValueError("Binary evidence likelihood must have shape (n, 2).")
    if len(truth) != len(likelihood) or len(mask) != len(likelihood):
        raise ValueError("Calibration arrays must have the same length.")
    clipped = np.clip(likelihood, 1e-8, 1.0)
    margin = np.log(clipped[:, 1]) - np.log(clipped[:, 0])
    oof = np.full((len(likelihood), 2), 0.5, dtype=float)
    usable_truth = truth[mask]
    class_counts = np.bincount(usable_truth, minlength=2)
    fold_count = min(int(maximum_folds), int(class_counts.min()))
    if mask.sum() < 10 or fold_count < 2:
        return {
            "status": "failed_closed_insufficient_cases",
            "coefficient": 0.0,
            "intercept": 0.0,
            "available_cases": int(mask.sum()),
            "class_counts": class_counts.astype(int).tolist(),
            "oof_likelihood": oof,
        }

    usable_indices = np.flatnonzero(mask)
    splitter = StratifiedKFold(
        n_splits=fold_count,
        shuffle=True,
        random_state=int(random_state),
    )
    for train_local, validation_local in splitter.split(
        margin[mask].reshape(-1, 1), usable_truth
    ):
        model = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=int(random_state),
        )
        model.fit(
            margin[usable_indices[train_local]].reshape(-1, 1),
            truth[usable_indices[train_local]],
        )
        positive = model.predict_proba(
            margin[usable_indices[validation_local]].reshape(-1, 1)
        )[:, int(np.flatnonzero(model.classes_ == 1)[0])]
        oof[usable_indices[validation_local], 1] = positive
        oof[usable_indices[validation_local], 0] = 1.0 - positive

    final_model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        solver="lbfgs",
        max_iter=1000,
        random_state=int(random_state),
    )
    final_model.fit(margin[mask].reshape(-1, 1), usable_truth)
    return {
        "status": "calibrated_on_development_oof",
        "coefficient": float(final_model.coef_[0, 0]),
        "intercept": float(final_model.intercept_[0]),
        "available_cases": int(mask.sum()),
        "class_counts": class_counts.astype(int).tolist(),
        "folds": int(fold_count),
        "oof_likelihood": oof,
    }


def apply_binary_evidence_calibrator(
    evidence_likelihood: np.ndarray,
    calibrator: dict[str, Any] | None,
) -> np.ndarray:
    """Apply a frozen ordinal-score calibrator; unavailable fits stay neutral."""

    likelihood = np.asarray(evidence_likelihood, dtype=float)
    if not calibrator or calibrator.get("status") != "calibrated_on_development_oof":
        return np.full_like(likelihood, 0.5, dtype=float)
    clipped = np.clip(likelihood, 1e-8, 1.0)
    margin = np.log(clipped[:, 1]) - np.log(clipped[:, 0])
    logit = (
        float(calibrator["intercept"])
        + float(calibrator["coefficient"]) * margin
    )
    positive = 1.0 / (1.0 + np.exp(-np.clip(logit, -30.0, 30.0)))
    return np.column_stack([1.0 - positive, positive])


def two_stage_route_parameters(
    base_probability: np.ndarray,
    screening_likelihood: np.ndarray,
    staging_likelihood: np.ndarray,
    gate: float,
    staging_available: bool,
) -> dict[str, Any]:
    """Use the same screening/staging routing in calibration and inference."""

    base = np.asarray(base_probability, dtype=float)
    screening = np.asarray(screening_likelihood, dtype=float)
    staging = np.asarray(staging_likelihood, dtype=float)
    base_screening = np.asarray([base[0], base[1:].sum()], dtype=float)
    base_staging = base[1:] / max(base[1:].sum(), 1e-12)
    screening_route = route_case(base_screening, screening, float(gate))
    staging_gate = float(gate) * float(staging_available)
    staging_route = route_case(base_staging, staging, staging_gate)
    return {
        "screening_route": screening_route,
        "staging_route": staging_route,
        "screening_multiplier": float(screening_route["route_multiplier"]),
        "staging_multiplier": float(staging_route["route_multiplier"]),
        "staging_gate": float(staging_gate),
    }


def _select_correction_candidate(
    rows: list[dict[str, float]],
    *,
    minimum_macro_f1_gain: float,
    auroc_noninferiority_margin: float,
) -> dict[str, Any]:
    """Apply the preregistered joint validation gate to strength candidates."""

    if not rows:
        raise ValueError("At least one correction-strength candidate is required.")
    baseline = next((row for row in rows if float(row["strength"]) == 0.0), None)
    if baseline is None:
        raise ValueError("The correction-strength grid must include zero.")
    baseline_f1 = float(baseline["macro_f1"])
    baseline_auroc = float(baseline["macro_auroc"])
    eligible = [
        row
        for row in rows
        if float(row["strength"]) > 0.0
        and np.isfinite(float(row["macro_f1"]))
        and np.isfinite(float(row["macro_auroc"]))
        and float(row["macro_f1"]) >= baseline_f1 + float(minimum_macro_f1_gain)
        and float(row["macro_auroc"])
        >= baseline_auroc - float(auroc_noninferiority_margin)
    ]
    if not eligible:
        return {
            "strength": 0.0,
            "selection_status": "failed_closed_no_joint_gain",
            "baseline_macro_f1": baseline_f1,
            "baseline_macro_auroc": baseline_auroc,
        }
    best = max(
        eligible,
        key=lambda row: (
            float(row["macro_f1"]),
            float(row["macro_auroc"]),
            -float(row["strength"]),
        ),
    )
    return {
        "strength": float(best["strength"]),
        "selection_status": "validated_joint_gain",
        "baseline_macro_f1": baseline_f1,
        "baseline_macro_auroc": baseline_auroc,
    }


def validate_candidate(
    candidate: dict[str, Any],
    workspace: dict[str, Any],
    labels: list[str],
) -> dict[str, Any]:
    """Deterministic CooT-style evidence audit. No model text can bypass it."""

    violations: list[str] = []
    clinical, available_counter, quality = _allowed_ids(workspace)
    used = {str(value) for value in candidate.get("used_evidence_ids", [])}
    cited_counter = {
        str(value) for value in candidate.get("counterevidence_ids", [])
    }
    cited_quality = {str(value) for value in candidate.get("quality_evidence_ids", [])}
    if used - clinical or cited_counter - clinical:
        violations.append("V04_INVALID_REFERENCE")
    if cited_quality - quality:
        violations.append("V04_INVALID_QUALITY_REFERENCE")
    if (used | cited_counter) & quality:
        violations.append("V02_QC_AS_DISEASE")
    if not bool(candidate.get("counterevidence_checked")):
        violations.append("V06_COUNTEREVIDENCE_OMITTED")
    if available_counter and not (cited_counter & available_counter):
        violations.append("V06_MATERIAL_COUNTEREVIDENCE_NOT_CITED")
    observable_state_lookup = {
        (str(item["state_id"]), str(item.get("task_scope", "overall"))): item
        for item in workspace.get("state_observations", [])
    }
    unavailable_state_lookup = {
        (str(item["state_id"]), str(item.get("task_scope", "overall"))): item
        for item in workspace.get("model_only_state_observations", [])
    }

    def resolve_state_key(
        state_id: str, task_scope: str
    ) -> tuple[str, str] | None:
        requested = (state_id, task_scope)
        if requested in observable_state_lookup or requested in unavailable_state_lookup:
            return requested
        if task_scope not in {"", "overall", "nan"}:
            task_view = (f"{state_id}__task_{task_scope}", task_scope)
            if task_view in observable_state_lookup or task_view in unavailable_state_lookup:
                return task_view
        return None

    for update in candidate.get("state_updates", []):
        requested_state_id = str(update.get("state_id", ""))
        requested_task_scope = str(update.get("task_scope", "overall"))
        key = resolve_state_key(requested_state_id, requested_task_scope)
        action = str(update.get("action", ""))
        references = set(map(str, update.get("evidence_ids", [])))
        is_observable = key is not None and key in observable_state_lookup
        is_unavailable_mark = (
            key is not None
            and key in unavailable_state_lookup
            and action == "mark_unavailable"
        )
        if not is_observable and not is_unavailable_mark:
            violations.append("V03_UNOBSERVABLE_STATE")
        allowed_state_references = set(clinical)
        if action in {"downweight", "invalidate", "mark_unavailable"}:
            allowed_state_references.update(quality)
        if is_unavailable_mark:
            assert key is not None
            state_item = unavailable_state_lookup[key]
            allowed_state_references.add(str(state_item["evidence_id"]))
            # Raw aliases keep migration records readable. New decisions still
            # use typed evidence IDs from the registry.
            allowed_state_references.add(key[0])
            allowed_state_references.add(f"state:{key[0]}")
        if references - allowed_state_references:
            violations.append("V04_INVALID_STATE_REFERENCE")
        if action == "keep" and references & quality:
            violations.append("V02_QC_AS_DISEASE")
    # Historical cache fields remain readable, but 9.2 prompts only emit the
    # evidence_* contract. This avoids invalidating prior audit fixtures while
    # removing supervised probability anchoring from every new Agent call.
    screening_scores = candidate.get("screening_scores")
    staging_scores = candidate.get("staging_scores")
    hierarchical_scores = (
        labels == ["HC", "MCI", "AD"]
        and isinstance(screening_scores, dict)
        and isinstance(staging_scores, dict)
        and all(label in screening_scores for label in ["HC", "impaired"])
        and all(label in staging_scores for label in ["MCI", "AD"])
    )
    if hierarchical_scores:
        screen_vector = np.asarray(
            [float(screening_scores["HC"]), float(screening_scores["impaired"])]
        )
        screen_vector -= screen_vector.max()
        screen_likelihood = np.exp(screen_vector)
        screen_likelihood /= screen_likelihood.sum()
        stage_vector = np.asarray(
            [float(staging_scores["MCI"]), float(staging_scores["AD"])]
        )
        staging_available = candidate.get("staging_action") == "classify"
        if not staging_available:
            if not np.isclose(stage_vector[0], stage_vector[1]):
                violations.append("V09_STAGING_INSUFFICIENT_NONNEUTRAL")
            if candidate.get("staging_class") != "undetermined":
                violations.append("V09_STAGING_INSUFFICIENT_CLASSIFIED")
            stage_vector = np.zeros(2, dtype=float)
        stage_vector -= stage_vector.max()
        stage_likelihood = np.exp(stage_vector)
        stage_likelihood /= stage_likelihood.sum()
        probabilities = {
            "HC": float(screen_likelihood[0]),
            "MCI": float(screen_likelihood[1] * stage_likelihood[0]),
            "AD": float(screen_likelihood[1] * stage_likelihood[1]),
        }
        if candidate.get("screening_class") != ["HC", "impaired"][
            int(np.argmax(screen_vector))
        ]:
            violations.append("V08_SCREENING_SCORE_MISMATCH")
        if staging_available and candidate.get("staging_class") != ["MCI", "AD"][
            int(np.argmax(stage_vector))
        ]:
            violations.append("V08_STAGING_SCORE_MISMATCH")
    else:
        screen_likelihood = None
        stage_likelihood = None
        raw_scores = candidate.get("evidence_scores")
        if isinstance(raw_scores, dict) and all(label in raw_scores for label in labels):
            score_vector = np.asarray([float(raw_scores[label]) for label in labels])
            score_vector -= score_vector.max()
            likelihood_vector = np.exp(score_vector)
            likelihood_vector /= likelihood_vector.sum()
            probabilities = {
                label: float(likelihood_vector[index])
                for index, label in enumerate(labels)
            }
            highest_evidence_label = labels[int(np.argmax(score_vector))]
        else:
            # Compatibility path for 8.27 caches and migration fixtures only.
            raw_likelihoods = candidate.get(
                "evidence_likelihoods", candidate.get("proposed_probabilities", {})
            )
            probabilities = normalize_probabilities(raw_likelihoods, labels)
            highest_evidence_label = max(probabilities, key=probabilities.get)
        evidence_class = str(
            candidate.get("evidence_class", candidate.get("proposed_class", ""))
        )
        if evidence_class not in labels or highest_evidence_label != evidence_class:
            violations.append("V08_CLASS_SCORE_MISMATCH")
    if candidate.get("action") == "classify" and not used:
        violations.append("V04_UNSUPPORTED_CLASSIFICATION")
    result = {
        "valid": not violations,
        "violations": list(dict.fromkeys(violations)),
        "normalized_evidence_likelihoods": probabilities,
        "state_update_factor": _state_update_factor(candidate, workspace),
        # Compatibility alias for downstream readers of 8.27 audit files.
        "normalized_probabilities": probabilities,
        "rollback": bool(violations),
    }
    if hierarchical_scores:
        result["normalized_screening_likelihoods"] = {
            "HC": float(screen_likelihood[0]),
            "impaired": float(screen_likelihood[1]),
        }
        result["normalized_staging_likelihoods"] = {
            "MCI": float(stage_likelihood[0]),
            "AD": float(stage_likelihood[1]),
        }
        result["staging_available"] = staging_available
    return result


def _state_update_factor(
    candidate: dict[str, Any],
    workspace: dict[str, Any],
    downweight_factor: float = 0.5,
) -> float:
    """Convert validated state maintenance actions into an evidence-permission factor.

    State updates cannot create disease evidence. They can only retain or reduce
    the Agent correction when a state is unreliable, invalid, or unobservable.
    """

    states = {
        (str(item.get("state_id", "")), str(item.get("task_scope", "overall"))): item
        for item in workspace.get("state_observations", [])
    }
    if not states:
        return 0.0
    factors = {key: 1.0 for key in states}
    action_factor = {
        "keep": 1.0,
        "downweight": float(np.clip(downweight_factor, 0.0, 1.0)),
        "invalidate": 0.0,
        "mark_unavailable": 0.0,
    }
    for update in candidate.get("state_updates", []):
        state_id = str(update.get("state_id", ""))
        task_scope = str(update.get("task_scope", "overall"))
        key = (state_id, task_scope)
        if key not in states and task_scope not in {"", "overall", "nan"}:
            key = (f"{state_id}__task_{task_scope}", task_scope)
        if key in factors:
            factors[key] = min(
                factors[key], action_factor.get(str(update.get("action", "")), 0.0)
            )
    weights = np.asarray(
        [
            max(
                abs(float(item.get("state_z", 0.0)))
                * float(item.get("confidence", 0.0))
                * (1.0 - float(item.get("missing_fraction", 0.0))),
                1e-6,
            )
            for item in states.values()
        ],
        dtype=float,
    )
    retained = np.asarray([factors[key] for key in states], dtype=float)
    return float(np.clip(np.average(retained, weights=weights), 0.0, 1.0))


def _prediction_rows(
    prior: pd.DataFrame,
    labels: list[str],
    decisions: dict[str, dict[str, Any]],
    audits: dict[str, dict[str, Any]],
    workspaces: dict[str, dict[str, Any]],
    correction_strength: float,
    staging_correction_strength: float = 0.0,
    screening_calibrator: dict[str, Any] | None = None,
    staging_calibrator: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    locked_workspaces: list[dict[str, Any]] = []
    for item in prior.to_dict("records"):
        subject_id = str(item["subject_id"])
        case_id = case_pseudonym(subject_id)
        workspace = workspaces.get(case_id, {})
        base = np.asarray([[float(item[f"prob_{label}"]) for label in labels]])
        candidate = decisions.get(case_id)
        audit = audits.get(case_id)
        action = str(candidate.get("action")) if candidate else "not_run"
        raw_gate = float(workspace.get("correction_gate", 0.0))
        state_update_factor = float(audit.get("state_update_factor", 0.0)) if audit else 0.0
        gate = raw_gate * state_update_factor
        candidate_valid = bool(candidate and audit and audit["valid"])
        hierarchical_agent = bool(
            audit
            and "normalized_screening_likelihoods" in audit
            and "normalized_staging_likelihoods" in audit
        )
        eligible = bool(
            candidate_valid
            and action == "classify"
            and (correction_strength > 0 or staging_correction_strength > 0)
            and gate > 0
        )
        if eligible:
            if hierarchical_agent and labels == ["HC", "MCI", "AD"]:
                screening = np.asarray(
                    [[audit["normalized_screening_likelihoods"][key] for key in ["HC", "impaired"]]]
                )
                staging = np.asarray(
                    [[audit["normalized_staging_likelihoods"][key] for key in ["MCI", "AD"]]]
                )
                screening = apply_binary_evidence_calibrator(
                    screening, screening_calibrator
                )
                staging = apply_binary_evidence_calibrator(staging, staging_calibrator)
                route_parameters = two_stage_route_parameters(
                    base[0],
                    screening[0],
                    staging[0],
                    gate,
                    bool(audit.get("staging_available", False)),
                )
                screening_route = route_parameters["screening_route"]
                staging_route = route_parameters["staging_route"]
                staging_gate = route_parameters["staging_gate"]
                final = fuse_two_stage_evidence(
                    base,
                    screening,
                    staging,
                    np.asarray([gate]),
                    np.asarray([staging_gate]),
                    correction_strength,
                    staging_correction_strength,
                    np.asarray([route_parameters["screening_multiplier"]]),
                    np.asarray([route_parameters["staging_multiplier"]]),
                )[0]
                route = {
                    "route": f"screening:{screening_route['route']}|staging:{staging_route['route']}",
                    "route_multiplier": screening_route["route_multiplier"],
                }
            else:
                evidence_likelihood = np.asarray(
                    [[audit["normalized_evidence_likelihoods"][label] for label in labels]]
                )
                route = route_case(
                    base[0],
                    evidence_likelihood[0],
                    gate,
                    hierarchical_reference_index=0 if len(labels) > 2 else None,
                )
                final = fuse_evidence_likelihood(
                    base,
                    evidence_likelihood,
                    np.asarray([gate]),
                    correction_strength,
                    np.asarray([route["route_multiplier"]]),
                    hierarchical_reference_index=0 if len(labels) > 2 else None,
                )[0]
            correction_applied = bool(np.abs(final - base[0]).sum() > 1e-9)
            status = (
                "applied_evidence_likelihood_correction"
                if correction_applied
                else "validated_no_effect"
            )
        else:
            final = base[0]
            route = {
                "route": "not_eligible",
                "route_multiplier": 0.0,
                "prior_margin": float("nan"),
                "prior_evidence_agreement": False,
            }
            correction_applied = False
            status = (
                "rolled_back_to_prior"
                if candidate and audit and not audit["valid"]
                else "held_supervised_prior"
            )
        decision_changed = bool(labels[int(np.argmax(final))] != str(item["predicted_label"]))
        row = dict(item)
        row.update({f"prob_{label}": float(final[index]) for index, label in enumerate(labels)})
        row["predicted_label"] = labels[int(np.argmax(final))]
        row["condition"] = "Ours"
        row["agent_decision_status"] = status
        row["agent_action"] = action
        row["agent_gate"] = gate
        row["agent_raw_gate"] = raw_gate
        row["agent_state_update_factor"] = state_update_factor
        row["agent_case_route"] = route["route"]
        row["agent_route_multiplier"] = route["route_multiplier"]
        row["agent_correction_strength"] = float(correction_strength)
        row["agent_staging_correction_strength"] = float(staging_correction_strength)
        row["agent_candidate_valid"] = candidate_valid
        row["agent_correction_applied"] = correction_applied
        row["agent_decision_changed"] = decision_changed
        rows.append(row)
        if workspace:
            locked = json.loads(json.dumps(workspace, ensure_ascii=False))
            locked["supervised_prior_probabilities"] = {
                label: float(base[0, index]) for index, label in enumerate(labels)
            }
            locked["agent_candidate"] = candidate
            locked["agent_validation"] = audit
            locked["agent_case_route"] = route
            locked["final_probabilities"] = {
                label: float(final[index]) for index, label in enumerate(labels)
            }
            locked["final_prediction"] = row["predicted_label"]
            locked["agent_decision_status"] = status
            locked_workspaces.append(locked)
    return pd.DataFrame(rows), locked_workspaces


def _test_agent_gate_passed(
    provider: str,
    correction_strength: float,
    staging_correction_strength: float,
    calibration_summary: dict[str, Any],
) -> bool:
    """Run held-out Agent inference only after a development-set gain gate passes."""

    return bool(
        provider != "disabled"
        and calibration_summary.get("status") == "completed"
        and calibration_summary.get("selection_status") == "validated_joint_gain"
        and (correction_strength > 0.0 or staging_correction_strength > 0.0)
    )


def run_cognitive_diagnostic_agent(
    root: Path,
    prior_predictions_path: Path,
    workspaces_path: Path,
    agents_config: dict[str, Any],
    provider: str,
    predictions_path: Path,
    decisions_path: Path,
    audit_path: Path,
    locked_workspaces_path: Path,
    status_path: Path,
    prompt_path: Path,
    calibration_predictions_path: Path | None = None,
    calibration_workspaces_path: Path | None = None,
    calibration_result_path: Path | None = None,
) -> None:
    labels = [str(label) for label in agents_config["labels"]]
    prior = pd.read_csv(prior_predictions_path, dtype={"subject_id": str})
    workspaces = _read_workspaces(workspaces_path)
    skill = _skill_text(root)
    evidence_scoring_instruction = (
        "先独立判断证据更支持认知保留（HC）还是认知受损（impaired），分别给出0至4级证据分。"
        "再判断受损证据是否足以区分MCI与AD：足够时 staging_action=classify，不足时 staging_action=insufficient；"
        "足够时填写MCI或AD作为staging_class，并使其对应较高的证据分；"
        "不足时staging_class必须为undetermined，MCI与AD分数必须相同。"
        "screening_class必须对应较高的筛查证据分；不要生成最终风险概率。"
        if labels == ["HC", "MCI", "AD"]
        else "evidence_scores 必须对每个类别给出0至4的整数证据等级，evidence_class必须是最高等级类别。"
    )
    instruction = (
        skill
        + "\n\n你正在执行盲化的证据诊断阶段，不是撰写报告。逐例检查状态、指标、片段、反证和质量限制。"
        + "你看不到监督模型概率，也不得猜测或复制其他模型的输出。"
        + "cognitive_state_reference 是仅由当前训练折建立的类别平衡认知状态参照："
        + "比较病例状态与HC、MCI、AD训练中位数，但不得把任务类型、语言或类别样本量当作疾病证据。"
        + "参照卡没有提供某病例的类别概率；最终判断仍必须引用该病例自身的状态、指标或片段证据ID。"
        + "只有临床证据 ID 可以进入 used_evidence_ids/counterevidence_ids；质量证据只能进入 quality_evidence_ids。"
        + "所有 evidence_ids 必须复制 evidence_registry 中带类型前缀的 evidence_id；state_updates.state_id 则复制状态对象中的原始 state_id 字段。"
        + "inference_only_state_observations 和 inference_only_metric_observations 可以用于研究性类别判断，"
        + "但没有临床主张权限，不能被描述成疾病机制，也不能进入医生报告。"
        + "state_updates 中，质量证据只能用于 downweight、invalidate 或 mark_unavailable；不得用于 keep 或支持风险。"
        + "model_only_state_observations 不能作为临床支持，但可以且只能用 mark_unavailable 明确排除。"
        + "mark_unavailable 的 evidence_ids 可以包含目标 state_id 本身及说明缺失原因的质量证据 ID。"
        + evidence_scoring_instruction
        + "系统会在 Agent 之外校准离散等级并与监督概率融合。"
        + "如果证据冲突但仍足以形成判断，选择 classify 并如实给出相对似然；质量不足时选择 abstain 或 retest。不要输出隐藏推理过程。"
        + f"允许类别仅为：{', '.join(labels)}。"
    )
    prompt_path.write_text(instruction, encoding="utf-8")
    request_failures: list[dict[str, Any]] = []

    def request_cases(
        cases_to_run: list[dict[str, Any]], artifact_prefix: str
    ) -> dict[str, dict[str, Any]]:
        if provider == "disabled" or not cases_to_run:
            return {}
        schema_path = status_path.parent / "cognitive_agent_schema.json"
        _batch_schema(schema_path, labels)
        batch_size = max(1, int(agents_config.get("diagnostic_agent_batch_size", 4)))
        workers = max(1, int(agents_config.get("diagnostic_agent_workers", 2)))
        jobs = []
        for start in range(0, len(cases_to_run), batch_size):
            cases = cases_to_run[start : start + batch_size]
            number = start // batch_size + 1
            jobs.append(
                {
                    "number": number,
                    "cases": cases,
                    "case_ids": [str(case["case_id"]) for case in cases],
                    "output": status_path.parent / f"{artifact_prefix}_batch_{number:03d}.json",
                    "meta": status_path.parent / f"{artifact_prefix}_batch_{number:03d}.meta.json",
                }
            )
        responses: dict[int, dict[str, Any]] = {}
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(
                    _run_batch,
                    root=root,
                    provider=provider,
                    model=str(agents_config["model"]),
                    prompt=instruction
                    + "\n\n病例工作区如下。病例相互独立，每例必须返回一次：\n"
                    + json.dumps(job["cases"], ensure_ascii=False),
                    schema_path=schema_path,
                    output_path=job["output"],
                    metadata_path=job["meta"],
                    case_ids=job["case_ids"],
                ): job
                for job in jobs
            }
            for future in as_completed(futures):
                job = futures[future]
                try:
                    responses[job["number"]] = future.result()
                except Exception as error:
                    request_failures.append(
                        {
                            "artifact_prefix": artifact_prefix,
                            "batch": int(job["number"]),
                            "case_ids": list(job["case_ids"]),
                            "error": f"{type(error).__name__}: {error}",
                        }
                    )
        expected_case_ids = {str(case["case_id"]) for case in cases_to_run}
        requested: dict[str, dict[str, Any]] = {}
        for number in sorted(responses):
            indexed, unexpected = _index_expected_candidates(
                responses[number].get("cases", []), expected_case_ids
            )
            requested.update(indexed)
            if unexpected:
                request_failures.append(
                    {
                        "artifact_prefix": artifact_prefix,
                        "batch": int(number),
                        "case_ids": unexpected,
                        "error": "Unexpected case_id returned by Agent",
                    }
                )
        missing_cases = [
            case for case in cases_to_run if str(case["case_id"]) not in requested
        ]
        if missing_cases:
            retry_responses: dict[int, dict[str, Any]] = {}
            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = {
                    executor.submit(
                        _run_batch,
                        root=root,
                        provider=provider,
                        model=str(agents_config["model"]),
                        prompt=instruction
                        + "\n\n首次批量响应漏掉了该病例。只返回下面这一例，case_id 必须完全一致：\n"
                        + json.dumps([case], ensure_ascii=False),
                        schema_path=schema_path,
                        output_path=status_path.parent
                        / f"{artifact_prefix}_retry_{index:03d}.json",
                        metadata_path=status_path.parent
                        / f"{artifact_prefix}_retry_{index:03d}.meta.json",
                        case_ids=[str(case["case_id"])],
                    ): {
                        "number": index,
                        "case_ids": [str(case["case_id"])],
                    }
                    for index, case in enumerate(missing_cases, start=1)
                }
                for future in as_completed(futures):
                    job = futures[future]
                    try:
                        retry_responses[job["number"]] = future.result()
                    except Exception as error:
                        request_failures.append(
                            {
                                "artifact_prefix": f"{artifact_prefix}_retry",
                                "batch": int(job["number"]),
                                "case_ids": list(job["case_ids"]),
                                "error": f"{type(error).__name__}: {error}",
                            }
                        )
            for number in sorted(retry_responses):
                indexed, unexpected = _index_expected_candidates(
                    retry_responses[number].get("cases", []), expected_case_ids
                )
                requested.update(indexed)
                if unexpected:
                    request_failures.append(
                        {
                            "artifact_prefix": f"{artifact_prefix}_retry",
                            "batch": int(number),
                            "case_ids": unexpected,
                            "error": "Unexpected case_id returned by Agent",
                        }
                    )
        return requested

    def repair_invalid_cases(
        decisions_to_check: dict[str, dict[str, Any]],
        audits_to_check: dict[str, dict[str, Any]],
        source_workspaces: dict[str, dict[str, Any]],
        artifact_prefix: str,
    ) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], int, int]:
        """Retry contract failures once without exposing labels or model priors."""

        repair_cases: list[dict[str, Any]] = []
        for case_id, audit in audits_to_check.items():
            if audit["valid"] or case_id not in source_workspaces:
                continue
            repair_case = blind_workspace(source_workspaces[case_id])
            repair_case["contract_repair"] = {
                "previous_violations": list(audit["violations"]),
                "instruction": (
                    "Return a complete replacement decision for this case. Correct only "
                    "the listed contract violations by copying valid IDs from this workspace. "
                    "Do not infer labels or probabilities from the violation codes."
                ),
            }
            repair_cases.append(repair_case)
        repaired = request_cases(repair_cases, artifact_prefix)
        accepted_repairs = 0
        for case_id, candidate in repaired.items():
            if case_id not in source_workspaces:
                continue
            repaired_audit = validate_candidate(
                candidate, source_workspaces[case_id], labels
            )
            decisions_to_check[case_id] = candidate
            audits_to_check[case_id] = repaired_audit
            accepted_repairs += int(repaired_audit["valid"])
        return (
            decisions_to_check,
            audits_to_check,
            int(len(repair_cases)),
            int(accepted_repairs),
        )

    correction_strength = float(
        agents_config.get("diagnostic_agent_correction_strength", 0.0)
    )
    staging_correction_strength = 0.0
    screening_calibrator: dict[str, Any] | None = None
    staging_calibrator: dict[str, Any] | None = None
    calibration_summary: dict[str, Any] = {
        "status": "not_available",
        "selected_strength": correction_strength,
        "selected_screening_strength": correction_strength,
        "selected_staging_strength": staging_correction_strength,
    }
    if (
        provider != "disabled"
        and calibration_predictions_path is not None
        and calibration_workspaces_path is not None
        and calibration_predictions_path.exists()
        and calibration_workspaces_path.exists()
    ):
        calibration_prior = pd.read_csv(
            calibration_predictions_path, dtype={"subject_id": str}
        )
        calibration_workspaces = _read_workspaces(calibration_workspaces_path)
        calibration_cases = [
            blind_workspace(calibration_workspaces[case_pseudonym(subject_id)])
            for subject_id in calibration_prior["subject_id"].astype(str)
            if case_pseudonym(subject_id) in calibration_workspaces
        ]
        calibration_decisions = request_cases(
            calibration_cases, "cognitive_agent_calibration"
        )
        calibration_audits = {
            case_id: validate_candidate(
                candidate, calibration_workspaces[case_id], labels
            )
            for case_id, candidate in calibration_decisions.items()
            if case_id in calibration_workspaces
        }
        (
            calibration_decisions,
            calibration_audits,
            calibration_repair_requested,
            calibration_repair_accepted,
        ) = repair_invalid_cases(
            calibration_decisions,
            calibration_audits,
            calibration_workspaces,
            "cognitive_agent_calibration_repair",
        )
        likelihood_rows: list[list[float]] = []
        screening_rows: list[list[float]] = []
        staging_rows: list[list[float]] = []
        gate_rows: list[float] = []
        staging_gate_rows: list[float] = []
        multiplier_rows: list[float] = []
        staging_multiplier_rows: list[float] = []
        valid_classify_count = 0
        for row_index, item in enumerate(calibration_prior.to_dict("records")):
            case_id = case_pseudonym(str(item["subject_id"]))
            candidate = calibration_decisions.get(case_id)
            audit = calibration_audits.get(case_id)
            workspace = calibration_workspaces.get(case_id, {})
            base = [float(item[f"prob_{label}"]) for label in labels]
            if (
                candidate
                and audit
                and audit["valid"]
                and str(candidate.get("action")) == "classify"
            ):
                likelihood = [
                    float(audit["normalized_evidence_likelihoods"][label])
                    for label in labels
                ]
                raw_gate = float(workspace.get("correction_gate", 0.0))
                gate = raw_gate * float(audit.get("state_update_factor", 0.0))
                if "normalized_screening_likelihoods" in audit:
                    screening = [
                        float(audit["normalized_screening_likelihoods"][key])
                        for key in ["HC", "impaired"]
                    ]
                    staging = [
                        float(audit["normalized_staging_likelihoods"][key])
                        for key in ["MCI", "AD"]
                    ]
                    route_parameters = two_stage_route_parameters(
                        np.asarray(base, dtype=float),
                        np.asarray(screening, dtype=float),
                        np.asarray(staging, dtype=float),
                        gate,
                        bool(audit.get("staging_available", False)),
                    )
                    multiplier = route_parameters["screening_multiplier"]
                    staging_gate = route_parameters["staging_gate"]
                    staging_multiplier = route_parameters["staging_multiplier"]
                else:
                    route = route_case(
                        base,
                        likelihood,
                        gate,
                        hierarchical_reference_index=0 if len(labels) > 2 else None,
                    )
                    multiplier = float(route["route_multiplier"])
                    screening = [float(likelihood[0]), float(sum(likelihood[1:]))]
                    staging = [0.5, 0.5]
                    staging_gate = 0.0
                    staging_multiplier = 0.0
                valid_classify_count += 1
            else:
                likelihood = [1.0 / len(labels) for _ in labels]
                screening = [0.5, 0.5]
                staging = [0.5, 0.5]
                gate = 0.0
                staging_gate = 0.0
                multiplier = 0.0
                staging_multiplier = 0.0
            likelihood_rows.append(likelihood)
            screening_rows.append(screening)
            staging_rows.append(staging)
            gate_rows.append(gate)
            staging_gate_rows.append(staging_gate)
            multiplier_rows.append(multiplier)
            staging_multiplier_rows.append(staging_multiplier)
        minimum_cases = max(
            len(labels) * 5,
            int(agents_config.get("diagnostic_agent_min_calibration_cases", 30)),
        )
        if valid_classify_count >= minimum_cases:
            strength_grid = [
                    float(value)
                    for value in agents_config.get(
                        "diagnostic_agent_strength_grid", [0.0, 0.25, 0.5, 1.0]
                    )
                ]
            minimum_gain = float(
                agents_config.get("diagnostic_agent_min_macro_f1_gain", 0.0)
            )
            noninferiority_margin = float(
                agents_config.get("diagnostic_agent_auroc_noninferiority_margin", 0.001)
            )
            if labels == ["HC", "MCI", "AD"]:
                calibration_truth = calibration_prior["label"].astype(str).to_numpy()
                screening_calibrator = fit_binary_evidence_calibrator(
                    (calibration_truth != "HC").astype(int),
                    np.asarray(screening_rows, dtype=float),
                    np.asarray(gate_rows, dtype=float) > 0,
                )
                staging_calibrator = fit_binary_evidence_calibrator(
                    (calibration_truth == "AD").astype(int),
                    np.asarray(staging_rows, dtype=float),
                    (np.asarray(staging_gate_rows, dtype=float) > 0)
                    & np.isin(calibration_truth, ["MCI", "AD"]),
                )
                calibration_fit = fit_agent_two_stage_strengths(
                    calibration_truth,
                    calibration_prior[[f"prob_{label}" for label in labels]].to_numpy(dtype=float),
                    screening_calibrator["oof_likelihood"],
                    staging_calibrator["oof_likelihood"],
                    np.asarray(gate_rows, dtype=float),
                    np.asarray(staging_gate_rows, dtype=float),
                    np.asarray(multiplier_rows, dtype=float),
                    np.asarray(staging_multiplier_rows, dtype=float),
                    strength_grid,
                    minimum_macro_f1_gain=minimum_gain,
                    auroc_noninferiority_margin=noninferiority_margin,
                )
                correction_strength = float(calibration_fit["selected_screening_strength"])
                staging_correction_strength = float(calibration_fit["selected_staging_strength"])
                calibration_fit["screening_score_calibrator"] = {
                    key: value
                    for key, value in screening_calibrator.items()
                    if key != "oof_likelihood"
                }
                calibration_fit["staging_score_calibrator"] = {
                    key: value
                    for key, value in staging_calibrator.items()
                    if key != "oof_likelihood"
                }
            else:
                calibration_fit = fit_agent_correction_strength(
                    calibration_prior["label"].astype(str).to_numpy(),
                    labels,
                    calibration_prior[[f"prob_{label}" for label in labels]].to_numpy(dtype=float),
                    np.asarray(likelihood_rows, dtype=float),
                    np.asarray(gate_rows, dtype=float),
                    np.asarray(multiplier_rows, dtype=float),
                    strength_grid,
                    minimum_macro_f1_gain=minimum_gain,
                    auroc_noninferiority_margin=noninferiority_margin,
                )
                correction_strength = float(calibration_fit["selected_strength"])
            calibration_summary = {
                "status": "completed",
                "available_cases": int(len(calibration_prior)),
                "agent_returned_cases": int(len(calibration_decisions)),
                "valid_classify_cases": int(valid_classify_count),
                "contract_repair_requested": calibration_repair_requested,
                "contract_repair_accepted": calibration_repair_accepted,
                **calibration_fit,
            }
        else:
            correction_strength = 0.0
            staging_correction_strength = 0.0
            calibration_summary = {
                "status": "failed_closed_insufficient_valid_cases",
                "available_cases": int(len(calibration_prior)),
                "agent_returned_cases": int(len(calibration_decisions)),
                "valid_classify_cases": int(valid_classify_count),
                "contract_repair_requested": calibration_repair_requested,
                "contract_repair_accepted": calibration_repair_accepted,
                "minimum_required": int(minimum_cases),
                "selected_strength": 0.0,
                "selected_screening_strength": 0.0,
                "selected_staging_strength": 0.0,
            }
    if calibration_result_path is not None:
        json_dump(calibration_summary, calibration_result_path)

    test_agent_gate_passed = _test_agent_gate_passed(
        provider,
        correction_strength,
        staging_correction_strength,
        calibration_summary,
    )
    if test_agent_gate_passed:
        selected = select_agent_cohort(
            prior,
            int(agents_config.get("diagnostic_agent_evaluation_cap", len(prior))),
        )
        selected_cases = [
            blind_workspace(workspaces[case_pseudonym(subject_id)])
            for subject_id in selected["subject_id"].astype(str)
            if case_pseudonym(subject_id) in workspaces
        ]
        decisions = request_cases(selected_cases, "cognitive_agent_test")
    else:
        selected_cases = []
        decisions = {}

    audits = {
        case_id: validate_candidate(candidate, workspaces[case_id], labels)
        for case_id, candidate in decisions.items()
        if case_id in workspaces
    }
    (
        decisions,
        audits,
        test_repair_requested,
        test_repair_accepted,
    ) = repair_invalid_cases(
        decisions,
        audits,
        workspaces,
        "cognitive_agent_test_repair",
    )
    predictions, locked = _prediction_rows(
        prior,
        labels,
        decisions,
        audits,
        workspaces,
        correction_strength,
        staging_correction_strength,
        screening_calibrator,
        staging_calibrator,
    )
    predictions.to_csv(predictions_path, index=False)
    with decisions_path.open("w", encoding="utf-8") as handle:
        for candidate in decisions.values():
            handle.write(json.dumps(candidate, ensure_ascii=False) + "\n")
    with audit_path.open("w", encoding="utf-8") as handle:
        for case_id, audit in audits.items():
            handle.write(json.dumps({"case_id": case_id, **audit}, ensure_ascii=False) + "\n")
    with locked_workspaces_path.open("w", encoding="utf-8") as handle:
        for workspace in locked:
            handle.write(json.dumps(workspace, ensure_ascii=False) + "\n")
    accepted = int(predictions["agent_correction_applied"].astype(bool).sum())
    decision_changes = int(predictions["agent_decision_changed"].astype(bool).sum())
    valid_candidates = int(predictions["agent_candidate_valid"].astype(bool).sum())
    rollback = int(predictions["agent_decision_status"].eq("rolled_back_to_prior").sum())
    json_dump(
        {
            "condition": "B3_evidence_governed_single_diagnostic_agent",
            "status": (
                "completed"
                if test_agent_gate_passed
                else "held_prior_development_gate_not_passed"
            ),
            "provider": provider,
            "model": agents_config.get("model"),
            "held_out_cases": int(len(prior)),
            "agent_requested_cases": int(len(selected_cases)),
            "agent_returned_cases": int(len(decisions)),
            "accepted_bounded_corrections": accepted,
            "valid_candidates": valid_candidates,
            "applied_probability_corrections": accepted,
            "changed_class_decisions": decision_changes,
            "contract_repair_requested": test_repair_requested,
            "contract_repair_accepted": test_repair_accepted,
            "request_failure_count": int(len(request_failures)),
            "request_failures": request_failures,
            "correction_strength": correction_strength,
            "staging_correction_strength": staging_correction_strength,
            "correction_calibration_status": calibration_summary["status"],
            "correction_selection_status": calibration_summary.get(
                "selection_status", "not_available"
            ),
            "test_agent_gate_passed": test_agent_gate_passed,
            "rolled_back_candidates": rollback,
            "unchanged_or_outside_cohort": int(len(prior) - accepted - rollback),
            "test_labels_exposed_to_agent": False,
            "supervised_prior_exposed_to_agent": False,
            "evidence_output_contract": (
                "two_stage_discrete_scores_0_4"
                if labels == ["HC", "MCI", "AD"]
                else "discrete_scores_0_4"
            ),
            "fusion_policy": (
                "separate_hc_vs_impairment_and_mci_vs_ad"
                if len(labels) > 2
                else "binary_log_opinion_pool"
            ),
            "created_at_utc": now_utc(),
        },
        status_path,
    )
