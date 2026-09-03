from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from .agent_runtime import case_pseudonym


DIAGNOSTIC_ROLES = {
    "clinical",
    "clinical_support",
    "cautious_support",
    "model_and_report",
}
QUALITY_ROLES = {"qc", "qc_only", "quality_control"}
INFERENCE_AUXILIARY_BRANCHES = {
    "language",
    "task_performance",
    "interaction",
    "speech_behavior",
}


def _private_id(value: Any, prefix: str) -> str:
    if prefix == "CASE":
        return case_pseudonym(str(value))
    import hashlib

    digest = hashlib.sha256(f"advoice-8.27::{value}".encode("utf-8")).hexdigest()[:12]
    return f"{prefix}_{digest}"


def _typed_evidence_id(kind: str, value: Any) -> str:
    text = str(value)
    return text if text.startswith(f"{kind}:") else f"{kind}:{text}"


def _deduplicate_evidence_registry(
    items: Sequence[Mapping[str, Any]],
) -> list[dict[str, str]]:
    """Return one typed registry entry per evidence object."""

    registry: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for item in items:
        evidence_id = str(item.get("evidence_id", ""))
        evidence_type = str(item.get("evidence_type", ""))
        key = (evidence_id, evidence_type)
        if not evidence_id or not evidence_type or key in seen:
            continue
        seen.add(key)
        registry.append(
            {"evidence_id": evidence_id, "evidence_type": evidence_type}
        )
    return registry


def _task_family(task_scope: str) -> str:
    scope = str(task_scope).lower()
    if any(token in scope for token in ("picture", "cookie", "pitt")):
        return "picture_description"
    if any(token in scope for token in ("fluency", "animal", "letter")):
        return "verbal_fluency"
    if any(token in scope for token in ("recall", "memory", "story")):
        return "memory_recall"
    if any(token in scope for token in ("read", "reading")):
        return "reading"
    if any(token in scope for token in ("interview", "dialog", "question")):
        return "structured_interview"
    return "other_speech_task"


def case_input_route(
    languages: Sequence[str], task_scopes: Sequence[str], branches: Sequence[str]
) -> dict[str, Any]:
    """Describe the observable case type without using an outcome label."""

    tasks = sorted(
        {
            str(value)
            for value in task_scopes
            if str(value).lower() not in {"", "overall", "nan"}
        }
    )
    families = sorted({_task_family(task) for task in tasks})
    language_values = sorted(
        {str(value) for value in languages if str(value).lower() not in {"", "nan"}}
    )
    return {
        "task_structure": "multitask" if len(tasks) > 1 else "single_task",
        "task_families": families or ["unspecified_speech_task"],
        "task_scopes": tasks,
        "language_scope": "multilingual" if len(language_values) > 1 else "single_language",
        "languages": language_values,
        "available_branches": sorted({str(value) for value in branches}),
        "fallback_policy": "shared_model_then_parent_task_family",
    }


def _sanitize_segment(segment: Mapping[str, Any]) -> dict[str, Any]:
    item = dict(segment)
    if item.get("segment_id"):
        item["segment_id"] = _private_id(item["segment_id"], "SEG")
    if item.get("case_id"):
        item["case_id"] = _private_id(item["case_id"], "REC")
    item.pop("audio_path", None)
    item.pop("source_path", None)
    return item


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return []
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _bounded(value: float) -> float:
    return float(np.clip(float(value), 0.0, 1.0))


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes"}
    return bool(value)


def evidence_gate(
    coverage: float | np.ndarray,
    reliability: float | np.ndarray,
    confound_burden: float | np.ndarray,
) -> float | np.ndarray:
    """Return evidence permission without multiplying three weak terms together."""

    coverage_array = np.clip(np.asarray(coverage, dtype=float), 0.0, 1.0)
    reliability_array = np.clip(np.asarray(reliability, dtype=float), 0.0, 1.0)
    confound_array = np.clip(np.asarray(confound_burden, dtype=float), 0.0, 1.0)
    result = np.sqrt(coverage_array * reliability_array) * np.sqrt(
        1.0 - confound_array
    )
    return float(result) if result.ndim == 0 else result


def structured_evidence_coverage(
    all_states: Sequence[Mapping[str, Any]],
    observable_states: Sequence[Mapping[str, Any]],
) -> dict[str, float]:
    """Measure whether expected states, tasks and branches are observable.

    Metric count is deliberately excluded: adding correlated measurements must not
    reduce the permission granted to a case.
    """

    def valid_scope(value: Any) -> str:
        scope = str(value or "overall")
        return "overall" if scope.lower() in {"", "nan"} else scope

    expected = {
        (str(item.get("state_id", "")), valid_scope(item.get("task_scope")))
        for item in all_states
        if str(item.get("state_id", "")) not in {"", "QC"}
        and str(item.get("category", "")).lower() not in {"not_applicable"}
    }
    observed = {
        (str(item.get("state_id", "")), valid_scope(item.get("task_scope")))
        for item in observable_states
        if float(item.get("confidence", 0.0)) > 0.0
        and float(item.get("missing_fraction", 1.0)) < 1.0
    }
    expected_tasks = {scope for _, scope in expected if scope != "overall"}
    observed_tasks = {scope for _, scope in observed if scope != "overall"}
    expected_branches = {
        str(item.get("branch", ""))
        for item in all_states
        if str(item.get("branch", "")) not in {"", "qc"}
        and (str(item.get("state_id", "")), valid_scope(item.get("task_scope")))
        in expected
    }
    observed_branches = {
        str(item.get("branch", ""))
        for item in observable_states
        if str(item.get("branch", "")) not in {"", "qc"}
        and (str(item.get("state_id", "")), valid_scope(item.get("task_scope")))
        in observed
    }
    state_coverage = len(observed & expected) / max(len(expected), 1)
    task_coverage = (
        len(observed_tasks & expected_tasks) / len(expected_tasks)
        if expected_tasks
        else state_coverage
    )
    branch_coverage = (
        len(observed_branches & expected_branches) / len(expected_branches)
        if expected_branches
        else state_coverage
    )
    overall = 0.6 * state_coverage + 0.25 * task_coverage + 0.15 * branch_coverage
    return {
        "overall": float(np.clip(overall, 0.0, 1.0)),
        "state": float(state_coverage),
        "task": float(task_coverage),
        "branch": float(branch_coverage),
        "expected_state_views": int(len(expected)),
        "observed_state_views": int(len(observed & expected)),
    }


def route_case(
    base_probability: Sequence[float],
    evidence_likelihood: Sequence[float],
    gate: float,
    *,
    hierarchical_reference_index: int | None = None,
    low_gate_threshold: float = 0.25,
    boundary_margin: float = 0.15,
    stable_confidence: float = 0.80,
) -> dict[str, Any]:
    """Assign a label-free decision route after blind evidence assessment."""

    base = np.asarray(base_probability, dtype=float)
    evidence = np.asarray(evidence_likelihood, dtype=float)
    if base.ndim != 1 or evidence.shape != base.shape or len(base) < 2:
        raise ValueError("Base and evidence values must be equally shaped vectors.")
    base = base / max(base.sum(), 1e-12)
    evidence = evidence / max(evidence.sum(), 1e-12)
    if hierarchical_reference_index is not None and len(base) > 2:
        reference = int(hierarchical_reference_index)
        base_route = np.asarray([base[reference], base.sum() - base[reference]])
        nonreference = [index for index in range(len(base)) if index != reference]
        subtype = base[nonreference]
        subtype = (
            subtype / subtype.sum()
            if subtype.sum() > 0
            else np.full(len(nonreference), 1.0 / len(nonreference))
        )
        evidence_route = np.asarray(
            [evidence[reference], float(np.dot(evidence[nonreference], subtype))]
        )
        evidence_route /= max(evidence_route.sum(), 1e-12)
    else:
        base_route = base
        evidence_route = evidence
    ordered = np.sort(base_route)
    margin = float(ordered[-1] - ordered[-2])
    agreement = int(np.argmax(base_route)) == int(np.argmax(evidence_route))
    if float(gate) < low_gate_threshold:
        route, multiplier = "quality_limited", 0.0
    elif not agreement:
        # Conflict is a review signal, not permission to amplify an unvalidated
        # Agent correction beyond the ordinary evidence bound.
        route, multiplier = "model_evidence_conflict", 1.0
    elif margin <= boundary_margin:
        route, multiplier = "boundary_case", 1.0
    elif float(base_route.max()) >= stable_confidence:
        route, multiplier = "stable_agreement", 0.25
    else:
        route, multiplier = "routine_review", 0.60
    return {
        "route": route,
        "route_multiplier": multiplier,
        "prior_margin": margin,
        "prior_evidence_agreement": agreement,
    }


def fuse_evidence_likelihood(
    base_probability: np.ndarray,
    evidence_likelihood: np.ndarray,
    gate: np.ndarray,
    strength: float,
    route_multiplier: np.ndarray | None = None,
    hierarchical_reference_index: int | None = None,
) -> np.ndarray:
    """Combine a supervised prior with a blind Agent evidence likelihood."""

    base = np.asarray(base_probability, dtype=float)
    evidence = np.asarray(evidence_likelihood, dtype=float)
    if base.shape != evidence.shape or base.ndim != 2:
        raise ValueError("Base and evidence likelihoods must be equally shaped matrices.")
    gate_array = np.clip(np.asarray(gate, dtype=float).reshape(-1, 1), 0.0, 1.0)
    if gate_array.shape[0] != base.shape[0]:
        raise ValueError("A gate value is required for every case.")
    if route_multiplier is None:
        route_array = np.ones_like(gate_array)
    else:
        route_array = np.clip(
            np.asarray(route_multiplier, dtype=float).reshape(-1, 1), 0.0, 2.0
        )
    beta = max(float(strength), 0.0) * gate_array * route_array
    if hierarchical_reference_index is None or base.shape[1] <= 2:
        log_score = np.log(np.clip(base, 1e-8, 1.0)) + beta * np.log(
            np.clip(evidence, 1e-8, 1.0)
        )
        log_score -= log_score.max(axis=1, keepdims=True)
        score = np.exp(log_score)
        return score / score.sum(axis=1, keepdims=True)

    reference = int(hierarchical_reference_index)
    if reference < 0 or reference >= base.shape[1]:
        raise ValueError("hierarchical_reference_index is outside the class vector.")
    nonreference = [index for index in range(base.shape[1]) if index != reference]
    base_binary = np.column_stack(
        [base[:, reference], base[:, nonreference].sum(axis=1)]
    )
    # The evidence vector contains class-conditional likelihoods.  Collapsing a
    # K-class vector with sum(non-reference) makes uniform evidence directional
    # (for HC/MCI/AD, 1/3 vs 2/3) and therefore changes an otherwise neutral
    # supervised prior.  Marginalise the non-reference likelihood using the
    # supervised conditional subtype distribution instead.
    subtype_probability = base[:, nonreference]
    subtype_probability = np.divide(
        subtype_probability,
        subtype_probability.sum(axis=1, keepdims=True),
        out=np.full_like(subtype_probability, 1.0 / len(nonreference)),
        where=subtype_probability.sum(axis=1, keepdims=True) > 0,
    )
    evidence_nonreference = (
        evidence[:, nonreference] * subtype_probability
    ).sum(axis=1)
    evidence_binary = np.column_stack(
        [evidence[:, reference], evidence_nonreference]
    )
    binary_log_score = np.log(np.clip(base_binary, 1e-8, 1.0)) + beta * np.log(
        np.clip(evidence_binary, 1e-8, 1.0)
    )
    binary_log_score -= binary_log_score.max(axis=1, keepdims=True)
    binary_score = np.exp(binary_log_score)
    binary_probability = binary_score / binary_score.sum(axis=1, keepdims=True)
    result = np.zeros_like(base)
    result[:, reference] = binary_probability[:, 0]
    result[:, nonreference] = binary_probability[:, 1, None] * subtype_probability
    return result


def fuse_two_stage_evidence(
    base_probability: np.ndarray,
    screening_likelihood: np.ndarray,
    staging_likelihood: np.ndarray,
    screening_gate: np.ndarray,
    staging_gate: np.ndarray,
    screening_strength: float,
    staging_strength: float,
    screening_route_multiplier: np.ndarray | None = None,
    staging_route_multiplier: np.ndarray | None = None,
) -> np.ndarray:
    """Update HC/impaired and MCI/AD boundaries with separate evidence channels."""

    base = np.asarray(base_probability, dtype=float)
    screen = np.asarray(screening_likelihood, dtype=float)
    stage = np.asarray(staging_likelihood, dtype=float)
    if base.ndim != 2 or base.shape[1] != 3:
        raise ValueError("Two-stage fusion requires HC/MCI/AD base probabilities.")
    if screen.shape != (len(base), 2) or stage.shape != (len(base), 2):
        raise ValueError("Screening and staging likelihoods must each have shape (n, 2).")
    base = base / base.sum(axis=1, keepdims=True).clip(min=1e-12)
    screen = screen / screen.sum(axis=1, keepdims=True).clip(min=1e-12)
    stage = stage / stage.sum(axis=1, keepdims=True).clip(min=1e-12)

    def multiplier(value: np.ndarray | None) -> np.ndarray:
        if value is None:
            return np.ones((len(base), 1), dtype=float)
        return np.clip(np.asarray(value, dtype=float).reshape(-1, 1), 0.0, 1.0)

    screen_beta = (
        max(float(screening_strength), 0.0)
        * np.clip(np.asarray(screening_gate, dtype=float).reshape(-1, 1), 0.0, 1.0)
        * multiplier(screening_route_multiplier)
    )
    stage_beta = (
        max(float(staging_strength), 0.0)
        * np.clip(np.asarray(staging_gate, dtype=float).reshape(-1, 1), 0.0, 1.0)
        * multiplier(staging_route_multiplier)
    )
    impaired = base[:, 1:].sum(axis=1, keepdims=True)
    base_screen = np.concatenate([base[:, :1], impaired], axis=1)
    base_stage = np.divide(
        base[:, 1:],
        impaired,
        out=np.full((len(base), 2), 0.5, dtype=float),
        where=impaired > 0.0,
    )

    def update(prior: np.ndarray, likelihood: np.ndarray, beta: np.ndarray) -> np.ndarray:
        score = np.log(np.clip(prior, 1e-8, 1.0)) + beta * np.log(
            np.clip(likelihood, 1e-8, 1.0)
        )
        score -= score.max(axis=1, keepdims=True)
        score = np.exp(score)
        return score / score.sum(axis=1, keepdims=True)

    updated_screen = update(base_screen, screen, screen_beta)
    updated_stage = update(base_stage, stage, stage_beta)
    result = np.column_stack(
        [
            updated_screen[:, 0],
            updated_screen[:, 1] * updated_stage[:, 0],
            updated_screen[:, 1] * updated_stage[:, 1],
        ]
    )
    return result / result.sum(axis=1, keepdims=True)


def fuse_corrected_probability(
    base_probability: np.ndarray,
    corrected_probability: np.ndarray,
    gate: np.ndarray,
    alpha: float,
) -> np.ndarray:
    """Apply a bounded residual correction while preserving valid probabilities."""

    base = np.asarray(base_probability, dtype=float)
    corrected = np.asarray(corrected_probability, dtype=float)
    if base.shape != corrected.shape or base.ndim != 2:
        raise ValueError("Base and corrected probabilities must be equally shaped matrices.")
    gate_array = np.clip(np.asarray(gate, dtype=float).reshape(-1, 1), 0.0, 1.0)
    if gate_array.shape[0] != base.shape[0]:
        raise ValueError("A gate value is required for every case.")
    mixing = np.clip(float(alpha), 0.0, 1.0) * gate_array
    probability = (1.0 - mixing) * base + mixing * corrected
    denominator = probability.sum(axis=1, keepdims=True)
    return np.divide(
        probability,
        denominator,
        out=np.full_like(probability, 1.0 / probability.shape[1]),
        where=denominator > 0,
    )


def validate_agent_trace(
    trace: Sequence[Mapping[str, Any]], valid_evidence_ids: Iterable[str]
) -> dict[str, Any]:
    valid = {str(value) for value in valid_evidence_ids}
    unknown: list[str] = []
    for action in trace:
        evidence_id = action.get("evidence_id")
        if evidence_id is not None and str(evidence_id) not in valid:
            unknown.append(str(evidence_id))
    return {
        "valid": not unknown,
        "unknown_evidence_ids": list(dict.fromkeys(unknown)),
        "action_count": len(trace),
    }


def _metric_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    confounds = [str(value) for value in _json_list(row.get("confound_tags"))]
    branch = str(row.get("branch", ""))
    role = str(row.get("evidence_role", ""))
    report_permission = _truthy(row.get("report_permission", False))
    return {
        "evidence_id": str(
            row.get("metric_instance_id") or row.get("metric_id") or "unknown_metric"
        ),
        "metric_id": str(row.get("metric_id", "")),
        "state_id": str(row.get("state_id", "")),
        "branch": branch,
        "task_scope": str(row.get("task_scope", "overall")),
        "value": float(pd.to_numeric(row.get("value"), errors="coerce")),
        "directional_z": float(
            np.nan_to_num(pd.to_numeric(row.get("directional_z"), errors="coerce"))
        ),
        "reliability": _bounded(
            np.nan_to_num(pd.to_numeric(row.get("reliability"), errors="coerce"))
        ),
        "confound_tags": confounds,
        "inference_permission": bool(
            report_permission
            or (role == "model_auxiliary" and branch in INFERENCE_AUXILIARY_BRANCHES)
        ),
        "report_permission": report_permission,
        "clinical_claim_permission": report_permission,
        "evidence_role": role,
    }


def _state_payload(row: Mapping[str, Any]) -> dict[str, Any]:
    model_state_z = float(
        np.nan_to_num(pd.to_numeric(row.get("state_z"), errors="coerce"))
    )
    report_permission = _truthy(row.get("report_permission", False))
    report_state_z = float(
        np.nan_to_num(
            pd.to_numeric(row.get("report_state_z", model_state_z), errors="coerce")
        )
    )
    report_confidence = _bounded(
        np.nan_to_num(
            pd.to_numeric(
                row.get("report_confidence", row.get("confidence")), errors="coerce"
            )
        )
    )
    supporting_metrics = [
        dict(item)
        for item in _json_list(row.get("supporting_metrics"))
        if isinstance(item, Mapping)
    ]
    counter_evidence = [
        dict(item)
        for item in _json_list(row.get("counter_evidence"))
        if isinstance(item, Mapping)
    ]
    branch = str(row.get("branch", ""))
    inference_permission = bool(
        report_permission or branch in INFERENCE_AUXILIARY_BRANCHES
    )
    return {
        "evidence_id": str(row.get("state_id", "unknown_state")),
        "state_id": str(row.get("state_id", "")),
        "state_base_id": str(row.get("state_base_id", row.get("state_id", ""))),
        "state_name_zh": str(row.get("state_name_zh", "")),
        "clinical_question": str(row.get("clinical_question", "")),
        "branch": branch,
        "task_scope": str(row.get("task_scope", "overall")),
        "state_z": report_state_z if report_permission else model_state_z,
        "model_state_z": model_state_z,
        "report_permission": report_permission,
        "inference_permission": inference_permission,
        "clinical_claim_permission": report_permission,
        "category": str(row.get("category", "unreliable")),
        "severity": float(
            np.nan_to_num(pd.to_numeric(row.get("severity"), errors="coerce"))
        ),
        "confidence": (
            report_confidence
            if report_permission
            else _bounded(
                np.nan_to_num(
                    pd.to_numeric(row.get("confidence"), errors="coerce")
                )
            )
        ),
        "missing_fraction": _bounded(
            np.nan_to_num(pd.to_numeric(row.get("missing_fraction"), errors="coerce"))
        ),
        "supporting_metrics": supporting_metrics,
        "counter_evidence": counter_evidence,
        "metric_evidence_ids": [
            str(item.get("metric_instance_id") or item.get("metric_id"))
            for item in supporting_metrics + counter_evidence
            if item.get("metric_instance_id") or item.get("metric_id")
        ],
        "evidence_segments": [
            _sanitize_segment(segment)
            for segment in _json_list(row.get("evidence_segments"))
            if isinstance(segment, Mapping)
        ],
    }


def build_case_workspace(
    *,
    subject_id: str,
    base_probabilities: Mapping[str, float],
    state_cards: pd.DataFrame,
    metric_evidence: pd.DataFrame,
    class_support: Mapping[str, float],
    max_supporting_evidence: int = 8,
) -> dict[str, Any]:
    """Build the label-free, auditable working memory used by the diagnostic Agent."""

    cards = state_cards[
        state_cards["subject_id"].astype(str).eq(str(subject_id))
    ].copy()
    evidence = metric_evidence[
        metric_evidence["subject_id"].astype(str).eq(str(subject_id))
    ].copy()

    all_state_items = [_state_payload(row) for row in cards.to_dict("records")]
    task_specific_states = {
        item["state_base_id"]
        for item in all_state_items
        if item["task_scope"] not in {"", "overall", "nan"}
        and item["confidence"] > 0.0
    }
    all_state_items = [
        item
        for item in all_state_items
        if not (
            item["task_scope"] == "overall"
            and item["state_base_id"] in task_specific_states
        )
    ]
    all_state_items.sort(
        key=lambda item: abs(item["state_z"])
        * item["confidence"]
        * (1.0 - item["missing_fraction"]),
        reverse=True,
    )
    reportable_state_items = [
        item
        for item in all_state_items
        if item["report_permission"] and item["confidence"] > 0.0
    ]
    inference_only_state_items = [
        item
        for item in all_state_items
        if item["inference_permission"]
        and not item["report_permission"]
        and item["confidence"] > 0.0
    ]
    state_items = reportable_state_items + inference_only_state_items
    model_only_state_items = [
        item for item in all_state_items if not item["inference_permission"]
    ]

    clinical: list[dict[str, Any]] = []
    quality: list[dict[str, Any]] = []
    for row in evidence.to_dict("records"):
        item = _metric_payload(row)
        role = item["evidence_role"]
        is_missing = _truthy(row.get("missing", False))
        if role in QUALITY_ROLES or item["branch"] == "qc" or item["state_id"] == "QC":
            quality.append(item)
        elif item["inference_permission"] and not is_missing and item["reliability"] > 0.0:
            clinical.append(item)

    task_specific_metric_keys = {
        (item["state_id"], item["metric_id"])
        for item in clinical
        if item["task_scope"] not in {"", "overall", "nan"}
    }
    clinical = [
        item
        for item in clinical
        if not (
            item["task_scope"] == "overall"
            and (item["state_id"], item["metric_id"]) in task_specific_metric_keys
        )
    ]

    clinical.sort(
        key=lambda item: abs(item["directional_z"]) * item["reliability"],
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    seen_states: set[str] = set()
    for item in clinical:
        state_key = f"{item['state_id']}::{item['task_scope']}"
        if state_key in seen_states and len(seen_states) < max_supporting_evidence:
            continue
        selected.append(item)
        seen_states.add(state_key)
        if len(selected) >= max_supporting_evidence:
            break
    if len(selected) < max_supporting_evidence:
        selected_ids = {item["evidence_id"] for item in selected}
        selected.extend(
            item
            for item in clinical
            if item["evidence_id"] not in selected_ids
        )
        selected = selected[:max_supporting_evidence]

    counter = [
        item
        for item in selected
        if item["directional_z"] < 0.0 and item["report_permission"]
    ]
    supportive = [
        item
        for item in selected
        if item["directional_z"] >= 0.0 and item["report_permission"]
    ]
    inference_only_metrics = [
        item for item in selected if not item["report_permission"]
    ]
    for state in all_state_items:
        state["evidence_id"] = _typed_evidence_id("state", state["evidence_id"])
        state["metric_evidence_ids"] = [
            _typed_evidence_id("metric", value)
            for value in state.get("metric_evidence_ids", [])
        ]
        for segment in state.get("evidence_segments", []):
            if segment.get("segment_id"):
                segment["segment_id"] = _typed_evidence_id(
                    "segment", segment["segment_id"]
                )
    for item in selected:
        item["evidence_id"] = _typed_evidence_id("metric", item["evidence_id"])
    for item in quality:
        item["evidence_id"] = _typed_evidence_id("qc", item["evidence_id"])
    coverage_components = structured_evidence_coverage(all_state_items, state_items)
    coverage = coverage_components["overall"]
    reliability = (
        float(np.mean([item["reliability"] for item in selected]))
        if selected
        else 0.0
    )
    confound_burden = (
        float(
            np.mean(
                [min(len(item["confound_tags"]) / 3.0, 1.0) for item in selected]
            )
        )
        if selected
        else 1.0
    )
    gate = evidence_gate(coverage, reliability, confound_burden)

    review_plan: list[dict[str, Any]] = [{"action": "inspect_quality"}]
    review_plan.extend(
        {
            "action": "inspect_state",
            "evidence_id": item["evidence_id"],
            "task_scope": item["task_scope"],
        }
        for item in state_items[: min(4, len(state_items))]
    )
    inspected_segments: set[str] = set()
    for item in state_items:
        for segment in item["evidence_segments"][:1]:
            segment_id = str(segment.get("segment_id", ""))
            if not segment_id or segment_id in inspected_segments:
                continue
            review_plan.append(
                {
                    "action": "inspect_segment",
                    "evidence_id": segment_id,
                    "state_id": item["state_id"],
                    "task_scope": item["task_scope"],
                }
            )
            inspected_segments.add(segment_id)
            if len(inspected_segments) >= 3:
                break
        if len(inspected_segments) >= 3:
            break
    task_scopes = sorted(
        {
            item["task_scope"]
            for item in state_items
            if item["task_scope"] not in {"", "overall", "nan"}
        }
    )
    if len(task_scopes) >= 2:
        review_plan.append(
            {
                "action": "compare_tasks",
                "task_scopes": task_scopes,
            }
        )
    review_plan.extend(
        {"action": "inspect_metric", "evidence_id": item["evidence_id"]}
        for item in selected
    )
    if counter:
        review_plan.append(
            {
                "action": "check_counterevidence",
                "evidence_id": counter[0]["evidence_id"],
            }
        )
    review_plan.extend([{"action": "update_hypothesis"}, {"action": "stop"}])

    languages = sorted(
        {
            str(value)
            for value in evidence.get("language", pd.Series(dtype=str)).dropna()
            if str(value).strip() and str(value).lower() != "nan"
        }
    )
    context_tasks = sorted(
        {
            str(value)
            for value in cards.get("task_scope", pd.Series(dtype=str)).dropna()
            if str(value).strip() and str(value).lower() != "nan"
        }
    )
    context_branches = sorted(
        {
            str(value)
            for value in cards.get("branch", pd.Series(dtype=str)).dropna()
            if str(value).strip() and str(value).lower() != "nan"
        }
    )

    return {
        "workspace_version": "9.2-v1",
        "case_id": _private_id(subject_id, "CASE"),
        "case_context": {
            "languages": languages,
            "task_scopes": context_tasks,
            "available_branches": context_branches,
        },
        "case_input_route": case_input_route(
            languages, context_tasks, context_branches
        ),
        "base_probabilities": {
            str(key): float(value) for key, value in base_probabilities.items()
        },
        "class_support": {
            str(key): float(value) for key, value in class_support.items()
        },
        "state_observations": state_items,
        "reportable_state_observations": reportable_state_items,
        "inference_only_state_observations": inference_only_state_items,
        "model_only_state_observations": model_only_state_items,
        "selected_supporting_evidence": supportive,
        "selected_counterevidence": counter,
        "inference_only_metric_observations": inference_only_metrics,
        "quality_observations": quality,
        "evidence_registry": _deduplicate_evidence_registry([
            {
                "evidence_id": item["evidence_id"],
                "evidence_type": "state",
            }
            for item in all_state_items
        ]
        + [
            {
                "evidence_id": item["evidence_id"],
                "evidence_type": "metric",
            }
            for item in selected
        ]
        + [
            {
                "evidence_id": item["evidence_id"],
                "evidence_type": "quality",
            }
            for item in quality
        ]
        + [
            {
                "evidence_id": segment["segment_id"],
                "evidence_type": "segment",
            }
            for state in all_state_items
            for segment in state.get("evidence_segments", [])
            if segment.get("segment_id")
        ]),
        "evidence_coverage": float(coverage),
        "evidence_coverage_components": coverage_components,
        "evidence_reliability": reliability,
        "confound_burden": confound_burden,
        "correction_gate": float(gate),
        "precomputed_review_plan": review_plan,
    }
