"""Task-conditioned state graph with correlation-family evidence budgets.

This module is intentionally separate from :mod:`advoice.states`.  The legacy
state-card builder remains available while callers opt into family-aware
aggregation and hierarchical shared/task components.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from scipy.special import expit

from .states import (
    _normalize_evidence_schema,
    _segment_evidence,
    _state_category,
)
from .utils import hash_values


IDENTITY_COLUMNS = ["dataset_id", "subject_id", "label", "split"]


@dataclass(frozen=True, slots=True)
class StateGraphV2:
    """Immutable handle for a rebuilt, family-aware state graph.

    The existing dataframe API remains the compatibility path.  This small
    wrapper gives replay consumers an explicit evidence/state binding without
    making pandas frames part of their public identity.
    """

    evidence_hash: str
    state_hash: str
    cards: pd.DataFrame
    wide: pd.DataFrame

    @classmethod
    def from_evidence_frame(
        cls,
        evidence: pd.DataFrame,
        states_config: dict[str, Any],
        correlation_config: dict[str, Any] | None = None,
    ) -> "StateGraphV2":
        cards, wide = build_state_graph_frame(evidence, states_config, correlation_config)
        evidence_hash = state_graph_evidence_hash(evidence)
        state_hash = hash_values([{
            "cards": _json_safe(cards.sort_index(axis=1).to_dict("records")),
            "wide": _json_safe(wide.sort_index(axis=1).to_dict("records")),
        }])
        return cls(evidence_hash=evidence_hash, state_hash=state_hash, cards=cards, wide=wide)


def _as_bool(series: pd.Series, default: bool = False) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.lower()
        .map({"true": True, "false": False, "1": True, "0": False,
              "1.0": True, "0.0": False})
        .fillna(default)
        .astype(bool)
    )


def _json_value(value: Any) -> Any:
    if value is None or value is pd.NA:
        return []
    if np.isscalar(value) and bool(pd.isna(value)):
        return []
    if isinstance(value, str):
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return [value] if value else []
    if isinstance(value, (list, tuple, set)):
        return list(value)
    if isinstance(value, dict):
        return value
    return [value]


def _json_safe(value: Any) -> Any:
    if value is None or value is pd.NA:
        return None
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return value


def state_graph_evidence_hash(evidence: pd.DataFrame) -> str:
    """Canonical identity for dataframe evidence passed to StateGraphV2."""

    records = _json_safe(evidence.sort_index(axis=1).to_dict("records"))
    return hash_values([sorted(
        records,
        key=lambda row: json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
    )])


def _json_dump(value: Any) -> str:
    return json.dumps(
        _json_safe(value),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _weighted_mean(values: Iterable[float], weights: Iterable[float]) -> float:
    pairs = sorted(
        (float(value), float(weight))
        for value, weight in zip(values, weights, strict=True)
        if np.isfinite(value) and np.isfinite(weight) and weight > 0.0
    )
    denominator = math.fsum(weight for _, weight in pairs)
    if denominator <= 0.0:
        return 0.0
    return math.fsum(value * weight for value, weight in pairs) / denominator


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.lexsort((weights, values))
    ordered_values = values[order]
    ordered_weights = weights[order]
    cutoff = float(ordered_weights.sum()) / 2.0
    return float(ordered_values[np.searchsorted(np.cumsum(ordered_weights), cutoff, side="left")])


def _huber_location(
    values: Iterable[float],
    weights: Iterable[float],
    *,
    delta: float,
    iterations: int = 25,
) -> float:
    """Return a deterministic weighted Huber location estimate."""

    pairs = sorted(
        (float(value), float(weight))
        for value, weight in zip(values, weights, strict=True)
        if np.isfinite(value) and np.isfinite(weight) and weight > 0.0
    )
    if not pairs:
        return 0.0
    array = np.asarray([value for value, _ in pairs], dtype=float)
    base_weights = np.asarray([weight for _, weight in pairs], dtype=float)
    location = _weighted_median(array, base_weights)
    delta = max(float(delta), 1e-6)
    for _ in range(iterations):
        residual = array - location
        robust_weights = np.minimum(1.0, delta / np.maximum(np.abs(residual), 1e-12))
        updated = _weighted_mean(array, base_weights * robust_weights)
        if abs(updated - location) <= 1e-12:
            break
        location = updated
    return float(location)


def _family_definitions(
    state_definition: dict[str, Any],
    correlation_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    metric_weights = {
        str(metric): float(weight)
        for metric, weight in zip(
            state_definition.get("metrics", []),
            state_definition.get("weights", []),
            strict=True,
        )
    }
    state_id = str(state_definition["id"])
    configured: Any = None
    if correlation_config:
        state_configs = correlation_config.get("states", {})
        if isinstance(state_configs, list):
            state_configs = {str(item["id"]): item for item in state_configs}
        configured = state_configs.get(state_id)
    raw_families = configured.get("families", []) if configured else []
    if isinstance(raw_families, dict):
        raw_families = [dict(value, id=key) for key, value in raw_families.items()]

    families: list[dict[str, Any]] = []
    assigned: set[str] = set()
    family_ids: set[str] = set()
    for raw in raw_families:
        family_id = str(raw.get("id", "family"))
        budget = float(raw.get("budget", 1.0))
        if family_id in family_ids:
            raise ValueError(f"{state_id} repeats correlation family id {family_id!r}")
        if not np.isfinite(budget) or budget <= 0.0:
            raise ValueError(f"{state_id} correlation family {family_id!r} needs a positive budget")
        metrics = sorted(str(metric) for metric in raw.get("metrics", []))
        unknown = set(metrics).difference(metric_weights)
        overlap = set(metrics).intersection(assigned)
        if unknown:
            raise ValueError(f"{state_id} correlation family contains unknown metrics: {sorted(unknown)}")
        if overlap:
            raise ValueError(f"{state_id} metrics occur in more than one family: {sorted(overlap)}")
        if not metrics:
            continue
        family_ids.add(family_id)
        assigned.update(metrics)
        families.append(
            {
                "id": family_id,
                "metrics": metrics,
                "budget": budget,
                "metric_weights": {metric: abs(metric_weights[metric]) for metric in metrics},
            }
        )
    for metric in sorted(set(metric_weights).difference(assigned)):
        families.append(
            {
                "id": f"metric:{metric}",
                "metrics": [metric],
                "budget": abs(metric_weights[metric]) or 1.0,
                "metric_weights": {metric: 1.0},
            }
        )
    return sorted(families, key=lambda item: item["id"])


def _metric_records(frame: pd.DataFrame, *, state_id: str = "") -> list[dict[str, Any]]:
    columns = [
        "evidence_id", "metric_id", "metric_instance_id", "state_id", "dataset_id",
        "subject_id", "case_id", "task_id", "task_scope", "value", "reference_label",
        "reference_median", "reference_scale", "cn_train_median", "robust_z",
        "directional_z", "reliability", "missing", "evidence_status",
        "report_permission", "confound_tags", "source_segment_ids", "segment_ids",
        "evidence_segments", "source_asset_id", "transcript_id", "method_version",
        "measurement_version", "generated_by", "provenance", "unavailable_reason",
    ]
    present = [column for column in columns if column in frame.columns]
    records = frame[present].copy().to_dict("records")
    for record in records:
        if not record.get("state_id") and state_id:
            record["state_id"] = state_id
        if not record.get("evidence_id"):
            record["evidence_id"] = str(
                record.get("metric_instance_id") or record.get("metric_id") or ""
            )
    records.sort(
        key=lambda item: (
            str(item.get("evidence_id", "")),
            str(item.get("metric_id", "")),
            str(item.get("metric_instance_id", "")),
            float(pd.to_numeric(pd.Series([item.get("directional_z")]), errors="coerce").fillna(0.0).iloc[0]),
        )
    )
    return _json_safe(records)


def _confound_trace(frame: pd.DataFrame) -> list[dict[str, Any]]:
    columns = [
        column
        for column in (
            "confound_tags", "potential_confounds", "observed_confounds", "ruled_out_confounds"
        )
        if column in frame.columns
    ]
    trace: list[dict[str, Any]] = []
    for row in frame.sort_values(["metric_id", "metric_instance_id"], kind="mergesort").to_dict("records"):
        payload = {column: _json_value(row.get(column)) for column in columns}
        if any(payload.values()):
            trace.append({"metric_id": str(row["metric_id"]), **payload})
    return trace


def _direct_segment_trace(frame: pd.DataFrame) -> list[Any]:
    trace: list[Any] = []
    for column in ("evidence_segments", "source_segment_ids", "segment_ids"):
        if column not in frame.columns:
            continue
        for value in frame[column]:
            parsed = _json_value(value)
            if isinstance(parsed, list):
                trace.extend(parsed)
            elif parsed:
                trace.append(parsed)
    unique = {_json_dump(item): item for item in trace}
    return [unique[key] for key in sorted(unique)]


def _aggregate_scope(
    state: pd.DataFrame,
    families: list[dict[str, Any]],
    *,
    contribution_limit: float,
    huber_delta: float,
) -> dict[str, Any]:
    family_rows: list[dict[str, Any]] = []
    total_budget = math.fsum(max(0.0, float(item["budget"])) for item in families)
    for family in families:
        points: list[dict[str, Any]] = []
        for metric in family["metrics"]:
            metric_rows = state[state["metric_id"].astype(str).eq(metric)].copy()
            if metric_rows.empty:
                continue
            missing = metric_rows["missing"].astype(bool)
            status_ok = ~metric_rows.get("evidence_status", "available").astype(str).isin(
                {"missing", "unavailable", "unobservable"}
            )
            values = pd.to_numeric(metric_rows["directional_z"], errors="coerce")
            reliability = pd.to_numeric(metric_rows["reliability"], errors="coerce").fillna(0.0)
            usable = (~missing) & status_ok & values.notna() & np.isfinite(values) & reliability.gt(0.0)
            if not usable.any():
                continue
            metric_values = values[usable].clip(-abs(contribution_limit), abs(contribution_limit))
            metric_reliability = float(np.median(reliability[usable].clip(0.0, 1.0)))
            metric_score = _huber_location(
                metric_values,
                reliability[usable].clip(lower=0.0),
                delta=huber_delta,
            )
            points.append(
                {
                    "metric_id": metric,
                    "score": metric_score,
                    "reliability": metric_reliability,
                    "weight": float(family["metric_weights"].get(metric, 1.0)),
                }
            )
        configured_weight = math.fsum(float(value) for value in family["metric_weights"].values())
        available_weight = math.fsum(point["weight"] for point in points)
        coverage = available_weight / configured_weight if configured_weight > 0.0 else 0.0
        family_score = _huber_location(
            [point["score"] for point in points],
            [point["weight"] * point["reliability"] for point in points],
            delta=huber_delta,
        )
        family_reliability = _weighted_mean(
            [point["reliability"] for point in points],
            [point["weight"] for point in points],
        )
        effective_budget = float(family["budget"]) * family_reliability * coverage
        family_rows.append(
            {
                "family_id": family["id"],
                "budget": float(family["budget"]),
                "effective_budget": effective_budget,
                "score": family_score,
                "reliability": family_reliability,
                "coverage": coverage,
                "available_metrics": [point["metric_id"] for point in points],
            }
        )
    effective_total = math.fsum(row["effective_budget"] for row in family_rows)
    score = _weighted_mean(
        [row["score"] for row in family_rows],
        [row["effective_budget"] for row in family_rows],
    )
    confidence = effective_total / total_budget if total_budget > 0.0 else 0.0
    coverage = (
        math.fsum(row["budget"] * row["coverage"] for row in family_rows) / total_budget
        if total_budget > 0.0
        else 0.0
    )
    available_families = sum(row["effective_budget"] > 0.0 for row in family_rows)
    return {
        "state_z": score,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
        "missing_fraction": float(np.clip(1.0 - coverage, 0.0, 1.0)),
        "available": bool(effective_total > 0.0),
        "independent_family_count": available_families,
        "family_scores": family_rows,
    }


def _evidence_lists(
    frame: pd.DataFrame, *, state_id: str = ""
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    status_ok = ~frame["evidence_status"].astype(str).isin(
        {"missing", "unavailable", "unobservable"}
    )
    reportable = frame[
        frame["report_permission"].astype(bool)
        & ~frame["missing"].astype(bool)
        & status_ok
        & pd.to_numeric(frame["reliability"], errors="coerce").fillna(0.0).gt(0.0)
        & pd.to_numeric(frame["directional_z"], errors="coerce").notna()
    ].copy()
    reportable["evidence_strength"] = (
        pd.to_numeric(reportable["directional_z"], errors="coerce").abs()
        * pd.to_numeric(reportable["reliability"], errors="coerce").fillna(0.0)
    )
    reportable = reportable.sort_values(
        ["evidence_strength", "metric_id", "metric_instance_id"],
        ascending=[False, True, True],
        kind="mergesort",
    )
    support = reportable[pd.to_numeric(reportable["directional_z"], errors="coerce").ge(0.0)]
    counter = reportable[pd.to_numeric(reportable["directional_z"], errors="coerce").lt(0.0)]
    return (
        _metric_records(support.head(3), state_id=state_id),
        _metric_records(counter.head(2), state_id=state_id),
    )


def _scope_payload(
    state: pd.DataFrame,
    aggregate: dict[str, Any],
    *,
    subject_id: str,
    state_id: str,
    task_scope: str,
    recordings: pd.DataFrame,
    segments: pd.DataFrame,
    segment_lookup: dict[tuple[str, str], pd.DataFrame],
) -> dict[str, Any]:
    support, counter = _evidence_lists(state, state_id=state_id)
    segment_trace = _direct_segment_trace(state)
    if not segment_trace:
        segment_trace = _segment_evidence(
            subject_id,
            state_id,
            recordings,
            segments,
            task_scope=task_scope,
            segment_lookup=segment_lookup,
        )
    metric_trace = _metric_records(state, state_id=state_id)
    supporting_ids = sorted({
        str(item["evidence_id"])
        for item in support
        if item.get("evidence_id")
    })
    counter_ids = sorted({
        str(item["evidence_id"])
        for item in counter
        if item.get("evidence_id")
    })
    provenance_trace = [
        {
            "evidence_id": item.get("evidence_id", ""),
            "state_id": item.get("state_id", state_id),
            "dataset_id": item.get("dataset_id", ""),
            "subject_id": item.get("subject_id", subject_id),
            "case_id": item.get("case_id", ""),
            "task_id": item.get("task_id"),
            "task_scope": item.get("task_scope", task_scope),
            "segment_ids": _json_value(
                item.get("source_segment_ids", item.get("segment_ids", []))
            ),
            "source_asset_id": item.get("source_asset_id", ""),
            "transcript_id": item.get("transcript_id"),
            "method_version": item.get("method_version", ""),
            "measurement_version": item.get("measurement_version", ""),
            "generated_by": item.get("generated_by", ""),
        }
        for item in metric_trace
    ]
    case_ids = sorted({
        str(item.get("case_id"))
        for item in metric_trace
        if item.get("case_id") not in (None, "", "nan")
    })
    task_ids = sorted({
        str(item.get("task_id") or item.get("task_scope") or "overall")
        for item in metric_trace
    })
    return {
        **aggregate,
        "supporting_metrics": support,
        "counter_evidence": counter,
        "confounds": _confound_trace(state),
        "evidence_segments": segment_trace,
        "metric_trace": metric_trace,
        "supporting_evidence_ids": supporting_ids,
        "counter_evidence_ids": counter_ids,
        "case_ids": case_ids,
        "task_ids": task_ids,
        "provenance_trace": provenance_trace,
        "provenance": provenance_trace,
    }


def _combined_payload(payloads: list[dict[str, Any]]) -> dict[str, Any]:
    combined: dict[str, Any] = {}
    for key in (
        "supporting_metrics", "counter_evidence", "confounds", "evidence_segments",
        "metric_trace", "supporting_evidence_ids", "counter_evidence_ids", "case_ids",
        "task_ids", "provenance_trace",
        "provenance",
    ):
        values = [item for payload in payloads for item in payload[key]]
        unique = {_json_dump(item): item for item in values}
        combined[key] = [unique[token] for token in sorted(unique)]
    combined["supporting_metrics"] = sorted(
        combined["supporting_metrics"],
        key=lambda item: (-abs(float(item.get("directional_z", 0.0))), str(item.get("metric_id", ""))),
    )[:3]
    combined["counter_evidence"] = sorted(
        combined["counter_evidence"],
        key=lambda item: (-abs(float(item.get("directional_z", 0.0))), str(item.get("metric_id", ""))),
    )[:2]
    combined["supporting_evidence_ids"] = sorted({
        str(item.get("evidence_id"))
        for item in combined["supporting_metrics"]
        if item.get("evidence_id")
    })
    combined["counter_evidence_ids"] = sorted({
        str(item.get("evidence_id"))
        for item in combined["counter_evidence"]
        if item.get("evidence_id")
    })
    combined["provenance"] = combined["provenance_trace"]
    return combined


def _card(
    *,
    identity: dict[str, Any],
    definition: dict[str, Any],
    state_id: str,
    task_scope: str,
    graph_level: str,
    state_z: float,
    confidence: float,
    missing_fraction: float,
    available: bool,
    payload: dict[str, Any],
    contribution_limit: float,
    unavailable_reason: str = "",
    task_state_z: float = np.nan,
    residual_shrinkage_factor: float = np.nan,
    independent_family_count: int = 0,
    family_scores: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    report_permission = bool(
        available
        and (payload["supporting_metrics"] or payload["counter_evidence"])
    )
    finite_z = float(state_z) if available and np.isfinite(state_z) else np.nan
    return {
        **identity,
        "state_id": state_id,
        "state_base_id": str(definition["id"]),
        "task_scope": task_scope,
        "graph_level": graph_level,
        "state_view": graph_level,
        "state_name_zh": definition.get("name_zh", definition["id"]),
        "branch": definition.get("branch", "unassigned"),
        "clinical_question": definition.get("clinical_question", ""),
        "state_z": finite_z,
        "raw_state_z": finite_z,
        "task_state_z": task_state_z,
        "report_state_z": finite_z if report_permission else np.nan,
        "severity": float(expit(finite_z)) if np.isfinite(finite_z) else np.nan,
        "category": _state_category(finite_z, confidence, missing_fraction) if np.isfinite(finite_z) else "unavailable",
        "confidence": float(confidence),
        "report_confidence": float(confidence) if report_permission else 0.0,
        "report_permission": report_permission,
        "missing_fraction": float(missing_fraction),
        "available": bool(available),
        "evidence_status": "available" if available else "unavailable",
        "unavailable_reason": unavailable_reason,
        "independent_family_count": int(independent_family_count),
        "metric_contribution_clip_z": contribution_limit,
        "residual_shrinkage_factor": residual_shrinkage_factor,
        "supporting_metrics": _json_dump(payload["supporting_metrics"]),
        "counter_evidence": _json_dump(payload["counter_evidence"]),
        "confounds": _json_dump(payload["confounds"]),
        "evidence_segments": _json_dump(payload["evidence_segments"]),
        "metric_trace": _json_dump(payload["metric_trace"]),
        "evidence_id": state_id,
        "supporting_evidence_ids": _json_dump(payload.get("supporting_evidence_ids", [])),
        "counter_evidence_ids": _json_dump(payload.get("counter_evidence_ids", [])),
        "metric_evidence_ids": _json_dump(sorted({
            *payload.get("supporting_evidence_ids", []),
            *payload.get("counter_evidence_ids", []),
        })),
        "case_ids": _json_dump(payload.get("case_ids", [])),
        "case_id": payload.get("case_ids", [""])[0] if len(payload.get("case_ids", [])) == 1 else "",
        "task_ids": _json_dump(payload.get("task_ids", [])),
        "provenance_trace": _json_dump(payload.get("provenance_trace", [])),
        "provenance": _json_dump(payload.get("provenance", payload.get("provenance_trace", []))),
        "segment_ids": _json_dump(sorted({
            str(segment_id)
            for item in payload.get("provenance_trace", [])
            for segment_id in item.get("segment_ids", [])
        })),
        "family_scores": _json_dump(family_scores or []),
        "trace_resolution": "task_and_segment" if task_scope != "shared" and payload["evidence_segments"] else "segment" if payload["evidence_segments"] else "task_and_metric" if task_scope != "shared" else "metric",
        "evidence_budget_id": f"{identity['dataset_id']}:{identity['subject_id']}:{definition['id']}",
    }


def build_state_graph_frame(
    evidence: pd.DataFrame,
    states_config: dict[str, Any],
    correlation_config: dict[str, Any] | None = None,
    *,
    correlation_family_config: dict[str, Any] | None = None,
    recordings: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build shared states and task residuals without duplicate family votes.

    Old state configuration files remain valid: metrics omitted from the family
    registry become singleton families with their legacy state weights.
    """

    if correlation_config is not None and correlation_family_config is not None:
        raise ValueError(
            "Supply correlation_config or correlation_family_config, not both."
        )
    correlation_config = correlation_config or correlation_family_config
    frame = _normalize_evidence_schema(evidence)
    for column, default in (("dataset_id", "unknown"), ("label", "unknown"), ("split", "unknown")):
        if column not in frame:
            frame[column] = default
    if "subject_id" not in frame or "metric_id" not in frame:
        raise ValueError("StateGraph evidence requires subject_id and metric_id columns.")
    if "directional_z" not in frame:
        raise ValueError("StateGraph evidence requires directional_z values.")
    if "reliability" not in frame:
        frame["reliability"] = 0.0
    if "evidence_status" not in frame:
        frame["evidence_status"] = np.where(frame["missing"].astype(bool), "missing", "available")
    frame["missing"] = _as_bool(frame["missing"], default=True)
    frame["report_permission"] = _as_bool(frame["report_permission"], default=False)
    frame["subject_id"] = frame["subject_id"].astype(str)
    frame["task_scope"] = frame["task_scope"].fillna("overall").astype(str)
    frame["metric_id"] = frame["metric_id"].astype(str)
    frame["metric_instance_id"] = frame["metric_instance_id"].astype(str)
    frame = frame.sort_values(
        IDENTITY_COLUMNS + ["metric_id", "metric_instance_id", "task_scope"],
        kind="mergesort",
    ).reset_index(drop=True)

    if (recordings is None) != (segments is None):
        raise ValueError("recordings and segments must be supplied together")
    recordings = recordings.copy() if recordings is not None else pd.DataFrame(columns=["case_id", "subject_id", "task_type"])
    segments = segments.copy() if segments is not None else pd.DataFrame(columns=["case_id"])
    segment_lookup: dict[tuple[str, str], pd.DataFrame] = {}
    if not recordings.empty and not segments.empty:
        recording_index = recordings[["case_id", "subject_id", "task_type"]].copy()
        recording_index["case_id"] = recording_index["case_id"].astype(str)
        recording_index["task_scope"] = recording_index["task_type"].fillna("").map(
            lambda value: re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
        )
        indexed = segments.copy()
        indexed["case_id"] = indexed["case_id"].astype(str)
        indexed = indexed.merge(recording_index, on="case_id", how="inner", validate="many_to_one")
        for subject_id, group in indexed.groupby("subject_id", sort=True):
            segment_lookup[(str(subject_id), "overall")] = group
        for (subject_id, task_scope), group in indexed.groupby(["subject_id", "task_scope"], sort=True):
            segment_lookup[(str(subject_id), str(task_scope))] = group

    defaults = (correlation_config or {}).get("defaults", {})
    contribution_limit = float(
        (correlation_config or {}).get(
            "metric_contribution_clip_z", states_config.get("metric_contribution_clip_z", 5.0)
        )
    )
    huber_delta = float(defaults.get("robust_huber_delta", 1.5))
    residual_prior_precision = max(float(defaults.get("task_residual_prior_precision", 1.0)), 0.0)
    min_tasks = max(int(defaults.get("min_tasks_for_residual", 2)), 2)
    rows: list[dict[str, Any]] = []

    for definition in sorted(states_config.get("states", []), key=lambda item: str(item["id"])):
        metrics = {str(metric) for metric in definition.get("metrics", [])}
        if not metrics:
            continue
        families = _family_definitions(definition, correlation_config)
        state_frame = frame[frame["metric_id"].isin(metrics)]
        grouping = IDENTITY_COLUMNS
        for keys, subject_state in state_frame.groupby(grouping, sort=True, dropna=False):
            identity = dict(zip(grouping, keys, strict=True))
            scopes: dict[str, dict[str, Any]] = {}
            for task_scope, scoped in subject_state.groupby("task_scope", sort=True):
                aggregate = _aggregate_scope(
                    scoped,
                    families,
                    contribution_limit=contribution_limit,
                    huber_delta=huber_delta,
                )
                scopes[str(task_scope)] = _scope_payload(
                    scoped,
                    aggregate,
                    subject_id=str(identity["subject_id"]),
                    state_id=str(definition["id"]),
                    task_scope=str(task_scope),
                    recordings=recordings,
                    segments=segments,
                    segment_lookup=segment_lookup,
                )

            named_tasks = sorted(scope for scope in scopes if scope != "overall")
            hierarchy_scopes = named_tasks if named_tasks else (["overall"] if "overall" in scopes else [])
            available_scopes = [scope for scope in hierarchy_scopes if scopes[scope]["available"]]
            selected_payloads = [scopes[scope] for scope in hierarchy_scopes]
            combined = _combined_payload(selected_payloads) if selected_payloads else {
                "supporting_metrics": [], "counter_evidence": [], "confounds": [],
                "evidence_segments": [], "metric_trace": [],
            }
            task_weights = [
                scopes[scope]["confidence"] * max(scopes[scope]["independent_family_count"], 1)
                for scope in available_scopes
            ]
            shared_available = bool(available_scopes)
            shared_z = _huber_location(
                [scopes[scope]["state_z"] for scope in available_scopes],
                task_weights,
                delta=huber_delta,
            ) if shared_available else np.nan
            shared_confidence = _weighted_mean(
                [scopes[scope]["confidence"] for scope in available_scopes],
                task_weights,
            ) if shared_available else 0.0
            shared_missing = _weighted_mean(
                [scopes[scope]["missing_fraction"] for scope in hierarchy_scopes],
                [1.0] * len(hierarchy_scopes),
            ) if hierarchy_scopes else 1.0
            family_trace = [
                {"task_scope": scope, **family}
                for scope in hierarchy_scopes
                for family in scopes[scope]["family_scores"]
            ]
            rows.append(
                _card(
                    identity=identity,
                    definition=definition,
                    state_id=str(definition["id"]),
                    task_scope="shared",
                    graph_level="shared",
                    state_z=shared_z,
                    confidence=shared_confidence,
                    missing_fraction=shared_missing,
                    available=shared_available,
                    payload=combined,
                    contribution_limit=contribution_limit,
                    unavailable_reason="" if shared_available else "no_observable_family_evidence",
                    independent_family_count=max(
                        (scopes[scope]["independent_family_count"] for scope in available_scopes),
                        default=0,
                    ),
                    family_scores=family_trace,
                )
            )

            if not named_tasks:
                continue
            residuals: dict[str, float] = {}
            shrinkage: dict[str, float] = {}
            if len(available_scopes) >= min_tasks:
                for scope, precision in zip(available_scopes, task_weights, strict=True):
                    alpha = precision / (precision + residual_prior_precision) if precision > 0.0 else 0.0
                    shrinkage[scope] = alpha
                    residuals[scope] = alpha * (scopes[scope]["state_z"] - float(shared_z))
                center = _weighted_mean(
                    [residuals[scope] for scope in available_scopes],
                    task_weights,
                )
                residuals = {scope: value - center for scope, value in residuals.items()}
            for scope in named_tasks:
                source = scopes[scope]
                residual_available = scope in residuals
                if residual_available:
                    reason = ""
                elif not source["available"]:
                    reason = "task_state_unavailable"
                else:
                    reason = "single_task_residual_not_identifiable"
                rows.append(
                    _card(
                        identity=identity,
                        definition=definition,
                        state_id=f"{definition['id']}__task_{scope}_residual",
                        task_scope=scope,
                        graph_level="task_residual",
                        state_z=residuals.get(scope, np.nan),
                        confidence=source["confidence"] * shrinkage.get(scope, 0.0),
                        missing_fraction=source["missing_fraction"],
                        available=residual_available,
                        payload=source,
                        contribution_limit=contribution_limit,
                        unavailable_reason=reason,
                        task_state_z=source["state_z"] if source["available"] else np.nan,
                        residual_shrinkage_factor=shrinkage.get(scope, np.nan),
                        independent_family_count=source["independent_family_count"],
                        family_scores=source["family_scores"],
                    )
                )

    cards = pd.DataFrame(rows)
    if cards.empty:
        return cards, pd.DataFrame(columns=IDENTITY_COLUMNS)
    cards = cards.sort_values(IDENTITY_COLUMNS + ["state_base_id", "graph_level", "task_scope"], kind="mergesort").reset_index(drop=True)
    score = cards.pivot(index=IDENTITY_COLUMNS, columns="state_id", values="state_z").add_prefix("state_")
    confidence = cards.pivot(index=IDENTITY_COLUMNS, columns="state_id", values="confidence").add_prefix("rel_")
    availability = cards.pivot(index=IDENTITY_COLUMNS, columns="state_id", values="available").add_prefix("available_")
    wide = score.join(confidence).join(availability).reset_index()
    return cards, wide


def build_state_graph(
    evidence_path: Path,
    recording_features_path: Path,
    segments_path: Path,
    states_config: dict[str, Any],
    state_cards_path: Path,
    state_wide_path: Path,
    *,
    correlation_config: dict[str, Any] | None = None,
    correlation_family_config: dict[str, Any] | None = None,
) -> None:
    """Path-based counterpart to :func:`build_state_graph_frame`."""

    cards, wide = build_state_graph_frame(
        pd.read_csv(evidence_path, dtype={"subject_id": str}),
        states_config,
        correlation_config,
        correlation_family_config=correlation_family_config,
        recordings=pd.read_csv(recording_features_path, dtype={"subject_id": str}),
        segments=pd.read_csv(segments_path),
    )
    cards.to_csv(state_cards_path, index=False)
    wide.to_csv(state_wide_path, index=False)


build_state_graph_cards_frame = build_state_graph_frame
