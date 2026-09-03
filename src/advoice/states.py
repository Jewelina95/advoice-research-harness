from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.special import expit

from .evidence import LANGUAGE_DEPENDENT_METRICS, MIN_LANGUAGE_REFERENCE_SUBJECTS


def _state_category(state_z: float, reliability: float, missing_fraction: float) -> str:
    if reliability < 0.45 or missing_fraction > 0.50:
        return "unreliable"
    if state_z >= 2.0:
        return "impaired"
    if state_z >= 1.0:
        return "borderline"
    return "normal"


def _robust_reference(values: pd.Series) -> tuple[float, float, bool]:
    finite = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if finite.empty:
        return 0.0, 1.0, False
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    scale = max(1.4826 * mad, float(finite.std(ddof=0)) * 0.25)
    tolerance = max(1e-8, abs(median) * 1e-8)
    available = bool(finite.nunique(dropna=True) >= 2 and scale > tolerance)
    return median, scale if available else 1.0, available


def _bounded_metric_contribution(values: pd.Series, limit: float) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce").fillna(0.0)
    return numeric.clip(lower=-abs(limit), upper=abs(limit))


def _normalize_evidence_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Keep cached pre-task evidence readable while preserving explicit unavailable states."""
    normalized = frame.copy()
    defaults: dict[str, Any] = {
        "task_scope": "overall",
        "reference_label": "unknown",
        "reference_median": np.nan,
        "reference_scale": np.nan,
        "cn_train_median": np.nan,
        "report_permission": False,
        "confound_tags": "[]",
    }
    for column, default in defaults.items():
        if column not in normalized.columns:
            normalized[column] = default
    if "metric_instance_id" not in normalized.columns:
        normalized["metric_instance_id"] = normalized["metric_id"]
    return normalized


def build_fold_calibrated_state_frame(
    evidence: pd.DataFrame,
    states_config: dict[str, Any],
    reference_subject_ids: set[str],
    reference_label: str,
) -> pd.DataFrame:
    """Rebuild state values using only the current training fold's reference subjects."""
    frame = _normalize_evidence_schema(evidence)
    frame["subject_id"] = frame["subject_id"].astype(str)
    reference = frame[
        frame["subject_id"].isin({str(value) for value in reference_subject_ids})
        & frame["label"].astype(str).eq(str(reference_label))
    ]
    if reference.empty:
        raise ValueError(
            f"No {reference_label!r} reference subjects are available in the current training fold."
        )
    calibrated_parts: list[pd.DataFrame] = []
    frame["language"] = frame.get("language", "unknown")
    frame["language"] = frame["language"].fillna("unknown").astype(str)
    reference["language"] = reference.get("language", "unknown")
    reference["language"] = reference["language"].fillna("unknown").astype(str)
    for metric_instance, metric_rows in frame.groupby("metric_instance_id", sort=False):
        metric_id = str(metric_rows["metric_id"].iloc[0])
        groups = (
            metric_rows.groupby("language", sort=False)
            if metric_id in LANGUAGE_DEPENDENT_METRICS
            else [("pooled", metric_rows)]
        )
        for language, scoped_rows in groups:
            reference_mask = reference["metric_instance_id"].eq(metric_instance)
            if metric_id in LANGUAGE_DEPENDENT_METRICS:
                reference_mask &= reference["language"].eq(str(language))
            reference_values = reference.loc[reference_mask, "value"]
            median, scale, variable = _robust_reference(reference_values)
            finite_count = int(
                pd.to_numeric(reference_values, errors="coerce")
                .replace([np.inf, -np.inf], np.nan)
                .notna()
                .sum()
            )
            available = bool(
                variable
                and (
                    metric_id not in LANGUAGE_DEPENDENT_METRICS
                    or finite_count >= MIN_LANGUAGE_REFERENCE_SUBJECTS
                )
            )
            part = scoped_rows.copy()
            values = pd.to_numeric(part["value"], errors="coerce")
            missing = values.isna() | (not available)
            part["fold_directional_z"] = np.where(
                missing,
                0.0,
                pd.to_numeric(part["direction"], errors="coerce").fillna(0.0)
                * (values - median)
                / scale,
            )
            part["fold_missing"] = missing
            calibrated_parts.append(part)
    calibrated = pd.concat(calibrated_parts, ignore_index=True)
    identity = ["dataset_id", "subject_id", "label", "split"]
    mapping_rows = [
        {
            "metric_id": metric_id,
            "state_base_id": definition["id"],
            "configured_weight": float(weight),
        }
        for definition in states_config["states"]
        for metric_id, weight in zip(
            definition["metrics"], definition["weights"], strict=True
        )
    ]
    state_evidence = calibrated.merge(
        pd.DataFrame(mapping_rows),
        on="metric_id",
        how="inner",
        validate="many_to_many",
    )
    state_evidence["state_id"] = np.where(
        state_evidence["task_scope"].eq("overall"),
        state_evidence["state_base_id"],
        state_evidence["state_base_id"]
        + "__task_"
        + state_evidence["task_scope"].astype(str),
    )
    state_evidence["effective_weight"] = (
        state_evidence["configured_weight"]
        * pd.to_numeric(state_evidence["reliability"], errors="coerce").fillna(0.0)
        * (~state_evidence["fold_missing"]).astype(float)
    )
    contribution_limit = float(states_config.get("metric_contribution_clip_z", 5.0))
    state_evidence["bounded_fold_directional_z"] = _bounded_metric_contribution(
        state_evidence["fold_directional_z"], contribution_limit
    )
    state_evidence["weighted_directional_z"] = (
        state_evidence["bounded_fold_directional_z"]
        * state_evidence["effective_weight"]
    )
    rows = (
        state_evidence.groupby(identity + ["state_id"], sort=False, as_index=False)
        .agg(
            weighted_sum=("weighted_directional_z", "sum"),
            effective_weight=("effective_weight", "sum"),
            configured_weight=("configured_weight", "sum"),
        )
    )
    rows["state_z"] = np.divide(
        rows["weighted_sum"],
        rows["effective_weight"],
        out=np.zeros(len(rows), dtype=float),
        where=rows["effective_weight"].to_numpy() > 0,
    )
    rows["reliability"] = np.divide(
        rows["effective_weight"],
        rows["configured_weight"],
        out=np.zeros(len(rows), dtype=float),
        where=rows["configured_weight"].to_numpy() > 0,
    )
    score = rows.pivot(index=identity, columns="state_id", values="state_z").add_prefix(
        "state_"
    )
    confidence = rows.pivot(
        index=identity, columns="state_id", values="reliability"
    ).add_prefix("rel_")
    return score.join(confidence).reset_index()


def _segment_evidence(
    subject_id: str,
    state_id: str,
    recordings: pd.DataFrame,
    segments: pd.DataFrame,
    task_scope: str = "overall",
    segment_lookup: dict[tuple[str, str], pd.DataFrame] | None = None,
) -> list[dict[str, Any]]:
    if segment_lookup is not None:
        available = segment_lookup.get((str(subject_id), task_scope), pd.DataFrame()).copy()
    else:
        subject_recordings = recordings[
            recordings["subject_id"].astype(str).eq(str(subject_id))
        ].copy()
        if task_scope != "overall":
            task_slug = subject_recordings["task_type"].fillna("").map(
                lambda value: re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
            )
            subject_recordings = subject_recordings[task_slug.eq(task_scope)]
        case_ids = subject_recordings["case_id"]
        available = segments[segments["case_id"].isin(case_ids)].copy()
    if available.empty or state_id not in {"S01", "S02", "S03"}:
        return []
    if "voiced_fraction" not in available:
        available["voiced_fraction"] = 1.0 - pd.to_numeric(
            available["silence_fraction"], errors="coerce"
        )
    if state_id == "S01":
        chosen = available.nlargest(2, "silence_fraction")
        selection_basis = "highest_silence_fraction"
    elif state_id == "S02":
        chosen = available.nsmallest(2, "voiced_fraction")
        selection_basis = "lowest_voiced_fraction"
    elif {
        "activity_transition_rate_hz",
        "voiced_run_mean_sec",
    }.issubset(available.columns):
        transition = pd.to_numeric(
            available["activity_transition_rate_hz"], errors="coerce"
        ).fillna(0.0)
        run_length = pd.to_numeric(
            available["voiced_run_mean_sec"], errors="coerce"
        ).fillna(0.0)
        available["continuity_burden"] = transition / np.maximum(run_length, 0.05)
        chosen = available.nlargest(2, "continuity_burden")
        selection_basis = "high_activity_switching_and_short_voiced_runs"
    else:
        # Loudness is device-sensitive and is not a valid proxy for speech continuity.
        return []
    columns = [
        "segment_id",
        "case_id",
        "start_sec",
        "end_sec",
        "silence_fraction",
        "voiced_fraction",
        "rms_db_mean",
    ]
    columns.extend(
        column
        for column in [
            "activity_transition_rate_hz",
            "voiced_run_mean_sec",
            "continuity_burden",
        ]
        if column in chosen.columns
    )
    if "source_spans" in chosen.columns:
        columns.append("source_spans")
    rows = chosen[columns].to_dict("records")
    for row in rows:
        row["selection_basis"] = selection_basis
    return rows


def build_state_cards(
    evidence_path: Path,
    recording_features_path: Path,
    segments_path: Path,
    states_config: dict[str, Any],
    state_cards_path: Path,
    state_wide_path: Path,
) -> None:
    evidence = _normalize_evidence_schema(pd.read_csv(evidence_path, dtype={"subject_id": str}))
    recordings = pd.read_csv(recording_features_path, dtype={"subject_id": str})
    segments = pd.read_csv(segments_path)
    recording_index = recordings[["case_id", "subject_id", "task_type"]].copy()
    recording_index["case_id"] = recording_index["case_id"].astype(str)
    recording_index["task_scope"] = recording_index["task_type"].fillna("").map(
        lambda value: re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")
    )
    indexed_segments = segments.copy()
    indexed_segments["case_id"] = indexed_segments["case_id"].astype(str)
    indexed_segments = indexed_segments.merge(
        recording_index[["case_id", "subject_id", "task_scope"]],
        on="case_id",
        how="inner",
        validate="many_to_one",
    )
    segment_lookup = {
        (str(subject_id), "overall"): group
        for subject_id, group in indexed_segments.groupby("subject_id", sort=False)
    }
    segment_lookup.update(
        {
            (str(subject_id), str(task_scope)): group
            for (subject_id, task_scope), group in indexed_segments.groupby(
                ["subject_id", "task_scope"], sort=False
            )
        }
    )
    state_mapping = pd.DataFrame(
        [
            {
                "metric_id": metric_id,
                "state_base_id": definition["id"],
                "state_name_zh": definition["name_zh"],
                "state_branch": definition["branch"],
                "clinical_question": definition["clinical_question"],
                "configured_weight": float(weight),
            }
            for definition in states_config["states"]
            for metric_id, weight in zip(
                definition["metrics"], definition["weights"], strict=True
            )
        ]
    )
    state_evidence = evidence.merge(
        state_mapping,
        on="metric_id",
        how="inner",
        validate="many_to_many",
    )
    state_evidence["effective_weight"] = (
        state_evidence["configured_weight"]
        * pd.to_numeric(state_evidence["reliability"], errors="coerce").fillna(0.0)
        * (~state_evidence["missing"].astype(bool)).astype(float)
    )
    contribution_limit = float(states_config.get("metric_contribution_clip_z", 5.0))
    state_evidence["bounded_directional_z"] = _bounded_metric_contribution(
        state_evidence["directional_z"], contribution_limit
    )
    rows: list[dict[str, Any]] = []
    grouping = [
        "dataset_id",
        "subject_id",
        "label",
        "split",
        "state_base_id",
        "state_name_zh",
        "state_branch",
        "clinical_question",
        "task_scope",
    ]
    for keys, state in state_evidence.groupby(grouping, sort=False):
        (
            dataset_id,
            subject_id,
            label,
            split,
            state_base_id,
            state_name_zh,
            state_branch,
            clinical_question,
            task_scope,
        ) = keys
        state_id = (
            state_base_id
            if task_scope == "overall"
            else f"{state_base_id}__task_{task_scope}"
        )
        denominator = float(state["effective_weight"].sum())
        if denominator:
            raw_state_z = float(
                (state["directional_z"] * state["effective_weight"]).sum()
                / denominator
            )
            state_z = float(
                (state["bounded_directional_z"] * state["effective_weight"]).sum()
                / denominator
            )
            reliability = denominator / max(float(state["configured_weight"].sum()), 1e-9)
        else:
            raw_state_z, state_z, reliability = 0.0, 0.0, 0.0
        missing_fraction = float(state["missing"].mean()) if len(state) else 1.0
        reportable = state[
            state["report_permission"].astype(bool)
            & ~state["missing"].astype(bool)
        ].copy()
        report_denominator = float(reportable["effective_weight"].sum())
        if report_denominator:
            report_state_z = float(
                (
                    reportable["bounded_directional_z"]
                    * reportable["effective_weight"]
                ).sum()
                / report_denominator
            )
            report_confidence = report_denominator / max(
                float(reportable["configured_weight"].sum()), 1e-9
            )
        else:
            report_state_z, report_confidence = 0.0, 0.0
        reportable["evidence_strength"] = (
            pd.to_numeric(reportable["directional_z"], errors="coerce").abs()
            * pd.to_numeric(reportable["reliability"], errors="coerce").fillna(0.0)
        )
        support = reportable[
            pd.to_numeric(reportable["directional_z"], errors="coerce").ge(0.0)
        ].nlargest(3, "evidence_strength")[
            [
                "metric_id",
                "metric_instance_id",
                "task_scope",
                "value",
                "reference_label",
                "reference_median",
                "reference_scale",
                "cn_train_median",
                "robust_z",
                "directional_z",
                "reliability",
                "report_permission",
                "confound_tags",
            ]
        ].to_dict("records")
        counter = reportable[
            pd.to_numeric(reportable["directional_z"], errors="coerce").lt(0.0)
        ].nlargest(2, "evidence_strength")[
            [
                "metric_id",
                "metric_instance_id",
                "task_scope",
                "value",
                "reference_label",
                "reference_median",
                "directional_z",
                "reliability",
            ]
        ].to_dict("records")
        local_segments = _segment_evidence(
            subject_id,
            state_base_id,
            recordings,
            segments,
            task_scope=task_scope,
            segment_lookup=segment_lookup,
        )
        rows.append(
            {
                "dataset_id": dataset_id,
                "subject_id": subject_id,
                "label": label,
                "split": split,
                "state_id": state_id,
                "state_base_id": state_base_id,
                "task_scope": task_scope,
                "state_name_zh": state_name_zh,
                "branch": state_branch,
                "clinical_question": clinical_question,
                "state_z": state_z,
                "raw_state_z": raw_state_z,
                "report_state_z": report_state_z,
                "report_confidence": report_confidence,
                "report_permission": bool(report_denominator > 0.0),
                "metric_contribution_clip_z": contribution_limit,
                "severity": float(expit(state_z)),
                "category": _state_category(state_z, reliability, missing_fraction),
                "confidence": reliability,
                "missing_fraction": missing_fraction,
                "supporting_metrics": json.dumps(support, ensure_ascii=False),
                "counter_evidence": json.dumps(counter, ensure_ascii=False),
                "evidence_segments": json.dumps(local_segments, ensure_ascii=False),
                "trace_resolution": (
                    "task_and_segment"
                    if task_scope != "overall" and local_segments
                    else "segment"
                    if local_segments
                    else "task_and_metric"
                    if task_scope != "overall"
                    else "metric"
                ),
            }
        )
    cards = pd.DataFrame(rows)
    cards.to_csv(state_cards_path, index=False)
    identity = ["dataset_id", "subject_id", "label", "split"]
    score = cards.pivot(index=identity, columns="state_id", values="state_z").add_prefix("state_")
    confidence = cards.pivot(index=identity, columns="state_id", values="confidence").add_prefix("rel_")
    wide = score.join(confidence).reset_index()
    wide.to_csv(state_wide_path, index=False)
