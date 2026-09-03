from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import json_dump


TASK_METRIC = re.compile(r"^task_(.+?)__(.+)$")
LANGUAGE_DEPENDENT_METRICS = {
    "word_count",
    "speech_rate_wpm",
    "lexical_ttr",
    "lexical_mattr50",
    "filler_rate_100w",
    "repair_rate_100w",
    "pronoun_ratio",
    "content_word_ratio",
    "mean_utterance_words",
    "picture_content_unit_coverage",
    "picture_information_density",
    "picture_content_redundancy",
    "picture_uncertainty_rate_100w",
}
MIN_LANGUAGE_REFERENCE_SUBJECTS = 8


def _robust_reference(values: pd.Series) -> tuple[float, float, bool]:
    finite = values.replace([np.inf, -np.inf], np.nan).dropna().astype(float)
    if finite.empty:
        return 0.0, 1.0, False
    median = float(finite.median())
    mad = float((finite - median).abs().median())
    scale = max(1.4826 * mad, float(finite.std(ddof=0)) * 0.25)
    tolerance = max(1e-8, abs(median) * 1e-8)
    available = bool(finite.nunique(dropna=True) >= 2 and scale > tolerance)
    return median, scale if available else 1.0, available


def recalibrate_metric_evidence_frame(
    evidence: pd.DataFrame,
    *,
    reference_subject_ids: set[str],
    target_subject_ids: set[str],
) -> pd.DataFrame:
    """Rebuild target evidence using only an outer-fit HC reference cohort."""

    frame = evidence.copy()
    subject = frame["subject_id"].astype(str)
    reference = frame[subject.isin({str(value) for value in reference_subject_ids})]
    target = frame[subject.isin({str(value) for value in target_subject_ids})].copy()
    if reference.empty or target.empty:
        raise ValueError("Fold-local metric evidence requires reference and target cases.")
    for metric_instance, row_indices in target.groupby("metric_instance_id").groups.items():
        target_rows = target.loc[row_indices]
        metric_id = str(target_rows["metric_id"].iloc[0])
        reference_rows = reference[reference["metric_instance_id"].eq(metric_instance)]
        for index, row in target_rows.iterrows():
            scoped_reference = reference_rows
            reference_scope = "outer_fit_hc_reference"
            if metric_id in LANGUAGE_DEPENDENT_METRICS:
                language = str(row.get("language", "unknown"))
                scoped_reference = reference_rows[
                    reference_rows["language"].fillna("unknown").astype(str).eq(language)
                ]
                reference_scope = f"outer_fit_hc_reference_language:{language}"
            values = pd.to_numeric(scoped_reference["value"], errors="coerce")
            median, scale, variable = _robust_reference(values)
            enough = (
                int(values.notna().sum()) >= MIN_LANGUAGE_REFERENCE_SUBJECTS
                if metric_id in LANGUAGE_DEPENDENT_METRICS
                else int(values.notna().sum()) >= 2
            )
            available = bool(variable and enough)
            value = pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0]
            missing = bool(pd.isna(value) or not available)
            robust_z = float((float(value) - median) / scale) if not missing else np.nan
            direction = int(row.get("direction", 0))
            target.at[index, "reference_scope"] = reference_scope
            target.at[index, "reference_median"] = median
            target.at[index, "reference_scale"] = scale
            target.at[index, "cn_train_median"] = median
            target.at[index, "cn_train_scale"] = scale
            target.at[index, "robust_z"] = robust_z
            target.at[index, "directional_z"] = (
                float(direction * robust_z) if direction and not missing else 0.0
            )
            target.at[index, "missing"] = missing
            if not available:
                target.at[index, "reliability"] = 0.0
    return target.reset_index(drop=True)


def build_metric_evidence(
    subject_features_path: Path,
    metrics_config: dict[str, Any],
    evidence_path: Path,
    reference_path: Path,
    reference_label: str = "HC",
) -> None:
    subjects = pd.read_csv(subject_features_path, dtype={"subject_id": str})
    metric_defs = metrics_config["metrics"]
    controls = subjects[
        subjects["split"].eq("train") & subjects["label"].eq(str(reference_label))
    ]
    if controls.empty:
        raise ValueError(
            f"No training subjects found for configured metric reference label {reference_label!r}."
        )
    references: dict[str, dict[str, float]] = {}
    rows: list[dict[str, Any]] = []
    task_scopes = sorted(
        {
            match.group(1)
            for column in subjects.columns
            if (match := TASK_METRIC.match(column)) is not None
        }
    )
    use_task_specific_evidence = len(task_scopes) > 1
    for definition in metric_defs:
        metric = definition["id"]
        metric_instances = [("overall", metric)] if metric in subjects.columns else []
        if use_task_specific_evidence:
            metric_instances.extend(
                (task_scope, f"task_{task_scope}__{metric}")
                for task_scope in task_scopes
                if f"task_{task_scope}__{metric}" in subjects.columns
            )
        if not metric_instances:
            references[metric] = {"median": 0.0, "scale": 1.0, "available": False}
            continue
        for task_scope, metric_instance in metric_instances:
            finite_reference = controls[metric_instance].replace([np.inf, -np.inf], np.nan).dropna()
            median, scale, reference_available = _robust_reference(
                controls[metric_instance]
            )
            references[metric_instance] = {
                "metric_id": metric,
                "task_scope": task_scope,
                "median": median,
                "scale": scale,
                "available": reference_available,
            }
            language_references: dict[str, dict[str, Any]] = {}
            if metric in LANGUAGE_DEPENDENT_METRICS and "language" in subjects.columns:
                for language_value in subjects["language"].fillna("unknown").astype(str).unique():
                    language_controls = controls[
                        controls["language"].fillna("unknown").astype(str).eq(language_value)
                    ][metric_instance]
                    language_finite = language_controls.replace([np.inf, -np.inf], np.nan).dropna()
                    language_median, language_scale, language_variable = (
                        _robust_reference(language_controls)
                    )
                    language_references[language_value] = {
                        "median": language_median,
                        "scale": language_scale,
                        "n": int(len(language_finite)),
                        "available": bool(
                            len(language_finite) >= MIN_LANGUAGE_REFERENCE_SUBJECTS
                            and language_variable
                        ),
                    }
                references[metric_instance]["language_references"] = language_references
            prefix = "" if task_scope == "overall" else f"task_{task_scope}__"
            for subject in subjects.to_dict("records"):
                if task_scope != "overall":
                    duration_column = f"{prefix}duration_sec"
                    task_duration = subject.get(duration_column)
                    if duration_column in subjects.columns and (
                        task_duration is None or not np.isfinite(task_duration)
                    ):
                        # A task-specific scope is non-applicable when the subject
                        # did not perform that task; it is not missing clinical evidence.
                        continue
                value = subject.get(metric_instance)
                subject_language = str(subject.get("language", "unknown"))
                reference_scope = "pooled_training_reference"
                subject_median, subject_scale = median, scale
                if metric in LANGUAGE_DEPENDENT_METRICS:
                    language_reference = language_references.get(subject_language, {})
                    subject_median = float(language_reference.get("median", 0.0))
                    subject_scale = float(language_reference.get("scale", 1.0))
                    reference_available = bool(language_reference.get("available", False))
                    reference_scope = f"training_reference_language:{subject_language}"
                missing = value is None or not np.isfinite(value) or not reference_available
                z = float((value - subject_median) / subject_scale) if not missing else np.nan
                direction = int(definition["direction"])
                directional_z = float(direction * z) if direction and not missing else 0.0
                branch = definition["branch"]
                audio_reliability = subject.get(f"{prefix}audio_reliability", subject.get("audio_reliability", 0.0))
                text_reliability = subject.get(f"{prefix}text_reliability", subject.get("text_reliability", 0.0))
                role_filtered = subject.get(f"{prefix}role_filtered_audio", subject.get("role_filtered_audio", 0.0))
                if metric == "speech_rate_wpm":
                    source_modality = "audio_transcript"
                    source_reliability = min(audio_reliability, text_reliability)
                elif branch == "language":
                    source_modality = "transcript"
                    source_reliability = text_reliability
                elif branch == "interaction":
                    source_modality = "role_aligned_audio_transcript"
                    source_reliability = min(text_reliability, 0.95 if role_filtered > 0 else 0.55)
                elif branch == "qc":
                    source_modality = "quality_control"
                    source_reliability = 1.0
                else:
                    source_modality = "audio"
                    source_reliability = audio_reliability
                source_reliability = float(source_reliability) if np.isfinite(source_reliability) else 0.0
                reliability = float(definition["reliability"] * source_reliability)
                if not reference_available:
                    reliability = 0.0
                if metric.startswith("f0_"):
                    f0_valid = subject.get(f"{prefix}f0_valid_fraction", subject.get("f0_valid_fraction", 0.0))
                    reliability *= float(np.clip(f0_valid / 0.45, 0.0, 1.0))
                rows.append(
                    {
                        "dataset_id": subject["dataset_id"],
                        "subject_id": subject["subject_id"],
                        "label": subject["label"],
                        "split": subject["split"],
                        "language": subject_language,
                        "metric_id": metric,
                        "metric_instance_id": metric_instance,
                        "task_scope": task_scope,
                        "state_id": definition["state"],
                        "branch": definition["branch"],
                        "source_modality": source_modality,
                        "source_reliability": source_reliability,
                        "value": value,
                        "reference_label": str(reference_label),
                        "reference_scope": reference_scope,
                        "reference_median": subject_median,
                        "reference_scale": subject_scale,
                        "cn_train_median": subject_median,
                        "cn_train_scale": subject_scale,
                        "robust_z": z,
                        "direction": direction,
                        "directional_z": directional_z,
                        "evidence_role": definition["role"],
                        "reliability": reliability,
                        "missing": bool(missing),
                        "confound_tags": json.dumps(definition.get("confounds", []), ensure_ascii=False),
                        "report_permission": bool(definition["report_permission"]),
                    }
                )
    pd.DataFrame(rows).to_csv(evidence_path, index=False)
    json_dump(
        {
            "reference_population": f"official training split, {reference_label} subjects only",
            "reference_label": str(reference_label),
            "task_specific_evidence": use_task_specific_evidence,
            "task_scopes": task_scopes if use_task_specific_evidence else [],
            "normalization": "median and max(1.4826*MAD, 0.25*SD); constant or near-constant training references are unavailable",
            "metrics": references,
        },
        reference_path,
    )
