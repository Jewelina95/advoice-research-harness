"""Read-only adapter from legacy frozen artifacts to conditional-authority evidence.

This module is deliberately limited to validation and conversion.  It does
not train a model, construct an Agent runtime, or invoke an external API.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping
from urllib.parse import quote

import pandas as pd

from .evidence import (
    ConfoundSets,
    EvidencePermissions,
    EvidenceProvenance,
    MetricEvidenceV2,
    ReferenceMetadata,
    ReliabilityComponents,
)


_REQUIRED_FILES = ("manifest.csv", "metric_evidence.csv", "segments.csv", "state_wide.csv")
_MANIFEST_REQUIRED = ("dataset_id", "case_id", "subject_id", "label", "split")
_EVIDENCE_REQUIRED = (
    "subject_id", "label", "split", "metric_id", "metric_instance_id", "state_id",
    "task_scope", "value", "direction", "reference_median", "reference_scale",
    "reliability", "missing", "confound_tags", "report_permission",
)
_STATE_REQUIRED = ("subject_id", "label", "split")
_LABEL_LEAKAGE_NAME = re.compile(
    r"(?:^|_)(?:label|labels|target|targets|diagnosis|diagnostic|class|outcome|ground_truth|truth)(?:$|_)",
    re.IGNORECASE,
)
_STATE_DRIVING_ROLES = frozenset({"clinical_support", "cautious_support", "model_auxiliary"})


class FrozenAuthorityArtifactError(ValueError):
    """Raised when a frozen artifact cannot be safely admitted."""


def _read_csv(path: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(path, dtype="string", keep_default_na=False)
    except (OSError, pd.errors.EmptyDataError) as exc:
        raise FrozenAuthorityArtifactError(f"Cannot read required artifact {path.name}.") from exc


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_columns(frame: pd.DataFrame, columns: tuple[str, ...], name: str) -> None:
    missing = [column for column in columns if column not in frame]
    if missing:
        raise FrozenAuthorityArtifactError(f"{name} is missing required columns: {missing}")


def _text(value: Any, field: str, *, row: int | None = None) -> str:
    result = str(value).strip()
    if not result or result.lower() in {"nan", "none", "null", "<na>"}:
        location = "" if row is None else f" at row {row}"
        raise FrozenAuthorityArtifactError(f"{field} cannot be blank{location}.")
    return result


def _optional_text(value: Any) -> str:
    result = str(value).strip()
    return "" if result.lower() in {"", "nan", "none", "null", "<na>"} else result


def _strict_bool(value: Any, field: str, *, row: int) -> bool:
    if isinstance(value, bool):
        return value
    normalized = _optional_text(value).lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise FrozenAuthorityArtifactError(f"{field} must be true/false or 1/0 at row {row}.")


def _finite(value: Any, field: str, *, row: int, minimum: float | None = None) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise FrozenAuthorityArtifactError(f"{field} must be numeric at row {row}.") from exc
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        comparator = "finite" if minimum is None else f"finite and >= {minimum}"
        raise FrozenAuthorityArtifactError(f"{field} must be {comparator} at row {row}.")
    return parsed


def _parse_direction(value: Any, *, row: int) -> int:
    parsed = _finite(value, "direction", row=row)
    if parsed not in {-1.0, 0.0, 1.0}:
        raise FrozenAuthorityArtifactError(f"direction must be -1, 0, or 1 at row {row}.")
    return int(parsed)


def _parse_confound_tags(value: Any, *, row: int) -> ConfoundSets:
    try:
        parsed = json.loads(_text(value, "confound_tags", row=row))
    except json.JSONDecodeError as exc:
        raise FrozenAuthorityArtifactError(f"confound_tags must be JSON at row {row}.") from exc
    if isinstance(parsed, list) and all(isinstance(item, str) and item.strip() for item in parsed):
        return ConfoundSets(potential=tuple(parsed))
    if isinstance(parsed, dict) and set(parsed).issubset({"potential", "observed", "ruled_out"}):
        normalized: dict[str, tuple[str, ...]] = {}
        for key, items in parsed.items():
            if not isinstance(items, list) or any(not isinstance(item, str) or not item.strip() for item in items):
                raise FrozenAuthorityArtifactError(f"confound_tags.{key} must be a JSON string list at row {row}.")
            normalized[key] = tuple(items)
        return ConfoundSets(**normalized)
    raise FrozenAuthorityArtifactError(f"confound_tags must be a JSON list or confound-set object at row {row}.")


def _parse_id_list(value: Any, *, field: str, row: int) -> tuple[str, ...]:
    text = _optional_text(value)
    if not text:
        return ()
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise FrozenAuthorityArtifactError(f"{field} must be a JSON string list at row {row}.") from exc
    if not isinstance(parsed, list) or any(not isinstance(item, str) or not item.strip() for item in parsed):
        raise FrozenAuthorityArtifactError(f"{field} must be a JSON string list at row {row}.")
    return tuple(dict.fromkeys(item.strip() for item in parsed))


def _task_scope(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def _identity_component(value: str) -> str:
    """Encode an identity component without changing its stable meaning."""

    return quote(_text(value, "identity"), safe="-_.~")


def _asset_id(record: Mapping[str, Any]) -> str:
    """Return only an asset identity that is actually present in the manifest."""

    digest = _optional_text(record.get("audio_sha256", ""))
    if digest:
        return f"sha256:{digest}"
    path = _optional_text(record.get("audio_path", ""))
    return f"path:{path}" if path else ""


def _manifest_recordings(manifest: pd.DataFrame) -> Mapping[str, tuple[Mapping[str, str], ...]]:
    recordings: dict[str, list[Mapping[str, str]]] = {}
    for row_number, row in manifest.iterrows():
        subject = _text(row["subject_id"], "manifest.subject_id", row=row_number)
        case_id = _text(row["case_id"], "manifest.case_id", row=row_number) if "case_id" in manifest else subject
        record = {str(key): _optional_text(value) for key, value in row.items()}
        record["case_id"] = case_id
        recordings.setdefault(subject, []).append(MappingProxyType(record))
    return MappingProxyType({key: tuple(value) for key, value in recordings.items()})


def _recording_candidates(
    subject: str,
    scope: str,
    row: Mapping[str, Any],
    recordings: Mapping[str, tuple[Mapping[str, str], ...]],
) -> tuple[Mapping[str, str], ...]:
    available = recordings.get(subject, ())
    if not available:
        raise FrozenAuthorityArtifactError(f"No manifest recording is available for subject {subject!r}.")

    explicit_case = _optional_text(row.get("case_id", ""))
    explicit_session = _optional_text(row.get("session_id", row.get("recording_id", "")))
    if explicit_case:
        available = tuple(item for item in available if item.get("case_id") == explicit_case)
    if explicit_session:
        session_matches = tuple(
            item for item in available
            if item.get("case_id") == explicit_session
            or item.get("session_id") == explicit_session
            or item.get("recording_id") == explicit_session
        )
        if not session_matches:
            raise FrozenAuthorityArtifactError(
                f"Evidence session identity cannot be resolved for subject {subject!r}."
            )
        available = session_matches
    if not available:
        raise FrozenAuthorityArtifactError(
            f"Evidence recording identity cannot be resolved for subject {subject!r}."
        )
    normalized_scope = _task_scope(scope)
    if normalized_scope not in {"", "overall"}:
        task_matches = tuple(
            item for item in available
            if _task_scope(item.get("task_scope", "")) == normalized_scope
            or _task_scope(item.get("task_type", "")) == normalized_scope
            or _task_scope(item.get("task_id", "")) == normalized_scope
        )
        if task_matches:
            available = task_matches
        elif len(available) > 1:
            raise FrozenAuthorityArtifactError(
                f"Task scope {scope!r} cannot be mapped to one manifest recording for {subject!r}."
            )
    return available


def _recording_session_id(recordings: tuple[Mapping[str, str], ...]) -> str:
    cases = tuple(_text(item["case_id"], "manifest.case_id") for item in recordings)
    if len(cases) == 1:
        return cases[0]
    return "recording-set:" + "|".join(cases)


def _manifest_metadata(manifest: pd.DataFrame) -> Mapping[str, Mapping[str, tuple[str, ...]]]:
    metadata: dict[str, Mapping[str, tuple[str, ...]]] = {}
    preserved = ("task_scope", "task_id", "task_type", "language", "channel", "dataset_id", "case_id")
    for subject_id, group in manifest.groupby("subject_id", sort=False):
        values: dict[str, tuple[str, ...]] = {}
        for column in preserved:
            if column in group:
                items = tuple(dict.fromkeys(
                    text for value in group[column].tolist() if (text := _optional_text(value))
                ))
                if items:
                    values[column] = items
        metadata[str(subject_id)] = MappingProxyType(values)
    return MappingProxyType(metadata)


def _subject_contract(manifest: pd.DataFrame) -> tuple[Mapping[str, str], Mapping[str, str]]:
    labels: dict[str, str] = {}
    splits: dict[str, str] = {}
    for subject_id, group in manifest.groupby("subject_id", sort=False):
        subject = _text(subject_id, "manifest.subject_id")
        subject_labels = {_text(value, "manifest.label") for value in group["label"].tolist()}
        subject_splits = {_text(value, "manifest.split") for value in group["split"].tolist()}
        if len(subject_labels) != 1 or len(subject_splits) != 1:
            raise FrozenAuthorityArtifactError(
                f"Manifest subject {subject!r} has inconsistent labels or splits."
            )
        labels[subject] = next(iter(subject_labels))
        splits[subject] = next(iter(subject_splits))
    return MappingProxyType(labels), MappingProxyType(splits)


def _validate_state_wide(
    state_wide: pd.DataFrame,
    subject_labels: Mapping[str, str],
    subject_splits: Mapping[str, str],
) -> None:
    _required_columns(state_wide, _STATE_REQUIRED, "state_wide.csv")
    subjects = [_text(value, "state_wide.subject_id", row=index) for index, value in state_wide["subject_id"].items()]
    duplicates = sorted({item for item in subjects if subjects.count(item) > 1})
    if duplicates:
        raise FrozenAuthorityArtifactError(f"state_wide.csv has duplicate subject IDs: {duplicates}")
    if set(subjects) != set(subject_labels):
        raise FrozenAuthorityArtifactError("state_wide.csv subjects must exactly match manifest subjects.")
    for index, row in state_wide.iterrows():
        subject = _text(row["subject_id"], "state_wide.subject_id", row=index)
        if _text(row["label"], "state_wide.label", row=index) != subject_labels[subject]:
            raise FrozenAuthorityArtifactError(f"state_wide.csv label disagrees with manifest for {subject!r}.")
        if _text(row["split"], "state_wide.split", row=index) != subject_splits[subject]:
            raise FrozenAuthorityArtifactError(f"state_wide.csv split disagrees with manifest for {subject!r}.")

    identities = {"dataset_id", "subject_id", "label", "split"}
    test_rows = state_wide["split"].map(_optional_text).eq("test")
    for column in state_wide.columns:
        if column in identities:
            continue
        if _LABEL_LEAKAGE_NAME.search(column):
            raise FrozenAuthorityArtifactError(f"state_wide.csv feature column {column!r} is label leakage.")
        if test_rows.any():
            feature = state_wide.loc[test_rows, column].map(_optional_text)
            labels = state_wide.loc[test_rows, "label"].map(_optional_text)
            if not feature.empty and feature.eq(labels).all():
                raise FrozenAuthorityArtifactError(f"state_wide.csv feature column {column!r} reproduces test labels.")


def _validate_evidence_contract(
    evidence: pd.DataFrame,
    subject_labels: Mapping[str, str],
    subject_splits: Mapping[str, str],
) -> None:
    _required_columns(evidence, _EVIDENCE_REQUIRED, "metric_evidence.csv")
    subjects: set[str] = set()
    for index, row in evidence.iterrows():
        subject = _text(row["subject_id"], "metric_evidence.subject_id", row=index)
        subjects.add(subject)
        if subject not in subject_labels:
            raise FrozenAuthorityArtifactError(f"metric_evidence.csv has unknown subject {subject!r}.")
        if _text(row["label"], "metric_evidence.label", row=index) != subject_labels[subject]:
            raise FrozenAuthorityArtifactError(f"metric_evidence.csv label disagrees with manifest for {subject!r}.")
        if _text(row["split"], "metric_evidence.split", row=index) != subject_splits[subject]:
            raise FrozenAuthorityArtifactError(f"metric_evidence.csv split disagrees with manifest for {subject!r}.")
    if subjects != set(subject_labels):
        raise FrozenAuthorityArtifactError("metric_evidence.csv subjects must exactly match manifest subjects.")


def _normalize_segments(
    segments: pd.DataFrame,
    *,
    manifest: pd.DataFrame,
    recordings: Mapping[str, tuple[Mapping[str, str], ...]],
) -> pd.DataFrame:
    """Bind recording segments to a subject execution case without losing source identity."""

    if segments.empty:
        return segments.copy(deep=True)
    if "segment_id" not in segments:
        raise FrozenAuthorityArtifactError("Non-empty segments.csv requires segment_id.")

    case_to_subject: dict[str, str] = {}
    case_to_recording: dict[str, Mapping[str, str]] = {}
    for subject, items in recordings.items():
        for item in items:
            case = _text(item["case_id"], "manifest.case_id")
            if case in case_to_subject and case_to_subject[case] != subject:
                raise FrozenAuthorityArtifactError(f"Manifest case_id {case!r} belongs to multiple subjects.")
            case_to_subject[case] = subject
            case_to_recording[case] = item

    normalized_rows: list[dict[str, Any]] = []
    seen_segments: dict[str, tuple[str, str]] = {}
    for row_number, raw in segments.iterrows():
        row = {str(key): value for key, value in raw.items()}
        explicit_subject = _optional_text(row.get("subject_id", ""))
        raw_case = _optional_text(row.get("case_id", ""))
        if raw_case and raw_case not in case_to_subject:
            raise FrozenAuthorityArtifactError(
                f"segments.case_id {raw_case!r} cannot be resolved to manifest provenance at row {row_number}."
            )
        if not explicit_subject and raw_case:
            explicit_subject = case_to_subject[raw_case]
        if explicit_subject not in recordings:
            raise FrozenAuthorityArtifactError(
                f"segments.subject_id {explicit_subject!r} cannot be resolved at row {row_number}."
            )

        task_column = next((column for column in ("task_scope", "task_id", "task_type") if column in row), None)
        requested_scope = _task_scope(_optional_text(row.get(task_column, ""))) if task_column else ""
        candidates = recordings[explicit_subject]
        if raw_case:
            candidates = tuple(item for item in candidates if item["case_id"] == raw_case)
        elif requested_scope and requested_scope != "overall":
            candidates = tuple(
                item for item in candidates
                if requested_scope in {
                    _task_scope(item.get("task_scope", "")),
                    _task_scope(item.get("task_type", "")),
                    _task_scope(item.get("task_id", "")),
                }
            )
        if len(candidates) != 1:
            raise FrozenAuthorityArtifactError(
                f"Segment row {row_number} does not identify exactly one recording case for {explicit_subject!r}."
            )
        recording = candidates[0]
        original_case = recording["case_id"]
        scope = requested_scope or _task_scope(
            recording.get("task_scope", recording.get("task_type", ""))
        ) or "overall"
        segment_id = _text(row["segment_id"], "segments.segment_id", row=row_number)
        identity = (explicit_subject, original_case)
        if segment_id in seen_segments and seen_segments[segment_id] != identity:
            raise FrozenAuthorityArtifactError(f"segment_id {segment_id!r} maps to multiple recording cases.")
        seen_segments[segment_id] = identity

        explicit_asset = _optional_text(
            row.get("source_asset_id", row.get("audio_asset_id", row.get("audio_path", "")))
        )
        asset = explicit_asset or _asset_id(recording)
        row["subject_id"] = explicit_subject
        row["original_recording_case_id"] = original_case
        # case_id is the subject-level execution identity consumed by the lock.
        row["case_id"] = explicit_subject
        row["_authority_task_scope"] = scope
        row["source_asset_id"] = asset
        normalized_rows.append(row)

    return pd.DataFrame(normalized_rows, columns=list(dict.fromkeys(
        [*segments.columns, "subject_id", "case_id", "original_recording_case_id",
         "source_asset_id", "_authority_task_scope"]
    )))


def _segment_index(segments: pd.DataFrame) -> Mapping[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]]:
    if segments.empty:
        return MappingProxyType({})
    result: dict[tuple[str, str], tuple[tuple[str, ...], tuple[str, ...]]] = {}
    grouped = segments.groupby(["subject_id", "_authority_task_scope"], sort=False)
    for (subject, scope), group in grouped:
        segment_ids = tuple(dict.fromkeys(_text(value, "segments.segment_id") for value in group["segment_id"].tolist()))
        assets = tuple(dict.fromkeys(
            text for value in group["source_asset_id"].tolist() if (text := _optional_text(value))
        ))
        result[(str(subject), str(scope))] = (segment_ids, assets)

    for subject, group in segments.groupby("subject_id", sort=False):
        segment_ids = tuple(dict.fromkeys(_text(value, "segments.segment_id") for value in group["segment_id"].tolist()))
        assets = tuple(dict.fromkeys(
            text for value in group["source_asset_id"].tolist() if (text := _optional_text(value))
        ))
        result[(str(subject), "overall")] = (segment_ids, assets)
    return MappingProxyType(result)


def _evidence_id(
    dataset_id: str,
    session_id: str,
    case_id: str,
    subject_id: str,
    metric_instance_id: str,
    state_id: str,
    task_scope: str,
) -> str:
    """Build a row-order-independent identity with all execution boundaries visible."""

    parts = (
        ("dataset", dataset_id),
        ("session", session_id or "unknown"),
        ("case", case_id or "unknown"),
        ("subject", subject_id),
        ("task", task_scope),
        ("metric", metric_instance_id),
        ("state", state_id),
    )
    return "metric:" + ":".join(f"{key}={_identity_component(value)}" for key, value in parts)


def _convert_evidence(
    evidence: pd.DataFrame,
    *,
    manifest: pd.DataFrame,
    segments: pd.DataFrame,
) -> Mapping[str, tuple[MetricEvidenceV2, ...]]:
    recordings = _manifest_recordings(manifest)
    index = _segment_index(segments)
    converted: dict[str, list[MetricEvidenceV2]] = {}
    seen_ids: set[str] = set()
    for row_number, row in evidence.iterrows():
        subject = _text(row["subject_id"], "metric_evidence.subject_id", row=row_number)
        metric_id = _text(row["metric_id"], "metric_evidence.metric_id", row=row_number)
        metric_instance_id = _text(row["metric_instance_id"], "metric_evidence.metric_instance_id", row=row_number)
        state_id = _text(row["state_id"], "metric_evidence.state_id", row=row_number)
        scope = _text(row["task_scope"], "metric_evidence.task_scope", row=row_number)
        candidate_recordings = _recording_candidates(subject, scope, row, recordings)
        dataset_ids = tuple(dict.fromkeys(
            text for item in candidate_recordings
            if (text := _optional_text(item.get("dataset_id", "")))
        ))
        explicit_dataset = _optional_text(row.get("dataset_id", ""))
        if explicit_dataset and dataset_ids and explicit_dataset not in dataset_ids:
            raise FrozenAuthorityArtifactError(
                f"metric_evidence.dataset_id disagrees with manifest for {subject!r} at row {row_number}."
            )
        dataset_id = explicit_dataset or (dataset_ids[0] if len(dataset_ids) == 1 else "")
        if not dataset_id:
            raise FrozenAuthorityArtifactError(
                f"metric_evidence.dataset_id is unknown for {subject!r} at row {row_number}."
            )
        session_id = _recording_session_id(candidate_recordings)
        normalized_case_id = subject
        missing = _strict_bool(row["missing"], "metric_evidence.missing", row=row_number)
        report_permission = _strict_bool(
            row["report_permission"], "metric_evidence.report_permission", row=row_number
        )
        reliability = _finite(row["reliability"], "metric_evidence.reliability", row=row_number, minimum=0.0)
        if reliability > 1.0:
            raise FrozenAuthorityArtifactError(f"metric_evidence.reliability must be <= 1 at row {row_number}.")
        median = _finite(row["reference_median"], "metric_evidence.reference_median", row=row_number)
        scale = _finite(row["reference_scale"], "metric_evidence.reference_scale", row=row_number)
        if scale <= 0.0:
            raise FrozenAuthorityArtifactError(f"metric_evidence.reference_scale must be > 0 at row {row_number}.")
        raw_value = _optional_text(row["value"])
        if raw_value:
            value: float | None = _finite(raw_value, "metric_evidence.value", row=row_number)
        elif missing:
            value = None
        else:
            raise FrozenAuthorityArtifactError(f"metric_evidence.value cannot be blank when missing is false at row {row_number}.")
        if "inference_permission" in evidence:
            inference_permission = _strict_bool(
                row["inference_permission"], "metric_evidence.inference_permission", row=row_number
            )
        else:
            # Legacy artifacts did not persist this field.  Missing evidence is never admitted to inference.
            inference_permission = not missing and reliability > 0.0
        if "observable" in evidence:
            observable = _strict_bool(row["observable"], "metric_evidence.observable", row=row_number)
        else:
            observable = not missing
        evidence_role = _optional_text(row.get("evidence_role", ""))
        branch = _optional_text(row.get("branch", ""))
        state_driving = (
            state_id.lower() != "qc"
            and branch.lower() != "qc"
            and (not evidence_role or evidence_role.lower() in _STATE_DRIVING_ROLES)
        )
        if "consumed_by_supervised" in evidence:
            consumed = _strict_bool(
                row["consumed_by_supervised"], "metric_evidence.consumed_by_supervised", row=row_number
            )
            if state_driving and not consumed:
                raise FrozenAuthorityArtifactError(
                    f"State-driving evidence must be consumed_by_supervised=true at row {row_number}."
                )
        else:
            consumed = state_driving
        incremental = _strict_bool(row["incremental_for_agent"], "metric_evidence.incremental_for_agent", row=row_number) if "incremental_for_agent" in evidence else False
        if consumed and incremental:
            raise FrozenAuthorityArtifactError(f"Evidence cannot be both consumed and incremental at row {row_number}.")
        source_segments = _parse_id_list(row["source_segment_ids"], field="source_segment_ids", row=row_number) if "source_segment_ids" in evidence else ()
        source_asset = _optional_text(row["source_asset_id"]) if "source_asset_id" in evidence else ""
        normalized_scope = _task_scope(scope) or "overall"
        indexed_segments, indexed_assets = index.get(
            (subject, normalized_scope), index.get((subject, "overall"), ((), ()))
        )
        if not source_segments:
            source_segments = indexed_segments
        if not source_asset and len(indexed_assets) == 1:
            source_asset = indexed_assets[0]
        known_segment_ids = set(indexed_segments)
        source_segments_proven = bool(source_segments) and set(source_segments).issubset(known_segment_ids)
        source_asset_proven = bool(source_asset) and bool(indexed_assets) and source_asset in set(indexed_assets)
        report_provenance = source_segments_proven and source_asset_proven and not missing and observable
        if source_asset and indexed_assets and source_asset not in set(indexed_assets):
            report_provenance = False
        report_permission = report_permission and report_provenance
        evidence_id = _evidence_id(
            dataset_id, session_id, normalized_case_id, subject,
            metric_instance_id, state_id, normalized_scope,
        )
        if evidence_id in seen_ids:
            raise FrozenAuthorityArtifactError(f"metric_evidence.csv has duplicate evidence identity {evidence_id!r}.")
        seen_ids.add(evidence_id)
        converted.setdefault(subject, []).append(MetricEvidenceV2(
            evidence_id=evidence_id,
            metric_id=metric_id,
            metric_instance_id=metric_instance_id,
            subject_id=subject,
            session_id=session_id,
            case_id=normalized_case_id,
            state_id=state_id,
            task_id=normalized_scope,
            value=value,
            unit=_optional_text(row.get("unit", "")),
            source_modality=_optional_text(row.get("source_modality", "")),
            direction=_parse_direction(row["direction"], row=row_number),
            direction_provenance=_optional_text(row.get("direction_provenance", "legacy_metric_evidence")),
            observable=observable,
            unavailable_reason=_optional_text(row.get("unavailable_reason", "")) or None,
            reference=ReferenceMetadata(
                median=median,
                scale=scale,
                sample_size=int(_finite(row["reference_sample_size"], "metric_evidence.reference_sample_size", row=row_number, minimum=0.0)) if "reference_sample_size" in evidence else 0,
                artifact_id=_optional_text(row.get("reference_artifact_id", "")),
                artifact_hash=_optional_text(row.get("reference_artifact_hash", "")),
                scope=_optional_text(row.get("reference_scope", "")),
                reference_label=_optional_text(row.get("reference_label", "")) or "unknown",
            ),
            reliability_components=ReliabilityComponents(source=reliability),
            provenance=EvidenceProvenance(
                source_asset_id=source_asset,
                source_segment_ids=source_segments,
                transcript_id=_optional_text(row.get("transcript_id", "")) or None,
                method_version=_optional_text(row.get("method_version", "")),
                measurement_version=_optional_text(row.get("measurement_version", "")),
                generated_by=_optional_text(row.get("generated_by", "")),
            ),
            confounds=_parse_confound_tags(row["confound_tags"], row=row_number),
            permissions=EvidencePermissions(inference=inference_permission, report=report_permission),
            consumed_by_supervised=consumed,
            incremental_for_agent=incremental,
            reliability_migration="legacy_scalar_total",
        ))
    return MappingProxyType({key: tuple(value) for key, value in converted.items()})


@dataclass(frozen=True, slots=True)
class FrozenAuthorityDataset:
    """Validated, read-only projection of one frozen artifact directory."""

    artifact_dir: Path
    manifest: pd.DataFrame
    metric_evidence: pd.DataFrame
    segments: pd.DataFrame
    state_wide: pd.DataFrame
    evidence_by_subject: Mapping[str, tuple[MetricEvidenceV2, ...]]
    subject_labels: Mapping[str, str]
    subject_splits: Mapping[str, str]
    subject_metadata: Mapping[str, Mapping[str, tuple[str, ...]]]
    artifact_hashes: Mapping[str, str]

    @property
    def evidence(self) -> tuple[MetricEvidenceV2, ...]:
        return tuple(item for subject in self.evidence_by_subject.values() for item in subject)

    def evidence_for_subject(self, subject_id: str) -> tuple[MetricEvidenceV2, ...]:
        return self.evidence_by_subject.get(str(subject_id), ())


def load_frozen_authority_dataset(path: str | Path) -> FrozenAuthorityDataset:
    """Load one immutable-authority dataset without fitting or contacting any service."""

    artifact_dir = Path(path).expanduser().resolve()
    if not artifact_dir.is_dir():
        raise FrozenAuthorityArtifactError(f"Artifact directory does not exist: {artifact_dir}")
    paths = {name: artifact_dir / name for name in _REQUIRED_FILES}
    missing = [name for name, file_path in paths.items() if not file_path.is_file()]
    if missing:
        raise FrozenAuthorityArtifactError(f"Artifact directory is missing required files: {missing}")
    manifest = _read_csv(paths["manifest.csv"])
    evidence = _read_csv(paths["metric_evidence.csv"])
    segments = _read_csv(paths["segments.csv"])
    state_wide = _read_csv(paths["state_wide.csv"])
    _required_columns(manifest, _MANIFEST_REQUIRED, "manifest.csv")
    labels, splits = _subject_contract(manifest)
    _validate_state_wide(state_wide, labels, splits)
    _validate_evidence_contract(evidence, labels, splits)
    recordings = _manifest_recordings(manifest)
    normalized_segments = _normalize_segments(
        segments, manifest=manifest, recordings=recordings
    )
    return FrozenAuthorityDataset(
        artifact_dir=artifact_dir,
        manifest=manifest.copy(deep=True),
        metric_evidence=evidence.copy(deep=True),
        segments=normalized_segments.copy(deep=True),
        state_wide=state_wide.copy(deep=True),
        evidence_by_subject=_convert_evidence(
            evidence, manifest=manifest, segments=normalized_segments
        ),
        subject_labels=labels,
        subject_splits=splits,
        subject_metadata=_manifest_metadata(manifest),
        artifact_hashes=MappingProxyType({name: _hash_file(file_path) for name, file_path in paths.items()}),
    )
