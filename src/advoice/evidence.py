from __future__ import annotations

import json
import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

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


_MISSING = object()


def _strict_bool(value: Any, *, field: str, default: bool | object = _MISSING) -> bool:
    """Parse contract booleans without Python's truthiness traps."""

    if value is None or (isinstance(value, float) and np.isnan(value)):
        if default is not _MISSING:
            return bool(default)
        raise ValueError(f"{field} cannot be null.")
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1"}:
            return True
        if normalized in {"false", "0"}:
            return False
        raise ValueError(f"{field} must be a boolean, not {value!r}.")
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return bool(value)
    if isinstance(value, (float, np.floating)) and np.isfinite(value) and value in (0.0, 1.0):
        return bool(value)
    raise ValueError(f"{field} must be a boolean, not {value!r}.")


def _legacy_total_reliability(value: Any) -> tuple["ReliabilityComponents", str]:
    """Migrate one legacy scalar into one explicit reliability component."""

    try:
        total = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Legacy scalar reliability must be numeric.") from exc
    if not np.isfinite(total) or not 0.0 <= total <= 1.0:
        raise ValueError("Legacy scalar reliability must be in [0, 1].")
    # ``source`` carries the historical total while the remaining dimensions
    # stay neutral, so the product remains exactly the legacy scalar.
    return ReliabilityComponents(source=total), "legacy_scalar_total"


def _canonical_json(value: Any) -> str:
    """Serialize contract values without platform- or insertion-order drift."""

    def normalize(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {str(key): normalize(item[key]) for key in sorted(item, key=str)}
        if isinstance(item, (list, tuple)):
            return [normalize(child) for child in item]
        if isinstance(item, (np.integer, np.floating)):
            return item.item()
        if isinstance(item, float) and not np.isfinite(item):
            return None
        return item

    return json.dumps(normalize(value), ensure_ascii=False, sort_keys=True, separators=(",", ":"))


@dataclass(frozen=True, slots=True)
class ReferenceMetadata:
    """Fold-bound reference statistics and their content identity."""

    median: float = 0.0
    scale: float = 1.0
    sample_size: int = 0
    artifact_id: str = ""
    artifact_hash: str = ""
    scope: str = ""
    reference_label: str = "HC"

    def __post_init__(self) -> None:
        if self.sample_size < 0:
            raise ValueError("Reference sample_size cannot be negative.")
        if self.scale <= 0:
            raise ValueError("Reference scale must be positive.")

    @classmethod
    def from_values(
        cls,
        values: Iterable[Any],
        *,
        artifact_id: str = "",
        scope: str = "",
        reference_label: str = "HC",
    ) -> "ReferenceMetadata":
        numeric = []
        for value in values:
            try:
                converted = float(value)
            except (TypeError, ValueError):
                continue
            if np.isfinite(converted):
                numeric.append(converted)
        median, scale, _ = _robust_reference(pd.Series(numeric, dtype=float))
        digest = hashlib.sha256(_canonical_json(numeric).encode("utf-8")).hexdigest()
        return cls(
            median=median,
            scale=scale,
            sample_size=len(numeric),
            artifact_id=artifact_id,
            artifact_hash=digest,
            scope=scope,
            reference_label=reference_label,
        )

    @property
    def reference_sample_size(self) -> int:
        return self.sample_size

    @property
    def reference_artifact_hash(self) -> str:
        return self.artifact_hash

    @property
    def reference_hash(self) -> str:
        return self.artifact_hash

    def to_dict(self) -> dict[str, Any]:
        return {
            "median": self.median,
            "scale": self.scale,
            "sample_size": self.sample_size,
            "artifact_id": self.artifact_id,
            "artifact_hash": self.artifact_hash,
            "scope": self.scope,
            "reference_label": self.reference_label,
        }

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any]) -> "ReferenceMetadata":
        aliases = {
            "reference_median": "median",
            "reference_scale": "scale",
            "reference_sample_size": "sample_size",
            "reference_artifact_id": "artifact_id",
            "reference_artifact_hash": "artifact_hash",
            "reference_scope": "scope",
            "reference_label": "reference_label",
        }
        normalized = {
            aliases.get(str(key), str(key)): item
            for key, item in value.items()
            if aliases.get(str(key), str(key)) in {
                "median", "scale", "sample_size", "artifact_id",
                "artifact_hash", "scope", "reference_label",
            }
        }
        return cls(**normalized)


@dataclass(frozen=True, slots=True)
class ReliabilityComponents:
    """Separate technical reliability dimensions; no scalar is inferred here."""

    source: float = 1.0
    role: float = 1.0
    alignment: float = 1.0
    asr: float = 1.0
    reference_support: float = 1.0
    measurement_stability: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "source", "role", "alignment", "asr", "reference_support",
            "measurement_stability",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"Reliability component {name!r} must be in [0, 1].")
            object.__setattr__(self, name, value)

    @property
    def source_reliability(self) -> float:
        return self.source

    @property
    def role_reliability(self) -> float:
        return self.role

    @property
    def alignment_reliability(self) -> float:
        return self.alignment

    @property
    def asr_reliability(self) -> float:
        return self.asr

    @property
    def role_coverage(self) -> float:
        return self.role

    @property
    def alignment_quality(self) -> float:
        return self.alignment

    @property
    def asr_quality(self) -> float:
        return self.asr

    @property
    def reference_support_reliability(self) -> float:
        return self.reference_support

    @property
    def measurement_stability_score(self) -> float:
        return self.measurement_stability

    def to_dict(self) -> dict[str, float]:
        return {
            "source": self.source,
            "role": self.role,
            "alignment": self.alignment,
            "asr": self.asr,
            "reference_support": self.reference_support,
            "measurement_stability": self.measurement_stability,
        }


@dataclass(frozen=True, slots=True)
class ConfoundSets:
    """Confounds are classified by status instead of collapsed into one list."""

    potential: tuple[str, ...] = ()
    observed: tuple[str, ...] = ()
    ruled_out: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in ("potential", "observed", "ruled_out"):
            raw = getattr(self, name)
            if isinstance(raw, str):
                raw = (raw,)
            values = tuple(dict.fromkeys(str(value) for value in raw))
            object.__setattr__(self, name, values)

    def to_dict(self) -> dict[str, list[str]]:
        return {
            "potential": list(self.potential),
            "observed": list(self.observed),
            "ruled_out": list(self.ruled_out),
        }

    @property
    def potential_confound_ids(self) -> tuple[str, ...]:
        return self.potential

    @property
    def observed_confound_ids(self) -> tuple[str, ...]:
        return self.observed

    @property
    def ruled_out_confound_ids(self) -> tuple[str, ...]:
        return self.ruled_out


@dataclass(frozen=True, slots=True)
class EvidenceProvenance:
    """Trace from a measurement to source assets and the producing version."""

    source_asset_id: str = ""
    source_segment_ids: tuple[str, ...] = ()
    transcript_id: str | None = None
    method_version: str = ""
    measurement_version: str = ""
    generated_by: str = ""

    def __post_init__(self) -> None:
        raw = (self.source_segment_ids,) if isinstance(self.source_segment_ids, str) else self.source_segment_ids
        object.__setattr__(self, "source_segment_ids", tuple(dict.fromkeys(
            str(value) for value in raw
        )))

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_asset_id": self.source_asset_id,
            "source_segment_ids": list(self.source_segment_ids),
            "transcript_id": self.transcript_id,
            "method_version": self.method_version,
            "measurement_version": self.measurement_version,
            "generated_by": self.generated_by,
        }

    @property
    def segment_ids(self) -> tuple[str, ...]:
        return self.source_segment_ids


@dataclass(frozen=True, slots=True)
class EvidencePermissions:
    """Immutable, independent permissions for model inference and reporting."""

    inference: bool = True
    report: bool = False

    @property
    def inference_permission(self) -> bool:
        return self.inference

    @property
    def report_permission(self) -> bool:
        return self.report

    def to_dict(self) -> dict[str, bool]:
        return {"inference": self.inference, "report": self.report}


@dataclass(frozen=True, slots=True)
class MetricEvidenceV2:
    """Typed Stage 2 evidence contract.

    The legacy CSV builder below remains unchanged.  This object is an additive
    contract for new consumers and deliberately keeps raw value, reference,
    reliability, confounds, provenance and permissions separate.
    """

    evidence_id: str = ""
    metric_id: str = ""
    subject_id: str = ""
    session_id: str = ""
    case_id: str = ""
    metric_instance_id: str = ""
    state_id: str = ""
    task_id: str | None = None
    value: Any = None
    unit: str = ""
    source_modality: str = ""
    direction: int = 0
    direction_provenance: str = ""
    observable: bool = True
    unavailable_reason: str | None = None
    reference: ReferenceMetadata = field(default_factory=ReferenceMetadata)
    reliability_components: ReliabilityComponents = field(default_factory=ReliabilityComponents)
    provenance: EvidenceProvenance = field(default_factory=EvidenceProvenance)
    confounds: ConfoundSets = field(default_factory=ConfoundSets)
    permissions: EvidencePermissions = field(default_factory=EvidencePermissions)
    consumed_by_supervised: bool = False
    incremental_for_agent: bool = False
    reliability_migration: str = ""

    def __post_init__(self) -> None:
        if isinstance(self.direction, bool) or self.direction not in (-1, 0, 1):
            raise ValueError("Evidence direction must be exactly -1, 0, or 1.")
        if self.consumed_by_supervised and self.incremental_for_agent:
            raise ValueError(
                "Evidence cannot be both consumed_by_supervised and incremental_for_agent."
            )

    @property
    def inference_permission(self) -> bool:
        return self.permissions.inference

    @property
    def report_permission(self) -> bool:
        return self.permissions.report

    @property
    def reference_stats(self) -> ReferenceMetadata:
        return self.reference

    @property
    def observability(self) -> bool:
        return self.observable

    @property
    def reliability(self) -> ReliabilityComponents:
        """Compatibility name for callers using the contract's short form."""
        return self.reliability_components

    @property
    def source_segment_ids(self) -> tuple[str, ...]:
        return self.provenance.source_segment_ids

    @property
    def reference_sample_size(self) -> int:
        return self.reference.sample_size

    @property
    def reference_artifact_id(self) -> str:
        return self.reference.artifact_id

    @property
    def reference_artifact_hash(self) -> str:
        return self.reference.artifact_hash

    @property
    def potential_confound_ids(self) -> tuple[str, ...]:
        return self.confounds.potential

    @property
    def observed_confound_ids(self) -> tuple[str, ...]:
        return self.confounds.observed

    @property
    def ruled_out_confound_ids(self) -> tuple[str, ...]:
        return self.confounds.ruled_out

    def to_dict(self) -> dict[str, Any]:
        return {
            "evidence_id": self.evidence_id,
            "metric_id": self.metric_id,
            "subject_id": self.subject_id,
            "session_id": self.session_id,
            "case_id": self.case_id,
            "metric_instance_id": self.metric_instance_id or self.metric_id,
            "state_id": self.state_id,
            "task_id": self.task_id,
            "value": self.value,
            "unit": self.unit,
            "source_modality": self.source_modality,
            "direction": self.direction,
            "direction_provenance": self.direction_provenance,
            "observable": self.observable,
            "unavailable_reason": self.unavailable_reason,
            "reference": self.reference.to_dict(),
            "reliability_components": self.reliability_components.to_dict(),
            "provenance": self.provenance.to_dict(),
            "confounds": self.confounds.to_dict(),
            "permissions": self.permissions.to_dict(),
            "inference_permission": self.inference_permission,
            "report_permission": self.report_permission,
            "consumed_by_supervised": self.consumed_by_supervised,
            "incremental_for_agent": self.incremental_for_agent,
            "reliability_migration": self.reliability_migration,
        }

    def to_json(self) -> str:
        return _canonical_json(self.to_dict())

    @classmethod
    def from_mapping(cls, row: Mapping[str, Any]) -> "MetricEvidenceV2":
        reference = row.get("reference", {})
        if not isinstance(reference, Mapping):
            reference = {}
        reference = {
            **reference,
            **{
                key: row[key]
                for key in (
                    "reference_median", "reference_scale", "reference_sample_size",
                    "reference_artifact_id", "reference_artifact_hash", "reference_scope",
                    "reference_label",
                )
                if key in row
            },
        }
        raw_components = row.get("reliability_components", _MISSING)
        raw_legacy_reliability = row.get("reliability", _MISSING)
        reliability_migration = str(row.get("reliability_migration", ""))
        if raw_components is _MISSING:
            if raw_legacy_reliability is _MISSING:
                # Missing quality metadata must not become six perfect
                # dimensions through ReliabilityComponents' constructor.
                reliability = {"source": 0.0}
                reliability_migration = reliability_migration or "missing_fail_closed"
            elif isinstance(raw_legacy_reliability, Mapping):
                reliability = raw_legacy_reliability
            else:
                migrated, migration = _legacy_total_reliability(raw_legacy_reliability)
                reliability = migrated.to_dict()
                reliability_migration = reliability_migration or migration
        elif not isinstance(raw_components, Mapping):
            raise ValueError("reliability_components must be a mapping.")
        else:
            reliability = raw_components
        reliability_aliases = {
            "source_reliability": "source",
            "role_reliability": "role",
            "alignment_reliability": "alignment",
            "asr_reliability": "asr",
            "reference_support_reliability": "reference_support",
            "measurement_stability_score": "measurement_stability",
        }
        reliability = {
            reliability_aliases.get(str(key), str(key)): value
            for key, value in reliability.items()
            if reliability_aliases.get(str(key), str(key)) in {
                "source", "role", "alignment", "asr", "reference_support",
                "measurement_stability",
            }
        }
        if not reliability:
            raise ValueError("Reliability metadata cannot be empty.")
        provenance = row.get("provenance", {})
        if not isinstance(provenance, Mapping):
            provenance = {}
        provenance = {
            **provenance,
            "source_asset_id": row.get("source_asset_id", provenance.get("source_asset_id", "")),
            "source_segment_ids": row.get("source_segment_ids", provenance.get("source_segment_ids", ())),
            "transcript_id": row.get("transcript_id", provenance.get("transcript_id")),
            "method_version": row.get("method_version", provenance.get("method_version", "")),
            "measurement_version": row.get("measurement_version", provenance.get("measurement_version", "")),
            "generated_by": row.get("generated_by", provenance.get("generated_by", "")),
        }
        confounds = row.get("confounds", {})
        if not isinstance(confounds, Mapping):
            confounds = {}
        confounds = {
            **confounds,
            "potential": row.get("potential_confound_ids", confounds.get("potential", ())),
            "observed": row.get("observed_confound_ids", confounds.get("observed", ())),
            "ruled_out": row.get("ruled_out_confound_ids", confounds.get("ruled_out", ())),
        }
        permissions = row.get("permissions", {})
        if not isinstance(permissions, Mapping):
            permissions = {}
        return cls(
            evidence_id=str(row.get("evidence_id", "")),
            metric_id=str(row.get("metric_id", "")),
            subject_id=str(row.get("subject_id", "")),
            session_id=str(row.get("session_id", row.get("recording_id", ""))),
            case_id=str(row.get("case_id", row.get("case", ""))),
            metric_instance_id=str(row.get("metric_instance_id", row.get("metric_id", ""))),
            state_id=str(row.get("state_id", "")),
            task_id=str(row["task_id"]) if row.get("task_id") is not None else None,
            value=row.get("value"),
            unit=str(row.get("unit", "")),
            source_modality=str(row.get("source_modality", "")),
            direction=int(row.get("direction", 0)),
            direction_provenance=str(row.get("direction_provenance", "")),
            observable=_strict_bool(
                row.get("observable", row.get("observability", True)),
                field="observable",
                default=True,
            ),
            unavailable_reason=row.get("unavailable_reason"),
            reference=ReferenceMetadata.from_mapping(reference),
            reliability_components=ReliabilityComponents(**reliability),
            provenance=EvidenceProvenance(**provenance) if isinstance(provenance, Mapping) else EvidenceProvenance(),
            confounds=ConfoundSets(**confounds) if isinstance(confounds, Mapping) else ConfoundSets(),
            permissions=EvidencePermissions(
                inference=_strict_bool(
                    permissions.get("inference", row.get("inference_permission", True)),
                    field="inference_permission",
                    default=True,
                ),
                report=_strict_bool(
                    permissions.get("report", row.get("report_permission", False)),
                    field="report_permission",
                    default=False,
                ),
            ),
            consumed_by_supervised=_strict_bool(
                row.get("consumed_by_supervised", False),
                field="consumed_by_supervised",
                default=False,
            ),
            incremental_for_agent=_strict_bool(
                row.get("incremental_for_agent", False),
                field="incremental_for_agent",
                default=False,
            ),
            reliability_migration=reliability_migration,
        )


# Vocabulary aliases for downstream callers and schema documentation.
ReferenceStats = ReferenceMetadata
MetricReference = ReferenceMetadata
Reliability = ReliabilityComponents
Provenance = EvidenceProvenance


def serialize_metric_evidence_v2(evidence: MetricEvidenceV2) -> str:
    """Return the canonical JSON representation used in audit artifacts."""
    return evidence.to_json()


def deserialize_metric_evidence_v2(payload: str | bytes) -> MetricEvidenceV2:
    """Parse canonical or compatible JSON into an immutable evidence object."""
    return MetricEvidenceV2.from_mapping(json.loads(payload))


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
            values = pd.to_numeric(scoped_reference["value"], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            median, scale, variable = _robust_reference(values)
            enough = (
                int(values.notna().sum()) >= MIN_LANGUAGE_REFERENCE_SUBJECTS
                if metric_id in LANGUAGE_DEPENDENT_METRICS
                else int(values.notna().sum()) >= 2
            )
            available = bool(variable and enough)
            value = pd.to_numeric(pd.Series([row.get("value")]), errors="coerce").iloc[0]
            value_missing = bool(pd.isna(value) or not np.isfinite(value))
            missing = value_missing or not available
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
            target.at[index, "value_missing"] = value_missing
            target.at[index, "reference_available"] = available
            target.at[index, "evidence_status"] = (
                "missing" if value_missing else "unavailable" if not available else "available"
            )
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
        # The channel-filtered config defines expected evidence, not the columns
        # that happened to survive extraction. Applicability is checked per subject.
        metric_instances = [("overall", metric)]
        if use_task_specific_evidence:
            metric_instances.extend(
                (task_scope, f"task_{task_scope}__{metric}")
                for task_scope in task_scopes
            )
        for task_scope, metric_instance in metric_instances:
            reference_values = controls.get(metric_instance, pd.Series(dtype=float))
            median, scale, reference_available = _robust_reference(
                reference_values
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
                    ].get(metric_instance, pd.Series(dtype=float))
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
                value_missing = value is None or not np.isfinite(value)
                missing = value_missing or not reference_available
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
                        "value_missing": bool(value_missing),
                        "reference_available": bool(reference_available),
                        "evidence_status": (
                            "missing" if value_missing
                            else "unavailable" if not reference_available
                            else "available"
                        ),
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
