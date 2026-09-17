"""Read-only adapter for historical Condition C prediction artifacts.

This module implements only the C0 -> C1 numerical-identity boundary.  It
does not deserialize the historical model, recompute probabilities, or claim
to support evidence replay.  The recorded prediction table is the immutable
source of probabilities; model artifacts and evidence snapshots are bound by
hash for later audit.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
import json
import math
from pathlib import Path
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd

from .module_a import EXPLANATION_PACKET_VERSION, ExplanationPacket, snapshot_hash
from .utils import hash_values, sha256_file


FROZEN_CONDITION_C_ADVISOR_VERSION = "advoice.condition_c_advisor.frozen.v1"
_PREDICTIONS_FILE = "ours_predictions.csv"
_MODEL_METADATA_FILE = "ours_model.json"
_MODEL_BINARY_FILE = "ours_model.joblib"
_REQUIRED_FILES = (_PREDICTIONS_FILE, _MODEL_METADATA_FILE, _MODEL_BINARY_FILE)
_PROBABILITY_TOLERANCE = 1e-8


class FrozenConditionCAdvisor:
    """Serve immutable historical Condition C probabilities by subject.

    The public inference boundary returns :class:`ExplanationPacket` objects
    without source labels or split assignments.  This adapter deliberately
    has no fit, update, revision, or replay operation.
    """

    def __init__(
        self,
        *,
        artifact_dir: Path,
        dataset_id: str,
        class_order: tuple[str, ...],
        probabilities_by_subject: Mapping[str, tuple[float, ...]],
        predicted_labels: Mapping[str, str],
        artifact_hashes: Mapping[str, str],
        calibration_status: str,
    ) -> None:
        self._artifact_dir = Path(artifact_dir)
        self._dataset_id = str(dataset_id)
        self._class_order = tuple(class_order)
        self._probabilities_by_subject = MappingProxyType(dict(probabilities_by_subject))
        self._predicted_labels = MappingProxyType(dict(predicted_labels))
        self._artifact_hashes = MappingProxyType(dict(artifact_hashes))
        self._calibration_status = str(calibration_status)

    @classmethod
    def from_artifact_dir(
        cls,
        artifact_dir: str | Path,
        *,
        expected_dataset_id: str | None = None,
        expected_subject_ids: Iterable[str] | None = None,
    ) -> "FrozenConditionCAdvisor":
        """Validate and load a frozen historical Condition C cohort."""

        root = Path(artifact_dir)
        paths = {name: root / name for name in _REQUIRED_FILES}
        for name, path in paths.items():
            if not path.is_file():
                raise FileNotFoundError(f"Missing required frozen Condition C artifact: {name}")

        try:
            metadata = json.loads(paths[_MODEL_METADATA_FILE].read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise ValueError("ours_model.json must contain valid UTF-8 JSON.") from exc
        if not isinstance(metadata, dict):
            raise ValueError("ours_model.json must contain a JSON object.")

        class_order = _validated_class_order(metadata.get("labels"))
        frame = pd.read_csv(
            paths[_PREDICTIONS_FILE],
            dtype={"dataset_id": "string", "subject_id": "string", "predicted_label": "string"},
        )
        dataset_id = _validated_dataset(frame, expected_dataset_id)
        subjects = _validated_subjects(frame, expected_subject_ids)
        probability_columns = [f"prob_{label}" for label in class_order]
        observed_probability_columns = [
            str(column) for column in frame.columns if str(column).startswith("prob_")
        ]
        if observed_probability_columns != probability_columns:
            raise ValueError(
                "Prediction probability columns do not match fixed class order: "
                f"expected {probability_columns}, observed {observed_probability_columns}."
            )
        if "predicted_label" not in frame.columns:
            raise ValueError("ours_predictions.csv is missing predicted_label.")

        probabilities_by_subject: dict[str, tuple[float, ...]] = {}
        predicted_labels: dict[str, str] = {}
        for row_index, row in frame.iterrows():
            subject_id = subjects[row_index]
            values = _validated_probability_vector(
                row[probability_columns].tolist(), subject_id=subject_id
            )
            predicted_label = str(row["predicted_label"])
            if predicted_label not in class_order:
                raise ValueError(
                    f"predicted_label for subject {subject_id!r} is outside fixed class order."
                )
            expected_prediction = class_order[int(np.argmax(np.asarray(values, dtype=float)))]
            if predicted_label != expected_prediction:
                raise ValueError(
                    "predicted_label does not match the recorded probability argmax for "
                    f"subject {subject_id!r}."
                )
            probabilities_by_subject[subject_id] = values
            predicted_labels[subject_id] = predicted_label

        artifact_hashes = {
            "model_joblib_sha256": sha256_file(paths[_MODEL_BINARY_FILE]),
            "model_metadata_sha256": sha256_file(paths[_MODEL_METADATA_FILE]),
            "prediction_table_sha256": sha256_file(paths[_PREDICTIONS_FILE]),
        }
        artifact_hashes["model_snapshot_sha256"] = hash_values(
            [
                FROZEN_CONDITION_C_ADVISOR_VERSION,
                artifact_hashes["model_joblib_sha256"],
                artifact_hashes["model_metadata_sha256"],
                list(class_order),
            ]
        )

        calibration_status = (
            "frozen_historical_calibrated"
            if _has_recorded_calibration(metadata)
            else "frozen_historical_calibration_unknown"
        )
        return cls(
            artifact_dir=root,
            dataset_id=dataset_id,
            class_order=class_order,
            probabilities_by_subject=probabilities_by_subject,
            predicted_labels=predicted_labels,
            artifact_hashes=artifact_hashes,
            calibration_status=calibration_status,
        )

    @property
    def class_order(self) -> tuple[str, ...]:
        return self._class_order

    @property
    def dataset_id(self) -> str:
        return self._dataset_id

    @property
    def subject_ids(self) -> tuple[str, ...]:
        return tuple(self._probabilities_by_subject)

    @property
    def artifact_hashes(self) -> Mapping[str, str]:
        return self._artifact_hashes

    def explain_subject(
        self,
        subject_id: str,
        *,
        evidence_snapshot: Any | None = None,
    ) -> ExplanationPacket:
        """Return the frozen packet for one subject without labels or split data."""

        key = str(subject_id)
        if key not in self._probabilities_by_subject:
            raise KeyError(f"Unknown subject_id for frozen Condition C cohort: {key!r}")
        values = self._probabilities_by_subject[key]
        probabilities = {
            label: values[index] for index, label in enumerate(self._class_order)
        }
        row_hash = hash_values(
            [
                FROZEN_CONDITION_C_ADVISOR_VERSION,
                self._dataset_id,
                key,
                list(self._class_order),
                list(values),
                self._predicted_labels[key],
            ]
        )
        hashes = {
            **self._artifact_hashes,
            "prediction_row_sha256": row_hash,
            "evidence_snapshot_sha256": snapshot_hash(evidence_snapshot),
        }
        entropy = -sum(value * math.log(value) for value in values if value > 0.0)
        return ExplanationPacket(
            schema_version=EXPLANATION_PACKET_VERSION,
            module_version=FROZEN_CONDITION_C_ADVISOR_VERSION,
            class_order=self._class_order,
            predicted_label=self._predicted_labels[key],
            raw_probabilities=probabilities,
            calibrated_probabilities=dict(probabilities),
            calibration_status=self._calibration_status,
            logits={},
            intercepts={},
            feature_contributions={label: {} for label in self._class_order},
            branch_contributions={label: {} for label in self._class_order},
            consumed_evidence_ids=(),
            uncertainty={
                "entropy": float(entropy),
                "max_probability": float(max(values)),
            },
            fold_disagreement=None,
            ood={
                "available": False,
                "reason": "not_recorded_in_frozen_prediction_table",
                "static_frozen": True,
                "evidence_replay_supported": False,
            },
            applicability_status="static_frozen_lookup_only",
            hashes=hashes,
        )


def _validated_class_order(value: Any) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("ours_model.json must define labels as a fixed ordered list.")
    labels = tuple(str(label).strip() for label in value)
    if len(labels) < 2 or any(not label for label in labels) or len(set(labels)) != len(labels):
        raise ValueError("ours_model.json labels must contain at least two unique non-empty labels.")
    return labels


def _validated_dataset(frame: pd.DataFrame, expected_dataset_id: str | None) -> str:
    if "dataset_id" not in frame.columns:
        raise ValueError("ours_predictions.csv is missing dataset_id.")
    if frame["dataset_id"].isna().any():
        raise ValueError("ours_predictions.csv contains missing dataset_id values.")
    dataset_ids = tuple(dict.fromkeys(str(value) for value in frame["dataset_id"].tolist()))
    if len(dataset_ids) != 1:
        raise ValueError("ours_predictions.csv must contain exactly one dataset cohort.")
    dataset_id = dataset_ids[0]
    if expected_dataset_id is not None and dataset_id != str(expected_dataset_id):
        raise ValueError(
            f"Frozen Condition C dataset cohort mismatch: expected {expected_dataset_id!r}, "
            f"observed {dataset_id!r}."
        )
    return dataset_id


def _validated_subjects(
    frame: pd.DataFrame, expected_subject_ids: Iterable[str] | None
) -> list[str]:
    if frame.empty:
        raise ValueError("ours_predictions.csv contains no subjects.")
    if "subject_id" not in frame.columns or frame["subject_id"].isna().any():
        raise ValueError("ours_predictions.csv contains missing subject_id values.")
    subjects = [str(value).strip() for value in frame["subject_id"].tolist()]
    if any(not subject for subject in subjects):
        raise ValueError("ours_predictions.csv contains empty subject_id values.")
    duplicate_mask = pd.Series(subjects).duplicated(keep=False)
    if duplicate_mask.any():
        duplicates = sorted(set(pd.Series(subjects)[duplicate_mask].tolist()))
        raise ValueError(f"Duplicate subject_id values in ours_predictions.csv: {duplicates}")
    if expected_subject_ids is not None:
        expected = {str(value) for value in expected_subject_ids}
        observed = set(subjects)
        if observed != expected:
            missing = sorted(expected - observed)
            unexpected = sorted(observed - expected)
            raise ValueError(
                "Frozen Condition C subject cohort mismatch: "
                f"missing={missing}, unexpected={unexpected}."
            )
    return subjects


def _validated_probability_vector(values: list[Any], *, subject_id: str) -> tuple[float, ...]:
    try:
        probabilities = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid probabilities for subject {subject_id!r}.") from exc
    if not all(math.isfinite(value) for value in probabilities):
        raise ValueError(f"Non-finite probabilities for subject {subject_id!r}.")
    if any(value < 0.0 or value > 1.0 for value in probabilities):
        raise ValueError(f"Out-of-range probabilities for subject {subject_id!r}.")
    if not math.isclose(sum(probabilities), 1.0, rel_tol=0.0, abs_tol=_PROBABILITY_TOLERANCE):
        raise ValueError(f"Recorded probabilities do not sum to one for subject {subject_id!r}.")
    return probabilities


def _has_recorded_calibration(metadata: Mapping[str, Any]) -> bool:
    temperature = metadata.get("final_probability_temperature")
    try:
        return math.isfinite(float(temperature)) and float(temperature) > 0.0
    except (TypeError, ValueError):
        return False


__all__ = ["FROZEN_CONDITION_C_ADVISOR_VERSION", "FrozenConditionCAdvisor"]
