from __future__ import annotations

import json

import pytest

from advoice.calibration_registry import (
    CalibrationRegistryError,
    REGISTRY_SCHEMA_VERSION,
    validate_registered_calibration_artifact,
)
from advoice.decision_lock import hash_artifact


def _artifact() -> dict[str, object]:
    run = {
        "partition_role": "development_calibration",
        "test_labels_consumed": False,
        "development_subject_manifest_hash": "1" * 64,
        "oof_predictions_hash": "2" * 64,
        "acceptance_criteria_hash": "3" * 64,
        "training_code_hash": "4" * 64,
        "training_config_hash": "5" * 64,
    }
    return {
        "selection_status": "validated_joint_gain",
        "selected_screening_strength": 0.25,
        "selected_staging_strength": 0.0,
        "deployment_context_hash": "6" * 64,
        "calibration_run": run,
        "calibration_run_hash": hash_artifact(run),
    }


def test_self_declared_calibration_is_rejected_without_registry_entry(tmp_path) -> None:
    artifact = _artifact()
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "approved_artifacts": [],
    }))

    with pytest.raises(CalibrationRegistryError, match="not approved"):
        validate_registered_calibration_artifact(
            artifact,
            artifact_hash=hash_artifact(artifact),
            registry_path=registry,
        )


def test_registry_binds_artifact_run_and_deployment_context(tmp_path) -> None:
    artifact = _artifact()
    artifact_hash = hash_artifact(artifact)
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps({
        "schema_version": REGISTRY_SCHEMA_VERSION,
        "approved_artifacts": [{
            "artifact_hash": artifact_hash,
            "calibration_run_hash": artifact["calibration_run_hash"],
            "deployment_context_hash": artifact["deployment_context_hash"],
            "status": "approved_for_inference",
        }],
    }))

    validate_registered_calibration_artifact(
        artifact,
        artifact_hash=artifact_hash,
        registry_path=registry,
    )
