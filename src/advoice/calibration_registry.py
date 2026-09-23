"""Repository-controlled trust boundary for nonzero inference authority."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .decision_lock import hash_artifact


REGISTRY_SCHEMA_VERSION = "advoice.authority_calibration_registry.v1"
DEFAULT_REGISTRY_PATH = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "calibration"
    / "authority_registry.json"
)
_HASH_FIELDS = (
    "development_subject_manifest_hash",
    "oof_predictions_hash",
    "acceptance_criteria_hash",
    "training_code_hash",
    "training_config_hash",
)


class CalibrationRegistryError(ValueError):
    """Raised when a calibration artifact has no repository trust anchor."""


def _sha256(value: Any, *, field: str) -> str:
    text = str(value)
    if len(text) != 64 or any(character not in "0123456789abcdef" for character in text):
        raise CalibrationRegistryError(f"{field} must be a lowercase SHA-256 digest.")
    return text


def validate_registered_calibration_artifact(
    artifact: Mapping[str, Any],
    *,
    artifact_hash: str,
    registry_path: Path | None = None,
) -> None:
    """Require a version-controlled attestation before enabling nonzero authority."""

    path = (registry_path or DEFAULT_REGISTRY_PATH).expanduser().resolve()
    if not path.is_file():
        raise CalibrationRegistryError(f"Calibration registry does not exist: {path}")
    registry = json.loads(path.read_text(encoding="utf-8"))
    if registry.get("schema_version") != REGISTRY_SCHEMA_VERSION:
        raise CalibrationRegistryError("Calibration registry schema is not supported.")
    run = artifact.get("calibration_run")
    if not isinstance(run, Mapping):
        raise CalibrationRegistryError("Calibration artifact requires calibration_run provenance.")
    if run.get("partition_role") != "development_calibration":
        raise CalibrationRegistryError(
            "Calibration run must declare partition_role=development_calibration."
        )
    if run.get("test_labels_consumed") is not False:
        raise CalibrationRegistryError("Calibration run must attest test_labels_consumed=false.")
    for field in _HASH_FIELDS:
        _sha256(run.get(field), field=f"calibration_run.{field}")
    run_hash = _sha256(artifact.get("calibration_run_hash"), field="calibration_run_hash")
    if run_hash != hash_artifact(dict(run)):
        raise CalibrationRegistryError("calibration_run_hash does not match calibration_run.")
    normalized_artifact_hash = _sha256(artifact_hash, field="artifact_hash")
    context_hash = _sha256(
        artifact.get("deployment_context_hash"), field="deployment_context_hash"
    )
    entries = registry.get("approved_artifacts", [])
    if not isinstance(entries, list):
        raise CalibrationRegistryError("approved_artifacts must be a list.")
    approved = any(
        isinstance(entry, Mapping)
        and entry.get("artifact_hash") == normalized_artifact_hash
        and entry.get("calibration_run_hash") == run_hash
        and entry.get("deployment_context_hash") == context_hash
        and entry.get("status") == "approved_for_inference"
        for entry in entries
    )
    if not approved:
        raise CalibrationRegistryError(
            "Calibration artifact is not approved by the repository-controlled registry."
        )


__all__ = [
    "CalibrationRegistryError",
    "DEFAULT_REGISTRY_PATH",
    "REGISTRY_SCHEMA_VERSION",
    "validate_registered_calibration_artifact",
]
