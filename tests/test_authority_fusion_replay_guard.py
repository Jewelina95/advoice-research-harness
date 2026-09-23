from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/replay_authority_fusion_audit.py"
SPEC = importlib.util.spec_from_file_location("replay_authority_fusion_audit", SCRIPT)
assert SPEC and SPEC.loader
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def _source(tmp_path: Path) -> Path:
    path = tmp_path / "case_audit.json"
    path.write_text(json.dumps({
        "study_hash": "a" * 64,
        "cases": [{
            "case_id": "case-1",
            "status": "completed",
            "prepared": {"class_order": ["HC", "AD"]},
            "frozen": {"probabilities": {"HC": 0.6, "AD": 0.4}},
            "pre_state": {"probabilities": {"HC": 0.6, "AD": 0.4}},
            "post_state": {"probabilities": {"HC": 0.4, "AD": 0.6}},
            "fusion": {
                "blind_ordinal_scores": {"HC": 0, "AD": 4},
                "config": {
                    "state_strength": 0.5,
                    "agent_strength": 0.0,
                    "staging_strength": 0.0,
                    "max_abs_state_delta": 0.75,
                    "ordinal_temperature": 1.0,
                    "conflict_aware_gating": True,
                    "min_state_uncertainty": 0.65,
                    "min_frozen_uncertainty": 0.75,
                    "min_counterevidence_margin": 0.75,
                },
                "audit_hash": "b" * 64,
                "probabilities": {"HC": 0.5, "AD": 0.5},
                "predicted_label": "HC",
                "provenance": {},
            },
        }],
    }))
    return path


def test_nonzero_historical_replay_requires_explicit_acknowledgement(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Refusing to replay nonzero historical authority"):
        MODULE.replay_audit(_source(tmp_path), channel="picture_description")


def test_acknowledged_replay_is_machine_marked_non_deployable(tmp_path: Path) -> None:
    result = MODULE.replay_audit(
        _source(tmp_path),
        channel="picture_description",
        allow_unvalidated_pilot=True,
    )

    assert result["deployment_status"] == MODULE.PILOT_STATUS
    assert result["publication_eligible"] is False
    assert result["calibrated_authority"] is False
