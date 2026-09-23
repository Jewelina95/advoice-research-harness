from __future__ import annotations

import importlib.util
import os
import json
import subprocess
import sys
from pathlib import Path

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/analyze_latest_evidence_agent_multidataset.py"


def test_historical_multidataset_pilot_requires_explicit_acknowledgement() -> None:
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, str(root / "scripts/analyze_latest_evidence_agent_multidataset.py")],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "Refusing to run an uncalibrated historical pilot" in result.stderr


def test_acknowledged_historical_pilot_marks_outputs_non_deployable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = importlib.util.spec_from_file_location("historical_multidataset_pilot", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = [{
        "arm": arm,
        "label": label,
        "accuracy": 0.5,
        "balanced_accuracy": 0.5,
        "macro_f1": 0.5,
        "log_loss": 0.7,
    } for arm, label in module.ARMS.items()]
    result = {
        "dataset": "PREPARE",
        "n": 1,
        "summary": rows,
        "changed": 0,
        "helped": 0,
        "harmed": 0,
        "state_gate_rate": 0.0,
        "agent_gate_rate": 0.0,
        "staging_gate_rate": 0.0,
        "conflict_rate": 0.0,
        "cases": [],
    }
    monkeypatch.setattr(module, "SPECS", {"PREPARE": (Path("a"), Path("b"))})
    monkeypatch.setattr(module, "evaluate", lambda *args: result)
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--allow-unvalidated-pilot"])
    monkeypatch.chdir(tmp_path)

    module.main()

    output = tmp_path / "reports/latest_evidence_agent_multidataset_2026-09-17"
    payload = json.loads((output / "results.json").read_text())
    assert payload["deployment_status"] == module.PILOT_STATUS
    assert payload["publication_eligible"] is False
    assert payload["calibrated_authority"] is False
    assert "不可部署的历史 pilot" in (output / "report.html").read_text()
