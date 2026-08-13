from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .utils import json_dump


LABELS = ["HC", "MCI", "AD"]


def pseudonym(value: str, prefix: str = "P") -> str:
    return prefix + "-" + hashlib.sha256(value.encode()).hexdigest()[:10].upper()


def output_schema(path: Path, include_probability: bool = True) -> None:
    case_properties: dict[str, Any] = {
        "case_id": {"type": "string"},
        "predicted_label": {"type": "string", "enum": LABELS},
        "report_zh": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "uncertainty_zh": {"type": "string"},
    }
    required = list(case_properties)
    if include_probability:
        case_properties["probabilities"] = {
            "type": "object",
            "properties": {label: {"type": "number", "minimum": 0, "maximum": 1} for label in LABELS},
            "required": LABELS,
            "additionalProperties": False,
        }
        required.append("probabilities")
    json_dump(
        {
            "$schema": "http://json-schema.org/draft-07/schema#",
            "type": "object",
            "properties": {
                "cases": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": case_properties,
                        "required": required,
                        "additionalProperties": False,
                    },
                }
            },
            "required": ["cases"],
            "additionalProperties": False,
        },
        path,
    )


def run_codex_batch(
    root: Path,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    model: str,
) -> dict[str, Any]:
    codex_binary = shutil.which("codex")
    if codex_binary is None:
        raise RuntimeError(
            "Codex CLI was not found on PATH. Install it or expose the codex executable before running agent stages."
        )
    command = [
        codex_binary,
        "exec",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-s",
        "read-only",
        "-m",
        model,
        "--output-schema",
        str(schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    result = subprocess.run(
        command,
        input=prompt,
        text=True,
        cwd=root,
        capture_output=True,
        timeout=1200,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(f"Codex agent failed: {result.stderr[-4000:]}")
    return json.loads(output_path.read_text(encoding="utf-8"))


def normalize_probabilities(values: dict[str, float]) -> dict[str, float]:
    array = np.array([max(float(values.get(label, 0.0)), 0.0) for label in LABELS])
    if not np.isfinite(array).all() or array.sum() <= 0:
        array = np.ones(len(LABELS)) / len(LABELS)
    else:
        array /= array.sum()
    return dict(zip(LABELS, array.tolist(), strict=True))
