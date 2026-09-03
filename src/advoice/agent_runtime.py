from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .utils import json_dump


def pseudonym(value: str, prefix: str = "P") -> str:
    return prefix + "-" + hashlib.sha256(value.encode()).hexdigest()[:10].upper()


def case_pseudonym(value: str) -> str:
    digest = hashlib.sha256(f"advoice-8.27::{value}".encode("utf-8")).hexdigest()[:12]
    return f"case_{digest}"


def select_agent_cohort(truth: pd.DataFrame, cap: int) -> pd.DataFrame:
    unique = truth.drop_duplicates("subject_id").copy()
    if cap >= len(unique):
        return unique
    # Held-out labels cannot influence which cases receive an Agent call.
    return (
        unique.assign(_order=unique["subject_id"].astype(str).map(pseudonym))
        .sort_values("_order")
        .head(cap)
        .drop(columns="_order")
        .reset_index(drop=True)
    )


def output_schema(path: Path, labels: list[str], include_probability: bool = True) -> None:
    case_properties: dict[str, Any] = {
        "case_id": {"type": "string"},
        "predicted_label": {"type": "string", "enum": labels},
        "report_zh": {"type": "string"},
        "evidence": {"type": "array", "items": {"type": "string"}},
        "uncertainty_zh": {"type": "string"},
    }
    required = list(case_properties)
    if include_probability:
        case_properties["probabilities"] = {
            "type": "object",
            "properties": {label: {"type": "number", "minimum": 0, "maximum": 1} for label in labels},
            "required": labels,
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


def run_openai_batch(
    prompt: str,
    schema_path: Path,
    output_path: Path,
    model: str,
) -> dict[str, Any]:
    """Run a stateless structured-output request through the Responses API."""

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is required for the openai_api provider.")
    from openai import OpenAI

    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    def api_schema(value: Any) -> Any:
        if isinstance(value, dict):
            unsupported = {"$schema", "$id", "uniqueItems"}
            return {
                key: api_schema(item)
                for key, item in value.items()
                if key not in unsupported
            }
        if isinstance(value, list):
            return [api_schema(item) for item in value]
        return value

    schema = api_schema(schema)
    client = OpenAI()
    response = None
    for attempt, delay_seconds in enumerate((0, 20, 40, 80, 120), start=1):
        if delay_seconds:
            time.sleep(delay_seconds)
        try:
            response = client.responses.create(
                model=model,
                input=prompt,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "advoice_structured_output",
                        "strict": True,
                        "schema": schema,
                    }
                },
                reasoning={"effort": "low"},
                max_output_tokens=12000,
                store=False,
            )
            break
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            if status_code != 429 or attempt == 5:
                raise
    if response is None:
        raise RuntimeError("OpenAI agent request exhausted its rate-limit retries.")
    if response.status != "completed" or not response.output_text:
        raise RuntimeError(
            f"OpenAI agent failed with status={response.status}: {response.error}"
        )
    payload = json.loads(response.output_text)
    json_dump(payload, output_path)
    return payload


def run_structured_batch(
    root: Path,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    model: str,
    provider: str,
) -> dict[str, Any]:
    if provider == "codex_cli":
        return run_codex_batch(root, prompt, schema_path, output_path, model)
    if provider == "openai_api":
        return run_openai_batch(prompt, schema_path, output_path, model)
    raise ValueError(f"Unsupported agent provider: {provider}")


def normalize_probabilities(values: dict[str, float], labels: list[str]) -> dict[str, float]:
    array = np.array([max(float(values.get(label, 0.0)), 0.0) for label in labels])
    if not np.isfinite(array).all() or array.sum() <= 0:
        array = np.ones(len(labels)) / len(labels)
    else:
        array /= array.sum()
    return dict(zip(labels, array.tolist(), strict=True))
