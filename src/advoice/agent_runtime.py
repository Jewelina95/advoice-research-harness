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


OPENAI_RETRY_DELAYS_SECONDS = (0, 20, 60)
OPENAI_MAX_OUTPUT_TOKENS = 4096


def _usage_counts(value: Any) -> dict[str, Any] | None:
    if hasattr(value, "model_dump"):
        value = value.model_dump()
    if not isinstance(value, dict):
        return None
    counts: dict[str, Any] = {}
    for key, item in value.items():
        if "token" in key and isinstance(item, int) and not isinstance(item, bool):
            counts[key] = item
        elif key.endswith("_details"):
            nested = _usage_counts(item)
            if nested:
                counts[key] = nested
    return counts or None


def _call_event(output: Path, **fields: Any) -> None:
    # Store counters and request identity, never prompts, transcripts or errors.
    path = output.with_name(output.name + ".calls.jsonl")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"timestamp": time.time(), **fields}) + "\n")


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
        "--json",
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
    started = time.monotonic()
    metadata = {"provider": "codex_cli", "model": model,
                "prompt_chars": len(prompt), "schema_bytes": schema_path.stat().st_size,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    _call_event(output_path, event="request_started", **metadata)
    try:
        result = subprocess.run(
            command, input=prompt, text=True, cwd=root,
            capture_output=True, timeout=1200, check=False,
        )
    except Exception as error:
        _call_event(output_path, event="request_failed", **metadata,
                    elapsed_seconds=time.monotonic() - started,
                    error_type=type(error).__name__, usage=None)
        raise
    usages = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed":
            counts = _usage_counts(event.get("usage"))
            if counts:
                usages.append(counts)
    _call_event(output_path, event="request_finished", **metadata,
                elapsed_seconds=time.monotonic() - started,
                returncode=result.returncode, turn_usage=usages,
                usage_available=bool(usages))
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
    # Retry only here: SDK retries multiplied the five outer attempts by three.
    client = OpenAI(max_retries=0)
    response = None
    retry_delays = OPENAI_RETRY_DELAYS_SECONDS
    metadata = {"provider": "openai_api", "model": model,
                "prompt_chars": len(prompt), "schema_bytes": len(json.dumps(schema).encode()),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    for attempt, delay_seconds in enumerate(retry_delays, start=1):
        if delay_seconds:
            time.sleep(delay_seconds)
        started = time.monotonic()
        _call_event(output_path, event="request_started", attempt=attempt, **metadata)
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
                max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
                store=False,
            )
            usage = _usage_counts(getattr(response, "usage", None))
            _call_event(output_path, event="request_finished", attempt=attempt, **metadata,
                        elapsed_seconds=time.monotonic() - started,
                        status=response.status, usage=usage, usage_available=usage is not None)
            break
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            error_name = type(error).__name__
            _call_event(output_path, event="request_failed", attempt=attempt, **metadata,
                        elapsed_seconds=time.monotonic() - started,
                        error_type=error_name, status_code=status_code, usage=None)
            transient_transport = error_name in {
                "APIConnectionError", "APITimeoutError", "TimeoutError",
            }
            transient_status = status_code in {408, 409, 429} or (
                isinstance(status_code, int) and 500 <= status_code < 600
            )
            if not (transient_transport or transient_status) or attempt == len(retry_delays):
                raise
    if response is None:
        raise RuntimeError("OpenAI agent request exhausted its rate-limit retries.")
    if response.status != "completed" or not response.output_text:
        raise RuntimeError(
            f"OpenAI agent failed with status={response.status}: {response.error}"
        )
    raw_path = output_path.with_name(f"{output_path.name}.raw.txt")
    raw_path.write_text(response.output_text, encoding="utf-8")
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
    # The caller binds output_path to a request hash. Reusing that exact path
    # makes long external-provider studies resumable without silently issuing
    # the same paid request twice. Corrupt cache entries fail closed.
    if output_path.is_file():
        try:
            cached = json.loads(output_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise RuntimeError(f"Cached structured output is invalid: {output_path}") from exc
        if not isinstance(cached, dict):
            raise RuntimeError(f"Cached structured output must be a JSON object: {output_path}")
        _call_event(output_path, event="cache_hit", provider=provider, model=model,
                    provider_calls=0)
        return cached
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
