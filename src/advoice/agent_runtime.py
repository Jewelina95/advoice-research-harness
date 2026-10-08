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


def _provider_schema(value: Any) -> Any:
    """Remove JSON Schema keywords rejected by strict structured-output APIs."""
    if isinstance(value, dict):
        unsupported = {"$schema", "$id", "uniqueItems"}
        return {
            key: _provider_schema(item)
            for key, item in value.items()
            if key not in unsupported
        }
    if isinstance(value, list):
        return [_provider_schema(item) for item in value]
    return value


def _codex_event_error(event: Any) -> str | None:
    if not isinstance(event, dict):
        return None
    if event.get("type") == "item.completed":
        item = event.get("item")
        if isinstance(item, dict) and item.get("type") == "error":
            return str(item.get("message") or "") or None
    if event.get("type") in {"error", "turn.failed"}:
        error = event.get("error")
        if isinstance(error, dict):
            return str(error.get("message") or "") or None
        return str(event.get("message") or "") or None
    return None


def _diagnostic_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8", errors="replace")).hexdigest()


def _provider_runtime_error(category: str) -> RuntimeError:
    return RuntimeError(f"Agent provider request failed [{category}].")


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
    provider_schema_path = schema_path.with_name(f"{schema_path.stem}.codex.json")
    json_dump(
        _provider_schema(json.loads(schema_path.read_text(encoding="utf-8"))),
        provider_schema_path,
    )
    command = [
        codex_binary,
        "exec",
        "--json",
        "--skip-git-repo-check",
        "--ephemeral",
        "--ignore-user-config",
        "--ignore-rules",
        "-c",
        "features.plugins=false",
        "-c",
        "features.skill_search=false",
        "-c",
        "features.apps=false",
        "-s",
        "read-only",
        "-m",
        model,
        "--output-schema",
        str(provider_schema_path),
        "-o",
        str(output_path),
        "-",
    ]
    started = time.monotonic()
    metadata = {"provider": "codex_cli", "model": model,
                "prompt_chars": len(prompt), "schema_bytes": schema_path.stat().st_size,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    _call_event(output_path, event="request_started", **metadata)
    environment = os.environ.copy()
    isolated_home = environment.get("ADVOICE_CODEX_HOME")
    if isolated_home:
        environment["CODEX_HOME"] = isolated_home
        # Codex also discovers user-level skills and plugins through HOME.
        # Keep clinical inference stateless and avoid unrelated context injection.
        environment["HOME"] = isolated_home
    try:
        result = subprocess.run(
            command, input=prompt, text=True, cwd=root,
            capture_output=True, timeout=1200, check=False, env=environment,
        )
    except Exception as error:
        category = "codex_transport_failed"
        _call_event(output_path, event="request_failed", **metadata,
                    elapsed_seconds=time.monotonic() - started,
                    error_type=type(error).__name__, error_category=category,
                    error_sha256=_diagnostic_hash(error), usage=None)
        raise _provider_runtime_error(category) from None
    usages = []
    provider_errors = []
    for line in result.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(event, dict) and event.get("type") == "turn.completed":
            counts = _usage_counts(event.get("usage"))
            if counts:
                usages.append(counts)
        provider_error = _codex_event_error(event)
        if provider_error:
            provider_errors.append(provider_error)
    failure_fields: dict[str, Any] = {}
    if result.returncode != 0:
        diagnostic = " | ".join(provider_errors) or result.stderr
        failure_fields = {
            "error_category": "codex_process_failed",
            "error_sha256": _diagnostic_hash(diagnostic),
        }
    _call_event(output_path, event="request_finished", **metadata,
                elapsed_seconds=time.monotonic() - started,
                returncode=result.returncode, turn_usage=usages,
                usage_available=bool(usages), **failure_fields)
    if result.returncode != 0:
        raise _provider_runtime_error("codex_process_failed")
    return json.loads(output_path.read_text(encoding="utf-8"))


API_PROVIDERS = ("openai_api", "anthropic_api", "deepseek_api")
AGENT_PROVIDERS = ("disabled", "codex_cli", "claude_cli", *API_PROVIDERS)
_API_KEY_ENV = {
    "openai_api": "OPENAI_API_KEY",
    "anthropic_api": "ANTHROPIC_API_KEY",
    "deepseek_api": "DEEPSEEK_API_KEY",
}
_ERROR_PREFIX = {"openai_api": "openai", "anthropic_api": "anthropic", "deepseek_api": "deepseek"}
DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# One shared effort level keeps cross-provider comparisons on the same policy.
AGENT_REASONING_EFFORT = os.environ.get("ADVOICE_AGENT_EFFORT", "low")
_ANTHROPIC_UNSUPPORTED = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
                          "minItems", "maxItems", "minLength", "maxLength", "pattern"}


def _strip_keywords(value: Any, keywords: set[str]) -> Any:
    if isinstance(value, dict):
        return {key: _strip_keywords(item, keywords) for key, item in value.items() if key not in keywords}
    if isinstance(value, list):
        return [_strip_keywords(item, keywords) for item in value]
    return value


def _validate_payload(payload: Any, schema: dict[str, Any]) -> None:
    # Providers without strict server-side schemas are validated locally, so a
    # malformed answer is a recorded failure instead of a silently coerced row.
    import jsonschema

    try:
        jsonschema.validate(payload, schema)
    except jsonschema.ValidationError:
        raise _provider_runtime_error("schema_validation_failed") from None


def _openai_call(prompt: str, schema: dict[str, Any], model: str) -> tuple[str | None, Any, str]:
    from openai import OpenAI

    response = OpenAI(max_retries=0).responses.create(
        model=model,
        input=prompt,
        text={"format": {"type": "json_schema", "name": "advoice_structured_output",
                         "strict": True, "schema": schema}},
        reasoning={"effort": AGENT_REASONING_EFFORT},
        max_output_tokens=OPENAI_MAX_OUTPUT_TOKENS,
        store=False,
    )
    ok = response.status == "completed" and bool(response.output_text)
    return (response.output_text if ok else None), getattr(response, "usage", None), str(response.status)


def _anthropic_call(prompt: str, schema: dict[str, Any], model: str) -> tuple[str | None, Any, str]:
    import anthropic

    # No server-side refusal fallback: a silent switch to another model would
    # break the model identity that cross-model comparisons depend on.
    with anthropic.Anthropic(max_retries=0).messages.stream(
        model=model,
        max_tokens=16000,
        messages=[{"role": "user", "content": prompt}],
        output_config={"effort": AGENT_REASONING_EFFORT,
                       "format": {"type": "json_schema",
                                  "schema": _strip_keywords(schema, _ANTHROPIC_UNSUPPORTED)}},
    ) as stream:
        message = stream.get_final_message()
    text = next((block.text for block in message.content if block.type == "text"), None)
    ok = message.stop_reason == "end_turn" and bool(text)
    return (text if ok else None), message.usage, str(message.stop_reason)


def _deepseek_call(prompt: str, schema: dict[str, Any], model: str) -> tuple[str | None, Any, str]:
    from openai import OpenAI

    client = OpenAI(api_key=os.environ["DEEPSEEK_API_KEY"], base_url=DEEPSEEK_BASE_URL, max_retries=0)
    # DeepSeek offers JSON mode but not strict schemas; the schema goes in the
    # prompt and the result is validated locally after parsing.
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Return one JSON object that validates against this JSON schema:\n"
                                          + json.dumps(schema, ensure_ascii=False)},
            {"role": "user", "content": prompt},
        ],
        response_format={"type": "json_object"},
        max_tokens=8192,
    )
    choice = response.choices[0]
    text = choice.message.content
    ok = choice.finish_reason == "stop" and bool(text)
    return (text if ok else None), getattr(response, "usage", None), str(choice.finish_reason)


_API_CALLERS = {"openai_api": _openai_call, "anthropic_api": _anthropic_call, "deepseek_api": _deepseek_call}


def run_api_batch(
    provider: str,
    prompt: str,
    schema_path: Path,
    output_path: Path,
    model: str,
) -> dict[str, Any]:
    """Run one stateless structured-output request against a hosted model API."""

    if provider not in _API_CALLERS:
        raise ValueError(f"Unsupported API provider: {provider}")
    key_env = _API_KEY_ENV[provider]
    if not os.environ.get(key_env):
        raise RuntimeError(f"{key_env} is required for the {provider} provider.")
    prefix = _ERROR_PREFIX[provider]
    schema = _provider_schema(json.loads(schema_path.read_text(encoding="utf-8")))
    metadata = {"provider": provider, "model": model, "effort": AGENT_REASONING_EFFORT,
                "prompt_chars": len(prompt), "schema_bytes": len(json.dumps(schema).encode()),
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    # Retry only here: SDK retries are disabled so attempts are counted once.
    result = None
    for attempt, delay_seconds in enumerate(OPENAI_RETRY_DELAYS_SECONDS, start=1):
        if delay_seconds:
            time.sleep(delay_seconds)
        started = time.monotonic()
        _call_event(output_path, event="request_started", attempt=attempt, **metadata)
        try:
            result = _API_CALLERS[provider](prompt, schema, model)
            usage = _usage_counts(result[1])
            _call_event(output_path, event="request_finished", attempt=attempt, **metadata,
                        elapsed_seconds=time.monotonic() - started,
                        status=result[2], usage=usage, usage_available=usage is not None)
            break
        except Exception as error:
            status_code = getattr(error, "status_code", None)
            error_name = type(error).__name__
            category = f"{prefix}_request_failed"
            _call_event(output_path, event="request_failed", attempt=attempt, **metadata,
                        elapsed_seconds=time.monotonic() - started,
                        error_type=error_name, status_code=status_code,
                        error_category=category,
                        error_sha256=_diagnostic_hash(error), usage=None)
            transient_transport = error_name in {
                "APIConnectionError", "APITimeoutError", "TimeoutError",
            }
            transient_status = status_code in {408, 409, 429} or (
                isinstance(status_code, int) and 500 <= status_code < 600
            )
            if not (transient_transport or transient_status) or attempt == len(OPENAI_RETRY_DELAYS_SECONDS):
                raise _provider_runtime_error(category) from None
    if result is None:
        raise RuntimeError("Agent API request exhausted its retries.")
    text, _, status = result
    if text is None:
        category = f"{prefix}_response_failed"
        _call_event(output_path, event="response_rejected", **metadata, status=status,
                    error_category=category, error_sha256=_diagnostic_hash(status), usage=None)
        raise _provider_runtime_error(category)
    raw_path = output_path.with_name(f"{output_path.name}.raw.txt")
    raw_path.write_text(text, encoding="utf-8")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        raise _provider_runtime_error(f"{prefix}_invalid_json") from None
    _validate_payload(payload, schema)
    json_dump(payload, output_path)
    return payload


CLAUDE_CLI_TIMEOUT_SECONDS = 600


def run_claude_cli_batch(prompt: str, schema_path: Path, output_path: Path, model: str) -> dict[str, Any]:
    """Structured call through the local Claude Code CLI (subscription login, no API key).

    Tools, MCP servers, settings files and session persistence are disabled and the
    process runs in an empty directory, so the call is a single stateless completion.
    """

    binary = shutil.which("claude")
    if binary is None:
        raise RuntimeError("Claude Code CLI was not found on PATH.")
    schema = _provider_schema(json.loads(schema_path.read_text(encoding="utf-8")))
    metadata = {"provider": "claude_cli", "model": model, "effort": AGENT_REASONING_EFFORT,
                "prompt_chars": len(prompt), "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()}
    command = [binary, "-p", "--model", model, "--effort", AGENT_REASONING_EFFORT,
               "--output-format", "json", "--json-schema", json.dumps(schema),
               "--tools", "", "--strict-mcp-config", "--setting-sources", "",
               "--no-session-persistence",
               "--system-prompt", "You are a measurement instrument. Answer only via the required structured output."]
    sandbox = output_path.parent / ".claude_cli_cwd"
    sandbox.mkdir(parents=True, exist_ok=True)
    for attempt, delay_seconds in enumerate(OPENAI_RETRY_DELAYS_SECONDS, start=1):
        if delay_seconds:
            time.sleep(delay_seconds)
        started = time.monotonic()
        _call_event(output_path, event="request_started", attempt=attempt, **metadata)
        try:
            result = subprocess.run(command, input=prompt, capture_output=True, text=True, cwd=sandbox,
                                    timeout=CLAUDE_CLI_TIMEOUT_SECONDS, check=False)
            envelope = json.loads(result.stdout)
        except (subprocess.TimeoutExpired, json.JSONDecodeError) as error:
            _call_event(output_path, event="request_failed", attempt=attempt, **metadata,
                        elapsed_seconds=time.monotonic() - started, error_type=type(error).__name__,
                        error_category="claude_cli_transport", usage=None)
            if attempt == len(OPENAI_RETRY_DELAYS_SECONDS):
                raise _provider_runtime_error("claude_cli_transport") from None
            continue
        usage = _usage_counts(envelope.get("usage"))
        _call_event(output_path, event="request_finished", attempt=attempt, **metadata,
                    elapsed_seconds=time.monotonic() - started, status=envelope.get("subtype"),
                    usage=usage, usage_available=usage is not None,
                    reported_cost_usd=envelope.get("total_cost_usd"))
        payload = envelope.get("structured_output")
        if not envelope.get("is_error") and isinstance(payload, dict):
            break
        message = str(envelope.get("result") or "")
        if "authenticate" in message.lower() or "login" in message.lower():
            raise RuntimeError("Claude Code CLI is not logged in; run `claude` and /login once.")
        transient = any(word in message.lower() for word in ("rate", "overloaded", "timeout", "529", "503"))
        if not transient or attempt == len(OPENAI_RETRY_DELAYS_SECONDS):
            raise _provider_runtime_error("claude_cli_response_failed")
    _validate_payload(payload, schema)
    json_dump(payload, output_path)
    return payload


def run_openai_batch(prompt: str, schema_path: Path, output_path: Path, model: str) -> dict[str, Any]:
    return run_api_batch("openai_api", prompt, schema_path, output_path, model)


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
    if provider == "claude_cli":
        return run_claude_cli_batch(prompt, schema_path, output_path, model)
    if provider == "openai_api":
        return run_openai_batch(prompt, schema_path, output_path, model)
    if provider in API_PROVIDERS:
        return run_api_batch(provider, prompt, schema_path, output_path, model)
    raise ValueError(f"Unsupported agent provider: {provider}")


def normalize_probabilities(values: dict[str, float], labels: list[str]) -> dict[str, float]:
    array = np.array([max(float(values.get(label, 0.0)), 0.0) for label in labels])
    if not np.isfinite(array).all() or array.sum() <= 0:
        array = np.ones(len(labels)) / len(labels)
    else:
        array /= array.sum()
    return dict(zip(labels, array.tolist(), strict=True))
