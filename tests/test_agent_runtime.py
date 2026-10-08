import json
import sys
import types

import pytest

from advoice.agent_runtime import run_codex_batch, run_openai_batch, run_structured_batch


class _Response:
    status = "completed"
    output_text = '{"action":"finish"}'
    error = None
    usage = {"input_tokens": 120, "output_tokens": 30, "total_tokens": 150,
             "input_tokens_details": {"cached_tokens": 100},
             "output_tokens_details": {"reasoning_tokens": 10}}


class _TransientError(Exception):
    pass


_TransientError.__name__ = "APIConnectionError"


def _install_openai(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> None:
    class _Responses:
        def create(self, **kwargs: object) -> object:
            assert kwargs["max_output_tokens"] == 4096
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class _OpenAI:
        def __init__(self, *, max_retries: int) -> None:
            assert max_retries == 0
            self.responses = _Responses()

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))


def test_openai_batch_retries_transient_connection_failure(monkeypatch, tmp_path) -> None:
    _install_openai(monkeypatch, [_TransientError("temporary"), _Response()])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("advoice.agent_runtime.time.sleep", lambda _: None)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    result = run_openai_batch("prompt", schema, tmp_path / "output.json", "test-model")

    assert result == {"action": "finish"}
    events = [json.loads(line) for line in (tmp_path / "output.json.calls.jsonl").read_text().splitlines()]
    assert [e["event"] for e in events] == [
        "request_started", "request_failed", "request_started", "request_finished",
    ]
    assert events[-1]["attempt"] == 2
    assert events[-1]["usage"]["input_tokens_details"]["cached_tokens"] == 100
    assert events[-1]["usage"]["output_tokens_details"]["reasoning_tokens"] == 10
    assert "test-key" not in json.dumps(events)
    assert all("prompt" not in event for event in events)


def test_openai_batch_does_not_retry_nontransient_request_error(monkeypatch, tmp_path) -> None:
    class BadRequestError(Exception):
        status_code = 400

    sensitive = "invalid schema echoed patient Alice and private prompt"
    outcomes: list[object] = [BadRequestError(sensitive), _Response()]
    _install_openai(monkeypatch, outcomes)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("advoice.agent_runtime.time.sleep", lambda _: None)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    with pytest.raises(RuntimeError, match=r"\[openai_request_failed\]") as captured:
        run_openai_batch("prompt", schema, tmp_path / "output.json", "test-model")
    assert len(outcomes) == 1
    ledger = (tmp_path / "output.json.calls.jsonl").read_text(encoding="utf-8")
    event = json.loads(ledger.splitlines()[-1])
    assert sensitive not in str(captured.value)
    assert sensitive not in ledger
    assert event["error_category"] == "openai_request_failed"
    assert len(event["error_sha256"]) == 64


def test_structured_batch_reuses_hash_bound_output_without_provider_call(
    monkeypatch, tmp_path,
) -> None:
    output = tmp_path / "request-hash.output.json"
    output.write_text('{"action":"cached"}', encoding="utf-8")
    monkeypatch.setattr(
        "advoice.agent_runtime.run_openai_batch",
        lambda *args: (_ for _ in ()).throw(AssertionError("provider called")),
    )

    result = run_structured_batch(
        tmp_path, "prompt", tmp_path / "schema.json", output, "model", "openai_api"
    )

    assert result == {"action": "cached"}
    event = json.loads((tmp_path / "request-hash.output.json.calls.jsonl").read_text())
    assert event["event"] == "cache_hit" and event["provider_calls"] == 0


@pytest.mark.parametrize("contents", ["not json", "[]"])
def test_structured_batch_rejects_corrupt_cached_output(tmp_path, contents) -> None:
    output = tmp_path / "request-hash.output.json"
    output.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cached structured output"):
        run_structured_batch(
            tmp_path, "prompt", tmp_path / "schema.json", output, "model", "openai_api"
        )


def test_codex_records_usage_without_recording_conversation(monkeypatch, tmp_path):
    output = tmp_path / "output.json"
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}')
    monkeypatch.setattr("advoice.agent_runtime.shutil.which", lambda _: "codex")

    def fake_run(command, **kwargs):
        assert "--json" in command
        assert "features.plugins=false" in command
        assert "features.skill_search=false" in command
        assert "features.apps=false" in command
        output.write_text('{"action":"finish"}')
        return types.SimpleNamespace(returncode=0, stderr="", stdout='\n'.join([
            'not a json event',
            json.dumps({"type": "item.completed", "text": "private patient text"}),
            json.dumps({"type": "turn.completed", "usage": {
                "input_tokens": 100, "cached_input_tokens": 20, "output_tokens": 10,
            }}),
        ]))

    monkeypatch.setattr("advoice.agent_runtime.subprocess.run", fake_run)
    assert run_codex_batch(tmp_path, "private prompt", schema, output, "test") == {"action": "finish"}
    ledger = (tmp_path / "output.json.calls.jsonl").read_text()
    event = json.loads(ledger.splitlines()[-1])
    assert event["turn_usage"][0]["cached_input_tokens"] == 20
    assert "private" not in ledger


def test_codex_uses_isolated_home_and_redacts_json_error(monkeypatch, tmp_path):
    output = tmp_path / "output.json"
    schema = tmp_path / "schema.json"
    schema.write_text('{"type":"object"}')
    isolated = tmp_path / "codex-home"
    monkeypatch.setenv("ADVOICE_CODEX_HOME", str(isolated))
    monkeypatch.setattr("advoice.agent_runtime.shutil.which", lambda _: "codex")

    sensitive = "structured output rejected for patient Alice private prompt"

    def fake_run(command, **kwargs):
        assert kwargs["env"]["CODEX_HOME"] == str(isolated)
        assert kwargs["env"]["HOME"] == str(isolated)
        return types.SimpleNamespace(
            returncode=1,
            stderr="",
            stdout=json.dumps({
                "type": "turn.failed",
                "error": {"message": sensitive},
            }),
        )

    monkeypatch.setattr("advoice.agent_runtime.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match=r"\[codex_process_failed\]") as captured:
        run_codex_batch(tmp_path, "prompt", schema, output, "test")
    ledger = (tmp_path / "output.json.calls.jsonl").read_text(encoding="utf-8")
    event = json.loads(ledger.splitlines()[-1])
    assert sensitive not in str(captured.value)
    assert sensitive not in ledger
    assert event["error_category"] == "codex_process_failed"
    assert len(event["error_sha256"]) == 64


def test_codex_strips_unsupported_schema_keywords(monkeypatch, tmp_path):
    output = tmp_path / "output.json"
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "array",
        "uniqueItems": True,
        "items": {"type": "string"},
    }))
    monkeypatch.setattr("advoice.agent_runtime.shutil.which", lambda _: "codex")

    def fake_run(command, **kwargs):
        provider_schema = json.loads(open(command[command.index("--output-schema") + 1]).read())
        assert "$schema" not in provider_schema
        assert "uniqueItems" not in provider_schema
        output.write_text('{"action":"finish"}')
        return types.SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr("advoice.agent_runtime.subprocess.run", fake_run)
    run_codex_batch(tmp_path, "prompt", schema, output, "test")


def test_codex_timeout_is_recorded_as_unknown_usage(monkeypatch, tmp_path):
    import subprocess

    schema = tmp_path / "schema.json"
    schema.write_text('{}')
    monkeypatch.setattr("advoice.agent_runtime.shutil.which", lambda _: "codex")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 1200)

    monkeypatch.setattr("advoice.agent_runtime.subprocess.run", timeout)
    with pytest.raises(RuntimeError, match=r"\[codex_transport_failed\]"):
        run_codex_batch(tmp_path, "prompt", schema, tmp_path / "out.json", "test")
    event = json.loads((tmp_path / "out.json.calls.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "request_failed" and event["usage"] is None
    assert event["error_category"] == "codex_transport_failed"


def test_openai_missing_usage_is_not_reported_as_zero(monkeypatch, tmp_path):
    response = _Response()
    response.usage = None
    _install_openai(monkeypatch, [response])
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    schema = tmp_path / "schema.json"
    schema.write_text('{}')
    run_openai_batch("prompt", schema, tmp_path / "out.json", "test")
    event = json.loads((tmp_path / "out.json.calls.jsonl").read_text().splitlines()[-1])
    assert event["usage"] is None and event["usage_available"] is False


_ACTION_SCHEMA = {"type": "object", "properties": {"action": {"type": "string", "minLength": 1}},
                  "required": ["action"], "additionalProperties": False}


def _install_anthropic(monkeypatch: pytest.MonkeyPatch, text: str, stop_reason: str, seen: dict) -> None:
    message = types.SimpleNamespace(
        content=[types.SimpleNamespace(type="thinking", thinking=""), types.SimpleNamespace(type="text", text=text)],
        stop_reason=stop_reason, usage={"input_tokens": 50, "output_tokens": 9})

    class _Stream:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def get_final_message(self):
            return message

    class _Messages:
        def stream(self, **kwargs):
            seen.update(kwargs)
            return _Stream()

    class _Anthropic:
        def __init__(self, *, max_retries: int) -> None:
            assert max_retries == 0
            self.messages = _Messages()

    monkeypatch.setitem(sys.modules, "anthropic", types.SimpleNamespace(Anthropic=_Anthropic))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")


def test_anthropic_batch_uses_structured_output_without_fallback(monkeypatch, tmp_path) -> None:
    from advoice.agent_runtime import run_structured_batch

    seen: dict = {}
    _install_anthropic(monkeypatch, '{"action":"finish"}', "end_turn", seen)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_ACTION_SCHEMA))
    result = run_structured_batch(tmp_path, "prompt", schema, tmp_path / "out.json", "claude-opus-5-5", "anthropic_api")
    assert result == {"action": "finish"}
    fmt = seen["output_config"]["format"]
    assert fmt["type"] == "json_schema" and "minLength" not in json.dumps(fmt["schema"])
    assert "fallbacks" not in seen and "betas" not in seen
    events = [json.loads(line) for line in (tmp_path / "out.json.calls.jsonl").read_text().splitlines()]
    assert events[-1]["usage"] == {"input_tokens": 50, "output_tokens": 9}


def test_anthropic_refusal_is_a_recorded_failure(monkeypatch, tmp_path) -> None:
    from advoice.agent_runtime import run_structured_batch

    _install_anthropic(monkeypatch, "", "refusal", {})
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_ACTION_SCHEMA))
    with pytest.raises(RuntimeError, match=r"\[anthropic_response_failed\]"):
        run_structured_batch(tmp_path, "p", schema, tmp_path / "out.json", "claude-opus-5-5", "anthropic_api")
    assert not (tmp_path / "out.json").exists()


def _install_deepseek(monkeypatch: pytest.MonkeyPatch, content: str, seen: dict) -> None:
    class _Completions:
        def create(self, **kwargs):
            seen.update(kwargs)
            choice = types.SimpleNamespace(finish_reason="stop", message=types.SimpleNamespace(content=content))
            return types.SimpleNamespace(choices=[choice], usage={"prompt_tokens": 40, "completion_tokens": 7})

    class _OpenAI:
        def __init__(self, *, api_key: str, base_url: str, max_retries: int) -> None:
            assert base_url == "https://api.deepseek.com" and max_retries == 0
            self.chat = types.SimpleNamespace(completions=_Completions())

    monkeypatch.setitem(sys.modules, "openai", types.SimpleNamespace(OpenAI=_OpenAI))
    monkeypatch.setenv("DEEPSEEK_API_KEY", "test")


def test_deepseek_batch_validates_json_mode_output_locally(monkeypatch, tmp_path) -> None:
    from advoice.agent_runtime import run_structured_batch

    seen: dict = {}
    _install_deepseek(monkeypatch, '{"action":"finish"}', seen)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_ACTION_SCHEMA))
    assert run_structured_batch(tmp_path, "p", schema, tmp_path / "out.json", "deepseek-chat", "deepseek_api") == {"action": "finish"}
    assert seen["response_format"] == {"type": "json_object"}
    assert '"action"' in seen["messages"][0]["content"]


def test_deepseek_schema_violation_fails_closed(monkeypatch, tmp_path) -> None:
    from advoice.agent_runtime import run_structured_batch

    _install_deepseek(monkeypatch, '{"action":"finish","extra":1}', {})
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_ACTION_SCHEMA))
    with pytest.raises(RuntimeError, match=r"\[schema_validation_failed\]"):
        run_structured_batch(tmp_path, "p", schema, tmp_path / "out.json", "deepseek-chat", "deepseek_api")
    assert not (tmp_path / "out.json").exists()


def test_missing_provider_key_fails_before_any_call(monkeypatch, tmp_path) -> None:
    from advoice.agent_runtime import run_api_batch

    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_ACTION_SCHEMA))
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        run_api_batch("deepseek_api", "p", schema, tmp_path / "out.json", "deepseek-chat")
    assert not (tmp_path / "out.json.calls.jsonl").exists()


def test_claude_cli_batch_reads_structured_output_and_cost(monkeypatch, tmp_path) -> None:
    from advoice import agent_runtime

    seen: dict = {}

    def fake_run(command, **kwargs):
        seen["command"] = command
        seen["cwd"] = kwargs["cwd"]
        envelope = {"type": "result", "subtype": "success", "is_error": False,
                    "structured_output": {"action": "finish"}, "total_cost_usd": 0.01,
                    "usage": {"input_tokens": 30, "output_tokens": 5}}
        return types.SimpleNamespace(stdout=json.dumps(envelope), returncode=0)

    monkeypatch.setattr("advoice.agent_runtime.shutil.which", lambda _: "claude")
    monkeypatch.setattr("advoice.agent_runtime.subprocess.run", fake_run)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_ACTION_SCHEMA))
    out = tmp_path / "out.json"
    assert agent_runtime.run_structured_batch(tmp_path, "p", schema, out, "claude-opus-5-5", "claude_cli") == {"action": "finish"}
    command = seen["command"]
    assert command[command.index("--tools") + 1] == "" and "--no-session-persistence" in command
    assert seen["cwd"].name == ".claude_cli_cwd"
    event = json.loads((tmp_path / "out.json.calls.jsonl").read_text().splitlines()[-1])
    assert event["reported_cost_usd"] == 0.01


def test_claude_cli_login_error_is_explicit(monkeypatch, tmp_path) -> None:
    from advoice import agent_runtime

    envelope = {"is_error": True, "result": "Failed to authenticate: OAuth session expired", "structured_output": None}
    monkeypatch.setattr("advoice.agent_runtime.shutil.which", lambda _: "claude")
    monkeypatch.setattr("advoice.agent_runtime.subprocess.run",
                        lambda *a, **k: types.SimpleNamespace(stdout=json.dumps(envelope), returncode=0))
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps(_ACTION_SCHEMA))
    with pytest.raises(RuntimeError, match="not logged in"):
        agent_runtime.run_claude_cli_batch("p", schema, tmp_path / "out.json", "claude-opus-5-5")
