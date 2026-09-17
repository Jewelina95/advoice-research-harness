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

    outcomes: list[object] = [BadRequestError("invalid schema"), _Response()]
    _install_openai(monkeypatch, outcomes)
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setattr("advoice.agent_runtime.time.sleep", lambda _: None)
    schema = tmp_path / "schema.json"
    schema.write_text(json.dumps({"type": "object"}), encoding="utf-8")

    with pytest.raises(BadRequestError, match="invalid schema"):
        run_openai_batch("prompt", schema, tmp_path / "output.json", "test-model")
    assert len(outcomes) == 1


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


def test_codex_timeout_is_recorded_as_unknown_usage(monkeypatch, tmp_path):
    import subprocess

    schema = tmp_path / "schema.json"
    schema.write_text('{}')
    monkeypatch.setattr("advoice.agent_runtime.shutil.which", lambda _: "codex")

    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired("codex", 1200)

    monkeypatch.setattr("advoice.agent_runtime.subprocess.run", timeout)
    with pytest.raises(subprocess.TimeoutExpired):
        run_codex_batch(tmp_path, "prompt", schema, tmp_path / "out.json", "test")
    event = json.loads((tmp_path / "out.json.calls.jsonl").read_text().splitlines()[-1])
    assert event["event"] == "request_failed" and event["usage"] is None


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
