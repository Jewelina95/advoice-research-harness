import json
import sys
import types

import pytest

from advoice.agent_runtime import run_openai_batch, run_structured_batch


class _Response:
    status = "completed"
    output_text = '{"action":"finish"}'
    error = None


class _TransientError(Exception):
    pass


_TransientError.__name__ = "APIConnectionError"


def _install_openai(monkeypatch: pytest.MonkeyPatch, outcomes: list[object]) -> None:
    class _Responses:
        def create(self, **_: object) -> object:
            outcome = outcomes.pop(0)
            if isinstance(outcome, BaseException):
                raise outcome
            return outcome

    class _OpenAI:
        def __init__(self) -> None:
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


@pytest.mark.parametrize("contents", ["not json", "[]"])
def test_structured_batch_rejects_corrupt_cached_output(tmp_path, contents) -> None:
    output = tmp_path / "request-hash.output.json"
    output.write_text(contents, encoding="utf-8")

    with pytest.raises(RuntimeError, match="Cached structured output"):
        run_structured_batch(
            tmp_path, "prompt", tmp_path / "schema.json", output, "model", "openai_api"
        )
