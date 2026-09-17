from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import advoice.authority_state_delta_cli as cli


def _args(tmp_path: Path, **overrides: object) -> object:
    values = {
        "artifact_dir": tmp_path / "artifact",
        "output_dir": tmp_path / "output",
        "cache_dir": tmp_path / "cache",
        "provider": "openai_api",
        "model": "gpt-test",
        "max_cases": 2,
        "selection_order": "longest_first",
        "alpha": 0.25,
        "max_abs_delta": 0.75,
        "skill_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _result(tmp_path: Path, *, failed: tuple[str, ...] = ()) -> SimpleNamespace:
    return SimpleNamespace(
        study_hash="a" * 64,
        attempted_case_ids=("one", "two"),
        completed_case_ids=tuple(case_id for case_id in ("one", "two") if case_id not in failed),
        failed_case_ids=failed,
        frozen_metrics={"accuracy": 0.5},
        fused_metrics={"accuracy": 0.75},
        paired_counts={"changed": 1, "unchanged": 1, "corrected": 1, "harmed": 0},
        audit_jsonl_path=tmp_path / "case_audit.jsonl",
        audit_json_path=tmp_path / "case_audit.json",
        aggregate_json_path=tmp_path / "aggregate.json",
    )


def test_parser_requires_provider_and_model() -> None:
    parser = cli.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([
            "--artifact-dir", "artifact", "--output-dir", "output", "--cache-dir", "cache",
        ])


def test_run_passes_all_controls_without_provider_fallback(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    calls: dict[str, object] = {}

    class Dataset:
        pass

    class Runtime:
        pass

    def load_dataset(path: Path) -> Dataset:
        calls["artifact"] = path
        return Dataset()

    def build_runtime(**kwargs: object) -> Runtime:
        calls["runtime"] = kwargs
        return Runtime()

    def run_study(dataset: object, runtime: object, **kwargs: object) -> SimpleNamespace:
        calls["study"] = (dataset, runtime, kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(cli.AuthorityStudyDataset, "from_artifact_dir", load_dataset)
    monkeypatch.setattr(cli, "build_authority_review_runtime", build_runtime)
    monkeypatch.setattr(cli, "run_authority_state_delta_cohort", run_study)

    status, summary = cli.run(_args(tmp_path, provider="codex_cli", model="explicit-model"))

    assert status == 0
    assert summary["provider"] == "codex_cli"
    assert summary["model"] == "explicit-model"
    assert calls["runtime"]["provider"] == "codex_cli"  # type: ignore[index]
    assert calls["runtime"]["model"] == "explicit-model"  # type: ignore[index]
    study_kwargs = calls["study"][2]  # type: ignore[index]
    assert study_kwargs["cache_dir"] == (tmp_path / "cache").resolve()  # type: ignore[index]
    assert study_kwargs["config"].selection_order == "longest_first"  # type: ignore[index]
    assert study_kwargs["config"].delta_fusion.alpha == 0.25  # type: ignore[index]


def test_failed_cases_return_nonzero_and_keep_json_summary(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(cli.AuthorityStudyDataset, "from_artifact_dir", lambda path: object())
    monkeypatch.setattr(cli, "build_authority_review_runtime", lambda **kwargs: object())
    monkeypatch.setattr(cli, "run_authority_state_delta_cohort", lambda *args, **kwargs: _result(tmp_path, failed=("two",)))

    status, summary = cli.run(_args(tmp_path))

    assert status == 1
    assert summary["status"] == "failed"
    assert summary["failed_cases"] == 1


def test_main_emits_compact_json_on_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    monkeypatch.setattr(cli, "run", lambda args: (0, {"status": "completed", "attempted_cases": 1}))

    status = cli.main([
        "--artifact-dir", str(artifact), "--output-dir", str(tmp_path / "out"),
        "--cache-dir", str(tmp_path / "cache"), "--provider", "disabled", "--model", "test",
    ])

    assert status == 0
    assert json.loads(capsys.readouterr().out) == {"status": "completed", "attempted_cases": 1}


def test_main_emits_json_error_and_nonzero_for_invalid_controls(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()

    status = cli.main([
        "--artifact-dir", str(artifact), "--output-dir", str(tmp_path / "out"),
        "--cache-dir", str(tmp_path / "cache"), "--provider", "openai_api", "--model", "",
    ])

    assert status == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "failed"
    assert "--model" in payload["error"]
