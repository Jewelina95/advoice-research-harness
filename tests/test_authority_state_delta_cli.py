from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import advoice.authority_state_delta_cli as cli
from advoice.decision_lock import hash_artifact


def _args(tmp_path: Path, **overrides: object) -> object:
    values = {
        "artifact_dir": tmp_path / "artifact",
        "output_dir": tmp_path / "output",
        "cache_dir": tmp_path / "cache",
        "provider": "openai_api",
        "model": "gpt-test",
        "max_cases": 2,
        "selection_order": "longest_first",
        "selection_salt": "fixture-pilot-v1",
        "state_strength": 0.0,
        "agent_strength": 0.0,
        "staging_strength": 0.0,
        "ordinal_temperature": 1.0,
        "alpha": None,
        "max_abs_delta": 0.75,
        "skill_path": None,
        "decision_only": True,
        "review_mode": "single_blind",
        "calibration_artifact": None,
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
    assert calls["runtime"]["review_mode"] == "single_blind"  # type: ignore[index]
    study_kwargs = calls["study"][2]  # type: ignore[index]
    assert study_kwargs["cache_dir"] == (tmp_path / "cache").resolve()  # type: ignore[index]
    assert study_kwargs["config"].selection_order == "longest_first"  # type: ignore[index]
    assert study_kwargs["config"].selection_salt == "fixture-pilot-v1"  # type: ignore[index]
    assert study_kwargs["config"].joint_fusion.state_strength == 0.0  # type: ignore[index]
    assert study_kwargs["config"].joint_fusion.agent_strength == 0.0  # type: ignore[index]
    assert study_kwargs["config"].joint_fusion.staging_strength == 0.0  # type: ignore[index]
    assert study_kwargs["decision_only"] is True
    assert summary["decision_only"] is True
    assert summary["report_generation"] == "deferred"


def test_report_opt_in_is_deferred_before_loading_dataset(tmp_path, monkeypatch, capsys):
    def forbidden(*args, **kwargs):
        raise AssertionError("Report request must fail before dataset or provider work")
    monkeypatch.setattr(cli.AuthorityStudyDataset, "from_artifact_dir", forbidden)
    monkeypatch.setattr(cli, "build_authority_review_runtime", forbidden)
    status = cli.main([
        "--artifact-dir", str(tmp_path / "missing"), "--output-dir", str(tmp_path / "out"),
        "--cache-dir", str(tmp_path / "cache"), "--provider", "openai_api",
        "--model", "test", "--report",
    ])
    assert status == 1
    assert "report generation is deferred" in json.loads(capsys.readouterr().out)["error"]
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("flags", [[], ["--decision-only"], ["--report"]])
def test_parser_defaults_to_decision_only(flags):
    args = cli.build_parser().parse_args([
        "--artifact-dir", "artifact", "--output-dir", "output", "--cache-dir", "cache",
        "--provider", "disabled", "--model", "test", *flags,
    ])
    assert args.decision_only is (flags != ["--report"])
    assert args.review_mode == "single_blind"


def test_parser_keeps_legacy_two_pass_only_as_explicit_reproduction_mode():
    args = cli.build_parser().parse_args([
        "--artifact-dir", "artifact", "--output-dir", "output", "--cache-dir", "cache",
        "--provider", "disabled", "--model", "test",
        "--review-mode", "legacy_two_pass",
    ])
    assert args.review_mode == "legacy_two_pass"


def test_parser_exposes_separate_staging_strength():
    args = cli.build_parser().parse_args([
        "--artifact-dir", "artifact", "--output-dir", "output", "--cache-dir", "cache",
        "--provider", "disabled", "--model", "test", "--staging-strength", "0.25",
    ])
    assert args.staging_strength == 0.25


def test_parser_fails_closed_before_development_calibration():
    args = cli.build_parser().parse_args([
        "--artifact-dir", "artifact", "--output-dir", "output", "--cache-dir", "cache",
        "--provider", "disabled", "--model", "test",
    ])
    assert args.state_strength == 0.0
    assert args.agent_strength == 0.0
    assert args.staging_strength == 0.0


def test_validated_calibration_artifact_controls_screening_and_staging_strengths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "artifact"
    artifact.mkdir()
    calibration = tmp_path / "agent_correction_calibration.json"
    fusion = cli.AuthorityJointFusionConfig(
        state_strength=0.0,
        agent_strength=0.25,
        staging_strength=0.5,
        max_abs_state_delta=0.75,
        ordinal_temperature=1.0,
    )
    calibration_payload = {
        "selection_status": "validated_joint_gain",
        "selected_screening_strength": 0.25,
        "selected_staging_strength": 0.5,
        "joint_fusion_config": fusion.to_dict(),
        "joint_fusion_config_hash": hash_artifact(fusion.to_dict()),
    }
    calibration.write_text(json.dumps(calibration_payload), encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr(cli.AuthorityStudyDataset, "from_artifact_dir", lambda path: object())
    monkeypatch.setattr(cli, "build_authority_review_runtime", lambda **kwargs: object())

    def run_study(*args: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return _result(tmp_path)

    monkeypatch.setattr(cli, "run_authority_state_delta_cohort", run_study)
    args = _args(tmp_path, calibration_artifact=calibration, state_strength=0.0, agent_strength=0.0)
    status, _ = cli.run(args)

    assert status == 0
    fusion = captured["config"].joint_fusion  # type: ignore[union-attr]
    assert fusion.state_strength == 0.0
    assert fusion.agent_strength == 0.25
    assert fusion.staging_strength == 0.5
    assert captured["config"].calibration_artifact_hash == hash_artifact(calibration_payload)  # type: ignore[union-attr]
    assert captured["config"].calibration_artifact["selection_status"] == "validated_joint_gain"  # type: ignore[index,union-attr]


def test_state_replay_requires_its_own_validated_strength(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact = tmp_path / "agent_correction_calibration.json"
    artifact.write_text(json.dumps({
        "selection_status": "validated_joint_gain",
        "selected_screening_strength": 0.25,
        "selected_staging_strength": 0.5,
        "state_selection_status": "validated_joint_gain",
        "selected_state_strength": 0.125,
    }), encoding="utf-8")
    args = _args(tmp_path, calibration_artifact=artifact)
    state, screening, staging, _, payload = cli._resolved_strengths(args)
    assert (state, screening, staging) == (0.125, 0.25, 0.5)
    assert payload is not None


def test_manual_nonzero_strength_is_rejected_without_calibration(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="require --calibration-artifact"):
        cli._validate_args(_args(tmp_path, agent_strength=0.25))


def test_unvalidated_calibration_artifact_is_rejected(tmp_path: Path) -> None:
    calibration = tmp_path / "agent_correction_calibration.json"
    calibration.write_text(json.dumps({
        "selection_status": "failed_closed_no_joint_gain",
        "selected_screening_strength": 1.0,
        "selected_staging_strength": 1.0,
    }), encoding="utf-8")
    with pytest.raises(ValueError, match="validated_joint_gain"):
        cli._resolved_strengths(_args(tmp_path, calibration_artifact=calibration))


def test_parser_rejects_conflicting_output_modes():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([
            "--artifact-dir", "artifact", "--output-dir", "output", "--cache-dir", "cache",
            "--provider", "disabled", "--model", "test", "--decision-only", "--report",
        ])


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
