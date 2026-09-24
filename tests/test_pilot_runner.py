from __future__ import annotations

import json
from pathlib import Path
from types import MappingProxyType

import pytest

from advoice.pilot import learning as learning
from advoice.pilot import runner as runner_module
from advoice.pilot.runner import (
    DiskAssessmentCache, LABEL_CAPABILITIES, PilotRunner, PreflightError, RunnerError, STAGES,
    paired_accuracy_summary, prediction_lock, score_locked_predictions,
)
from advoice.pilot.runtime import CacheIdentity, RuntimeBudget, RuntimeValidationError


def test_dry_run_materializes_entire_dag_and_fake_stamp(tmp_path: Path) -> None:
    records = PilotRunner({}, tmp_path, provider="fake", dry_run=True).run("all")

    assert [record["stage"] for record in records] == list(STAGES)
    assert all(record["status"] == "dry_run" for record in records)
    assert all(record["output"]["analytical"] is False for record in records)
    assert all(record["output"]["provider_stamp"] == "TEST_ONLY_FAKE_PROVIDER" for record in records)
    assert records[-1]["label_capabilities"] == []
    assert records[-2]["label_capabilities"] == ["holdout", "stress"]


def test_no_paid_execution_is_default_and_fake_cannot_be_analytical(tmp_path: Path) -> None:
    with pytest.raises(PreflightError, match="disabled by default"):
        PilotRunner({}, tmp_path)
    with pytest.raises(PreflightError, match="test-only"):
        PilotRunner({}, tmp_path, provider="fake", allow_paid=True)


def test_budget_deadline_and_holdout_label_capability_are_fail_closed() -> None:
    with pytest.raises(RuntimeValidationError, match="between 1 and 180"):
        RuntimeBudget(0, 0, 0, 0, 0, 0, 0.0, timeout_seconds=181)
    exhausted = RuntimeBudget(0, 0, 0, 0, 0, 0, 0.0)
    with pytest.raises(RuntimeValidationError, match="budget exhausted"):
        exhausted.reserve_semantic_call()
    assert all("holdout" not in LABEL_CAPABILITIES[stage] for stage in STAGES if stage != "score")
    assert LABEL_CAPABILITIES["score"] == frozenset({"holdout", "stress"})


def test_resume_is_idempotent_and_config_hash_mismatch_is_rejected(tmp_path: Path) -> None:
    first = PilotRunner({"seed": 1}, tmp_path, provider="fake", dry_run=True).run("inventory")[0]
    second = PilotRunner({"seed": 1}, tmp_path, provider="fake", dry_run=True, resume=True).run("inventory")[0]
    assert second == first

    with pytest.raises(RunnerError, match="Resume code/config SHA mismatch"):
        PilotRunner({"seed": 2}, tmp_path, provider="fake", dry_run=True, resume=True).run("inventory")


def test_resume_rejects_tampered_output_and_incomplete_status(tmp_path: Path) -> None:
    runner = PilotRunner({"seed": 1}, tmp_path, provider="fake", dry_run=True)
    runner.run("inventory")
    path = tmp_path / "stages" / "inventory.json"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["output"]["analytical"] = True
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RunnerError, match="output hash mismatch"):
        PilotRunner({"seed": 1}, tmp_path, provider="fake", dry_run=True, resume=True).run("inventory")

    record["output"]["analytical"] = False
    record["output_hash"] = runner_module._hash(record["output"])
    record["status"] = "blocked"
    path.write_text(json.dumps(record), encoding="utf-8")
    with pytest.raises(RunnerError, match="not a completed stage"):
        PilotRunner({"seed": 1}, tmp_path, provider="fake", dry_run=True, resume=True).run("inventory")


def test_real_run_preflight_requires_only_stage_visible_artifacts(tmp_path: Path) -> None:
    runner = PilotRunner({"paid_execution": True, "artifacts": {}}, tmp_path, provider="configured", allow_paid=True)
    with pytest.raises(PreflightError) as raised:
        runner.run("inventory")
    text = str(raised.value)
    for name in ("split_manifest", "cohort_status", "prepared_cases"):
        assert name in text
    assert "holdout_labels" not in text and "stress_labels" not in text
    assert (tmp_path / "stages" / "inventory.json").is_file()


def test_disk_cache_survives_restart_and_rejects_tampering(tmp_path: Path) -> None:
    identity = CacheIdentity(fields=MappingProxyType({"model": "fixture", "round": "v0", "nested": MappingProxyType({"schema": "v1"})}), key="a" * 64)
    first = DiskAssessmentCache(tmp_path / "cache")
    first.store(identity, {"status": "ok"})
    second = DiskAssessmentCache(tmp_path / "cache")
    payload, event = second.lookup(identity)
    assert payload == {"status": "ok"}
    assert event.status == "hit"

    path = tmp_path / "cache" / "aa" / ("a" * 64 + ".json")
    body = json.loads(path.read_text())
    body["payload"] = {"status": "changed"}
    path.write_text(json.dumps(body), encoding="utf-8")
    with pytest.raises(RunnerError, match="content hash mismatch"):
        second.lookup(identity)


def test_closed_portable_adapters_are_allowlisted_and_synthetic_is_not() -> None:
    base = learning.PortableBasePredictorAdapter()
    replay = learning.PortableReplayPredictorAdapter()
    registry = learning.create_trusted_adapter_registry(
        (base, replay), class_order=("HC", "AD"),
        feature_names_by_role={"base": ("acoustic", "nonstate"), "replay": ("state",)},
    )
    assert len(registry.attestations) == 2
    with pytest.raises(learning.LearningError, match="trusted analytical adapter"):
        learning.create_trusted_adapter_registry(
            (learning.SyntheticLinearPredictorAdapter("base"), replay),
            class_order=("HC", "AD"),
            feature_names_by_role={"base": ("acoustic", "nonstate"), "replay": ("state",)},
        )


def _locked_rows() -> tuple[dict, ...]:
    rows = []
    probabilities = {
        "B_raw": ((0.8, 0.2), (0.8, 0.2)),
        "B": ((0.8, 0.2), (0.8, 0.2)),
        "J-A": ((0.8, 0.2), (0.1, 0.9)),
        "J-S": ((0.8, 0.2), (0.1, 0.9)),
        "J-AS": ((0.7, 0.3), (0.1, 0.9)),
    }
    for arm, per_subject in probabilities.items():
        for subject_id, probability in zip(("sub_a", "sub_b"), per_subject, strict=True):
            rows.append({"dataset_id": "fixture", "subject_id": subject_id, "arm": arm,
                         "class_order": ("HC", "AD"), "probabilities": probability})
    return tuple(rows)


def test_locked_score_has_paired_bootstrap_mcnemar_and_per_class_help_harm() -> None:
    rows = _locked_rows()
    assigned = {"fixture": ("sub_a", "sub_b")}
    scored = score_locked_predictions(rows, {"sub_a": "HC", "sub_b": "AD"}, seed=7,
                                      assigned_subjects=assigned, lock=prediction_lock(rows, assigned))
    paired = next(row for row in scored["paired"] if row["arm"] == "J-AS")
    assert paired["helped"] == 1 and paired["harmed"] == 0
    assert paired["bootstrap"]["replicates"] == 2000
    assert paired["mcnemar"]["base_wrong_arm_right"] == 1
    assert paired["mcnemar"]["base_right_arm_wrong"] == 0
    assert paired["baseline_arm"] == "B"
    assert paired["per_class"]["AD"] == {"n": 1, "helped": 1, "harmed": 0}

    direct = paired_accuracy_summary(("HC", "AD"), ("HC", "HC"), ("HC", "AD"), seed=7)
    assert direct["bootstrap"]["estimate"] == 0.5


def test_score_rejects_unlocked_missing_duplicate_or_contradictory_rows() -> None:
    rows = list(_locked_rows())
    assigned = {"fixture": ("sub_a", "sub_b")}
    labels = {"sub_a": "HC", "sub_b": "AD"}
    with pytest.raises(RunnerError, match="requires a locked"):
        score_locked_predictions(rows, labels, seed=7, assigned_subjects=assigned, lock=None)

    contradictory = [dict(row) for row in rows]
    contradictory[0]["predicted"] = "AD"
    with pytest.raises(RunnerError, match="contradicts"):
        score_locked_predictions(contradictory, labels, seed=7, assigned_subjects=assigned,
                                 lock=prediction_lock(contradictory, assigned))

    duplicate = [dict(row) for row in rows]
    duplicate.append(dict(duplicate[0]))
    with pytest.raises(RunnerError, match="duplicate"):
        score_locked_predictions(duplicate, labels, seed=7, assigned_subjects=assigned,
                                 lock=prediction_lock(duplicate, assigned))

    missing = [row for row in rows if not (row["arm"] == "J-AS" and row["subject_id"] == "sub_b")]
    with pytest.raises(RunnerError, match="omit"):
        score_locked_predictions(missing, labels, seed=7, assigned_subjects=assigned,
                                 lock=prediction_lock(missing, assigned))


def test_score_derives_prediction_and_retains_zero_f1() -> None:
    rows = list(_locked_rows())
    for row in rows:
        if row["arm"] == "B":
            row["probabilities"] = (1.0, 0.0)
    assigned = {"fixture": ("sub_a", "sub_b")}
    scored = score_locked_predictions(rows, {"sub_a": "HC", "sub_b": "AD"}, seed=7,
                                      assigned_subjects=assigned, lock=prediction_lock(rows, assigned))
    f1 = next(row for row in scored["metrics"] if row["arm"] == "B" and row["metric"] == "f1_AD")
    assert f1["value"] == 0.0 and f1["undefined_reason"] is None


def test_real_stages_are_blocked_without_adapters_and_t7_binds_code_and_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    gate = tmp_path / "preflight.json"
    config = {"paid_execution": True, "t7_preflight_gate": str(gate), "artifacts": {}}
    runner = PilotRunner(config, tmp_path / "run", provider="configured", allow_paid=True)
    gate.write_text(json.dumps({"verdict": "ACCEPT", "code_sha": runner_module._git_sha(),
                                "config_hash": runner._config_hash()}), encoding="utf-8")
    assert runner._require_t7_gate()["verdict"] == "ACCEPT"

    mismatched = PilotRunner({**config, "seed": 2}, tmp_path / "other", provider="configured", allow_paid=True)
    with pytest.raises(PreflightError, match="config hash"):
        mismatched._require_t7_gate()

    monkeypatch.setattr(PilotRunner, "_preflight", lambda self, stage: {"checked": stage})
    record = runner.run("inventory")[0]
    assert record["status"] == "blocked"
    assert record["output"]["reason"] == "no_reviewed_stage_adapter"


def test_non_score_input_hash_never_hashes_sealed_labels(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    artifacts = {}
    for name in ("split_manifest", "cohort_status", "prepared_cases", "development_labels", "holdout_labels", "stress_labels"):
        path = tmp_path / f"{name}.json"
        path.write_text("x", encoding="utf-8")
        artifacts[name] = str(path)
    runner = PilotRunner({"paid_execution": True, "artifacts": artifacts}, tmp_path / "run", provider="configured", allow_paid=True)
    hashed: list[Path] = []
    monkeypatch.setattr(runner_module, "_file_hash", lambda path: hashed.append(path) or "hash")
    runner._input_hash("predict-holdout")
    assert all("holdout_labels" not in str(path) and "stress_labels" not in str(path) for path in hashed)


def test_real_stage_blocks_dry_run_dependencies(tmp_path: Path) -> None:
    PilotRunner({}, tmp_path, provider="fake", dry_run=True).run("inventory")
    runner = PilotRunner({"paid_execution": True, "artifacts": {}}, tmp_path, provider="configured", allow_paid=True)
    record = runner.run("split")[0]
    assert record["status"] == "blocked"
    assert record["output"]["reason"] == "dependencies_incomplete"
