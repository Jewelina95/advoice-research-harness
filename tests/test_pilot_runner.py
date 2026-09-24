from __future__ import annotations

import json
from pathlib import Path

import pytest

from advoice.pilot import learning as learning
from advoice.pilot.runner import (
    DiskAssessmentCache, LABEL_CAPABILITIES, PilotRunner, PreflightError, RunnerError, STAGES,
    paired_accuracy_summary, score_locked_predictions,
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

    with pytest.raises(RunnerError, match="Resume hash mismatch"):
        PilotRunner({"seed": 2}, tmp_path, provider="fake", dry_run=True, resume=True).run("inventory")


def test_real_run_preflight_lists_every_missing_artifact(tmp_path: Path) -> None:
    runner = PilotRunner({"artifacts": {}}, tmp_path, provider="configured", allow_paid=True)
    with pytest.raises(PreflightError) as raised:
        runner.run("inventory")
    text = str(raised.value)
    for name in ("split_manifest", "cohort_status", "prepared_cases", "development_labels", "holdout_labels", "stress_labels"):
        assert name in text
    assert (tmp_path / "stages" / "inventory.json").is_file()


def test_disk_cache_survives_restart_and_rejects_tampering(tmp_path: Path) -> None:
    identity = CacheIdentity(fields={"model": "fixture", "round": "v0"}, key="a" * 64)
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


def test_locked_score_has_paired_bootstrap_mcnemar_and_help_harm() -> None:
    rows = (
        {"dataset_id": "fixture", "subject_id": "sub_a", "arm": "B", "class_order": ("HC", "AD"), "probabilities": (0.8, 0.2), "predicted": "HC"},
        {"dataset_id": "fixture", "subject_id": "sub_b", "arm": "B", "class_order": ("HC", "AD"), "probabilities": (0.8, 0.2), "predicted": "HC"},
        {"dataset_id": "fixture", "subject_id": "sub_a", "arm": "J-AS", "class_order": ("HC", "AD"), "probabilities": (0.7, 0.3), "predicted": "HC"},
        {"dataset_id": "fixture", "subject_id": "sub_b", "arm": "J-AS", "class_order": ("HC", "AD"), "probabilities": (0.1, 0.9), "predicted": "AD"},
    )
    scored = score_locked_predictions(rows, {"sub_a": "HC", "sub_b": "AD"}, seed=7)
    paired = scored["paired"][0]
    assert paired["helped"] == 1 and paired["harmed"] == 0
    assert paired["bootstrap"]["replicates"] == 2000
    assert paired["mcnemar"]["base_wrong_arm_right"] == 1
    assert paired["mcnemar"]["base_right_arm_wrong"] == 0

    direct = paired_accuracy_summary(("HC", "AD"), ("HC", "HC"), ("HC", "AD"), seed=7)
    assert direct["bootstrap"]["estimate"] == 0.5
