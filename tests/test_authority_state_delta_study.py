from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest

import advoice.authority_state_delta_study as study_module
from advoice.authority_state_delta_study import (
    AuthorityStateDeltaStudyConfig,
    AuthorityStateDeltaStudyError,
    _packet_bound_hash,
    run_authority_state_delta_cohort,
)
from advoice.authority_joint_fusion import AuthorityJointFusionConfig
from advoice.authority_study_dataset import AuthorityStudyDataset, PreparedAuthorityStudyCase
from advoice.module_a import ExplanationPacket


LABELS = ("HC", "AD")


def _packet(
    probabilities: tuple[float, float],
    *,
    evidence_hash: str,
    state_hash: str | None,
) -> ExplanationPacket:
    values = dict(zip(LABELS, probabilities, strict=True))
    hashes = {"evidence_hash": evidence_hash}
    if state_hash is not None:
        hashes["state_hash"] = state_hash
    return ExplanationPacket(
        schema_version="fixture.packet.v1",
        module_version="fixture",
        class_order=LABELS,
        predicted_label=LABELS[max(range(2), key=probabilities.__getitem__)],
        raw_probabilities=values,
        calibrated_probabilities=dict(values),
        calibration_status="fixture",
        logits={},
        intercepts={},
        feature_contributions={label: {} for label in LABELS},
        branch_contributions={label: {} for label in LABELS},
        consumed_evidence_ids=(),
        uncertainty={},
        fold_disagreement=None,
        ood={},
        applicability_status="fixture",
        hashes=hashes,
    )


@dataclass
class _Transaction:
    transaction_hash: str = "9" * 64

    def to_dict(self) -> dict[str, Any]:
        return {"transaction_hash": self.transaction_hash, "batches": ["S01", "S02"]}


class _Advisor:
    dataset_id = "fixture"
    class_order = LABELS

    def __init__(self, frozen: Mapping[str, ExplanationPacket]) -> None:
        self.frozen = frozen

    def explain_subject(self, subject_id: str, *, evidence_snapshot: Any) -> ExplanationPacket:
        assert evidence_snapshot == (f"evidence:{subject_id}",)
        return self.frozen[subject_id]


class _Dataset(AuthorityStudyDataset):
    def __init__(
        self,
        cases: tuple[PreparedAuthorityStudyCase, ...],
        truth: Mapping[str, str],
        advisor: _Advisor,
    ) -> None:
        executor = SimpleNamespace(
            states_config={"states": []},
            module_a=object(),
            correlation_config=None,
        )
        super().__init__(
            frozen=SimpleNamespace(),
            advisor=advisor,  # type: ignore[arg-type]
            executor=executor,  # type: ignore[arg-type]
            _transcripts={},
            _subject_routing={},
            _evaluation_truth={},
        )
        self._cases = cases
        self._truth = dict(truth)
        self.review_calls = 0
        self.truth_calls = 0

    def prepare_test_cases(
        self,
        *,
        max_cases: int | None = None,
        order: str = "longest_first",
        selection_salt: str = "authority-pilot-v1",
    ):
        assert order in {"longest_first", "subject_id", "stable_hash"}
        assert selection_salt
        return self._cases if max_cases is None else self._cases[:max_cases]

    def evaluation_truth(self, subject_ids=None):
        self.truth_calls += 1
        # The runner may only read truth after every requested review has run.
        assert self.review_calls == len(subject_ids)
        return {subject_id: self._truth[subject_id] for subject_id in subject_ids}


class _Runtime:
    def __init__(self, dataset: _Dataset, *, status: str = "available") -> None:
        self.dataset = dataset
        self.status = status
        self.cache_dir = Path("/tmp/authority-study-cache")
        self.calls: list[str] = []

    def review(self, prepared, *, transcript=None):
        assert "label" not in prepared.case_metadata
        assert transcript == f"transcript:{prepared.case_id}"
        self.dataset.review_calls += 1
        self.calls.append(prepared.case_id)
        return SimpleNamespace(
            status=self.status,
            case_id=f"pseudo:{prepared.case_id}",
            blind_request_hash="a" * 64,
            reconciliation_request_hash="b" * 64,
            cache_key="c" * 64,
            error=None if self.status == "available" else "provider response invalid",
            blind_assessment=(
                SimpleNamespace(ordinal_scores={"HC": 1, "AD": 1})
                if self.status == "available"
                else None
            ),
        )


def _case(case_id: str, pre_packet: ExplanationPacket) -> PreparedAuthorityStudyCase:
    pre = SimpleNamespace(
        revised_evidence=(f"evidence:{case_id}",),
        packet=pre_packet,
        audit=SimpleNamespace(revision_hash="8" * 64, audit_hash="7" * 64),
    )
    prepared = SimpleNamespace(
        case_id=case_id,
        case_metadata={"case_id": case_id, "dataset_id": "fixture"},
        case_context={},
        route=SimpleNamespace(target_route=SimpleNamespace(id="diagnosis", labels=LABELS)),
        pre_replay=pre,
        module_a_evidence=(f"evidence:{case_id}",),
        reviewed_packet_hash="1" * 64,
        reviewed_evidence_hash="2" * 64,
        reviewed_state_graph_hash="3" * 64,
        advisor_packet_hash="4" * 64,
    )
    return PreparedAuthorityStudyCase(prepared_case=prepared, transcript=f"transcript:{case_id}")


def _fixture_dataset() -> tuple[_Dataset, tuple[PreparedAuthorityStudyCase, ...], dict[str, ExplanationPacket]]:
    pre_one = _packet((0.7, 0.3), evidence_hash="1" * 64, state_hash="2" * 64)
    pre_two = _packet((0.3, 0.7), evidence_hash="3" * 64, state_hash="4" * 64)
    cases = (_case("case-1", pre_one), _case("case-2", pre_two))
    frozen = {
        "case-1": _packet((0.8, 0.2), evidence_hash="5" * 64, state_hash=None),
        "case-2": _packet((0.2, 0.8), evidence_hash="6" * 64, state_hash=None),
    }
    return _Dataset(cases, {"case-1": "HC", "case-2": "AD"}, _Advisor(frozen)), cases, frozen


def _compiled(transaction: _Transaction | None):
    return SimpleNamespace(decision=SimpleNamespace(decision_hash="d" * 64), transaction=transaction)


def test_no_transaction_preserves_frozen_bits_and_reads_truth_only_after_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _, frozen = _fixture_dataset()
    runtime = _Runtime(dataset)
    monkeypatch.setattr(study_module, "compile_authority_review_decision", lambda *_: _compiled(None))

    result = run_authority_state_delta_cohort(
        dataset,
        runtime,
        output_dir=tmp_path,
        config=AuthorityStateDeltaStudyConfig(),
        cache_dir=runtime.cache_dir,
    )

    assert runtime.calls == ["case-1", "case-2"]
    assert dataset.truth_calls == 1
    assert result.failed_case_ids == ()
    payload = __import__("json").loads(result.audit_json_path.read_text(encoding="utf-8"))
    by_case = {item["case_id"]: item for item in payload["cases"]}
    for case_id, packet in frozen.items():
        assert by_case[case_id]["transaction"] is None
        assert tuple(
            float(by_case[case_id]["fusion"]["probabilities"][label]).hex()
            for label in LABELS
        ) == tuple(packet.calibrated_probabilities[label].hex() for label in LABELS)
    assert result.paired_counts == {"changed": 0, "unchanged": 2, "corrected": 0, "harmed": 0}


def test_atomic_transaction_replays_once_and_applies_predeclared_delta(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, cases, _ = _fixture_dataset()
    runtime = _Runtime(dataset)
    transaction = _Transaction()
    post_packets = {
        "case-1": _packet((0.4, 0.6), evidence_hash="a" * 64, state_hash="b" * 64),
        "case-2": _packet((0.6, 0.4), evidence_hash="c" * 64, state_hash="d" * 64),
    }
    replayed: list[tuple[Any, Any]] = []
    monkeypatch.setattr(study_module, "compile_authority_review_decision", lambda *_: _compiled(transaction))

    def replay(snapshot, revision, **kwargs):
        replayed.append((snapshot, revision))
        case_id = str(snapshot[0]).removeprefix("evidence:")
        return SimpleNamespace(
            packet=post_packets[case_id],
            audit=SimpleNamespace(revision_hash=transaction.transaction_hash, audit_hash=f"audit:{case_id}"),
        )

    monkeypatch.setattr(study_module, "replay_evidence", replay)
    result = run_authority_state_delta_cohort(
        dataset,
        runtime,
        output_dir=tmp_path,
        config=AuthorityStateDeltaStudyConfig(
            joint_fusion=AuthorityJointFusionConfig(
                state_strength=0.25,
                agent_strength=0.0,
                max_abs_state_delta=0.75,
                ordinal_temperature=1.0,
            )
        ),
    )

    assert len(replayed) == 2
    assert all(revision is transaction for _, revision in replayed)
    assert result.failed_case_ids == ()
    audit = __import__("json").loads(result.audit_json_path.read_text(encoding="utf-8"))
    assert all(item["transaction"]["transaction_hash"] == transaction.transaction_hash for item in audit["cases"])
    assert all(
        item["fusion"]["provenance"]["revision_action"] == "multi_state_evidence_review"
        for item in audit["cases"]
    )
    assert any(item["fusion"]["correction_applied"] for item in audit["cases"])


def test_failed_provider_keeps_full_queue_frozen_baseline_without_synthetic_fused_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _, _ = _fixture_dataset()
    runtime = _Runtime(dataset, status="failed_closed_provider_error")
    monkeypatch.setattr(study_module, "compile_authority_review_decision", lambda *_: pytest.fail("must not compile"))

    result = run_authority_state_delta_cohort(dataset, runtime, output_dir=tmp_path)

    assert result.completed_case_ids == ()
    assert result.failed_case_ids == ("case-1", "case-2")
    assert result.frozen_metrics["n"] == 2
    assert result.frozen_metrics["accuracy"] == 1.0
    assert result.fused_metrics is None
    aggregate = __import__("json").loads(result.aggregate_json_path.read_text(encoding="utf-8"))
    assert aggregate["metrics_exclude_failed_cases"] is False
    assert aggregate["failed_case_count"] == 2
    assert aggregate["frozen_full_queue_case_ids"] == ["case-1", "case-2"]
    assert aggregate["paired_case_ids"] == []
    assert aggregate["paired_frozen_metrics"] is None
    assert aggregate["paired_fused_metrics"] is None
    assert aggregate["agent_coverage"] == {
        "attempted_case_count": 2,
        "completed_case_count": 0,
        "failed_case_count": 2,
        "coverage_rate": 0.0,
        "failure_rate": 1.0,
    }
    audits = __import__("json").loads(result.audit_json_path.read_text(encoding="utf-8"))["cases"]
    assert all(item["status"] == "failed" and "frozen" in item and "fusion" not in item for item in audits)


def test_complete_case_pairing_is_separate_from_full_queue_frozen_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _, _ = _fixture_dataset()
    runtime = _Runtime(dataset)
    monkeypatch.setattr(study_module, "compile_authority_review_decision", lambda *_: _compiled(None))

    original_review = runtime.review

    def review_one_failure(prepared, *, transcript=None):
        if prepared.case_id == "case-2":
            dataset.review_calls += 1
            runtime.calls.append(prepared.case_id)
            return SimpleNamespace(
                status="failed_closed_provider_error",
                case_id=f"pseudo:{prepared.case_id}",
                blind_request_hash="a" * 64,
                reconciliation_request_hash="b" * 64,
                cache_key="c" * 64,
                error="provider response invalid",
            )
        return original_review(prepared, transcript=transcript)

    runtime.review = review_one_failure  # type: ignore[method-assign]
    result = run_authority_state_delta_cohort(dataset, runtime, output_dir=tmp_path)

    aggregate = __import__("json").loads(result.aggregate_json_path.read_text(encoding="utf-8"))
    assert result.frozen_metrics["n"] == 2
    assert result.fused_metrics["n"] == 1
    assert aggregate["paired_frozen_metrics"]["n"] == 1
    assert aggregate["paired_fused_metrics"]["n"] == 1
    assert aggregate["agent_coverage"]["coverage_rate"] == 0.5
    assert aggregate["agent_coverage"]["failure_rate"] == 0.5


def test_completed_cases_resume_without_a_second_provider_call(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset, _, _ = _fixture_dataset()
    runtime = _Runtime(dataset)
    monkeypatch.setattr(study_module, "compile_authority_review_decision", lambda *_: _compiled(None))
    first = run_authority_state_delta_cohort(dataset, runtime, output_dir=tmp_path)
    assert first.completed_case_ids == ("case-1", "case-2")

    second = run_authority_state_delta_cohort(dataset, runtime, output_dir=tmp_path)
    assert runtime.calls == ["case-1", "case-2"]
    assert second.completed_case_ids == ("case-1", "case-2")
    assert len(second.audit_jsonl_path.read_text(encoding="utf-8").splitlines()) == 2


def test_rejects_mismatched_runtime_cache(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    dataset, _, _ = _fixture_dataset()
    runtime = _Runtime(dataset)
    monkeypatch.setattr(study_module, "compile_authority_review_decision", lambda *_: _compiled(None))

    with pytest.raises(AuthorityStateDeltaStudyError, match="does not match"):
        run_authority_state_delta_cohort(
            dataset,
            runtime,
            output_dir=tmp_path,
            cache_dir=tmp_path / "other-cache",
        )


def test_packet_hash_aliases_accept_equal_values_and_reject_conflicts() -> None:
    packet = _packet((0.5, 0.5), evidence_hash="a" * 64, state_hash="b" * 64)
    packet.hashes["evidence_snapshot_hash"] = "a" * 64
    assert _packet_bound_hash(packet, ("evidence_hash", "evidence_snapshot_hash")) == "a" * 64

    packet.hashes["evidence_snapshot_hash"] = "c" * 64
    with pytest.raises(AuthorityStateDeltaStudyError, match="one consistent value"):
        _packet_bound_hash(packet, ("evidence_hash", "evidence_snapshot_hash"))


def test_single_case_smoke_reports_descriptive_metrics_without_calibration(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset, _, _ = _fixture_dataset()
    runtime = _Runtime(dataset)
    monkeypatch.setattr(study_module, "compile_authority_review_decision", lambda *_: _compiled(None))

    result = run_authority_state_delta_cohort(
        dataset,
        runtime,
        output_dir=tmp_path,
        config=AuthorityStateDeltaStudyConfig(max_cases=1),
    )

    assert result.frozen_metrics["n"] == 1
    assert result.fused_metrics["n"] == 1
    assert result.frozen_metrics["evaluation_status"] == "insufficient_class_coverage"
    assert result.fused_metrics["macro_auroc_ovr"] is None
