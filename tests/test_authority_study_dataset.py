from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import advoice.authority_study_dataset as study_module
from advoice.authority_study_dataset import AuthorityStudyDataset, AuthorityStudyDatasetError
from advoice.evidence import MetricEvidenceV2, ReferenceMetadata
from advoice.evidence_replay import build_state_graph_v2


BASIC_STATES = {"states": [{"id": "S01", "metrics": ["pause"], "weights": [1.0]}]}


@dataclass
class _FakeAdvisor:
    dataset_id: str = "fixture"
    class_order: tuple[str, ...] = ("HC", "AD")
    subject_ids: tuple[str, ...] = ("test-short", "test-long")

    @classmethod
    def from_artifact_dir(cls, path: Path, **kwargs: Any) -> "_FakeAdvisor":
        cls.called_with = (Path(path), kwargs)
        return cls()


class _FakeExecutor:
    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.calls: list[dict[str, Any]] = []
        _FakeExecutor.latest = self

    def prepare_case(self, *, case_metadata: dict[str, Any], evidence: tuple[Any, ...]) -> Any:
        self.calls.append({"case_metadata": dict(case_metadata), "evidence": evidence})
        return SimpleNamespace(case_id=case_metadata["case_id"], case_metadata=dict(case_metadata))


def _frozen(*, include_train: bool = True, advisor_subjects: tuple[str, ...] | None = None) -> Any:
    rows = [
        ("train-hc", "HC", "train"),
        ("train-ad", "AD", "train"),
        ("test-short", "HC", "test"),
        ("test-long", "AD", "test"),
    ]
    if not include_train:
        rows = [(subject, label, "test") for subject, label, _ in rows]
    manifest_rows = []
    for subject, label, split in rows:
        manifest_rows.append({
            "dataset_id": "fixture", "case_id": subject, "subject_id": subject,
            "label": label, "split": split, "channel": "picture_description",
            "task_type": "cookie", "language": "en", "audio_path": f"/hidden/{subject}.wav",
            "transcript_path": f"/hidden/{subject}.txt",
        })
    manifest = pd.DataFrame(manifest_rows)
    wide = pd.DataFrame([
        {
            "subject_id": subject, "label": label, "split": split,
            "state_S01": -1.0 if label == "HC" else 1.0,
            "rel_S01": 0.9, "available_S01": 1.0,
            "diagnosis_leakage": label,
        }
        for subject, label, split in rows
    ])
    labels = {subject: label for subject, label, _ in rows}
    splits = {subject: split for subject, _, split in rows}
    reference = ReferenceMetadata(median=0.0, scale=1.0, sample_size=20)
    evidence = {
        subject: (
            MetricEvidenceV2(
                evidence_id=f"evidence:{subject}",
                metric_id="pause",
                metric_instance_id="pause:cookie",
                subject_id=subject,
                state_id="S01",
                task_id="cookie",
                value=-1.0 if label == "HC" else 1.0,
                direction=1,
                reference=reference,
                consumed_by_supervised=True,
            ),
        )
        for subject, label, _ in rows
    }
    return SimpleNamespace(
        manifest=manifest,
        state_wide=wide,
        subject_labels=labels,
        subject_splits=splits,
        evidence_for_subject=lambda subject: evidence.get(subject, ()),
    )


def _write_transcripts(root: Path, *, duplicate: bool = False, omit: str | None = None) -> None:
    rows = [
        {"subject_id": "train-hc", "transcript": "training transcript one"},
        {"subject_id": "train-ad", "transcript": "training transcript two"},
        {"subject_id": "test-short", "transcript": "short"},
        {"subject_id": "test-long", "transcript": "this is the longest held out transcript"},
    ]
    if omit is not None:
        rows = [row for row in rows if row["subject_id"] != omit]
    if duplicate:
        rows.append(dict(rows[-1]))
    pd.DataFrame(rows).to_csv(root / "subject_transcripts.csv", index=False)


def _install(monkeypatch: pytest.MonkeyPatch, frozen: Any, advisor: _FakeAdvisor | None = None) -> None:
    monkeypatch.setattr(study_module, "load_frozen_authority_dataset", lambda path: frozen)
    chosen = advisor or _FakeAdvisor()
    monkeypatch.setattr(
        study_module.FrozenConditionCAdvisor,
        "from_artifact_dir",
        lambda path, **kwargs: chosen,
    )
    monkeypatch.setattr(
        study_module,
        "build_state_graph_v2",
        lambda evidence, states_config, **kwargs: SimpleNamespace(
            wide=frozen.state_wide.drop(columns=["label", "split"]).copy()
        ),
    )
    monkeypatch.setattr(study_module, "ConditionalAuthorityExecutor", _FakeExecutor)


def _process_like_frozen(states_config: dict[str, Any]) -> Any:
    subjects = (
        ("process-train-hc-1", "HC", "train", -2.0, -1.0),
        ("process-train-hc-2", "HC", "train", -1.5, -0.5),
        ("process-train-ad-1", "AD", "train", 1.0, 2.0),
        ("process-train-ad-2", "AD", "train", 1.5, 2.5),
        ("process-test", "AD", "test", 1.2, 2.2),
    )
    reference = ReferenceMetadata(median=0.0, scale=1.0, sample_size=20)
    evidence: dict[str, tuple[MetricEvidenceV2, ...]] = {}
    manifest_rows: list[dict[str, str]] = []
    all_evidence: list[MetricEvidenceV2] = []
    for subject_id, label, split, ctd_value, vf_value in subjects:
        subject_evidence = tuple(
            MetricEvidenceV2(
                evidence_id=f"metric:{subject_id}:{task_id}",
                metric_id="pause",
                metric_instance_id=f"pause:{task_id}",
                subject_id=subject_id,
                state_id="S01",
                task_id=task_id,
                value=value,
                direction=1,
                reference=reference,
                consumed_by_supervised=True,
            )
            for task_id, value in (("ctd", ctd_value), ("vf", vf_value))
        )
        evidence[subject_id] = subject_evidence
        all_evidence.extend(subject_evidence)
        for task_id in ("ctd", "vf"):
            manifest_rows.append({
                "dataset_id": "fixture", "case_id": f"{subject_id}:{task_id}",
                "subject_id": subject_id, "label": label, "split": split,
                "channel": "structured_multitask", "task_type": task_id, "language": "en",
            })

    graph = build_state_graph_v2(
        tuple(all_evidence),
        states_config,
        dataset_id="fixture",
        label="unknown",
        split="graph",
    )
    wide = graph.wide.drop(columns=["dataset_id", "label", "split"]).copy()
    labels = {subject_id: label for subject_id, label, _, _, _ in subjects}
    splits = {subject_id: split for subject_id, _, split, _, _ in subjects}
    wide.insert(1, "label", wide["subject_id"].map(labels))
    wide.insert(2, "split", wide["subject_id"].map(splits))
    for prefix in ("state_", "rel_", "available_"):
        residual = f"{prefix}S01__task_ctd_residual"
        wide[f"{prefix}S01__task_ctd"] = wide[residual]
    return SimpleNamespace(
        manifest=pd.DataFrame(manifest_rows),
        state_wide=wide,
        subject_labels=labels,
        subject_splits=splits,
        evidence_for_subject=lambda subject: evidence.get(subject, ()),
    )


def test_prepares_label_blind_test_case_with_explicit_train_state_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen()
    _write_transcripts(tmp_path)
    _install(monkeypatch, frozen)

    dataset = AuthorityStudyDataset.from_artifact_dir(
        tmp_path,
        states_config=BASIC_STATES,
    )
    selected = dataset.prepare_test_cases(order="longest-first", max_cases=1)

    assert [item.prepared_case.case_id for item in selected] == ["test-long"]
    assert selected[0].transcript == "this is the longest held out transcript"
    assert dataset.executor.kwargs["module_a_state_feature_whitelist"] == (
        "available_S01", "rel_S01", "state_S01",
    )
    trained = dataset.executor.kwargs["module_a"]
    assert trained.requested_feature_whitelist_ == dataset.executor.kwargs["module_a_state_feature_whitelist"]
    assert trained.feature_selection_mode_ == "explicit_state_feature_whitelist"
    assert trained.task_column == "__authority_task_type"
    assert trained.language_column == "__authority_language"

    metadata = _FakeExecutor.latest.calls[0]["case_metadata"]
    assert metadata["channel"] == "picture_description"
    assert metadata["task_type"] == "cookie"
    assert metadata["language"] == "en"
    assert metadata["target_labels"] == ["HC", "AD"]
    assert not {"label", "split", "audio_path", "transcript_path"} & set(metadata)
    assert "/hidden" not in repr(metadata)

    assert dataset.evaluation_truth(["test-long"]) == {"test-long": "AD"}


def test_preparation_does_not_read_truth_after_dataset_construction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen()
    _write_transcripts(tmp_path)
    _install(monkeypatch, frozen)
    dataset = AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config=BASIC_STATES)

    dataset._evaluation_truth = {"unexpected": "forbidden"}
    dataset.prepare_test_cases(order="subject_id", max_cases=1)

    assert _FakeExecutor.latest.calls[0]["case_metadata"]["case_id"] == "test-long"


def test_process_task_absolute_columns_are_excluded_but_replayable_residuals_are_retained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    states_config = {
        "states": [{"id": "S01", "metrics": ["pause"], "weights": [1.0]}]
    }
    frozen = _process_like_frozen(states_config)
    pd.DataFrame([
        {"subject_id": subject_id, "transcript": f"transcript for {subject_id}"}
        for subject_id in frozen.subject_splits
    ]).to_csv(tmp_path / "subject_transcripts.csv", index=False)
    monkeypatch.setattr(study_module, "load_frozen_authority_dataset", lambda path: frozen)
    monkeypatch.setattr(
        study_module.FrozenConditionCAdvisor,
        "from_artifact_dir",
        lambda path, **kwargs: _FakeAdvisor(subject_ids=("process-test",)),
    )
    rebuilt_subjects: list[tuple[str, ...]] = []

    def _recording_graph_builder(evidence: tuple[MetricEvidenceV2, ...], *args: Any, **kwargs: Any) -> Any:
        rebuilt_subjects.append(tuple(sorted({item.subject_id for item in evidence})))
        return build_state_graph_v2(evidence, *args, **kwargs)

    monkeypatch.setattr(study_module, "build_state_graph_v2", _recording_graph_builder)

    dataset = AuthorityStudyDataset.from_artifact_dir(
        tmp_path,
        states_config=states_config,
    )
    whitelist = dataset.executor.module_a_state_feature_whitelist

    assert "state_S01__task_ctd" not in whitelist
    assert "rel_S01__task_ctd" not in whitelist
    assert "available_S01__task_ctd" not in whitelist
    assert "state_S01__task_ctd_residual" in whitelist
    assert "rel_S01__task_ctd_residual" in whitelist
    assert "available_S01__task_ctd_residual" in whitelist
    assert rebuilt_subjects == [("process-train-ad-1",)]
    prepared = dataset.prepare_test_cases(order="subject_id", max_cases=1)
    assert prepared[0].prepared_case.case_id == "process-test"


@pytest.mark.parametrize(
    ("frozen_kwargs", "advisor", "transcript_kwargs", "match"),
    [
        (
            {"include_train": False},
            _FakeAdvisor(subject_ids=("train-hc", "train-ad", "test-short", "test-long")),
            {},
            "no split=train",
        ),
        ({}, _FakeAdvisor(class_order=("HC", "MCI")), {}, "Training labels and frozen"),
        ({}, None, {"duplicate": True}, "duplicate subject IDs"),
        ({}, None, {"omit": "test-long"}, "missing test subjects"),
    ],
)
def test_rejects_invalid_preparation_contracts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    frozen_kwargs: dict[str, Any],
    advisor: _FakeAdvisor | None,
    transcript_kwargs: dict[str, Any],
    match: str,
) -> None:
    _write_transcripts(tmp_path, **transcript_kwargs)
    _install(monkeypatch, _frozen(**frozen_kwargs), advisor)

    with pytest.raises(AuthorityStudyDatasetError, match=match):
        AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config=BASIC_STATES)


def test_rejects_advisor_subject_mismatch_and_unknown_truth_request(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_transcripts(tmp_path)
    _install(
        monkeypatch,
        _frozen(),
        _FakeAdvisor(subject_ids=("test-short",)),
    )
    with pytest.raises(AuthorityStudyDatasetError, match="subjects do not match"):
        AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config=BASIC_STATES)

    _install(monkeypatch, _frozen())
    dataset = AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config=BASIC_STATES)
    with pytest.raises(AuthorityStudyDatasetError, match="Unknown subjects"):
        dataset.evaluation_truth(["unknown"])
