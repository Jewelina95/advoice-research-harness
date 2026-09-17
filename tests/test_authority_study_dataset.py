from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

import advoice.authority_study_dataset as study_module
from advoice.authority_study_dataset import AuthorityStudyDataset, AuthorityStudyDatasetError


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
    evidence = {subject: (f"evidence:{subject}",) for subject, _, _ in rows}
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
    monkeypatch.setattr(study_module, "ConditionalAuthorityExecutor", _FakeExecutor)


def test_prepares_label_blind_test_case_with_explicit_train_state_whitelist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    frozen = _frozen()
    _write_transcripts(tmp_path)
    _install(monkeypatch, frozen)

    dataset = AuthorityStudyDataset.from_artifact_dir(
        tmp_path,
        states_config={"states": []},
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
    dataset = AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config={"states": []})

    dataset._evaluation_truth = {"unexpected": "forbidden"}
    dataset.prepare_test_cases(order="subject_id", max_cases=1)

    assert _FakeExecutor.latest.calls[0]["case_metadata"]["case_id"] == "test-long"


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
        AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config={"states": []})


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
        AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config={"states": []})

    _install(monkeypatch, _frozen())
    dataset = AuthorityStudyDataset.from_artifact_dir(tmp_path, states_config={"states": []})
    with pytest.raises(AuthorityStudyDatasetError, match="Unknown subjects"):
        dataset.evaluation_truth(["unknown"])
