"""Scorer-only label loading is separate from all provider-facing imports."""
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_pilot_contracts import assessment, snapshot, subject


@pytest.fixture
def c():
    return importlib.import_module("advoice.pilot.contracts")


@pytest.fixture
def labels():
    return importlib.import_module("advoice.pilot.labels")


def label_row(c, labels):
    return labels.LabelRow(subject=subject(c), source_label="case", label="AD",
                           label_mapping={"control": "HC", "case": "AD"}, source="registry_v1")


def test_provider_imports_never_import_or_export_label_capabilities():
    script = """
import sys
import advoice.pilot as pilot
import advoice.pilot.contracts as contracts
assert 'advoice.pilot.labels' not in sys.modules
for module in (pilot, contracts):
    assert not hasattr(module, 'LabelRow')
    assert not hasattr(module, 'read_labels_jsonl')
    assert not hasattr(module, 'serialize_labels_jsonl')
"""
    root = Path(__file__).resolve().parents[1]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, env=env,
    )
    assert result.returncode == 0, result.stderr


def test_explicit_scorer_label_roundtrip(c, labels):
    row = label_row(c, labels)
    assert labels.LabelRow.from_json(row.to_json()) == row
    for purpose in ("scoring", "training"):
        serialized = labels.serialize_labels_jsonl((row,), purpose=purpose)
        assert labels.read_labels_jsonl(serialized.splitlines(), purpose=purpose) == (row,)
    assert not hasattr(importlib.import_module("advoice.pilot"), "LabelRow")
    assert not hasattr(c, "LabelRow")


@pytest.mark.parametrize("change", [{"source": ""}, {"label_mapping": {}}, {"label": "MCI"},
                                     {"label_mapping": {"case": "mild_AD"}}])
def test_label_source_and_mapping_required(c, labels, change):
    with pytest.raises(c.PilotContractError):
        replace(label_row(c, labels), **change)


def test_labels_never_cross_safe_or_provider_boundaries(c, labels):
    row = label_row(c, labels)
    for value in (row, row.to_dict(), {"subject": row}, {"state_cards": [row.to_dict()]}):
        with pytest.raises(c.PilotContractError):
            c.safe_inference_payload(value)
    for serialize in (row.to_inference_dict, row.to_publishable_dict):
        with pytest.raises(c.PilotContractError):
            serialize()
    payload = assessment(c, snapshot(c)).to_dict()
    payload["state_judgments"][0]["label"] = row.label
    with pytest.raises(c.PilotContractError):
        c.AgentAssessment.from_provider_payload(payload, snapshot(c))


def test_label_reader_rejects_wrong_purpose_duplicates_and_malformed_rows(c, labels):
    row = label_row(c, labels)
    for purpose in ("inference", "publish", None):
        with pytest.raises(c.PilotContractError):
            labels.read_labels_jsonl([row.to_json()], purpose=purpose)
        with pytest.raises(c.PilotContractError):
            labels.serialize_labels_jsonl([row], purpose=purpose)
    for lines in ([row.to_json(), row.to_json()], ["{"], ["{}"],
                  [json.dumps({**row.to_dict(), "source": None})]):
        with pytest.raises(c.PilotContractError):
            labels.read_labels_jsonl(lines, purpose="scoring")


def test_label_json_rejects_duplicate_keys(c, labels):
    duplicate = '{"schema_version":"advoice.pilot.v1","schema_version":"advoice.pilot.v1"}'
    with pytest.raises(c.PilotContractError, match="Duplicate"):
        labels.LabelRow.from_json(duplicate)


def test_label_duplicate_identity_includes_dataset_task_subject_and_source(c, labels):
    row = label_row(c, labels)
    different_source = replace(row, source="registry_v2")
    serialized = labels.serialize_labels_jsonl((row, different_source), purpose="scoring")
    assert labels.read_labels_jsonl(serialized.splitlines(), purpose="scoring") == (
        row, different_source)
    same_identity = replace(row, source_label="control", label="HC")
    for operation in (
        lambda: labels.serialize_labels_jsonl((row, same_identity), purpose="scoring"),
        lambda: labels.read_labels_jsonl(
            (row.to_json(), same_identity.to_json()), purpose="scoring"),
    ):
        with pytest.raises(c.PilotContractError):
            operation()


def test_binary_label_mapping_cannot_fabricate_mci_or_ad(c, labels):
    binary = subject(c, task="hc_impairment", class_order=("HC", "IMPAIRED"))
    for fabricated in ("MCI", "AD"):
        with pytest.raises(c.PilotContractError):
            labels.LabelRow(subject=binary, source_label="case", label=fabricated,
                             label_mapping={"case": fabricated}, source="registry_v1")
