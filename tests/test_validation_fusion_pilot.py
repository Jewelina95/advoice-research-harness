"""Isolation and selection contracts for the frozen PREPARE validation pilot."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/run_validation_fusion_pilot.py"
spec = importlib.util.spec_from_file_location("validation_fusion_pilot", SCRIPT)
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def test_npz_reads_only_selected_rows(tmp_path):
    path = tmp_path / "mixed.npz"
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    values[[1, 3, 5]] = np.nan
    np.savez_compressed(path, embeddings=values)
    actual = pilot.selected_npz_rows(path, [4, 0, 2], expected_rows=6)
    np.testing.assert_array_equal(actual, values[[4, 0, 2]])
    with pytest.raises(ValueError):
        pilot.selected_npz_rows(path, [1], expected_rows=6)
    with pytest.raises(ValueError):
        pilot.selected_npz_rows(path, [0, 0], expected_rows=6)
    with pytest.raises(ValueError):
        pilot.selected_npz_rows(path, [0], expected_rows=7)


def test_selected_csv_does_not_decode_excluded_payload(tmp_path):
    path = tmp_path / "mixed.csv"
    path.write_text("subject_id,label,value\na,HC,1\ntest,POISON,not_numeric\nb,AD,2\n")
    rows = pilot.selected_csv(path, {"a", "b"}, ["subject_id", "label", "value"])
    assert [float(r["value"]) for r in rows] == [1, 2]
    assert {r["subject_id"] for r in rows} == {"a", "b"}


def test_public_split_rejects_duplicate_uid(tmp_path):
    path = tmp_path / "protocol.csv"
    path.write_text("uid,reference_partition\na,train\na,validation\n")
    with pytest.raises(ValueError, match="Duplicate"):
        pilot.public_split(path, expected_counts=None)


def test_acc_is_primary_and_ties_are_deterministic():
    rows = [
        {"candidate": "a", "accuracy": 0.7, "macro_f1": 0.9, "macro_auc_ovr": 0.9},
        {"candidate": "b", "accuracy": 0.8, "macro_f1": 0.5, "macro_auc_ovr": 0.5},
    ]
    assert pilot.select_winner(rows)["candidate"] == "b"
    rows[0].update(accuracy=0.8, macro_f1=0.5, macro_auc_ovr=0.5)
    assert pilot.select_winner(rows)["candidate"] == "a"
    assert len(pilot.preregistered_plan()["candidates"]) == 8


def evidence_fixture():
    ids = ["h1", "h2", "h3", "a1", "m1", "v"]
    return pd.DataFrame({
        "dataset_id": ["synthetic"] * 6,
        "subject_id": ids,
        "label": ["HC", "HC", "HC", "AD", "MCI", "HC"],
        "split": ["train"] * 5 + ["validation"],
        "language": ["english"] * 6,
        "metric_id": ["silence_fraction"] * 6,
        "metric_instance_id": ["silence_fraction"] * 6,
        "task_scope": ["overall"] * 6,
        "value": [0.1, 0.2, 0.3, 0.4, 0.35, 0.99],
        "direction": [1] * 6,
        "reliability": [0.85] * 6,
    })


def test_state_references_exclude_validation_values_and_labels():
    evidence = evidence_fixture()
    config = {"states": [{"id": "S01", "metrics": ["silence_fraction"], "weights": [1]}]}
    ids = evidence.subject_id.tolist()
    train = ids[:-1]
    tasks = dict.fromkeys(ids, "picture_description")
    first = pilot.state_matrix(evidence, config, train, ids, tasks)
    changed = evidence.copy()
    changed.loc[changed.subject_id.eq("v"), ["value", "label"]] = [10000, "AD"]
    second = pilot.state_matrix(changed, config, train, ids, tasks)
    np.testing.assert_array_equal(first[:-1], second[:-1])
    reduced = pilot.state_matrix(evidence, config, ["h1", "h2", "a1", "m1"], ids, tasks)
    assert not np.array_equal(first, reduced)


def test_logistic_is_reproducible_and_class_order_fixed():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(36, 8))
    y = np.tile(np.arange(3), 12)
    first = pilot.fit_head(x, y, 0.01)
    second = pilot.fit_head(x, y, 0.01)
    np.testing.assert_array_equal(first.predict_proba(x), second.predict_proba(x))
    np.testing.assert_array_equal(first.classes_, [0, 1, 2])


def test_output_cannot_overwrite_source_or_prior_run(tmp_path):
    source = tmp_path / "history"
    source.mkdir()
    with pytest.raises(ValueError):
        pilot.reserve_output(source / "artifacts" / "oops", source)
    output = tmp_path / "pilot"
    pilot.reserve_output(output, source)
    with pytest.raises(FileExistsError):
        pilot.reserve_output(output, source)


def test_candidate_training_excludes_validation_and_inner_held_labels(monkeypatch):
    rng = np.random.default_rng(12)
    ids = [f"s{i}" for i in range(45)]
    labels = np.tile(np.arange(3), 15)
    evidence = pd.concat([evidence_fixture().iloc[[0]]] * len(ids), ignore_index=True)
    evidence["subject_id"] = ids
    evidence["label"] = [pilot.LABELS[i] for i in labels]
    evidence["value"] = rng.uniform(size=len(ids))
    config = {"states": [{"id": "S01", "metrics": ["silence_fraction"], "weights": [1]}]}
    tasks = dict.fromkeys(ids, "picture_description")
    embeddings = {name: rng.normal(size=(45, 6)) for name in ["text", "audio"]}
    observed = []
    original = pilot.state_matrix

    def audited(evidence, config, reference_ids, output_ids, tasks):
        observed.append(set(reference_ids))
        assert set(reference_ids).issubset(ids[:36])
        return original(evidence, config, reference_ids, output_ids, tasks)

    monkeypatch.setattr(pilot, "state_matrix", audited)
    args = (ids, labels, embeddings, evidence, config, tasks, 36, pilot.preregistered_plan())
    results, predictions, models, folds = pilot.fit_candidates(*args)
    assert len(results) == len(predictions) == len(models) == 8
    assert len(observed) == 4
    assert set.union(*(set(fold["held_ids"]) for fold in folds)) == set(ids[:36])
    for fold in folds:
        assert not set(fold["fit_ids"]) & set(fold["held_ids"])
        assert set(fold["HC_reference_ids"]).issubset(fold["fit_ids"])
    changed_labels = labels.copy()
    changed_labels[36:] = (changed_labels[36:] + 1) % 3
    evidence.loc[evidence.subject_id.isin(ids[36:]), "label"] = "AD"
    _, changed_predictions, _, _ = pilot.fit_candidates(
        ids, changed_labels, embeddings, evidence, config, tasks, 36, pilot.preregistered_plan())
    for name, probability in predictions.items():
        np.testing.assert_array_equal(probability, changed_predictions[name])
        np.testing.assert_allclose(probability.sum(axis=1), 1)


def test_known_threadpool_platform_error_falls_back_without_hiding_other_errors(monkeypatch):
    def broken(**kwargs):
        raise AttributeError("'NoneType' object has no attribute 'split'")

    monkeypatch.setattr(pilot, "threadpool_limits", broken)
    monkeypatch.setattr(pilot, "THREAD_LIMIT_WARNINGS", [])
    with pytest.warns(RuntimeWarning, match="legacy OpenBLAS"):
        with pilot.bounded_cpu():
            pass
    assert len(pilot.THREAD_LIMIT_WARNINGS) == 1

    def unrelated(**kwargs):
        raise AttributeError("unrelated error")

    monkeypatch.setattr(pilot, "threadpool_limits", unrelated)
    with pytest.raises(AttributeError, match="unrelated"):
        with pilot.bounded_cpu():
            pass
