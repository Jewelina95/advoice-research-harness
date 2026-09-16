import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


PATH = Path(__file__).resolve().parents[1] / "scripts/evaluate_locked_prepare_test.py"
SPEC = importlib.util.spec_from_file_location("locked_prepare_test", PATH)
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def test_hash_guard_rejects_tampered_input(tmp_path):
    path = tmp_path / "input"
    path.write_text("original")
    digest = evaluation.sha256(path)
    evaluation.require_hash(path, digest)
    path.write_text("changed")
    with pytest.raises(ValueError, match="Hash mismatch"):
        evaluation.require_hash(path, digest)


def test_inference_never_calls_fit():
    class PredictOnly:
        classes_ = np.array([0, 1, 2])

        def predict_proba(self, x):
            return np.tile([0.2, 0.3, 0.5], (len(x), 1))

        def fit(self, *args, **kwargs):
            raise AssertionError("No test-time fit is permitted")

    candidate = {"head": "stack", "cognition": True}
    bundle = {"model": {"modalities": ["text", "audio", "state"],
                        "experts": {name: PredictOnly() for name in ["text", "audio", "state"]},
                        "head": PredictOnly()}}
    matrices = {name: np.zeros((4, 2)) for name in ["text", "audio", "state"]}
    result = evaluation.infer_locked(bundle, candidate, matrices)
    np.testing.assert_array_equal(result, np.tile([0.2, 0.3, 0.5], (4, 1)))


def test_paired_comparison_preserves_subject_alignment():
    y = np.array([0, 1, 2, 0, 1, 2])
    ours = np.eye(3)[[0, 0, 2, 0, 2, 2]]
    other = np.eye(3)[[0, 1, 0, 0, 1, 2]]
    result = evaluation.paired_correctness(y, ours, other)
    assert result["selected_only_correct"] == 1
    assert result["comparator_only_correct"] == 2
    assert result["accuracy_delta"] == pytest.approx(-1 / 6)


def test_test_ids_exclude_train_validation_and_reject_duplicates(tmp_path):
    path = tmp_path / "protocol.csv"
    path.write_text("uid,reference_partition\na,train\nb,validation\nc,test\nd,test\n")
    assert evaluation.test_ids(path, expected_count=2) == ["c", "d"]
    path.write_text("uid,reference_partition\na,train\na,test\n")
    with pytest.raises(ValueError, match="Duplicate"):
        evaluation.test_ids(path, expected_count=1)


def test_authorization_does_not_modify_original_lock(tmp_path):
    lock = tmp_path / "selected_config.lock.json"
    lock.write_text(json.dumps({"test_evaluation_authorized": False}))
    original = lock.read_bytes()
    output = tmp_path / "test_evaluation"
    evaluation.authorize_execution(output, lock)
    assert lock.read_bytes() == original
    authorization = json.loads((output / "authorization.json").read_text())
    assert authorization["test_evaluation_authorized"] is True
    assert authorization["original_lock_sha256"] == evaluation.sha256(lock)
    with pytest.raises(FileExistsError):
        evaluation.authorize_execution(output, lock)


def test_bias_release_without_labels_requires_explicit_prior_label_verification(tmp_path):
    path = tmp_path / "bias.csv"
    path.write_text("uid,C,MCI,ADRD\nb,0.2,0.3,0.5\na,0.8,0.1,0.1\n")
    with pytest.raises(ValueError, match="labels are required"):
        evaluation.released_probabilities(path, ["a", "b"], np.array([0, 2]))
    actual = evaluation.released_probabilities(path, ["a", "b"], np.array([0, 2]), require_labels=False)
    np.testing.assert_allclose(actual, [[0.8, 0.1, 0.1], [0.2, 0.3, 0.5]])
