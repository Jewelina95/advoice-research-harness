from __future__ import annotations

import numpy as np
import pytest

from advoice.module_b import ConditionalArbitrator


LABELS = ("HC", "MCI", "AD")


def _row(index: int, **updates):
    post = [(.70, .20, .10), (.15, .65, .20), (.10, .25, .65)][index % 3]
    scores = [
        {"HC": 4, "MCI": 1, "AD": 0},
        {"HC": 0, "MCI": 4, "AD": 1},
        {"HC": 0, "MCI": 1, "AD": 4},
    ][index % 3]
    row = {
        "module_a_pre_replay": dict(zip(LABELS, (.65, .25, .10), strict=True)),
        "module_a_post_replay": dict(zip(LABELS, post, strict=True)),
        "agent_ordinal_scores": scores,
        "agent_scores_validated": True,
        "eligible": True,
        "revision_type": "downweight" if index % 2 else "none",
        "action_type": "classify" if index % 2 else "review",
        "agreement": True,
        "evidence_coverage": .95,
        "evidence_reliability": .90,
        "confound_burden": .05,
        "route": "picture",
        "language": "en",
        "ood": .0,
        "incremental_evidence_declared": True,
        "evidence_consumed_by_module_a": False,
        "incremental_evidence_ids": [f"metric:incremental-{index}"],
        "consumed_evidence_ids": [],
        "route_supported": True,
        "replay_performed": True,
        "cross_fit_fold": index % 3,
        "true_label": LABELS[index % 3],
    }
    row.update(updates)
    return row


@pytest.fixture
def fitted() -> ConditionalArbitrator:
    rows = [_row(index) for index in range(18)]
    return ConditionalArbitrator(LABELS, ridge_alpha=.5, max_logit_correction=.12).fit(rows)


def _post_values(row):
    return np.asarray([row["module_a_post_replay"][label] for label in LABELS])


def test_ineligible_is_bitwise_identical_to_post_replay_module_a(fitted) -> None:
    row = _row(1, eligible=False)
    prediction = fitted.predict_one(row)
    actual = np.asarray([prediction.probabilities[label] for label in LABELS])
    assert np.array_equal(actual, _post_values(row))
    assert prediction.fallback == "module_a_post_replay"
    assert not prediction.additive_correction_applied


@pytest.mark.parametrize(
    "updates",
    [
        {"agent_scores_validated": False},
        {"incremental_evidence_declared": False, "incremental_evidence_ids": []},
        {
            "incremental_evidence_declared": False,
            "incremental_evidence_ids": [],
            "evidence_consumed_by_module_a": True,
            "consumed_evidence_ids": ["metric:consumed-2"],
        },
    ],
)
def test_no_validated_incremental_evidence_has_zero_additive_correction(fitted, updates) -> None:
    row = _row(2, **updates)
    prediction = fitted.predict_one(row)
    assert all(value == 0.0 for value in prediction.additive_logit_correction.values())
    np.testing.assert_array_equal(
        [prediction.probabilities[label] for label in LABELS], _post_values(row)
    )
    assert prediction.reason == "no_validated_incremental_evidence"


def test_consumed_evidence_changes_only_through_supplied_replay(fitted) -> None:
    row = _row(
        0,
        incremental_evidence_declared=False,
        incremental_evidence_ids=[],
        evidence_consumed_by_module_a=True,
        consumed_evidence_ids=["metric:replayed"],
        module_a_pre_replay={"HC": .80, "MCI": .10, "AD": .10},
        module_a_post_replay={"HC": .20, "MCI": .30, "AD": .50},
    )
    prediction = fitted.predict_one(row)
    assert prediction.probabilities == row["module_a_post_replay"]
    assert prediction.reason == "no_validated_incremental_evidence"


def test_disjoint_incremental_evidence_remains_eligible_when_other_evidence_was_consumed(fitted) -> None:
    row = _row(
        2,
        evidence_consumed_by_module_a=True,
        consumed_evidence_ids=["metric:already-replayed"],
        incremental_evidence_declared=True,
        incremental_evidence_ids=["metric:new-agent-evidence"],
    )
    prediction = fitted.predict_one(row)

    assert prediction.additive_correction_applied
    assert prediction.reason == "validated_incremental_evidence"
    assert prediction.consumed_evidence_ids == ("metric:already-replayed",)
    assert prediction.incremental_evidence_ids == ("metric:new-agent-evidence",)


def test_unsupported_route_has_declared_post_replay_fallback(fitted) -> None:
    row = _row(0, route="spontaneous", language="zh", route_supported=False)
    prediction = fitted.predict_one(row)
    assert prediction.fallback == "module_a_post_replay"
    assert prediction.reason == "unsupported_route_language"
    assert prediction.probabilities == row["module_a_post_replay"]


def test_logit_correction_is_zero_sum_and_capped(fitted) -> None:
    prediction = fitted.predict_one(_row(2))
    correction = np.asarray([prediction.additive_logit_correction[label] for label in LABELS])
    assert prediction.additive_correction_applied
    assert correction.sum() == pytest.approx(0.0, abs=1e-15)
    assert np.abs(correction).max() <= fitted.max_logit_correction + 1e-15


def test_probabilities_are_normalized_and_serialization_is_reproducible(fitted, tmp_path) -> None:
    row = _row(1)
    first = fitted.predict_one(row)
    second = fitted.predict_one(row)
    assert first.to_json() == second.to_json()
    np.testing.assert_allclose(fitted.predict_proba([_row(0), row, _row(2)]).sum(axis=1), 1.0)
    restored = ConditionalArbitrator.from_json(fitted.to_json())
    assert restored.to_json() == fitted.to_json()
    assert restored.to_dict()["revision_types"] == ["downweight", "none"]
    path = tmp_path / "module_b.json"
    fitted.save(path)
    assert ConditionalArbitrator.load(path).predict_one(row).to_json() == first.to_json()


def test_fit_requires_cross_fitted_development_rows() -> None:
    with pytest.raises(ValueError, match="cross_fit_fold"):
        ConditionalArbitrator(LABELS).fit([_row(0, cross_fit_fold=None)])


def test_revision_without_replay_is_rejected() -> None:
    with pytest.raises(ValueError, match="requires Module A numerical replay"):
        ConditionalArbitrator(LABELS).fit([_row(1, replay_performed=False)])


def test_revision_type_is_a_serialized_conditional_feature(fitted) -> None:
    assert "score:HC|revision=downweight" in fitted.feature_names_
    assert "score:AD|revision=none" in fitted.feature_names_
    artifact = fitted.to_dict()
    assert artifact["revision_types"] == ["downweight", "none"]
    assert artifact["feature_names"] == list(fitted.feature_names_)


def test_incremental_and_consumed_evidence_ids_must_be_disjoint() -> None:
    with pytest.raises(ValueError, match="must be disjoint"):
        ConditionalArbitrator(LABELS).fit([
            _row(0, consumed_evidence_ids=["metric:incremental-0"], evidence_consumed_by_module_a=True)
        ])


@pytest.mark.parametrize(
    "updates",
    [
        {"incremental_evidence_declared": True, "incremental_evidence_ids": []},
        {"incremental_evidence_declared": False, "incremental_evidence_ids": ["metric:unexpected"]},
        {"evidence_consumed_by_module_a": True, "consumed_evidence_ids": []},
        {"evidence_consumed_by_module_a": False, "consumed_evidence_ids": ["metric:unexpected"]},
    ],
)
def test_evidence_id_declaration_mismatch_is_rejected(updates) -> None:
    with pytest.raises(ValueError, match="must exactly match"):
        ConditionalArbitrator(LABELS).fit([_row(0, **updates)])
