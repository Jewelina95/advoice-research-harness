import numpy as np
import pandas as pd
import pytest

from advoice.evidence_research import (
    effective_dimension_statistics, historical_declared_reuse,
    historical_inventory, overlap_components, overlap_statistics, replay_check,
    state_overlap_tables, task_scope_reuse_statistics, training_rows,
)


METRICS = [
    {"id": "silence_fraction", "role": "clinical_support"},
    {"id": "voiced_fraction", "role": "clinical_support"},
]


def subject_frame(n=25):
    x = np.linspace(0.1, 0.9, n)
    return pd.DataFrame({
        "subject_id": [f"synthetic-{i}" for i in range(n)],
        "split": "train", "language": "en", "label": 0,
        "silence_fraction": x, "voiced_fraction": 1 - x,
    })


def test_only_train_registered_values_are_analyzed():
    frame = subject_frame()
    heldout = frame.iloc[[0]].assign(
        subject_id="synthetic-test", split="test", silence_fraction=999,
        voiced_fraction=999, label=999,
    )
    coverage, pairs, meta = overlap_statistics(pd.concat([frame, heldout]), METRICS)
    assert meta["n_train_records"] == 25
    assert meta["excluded_nontrain_records"] == 1
    assert set(coverage.metric_id) == {m["id"] for m in METRICS}
    assert pairs.iloc[0].rho == pytest.approx(-1)
    assert pairs.iloc[0].sum_one_on_observed_rows
    assert pairs.iloc[0].affine_on_observed_rows
    assert pairs.iloc[0].affine_slope == pytest.approx(-1)
    assert pairs.iloc[0].affine_intercept == pytest.approx(1)
    assert pairs.iloc[0].reviewed_complement_formula
    assert not meta["feature_selection_performed"]


@pytest.mark.parametrize("change", ["duplicate", "blank", "null", "split", "no_train"])
def test_ambiguous_subject_tables_fail_closed(change):
    frame = subject_frame()
    if change == "duplicate":
        frame.loc[1, "subject_id"] = frame.loc[0, "subject_id"]
    elif change == "blank":
        frame.loc[0, "subject_id"] = " "
    elif change == "null":
        frame.loc[0, "subject_id"] = None
    elif change == "split":
        frame.loc[0, "split"] = "unverified"
    else:
        frame["split"] = "test"
    with pytest.raises(ValueError):
        training_rows(frame)


def test_task_and_language_blocks_are_separate():
    frame = subject_frame(50)
    frame.loc[25:, "language"] = "es"
    frame["task_recall__silence_fraction"] = frame.silence_fraction
    frame["task_recall__voiced_fraction"] = frame.voiced_fraction
    coverage, pairs, meta = overlap_statistics(frame, METRICS)
    assert len(pairs) == 4
    assert set(pairs.n_pairs) == {25}
    assert set(pairs.task_scope) == {"overall", "recall"}
    assert set(coverage.language) == {"en", "es"}
    assert meta["n_registered_metric_columns"] == 4


def test_missingness_is_not_imputed_and_unknown_language_is_disclosed():
    frame = subject_frame().drop(columns="language")
    frame.loc[:9, "voiced_fraction"] = np.nan
    coverage, pairs, _ = overlap_statistics(frame, METRICS)
    pair = pairs.iloc[0]
    assert pair.n_pairs == 15
    assert pair.status == "insufficient_pairs"
    assert not pair.language_conditioned
    assert not pair.review_candidate
    assert coverage.set_index("metric_id").loc["voiced_fraction", "missing_fraction"] == pytest.approx(.4)


def test_constant_metric_is_not_ranked_as_independent():
    frame = subject_frame().assign(voiced_fraction=0)
    coverage, pairs, _ = overlap_statistics(frame, METRICS)
    assert "constant" in set(coverage.status)
    assert pairs.iloc[0].status == "constant"
    assert pd.isna(pairs.iloc[0].rho)


def test_monotone_nonlinear_relation_is_not_misreported_as_affine():
    frame = subject_frame().assign(
        silence_fraction=np.linspace(.1, 2.5, 25),
        voiced_fraction=np.square(np.linspace(.1, 2.5, 25)),
    )
    _, pairs, _ = overlap_statistics(frame, METRICS)
    assert pairs.iloc[0].rho == pytest.approx(1)
    assert not pairs.iloc[0].affine_on_observed_rows


def test_overlap_components_retain_correlated_group_and_isolate():
    metrics = METRICS + [{"id": "independent", "role": "model_auxiliary", "branch": "language"}]
    frame = subject_frame(40)
    frame["independent"] = np.random.default_rng(7).normal(size=len(frame))
    coverage, pairs, _ = overlap_statistics(frame, metrics, threshold=.95)
    components = overlap_components(coverage, pairs, threshold=.95)
    assert sorted(components.n_members) == [1, 2]
    paired = components.loc[components.n_members.eq(2)].iloc[0]
    assert paired.contains_affine_relation
    assert paired.contains_complement
    assert set(paired.metric_ids.split(";")) == {"silence_fraction", "voiced_fraction"}


def test_effective_dimension_reports_duplicate_rank_without_selecting_features():
    metrics = [
        {"id": "x", "role": "clinical_support", "branch": "speech", "report_permission": True},
        {"id": "y", "role": "clinical_support", "branch": "speech", "report_permission": True},
        {"id": "z", "role": "model_auxiliary", "branch": "language", "report_permission": False},
    ]
    n = 50
    frame = pd.DataFrame({
        "subject_id": [f"s-{i}" for i in range(n)], "split": "train", "language": "en",
        "x": np.arange(n), "y": np.arange(n), "z": (np.arange(n) * 7) % 31,
    })
    dimensions = effective_dimension_statistics(frame, metrics)
    all_metrics = dimensions.loc[dimensions.metric_subset.eq("all_variable")].iloc[0]
    speech = dimensions.loc[dimensions.metric_subset.eq("branch:speech")].iloc[0]
    assert all_metrics.matrix_rank == 2
    assert all_metrics.effective_rank > 1
    assert speech.matrix_rank == 1
    assert speech.effective_rank == pytest.approx(1)
    assert speech.scope == "descriptive_observed_matrix_not_clinical_factor_count"


def test_effective_dimension_fails_closed_for_small_complete_case_block():
    frame = subject_frame(10)
    dimensions = effective_dimension_statistics(frame, METRICS, min_rows=20)
    assert set(dimensions.status) == {"insufficient_complete_rows", "no_variable_metrics"}


def test_task_scope_copy_is_marked_as_routing_alias_not_new_evidence():
    frame = subject_frame()
    frame["task_recall__silence_fraction"] = frame.silence_fraction
    frame["task_recall__voiced_fraction"] = frame.voiced_fraction * 2
    reuse = task_scope_reuse_statistics(frame, METRICS)
    silence = reuse.loc[reuse.metric_id.eq("silence_fraction")].iloc[0]
    voiced = reuse.loc[reuse.metric_id.eq("voiced_fraction")].iloc[0]
    assert silence.equal_on_observed_rows
    assert silence.interpretation == "routing_alias_not_independent_evidence"
    assert not voiced.equal_on_observed_rows
    assert voiced.affine_on_observed_rows
    assert voiced.interpretation == "task_specific_measurement_candidate"


def test_historical_inventory_preserves_repeated_names_without_guessing_aliases():
    history = pd.DataFrame({
        "metric_name": ["silence_fraction", "old_pause", "old_pause"],
        "state_id": ["S01", "S01", "S03"],
        "metric_definition": ["fraction", "pause", "pause"],
        "how_calculated": ["mean", "unknown", "unknown"],
    })
    result = historical_inventory(history, METRICS)
    assert len(result) == 3
    assert result.repeated_historical_name.sum() == 2
    assert "definition_review" in result.iloc[0].mapping_status
    assert set(result.iloc[1:].mapping_status) == {"unresolved_not_absent"}


def test_config_overlap_separates_reuse_and_formula_group():
    states = [
        {"id": "S01", "metrics": ["silence_fraction"], "weights": [1.]},
        {"id": "S02", "metrics": ["voiced_fraction"], "weights": [1.]},
        {"id": "S03", "metrics": ["voiced_fraction"], "weights": [1.]},
    ]
    edges, incidence, pairs = state_overlap_tables(states, {m["id"] for m in METRICS})
    assert len(edges) == 3
    assert len(incidence) == 2
    assert pairs.iloc[0].shared_metric_count == 0
    assert pairs.iloc[0].shared_reviewed_information_groups == "vad_occupancy"
    assert pairs.iloc[2].shared_metric_count == 1
    states[0]["weights"] = []
    with pytest.raises(ValueError):
        state_overlap_tables(states, {m["id"] for m in METRICS})


def test_historical_reuse_uses_explicit_references_only():
    history = pd.DataFrame({
        "metric_name": ["old_a", "alias_a", "apparently_similar"],
        "state_id": ["S01", "S03", "S01"],
        "matched_current_columns": ["old_column; second_column", "old_column", None],
    })
    edges, reuse = historical_declared_reuse(history)
    assert len(edges) == 3
    assert len(reuse) == 1
    assert reuse.iloc[0].n_metric_names == 2
    assert reuse.iloc[0].declared_historical_column == "old_column"


GRAPH = {
    "audio": [], "metric": ["audio"], "state_a": ["metric"],
    "state_b": ["metric"], "embedding": ["audio"],
    "risk": ["state_a", "state_b", "embedding"], "report": ["risk"],
    "independent_control": [],
}


def test_complete_replay_passes_declared_contract():
    replay = ["metric", "state_a", "state_b", "embedding", "risk", "report"]
    result = replay_check(GRAPH, {"audio"}, replay, set(), {"report"})
    assert result["passed"]
    assert "independent_control" not in result["affected_descendants"]


def test_shared_descendants_and_embedding_bypass_cannot_be_ignored():
    result = replay_check(GRAPH, {"audio"}, ["metric", "state_a", "risk", "report"], set(), {"report"})
    assert not result["passed"]
    assert result["stale_descendants"] == ["embedding", "report", "risk", "state_b"]
    assert "risk" in result["out_of_order_recomputations"]


def test_local_state_intervention_does_not_claim_source_removal():
    result = replay_check(GRAPH, {"state_a"}, ["risk", "report"], set(), {"report"})
    assert result["passed"]
    assert result["affected_descendants"] == ["report", "risk"]


def test_invalid_source_cannot_support_active_descendant():
    invalid = set(GRAPH) - {"independent_control"}
    result = replay_check(GRAPH, {"audio"}, [], invalid, {"report"})
    assert not result["passed"]
    assert result["active_nodes_with_invalid_ancestry"] == ["report"]
    assert replay_check(GRAPH, {"audio"}, [], invalid, {"independent_control"})["passed"]


def test_out_of_order_recomputation_fails():
    result = replay_check(GRAPH, {"state_a"}, ["report", "risk"], set(), {"report"})
    assert not result["passed"]
    assert result["out_of_order_recomputations"] == ["report"]


def test_source_removal_requires_recomputation_against_before_graph():
    before = {"source": [], "other": [], "risk": ["source", "other"], "report": ["risk"]}
    after = {**before, "risk": ["other"]}
    result = replay_check(before, {"source"}, [], {"source"}, {"report"}, revised_parents=after)
    assert not result["passed"]
    assert result["stale_descendants"] == ["report", "risk"]
    result = replay_check(before, {"source"}, ["risk", "report"], {"source"}, {"report"}, revised_parents=after)
    assert result["passed"]
    assert result["graph_transition_checked"]


@pytest.mark.parametrize("changed", [{"source"}, set()])
def test_recomputation_cannot_read_invalid_parent_even_when_not_active(changed):
    graph = {"source": [], "derived": ["source"]}
    result = replay_check(graph, changed, ["derived"], {"source"}, set())
    assert not result["passed"]
    assert result["recomputations_with_invalid_ancestry"] == ["derived"]


def test_whitespace_ids_and_mixed_datasets_are_rejected():
    frame = subject_frame()
    frame.loc[1, "subject_id"] = " " + frame.loc[0, "subject_id"] + " "
    with pytest.raises(ValueError, match="whitespace"):
        overlap_statistics(frame, METRICS)
    frame = subject_frame().assign(dataset_id="A")
    frame.loc[1, "dataset_id"] = "B"
    with pytest.raises(ValueError, match="dataset"):
        overlap_statistics(frame, METRICS)


def test_row_task_stratification_prevents_pooled_task_correlation():
    frame = subject_frame(60)
    frame["task_type"] = ["read"] * 30 + ["recall"] * 30
    frame["silence_fraction"] = list(range(30)) + list(range(100, 130))
    frame["voiced_fraction"] = list(range(29, -1, -1)) + list(range(129, 99, -1))
    _, pairs, _ = overlap_statistics(frame, METRICS)
    assert len(pairs) == 2
    assert set(pairs.rho.round(5)) == {-1.}
    assert pairs.task_conditioned.all()


@pytest.mark.parametrize("language", ["NaN", "N/A", "None", " null "])
def test_missing_language_sentinels_never_claim_conditioning(language):
    frame = subject_frame().assign(language=language)
    _, pairs, _ = overlap_statistics(frame, METRICS)
    assert pairs.iloc[0].language == "unknown"
    assert not pairs.iloc[0].language_conditioned


@pytest.mark.parametrize("graph,changed,recomputed", [
    ({"x": ["y"]}, {"x"}, []),
    ({"x": ["y"], "y": ["x"]}, {"x"}, []),
    ({"x": []}, {"x"}, ["x", "x"]),
])
def test_invalid_graphs_and_logs_fail_closed(graph, changed, recomputed):
    with pytest.raises(ValueError):
        replay_check(graph, changed, recomputed, set(), set())
