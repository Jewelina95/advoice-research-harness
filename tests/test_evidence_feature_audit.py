import numpy as np
import pandas as pd

from advoice.evidence_feature_audit import (
    benjamini_hochberg, candidate_metric_ids, choose_feature_counts,
    feature_inference, map_historical_metrics, stratified_bootstrap_indices,
)


METRICS = [
    {"id": "duration_sec", "state": "QC", "branch": "qc", "role": "qc_only"},
    {"id": "word_count", "state": "QC", "branch": "qc", "role": "qc_only"},
]


def history_frame():
    return pd.DataFrame({
        "state_id": ["S02", "S02"],
        "state_name": ["output", "output"],
        "metric_name": ["duration_sec", "unknown_old_metric"],
        "metric_definition": ["duration", "unknown"],
        "how_calculated": ["seconds", "unknown"],
        "current_evaluation_status": ["include_current_state_evaluation", "exclude_metric_not_available"],
    })


def test_mapping_preserves_rows_and_distinguishes_new_runtime_metrics():
    mapping, lineage, summary = map_historical_metrics(history_frame(), METRICS)
    assert len(mapping) == 2
    assert mapping.iloc[0].runtime_metric_ids == "duration_sec"
    assert mapping.iloc[1].disposition == "not_extracted_in_current_pipeline"
    assert lineage.set_index("runtime_metric_id").loc["word_count", "provenance_status"] == "new_runtime_metric"
    assert summary["count"].sum() >= 4


def test_benjamini_hochberg_is_monotone_in_rank_order():
    adjusted = benjamini_hochberg([.04, .001, .02, np.nan])
    ordered = adjusted[np.argsort([.04, .001, .02, np.nan])[:3]]
    assert np.all(np.diff(ordered) >= 0)
    assert np.isnan(adjusted[-1])


def test_feature_inference_uses_train_only():
    frame = pd.DataFrame({
        "split": ["train"] * 20 + ["test"] * 2,
        "label": ["HC"] * 10 + ["AD"] * 10 + ["HC", "AD"],
        "duration_sec": list(range(20)) + [9999, -9999],
    })

    def fake_kruskal(*groups):
        assert sum(len(group) for group in groups) == 20
        return 3.0, .05

    result = feature_inference(frame, ["duration_sec", "missing"], kruskal_fn=fake_kruskal)
    assert result.iloc[0].n_train == 20
    assert result.iloc[0].n_observed == 20
    assert result.iloc[1].status == "not_extracted"


def test_stratified_bootstrap_preserves_class_counts():
    labels = np.asarray([0, 0, 0, 1, 1])
    sampled = stratified_bootstrap_indices(labels, np.random.default_rng(7))
    assert len(sampled) == len(labels)
    assert np.bincount(labels[sampled]).tolist() == [3, 2]


def test_feature_count_grid_and_qc_exclusion():
    assert choose_feature_counts(10) == [1, 2, 4, 8, 10]
    assert candidate_metric_ids(METRICS) == []
