import numpy as np
import pandas as pd
import pytest

from advoice.condition_c import (
    _agent_feature_frame, _evidence_quality_by_subject, _fit_prototypes,
)


def test_case_features_do_not_depend_on_batch_companions():
    training = pd.DataFrame({"state_S01": [-1., -.5, .5, 1.], "label": ["HC", "HC", "AD", "AD"]})
    prototypes = _fit_prototypes(training, ["state_S01"], ["HC", "AD"])
    frame = pd.DataFrame({"subject_id": ["a", "b"], "state_S01": [.4, .6], "rel_S01": [.9, .9]})
    quality = {
        "a": {"metric_coverage": .25, "metric_reliability": .9, "confound_burden": 0.},
        "b": {"metric_coverage": 1., "metric_reliability": .9, "confound_burden": 0.},
    }
    one, _ = _agent_feature_frame(frame.iloc[:1], prototypes, ["HC", "AD"], np.array([[.4, .6]]), quality)
    batch, _ = _agent_feature_frame(frame, prototypes, ["HC", "AD"], np.array([[.4, .6], [.4, .6]]), quality)
    np.testing.assert_array_equal(one.iloc[0], batch.iloc[0])
    assert one.iloc[0].agent_evidence_coverage == pytest.approx(.5)


def test_coverage_counts_missing_expected_metrics_not_only_observed_rows():
    evidence = pd.DataFrame([
        {"subject_id": "a", "metric_id": f"m{i}", "evidence_role": "clinical_support",
         "report_permission": True, "missing": i > 0, "reliability": .9 if i == 0 else 0.,
         "confound_tags": "[]"}
        for i in range(4)
    ])
    quality = _evidence_quality_by_subject(evidence)
    assert quality["a"]["metric_coverage"] == pytest.approx(.25)
    evidence["missing"] = True
    assert _evidence_quality_by_subject(evidence)["a"]["metric_coverage"] == 0.


def test_csv_false_permissions_do_not_count_as_clinical_evidence():
    evidence = pd.DataFrame([{"subject_id": "a", "metric_id": "m", "evidence_role": "clinical_support",
                              "report_permission": "False", "missing": "False", "reliability": 1., "confound_tags": "[]"}])
    assert "a" not in _evidence_quality_by_subject(evidence)
