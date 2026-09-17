import numpy as np
import pytest

from advoice.evidence import recalibrate_metric_evidence_frame
from advoice.states import build_fold_calibrated_state_frame, build_state_cards_frame
from test_expected_evidence_coverage import _build, _configs, _subjects


@pytest.mark.parametrize("value", [np.inf, -np.inf, np.nan])
def test_fold_scoring_does_not_convert_nonfinite_value_to_abnormal_state(tmp_path, value):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects(), metrics)
    mask = evidence.subject_id.eq("case") & evidence.metric_id.eq("silence_fraction")
    evidence.loc[mask, "value"] = value
    model = build_fold_calibrated_state_frame(evidence, config, {"hc1", "hc2"}, "HC")
    recalibrated = recalibrate_metric_evidence_frame(
        evidence, reference_subject_ids={"hc1", "hc2"}, target_subject_ids={"case"},
    )
    _, cards = build_state_cards_frame(recalibrated, config)
    for column in ("state_S01", "rel_S01"):
        actual = model.set_index("subject_id").loc["case", column]
        assert actual == cards.iloc[0][column] == 0.0
