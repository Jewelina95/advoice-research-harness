import json

import pandas as pd

from advoice.condition_c import _rebuild_cards
from advoice.states import build_state_cards_frame
from test_expected_evidence_coverage import _build, _configs, _subjects


def test_rebuild_recomputes_scores_but_preserves_trace(tmp_path):
    metrics, config = _configs()
    evidence = _build(tmp_path, _subjects(), metrics)
    expected, _ = build_state_cards_frame(evidence, config)
    old = expected.copy()
    old["state_z"] = 99.0
    old["report_state_z"] = -99.0
    old["confidence"] = 1.0
    old["evidence_segments"] = json.dumps([{"segment_id": "segment-1", "start_sec": 0., "end_sec": 5.}])
    old["trace_resolution"] = "segment"
    rebuilt = _rebuild_cards(evidence, config, old)
    ignore = ["evidence_segments", "trace_resolution"]
    pd.testing.assert_frame_equal(rebuilt.drop(columns=ignore), expected.drop(columns=ignore))
    assert rebuilt.evidence_segments.tolist() == old.evidence_segments.tolist()
