from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from advoice.cache import StageCache
from advoice.data import NCMMSC_NAME
from advoice.evaluation import expected_calibration_error
from advoice.states import _state_category
from advoice.report_agent import _sanitized_segments


def test_ncmmsc_filename_contract() -> None:
    match = NCMMSC_NAME.match("AD_F_040349_003")
    assert match is not None
    assert match.groups() == ("AD", "F", "040349", "003")
    assert NCMMSC_NAME.match("MCI_M_061901-002") is not None
    assert NCMMSC_NAME.match("001") is None


def test_state_category_honors_reliability() -> None:
    assert _state_category(3.0, 0.2, 0.0) == "unreliable"
    assert _state_category(2.2, 0.9, 0.0) == "impaired"
    assert _state_category(1.2, 0.9, 0.0) == "borderline"
    assert _state_category(0.3, 0.9, 0.0) == "normal"


def test_ece_is_zero_for_perfect_predictions() -> None:
    labels = np.array(["HC", "MCI", "AD"])
    probability = np.eye(3)
    assert expected_calibration_error(labels, probability, 10) == 0.0


def test_stage_cache_skips_unchanged_inputs(tmp_path: Path) -> None:
    output = tmp_path / "value.txt"
    calls = []

    def write() -> None:
        calls.append(1)
        output.write_text("ok", encoding="utf-8")

    cache = StageCache(tmp_path / "cache")
    first = cache.execute("stage", [{"x": 1}], [output], write)
    second = cache.execute("stage", [{"x": 1}], [output], write)
    assert first.status == "executed"
    assert second.status == "cached"
    assert len(calls) == 1


def test_report_segments_remove_source_label_identifiers() -> None:
    raw = '[{"segment_id":"HC_F_019239_002:S06","case_id":"HC_F_019239_002","start_sec":50,"end_sec":60,"silence_fraction":0.8,"rms_db_mean":-32}]'
    result = _sanitized_segments(raw)
    assert result[0]["segment_id"].startswith("SEG-")
    assert "HC_F" not in str(result)
