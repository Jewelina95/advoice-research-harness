import math
from pathlib import Path

import pandas as pd
import yaml

from advoice.metric_governance import UNVALIDATED_CANDIDATE_METRICS
from advoice.models import _feature_columns
from advoice.transcripts import _immediate_repetition_rates, _mtld


def test_mtld_is_unavailable_for_short_samples():
    assert math.isnan(_mtld(["a"] * 9))


def test_mtld_distinguishes_repetition_from_diversity():
    repeated = _mtld(["same"] * 60)
    diverse = _mtld([f"word-{index}" for index in range(60)])
    assert repeated < diverse


def test_immediate_repetition_rates_are_explicitly_local():
    token_rate, bigram_rate = _immediate_repetition_rates(
        ["the", "boy", "boy", "takes", "a", "cookie", "a", "cookie"]
    )
    assert token_rate > 0
    assert bigram_rate > 0


def test_no_repetition_has_zero_rates():
    assert _immediate_repetition_rates(["one", "two", "three"]) == (0.0, 0.0)


def test_unvalidated_candidates_cannot_enter_model_feature_matrix():
    frame = pd.DataFrame({
        "subject_id": ["s1"],
        "label": ["HC"],
        "split": ["train"],
        "pause_sd_sec": [0.2],
        "task_cookie__pause_iqr_sec": [0.1],
        "lexical_mtld": [32.0],
        "repeat_token_rate_100w": [1.0],
        "repeat_bigram_rate_100w": [0.5],
        "silence_fraction": [0.3],
    })
    assert _feature_columns(frame) == ["silence_fraction"]


def test_research_candidate_config_matches_runtime_quarantine():
    root = Path(__file__).resolve().parents[1]
    payload = yaml.safe_load(
        (root / "configs/research/candidate_metric_extensions_2026-09-17.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert {item["id"] for item in payload["candidates"]} == set(
        UNVALIDATED_CANDIDATE_METRICS
    )
