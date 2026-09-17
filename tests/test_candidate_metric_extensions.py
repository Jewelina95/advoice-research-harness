import math

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
