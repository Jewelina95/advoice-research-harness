"""Prediction eligibility for extracted evidence-governance candidates."""

from __future__ import annotations


UNVALIDATED_CANDIDATE_METRICS = frozenset({
    "pause_sd_sec",
    "pause_iqr_sec",
    "lexical_mtld",
    "repeat_token_rate_100w",
    "repeat_bigram_rate_100w",
})


def base_metric_id(column: str) -> str:
    """Return the registry ID for an overall or task-prefixed feature column."""

    return str(column).split("__", 1)[-1]


def model_feature_allowed(column: str) -> bool:
    """Keep research candidates observable while excluding them from prediction."""

    return base_metric_id(column) not in UNVALIDATED_CANDIDATE_METRICS


__all__ = [
    "UNVALIDATED_CANDIDATE_METRICS",
    "base_metric_id",
    "model_feature_allowed",
]
