import pandas as pd

from advoice.evidence_domain_audit_run import state_observability


def test_constant_columns_do_not_count_as_observable_evidence():
    coverage = pd.DataFrame({
        "dataset_id": ["A", "A"],
        "language": ["en", "en"],
        "configured_state": ["S13", "S13"],
        "status": ["constant", "unobserved"],
    })
    result = state_observability(coverage).iloc[0]
    assert result.observability_score == 0
    assert result.observability_status == "not_observable"


def test_variable_fraction_is_reported_without_rewarding_constants():
    coverage = pd.DataFrame({
        "dataset_id": ["A", "A", "A"],
        "language": ["en", "en", "en"],
        "configured_state": ["S12", "S12", "S12"],
        "status": ["variable", "constant", "variable"],
    })
    result = state_observability(coverage).iloc[0]
    assert result.observability_score == 2 / 3
    assert result.observability_status == "partially_observable"
