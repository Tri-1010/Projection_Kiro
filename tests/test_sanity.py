"""
Unit tests for sanity checks.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np

from transitions import prepare_transitions, estimate_transition_matrices
from forecast import build_initial_vectors, forecast
from metrics import compute_del_from_snapshot, compute_del_from_forecast, make_mixed_report

# Synthetic data config
SYNTH_CFG = {
    "loan_id": "loan_id",
    "mob": "mob",
    "bucket": "bucket",
    "ead": "ead",
    "orig_date": "orig_date",
    "cutoff": "cutoff",
    "cohort": "cohort",
    "from_state": "from_state",
    "to_state": "to_state",
    "segment_key": "segment_key",
}

SYNTH_SEGMENT_COLS = ["product", "channel"]
SYNTH_BUCKETS_CANON = ["CURRENT", "DPD30", "DPD60", "DPD90+", "CLOSED"]
SYNTH_BUCKETS_30P = ["DPD30", "DPD60", "DPD90+"]
SYNTH_ABSORBING_BASE = ["DPD90+", "CLOSED"]


@pytest.fixture
def synthetic_data():
    """Generate minimal synthetic data for testing."""
    np.random.seed(42)
    records = []
    
    for loan_id in range(1, 11):
        for mob in range(5):
            records.append({
                "loan_id": loan_id,
                "mob": mob,
                "bucket": "CURRENT" if mob < 3 else "DPD30",
                "ead": 10000.0,
                "orig_date": "2024-01-01",
                "product": "A",
                "channel": "X",
            })
    
    return pd.DataFrame(records)


def test_prepare_transitions(synthetic_data):
    """Test prepare_transitions function."""
    df_trans = prepare_transitions(
        synthetic_data, SYNTH_CFG, SYNTH_SEGMENT_COLS, SYNTH_BUCKETS_CANON, SYNTH_ABSORBING_BASE
    )
    
    assert len(df_trans) > 0
    assert SYNTH_CFG["from_state"] in df_trans.columns
    assert SYNTH_CFG["to_state"] in df_trans.columns
    assert df_trans[SYNTH_CFG["to_state"]].notna().all()


def test_transition_matrix_row_sums(synthetic_data):
    """Test that transition matrix rows sum to 1."""
    df_trans = prepare_transitions(
        synthetic_data, SYNTH_CFG, SYNTH_SEGMENT_COLS, SYNTH_BUCKETS_CANON, SYNTH_ABSORBING_BASE
    )
    
    segment_levels = [("GLOBAL", [])]
    prior_strengths = {"coarse": 100.0, "full": 50.0}
    
    transitions_dict, _, _ = estimate_transition_matrices(
        df_trans, SYNTH_CFG, SYNTH_BUCKETS_CANON, segment_levels,
        max_mob=4, weight_mode="count", min_count=1,
        prior_strengths=prior_strengths, tail_pool_start=None
    )
    
    for key, P in transitions_dict.items():
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"Row sums != 1 for {key}"


def test_absorbing_states(synthetic_data):
    """Test that absorbing states have one-hot rows."""
    # Add some absorbing state data
    absorbing_records = []
    for loan_id in range(11, 16):
        for mob in range(5):
            absorbing_records.append({
                "loan_id": loan_id,
                "mob": mob,
                "bucket": "DPD90+" if mob >= 2 else "DPD60",
                "ead": 10000.0,
                "orig_date": "2024-01-01",
                "product": "A",
                "channel": "X",
            })
    
    df = pd.concat([synthetic_data, pd.DataFrame(absorbing_records)], ignore_index=True)
    df_trans = prepare_transitions(df, SYNTH_CFG, SYNTH_SEGMENT_COLS, SYNTH_BUCKETS_CANON, SYNTH_ABSORBING_BASE)
    
    segment_levels = [("GLOBAL", [])]
    prior_strengths = {"coarse": 100.0, "full": 50.0}
    
    transitions_dict, _, _ = estimate_transition_matrices(
        df_trans, SYNTH_CFG, SYNTH_BUCKETS_CANON, segment_levels,
        max_mob=4, weight_mode="count", min_count=1,
        prior_strengths=prior_strengths, tail_pool_start=None
    )
    
    for key, P in transitions_dict.items():
        for state in ["DPD90+", "CLOSED"]:
            if state in P.index:
                row = P.loc[state]
                assert row[state] == 1.0, f"Absorbing state {state} not one-hot"


def test_del_computation(synthetic_data):
    """Test DEL computation."""
    actual_del_long, denom_map = compute_del_from_snapshot(
        synthetic_data, SYNTH_CFG, SYNTH_BUCKETS_30P, SYNTH_SEGMENT_COLS, max_mob=4, denom_level="cohort"
    )
    
    assert len(actual_del_long) > 0
    assert "del_pct" in actual_del_long.columns
    assert (actual_del_long["del_pct"] >= 0).all()
    assert (actual_del_long["del_pct"] <= 1).all()


def test_mixed_report_flags(synthetic_data):
    """Test mixed report flags."""
    actual_del_long, denom_map = compute_del_from_snapshot(
        synthetic_data, SYNTH_CFG, SYNTH_BUCKETS_30P, SYNTH_SEGMENT_COLS, max_mob=4, denom_level="cohort"
    )
    
    # Create dummy forecast
    pred_del_long = actual_del_long.copy()
    pred_del_long["del_pct"] = pred_del_long["del_pct"] * 1.1
    
    mixed_wide, flags_wide, actual_wide, forecast_wide = make_mixed_report(
        actual_del_long, pred_del_long, max_mob=4
    )
    
    # Check flags are valid
    for col in [c for c in flags_wide.columns if c.startswith("MOB_")]:
        valid_flags = flags_wide[col].isin(["ACTUAL", "FORECAST", ""])
        assert valid_flags.all(), f"Invalid flags in {col}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
