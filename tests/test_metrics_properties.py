"""
Property-based tests for metrics module.

Tests the following properties from the design document:
- Property 1: Forecast Starts from Last Actual EAD
- Property 2: Mixed Report Correctness
- Property 5: DEL Denominator Invariant

Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 3.4
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from metrics import (
    compute_del_from_snapshot,
    forecast_from_actual,
    make_mixed_report_v2,
)
from forecast import get_best_matrix


# Test configuration
TEST_CFG = {
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

TEST_STATES = ["CURRENT", "DPD30+", "DPD60+", "DPD90+", "WRITEOFF"]
TEST_BAD_STATES = ["DPD30+", "DPD60+", "DPD90+", "WRITEOFF"]
TEST_SEGMENT_COLS = ["product"]


def create_valid_transition_matrix(states: list) -> pd.DataFrame:
    """Create a valid stochastic transition matrix (rows sum to 1)."""
    n = len(states)
    # Generate random positive values
    raw = np.random.rand(n, n) + 0.01
    # Normalize rows to sum to 1
    row_sums = raw.sum(axis=1, keepdims=True)
    P = raw / row_sums
    return pd.DataFrame(P, index=states, columns=states)


def create_synthetic_snapshot(
    n_loans: int,
    max_actual_mob: int,
    cohort: str = "2024-01",
    segment: str = "A"
) -> pd.DataFrame:
    """Create synthetic snapshot data for testing."""
    records = []
    for loan_id in range(1, n_loans + 1):
        for mob in range(max_actual_mob + 1):
            # Assign states based on MOB (simple progression)
            if mob < 2:
                bucket = "CURRENT"
            elif mob < 4:
                bucket = np.random.choice(["CURRENT", "DPD30+"], p=[0.8, 0.2])
            else:
                bucket = np.random.choice(["CURRENT", "DPD30+", "DPD60+"], p=[0.6, 0.3, 0.1])
            
            records.append({
                "loan_id": loan_id,
                "mob": mob,
                "bucket": bucket,
                "ead": 10000.0,
                "orig_date": f"{cohort}-01",
                "product": segment,
            })
    
    return pd.DataFrame(records)


def create_transitions_dict(states: list, max_mob: int, segment_key: str) -> dict:
    """Create a transitions dict with valid stochastic matrices."""
    transitions_dict = {}
    for mob in range(max_mob):
        P = create_valid_transition_matrix(states)
        transitions_dict[("FULL", segment_key, mob)] = P
        transitions_dict[("GLOBAL", "", mob)] = P
    return transitions_dict


# =============================================================================
# Property 1: Forecast Starts from Last Actual EAD
# =============================================================================
# **Feature: del-forecast-calibration, Property 1: Forecast Starts from Last Actual EAD**
# *For any* cohort-segment with actual data up to MOB m, the forecast at MOB m+1 
# SHALL be computed using the actual EAD distribution at MOB m as the starting vector.
# **Validates: Requirements 1.1, 1.2, 1.3**

@settings(max_examples=100, deadline=30000)
@given(
    n_loans=st.integers(min_value=3, max_value=10),
    max_actual_mob=st.integers(min_value=2, max_value=4),
    max_forecast_mob=st.integers(min_value=6, max_value=8),
)
def test_property_1_forecast_starts_from_last_actual_ead(
    n_loans: int,
    max_actual_mob: int,
    max_forecast_mob: int
):
    """
    Property 1: Forecast Starts from Last Actual EAD
    
    For any cohort-segment with actual data up to MOB m, the forecast at MOB m+1
    SHALL be computed using the actual EAD distribution at MOB m as the starting vector,
    not the EAD at MOB=0.
    
    **Validates: Requirements 1.1, 1.2, 1.3**
    """
    np.random.seed(42)
    
    # Create synthetic data
    df_snapshot = create_synthetic_snapshot(
        n_loans=n_loans,
        max_actual_mob=max_actual_mob,
        cohort="2024-01",
        segment="A"
    )
    
    # Create transitions dict
    transitions_dict = create_transitions_dict(
        TEST_STATES, max_forecast_mob, "A"
    )
    
    # Compute actual DEL to get denom_map
    actual_del_long, denom_map = compute_del_from_snapshot(
        df_snapshot, TEST_CFG, TEST_BAD_STATES, TEST_SEGMENT_COLS,
        max_mob=max_forecast_mob, denom_level="cohort_segment"
    )
    
    # Forecast from actual
    forecast_result = forecast_from_actual(
        df_snapshot, TEST_CFG, transitions_dict, TEST_STATES,
        TEST_SEGMENT_COLS, max_forecast_mob, TEST_BAD_STATES, denom_map
    )
    
    # Property check: For MOBs > max_actual_mob, is_actual should be False
    # and for MOBs <= max_actual_mob, is_actual should be True
    for _, row in forecast_result.iterrows():
        mob = row["mob"]
        is_actual = row["is_actual"]
        
        if mob <= max_actual_mob:
            assert is_actual == True, (
                f"MOB {mob} should be marked as ACTUAL (max_actual_mob={max_actual_mob})"
            )
        else:
            assert is_actual == False, (
                f"MOB {mob} should be marked as FORECAST (max_actual_mob={max_actual_mob})"
            )
    
    # Additional check: forecast should have data for all MOBs from 0 to max_forecast_mob
    mobs_in_result = set(forecast_result["mob"].unique())
    expected_mobs = set(range(max_forecast_mob + 1))
    assert mobs_in_result == expected_mobs, (
        f"Forecast should cover MOBs 0 to {max_forecast_mob}, got {mobs_in_result}"
    )


# =============================================================================
# Property 2: Mixed Report Correctness
# =============================================================================
# **Feature: del-forecast-calibration, Property 2: Mixed Report Correctness**
# *For any* cohort-segment and MOB, if actual data exists at that MOB, the mixed report
# SHALL contain the actual DEL value; otherwise, it SHALL contain the forecasted DEL value.
# **Validates: Requirements 1.4, 1.5**

@settings(max_examples=100, deadline=30000)
@given(
    n_loans=st.integers(min_value=3, max_value=10),
    max_actual_mob=st.integers(min_value=2, max_value=4),
    max_forecast_mob=st.integers(min_value=6, max_value=8),
)
def test_property_2_mixed_report_correctness(
    n_loans: int,
    max_actual_mob: int,
    max_forecast_mob: int
):
    """
    Property 2: Mixed Report Correctness
    
    For any cohort-segment and MOB, if actual data exists at that MOB, the mixed report
    SHALL contain the actual DEL value; otherwise, it SHALL contain the forecasted DEL
    value computed from the last actual EAD distribution.
    
    **Validates: Requirements 1.4, 1.5**
    """
    np.random.seed(42)
    
    # Create synthetic data
    df_snapshot = create_synthetic_snapshot(
        n_loans=n_loans,
        max_actual_mob=max_actual_mob,
        cohort="2024-01",
        segment="A"
    )
    
    # Create transitions dict
    transitions_dict = create_transitions_dict(
        TEST_STATES, max_forecast_mob, "A"
    )
    
    # Create mixed report
    mixed_wide, flags_wide, actual_wide, forecast_wide = make_mixed_report_v2(
        df_snapshot, TEST_CFG, transitions_dict, TEST_STATES,
        TEST_SEGMENT_COLS, max_forecast_mob, TEST_BAD_STATES, "cohort_segment"
    )
    
    # Property check: flags should be ACTUAL for MOBs with actual data, FORECAST otherwise
    mob_cols = [f"MOB_{i}" for i in range(max_forecast_mob + 1)]
    
    for idx, row in flags_wide.iterrows():
        for i, mob_col in enumerate(mob_cols):
            flag = row[mob_col]
            
            if i <= max_actual_mob:
                assert flag == "ACTUAL", (
                    f"MOB {i} should have flag ACTUAL (max_actual_mob={max_actual_mob}), got {flag}"
                )
            else:
                assert flag == "FORECAST", (
                    f"MOB {i} should have flag FORECAST (max_actual_mob={max_actual_mob}), got {flag}"
                )
    
    # Additional check: mixed values should match actual values where actual exists
    for idx in range(len(mixed_wide)):
        for i in range(max_actual_mob + 1):
            mob_col = f"MOB_{i}"
            mixed_val = mixed_wide.iloc[idx][mob_col]
            actual_val = actual_wide.iloc[idx][mob_col]
            
            # Both should be equal (or both NaN)
            if pd.notna(actual_val):
                assert np.isclose(mixed_val, actual_val, rtol=1e-9), (
                    f"Mixed value at MOB {i} should equal actual value: "
                    f"mixed={mixed_val}, actual={actual_val}"
                )


# =============================================================================
# Property 5: DEL Denominator Invariant
# =============================================================================
# **Feature: del-forecast-calibration, Property 5: DEL Denominator Invariant**
# *For any* DEL computation, the denominator SHALL be the sum of EAD at MOB=0 
# for the cohort (or cohort-segment), regardless of which MOB the DEL is being computed for.
# **Validates: Requirements 3.4**

@settings(max_examples=100, deadline=30000)
@given(
    n_loans=st.integers(min_value=3, max_value=10),
    max_mob=st.integers(min_value=3, max_value=6),
    ead_value=st.floats(min_value=1000.0, max_value=100000.0, allow_nan=False, allow_infinity=False),
)
def test_property_5_del_denominator_invariant(
    n_loans: int,
    max_mob: int,
    ead_value: float
):
    """
    Property 5: DEL Denominator Invariant
    
    For any DEL computation, the denominator SHALL be the sum of EAD at MOB=0
    for the cohort (or cohort-segment), regardless of which MOB the DEL is being computed for.
    
    **Validates: Requirements 3.4**
    """
    np.random.seed(42)
    
    # Create synthetic data with known EAD
    records = []
    for loan_id in range(1, n_loans + 1):
        for mob in range(max_mob + 1):
            bucket = "CURRENT" if mob < 2 else np.random.choice(["CURRENT", "DPD30+"])
            records.append({
                "loan_id": loan_id,
                "mob": mob,
                "bucket": bucket,
                "ead": ead_value,
                "orig_date": "2024-01-01",
                "product": "A",
            })
    
    df_snapshot = pd.DataFrame(records)
    
    # Compute DEL
    actual_del_long, denom_map = compute_del_from_snapshot(
        df_snapshot, TEST_CFG, TEST_BAD_STATES, TEST_SEGMENT_COLS,
        max_mob=max_mob, denom_level="cohort_segment"
    )
    
    # Expected denominator: n_loans * ead_value (sum of EAD at MOB=0)
    expected_denom = n_loans * ead_value
    
    # Property check: denom_ead should be the same for all MOBs
    unique_denoms = actual_del_long["denom_ead"].unique()
    
    # All denominators should be equal
    assert len(unique_denoms) == 1, (
        f"Denominator should be invariant across MOBs, got {len(unique_denoms)} unique values"
    )
    
    # Denominator should equal sum of EAD at MOB=0
    actual_denom = unique_denoms[0]
    assert np.isclose(actual_denom, expected_denom, rtol=1e-9), (
        f"Denominator should be {expected_denom} (n_loans * ead), got {actual_denom}"
    )
    
    # Additional check: DEL = numer_ead / denom_ead
    for _, row in actual_del_long.iterrows():
        if row["denom_ead"] > 0:
            expected_del = row["numer_ead"] / row["denom_ead"]
            assert np.isclose(row["del_pct"], expected_del, rtol=1e-9), (
                f"DEL should be numer/denom: expected {expected_del}, got {row['del_pct']}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
