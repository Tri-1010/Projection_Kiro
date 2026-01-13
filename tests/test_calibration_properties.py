"""
Property-based tests for calibration module.

Tests the following properties from the design document:
- Property 3: Step-wise Calibration Formula
- Property 4: Calibrated Matrix Row Sum

Validates: Requirements 2.1, 2.2, 2.3, 2.5
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import pandas as pd
import numpy as np
from hypothesis import given, strategies as st, settings, assume

from calibration import (
    apply_matrix_calibration,
    apply_stepwise_calibration_to_matrices,
    fit_stepwise_calibration_factors,
)


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
TEST_ABSORBING_STATES = ["DPD90+", "WRITEOFF"]
TEST_SEGMENT_COLS = ["product"]


def create_valid_transition_matrix(states: list, seed: int = None) -> pd.DataFrame:
    """Create a valid stochastic transition matrix (rows sum to 1)."""
    if seed is not None:
        np.random.seed(seed)
    n = len(states)
    raw = np.random.rand(n, n) + 0.01
    row_sums = raw.sum(axis=1, keepdims=True)
    P = raw / row_sums
    return pd.DataFrame(P, index=states, columns=states)


def create_transitions_dict(states: list, max_mob: int, segment_key: str, seed: int = None) -> dict:
    """Create a transitions dict with valid stochastic matrices."""
    transitions_dict = {}
    for mob in range(max_mob):
        P = create_valid_transition_matrix(states, seed=seed + mob if seed else None)
        transitions_dict[("FULL", segment_key, mob)] = P
        transitions_dict[("GLOBAL", "", mob)] = P.copy()
    return transitions_dict


def create_synthetic_snapshot(
    n_loans: int,
    max_actual_mob: int,
    cohort: str = "2024-01",
    segment: str = "A",
    seed: int = None
) -> pd.DataFrame:
    """Create synthetic snapshot data for testing."""
    if seed is not None:
        np.random.seed(seed)
    
    records = []
    for loan_id in range(1, n_loans + 1):
        for mob in range(max_actual_mob + 1):
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



# =============================================================================
# Property 4: Calibrated Matrix Row Sum
# =============================================================================
# **Feature: del-forecast-calibration, Property 4: Calibrated Matrix Row Sum**
# *For any* calibrated transition matrix, each row SHALL sum to 1.0 (within floating-point tolerance).
# **Validates: Requirements 2.3**

@settings(max_examples=100, deadline=30000)
@given(
    k_factor=st.floats(min_value=0.5, max_value=2.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_4_calibrated_matrix_row_sum(k_factor: float, seed: int):
    """
    Property 4: Calibrated Matrix Row Sum
    
    For any calibrated transition matrix, each row SHALL sum to 1.0
    (within floating-point tolerance).
    
    **Validates: Requirements 2.3**
    """
    np.random.seed(seed)
    
    # Create a valid transition matrix
    P = create_valid_transition_matrix(TEST_STATES, seed=seed)
    
    # Apply calibration
    P_calibrated = apply_matrix_calibration(
        P, TEST_BAD_STATES, k_factor, TEST_ABSORBING_STATES
    )
    
    # Property check: each row should sum to 1.0
    row_sums = P_calibrated.sum(axis=1)
    
    for state, row_sum in row_sums.items():
        assert np.isclose(row_sum, 1.0, rtol=1e-9), (
            f"Row '{state}' should sum to 1.0, got {row_sum} (k={k_factor})"
        )


@settings(max_examples=100, deadline=30000)
@given(
    max_mob=st.integers(min_value=3, max_value=8),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_4_stepwise_calibrated_matrices_row_sum(max_mob: int, seed: int):
    """
    Property 4: Calibrated Matrix Row Sum (via apply_stepwise_calibration_to_matrices)
    
    For any calibrated transition matrix produced by apply_stepwise_calibration_to_matrices,
    each row SHALL sum to 1.0 (within floating-point tolerance).
    
    **Validates: Requirements 2.3**
    """
    np.random.seed(seed)
    
    # Create transitions dict
    transitions_dict = create_transitions_dict(TEST_STATES, max_mob, "A", seed=seed)
    
    # Create factors DataFrame with random k values
    factors_df = pd.DataFrame({
        "mob": list(range(max_mob + 1)),
        "k": np.random.uniform(0.5, 2.0, max_mob + 1),
    })
    
    # Apply stepwise calibration
    calibrated_dict = apply_stepwise_calibration_to_matrices(
        transitions_dict, factors_df, TEST_BAD_STATES, TEST_ABSORBING_STATES
    )
    
    # Property check: each row in each calibrated matrix should sum to 1.0
    for key, P_calibrated in calibrated_dict.items():
        row_sums = P_calibrated.sum(axis=1)
        
        for state, row_sum in row_sums.items():
            assert np.isclose(row_sum, 1.0, rtol=1e-9), (
                f"Matrix {key}, row '{state}' should sum to 1.0, got {row_sum}"
            )



# =============================================================================
# Property 3: Step-wise Calibration Formula
# =============================================================================
# **Feature: del-forecast-calibration, Property 3: Step-wise Calibration Formula**
# *For any* MOB m > 0, the calibration factor k[m] SHALL equal actual_del[m] / expected_del[m],
# where expected_del[m] is computed from actual_ead[m-1] @ P[m-1], clipped to (k_min, k_max).
# **Validates: Requirements 2.1, 2.2, 2.5**

@settings(max_examples=100, deadline=60000)
@given(
    n_loans=st.integers(min_value=5, max_value=15),
    max_actual_mob=st.integers(min_value=3, max_value=6),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_3_stepwise_calibration_formula(
    n_loans: int,
    max_actual_mob: int,
    seed: int
):
    """
    Property 3: Step-wise Calibration Formula
    
    For any MOB m > 0, the calibration factor k[m] SHALL equal actual_del[m] / expected_del[m],
    where expected_del[m] is computed from actual_ead[m-1] @ P[m-1], clipped to (k_min, k_max).
    
    **Validates: Requirements 2.1, 2.2, 2.5**
    """
    np.random.seed(seed)
    
    k_clip = (0.5, 2.0)
    k_min, k_max = k_clip
    
    # Create synthetic data
    df_snapshot = create_synthetic_snapshot(
        n_loans=n_loans,
        max_actual_mob=max_actual_mob,
        cohort="2024-01",
        segment="A",
        seed=seed
    )
    
    # Create transitions dict
    transitions_dict = create_transitions_dict(
        TEST_STATES, max_actual_mob + 2, "A", seed=seed
    )
    
    # Fit stepwise calibration factors
    factors_df = fit_stepwise_calibration_factors(
        df_snapshot, TEST_CFG, transitions_dict, TEST_STATES,
        TEST_SEGMENT_COLS, TEST_BAD_STATES, max_actual_mob,
        "cohort_segment", k_clip
    )
    
    # Property checks:
    # 1. k values should be within clip bounds
    for _, row in factors_df.iterrows():
        mob = row["mob"]
        k = row["k"]
        
        assert k_min <= k <= k_max, (
            f"k[{mob}] = {k} should be within [{k_min}, {k_max}]"
        )
    
    # 2. For MOB=0, k should be 1.0 (no transition)
    k_mob0 = factors_df[factors_df["mob"] == 0]["k"].values[0]
    assert k_mob0 == 1.0, f"k[0] should be 1.0, got {k_mob0}"
    
    # 3. For MOBs with data, k = actual_mean / expected_mean (before clipping)
    for _, row in factors_df.iterrows():
        mob = row["mob"]
        k = row["k"]
        expected_mean = row["expected_mean"]
        actual_mean = row["actual_mean"]
        n_cohorts = row["n_cohorts_used"]
        
        if mob == 0 or n_cohorts == 0:
            continue
        
        # Compute expected k (before clipping)
        if expected_mean > 1e-10:
            expected_k = actual_mean / expected_mean
            expected_k_clipped = np.clip(expected_k, k_min, k_max)
            
            assert np.isclose(k, expected_k_clipped, rtol=1e-9), (
                f"k[{mob}] = {k} should equal clipped(actual/expected) = "
                f"clipped({actual_mean}/{expected_mean}) = {expected_k_clipped}"
            )


@settings(max_examples=100, deadline=60000)
@given(
    k_min=st.floats(min_value=0.1, max_value=0.9, allow_nan=False, allow_infinity=False),
    k_max=st.floats(min_value=1.1, max_value=3.0, allow_nan=False, allow_infinity=False),
    seed=st.integers(min_value=0, max_value=10000),
)
def test_property_3_k_clipping_bounds(k_min: float, k_max: float, seed: int):
    """
    Property 3: Step-wise Calibration Formula - K Clipping
    
    When k is outside the clip range, the Calibration_Engine SHALL clip k to (k_min, k_max) bounds.
    
    **Validates: Requirements 2.5**
    """
    np.random.seed(seed)
    
    k_clip = (k_min, k_max)
    n_loans = 10
    max_actual_mob = 4
    
    # Create synthetic data
    df_snapshot = create_synthetic_snapshot(
        n_loans=n_loans,
        max_actual_mob=max_actual_mob,
        cohort="2024-01",
        segment="A",
        seed=seed
    )
    
    # Create transitions dict
    transitions_dict = create_transitions_dict(
        TEST_STATES, max_actual_mob + 2, "A", seed=seed
    )
    
    # Fit stepwise calibration factors
    factors_df = fit_stepwise_calibration_factors(
        df_snapshot, TEST_CFG, transitions_dict, TEST_STATES,
        TEST_SEGMENT_COLS, TEST_BAD_STATES, max_actual_mob,
        "cohort_segment", k_clip
    )
    
    # Property check: all k values should be within [k_min, k_max]
    for _, row in factors_df.iterrows():
        mob = row["mob"]
        k = row["k"]
        
        assert k_min <= k <= k_max, (
            f"k[{mob}] = {k} should be within [{k_min}, {k_max}]"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
