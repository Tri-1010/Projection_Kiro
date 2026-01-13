"""
Sanity check script with synthetic data generation.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from transitions import prepare_transitions, estimate_transition_matrices
from forecast import build_initial_vectors, forecast
from metrics import compute_del_from_snapshot, compute_del_from_forecast, make_mixed_report

# Synthetic data config (different from actual data config)
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
SYNTH_MAX_MOB = 24


def generate_synthetic_data(n_loans=1000, n_cohorts=6, max_mob=12):
    """Generate synthetic loan data for testing."""
    np.random.seed(42)
    
    records = []
    loan_id = 0
    
    base_date = datetime(2024, 1, 1)
    
    for cohort_idx in range(n_cohorts):
        cohort_date = base_date + timedelta(days=30 * cohort_idx)
        n_loans_cohort = n_loans // n_cohorts
        
        for _ in range(n_loans_cohort):
            loan_id += 1
            ead = np.random.uniform(10000, 100000)
            state = "CURRENT"
            
            for mob in range(max_mob + 1):
                records.append({
                    "loan_id": loan_id,
                    "mob": mob,
                    "bucket": state,
                    "ead": ead,
                    "orig_date": cohort_date.strftime("%Y-%m-%d"),
                    "product": np.random.choice(["A", "B"]),
                    "channel": np.random.choice(["X", "Y"]),
                })
                
                # Transition logic
                if state == "CURRENT":
                    r = np.random.random()
                    if r < 0.02:
                        state = "DPD30"
                    elif r < 0.025:
                        state = "CLOSED"
                elif state == "DPD30":
                    r = np.random.random()
                    if r < 0.3:
                        state = "CURRENT"
                    elif r < 0.4:
                        state = "DPD60"
                    elif r < 0.42:
                        state = "CLOSED"
                elif state == "DPD60":
                    r = np.random.random()
                    if r < 0.2:
                        state = "DPD30"
                    elif r < 0.4:
                        state = "DPD90+"
                    elif r < 0.42:
                        state = "CLOSED"
                # DPD90+ and CLOSED are absorbing
    
    return pd.DataFrame(records)


def run_sanity_checks():
    """Run comprehensive sanity checks."""
    print("=" * 60)
    print("SANITY CHECK - Credit Risk Markov-Chain Projection Model")
    print("=" * 60)
    
    # Generate synthetic data
    print("\n1. Generating synthetic data...")
    df = generate_synthetic_data(n_loans=500, n_cohorts=4, max_mob=12)
    print(f"   Generated {len(df)} records, {df['loan_id'].nunique()} loans")
    
    # Prepare transitions
    print("\n2. Preparing transitions...")
    df_trans = prepare_transitions(df, SYNTH_CFG, SYNTH_SEGMENT_COLS, SYNTH_BUCKETS_CANON, SYNTH_ABSORBING_BASE)
    print(f"   Transitions: {len(df_trans)} records")
    
    # Check: to_state should follow from_state
    assert df_trans[SYNTH_CFG["to_state"]].notna().all(), "to_state has NaN values"
    print("   ✓ No NaN in to_state")
    
    # Estimate transition matrices
    print("\n3. Estimating transition matrices...")
    segment_levels = [
        ("GLOBAL", []),
        ("COARSE", [SYNTH_SEGMENT_COLS[0]]),
        ("FULL", SYNTH_SEGMENT_COLS),
    ]
    prior_strengths = {"coarse": 100.0, "full": 50.0}
    
    transitions_dict, transitions_long_df, meta_df = estimate_transition_matrices(
        df_trans, SYNTH_CFG, SYNTH_BUCKETS_CANON, segment_levels,
        max_mob=12, weight_mode="ead", min_count=30,
        prior_strengths=prior_strengths, tail_pool_start=10
    )
    print(f"   Matrices: {len(transitions_dict)} total")
    
    # Check: row sums = 1
    for key, P in transitions_dict.items():
        row_sums = P.sum(axis=1)
        assert np.allclose(row_sums, 1.0), f"Row sums != 1 for {key}"
    print("   ✓ All row sums = 1.0")
    
    # Check: absorbing rows are one-hot
    for key, P in transitions_dict.items():
        for state in SYNTH_ABSORBING_BASE:
            if state in P.index:
                row = P.loc[state]
                assert row[state] == 1.0, f"Absorbing state {state} not one-hot"
    print("   ✓ Absorbing rows are one-hot")
    
    # Build initial vectors
    print("\n4. Building initial vectors...")
    df_init, denom_map = build_initial_vectors(df, SYNTH_CFG, SYNTH_BUCKETS_CANON, SYNTH_SEGMENT_COLS, "cohort")
    print(f"   Initial vectors: {len(df_init)} records")
    
    # Forecast
    print("\n5. Forecasting...")
    forecast_df = forecast(df_init, transitions_dict, SYNTH_BUCKETS_CANON, max_mob=12)
    print(f"   Forecast: {len(forecast_df)} records")
    
    # Check: EAD conservation (approximately, due to CLOSED)
    for cohort in forecast_df["cohort"].unique():
        cohort_data = forecast_df[forecast_df["cohort"] == cohort]
        mob0_total = cohort_data[cohort_data["mob"] == 0]["ead"].sum()
        mob12_total = cohort_data[cohort_data["mob"] == 12]["ead"].sum()
        # Allow some loss due to CLOSED state
        assert mob12_total <= mob0_total * 1.01, "EAD increased unexpectedly"
    print("   ✓ EAD conservation check passed")
    
    # Compute DEL metrics
    print("\n6. Computing DEL metrics...")
    actual_del_long, _ = compute_del_from_snapshot(
        df, SYNTH_CFG, SYNTH_BUCKETS_30P, SYNTH_SEGMENT_COLS, max_mob=12, denom_level="cohort"
    )
    pred_del_long = compute_del_from_forecast(forecast_df, SYNTH_BUCKETS_30P, denom_map)
    print(f"   Actual DEL: {len(actual_del_long)} records")
    print(f"   Pred DEL: {len(pred_del_long)} records")
    
    # Check: DEL values in [0, 1]
    assert (actual_del_long["del_pct"] >= 0).all(), "Negative DEL values"
    assert (actual_del_long["del_pct"] <= 1).all(), "DEL > 100%"
    print("   ✓ DEL values in valid range")
    
    # Make mixed report
    print("\n7. Creating mixed report...")
    mixed_wide, flags_wide, actual_wide, forecast_wide = make_mixed_report(
        actual_del_long, pred_del_long, max_mob=12
    )
    print(f"   Mixed report: {len(mixed_wide)} rows")
    
    # Check: flags are ACTUAL or FORECAST
    for col in [c for c in flags_wide.columns if c.startswith("MOB_")]:
        valid_flags = flags_wide[col].isin(["ACTUAL", "FORECAST", ""])
        assert valid_flags.all(), f"Invalid flags in {col}"
    print("   ✓ Flags are valid")
    
    print("\n" + "=" * 60)
    print("ALL SANITY CHECKS PASSED!")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    success = run_sanity_checks()
    sys.exit(0 if success else 1)
