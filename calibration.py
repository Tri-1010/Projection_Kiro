"""
Calibration functions for adjusting forecasts to match actual DEL curves.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def fit_del_curve_factors(
    actual_del_long: pd.DataFrame,
    pred_del_long: pd.DataFrame,
    max_mob: int,
    k_clip: Tuple[float, float]
) -> pd.DataFrame:
    """
    Fit calibration factors per MOB to match actual DEL curve.
    
    Args:
        actual_del_long: Actual DEL DataFrame (cohort, mob, del_pct, ...).
        pred_del_long: Predicted DEL DataFrame (cohort, mob, del_pct, ...).
        max_mob: Maximum MOB.
        k_clip: (k_min, k_max) for clipping factors.
        
    Returns:
        DataFrame with columns: mob, k, n_cohorts_used, pred_mean, actual_mean
    """
    k_min, k_max = k_clip
    records = []
    
    for mob in range(max_mob + 1):
        actual_mob = actual_del_long[actual_del_long["mob"] == mob]
        pred_mob = pred_del_long[pred_del_long["mob"] == mob]
        
        if len(actual_mob) == 0 or len(pred_mob) == 0:
            records.append({
                "mob": mob,
                "k": 1.0,
                "n_cohorts_used": 0,
                "pred_mean": 0.0,
                "actual_mean": 0.0
            })
            continue
        
        actual_mean = actual_mob["del_pct"].mean()
        pred_mean = pred_mob["del_pct"].mean()
        
        # Safe division
        if pred_mean > 1e-10:
            k = actual_mean / pred_mean
        else:
            k = 1.0
        
        # Clip
        k = np.clip(k, k_min, k_max)
        
        records.append({
            "mob": mob,
            "k": k,
            "n_cohorts_used": len(actual_mob),
            "pred_mean": pred_mean,
            "actual_mean": actual_mean
        })
    
    return pd.DataFrame(records)


def apply_matrix_calibration(
    P: pd.DataFrame,
    bad_states: List[str],
    k: float,
    absorbing_states: List[str]
) -> pd.DataFrame:
    """
    Apply calibration factor to transition matrix.
    
    Args:
        P: Transition matrix DataFrame (index=states, columns=states).
        bad_states: List of bad/delinquent states to scale.
        k: Calibration factor.
        absorbing_states: List of absorbing states (keep one-hot).
        
    Returns:
        Calibrated transition matrix DataFrame.
    """
    P2 = P.copy()
    states = list(P.index)
    bad_set = set(bad_states)
    absorbing_set = set(absorbing_states)
    
    for from_state in states:
        if from_state in absorbing_set:
            # Keep one-hot for absorbing
            P2.loc[from_state, :] = 0.0
            P2.loc[from_state, from_state] = 1.0
            continue
        
        row = P2.loc[from_state, :].values.astype(float)
        
        # Identify bad and good columns
        bad_mask = np.array([s in bad_set for s in states])
        good_mask = ~bad_mask
        
        # Scale bad probabilities
        bad_total = row[bad_mask].sum()
        good_total = row[good_mask].sum()
        
        if bad_total > 1e-10:
            new_bad_total = min(bad_total * k, 1.0 - 1e-10)  # Cap at near 1
            scale_bad = new_bad_total / bad_total
            row[bad_mask] *= scale_bad
            
            # Adjust good to maintain row sum = 1
            new_good_total = 1.0 - row[bad_mask].sum()
            if good_total > 1e-10 and new_good_total > 0:
                scale_good = new_good_total / good_total
                row[good_mask] *= scale_good
            elif new_good_total <= 0:
                row[good_mask] = 0.0
        
        # Ensure row sums to 1
        row_sum = row.sum()
        if row_sum > 1e-10:
            row /= row_sum
        else:
            # Identity fallback
            row[:] = 0.0
            row[states.index(from_state)] = 1.0
        
        P2.loc[from_state, :] = row
    
    return P2


def apply_vector_calibration(
    v: pd.Series,
    bad_states: List[str],
    k: float
) -> pd.Series:
    """
    Apply calibration factor to EAD vector.
    
    Args:
        v: EAD vector (Series with state index).
        bad_states: List of bad/delinquent states to scale.
        k: Calibration factor.
        
    Returns:
        Calibrated EAD vector (total EAD preserved).
    """
    v2 = v.copy().astype(float)
    states = list(v.index)
    bad_set = set(bad_states)
    
    bad_mask = np.array([s in bad_set for s in states])
    good_mask = ~bad_mask
    
    total_ead = v2.sum()
    bad_ead = v2[bad_mask].sum()
    good_ead = v2[good_mask].sum()
    
    if total_ead < 1e-10:
        return v2
    
    # Scale bad EAD
    new_bad_ead = min(bad_ead * k, total_ead - 1e-10)  # Cap at near total
    new_bad_ead = max(new_bad_ead, 0.0)
    
    if bad_ead > 1e-10:
        scale_bad = new_bad_ead / bad_ead
        v2.iloc[bad_mask] *= scale_bad
    
    # Adjust good to preserve total
    new_good_ead = total_ead - v2[bad_mask].sum()
    if good_ead > 1e-10 and new_good_ead > 0:
        scale_good = new_good_ead / good_ead
        v2.iloc[good_mask] *= scale_good
    elif new_good_ead <= 0:
        v2.iloc[good_mask] = 0.0
    
    return v2
