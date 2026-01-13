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


def fit_stepwise_calibration_factors(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    transitions_dict: Dict,
    states: List[str],
    segment_cols: List[str],
    bad_states: List[str],
    max_mob: int,
    denom_level: str,
    k_clip: Tuple[float, float]
) -> pd.DataFrame:
    """
    Tính hệ số calibration theo từng bước chuyển (step-wise).
    Calculate calibration factors for each transition step.
    
    Logic:
    - Với mỗi MOB m, tính:
      + actual_del[m]: DEL thực tế tại MOB m
      + expected_del[m]: DEL dự kiến = actual_ead[m-1] * P[m-1] → tính DEL
      + k_step[m] = actual_del[m] / expected_del[m]
    
    Cách này đo lường sai số của từng bước transition, không phải sai số tích lũy từ MOB=0.
    
    Args:
        df_snapshot: Snapshot DataFrame.
        cfg: Configuration dict.
        transitions_dict: Dict of transition matrices.
        states: List of states.
        segment_cols: Segment columns.
        bad_states: List of bad states for DEL calculation.
        max_mob: Maximum MOB.
        denom_level: "cohort" or "cohort_segment".
        k_clip: (k_min, k_max) for clipping factors.
        
    Returns:
        DataFrame with columns: mob, k, n_cohorts_used, expected_mean, actual_mean
    """
    from forecast import get_best_matrix
    
    k_min, k_max = k_clip
    
    df = df_snapshot.copy()
    
    # Filter out rows with None/NaN bucket values
    df = df[df[cfg["bucket"]].notna()]
    
    # Parse cohort
    df[cfg["cohort"]] = pd.to_datetime(df[cfg["orig_date"]]).dt.to_period("M").astype(str)
    
    # Build segment key
    if segment_cols:
        df[cfg["segment_key"]] = df[segment_cols].astype(str).agg("|".join, axis=1)
    else:
        df[cfg["segment_key"]] = ""
    
    state_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    
    # Compute denom from MOB=0
    df_mob0 = df[df[cfg["mob"]] == 0]
    if denom_level == "cohort":
        denom_df = df_mob0.groupby(cfg["cohort"])[cfg["ead"]].sum().reset_index()
        denom_df.columns = ["cohort", "denom_ead"]
    else:
        denom_df = df_mob0.groupby([cfg["cohort"], cfg["segment_key"]])[cfg["ead"]].sum().reset_index()
        denom_df.columns = ["cohort", "segment_key", "denom_ead"]
    
    # Build denom_map
    denom_map = {}
    if denom_level == "cohort":
        for _, row in denom_df.iterrows():
            denom_map[row["cohort"]] = row["denom_ead"]
    else:
        for _, row in denom_df.iterrows():
            denom_map[(row["cohort"], row["segment_key"])] = row["denom_ead"]
    
    # Collect step-wise data for each MOB
    step_data = {mob: {"actual_dels": [], "expected_dels": []} for mob in range(1, max_mob + 1)}
    
    # Group by cohort and segment
    for (cohort, segment_key), grp in df.groupby([cfg["cohort"], cfg["segment_key"]]):
        # Get denom
        if denom_level == "cohort":
            denom_ead = denom_map.get(cohort, 0)
        else:
            denom_ead = denom_map.get((cohort, segment_key), 0)
        
        if denom_ead <= 0:
            continue
        
        # Get max actual MOB for this cohort-segment
        max_actual_mob = grp[cfg["mob"]].max()
        
        # For each MOB from 1 to max_actual_mob, compute step-wise calibration
        for mob in range(1, int(max_actual_mob) + 1):
            # Get actual EAD distribution at MOB-1
            actual_prev = grp[grp[cfg["mob"]] == mob - 1]
            if len(actual_prev) == 0:
                continue
            
            # Build EAD vector at MOB-1
            v_prev = np.zeros(n_states)
            ead_by_state = actual_prev.groupby(cfg["bucket"])[cfg["ead"]].sum().to_dict()
            for state, ead in ead_by_state.items():
                idx = state_idx.get(state)
                if idx is not None:
                    v_prev[idx] = ead
            
            # Get transition matrix for MOB-1
            P = get_best_matrix(transitions_dict, mob - 1, segment_key)
            if P is None:
                continue
            
            # Forecast one step: v_expected = v_prev @ P
            P_arr = P.reindex(index=states, columns=states, fill_value=0).values
            v_expected = v_prev @ P_arr
            
            # Compute expected DEL
            expected_bad_ead = sum(v_expected[state_idx[s]] for s in bad_states if s in state_idx)
            expected_del = expected_bad_ead / denom_ead if denom_ead > 0 else 0
            
            # Get actual DEL at MOB
            actual_at_mob = grp[grp[cfg["mob"]] == mob]
            if len(actual_at_mob) == 0:
                continue
            
            actual_bad_ead = actual_at_mob[actual_at_mob[cfg["bucket"]].isin(bad_states)][cfg["ead"]].sum()
            actual_del = actual_bad_ead / denom_ead if denom_ead > 0 else 0
            
            step_data[mob]["actual_dels"].append(actual_del)
            step_data[mob]["expected_dels"].append(expected_del)
    
    # Compute k for each MOB
    records = []
    
    # MOB=0: k=1 (no transition)
    records.append({
        "mob": 0,
        "k": 1.0,
        "n_cohorts_used": 0,
        "expected_mean": 0.0,
        "actual_mean": 0.0
    })
    
    for mob in range(1, max_mob + 1):
        actual_dels = step_data[mob]["actual_dels"]
        expected_dels = step_data[mob]["expected_dels"]
        
        if len(actual_dels) == 0 or len(expected_dels) == 0:
            records.append({
                "mob": mob,
                "k": 1.0,
                "n_cohorts_used": 0,
                "expected_mean": 0.0,
                "actual_mean": 0.0
            })
            continue
        
        actual_mean = np.mean(actual_dels)
        expected_mean = np.mean(expected_dels)
        
        # Safe division
        if expected_mean > 1e-10:
            k = actual_mean / expected_mean
        else:
            k = 1.0
        
        # Clip
        k = np.clip(k, k_min, k_max)
        
        records.append({
            "mob": mob,
            "k": k,
            "n_cohorts_used": len(actual_dels),
            "expected_mean": expected_mean,
            "actual_mean": actual_mean
        })
    
    return pd.DataFrame(records)


def apply_stepwise_calibration_to_matrices(
    transitions_dict: Dict,
    factors_df: pd.DataFrame,
    bad_states: List[str],
    absorbing_states: List[str]
) -> Dict:
    """
    Áp dụng hệ số calibration step-wise vào transition matrices.
    Apply step-wise calibration factors to transition matrices.
    
    Args:
        transitions_dict: Dict of transition matrices.
        factors_df: DataFrame with mob, k columns.
        bad_states: List of bad states.
        absorbing_states: List of absorbing states.
        
    Returns:
        Dict of calibrated transition matrices.
    """
    k_by_mob = factors_df.set_index("mob")["k"].to_dict()
    
    calibrated_dict = {}
    
    for key, P in transitions_dict.items():
        level, segment_key, mob = key
        
        # Get k for this MOB (k is for transition from MOB to MOB+1)
        # So we use k[mob+1] to calibrate matrix at MOB
        k = k_by_mob.get(mob + 1, 1.0)
        
        calibrated_P = apply_matrix_calibration(P, bad_states, k, absorbing_states)
        calibrated_dict[key] = calibrated_P
    
    return calibrated_dict
