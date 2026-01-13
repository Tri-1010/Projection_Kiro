"""
Forecasting using transition matrices.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def build_initial_vectors(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    states: List[str],
    segment_cols: List[str],
    denom_level: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    Build initial EAD vectors from snapshot data at MOB=0.
    
    Args:
        df_snapshot: Snapshot DataFrame.
        cfg: Configuration dict.
        states: List of states.
        segment_cols: Segment columns.
        denom_level: "cohort" or "cohort_segment".
        
    Returns:
        Tuple of:
        - df_init: DataFrame with cohort, segment_key, state, ead
        - denom_map: Dict mapping (cohort, segment_key) -> total EAD at MOB=0
    """
    df = df_snapshot.copy()
    
    # Parse cohort
    df[cfg["cohort"]] = pd.to_datetime(df[cfg["orig_date"]]).dt.to_period("M").astype(str)
    
    # Filter to MOB=0
    df_mob0 = df[df[cfg["mob"]] == 0].copy()
    
    # Build segment key
    if segment_cols:
        df_mob0[cfg["segment_key"]] = df_mob0[segment_cols].astype(str).agg("|".join, axis=1)
    else:
        df_mob0[cfg["segment_key"]] = ""
    
    # Aggregate EAD by cohort, segment, state
    init_agg = df_mob0.groupby([cfg["cohort"], cfg["segment_key"], cfg["bucket"]])[cfg["ead"]].sum().reset_index()
    init_agg.columns = ["cohort", "segment_key", "state", "ead"]
    
    # Build denom_map
    denom_map = {}
    if denom_level == "cohort":
        cohort_totals = init_agg.groupby("cohort")["ead"].sum().to_dict()
        for _, row in init_agg.iterrows():
            key = (row["cohort"], row["segment_key"])
            denom_map[key] = cohort_totals[row["cohort"]]
    else:  # cohort_segment
        seg_totals = init_agg.groupby(["cohort", "segment_key"])["ead"].sum().to_dict()
        denom_map = seg_totals
    
    return init_agg, denom_map


def get_best_matrix(
    transitions_dict: Dict,
    mob: int,
    segment_key: str
) -> Optional[pd.DataFrame]:
    """
    Get the best available transition matrix with hierarchy fallback.
    
    Args:
        transitions_dict: Dict of transition matrices.
        mob: MOB value.
        segment_key: Segment key string.
        
    Returns:
        Best available matrix DataFrame, or None if not found.
    """
    # Try FULL first
    key = ("FULL", segment_key, mob)
    if key in transitions_dict:
        return transitions_dict[key]
    
    # Try COARSE (first part of segment key)
    if "|" in segment_key:
        coarse_key = segment_key.split("|")[0]
    else:
        coarse_key = segment_key
    key = ("COARSE", coarse_key, mob)
    if key in transitions_dict:
        return transitions_dict[key]
    
    # Fall back to GLOBAL
    key = ("GLOBAL", "", mob)
    if key in transitions_dict:
        return transitions_dict[key]
    
    return None


def forecast(
    df_init: pd.DataFrame,
    transitions_dict: Dict,
    states: List[str],
    max_mob: int,
    actual_override_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Forecast EAD distribution over MOBs using transition matrices.
    
    Args:
        df_init: Initial vectors DataFrame (cohort, segment_key, state, ead).
        transitions_dict: Dict of transition matrices.
        states: List of states.
        max_mob: Maximum MOB to forecast to.
        actual_override_df: Optional DataFrame with actual values to override forecast.
            Columns: cohort, segment_key, mob, state, ead
            
    Returns:
        DataFrame with columns: cohort, segment_key, mob, state, ead
    """
    state_idx = {s: i for i, s in enumerate(states)}
    n_states = len(states)
    
    records = []
    
    # Group by cohort and segment
    for (cohort, segment_key), grp in df_init.groupby(["cohort", "segment_key"]):
        # Build initial vector
        v = np.zeros(n_states)
        for _, row in grp.iterrows():
            idx = state_idx.get(row["state"])
            if idx is not None:
                v[idx] = row["ead"]
        
        # Record MOB=0
        for i, s in enumerate(states):
            records.append({
                "cohort": cohort,
                "segment_key": segment_key,
                "mob": 0,
                "state": s,
                "ead": v[i]
            })
        
        # Forecast MOB 1 to max_mob
        for mob in range(1, max_mob + 1):
            # Check for actual override
            if actual_override_df is not None:
                override = actual_override_df[
                    (actual_override_df["cohort"] == cohort) &
                    (actual_override_df["segment_key"] == segment_key) &
                    (actual_override_df["mob"] == mob)
                ]
                if len(override) > 0:
                    v = np.zeros(n_states)
                    for _, row in override.iterrows():
                        idx = state_idx.get(row["state"])
                        if idx is not None:
                            v[idx] = row["ead"]
                    for i, s in enumerate(states):
                        records.append({
                            "cohort": cohort,
                            "segment_key": segment_key,
                            "mob": mob,
                            "state": s,
                            "ead": v[i]
                        })
                    continue
            
            # Get transition matrix for previous MOB
            P = get_best_matrix(transitions_dict, mob - 1, segment_key)
            if P is not None:
                P_arr = P.reindex(index=states, columns=states, fill_value=0).values
                v = v @ P_arr
            
            for i, s in enumerate(states):
                records.append({
                    "cohort": cohort,
                    "segment_key": segment_key,
                    "mob": mob,
                    "state": s,
                    "ead": v[i]
                })
    
    return pd.DataFrame(records)
