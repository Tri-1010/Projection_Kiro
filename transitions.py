"""
Transition matrix estimation with hierarchical shrinkage and tail pooling.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def prepare_transitions(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    segment_cols: List[str],
    states: List[str],
    absorbing_states: List[str]
) -> pd.DataFrame:
    """
    Prepare transition data from loan snapshots.
    
    Args:
        df: Input DataFrame with loan snapshots (loan_id x MOB).
        cfg: Configuration dict with column name mappings.
        segment_cols: List of segment column names.
        states: List of valid bucket/state values.
        absorbing_states: List of absorbing state names.
        
    Returns:
        DataFrame with columns: loan_id, mob, from_state, to_state, ead, cohort + segment_cols
    """
    df = df.copy()
    
    # Filter out rows with None/NaN bucket values
    df = df[df[cfg["bucket"]].notna()]
    
    # Filter to only valid states
    df = df[df[cfg["bucket"]].isin(states)]
    
    # Parse cohort from orig_date (YYYY-MM)
    df[cfg["cohort"]] = pd.to_datetime(df[cfg["orig_date"]]).dt.to_period("M").astype(str)
    
    # Handle duplicates: keep latest by cutoff if exists, else last occurrence
    if cfg["cutoff"] in df.columns:
        df = df.sort_values([cfg["loan_id"], cfg["mob"], cfg["cutoff"]])
    else:
        df = df.sort_values([cfg["loan_id"], cfg["mob"]])
    df = df.drop_duplicates(subset=[cfg["loan_id"], cfg["mob"]], keep="last")
    
    # Sort by loan_id, mob for next-state computation
    df = df.sort_values([cfg["loan_id"], cfg["mob"]]).reset_index(drop=True)
    
    # Compute from_state and to_state
    df[cfg["from_state"]] = df[cfg["bucket"]]
    df[cfg["to_state"]] = df.groupby(cfg["loan_id"])[cfg["bucket"]].shift(-1)
    
    # Auto-expand absorbing states: include worse delinquency states
    expanded_absorbing = set(absorbing_states)
    for state in states:
        for abs_state in absorbing_states:
            if "DPD" in abs_state and "DPD" in state:
                # Extract numeric part
                try:
                    abs_num = int(abs_state.replace("DPD", "").replace("+", ""))
                    state_num = int(state.replace("DPD", "").replace("+", ""))
                    if state_num >= abs_num:
                        expanded_absorbing.add(state)
                except ValueError:
                    pass
            elif state == abs_state:
                expanded_absorbing.add(state)
    
    # Force absorbing: if from_state is absorbing, to_state = from_state
    mask_absorbing = df[cfg["from_state"]].isin(expanded_absorbing)
    df.loc[mask_absorbing, cfg["to_state"]] = df.loc[mask_absorbing, cfg["from_state"]]
    
    # Filter: keep rows where mob in 0..(MAX_MOB-1) and to_state exists
    from config import MAX_MOB
    df = df[df[cfg["mob"]].between(0, MAX_MOB - 1)]
    df = df[df[cfg["to_state"]].notna()]
    
    # Filter to_state to valid states only
    df = df[df[cfg["to_state"]].isin(states)]
    
    # Select output columns
    output_cols = [
        cfg["loan_id"], cfg["mob"], cfg["from_state"], cfg["to_state"],
        cfg["ead"], cfg["cohort"]
    ] + segment_cols
    
    return df[output_cols].reset_index(drop=True)


def estimate_transition_matrices(
    df_trans: pd.DataFrame,
    cfg: Dict[str, Any],
    states: List[str],
    segment_levels: List[Tuple[str, List[str]]],
    max_mob: int,
    weight_mode: str,
    min_count: int,
    prior_strengths: Dict[str, float],
    tail_pool_start: Optional[int]
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]:
    """
    Estimate transition matrices with hierarchical shrinkage and tail pooling.
    Uses vectorized operations for performance.
    """
    n_states = len(states)
    state_idx = {s: i for i, s in enumerate(states)}
    
    # Identify absorbing states (auto-expand)
    from config import ABSORBING_BASE
    absorbing_states = set(ABSORBING_BASE)
    for state in states:
        for abs_state in ABSORBING_BASE:
            if "DPD" in abs_state and "DPD" in state:
                try:
                    abs_num = int(abs_state.replace("DPD", "").replace("+", ""))
                    state_num = int(state.replace("DPD", "").replace("+", ""))
                    if state_num >= abs_num:
                        absorbing_states.add(state)
                except ValueError:
                    pass
            elif state == abs_state:
                absorbing_states.add(state)
    # Add terminal states
    for s in ["CLOSED", "WRITEOFF", "PREPAY"]:
        if s in states:
            absorbing_states.add(s)
    
    transitions_dict = {}
    long_records = []
    meta_records = []
    
    def build_matrix_from_counts(counts_matrix, absorbing_states, states):
        """Build probability matrix from counts, enforcing absorbing and row sums."""
        n = len(states)
        P = np.zeros((n, n))
        for i, from_s in enumerate(states):
            row_sum = counts_matrix[i, :].sum()
            if from_s in absorbing_states:
                P[i, :] = 0.0
                P[i, i] = 1.0
            elif row_sum > 0:
                P[i, :] = counts_matrix[i, :] / row_sum
            else:
                P[i, :] = 0.0
                P[i, i] = 1.0
        return P
    
    def matrix_to_df(P, states):
        return pd.DataFrame(P, index=states, columns=states)
    
    def compute_counts_vectorized(df_mob, weight_col, state_idx, n_states):
        """Compute transition counts using vectorized operations."""
        counts = np.zeros((n_states, n_states))
        
        if len(df_mob) == 0:
            return counts
        
        # Map states to indices
        from_idx = df_mob[cfg["from_state"]].map(state_idx)
        to_idx = df_mob[cfg["to_state"]].map(state_idx)
        
        # Filter valid indices
        valid_mask = from_idx.notna() & to_idx.notna()
        if valid_mask.sum() == 0:
            return counts
            
        from_idx = from_idx[valid_mask].astype(int).values
        to_idx = to_idx[valid_mask].astype(int).values
        
        if weight_col:
            weights = df_mob.loc[valid_mask, weight_col].fillna(0).values
        else:
            weights = np.ones(len(from_idx))
        
        # Use numpy add.at for fast aggregation
        for i in range(len(from_idx)):
            counts[from_idx[i], to_idx[i]] += weights[i]
        
        return counts
    
    # Build level hierarchy
    level_dict = {name: cols for name, cols in segment_levels}
    
    # Weight column
    weight_col = cfg["ead"] if weight_mode == "ead" else None
    
    # Process each MOB
    for mob in range(max_mob):
        df_mob = df_trans[df_trans[cfg["mob"]] == mob]
        
        # GLOBAL level
        global_counts = compute_counts_vectorized(df_mob, weight_col, state_idx, n_states)
        P_global = build_matrix_from_counts(global_counts, absorbing_states, states)
        transitions_dict[("GLOBAL", "", mob)] = matrix_to_df(P_global, states)
        
        # Record meta
        n_trans = global_counts.sum()
        meta_records.append({
            "level": "GLOBAL",
            "segment_key": "",
            "mob": mob,
            "n_transitions": int(n_trans) if not np.isnan(n_trans) else 0,
            "n_rows_with_data": int((global_counts.sum(axis=1) > 0).sum())
        })
        
        # COARSE level
        if "COARSE" in level_dict and level_dict["COARSE"]:
            coarse_cols = level_dict["COARSE"]
            for seg_key, grp in df_mob.groupby(coarse_cols):
                seg_key_str = str(seg_key) if not isinstance(seg_key, tuple) else "|".join(str(s) for s in seg_key)
                
                seg_counts = compute_counts_vectorized(grp, weight_col, state_idx, n_states)
                
                # Hierarchical shrinkage
                tau_coarse = prior_strengths.get("coarse", 0)
                post_counts = seg_counts + tau_coarse * P_global
                
                P_coarse = build_matrix_from_counts(post_counts, absorbing_states, states)
                transitions_dict[("COARSE", seg_key_str, mob)] = matrix_to_df(P_coarse, states)
                
                meta_records.append({
                    "level": "COARSE",
                    "segment_key": seg_key_str,
                    "mob": mob,
                    "n_transitions": int(seg_counts.sum()) if not np.isnan(seg_counts.sum()) else 0,
                    "n_rows_with_data": int((seg_counts.sum(axis=1) > 0).sum())
                })
        
        # FULL level
        if "FULL" in level_dict and level_dict["FULL"]:
            full_cols = level_dict["FULL"]
            coarse_cols = level_dict.get("COARSE", [])
            
            for seg_key, grp in df_mob.groupby(full_cols):
                seg_key_str = str(seg_key) if not isinstance(seg_key, tuple) else "|".join(str(s) for s in seg_key)
                
                # Get parent matrix
                if coarse_cols:
                    if isinstance(seg_key, tuple):
                        parent_key = "|".join(str(s) for s in seg_key[:len(coarse_cols)])
                    else:
                        parent_key = str(seg_key)
                    parent_matrix = transitions_dict.get(("COARSE", parent_key, mob))
                    P_parent = parent_matrix.values if parent_matrix is not None else P_global
                else:
                    P_parent = P_global
                
                seg_counts = compute_counts_vectorized(grp, weight_col, state_idx, n_states)
                
                # Hierarchical shrinkage
                tau_full = prior_strengths.get("full", 0)
                post_counts = seg_counts + tau_full * P_parent
                
                P_full = build_matrix_from_counts(post_counts, absorbing_states, states)
                transitions_dict[("FULL", seg_key_str, mob)] = matrix_to_df(P_full, states)
                
                meta_records.append({
                    "level": "FULL",
                    "segment_key": seg_key_str,
                    "mob": mob,
                    "n_transitions": int(seg_counts.sum()) if not np.isnan(seg_counts.sum()) else 0,
                    "n_rows_with_data": int((seg_counts.sum(axis=1) > 0).sum())
                })
    
    # Tail pooling
    if tail_pool_start is not None and tail_pool_start < max_mob:
        tail_mobs = list(range(tail_pool_start, max_mob))
        
        level_seg_keys = set()
        for key in transitions_dict.keys():
            level, seg_key, mob = key
            level_seg_keys.add((level, seg_key))
        
        for level, seg_key in level_seg_keys:
            tail_matrices = []
            for mob in tail_mobs:
                key = (level, seg_key, mob)
                if key in transitions_dict:
                    tail_matrices.append(transitions_dict[key].values)
            
            if tail_matrices:
                pooled = np.mean(tail_matrices, axis=0)
                
                # Re-enforce absorbing and row sums
                for i, from_s in enumerate(states):
                    if from_s in absorbing_states:
                        pooled[i, :] = 0.0
                        pooled[i, i] = 1.0
                    else:
                        row_sum = pooled[i, :].sum()
                        if row_sum > 0:
                            pooled[i, :] /= row_sum
                        else:
                            pooled[i, :] = 0.0
                            pooled[i, i] = 1.0
                
                pooled_df = matrix_to_df(pooled, states)
                for mob in tail_mobs:
                    transitions_dict[(level, seg_key, mob)] = pooled_df.copy()
    
    # Build long form (simplified - just from transitions_dict)
    for key, P in transitions_dict.items():
        level, seg_key, mob = key
        for from_s in states:
            for to_s in states:
                long_records.append({
                    "level": level,
                    "segment_key": seg_key,
                    "mob": mob,
                    "from_state": from_s,
                    "to_state": to_s,
                    "prob": P.loc[from_s, to_s],
                })
    
    transitions_long_df = pd.DataFrame(long_records)
    meta_df = pd.DataFrame(meta_records)
    
    return transitions_dict, transitions_long_df, meta_df
