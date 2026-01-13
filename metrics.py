"""
Metrics computation for DEL curves and mixed reports.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional


def compute_del_from_snapshot(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    bad_states: List[str],
    segment_cols: List[str],
    max_mob: int,
    denom_level: str
) -> Tuple[pd.DataFrame, Dict]:
    """
    Compute DEL30 from snapshot data.
    
    Args:
        df_snapshot: Snapshot DataFrame.
        cfg: Configuration dict.
        bad_states: List of bad states (e.g., BUCKETS_30P).
        segment_cols: Segment columns.
        max_mob: Maximum MOB.
        denom_level: "cohort" or "cohort_segment".
        
    Returns:
        Tuple of:
        - actual_del_long: DataFrame with cohort, segment_key, mob, del_pct, denom_ead, numer_ead
        - denom_map: Dict mapping (cohort, segment_key) -> denom_ead
    """
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
    
    # Compute denominator from MOB=0
    df_mob0 = df[df[cfg["mob"]] == 0]
    
    if denom_level == "cohort":
        denom_df = df_mob0.groupby(cfg["cohort"])[cfg["ead"]].sum().reset_index()
        denom_df.columns = ["cohort", "denom_ead"]
    else:  # cohort_segment
        denom_df = df_mob0.groupby([cfg["cohort"], cfg["segment_key"]])[cfg["ead"]].sum().reset_index()
        denom_df.columns = ["cohort", "segment_key", "denom_ead"]
    
    # Build denom_map
    denom_map = {}
    if denom_level == "cohort":
        for _, row in denom_df.iterrows():
            # Will be applied to all segments in this cohort
            denom_map[row["cohort"]] = row["denom_ead"]
    else:
        for _, row in denom_df.iterrows():
            denom_map[(row["cohort"], row["segment_key"])] = row["denom_ead"]
    
    # Compute numerator (bad EAD) per cohort, segment, mob
    df["is_bad"] = df[cfg["bucket"]].isin(bad_states)
    numer_df = df[df["is_bad"]].groupby([cfg["cohort"], cfg["segment_key"], cfg["mob"]])[cfg["ead"]].sum().reset_index()
    numer_df.columns = ["cohort", "segment_key", "mob", "numer_ead"]
    
    # Get all cohort-segment-mob combinations
    all_combos = df.groupby([cfg["cohort"], cfg["segment_key"], cfg["mob"]]).size().reset_index()
    all_combos.columns = ["cohort", "segment_key", "mob", "_count"]
    all_combos = all_combos.drop(columns=["_count"])
    
    # Merge numerator
    result = all_combos.merge(numer_df, on=["cohort", "segment_key", "mob"], how="left")
    result["numer_ead"] = result["numer_ead"].fillna(0)
    
    # Merge denominator
    if denom_level == "cohort":
        result = result.merge(denom_df, on="cohort", how="left")
    else:
        result = result.merge(denom_df, on=["cohort", "segment_key"], how="left")
    
    result["denom_ead"] = result["denom_ead"].fillna(0)
    
    # Compute DEL percentage
    result["del_pct"] = np.where(
        result["denom_ead"] > 1e-10,
        result["numer_ead"] / result["denom_ead"],
        0.0
    )
    
    # Filter to max_mob
    result = result[result["mob"] <= max_mob]
    
    # Build proper denom_map for return
    denom_map_out = {}
    for _, row in result[["cohort", "segment_key", "denom_ead"]].drop_duplicates().iterrows():
        denom_map_out[(row["cohort"], row["segment_key"])] = row["denom_ead"]
    
    return result[["cohort", "segment_key", "mob", "del_pct", "denom_ead", "numer_ead"]], denom_map_out


def compute_del_from_forecast(
    forecast_df: pd.DataFrame,
    bad_states: List[str],
    denom_map: Dict
) -> pd.DataFrame:
    """
    Compute DEL30 from forecast data.
    
    Args:
        forecast_df: Forecast DataFrame (cohort, segment_key, mob, state, ead).
        bad_states: List of bad states.
        denom_map: Dict mapping (cohort, segment_key) -> denom_ead.
        
    Returns:
        DataFrame with cohort, segment_key, mob, del_pct, denom_ead, numer_ead
    """
    df = forecast_df.copy()
    
    # Compute numerator (bad EAD)
    df["is_bad"] = df["state"].isin(bad_states)
    numer_df = df[df["is_bad"]].groupby(["cohort", "segment_key", "mob"])["ead"].sum().reset_index()
    numer_df.columns = ["cohort", "segment_key", "mob", "numer_ead"]
    
    # Get all cohort-segment-mob combinations
    all_combos = df.groupby(["cohort", "segment_key", "mob"]).size().reset_index()
    all_combos.columns = ["cohort", "segment_key", "mob", "_count"]
    all_combos = all_combos.drop(columns=["_count"])
    
    # Merge numerator
    result = all_combos.merge(numer_df, on=["cohort", "segment_key", "mob"], how="left")
    result["numer_ead"] = result["numer_ead"].fillna(0)
    
    # Add denominator from map
    result["denom_ead"] = result.apply(
        lambda row: denom_map.get((row["cohort"], row["segment_key"]), 0),
        axis=1
    )
    
    # Compute DEL percentage
    result["del_pct"] = np.where(
        result["denom_ead"] > 1e-10,
        result["numer_ead"] / result["denom_ead"],
        0.0
    )
    
    return result[["cohort", "segment_key", "mob", "del_pct", "denom_ead", "numer_ead"]]


def make_mixed_report(
    actual_del_long: pd.DataFrame,
    pred_del_long: pd.DataFrame,
    max_mob: int,
    transitions_dict: Dict = None,
    states: List[str] = None,
    bad_states: List[str] = None,
    denom_map: Dict = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create mixed report with actual where available, forecast from last actual otherwise.
    Tạo báo cáo mixed: dùng actual nếu có, forecast từ actual gần nhất nếu không.
    
    QUAN TRỌNG: Forecast phải được tính từ actual gần nhất, không phải từ MOB=0.
    IMPORTANT: Forecast must be calculated from the last actual, not from MOB=0.
    
    Args:
        actual_del_long: Actual DEL DataFrame.
        pred_del_long: Predicted DEL DataFrame (used as fallback if no transitions_dict).
        max_mob: Maximum MOB.
        transitions_dict: Optional dict of transition matrices for re-forecasting.
        states: Optional list of states (required if transitions_dict provided).
        bad_states: Optional list of bad states (required if transitions_dict provided).
        denom_map: Optional dict mapping (cohort, segment_key) -> denom_ead.
        
    Returns:
        Tuple of:
        - mixed_wide: Wide DataFrame with MOB_0..MOB_24 columns (actual if exists, else forecast from last actual)
        - flags_wide: Wide DataFrame with "ACTUAL" or "FORECAST" flags
        - actual_wide: Wide DataFrame with actual values only
        - forecast_wide: Wide DataFrame with forecast values only
    """
    # Standardize column names
    actual = actual_del_long.copy()
    pred = pred_del_long.copy()
    
    # Ensure required columns
    for df in [actual, pred]:
        if "segment_key" not in df.columns:
            df["segment_key"] = ""
    
    # Create wide format for actual
    actual_pivot = actual.pivot_table(
        index=["cohort", "segment_key"],
        columns="mob",
        values="del_pct",
        aggfunc="first"
    ).reset_index()
    
    # Rename columns to MOB_X format
    actual_pivot.columns = [
        f"MOB_{c}" if isinstance(c, (int, np.integer)) else c
        for c in actual_pivot.columns
    ]
    
    # Create wide format for forecast
    pred_pivot = pred.pivot_table(
        index=["cohort", "segment_key"],
        columns="mob",
        values="del_pct",
        aggfunc="first"
    ).reset_index()
    
    pred_pivot.columns = [
        f"MOB_{c}" if isinstance(c, (int, np.integer)) else c
        for c in pred_pivot.columns
    ]
    
    # Ensure all MOB columns exist
    mob_cols = [f"MOB_{i}" for i in range(max_mob + 1)]
    for col in mob_cols:
        if col not in actual_pivot.columns:
            actual_pivot[col] = np.nan
        if col not in pred_pivot.columns:
            pred_pivot[col] = np.nan
    
    # Reorder columns
    id_cols = ["cohort", "segment_key"]
    actual_wide = actual_pivot[id_cols + mob_cols].copy()
    forecast_wide = pred_pivot[id_cols + mob_cols].copy()
    
    # Merge on cohort and segment_key
    merged = actual_wide.merge(
        forecast_wide,
        on=id_cols,
        how="outer",
        suffixes=("_actual", "_forecast")
    )
    
    # Build mixed and flags
    mixed_data = {col: [] for col in id_cols + mob_cols}
    flags_data = {col: [] for col in id_cols + mob_cols}
    
    for _, row in merged.iterrows():
        cohort = row["cohort"]
        segment_key = row["segment_key"]
        
        for col in id_cols:
            mixed_data[col].append(row[col])
            flags_data[col].append(row[col])
        
        # Track last actual MOB and value for forecasting
        last_actual_mob = -1
        last_actual_val = None
        
        # First pass: find all actual values and determine last actual MOB
        actual_values = {}
        for i, mob_col in enumerate(mob_cols):
            actual_val = row.get(f"{mob_col}_actual", np.nan)
            if pd.notna(actual_val):
                actual_values[i] = actual_val
                last_actual_mob = i
                last_actual_val = actual_val
        
        # Second pass: build mixed values
        for i, mob_col in enumerate(mob_cols):
            actual_val = row.get(f"{mob_col}_actual", np.nan)
            forecast_val = row.get(f"{mob_col}_forecast", np.nan)
            
            if pd.notna(actual_val):
                # Use actual value
                mixed_data[mob_col].append(actual_val)
                flags_data[mob_col].append("ACTUAL")
            elif i > last_actual_mob and last_actual_mob >= 0:
                # Need to forecast from last actual
                # Use the original forecast value but this is a known limitation
                # The proper fix requires re-running forecast from actual EAD distribution
                if pd.notna(forecast_val):
                    mixed_data[mob_col].append(forecast_val)
                    flags_data[mob_col].append("FORECAST")
                else:
                    mixed_data[mob_col].append(np.nan)
                    flags_data[mob_col].append("")
            elif pd.notna(forecast_val):
                mixed_data[mob_col].append(forecast_val)
                flags_data[mob_col].append("FORECAST")
            else:
                mixed_data[mob_col].append(np.nan)
                flags_data[mob_col].append("")
    
    mixed_wide = pd.DataFrame(mixed_data)
    flags_wide = pd.DataFrame(flags_data)
    
    # Rebuild actual_wide and forecast_wide with proper structure
    actual_wide = merged[id_cols].copy()
    forecast_wide = merged[id_cols].copy()
    
    for mob_col in mob_cols:
        actual_wide[mob_col] = merged.get(f"{mob_col}_actual", np.nan)
        forecast_wide[mob_col] = merged.get(f"{mob_col}_forecast", np.nan)
    
    return mixed_wide, flags_wide, actual_wide, forecast_wide



def compute_all_del_metrics(
    df_snapshot: pd.DataFrame,
    forecast_df: pd.DataFrame,
    cfg: Dict[str, Any],
    segment_cols: List[str],
    max_mob: int,
    denom_level: str,
    buckets_30p: List[str],
    buckets_60p: List[str],
    buckets_90p: List[str]
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Tính DEL30, DEL60, DEL90 và tạo mixed reports cho tất cả.
    Compute DEL30, DEL60, DEL90 and create mixed reports for all.
    
    Args:
        df_snapshot: Snapshot DataFrame.
        forecast_df: Forecast DataFrame.
        cfg: Configuration dict.
        segment_cols: Segment columns.
        max_mob: Maximum MOB.
        denom_level: "cohort" or "cohort_segment".
        buckets_30p: Bad states for DEL30 (e.g., ["DPD30+", "DPD60+", "DPD90+", "WRITEOFF"])
        buckets_60p: Bad states for DEL60 (e.g., ["DPD60+", "DPD90+", "WRITEOFF"])
        buckets_90p: Bad states for DEL90 (e.g., ["DPD90+", "WRITEOFF"])
        
    Returns:
        Dict with keys "DEL30", "DEL60", "DEL90", each containing:
        (mixed_wide, flags_wide, actual_wide, forecast_wide)
    """
    results = {}
    
    del_configs = [
        ("DEL30", buckets_30p),
        ("DEL60", buckets_60p),
        ("DEL90", buckets_90p),
    ]
    
    for del_name, bad_states in del_configs:
        # Compute actual DEL
        actual_del_long, denom_map = compute_del_from_snapshot(
            df_snapshot, cfg, bad_states, segment_cols, max_mob, denom_level
        )
        
        # Compute forecast DEL
        pred_del_long = compute_del_from_forecast(forecast_df, bad_states, denom_map)
        
        # Make mixed report
        mixed_wide, flags_wide, actual_wide, forecast_wide = make_mixed_report(
            actual_del_long, pred_del_long, max_mob
        )
        
        results[del_name] = (mixed_wide, flags_wide, actual_wide, forecast_wide)
    
    return results


def forecast_from_actual(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    transitions_dict: Dict,
    states: List[str],
    segment_cols: List[str],
    max_mob: int,
    bad_states: List[str],
    denom_map: Dict
) -> pd.DataFrame:
    """
    Forecast DEL từ actual EAD distribution tại mỗi MOB có actual data.
    Forecast DEL from actual EAD distribution at each MOB with actual data.
    
    Logic:
    - Với mỗi cohort-segment, tìm MOB cao nhất có actual data
    - Lấy actual EAD distribution tại MOB đó
    - Forecast từ MOB đó đến max_mob bằng transition matrices
    
    Args:
        df_snapshot: Snapshot DataFrame.
        cfg: Configuration dict.
        transitions_dict: Dict of transition matrices.
        states: List of states.
        segment_cols: Segment columns.
        max_mob: Maximum MOB.
        bad_states: List of bad states for DEL calculation.
        denom_map: Dict mapping (cohort, segment_key) -> denom_ead.
        
    Returns:
        DataFrame with cohort, segment_key, mob, del_pct, denom_ead, numer_ead
    """
    from forecast import get_best_matrix
    
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
    
    records = []
    
    # Group by cohort and segment
    for (cohort, segment_key), grp in df.groupby([cfg["cohort"], cfg["segment_key"]]):
        # Find max MOB with actual data
        max_actual_mob = grp[cfg["mob"]].max()
        
        # Get denom for this cohort-segment
        denom_ead = denom_map.get((cohort, segment_key), 0)
        if denom_ead <= 0:
            continue
        
        # Get actual EAD distribution at max_actual_mob
        actual_at_max = grp[grp[cfg["mob"]] == max_actual_mob]
        
        # Build EAD vector at max_actual_mob
        v = np.zeros(n_states)
        ead_by_state = actual_at_max.groupby(cfg["bucket"])[cfg["ead"]].sum().to_dict()
        for state, ead in ead_by_state.items():
            idx = state_idx.get(state)
            if idx is not None:
                v[idx] = ead
        
        # Record actual MOBs (0 to max_actual_mob)
        for mob in range(0, int(max_actual_mob) + 1):
            actual_at_mob = grp[grp[cfg["mob"]] == mob]
            ead_by_state_mob = actual_at_mob.groupby(cfg["bucket"])[cfg["ead"]].sum().to_dict()
            
            numer_ead = sum(ead_by_state_mob.get(s, 0) for s in bad_states)
            del_pct = numer_ead / denom_ead if denom_ead > 0 else 0
            
            records.append({
                "cohort": cohort,
                "segment_key": segment_key,
                "mob": mob,
                "del_pct": del_pct,
                "denom_ead": denom_ead,
                "numer_ead": numer_ead,
                "is_actual": True
            })
        
        # Forecast from max_actual_mob + 1 to max_mob
        for mob in range(int(max_actual_mob) + 1, max_mob + 1):
            # Get transition matrix for previous MOB
            P = get_best_matrix(transitions_dict, mob - 1, segment_key)
            if P is not None:
                P_arr = P.reindex(index=states, columns=states, fill_value=0).values
                v = v @ P_arr
            
            # Compute DEL from forecasted EAD distribution
            numer_ead = sum(v[state_idx[s]] for s in bad_states if s in state_idx)
            del_pct = numer_ead / denom_ead if denom_ead > 0 else 0
            
            records.append({
                "cohort": cohort,
                "segment_key": segment_key,
                "mob": mob,
                "del_pct": del_pct,
                "denom_ead": denom_ead,
                "numer_ead": numer_ead,
                "is_actual": False
            })
    
    return pd.DataFrame(records)


def make_mixed_report_v2(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    transitions_dict: Dict,
    states: List[str],
    segment_cols: List[str],
    max_mob: int,
    bad_states: List[str],
    denom_level: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create mixed report với forecast được tính từ actual EAD distribution gần nhất.
    Create mixed report with forecast calculated from the nearest actual EAD distribution.
    
    Đây là phiên bản đúng logic: forecast tiếp tục từ actual gần nhất,
    không phải forecast từ MOB=0.
    
    Args:
        df_snapshot: Snapshot DataFrame.
        cfg: Configuration dict.
        transitions_dict: Dict of transition matrices.
        states: List of states.
        segment_cols: Segment columns.
        max_mob: Maximum MOB.
        bad_states: List of bad states for DEL calculation.
        denom_level: "cohort" or "cohort_segment".
        
    Returns:
        Tuple of:
        - mixed_wide: Wide DataFrame with MOB_0..MOB_24 columns
        - flags_wide: Wide DataFrame with "ACTUAL" or "FORECAST" flags
        - actual_wide: Wide DataFrame with actual values only
        - forecast_wide: Wide DataFrame with forecast values only (from MOB=0)
    """
    # First compute actual DEL
    actual_del_long, denom_map = compute_del_from_snapshot(
        df_snapshot, cfg, bad_states, segment_cols, max_mob, denom_level
    )
    
    # Compute forecast from actual (this is the correct mixed data)
    mixed_del_long = forecast_from_actual(
        df_snapshot, cfg, transitions_dict, states, segment_cols, 
        max_mob, bad_states, denom_map
    )
    
    # Also compute pure forecast from MOB=0 for comparison
    from forecast import build_initial_vectors, forecast
    df_init, _ = build_initial_vectors(df_snapshot, cfg, states, segment_cols, denom_level)
    forecast_df = forecast(df_init, transitions_dict, states, max_mob)
    
    from metrics import compute_del_from_forecast
    pred_del_long = compute_del_from_forecast(forecast_df, bad_states, denom_map)
    
    # Create wide format for mixed (actual + forecast from actual)
    mixed_pivot = mixed_del_long.pivot_table(
        index=["cohort", "segment_key"],
        columns="mob",
        values="del_pct",
        aggfunc="first"
    ).reset_index()
    
    mixed_pivot.columns = [
        f"MOB_{c}" if isinstance(c, (int, np.integer)) else c
        for c in mixed_pivot.columns
    ]
    
    # Create flags from is_actual column
    flags_pivot = mixed_del_long.pivot_table(
        index=["cohort", "segment_key"],
        columns="mob",
        values="is_actual",
        aggfunc="first"
    ).reset_index()
    
    flags_pivot.columns = [
        f"MOB_{c}" if isinstance(c, (int, np.integer)) else c
        for c in flags_pivot.columns
    ]
    
    # Create wide format for actual only
    actual_pivot = actual_del_long.pivot_table(
        index=["cohort", "segment_key"],
        columns="mob",
        values="del_pct",
        aggfunc="first"
    ).reset_index()
    
    actual_pivot.columns = [
        f"MOB_{c}" if isinstance(c, (int, np.integer)) else c
        for c in actual_pivot.columns
    ]
    
    # Create wide format for pure forecast
    pred_pivot = pred_del_long.pivot_table(
        index=["cohort", "segment_key"],
        columns="mob",
        values="del_pct",
        aggfunc="first"
    ).reset_index()
    
    pred_pivot.columns = [
        f"MOB_{c}" if isinstance(c, (int, np.integer)) else c
        for c in pred_pivot.columns
    ]
    
    # Ensure all MOB columns exist
    mob_cols = [f"MOB_{i}" for i in range(max_mob + 1)]
    id_cols = ["cohort", "segment_key"]
    
    for col in mob_cols:
        if col not in mixed_pivot.columns:
            mixed_pivot[col] = np.nan
        if col not in flags_pivot.columns:
            flags_pivot[col] = np.nan
        if col not in actual_pivot.columns:
            actual_pivot[col] = np.nan
        if col not in pred_pivot.columns:
            pred_pivot[col] = np.nan
    
    # Reorder columns
    mixed_wide = mixed_pivot[id_cols + mob_cols].copy()
    actual_wide = actual_pivot[id_cols + mob_cols].copy()
    forecast_wide = pred_pivot[id_cols + mob_cols].copy()
    
    # Convert flags to ACTUAL/FORECAST strings
    flags_wide = flags_pivot[id_cols + mob_cols].copy()
    for col in mob_cols:
        flags_wide[col] = flags_wide[col].apply(
            lambda x: "ACTUAL" if x == True else ("FORECAST" if x == False else "")
        )
    
    return mixed_wide, flags_wide, actual_wide, forecast_wide


def compute_all_del_metrics_v2(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    transitions_dict: Dict,
    states: List[str],
    segment_cols: List[str],
    max_mob: int,
    denom_level: str,
    buckets_30p: List[str],
    buckets_60p: List[str],
    buckets_90p: List[str]
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Tính DEL30, DEL60, DEL90 với forecast được tính từ actual EAD distribution gần nhất.
    Compute DEL30, DEL60, DEL90 with forecast calculated from nearest actual EAD distribution.
    
    Đây là phiên bản đúng logic: forecast tiếp tục từ actual gần nhất.
    This is the correct logic version: forecast continues from the nearest actual.
    
    Args:
        df_snapshot: Snapshot DataFrame.
        cfg: Configuration dict.
        transitions_dict: Dict of transition matrices.
        states: List of states.
        segment_cols: Segment columns.
        max_mob: Maximum MOB.
        denom_level: "cohort" or "cohort_segment".
        buckets_30p: Bad states for DEL30.
        buckets_60p: Bad states for DEL60.
        buckets_90p: Bad states for DEL90.
        
    Returns:
        Dict with keys "DEL30", "DEL60", "DEL90", each containing:
        (mixed_wide, flags_wide, actual_wide, forecast_wide)
    """
    results = {}
    
    del_configs = [
        ("DEL30", buckets_30p),
        ("DEL60", buckets_60p),
        ("DEL90", buckets_90p),
    ]
    
    for del_name, bad_states in del_configs:
        mixed_wide, flags_wide, actual_wide, forecast_wide = make_mixed_report_v2(
            df_snapshot, cfg, transitions_dict, states, segment_cols,
            max_mob, bad_states, denom_level
        )
        
        results[del_name] = (mixed_wide, flags_wide, actual_wide, forecast_wide)
    
    return results
