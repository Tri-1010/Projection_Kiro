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
    max_mob: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create mixed report with actual where available, forecast otherwise.
    
    Args:
        actual_del_long: Actual DEL DataFrame.
        pred_del_long: Predicted DEL DataFrame.
        max_mob: Maximum MOB.
        
    Returns:
        Tuple of:
        - mixed_wide: Wide DataFrame with MOB_0..MOB_24 columns (actual if exists, else forecast)
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
        for col in id_cols:
            mixed_data[col].append(row[col])
            flags_data[col].append(row[col])
        
        for mob_col in mob_cols:
            actual_val = row.get(f"{mob_col}_actual", np.nan)
            forecast_val = row.get(f"{mob_col}_forecast", np.nan)
            
            if pd.notna(actual_val):
                mixed_data[mob_col].append(actual_val)
                flags_data[mob_col].append("ACTUAL")
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
