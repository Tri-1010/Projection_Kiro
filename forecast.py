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


def map_forecast_to_loans(
    df_snapshot: pd.DataFrame,
    forecast_df: pd.DataFrame,
    cfg: Dict[str, Any],
    segment_cols: List[str],
    states: List[str],
    max_mob: int
) -> pd.DataFrame:
    """
    Map forecast EAD back to individual loans based on their proportion in cohort-segment.
    Phân bổ forecast EAD về từng khoản vay dựa trên tỷ lệ EAD trong cohort-segment.
    
    Logic:
    - Với mỗi loan tại MOB=0, tính tỷ lệ EAD của loan / tổng EAD cohort-segment
    - Forecast EAD của loan = tỷ lệ × forecast EAD của cohort-segment
    - Forecast state distribution được phân bổ theo tỷ lệ này
    
    Args:
        df_snapshot: Original snapshot DataFrame with loan_id.
        forecast_df: Forecast DataFrame (cohort, segment_key, mob, state, ead).
        cfg: Configuration dict.
        segment_cols: Segment columns.
        states: List of states.
        max_mob: Maximum MOB.
        
    Returns:
        DataFrame with columns: loan_id, cohort, segment_key, mob, state, ead_forecast, ead_ratio
    """
    df = df_snapshot.copy()
    
    # Parse cohort
    df[cfg["cohort"]] = pd.to_datetime(df[cfg["orig_date"]]).dt.to_period("M").astype(str)
    
    # Build segment key
    if segment_cols:
        df[cfg["segment_key"]] = df[segment_cols].astype(str).agg("|".join, axis=1)
    else:
        df[cfg["segment_key"]] = ""
    
    # Get loans at MOB=0 with their EAD
    df_mob0 = df[df[cfg["mob"]] == 0][[cfg["loan_id"], cfg["cohort"], cfg["segment_key"], cfg["ead"]]].copy()
    df_mob0.columns = ["loan_id", "cohort", "segment_key", "ead_loan"]
    
    # Compute total EAD per cohort-segment at MOB=0
    seg_totals = df_mob0.groupby(["cohort", "segment_key"])["ead_loan"].sum().reset_index()
    seg_totals.columns = ["cohort", "segment_key", "ead_total"]
    
    # Merge to get ratio
    df_mob0 = df_mob0.merge(seg_totals, on=["cohort", "segment_key"], how="left")
    df_mob0["ead_ratio"] = df_mob0["ead_loan"] / df_mob0["ead_total"]
    df_mob0["ead_ratio"] = df_mob0["ead_ratio"].fillna(0)
    
    # Pivot forecast to wide format for easier computation
    # forecast_df: cohort, segment_key, mob, state, ead
    forecast_wide = forecast_df.pivot_table(
        index=["cohort", "segment_key", "mob"],
        columns="state",
        values="ead",
        fill_value=0
    ).reset_index()
    
    # Merge loans with forecast
    records = []
    
    for _, loan_row in df_mob0.iterrows():
        loan_id = loan_row["loan_id"]
        cohort = loan_row["cohort"]
        segment_key = loan_row["segment_key"]
        ratio = loan_row["ead_ratio"]
        
        # Get forecast for this cohort-segment
        fct = forecast_wide[
            (forecast_wide["cohort"] == cohort) &
            (forecast_wide["segment_key"] == segment_key)
        ]
        
        for _, fct_row in fct.iterrows():
            mob = fct_row["mob"]
            for state in states:
                ead_seg = fct_row.get(state, 0)
                ead_loan = ead_seg * ratio
                records.append({
                    "loan_id": loan_id,
                    "cohort": cohort,
                    "segment_key": segment_key,
                    "mob": mob,
                    "state": state,
                    "ead_forecast": ead_loan,
                    "ead_ratio": ratio
                })
    
    return pd.DataFrame(records)


def merge_forecast_to_snapshot(
    df_snapshot: pd.DataFrame,
    forecast_df: pd.DataFrame,
    cfg: Dict[str, Any],
    segment_cols: List[str],
    states: List[str],
    bad_states_30p: List[str],
    bad_states_60p: List[str] = None,
    bad_states_90p: List[str] = None
) -> pd.DataFrame:
    """
    Merge forecast DEL metrics back to original snapshot by loan_id and MOB.
    Gộp forecast DEL vào data gốc theo loan_id và MOB.
    
    Thêm các cột:
    - ead_forecast_total: Tổng EAD forecast của loan tại MOB
    - ead_forecast_bad30: EAD forecast ở trạng thái xấu 30+ của loan
    - del30_forecast: DEL30 forecast = ead_bad30 / ead_at_mob0
    - (tương tự cho DEL60, DEL90 nếu có)
    
    Args:
        df_snapshot: Original snapshot DataFrame.
        forecast_df: Forecast DataFrame (cohort, segment_key, mob, state, ead).
        cfg: Configuration dict.
        segment_cols: Segment columns.
        states: List of states.
        bad_states_30p: Bad states for DEL30.
        bad_states_60p: Bad states for DEL60 (optional).
        bad_states_90p: Bad states for DEL90 (optional).
        
    Returns:
        DataFrame = df_snapshot with additional forecast columns.
    """
    df = df_snapshot.copy()
    
    # Parse cohort
    df[cfg["cohort"]] = pd.to_datetime(df[cfg["orig_date"]]).dt.to_period("M").astype(str)
    
    # Build segment key
    if segment_cols:
        df[cfg["segment_key"]] = df[segment_cols].astype(str).agg("|".join, axis=1)
    else:
        df[cfg["segment_key"]] = ""
    
    # Get EAD at MOB=0 per loan (denominator for DEL)
    df_mob0 = df[df[cfg["mob"]] == 0][[cfg["loan_id"], cfg["ead"]]].copy()
    df_mob0.columns = ["loan_id", "ead_mob0"]
    
    # Compute total EAD per cohort-segment at MOB=0
    df_temp = df[df[cfg["mob"]] == 0].copy()
    if segment_cols:
        df_temp[cfg["segment_key"]] = df_temp[segment_cols].astype(str).agg("|".join, axis=1)
    else:
        df_temp[cfg["segment_key"]] = ""
    df_temp[cfg["cohort"]] = pd.to_datetime(df_temp[cfg["orig_date"]]).dt.to_period("M").astype(str)
    
    seg_totals = df_temp.groupby([cfg["cohort"], cfg["segment_key"]])[cfg["ead"]].sum().reset_index()
    seg_totals.columns = ["cohort", "segment_key", "ead_seg_total"]
    
    # Compute loan ratio
    df_ratio = df_temp[[cfg["loan_id"], cfg["cohort"], cfg["segment_key"], cfg["ead"]]].copy()
    df_ratio.columns = ["loan_id", "cohort", "segment_key", "ead_loan"]
    df_ratio = df_ratio.merge(seg_totals, on=["cohort", "segment_key"], how="left")
    df_ratio["ead_ratio"] = df_ratio["ead_loan"] / df_ratio["ead_seg_total"]
    df_ratio["ead_ratio"] = df_ratio["ead_ratio"].fillna(0)
    
    # Aggregate forecast by cohort-segment-mob
    fct_agg = forecast_df.groupby(["cohort", "segment_key", "mob"]).apply(
        lambda g: pd.Series({
            "ead_fct_total": g["ead"].sum(),
            "ead_fct_bad30": g[g["state"].isin(bad_states_30p)]["ead"].sum(),
            "ead_fct_bad60": g[g["state"].isin(bad_states_60p)]["ead"].sum() if bad_states_60p else 0,
            "ead_fct_bad90": g[g["state"].isin(bad_states_90p)]["ead"].sum() if bad_states_90p else 0,
        })
    ).reset_index()
    
    # Merge ratio to main df
    df = df.merge(df_ratio[["loan_id", "ead_ratio"]], on="loan_id", how="left")
    df = df.merge(df_mob0, on="loan_id", how="left")
    
    # Merge forecast aggregates
    df = df.merge(
        fct_agg,
        left_on=[cfg["cohort"], cfg["segment_key"], cfg["mob"]],
        right_on=["cohort", "segment_key", "mob"],
        how="left",
        suffixes=("", "_fct")
    )
    
    # Compute loan-level forecast
    df["ead_forecast_total"] = df["ead_fct_total"] * df["ead_ratio"]
    df["ead_forecast_bad30"] = df["ead_fct_bad30"] * df["ead_ratio"]
    df["del30_forecast"] = df["ead_forecast_bad30"] / df["ead_mob0"]
    df["del30_forecast"] = df["del30_forecast"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if bad_states_60p:
        df["ead_forecast_bad60"] = df["ead_fct_bad60"] * df["ead_ratio"]
        df["del60_forecast"] = df["ead_forecast_bad60"] / df["ead_mob0"]
        df["del60_forecast"] = df["del60_forecast"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    if bad_states_90p:
        df["ead_forecast_bad90"] = df["ead_fct_bad90"] * df["ead_ratio"]
        df["del90_forecast"] = df["ead_forecast_bad90"] / df["ead_mob0"]
        df["del90_forecast"] = df["del90_forecast"].replace([np.inf, -np.inf], np.nan).fillna(0)
    
    # Clean up temp columns
    drop_cols = ["ead_fct_total", "ead_fct_bad30", "ead_fct_bad60", "ead_fct_bad90", 
                 "cohort_fct", "segment_key_fct", "mob_fct"]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
    
    return df
