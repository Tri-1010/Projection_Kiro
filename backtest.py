"""
Backtest utilities with cohort split and metrics.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional

from config import (
    CFG, SEGMENT_COLS, BUCKETS_CANON, BUCKETS_30P, ABSORBING_BASE,
    MAX_MOB, MIN_COUNT, WEIGHT_MODE, PRIOR_STRENGTH_FULL, PRIOR_STRENGTH_COARSE,
    TAIL_POOL_START, K_CLIP, DENOM_LEVEL
)
from transitions import prepare_transitions, estimate_transition_matrices
from forecast import build_initial_vectors, forecast
from calibration import fit_del_curve_factors
from metrics import compute_del_from_snapshot, compute_del_from_forecast, make_mixed_report


def split_cohorts(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    train_ratio: float = 0.7
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data by cohort into train and test sets.
    
    Args:
        df: Input DataFrame.
        cfg: Configuration dict.
        train_ratio: Fraction of cohorts for training.
        
    Returns:
        Tuple of (train_df, test_df).
    """
    df = df.copy()
    df[cfg["cohort"]] = pd.to_datetime(df[cfg["orig_date"]]).dt.to_period("M").astype(str)
    
    cohorts = sorted(df[cfg["cohort"]].unique())
    n_train = int(len(cohorts) * train_ratio)
    
    train_cohorts = set(cohorts[:n_train])
    test_cohorts = set(cohorts[n_train:])
    
    train_df = df[df[cfg["cohort"]].isin(train_cohorts)]
    test_df = df[df[cfg["cohort"]].isin(test_cohorts)]
    
    return train_df, test_df


def compute_backtest_metrics(
    actual_del_long: pd.DataFrame,
    pred_del_long: pd.DataFrame,
    max_mob: int
) -> pd.DataFrame:
    """
    Compute backtest metrics (MAE, MAPE) by MOB.
    
    Args:
        actual_del_long: Actual DEL DataFrame.
        pred_del_long: Predicted DEL DataFrame.
        max_mob: Maximum MOB.
        
    Returns:
        DataFrame with mob, mae, mape, n_obs columns.
    """
    records = []
    
    for mob in range(max_mob + 1):
        actual_mob = actual_del_long[actual_del_long["mob"] == mob]
        pred_mob = pred_del_long[pred_del_long["mob"] == mob]
        
        if len(actual_mob) == 0 or len(pred_mob) == 0:
            records.append({
                "mob": mob,
                "mae": np.nan,
                "mape": np.nan,
                "n_obs": 0
            })
            continue
        
        # Merge on cohort and segment_key
        merged = actual_mob.merge(
            pred_mob,
            on=["cohort", "segment_key"],
            suffixes=("_actual", "_pred"),
            how="inner"
        )
        
        if len(merged) == 0:
            records.append({
                "mob": mob,
                "mae": np.nan,
                "mape": np.nan,
                "n_obs": 0
            })
            continue
        
        errors = merged["del_pct_actual"] - merged["del_pct_pred"]
        mae = np.abs(errors).mean()
        
        # MAPE with safe division
        mask = merged["del_pct_actual"] > 1e-10
        if mask.sum() > 0:
            mape = (np.abs(errors[mask]) / merged.loc[mask, "del_pct_actual"]).mean()
        else:
            mape = np.nan
        
        records.append({
            "mob": mob,
            "mae": mae,
            "mape": mape,
            "n_obs": len(merged)
        })
    
    return pd.DataFrame(records)


def run_backtest(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    segment_cols: List[str],
    states: List[str],
    bad_states: List[str],
    absorbing_states: List[str],
    max_mob: int,
    train_ratio: float = 0.7,
    calibrate: bool = False
) -> Dict[str, Any]:
    """
    Run full backtest pipeline with cohort split.
    
    Args:
        df: Input DataFrame.
        cfg: Configuration dict.
        segment_cols: Segment columns.
        states: List of states.
        bad_states: Bad states for DEL computation.
        absorbing_states: Absorbing states.
        max_mob: Maximum MOB.
        train_ratio: Fraction of cohorts for training.
        calibrate: Whether to apply calibration.
        
    Returns:
        Dict with backtest results.
    """
    # Split cohorts
    train_df, test_df = split_cohorts(df, cfg, train_ratio)
    
    # Prepare transitions from training data
    df_trans = prepare_transitions(train_df, cfg, segment_cols, states, absorbing_states)
    
    # Estimate transition matrices
    segment_levels = [
        ("GLOBAL", []),
        ("COARSE", [segment_cols[0]] if segment_cols else []),
        ("FULL", segment_cols),
    ]
    prior_strengths = {
        "coarse": PRIOR_STRENGTH_COARSE,
        "full": PRIOR_STRENGTH_FULL,
    }
    
    transitions_dict, transitions_long_df, meta_df = estimate_transition_matrices(
        df_trans, cfg, states, segment_levels,
        max_mob, WEIGHT_MODE, MIN_COUNT, prior_strengths, TAIL_POOL_START
    )
    
    # Build initial vectors from test data
    df_init, denom_map = build_initial_vectors(test_df, cfg, states, segment_cols, DENOM_LEVEL)
    
    # Forecast on test data
    forecast_df = forecast(df_init, transitions_dict, states, max_mob)
    
    # Compute actual DEL from test data
    actual_del_long, _ = compute_del_from_snapshot(
        test_df, cfg, bad_states, segment_cols, max_mob, DENOM_LEVEL
    )
    
    # Compute predicted DEL
    pred_del_long = compute_del_from_forecast(forecast_df, bad_states, denom_map)
    
    # Calibration (optional)
    factors_df = None
    if calibrate:
        # Fit factors on training data
        train_actual, train_denom = compute_del_from_snapshot(
            train_df, cfg, bad_states, segment_cols, max_mob, DENOM_LEVEL
        )
        train_init, _ = build_initial_vectors(train_df, cfg, states, segment_cols, DENOM_LEVEL)
        train_forecast = forecast(train_init, transitions_dict, states, max_mob)
        train_pred = compute_del_from_forecast(train_forecast, bad_states, train_denom)
        
        factors_df = fit_del_curve_factors(train_actual, train_pred, max_mob, K_CLIP)
    
    # Compute metrics
    metrics_df = compute_backtest_metrics(actual_del_long, pred_del_long, max_mob)
    
    # Make mixed report
    mixed_wide, flags_wide, actual_wide, forecast_wide = make_mixed_report(
        actual_del_long, pred_del_long, max_mob
    )
    
    return {
        "train_cohorts": sorted(train_df[cfg["cohort"]].unique()),
        "test_cohorts": sorted(test_df[cfg["cohort"]].unique()),
        "transitions_dict": transitions_dict,
        "transitions_long_df": transitions_long_df,
        "meta_df": meta_df,
        "forecast_df": forecast_df,
        "actual_del_long": actual_del_long,
        "pred_del_long": pred_del_long,
        "metrics_df": metrics_df,
        "factors_df": factors_df,
        "mixed_wide": mixed_wide,
        "flags_wide": flags_wide,
        "actual_wide": actual_wide,
        "forecast_wide": forecast_wide,
    }
