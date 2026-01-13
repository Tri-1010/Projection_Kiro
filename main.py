"""
Main CLI entry point for Credit Risk Markov-Chain Projection Model.
"""
import argparse
import os
import pandas as pd

from config import (
    CFG, SEGMENT_COLS, BUCKETS_CANON, BUCKETS_30P, ABSORBING_BASE,
    MAX_MOB, MIN_COUNT, WEIGHT_MODE, PRIOR_STRENGTH_FULL, PRIOR_STRENGTH_COARSE,
    TAIL_POOL_START, CALIBRATION_MODE, K_CLIP, DENOM_LEVEL
)
from data_io import load_parquet, validate_schema
from transitions import prepare_transitions, estimate_transition_matrices
from forecast import build_initial_vectors, forecast
from calibration import fit_del_curve_factors, apply_matrix_calibration
from metrics import compute_del_from_snapshot, compute_del_from_forecast, make_mixed_report
from export import export_to_excel


def main():
    parser = argparse.ArgumentParser(
        description="Credit Risk Markov-Chain Projection Model"
    )
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Path to input parquet file"
    )
    parser.add_argument(
        "--output", "-o",
        default="out/projection_report.xlsx",
        help="Path to output Excel file"
    )
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="Enable calibration"
    )
    parser.add_argument(
        "--max-mob",
        type=int,
        default=MAX_MOB,
        help=f"Maximum MOB to forecast (default: {MAX_MOB})"
    )
    
    args = parser.parse_args()
    
    # Create output directory if needed
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    
    # Load data
    print(f"Loading data from {args.input}...")
    df = load_parquet(args.input)
    validate_schema(df, CFG, SEGMENT_COLS, BUCKETS_CANON)
    
    # Prepare transitions
    print("Preparing transitions...")
    df_trans = prepare_transitions(df, CFG, SEGMENT_COLS, BUCKETS_CANON, ABSORBING_BASE)
    
    # Estimate transition matrices
    print("Estimating transition matrices...")
    segment_levels = [
        ("GLOBAL", []),
        ("COARSE", [SEGMENT_COLS[0]] if SEGMENT_COLS else []),
        ("FULL", SEGMENT_COLS),
    ]
    prior_strengths = {
        "coarse": PRIOR_STRENGTH_COARSE,
        "full": PRIOR_STRENGTH_FULL,
    }
    
    transitions_dict, transitions_long_df, meta_df = estimate_transition_matrices(
        df_trans, CFG, BUCKETS_CANON, segment_levels,
        args.max_mob, WEIGHT_MODE, MIN_COUNT, prior_strengths, TAIL_POOL_START
    )
    
    # Build initial vectors
    print("Building initial vectors...")
    df_init, denom_map = build_initial_vectors(df, CFG, BUCKETS_CANON, SEGMENT_COLS, DENOM_LEVEL)
    
    # Forecast
    print("Forecasting...")
    forecast_df = forecast(df_init, transitions_dict, BUCKETS_CANON, args.max_mob)
    
    # Compute DEL metrics
    print("Computing DEL metrics...")
    actual_del_long, _ = compute_del_from_snapshot(
        df, CFG, BUCKETS_30P, SEGMENT_COLS, args.max_mob, DENOM_LEVEL
    )
    pred_del_long = compute_del_from_forecast(forecast_df, BUCKETS_30P, denom_map)
    
    # Calibration (optional)
    factors_df = None
    if args.calibrate:
        print("Fitting calibration factors...")
        factors_df = fit_del_curve_factors(actual_del_long, pred_del_long, args.max_mob, K_CLIP)
    
    # Make mixed report
    print("Creating mixed report...")
    mixed_wide, flags_wide, actual_wide, forecast_wide = make_mixed_report(
        actual_del_long, pred_del_long, args.max_mob
    )
    
    # Export
    print(f"Exporting to {args.output}...")
    export_to_excel(
        args.output,
        transitions_long_df,
        mixed_wide,
        flags_wide,
        actual_wide,
        forecast_wide,
        factors_df=factors_df,
        forecast_df=forecast_df,
        meta_df=meta_df
    )
    
    print("Done!")


if __name__ == "__main__":
    main()
