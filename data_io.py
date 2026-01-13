"""
I/O utilities for loading and validating data.
"""
import pandas as pd
from typing import List, Dict, Any


def load_parquet(path: str) -> pd.DataFrame:
    """
    Load a parquet file into a DataFrame.
    
    Args:
        path: Path to the parquet file.
        
    Returns:
        DataFrame with loaded data.
    """
    return pd.read_parquet(path)


def validate_schema(
    df: pd.DataFrame,
    cfg: Dict[str, Any],
    segment_cols: List[str],
    states: List[str]
) -> None:
    """
    Validate that DataFrame has required columns and valid values.
    
    Args:
        df: Input DataFrame to validate.
        cfg: Configuration dict with column name mappings.
        segment_cols: List of segment column names.
        states: List of valid bucket/state values.
        
    Raises:
        ValueError: If validation fails.
    """
    required_cols = [
        cfg["loan_id"],
        cfg["mob"],
        cfg["bucket"],
        cfg["ead"],
        cfg["orig_date"],
    ]
    
    # Check required columns exist
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    # Check segment columns exist
    missing_seg = [c for c in segment_cols if c not in df.columns]
    if missing_seg:
        raise ValueError(f"Missing segment columns: {missing_seg}")
    
    # Check bucket values are valid (warn but don't fail for unknown values)
    actual_buckets = set(df[cfg["bucket"]].dropna().unique())
    invalid_buckets = actual_buckets - set(states)
    if invalid_buckets:
        print(f"Warning: Found bucket values not in states list: {invalid_buckets}")
        print(f"These will be filtered out during processing.")
    
    # Check MOB is non-negative
    if (df[cfg["mob"]] < 0).any():
        raise ValueError("MOB contains negative values")
    
    # Check EAD is non-negative
    if (df[cfg["ead"]] < 0).any():
        raise ValueError("EAD contains negative values")
