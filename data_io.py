"""
I/O utilities for loading and validating data.
Hỗ trợ load single file hoặc tất cả parquet files trong thư mục.
"""
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Union
import glob


def load_parquet(path: Union[str, Path]) -> pd.DataFrame:
    """
    Load parquet file(s) into a DataFrame.
    Hỗ trợ load single file hoặc tất cả files trong thư mục.
    
    Args:
        path: Đường dẫn đến file parquet hoặc thư mục chứa các file parquet.
              - Nếu là file: load file đó
              - Nếu là thư mục: load tất cả *.parquet trong thư mục
              - Hỗ trợ glob pattern (e.g., "data/*.parquet")
        
    Returns:
        DataFrame với dữ liệu đã load (concat nếu nhiều files).
        
    Examples:
        # Load single file
        df = load_parquet("data/Oct25.parquet")
        
        # Load all parquet files in directory
        df = load_parquet("data/")
        
        # Load with glob pattern
        df = load_parquet("data/2024*.parquet")
    """
    path = Path(path) if isinstance(path, str) and not any(c in path for c in ['*', '?']) else path
    
    # Case 1: Glob pattern (contains * or ?)
    if isinstance(path, str) and any(c in path for c in ['*', '?']):
        files = sorted(glob.glob(path))
        if not files:
            raise FileNotFoundError(f"No parquet files found matching pattern: {path}")
        print(f"Loading {len(files)} parquet files from pattern: {path}")
        dfs = []
        for f in files:
            print(f"  - {f}")
            dfs.append(pd.read_parquet(f))
        return pd.concat(dfs, ignore_index=True)
    
    # Case 2: Directory
    if isinstance(path, Path) and path.is_dir():
        files = sorted(path.glob("*.parquet"))
        if not files:
            raise FileNotFoundError(f"No parquet files found in directory: {path}")
        print(f"Loading {len(files)} parquet files from directory: {path}")
        dfs = []
        for f in files:
            print(f"  - {f.name}")
            dfs.append(pd.read_parquet(f))
        return pd.concat(dfs, ignore_index=True)
    
    # Case 3: Single file
    path = Path(path) if isinstance(path, str) else path
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    print(f"Loading parquet file: {path}")
    return pd.read_parquet(path)


def load_parquet_files(paths: List[Union[str, Path]]) -> pd.DataFrame:
    """
    Load multiple parquet files và concat thành 1 DataFrame.
    
    Args:
        paths: List các đường dẫn file parquet.
        
    Returns:
        DataFrame với dữ liệu đã concat.
        
    Example:
        df = load_parquet_files(["data/Oct25.parquet", "data/Nov25.parquet"])
    """
    if not paths:
        raise ValueError("No paths provided")
    
    print(f"Loading {len(paths)} parquet files...")
    dfs = []
    for p in paths:
        print(f"  - {p}")
        dfs.append(pd.read_parquet(p))
    
    return pd.concat(dfs, ignore_index=True)


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
