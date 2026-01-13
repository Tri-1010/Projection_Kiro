"""
Excel export utilities.
"""
import pandas as pd
from typing import Optional


def _sanitize_sheet_name(name: str, max_len: int = 31) -> str:
    """Sanitize sheet name for Excel (max 31 chars, no special chars)."""
    # Remove invalid characters
    invalid_chars = [":", "\\", "/", "?", "*", "[", "]"]
    for char in invalid_chars:
        name = name.replace(char, "_")
    # Truncate to max length
    return name[:max_len]


def _ensure_unique_names(names: list) -> list:
    """Ensure sheet names are unique by appending numbers if needed."""
    seen = {}
    result = []
    for name in names:
        if name in seen:
            seen[name] += 1
            result.append(f"{name[:28]}_{seen[name]}")
        else:
            seen[name] = 0
            result.append(name)
    return result


def export_to_excel(
    path: str,
    transitions_long_df: pd.DataFrame,
    mixed_wide: pd.DataFrame,
    flags_wide: pd.DataFrame,
    actual_wide: pd.DataFrame,
    forecast_wide: pd.DataFrame,
    factors_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    meta_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Export results to Excel workbook with multiple sheets.
    
    Args:
        path: Output file path.
        transitions_long_df: Long-form transition probabilities.
        mixed_wide: Mixed DEL report (actual where available, else forecast).
        flags_wide: Flags indicating ACTUAL or FORECAST.
        actual_wide: Actual DEL values.
        forecast_wide: Forecast DEL values.
        factors_df: Optional calibration factors.
        forecast_df: Optional long-form forecast.
        meta_df: Optional segment metadata.
    """
    # Define sheets and their data
    sheets = [
        ("transitions_long", transitions_long_df),
        ("segment_meta", meta_df),
        ("del30_mixed", mixed_wide),
        ("del30_flags", flags_wide),
        ("del30_actual", actual_wide),
        ("del30_forecast", forecast_wide),
        ("calibration_factors", factors_df),
        ("forecast_long", forecast_df),
    ]
    
    # Filter out None DataFrames and sanitize names
    sheets = [(name, df) for name, df in sheets if df is not None]
    sheet_names = [_sanitize_sheet_name(name) for name, _ in sheets]
    sheet_names = _ensure_unique_names(sheet_names)
    
    # Write to Excel
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        for (_, df), sheet_name in zip(sheets, sheet_names):
            df.to_excel(writer, sheet_name=sheet_name, index=False)
