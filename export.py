"""
Excel export utilities.
Xuất báo cáo Excel với sheet riêng cho mỗi product và sheet Portfolio tổng hợp.
"""
import pandas as pd
import numpy as np
from typing import Optional, List


def _sanitize_sheet_name(name: str, max_len: int = 31) -> str:
    """Sanitize sheet name for Excel (max 31 chars, no special chars)."""
    invalid_chars = [":", "\\", "/", "?", "*", "[", "]"]
    for char in invalid_chars:
        name = name.replace(char, "_")
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


def _compute_portfolio_del(
    mixed_wide: pd.DataFrame,
    actual_wide: pd.DataFrame,
    forecast_wide: pd.DataFrame,
    flags_wide: pd.DataFrame
) -> tuple:
    """
    Tính DEL cho Portfolio (tổng hợp tất cả products).
    Compute Portfolio DEL by aggregating all products.
    
    Returns:
        Tuple of (portfolio_mixed, portfolio_actual, portfolio_forecast, portfolio_flags)
    """
    mob_cols = [c for c in mixed_wide.columns if c.startswith("MOB_")]
    
    # Portfolio = mean across all segments per cohort
    # (Hoặc có thể dùng weighted average nếu có EAD weights)
    
    portfolio_mixed = mixed_wide.groupby("cohort")[mob_cols].mean().reset_index()
    portfolio_mixed.insert(1, "segment_key", "PORTFOLIO")
    
    portfolio_actual = actual_wide.groupby("cohort")[mob_cols].mean().reset_index()
    portfolio_actual.insert(1, "segment_key", "PORTFOLIO")
    
    portfolio_forecast = forecast_wide.groupby("cohort")[mob_cols].mean().reset_index()
    portfolio_forecast.insert(1, "segment_key", "PORTFOLIO")
    
    # Flags: ACTUAL nếu tất cả segments có ACTUAL, else FORECAST
    def agg_flags(group):
        result = {}
        for col in mob_cols:
            vals = group[col].values
            if all(v == "ACTUAL" for v in vals if v):
                result[col] = "ACTUAL"
            elif any(v == "ACTUAL" for v in vals if v):
                result[col] = "MIXED"
            else:
                result[col] = "FORECAST"
        return pd.Series(result)
    
    portfolio_flags = flags_wide.groupby("cohort").apply(agg_flags).reset_index()
    portfolio_flags.insert(1, "segment_key", "PORTFOLIO")
    
    return portfolio_mixed, portfolio_actual, portfolio_forecast, portfolio_flags


def _split_by_segment(
    df: pd.DataFrame,
    segment_col: str = "segment_key"
) -> dict:
    """
    Chia DataFrame theo segment_key thành dict.
    Split DataFrame by segment_key into dict.
    
    Returns:
        Dict[segment_key] -> DataFrame (without segment_key column)
    """
    if segment_col not in df.columns:
        return {"ALL": df}
    
    result = {}
    for seg_key, grp in df.groupby(segment_col):
        # Drop segment_key column for cleaner output
        grp_clean = grp.drop(columns=[segment_col]).reset_index(drop=True)
        result[str(seg_key)] = grp_clean
    
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
    Export results to Excel workbook with separate sheets per product.
    Xuất báo cáo Excel với sheet riêng cho mỗi product và sheet Portfolio.
    
    Sheet structure:
    - Portfolio_Mixed, Portfolio_Actual, Portfolio_Forecast, Portfolio_Flags
    - {Product}_Mixed, {Product}_Actual, {Product}_Forecast, {Product}_Flags (per product)
    - transitions_long, segment_meta, calibration_factors, forecast_long (metadata)
    
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
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        
        # === 1. PORTFOLIO SHEETS (tổng hợp tất cả products) ===
        portfolio_mixed, portfolio_actual, portfolio_forecast, portfolio_flags = \
            _compute_portfolio_del(mixed_wide, actual_wide, forecast_wide, flags_wide)
        
        portfolio_mixed.to_excel(writer, sheet_name="Portfolio_Mixed", index=False)
        portfolio_actual.to_excel(writer, sheet_name="Portfolio_Actual", index=False)
        portfolio_forecast.to_excel(writer, sheet_name="Portfolio_Forecast", index=False)
        portfolio_flags.to_excel(writer, sheet_name="Portfolio_Flags", index=False)
        
        # === 2. PER-PRODUCT SHEETS ===
        mixed_by_seg = _split_by_segment(mixed_wide)
        actual_by_seg = _split_by_segment(actual_wide)
        forecast_by_seg = _split_by_segment(forecast_wide)
        flags_by_seg = _split_by_segment(flags_wide)
        
        # Get all unique segments
        all_segments = sorted(set(mixed_by_seg.keys()) | set(actual_by_seg.keys()))
        
        for seg_key in all_segments:
            seg_name = _sanitize_sheet_name(seg_key, max_len=20)
            
            # Mixed
            if seg_key in mixed_by_seg:
                sheet_name = _sanitize_sheet_name(f"{seg_name}_Mixed")
                mixed_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Actual
            if seg_key in actual_by_seg:
                sheet_name = _sanitize_sheet_name(f"{seg_name}_Actual")
                actual_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Forecast
            if seg_key in forecast_by_seg:
                sheet_name = _sanitize_sheet_name(f"{seg_name}_Forecast")
                forecast_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
            
            # Flags
            if seg_key in flags_by_seg:
                sheet_name = _sanitize_sheet_name(f"{seg_name}_Flags")
                flags_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
        
        # === 3. METADATA SHEETS ===
        if transitions_long_df is not None:
            transitions_long_df.to_excel(writer, sheet_name="transitions_long", index=False)
        
        if meta_df is not None:
            meta_df.to_excel(writer, sheet_name="segment_meta", index=False)
        
        if factors_df is not None:
            factors_df.to_excel(writer, sheet_name="calibration_factors", index=False)
        
        if forecast_df is not None:
            forecast_df.to_excel(writer, sheet_name="forecast_long", index=False)



def export_all_del_to_excel(
    path: str,
    transitions_long_df: pd.DataFrame,
    del_results: dict,
    factors_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    meta_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Export DEL30, DEL60, DEL90 results to Excel with separate sheets per product.
    Xuất báo cáo DEL30, DEL60, DEL90 với sheet riêng cho mỗi product và Portfolio.
    
    Sheet structure:
    - DEL30_Portfolio, DEL30_{Product}_Mixed, DEL30_{Product}_Actual, etc.
    - DEL60_Portfolio, DEL60_{Product}_Mixed, DEL60_{Product}_Actual, etc.
    - DEL90_Portfolio, DEL90_{Product}_Mixed, DEL90_{Product}_Actual, etc.
    - Metadata sheets: transitions_long, segment_meta, calibration_factors, forecast_long
    
    Args:
        path: Output file path.
        transitions_long_df: Long-form transition probabilities.
        del_results: Dict from compute_all_del_metrics with keys "DEL30", "DEL60", "DEL90"
            Each value is (mixed_wide, flags_wide, actual_wide, forecast_wide)
        factors_df: Optional calibration factors.
        forecast_df: Optional long-form forecast.
        meta_df: Optional segment metadata.
    """
    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        
        # === PROCESS EACH DEL TYPE (DEL30, DEL60, DEL90) ===
        for del_type in ["DEL30", "DEL60", "DEL90"]:
            if del_type not in del_results:
                continue
                
            mixed_wide, flags_wide, actual_wide, forecast_wide = del_results[del_type]
            
            # === PORTFOLIO SHEETS ===
            portfolio_mixed, portfolio_actual, portfolio_forecast, portfolio_flags = \
                _compute_portfolio_del(mixed_wide, actual_wide, forecast_wide, flags_wide)
            
            # Write Portfolio sheets with DEL prefix
            sheet_name = _sanitize_sheet_name(f"{del_type}_Portfolio")
            portfolio_mixed.to_excel(writer, sheet_name=sheet_name, index=False)
            
            sheet_name = _sanitize_sheet_name(f"{del_type}_Port_Actual")
            portfolio_actual.to_excel(writer, sheet_name=sheet_name, index=False)
            
            sheet_name = _sanitize_sheet_name(f"{del_type}_Port_Forecast")
            portfolio_forecast.to_excel(writer, sheet_name=sheet_name, index=False)
            
            sheet_name = _sanitize_sheet_name(f"{del_type}_Port_Flags")
            portfolio_flags.to_excel(writer, sheet_name=sheet_name, index=False)
            
            # === PER-PRODUCT SHEETS ===
            mixed_by_seg = _split_by_segment(mixed_wide)
            actual_by_seg = _split_by_segment(actual_wide)
            forecast_by_seg = _split_by_segment(forecast_wide)
            flags_by_seg = _split_by_segment(flags_wide)
            
            all_segments = sorted(set(mixed_by_seg.keys()) | set(actual_by_seg.keys()))
            
            for seg_key in all_segments:
                # Truncate segment name to fit Excel limit (31 chars total)
                # Format: DEL30_TOPUP_Mixed = 16 chars max for seg_name
                seg_name = _sanitize_sheet_name(seg_key, max_len=12)
                
                # Mixed
                if seg_key in mixed_by_seg:
                    sheet_name = _sanitize_sheet_name(f"{del_type}_{seg_name}_Mix")
                    mixed_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Actual
                if seg_key in actual_by_seg:
                    sheet_name = _sanitize_sheet_name(f"{del_type}_{seg_name}_Act")
                    actual_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Forecast
                if seg_key in forecast_by_seg:
                    sheet_name = _sanitize_sheet_name(f"{del_type}_{seg_name}_Fct")
                    forecast_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
                
                # Flags
                if seg_key in flags_by_seg:
                    sheet_name = _sanitize_sheet_name(f"{del_type}_{seg_name}_Flg")
                    flags_by_seg[seg_key].to_excel(writer, sheet_name=sheet_name, index=False)
        
        # === METADATA SHEETS ===
        if transitions_long_df is not None:
            transitions_long_df.to_excel(writer, sheet_name="transitions_long", index=False)
        
        if meta_df is not None:
            meta_df.to_excel(writer, sheet_name="segment_meta", index=False)
        
        if factors_df is not None:
            factors_df.to_excel(writer, sheet_name="calibration_factors", index=False)
        
        if forecast_df is not None:
            forecast_df.to_excel(writer, sheet_name="forecast_long", index=False)
