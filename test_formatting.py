#!/usr/bin/env python3
"""
Test script để demo các tính năng formatting Excel mới.
Demo script for new Excel formatting features.
"""
import pandas as pd
import numpy as np
from export import export_all_del_to_excel

# Tạo dữ liệu test
# Create test data
def create_test_data():
    """Tạo dữ liệu test cho DEL30/60/90"""
    np.random.seed(42)
    
    # Cohorts và segments
    cohorts = ['2023-01', '2023-02', '2023-03']
    segments = ['CDLPIL', 'TOPUP', 'PERSONAL']
    
    # MOB columns
    mob_cols = [f'MOB_{i}' for i in range(25)]
    
    data = []
    flags_data = []
    
    for cohort in cohorts:
        for segment in segments:
            # Tạo DEL curve tăng dần
            base_del = np.random.uniform(0.01, 0.03)
            del_values = [base_del * (1 + i * 0.1) for i in range(25)]
            
            # Add some noise
            del_values = [max(0, val + np.random.normal(0, 0.005)) for val in del_values]
            
            # Mixed data
            row = {'cohort': cohort, 'segment_key': segment}
            row.update({f'MOB_{i}': del_values[i] for i in range(25)})
            data.append(row)
            
            # Flags data (ACTUAL for first 12 MOBs, FORECAST for rest)
            flag_row = {'cohort': cohort, 'segment_key': segment}
            flag_row.update({f'MOB_{i}': 'ACTUAL' if i < 12 else 'FORECAST' for i in range(25)})
            flags_data.append(flag_row)
    
    mixed_df = pd.DataFrame(data)
    flags_df = pd.DataFrame(flags_data)
    
    # Actual = Mixed với NaN cho FORECAST periods
    actual_df = mixed_df.copy()
    for i, row in flags_df.iterrows():
        for col in mob_cols:
            if row[col] == 'FORECAST':
                actual_df.loc[i, col] = np.nan
    
    # Forecast = Mixed với slight variations
    forecast_df = mixed_df.copy()
    for col in mob_cols:
        forecast_df[col] = forecast_df[col] * np.random.uniform(0.95, 1.05, len(forecast_df))
    
    return mixed_df, flags_df, actual_df, forecast_df

def main():
    """Chạy test formatting"""
    print("Creating test data...")
    mixed_df, flags_df, actual_df, forecast_df = create_test_data()
    
    # Tạo del_results dict - chỉ copy numeric columns
    mob_cols = [col for col in mixed_df.columns if col.startswith('MOB_')]
    
    # Create copies for DEL60 and DEL90 with different values
    mixed_60 = mixed_df.copy()
    mixed_90 = mixed_df.copy()
    actual_60 = actual_df.copy()
    actual_90 = actual_df.copy()
    forecast_60 = forecast_df.copy()
    forecast_90 = forecast_df.copy()
    
    # Multiply only numeric columns
    for col in mob_cols:
        mixed_60[col] = mixed_60[col] * 0.8
        mixed_90[col] = mixed_90[col] * 0.6
        actual_60[col] = actual_60[col] * 0.8
        actual_90[col] = actual_90[col] * 0.6
        forecast_60[col] = forecast_60[col] * 0.8
        forecast_90[col] = forecast_90[col] * 0.6
    
    del_results = {
        'del30': (mixed_df, flags_df, actual_df, forecast_df),
        'del60': (mixed_60, flags_df, actual_60, forecast_60),
        'del90': (mixed_90, flags_df, actual_90, forecast_90)
    }
    
    # Tạo transitions dummy
    transitions_df = pd.DataFrame({
        'level': ['FULL'] * 10,
        'segment_key': ['CDLPIL'] * 10,
        'mob': list(range(10)),
        'from_state': ['DPD0'] * 10,
        'to_state': ['DPD1+'] * 10,
        'probability': np.random.uniform(0.1, 0.3, 10)
    })
    
    print("Exporting formatted Excel file...")
    export_all_del_to_excel(
        'test_formatted_output.xlsx',
        transitions_df,
        del_results
    )
    
    print("✅ Test completed! Check 'test_formatted_output.xlsx' for formatting:")
    print("   - Title tại dòng 1 với font size 20, bold, màu Dark Blue")
    print("   - Headers từ dòng 3, tô đậm, tô cell, căn giữa")
    print("   - Values format 4 decimal places với %")
    print("   - Color scale Red-Yellow-Green cho MOB columns")
    print("   - Border phân biệt actual/forecast bằng màu đỏ nét đậm")
    print("   - Bỏ grid lines")

if __name__ == "__main__":
    main()