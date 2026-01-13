# Implementation Plan: DEL Forecast & Calibration System

## Overview

Implementation plan cho hệ thống DEL Forecast & Calibration. Các tasks được sắp xếp theo thứ tự dependencies và bao gồm cả property-based tests.

## Tasks

- [x] 1. Configuration Module (config.py)
  - [x] 1.1 Define CFG dictionary with all column mappings
    - loan_id, mob, bucket, ead, orig_date, cutoff, cohort, segment_key
    - _Requirements: 5.1, 5.2_
  - [x] 1.2 Define bucket lists for DEL metrics
    - BUCKETS_30P, BUCKETS_60P, BUCKETS_90P
    - _Requirements: 5.3_
  - [x] 1.3 Define absorbing states and canonical bucket order
    - ABSORBING_BASE, BUCKETS_CANON
    - _Requirements: 5.4_
  - [x] 1.4 Define calibration parameters
    - K_CLIP, CALIBRATION_MODE, MAX_MOB
    - _Requirements: 5.5_

- [x] 2. Metrics Module - Core DEL Computation (metrics.py)
  - [x] 2.1 Implement compute_del_from_snapshot()
    - Compute actual DEL from snapshot data
    - Use cfg["column_key"] for all column access
    - _Requirements: 3.1, 3.2, 3.3, 3.4_
  - [x] 2.2 Implement compute_del_from_forecast()
    - Compute DEL from forecast DataFrame
    - _Requirements: 3.4_
  - [x] 2.3 Implement forecast_from_actual()
    - Forecast DEL từ actual EAD distribution tại last actual MOB
    - _Requirements: 1.1, 1.2, 1.3_
  - [x] 2.4 Implement make_mixed_report_v2()
    - Create mixed report with correct forecast logic
    - _Requirements: 1.4, 1.5_
  - [x] 2.5 Implement compute_all_del_metrics_v2()
    - Compute DEL30, DEL60, DEL90 with correct forecast logic
    - _Requirements: 3.1, 3.2, 3.3, 3.5_

- [x] 2.6 Write property tests for metrics module
  - **Property 1: Forecast Starts from Last Actual EAD**
  - **Property 2: Mixed Report Correctness**
  - **Property 5: DEL Denominator Invariant**
  - **Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 3.4**

- [x] 3. Calibration Module (calibration.py)
  - [x] 3.1 Implement fit_stepwise_calibration_factors()
    - Compute k[m] = actual_del[m] / expected_del[m]
    - expected_del[m] = actual_ead[m-1] @ P[m-1]
    - _Requirements: 2.1, 2.2_
  - [x] 3.2 Implement apply_stepwise_calibration_to_matrices()
    - Apply k factors to transition matrices
    - Maintain row sum = 1
    - _Requirements: 2.3_
  - [x] 3.3 Implement k clipping logic
    - Clip k to (k_min, k_max) bounds
    - _Requirements: 2.5_

- [x] 3.4 Write property tests for calibration module
  - **Property 3: Step-wise Calibration Formula**
  - **Property 4: Calibrated Matrix Row Sum**
  - **Validates: Requirements 2.1, 2.2, 2.3, 2.5**

- [x] 4. Export Module (export.py)
  - [x] 4.1 Implement _format_mixed_sheet()
    - Title at row 1: {SEGMENT}_{DEL} Actual & Forecast
    - Headers from row 3 with formatting
    - _Requirements: 4.2, 4.3_
  - [x] 4.2 Implement percentage formatting
    - Format MOB values as 0.00% (2 decimal places)
    - _Requirements: 4.4_
  - [x] 4.3 Implement color scale
    - Green-Yellow-Red color scale (green=low, red=high)
    - _Requirements: 4.5_
  - [x] 4.4 Implement ACTUAL/FORECAST boundary borders
    - Thick red borders on RIGHT and BOTTOM of last ACTUAL cell
    - _Requirements: 4.6_
  - [x] 4.5 Implement grid removal
    - Remove grid lines from all sheets
    - _Requirements: 4.7_
  - [x] 4.6 Implement _compute_portfolio_del()
    - Aggregate DEL across segments by mean
    - Aggregate flags with ACTUAL/MIXED/FORECAST logic
    - _Requirements: 6.1, 6.2_
  - [x] 4.7 Implement export_all_del_to_excel()
    - Create sheets for Portfolio and each segment
    - Portfolio sheets before segment sheets
    - _Requirements: 4.1, 6.3_

- [x] 4.8 Write property tests for export module
  - **Property 6: Excel Percentage Format**
  - **Property 7: ACTUAL/FORECAST Boundary Border**
  - **Property 8: Portfolio DEL Aggregation**
  - **Property 9: Portfolio Flag Aggregation**
  - **Validates: Requirements 4.4, 4.6, 6.1, 6.2**

- [x] 5. Checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

- [x] 6. Integration - Update Notebooks
  - [x] 6.1 Update notebooks/03_calibration_demo.ipynb
    - Use fit_stepwise_calibration_factors()
    - Use apply_stepwise_calibration_to_matrices()
    - Use make_mixed_report_v2()
    - Use export_all_del_to_excel()
    - _Requirements: All_

- [x] 7. Final checkpoint - Ensure all tests pass
  - Ensure all tests pass, ask the user if questions arise.

## Notes

- All tasks are required including property tests
- Each task references specific requirements for traceability
- Checkpoints ensure incremental validation
- Property tests validate universal correctness properties
- Unit tests validate specific examples and edge cases
- All column access must use cfg["column_key"] pattern, not hardcoded strings
