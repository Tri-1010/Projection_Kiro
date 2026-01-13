# Requirements Document

## Introduction

Hệ thống dự báo DEL (Delinquency) sử dụng mô hình Markov Chain để tính toán và dự báo tỷ lệ nợ xấu (DEL30, DEL60, DEL90) theo cohort và segment. Hệ thống bao gồm các chức năng: tính transition matrices, calibration step-wise, forecast từ actual EAD distribution, và xuất báo cáo Excel với định dạng chuyên nghiệp.

This system forecasts DEL (Delinquency) rates using Markov Chain models to calculate and predict bad debt ratios (DEL30, DEL60, DEL90) by cohort and segment. The system includes: transition matrix computation, step-wise calibration, forecasting from actual EAD distribution, and professional Excel report export.

## Glossary

- **DEL (Delinquency)**: Tỷ lệ nợ xấu, tính bằng EAD của các bucket xấu chia cho tổng EAD tại MOB=0
- **DEL30/60/90**: DEL với ngưỡng 30+/60+/90+ ngày quá hạn
- **MOB (Months on Book)**: Số tháng kể từ ngày giải ngân
- **EAD (Exposure at Default)**: Dư nợ tại thời điểm
- **Cohort**: Nhóm khoản vay theo tháng giải ngân
- **Segment**: Phân khúc sản phẩm (PRODUCT_TYPE)
- **Transition_Matrix**: Ma trận xác suất chuyển trạng thái giữa các bucket
- **Calibration_Factor**: Hệ số điều chỉnh k để forecast khớp với actual
- **Mixed_Report**: Báo cáo kết hợp actual (nếu có) và forecast (nếu không có actual)
- **Step_wise_Calibration**: Calibration theo từng bước chuyển MOB, không phải tích lũy từ MOB=0

## Requirements

### Requirement 1: Forecast Logic - Forecast từ Actual EAD Distribution

**User Story:** As a risk analyst, I want the forecast to continue from the last actual EAD distribution, so that the forecast reflects the real portfolio state instead of starting from MOB=0.

#### Acceptance Criteria

1. WHEN creating a mixed report, THE Forecast_Engine SHALL identify the last MOB with actual data for each cohort-segment
2. WHEN forecasting beyond actual data, THE Forecast_Engine SHALL use the actual EAD distribution at the last actual MOB as the starting point
3. WHEN applying transition matrices, THE Forecast_Engine SHALL multiply the EAD vector by the transition matrix for each subsequent MOB
4. THE Mixed_Report SHALL contain actual DEL values for MOBs with actual data and forecasted DEL values for MOBs without actual data
5. THE Flags_Report SHALL indicate "ACTUAL" or "FORECAST" for each cell in the Mixed_Report

### Requirement 2: Step-wise Calibration

**User Story:** As a model developer, I want to calibrate transition matrices per MOB step, so that I can measure and correct the error at each transition rather than cumulative error from MOB=0.

#### Acceptance Criteria

1. WHEN fitting calibration factors, THE Calibration_Engine SHALL compute k[m] = actual_del[m] / expected_del[m] for each MOB m
2. WHEN computing expected_del[m], THE Calibration_Engine SHALL use actual_ead[m-1] multiplied by transition matrix P[m-1]
3. WHEN applying calibration, THE Calibration_Engine SHALL scale the bad state probabilities in each transition matrix by the corresponding k factor
4. THE Calibration_Factors DataFrame SHALL contain columns: mob, k, n_cohorts_used, expected_mean, actual_mean
5. WHEN k is outside the clip range, THE Calibration_Engine SHALL clip k to (k_min, k_max) bounds

### Requirement 3: DEL Metrics Computation

**User Story:** As a risk analyst, I want to compute DEL30, DEL60, and DEL90 metrics, so that I can analyze delinquency at different severity levels.

#### Acceptance Criteria

1. THE Metrics_Engine SHALL compute DEL30 using bad states: DPD30+, DPD60+, DPD90+, WRITEOFF
2. THE Metrics_Engine SHALL compute DEL60 using bad states: DPD60+, DPD90+, WRITEOFF
3. THE Metrics_Engine SHALL compute DEL90 using bad states: DPD90+, WRITEOFF
4. WHEN computing DEL, THE Metrics_Engine SHALL use EAD at MOB=0 as the denominator
5. THE Metrics_Engine SHALL output mixed_wide, flags_wide, actual_wide, forecast_wide DataFrames for each DEL type

### Requirement 4: Excel Export with Professional Formatting

**User Story:** As a business user, I want Excel reports with professional formatting, so that I can easily read and present the DEL forecasts.

#### Acceptance Criteria

1. WHEN exporting to Excel, THE Export_Engine SHALL create separate sheets for Portfolio and each Product segment
2. THE Mixed_Sheet SHALL have a title at row 1 in format "{SEGMENT}_{DEL} Actual & Forecast" with size 20, bold, Dark Blue color
3. THE Mixed_Sheet SHALL have headers from row 3 with bold font, light blue background, and center alignment
4. WHEN formatting MOB values, THE Export_Engine SHALL display them as percentages with 2 decimal places (0.00%)
5. THE Export_Engine SHALL apply Green-Yellow-Red color scale to MOB columns where green represents low values and red represents high values
6. WHEN there is a boundary between ACTUAL and FORECAST cells, THE Export_Engine SHALL place thick red borders on the RIGHT and BOTTOM edges of the last ACTUAL cell
7. THE Export_Engine SHALL remove grid lines from all sheets

### Requirement 5: Configuration Management

**User Story:** As a developer, I want all column names centralized in config, so that I can easily adapt the system to different data schemas.

#### Acceptance Criteria

1. THE Config SHALL define all column name mappings in the CFG dictionary
2. WHEN accessing data columns, THE System SHALL use cfg["column_key"] instead of hardcoded strings
3. THE Config SHALL define bucket lists for DEL30, DEL60, DEL90 (BUCKETS_30P, BUCKETS_60P, BUCKETS_90P)
4. THE Config SHALL define absorbing states and canonical bucket order
5. THE Config SHALL define calibration parameters (K_CLIP, CALIBRATION_MODE)

### Requirement 6: Portfolio Aggregation

**User Story:** As a risk manager, I want to see portfolio-level DEL metrics, so that I can understand the overall portfolio performance.

#### Acceptance Criteria

1. WHEN computing Portfolio DEL, THE Export_Engine SHALL aggregate DEL values across all segments by taking the mean
2. WHEN computing Portfolio flags, THE Export_Engine SHALL mark as "ACTUAL" only if all segments have ACTUAL, "MIXED" if some have ACTUAL, else "FORECAST"
3. THE Portfolio sheets SHALL be placed before individual segment sheets in the Excel workbook
