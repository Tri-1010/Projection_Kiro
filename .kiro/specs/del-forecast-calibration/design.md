# Design Document: DEL Forecast & Calibration System

## Overview

Hệ thống DEL Forecast & Calibration sử dụng mô hình Markov Chain để dự báo tỷ lệ nợ xấu (DEL30, DEL60, DEL90). Hệ thống được thiết kế với các module riêng biệt cho: metrics computation, calibration, forecasting, và Excel export.

This DEL Forecast & Calibration system uses Markov Chain models to forecast delinquency rates (DEL30, DEL60, DEL90). The system is designed with separate modules for: metrics computation, calibration, forecasting, and Excel export.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Main Pipeline                             │
├─────────────────────────────────────────────────────────────────┤
│  1. Load Data (df_snapshot)                                      │
│  2. Build Transition Matrices (transitions.py)                   │
│  3. Fit Step-wise Calibration (calibration.py)                   │
│  4. Apply Calibration to Matrices                                │
│  5. Compute DEL Metrics with Forecast (metrics.py)               │
│  6. Export to Excel (export.py)                                  │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  config.py   │───▶│  metrics.py  │───▶│  export.py   │
│  (CFG dict)  │    │  (DEL calc)  │    │  (Excel)     │
└──────────────┘    └──────────────┘    └──────────────┘
       │                   ▲
       │                   │
       ▼                   │
┌──────────────┐    ┌──────────────┐
│ calibration  │───▶│  forecast.py │
│    .py       │    │  (Markov)    │
└──────────────┘    └──────────────┘
```

## Components and Interfaces

### 1. Configuration Module (config.py)

```python
CFG = {
    "loan_id": "AGREEMENT_ID",
    "mob": "MOB",
    "bucket": "STATE_MODEL",
    "ead": "PRINCIPLE_OUTSTANDING",
    "orig_date": "DISBURSAL_DATE",
    "cutoff": "CUTOFF_DATE",
    "cohort": "cohort",
    "segment_key": "segment_key",
}

BUCKETS_30P = ["DPD30+", "DPD60+", "DPD90+", "WRITEOFF"]
BUCKETS_60P = ["DPD60+", "DPD90+", "WRITEOFF"]
BUCKETS_90P = ["DPD90+", "WRITEOFF"]
```

### 2. Metrics Module (metrics.py)

#### Key Functions:

```python
def forecast_from_actual(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    transitions_dict: Dict,
    states: List[str],
    segment_cols: List[str],
    max_mob: int,
    bad_states: List[str],
    denom_map: Dict
) -> pd.DataFrame:
    """
    Forecast DEL từ actual EAD distribution tại mỗi MOB có actual data.
    
    Logic:
    1. Với mỗi cohort-segment, tìm MOB cao nhất có actual data (max_actual_mob)
    2. Lấy actual EAD distribution tại max_actual_mob
    3. Forecast từ max_actual_mob + 1 đến max_mob bằng: v[m+1] = v[m] @ P[m]
    4. Tính DEL = sum(bad_ead) / denom_ead
    """

def make_mixed_report_v2(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    transitions_dict: Dict,
    states: List[str],
    segment_cols: List[str],
    max_mob: int,
    bad_states: List[str],
    denom_level: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create mixed report với forecast được tính từ actual EAD distribution gần nhất.
    
    Returns:
    - mixed_wide: Actual + Forecast DEL values
    - flags_wide: "ACTUAL" or "FORECAST" flags
    - actual_wide: Only actual DEL values
    - forecast_wide: Only forecast DEL values (from MOB=0)
    """

def compute_all_del_metrics_v2(
    df_snapshot, cfg, transitions_dict, states, segment_cols,
    max_mob, denom_level, buckets_30p, buckets_60p, buckets_90p
) -> Dict[str, Tuple]:
    """
    Compute DEL30, DEL60, DEL90 với forecast từ actual EAD distribution.
    
    Returns:
    - Dict with keys "DEL30", "DEL60", "DEL90"
    - Each value is (mixed_wide, flags_wide, actual_wide, forecast_wide)
    """
```

### 3. Calibration Module (calibration.py)

#### Key Functions:

```python
def fit_stepwise_calibration_factors(
    df_snapshot: pd.DataFrame,
    cfg: Dict[str, Any],
    transitions_dict: Dict,
    states: List[str],
    segment_cols: List[str],
    bad_states: List[str],
    max_mob: int,
    denom_level: str,
    k_clip: Tuple[float, float]
) -> pd.DataFrame:
    """
    Tính hệ số calibration theo từng bước chuyển (step-wise).
    
    Formula:
    - expected_del[m] = actual_ead[m-1] @ P[m-1] → compute DEL
    - k[m] = actual_del[m] / expected_del[m]
    - k[m] = clip(k[m], k_min, k_max)
    
    Returns:
    - DataFrame with columns: mob, k, n_cohorts_used, expected_mean, actual_mean
    """

def apply_stepwise_calibration_to_matrices(
    transitions_dict: Dict,
    factors_df: pd.DataFrame,
    bad_states: List[str],
    absorbing_states: List[str]
) -> Dict:
    """
    Áp dụng hệ số calibration step-wise vào transition matrices.
    
    Logic:
    - Với mỗi matrix P[m], lấy k[m+1] (k cho transition từ m sang m+1)
    - Scale bad state probabilities by k
    - Adjust good state probabilities to maintain row sum = 1
    """
```

### 4. Export Module (export.py)

#### Key Functions:

```python
def export_all_del_to_excel(
    path: str,
    transitions_long_df: pd.DataFrame,
    del_results: dict,
    factors_df: Optional[pd.DataFrame] = None,
    forecast_df: Optional[pd.DataFrame] = None,
    meta_df: Optional[pd.DataFrame] = None
) -> None:
    """
    Export DEL30, DEL60, DEL90 results to Excel.
    
    Sheet structure:
    - DEL30_Portfolio, DEL30_{Product}_Mixed, DEL30_{Product}_Actual, etc.
    - DEL60_Portfolio, DEL60_{Product}_Mixed, etc.
    - DEL90_Portfolio, DEL90_{Product}_Mixed, etc.
    - Metadata sheets: transitions_long, segment_meta, calibration_factors
    """

def _format_mixed_sheet(
    worksheet, segment_key: str, del_type: str, 
    data_df: pd.DataFrame, flags_df: pd.DataFrame
):
    """
    Format Mixed sheet với các yêu cầu:
    - Title tại dòng 1: {segment_key}_{del_type} Actual & Forecast
    - Header từ dòng 3, tô đậm, tô cell, căn giữa
    - Values format 2 decimal places với % (0.00%)
    - Color scale: xanh (thấp) → vàng (giữa) → đỏ (cao)
    - Border đỏ dày ở cạnh phải và dưới của cell ACTUAL cuối cùng
    - Bỏ grid
    """
```

## Data Models

### Transition Matrix Structure
```python
# Key: (level, segment_key, mob)
# Value: pd.DataFrame with states as index and columns
transitions_dict = {
    ("FULL", "TOPUP", 0): pd.DataFrame(...),  # P[0] for TOPUP segment
    ("FULL", "TOPUP", 1): pd.DataFrame(...),  # P[1] for TOPUP segment
    ...
}
```

### DEL Results Structure
```python
del_results = {
    "DEL30": (mixed_wide, flags_wide, actual_wide, forecast_wide),
    "DEL60": (mixed_wide, flags_wide, actual_wide, forecast_wide),
    "DEL90": (mixed_wide, flags_wide, actual_wide, forecast_wide),
}

# Each DataFrame has columns:
# cohort, segment_key, MOB_0, MOB_1, ..., MOB_24
```

### Calibration Factors Structure
```python
factors_df = pd.DataFrame({
    "mob": [0, 1, 2, ...],
    "k": [1.0, 1.05, 0.98, ...],
    "n_cohorts_used": [0, 50, 48, ...],
    "expected_mean": [0.0, 0.02, 0.03, ...],
    "actual_mean": [0.0, 0.021, 0.029, ...],
})
```

## Correctness Properties

*A property is a characteristic or behavior that should hold true across all valid executions of a system-essentially, a formal statement about what the system should do. Properties serve as the bridge between human-readable specifications and machine-verifiable correctness guarantees.*

### Property 1: Forecast Starts from Last Actual EAD

*For any* cohort-segment with actual data up to MOB m, the forecast at MOB m+1 SHALL be computed using the actual EAD distribution at MOB m as the starting vector, not the EAD at MOB=0.

**Validates: Requirements 1.1, 1.2, 1.3**

### Property 2: Mixed Report Correctness

*For any* cohort-segment and MOB, if actual data exists at that MOB, the mixed report SHALL contain the actual DEL value; otherwise, it SHALL contain the forecasted DEL value computed from the last actual EAD distribution.

**Validates: Requirements 1.4, 1.5**

### Property 3: Step-wise Calibration Formula

*For any* MOB m > 0, the calibration factor k[m] SHALL equal actual_del[m] / expected_del[m], where expected_del[m] is computed from actual_ead[m-1] @ P[m-1], clipped to (k_min, k_max).

**Validates: Requirements 2.1, 2.2, 2.5**

### Property 4: Calibrated Matrix Row Sum

*For any* calibrated transition matrix, each row SHALL sum to 1.0 (within floating-point tolerance).

**Validates: Requirements 2.3**

### Property 5: DEL Denominator Invariant

*For any* DEL computation, the denominator SHALL be the sum of EAD at MOB=0 for the cohort (or cohort-segment), regardless of which MOB the DEL is being computed for.

**Validates: Requirements 3.4**

### Property 6: Excel Percentage Format

*For any* MOB cell in the Excel output, the number format SHALL be '0.00%' (2 decimal places).

**Validates: Requirements 4.4**

### Property 7: ACTUAL/FORECAST Boundary Border

*For any* row in the Mixed sheet where there is a transition from ACTUAL to FORECAST, the last ACTUAL cell SHALL have thick red borders on its RIGHT and BOTTOM edges.

**Validates: Requirements 4.6**

### Property 8: Portfolio DEL Aggregation

*For any* MOB, the Portfolio DEL SHALL equal the mean of all segment DELs at that MOB.

**Validates: Requirements 6.1**

### Property 9: Portfolio Flag Aggregation

*For any* MOB, the Portfolio flag SHALL be "ACTUAL" if all segments have "ACTUAL", "MIXED" if some have "ACTUAL", and "FORECAST" if none have "ACTUAL".

**Validates: Requirements 6.2**

## Error Handling

1. **Missing Data**: If a cohort-segment has no data at MOB=0, skip it (denom_ead = 0)
2. **Division by Zero**: Use safe division with threshold 1e-10
3. **Missing Transition Matrix**: Use `get_best_matrix()` to fall back to coarser segment or global matrix
4. **Calibration Factor Bounds**: Clip k to (k_min, k_max) to prevent extreme adjustments
5. **Excel Sheet Name Limits**: Truncate and sanitize sheet names to 31 characters

## Testing Strategy

### Unit Tests
- Test individual functions with specific examples
- Test edge cases: empty data, single cohort, missing MOBs
- Test error handling: division by zero, missing matrices

### Property-Based Tests
- Use `hypothesis` library for Python
- Minimum 100 iterations per property test
- Generate random cohort-segment data with varying MOBs
- Generate random transition matrices (valid stochastic matrices)
- Verify properties hold across all generated inputs

### Integration Tests
- Test full pipeline from data load to Excel export
- Verify Excel file structure and formatting
- Compare output against known baseline results
