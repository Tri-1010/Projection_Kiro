# DICTIONARY - Từ điển Kỹ thuật / Technical Dictionary

## Mục lục / Table of Contents

1. [Tổng quan Mô hình / Model Overview](#1-tổng-quan-mô-hình--model-overview)
2. [Nguyên lý Markov Chain / Markov Chain Principles](#2-nguyên-lý-markov-chain--markov-chain-principles)
3. [Module: config.py](#3-module-configpy)
4. [Module: data_io.py](#4-module-data_iopy)
5. [Module: transitions.py](#5-module-transitionspy)
6. [Module: forecast.py](#6-module-forecastpy)
7. [Module: calibration.py](#7-module-calibrationpy)
8. [Module: metrics.py](#8-module-metricspy)
9. [Module: export.py](#9-module-exportpy)
10. [Module: backtest.py](#10-module-backtestpy)
11. [Thuật ngữ / Glossary](#11-thuật-ngữ--glossary)

---

## 1. Tổng quan Mô hình / Model Overview

### Mục đích / Purpose
Mô hình dự báo rủi ro tín dụng sử dụng chuỗi Markov không đồng nhất theo thời gian (time-inhomogeneous Markov chain) để dự báo phân bố trạng thái nợ xấu (delinquency) của danh mục cho vay theo MOB (Months on Book).

The credit risk projection model uses time-inhomogeneous Markov chains to forecast the delinquency state distribution of a loan portfolio by MOB (Months on Book).

### Luồng xử lý chính / Main Pipeline Flow

```
┌─────────────────┐
│   Raw Data      │  Parquet file với loan snapshots
│   (Parquet)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   data_io.py    │  Load & validate schema
│   load_parquet  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ transitions.py  │  Chuẩn bị dữ liệu chuyển đổi trạng thái
│prepare_transitions│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ transitions.py  │  Ước lượng ma trận chuyển đổi
│estimate_matrices│  với hierarchical shrinkage + tail pooling
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  forecast.py    │  Xây dựng vector khởi tạo từ MOB=0
│build_init_vectors│
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  forecast.py    │  Dự báo phân bố EAD theo MOB
│    forecast     │  v(t+1) = v(t) × P(t)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  metrics.py     │  Tính DEL30 từ actual & forecast
│ compute_del_*   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ calibration.py  │  Hiệu chỉnh để khớp với actual
│ fit_factors     │  (optional)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  export.py      │  Xuất báo cáo Excel
│export_to_excel  │
└─────────────────┘
```


---

## 2. Nguyên lý Markov Chain / Markov Chain Principles

### 2.1 Markov Chain là gì? / What is a Markov Chain?

**VI:** Chuỗi Markov là mô hình xác suất mô tả chuỗi các sự kiện trong đó xác suất của mỗi sự kiện chỉ phụ thuộc vào trạng thái đạt được ở sự kiện trước đó (tính chất Markov - "memoryless").

**EN:** A Markov chain is a stochastic model describing a sequence of events where the probability of each event depends only on the state attained in the previous event (Markov property - "memoryless").

### 2.2 Ma trận chuyển đổi / Transition Matrix

**Định nghĩa / Definition:**
```
P[i,j] = P(X(t+1) = j | X(t) = i)
```

Trong đó:
- `P[i,j]` = Xác suất chuyển từ trạng thái i sang trạng thái j
- Mỗi hàng tổng = 1 (row-stochastic)

**Ví dụ / Example:**
```
States: [DPD0, DPD1+, DPD30+, DPD60+, DPD90+, WRITEOFF, PREPAY]

         DPD0   DPD1+  DPD30+ DPD60+ DPD90+ WRITEOFF PREPAY
DPD0     0.85   0.10   0.02   0.01   0.00   0.00     0.02
DPD1+    0.30   0.40   0.20   0.05   0.03   0.00     0.02
DPD30+   0.10   0.15   0.35   0.25   0.10   0.03     0.02
DPD60+   0.05   0.05   0.10   0.30   0.35   0.10     0.05
DPD90+   0.00   0.00   0.00   0.00   1.00   0.00     0.00  ← Absorbing
WRITEOFF 0.00   0.00   0.00   0.00   0.00   1.00     0.00  ← Absorbing
PREPAY   0.00   0.00   0.00   0.00   0.00   0.00     1.00  ← Absorbing
```

### 2.3 Time-Inhomogeneous Markov Chain

**VI:** Mô hình sử dụng ma trận chuyển đổi khác nhau cho mỗi MOB (0→1, 1→2, ..., 23→24) vì hành vi trả nợ thay đổi theo tuổi khoản vay.

**EN:** The model uses different transition matrices for each MOB (0→1, 1→2, ..., 23→24) because repayment behavior changes with loan age.

```
v(1) = v(0) × P(0)      # MOB 0 → 1
v(2) = v(1) × P(1)      # MOB 1 → 2
...
v(24) = v(23) × P(23)   # MOB 23 → 24
```

### 2.4 Absorbing States / Trạng thái Hấp thụ

**VI:** Trạng thái hấp thụ là trạng thái mà một khi đã vào thì không thể thoát ra. Trong mô hình này: DPD90+, WRITEOFF, PREPAY.

**EN:** Absorbing states are states that once entered, cannot be left. In this model: DPD90+, WRITEOFF, PREPAY.

```python
# Absorbing row: P[i,i] = 1, P[i,j] = 0 for j ≠ i
# Ví dụ: DPD90+ row = [0, 0, 0, 0, 1, 0, 0]
```

### 2.5 Hierarchical Shrinkage / Co rút Phân cấp

**VI:** Kỹ thuật Bayesian để xử lý segment nhỏ có ít dữ liệu bằng cách "mượn sức mạnh" từ segment lớn hơn.

**EN:** Bayesian technique to handle small segments with limited data by "borrowing strength" from larger segments.

```
Hierarchy: GLOBAL → COARSE → FULL

posterior_counts = observed_counts + τ × prior_probabilities

Trong đó:
- τ (tau) = prior strength (e.g., 50 for FULL, 100 for COARSE)
- prior_probabilities = ma trận từ level cao hơn
```

**Ví dụ / Example:**
```python
# FULL segment "TOPUP" có ít data
# Mượn từ COARSE (cũng là "TOPUP" vì SEGMENT_COLS chỉ có 1 cột)
# Mượn từ GLOBAL (tất cả products)

P_full = normalize(counts_full + 50 * P_coarse)
P_coarse = normalize(counts_coarse + 100 * P_global)
```


---

## 3. Module: config.py

### Mục đích / Purpose
Tập trung tất cả cấu hình và tên cột để dễ bảo trì và tránh hardcode.

Centralize all configuration and column names for easy maintenance and avoid hardcoding.

### Các biến chính / Key Variables

| Variable | Type | Mô tả VI | Description EN |
|----------|------|----------|----------------|
| `CFG` | dict | Mapping tên cột dữ liệu | Data column name mappings |
| `SEGMENT_COLS` | list | Cột phân khúc (e.g., PRODUCT_TYPE) | Segment columns |
| `BUCKETS_CANON` | list | Thứ tự chuẩn các trạng thái | Canonical state order |
| `BUCKETS_30P` | list | Trạng thái nợ xấu 30+ ngày | 30+ days delinquent states |
| `ABSORBING_BASE` | list | Trạng thái hấp thụ cơ bản | Base absorbing states |
| `MAX_MOB` | int | MOB tối đa (24) | Maximum MOB |
| `MIN_COUNT` | int | Số lượng tối thiểu để ước lượng | Minimum count for estimation |
| `WEIGHT_MODE` | str | "ead" hoặc "count" | Weighting mode |
| `PRIOR_STRENGTH_*` | float | Độ mạnh prior cho shrinkage | Prior strength for shrinkage |
| `TAIL_POOL_ENABLED` | bool | Bật/tắt tail pooling | Enable/disable tail pooling |
| `TAIL_POOL_START` | int | MOB bắt đầu tail pooling | MOB to start tail pooling |
| `K_CLIP` | tuple | (k_min, k_max) cho calibration | Calibration factor bounds |
| `DENOM_LEVEL` | str | "cohort" hoặc "cohort_segment" | Denominator level for DEL |

### CFG Dictionary Chi tiết / CFG Dictionary Details

```python
CFG = {
    "loan_id": "AGREEMENT_ID",      # ID khoản vay / Loan identifier
    "mob": "MOB",                    # Months on Book
    "bucket": "STATE_MODEL",         # Trạng thái nợ / Delinquency state
    "ead": "PRINCIPLE_OUTSTANDING",  # Exposure at Default
    "orig_date": "DISBURSAL_DATE",   # Ngày giải ngân / Origination date
    "cutoff": "CUTOFF_DATE",         # Ngày cắt dữ liệu / Data cutoff date
    "cohort": "cohort",              # Cohort (derived: YYYY-MM)
    "from_state": "from_state",      # Trạng thái nguồn / Source state
    "to_state": "to_state",          # Trạng thái đích / Target state
    "segment_key": "segment_key",    # Key phân khúc / Segment key
}
```

---

## 4. Module: data_io.py

### 4.1 Function: `load_parquet(path)`

**Mục đích / Purpose:**
Đọc file parquet vào DataFrame.
Load parquet file into DataFrame.

**Input:**
- `path` (str): Đường dẫn file parquet

**Output:**
- `pd.DataFrame`: Dữ liệu đã load

**Code Logic:**
```python
def load_parquet(path: str) -> pd.DataFrame:
    return pd.read_parquet(path)
```

### 4.2 Function: `validate_schema(df, cfg, segment_cols, states)`

**Mục đích / Purpose:**
Kiểm tra DataFrame có đủ cột và giá trị hợp lệ.
Validate DataFrame has required columns and valid values.

**Input:**
- `df`: DataFrame cần validate
- `cfg`: Config dictionary
- `segment_cols`: Danh sách cột segment
- `states`: Danh sách trạng thái hợp lệ

**Output:**
- None (raise ValueError nếu lỗi)

**Validation Steps:**
1. Kiểm tra cột bắt buộc: loan_id, mob, bucket, ead, orig_date
2. Kiểm tra cột segment tồn tại
3. Cảnh báo bucket values không trong states list
4. Kiểm tra MOB >= 0
5. Kiểm tra EAD >= 0


---

## 5. Module: transitions.py

### 5.1 Function: `prepare_transitions(df, cfg, segment_cols, states, absorbing_states)`

**Mục đích / Purpose:**
Chuyển đổi dữ liệu snapshot (loan_id × MOB) thành dữ liệu transition (from_state → to_state).

Transform snapshot data (loan_id × MOB) into transition data (from_state → to_state).

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `df` | DataFrame | Dữ liệu snapshot gốc |
| `cfg` | dict | Config với tên cột |
| `segment_cols` | list | Cột phân khúc |
| `states` | list | Trạng thái hợp lệ |
| `absorbing_states` | list | Trạng thái hấp thụ |

**Output:**
- DataFrame với columns: `[loan_id, mob, from_state, to_state, ead, cohort] + segment_cols`

**Algorithm / Thuật toán:**

```
Step 1: Filter & Clean
├── Loại bỏ rows có bucket = NaN
├── Chỉ giữ rows có bucket trong states list
└── Parse cohort từ orig_date (YYYY-MM format)

Step 2: Handle Duplicates
├── Sort by [loan_id, mob, cutoff] (nếu có cutoff)
└── Drop duplicates, keep="last" (giữ record mới nhất)

Step 3: Compute Transitions
├── from_state = bucket hiện tại
├── to_state = bucket của MOB tiếp theo (shift -1 trong group loan_id)
└── Nếu from_state là absorbing → force to_state = from_state

Step 4: Auto-expand Absorbing States
├── Nếu "DPD90+" trong absorbing_states
└── Tự động thêm DPD120+, DPD180+, etc. nếu có trong states

Step 5: Filter Valid Transitions
├── Chỉ giữ MOB trong [0, MAX_MOB-1]
├── Loại bỏ rows có to_state = NaN
└── Chỉ giữ to_state trong states list
```

**Ví dụ / Example:**

Input (snapshot):
```
loan_id | mob | bucket  | ead
--------|-----|---------|-------
A001    | 0   | DPD0    | 100000
A001    | 1   | DPD0    | 98000
A001    | 2   | DPD1+   | 96000
A001    | 3   | DPD30+  | 94000
```

Output (transitions):
```
loan_id | mob | from_state | to_state | ead    | cohort
--------|-----|------------|----------|--------|--------
A001    | 0   | DPD0       | DPD0     | 100000 | 2023-01
A001    | 1   | DPD0       | DPD1+    | 98000  | 2023-01
A001    | 2   | DPD1+      | DPD30+   | 96000  | 2023-01
```

### 5.2 Function: `estimate_transition_matrices(...)`

**Mục đích / Purpose:**
Ước lượng ma trận chuyển đổi với hierarchical shrinkage và tail pooling (tùy chọn).

Estimate transition matrices with hierarchical shrinkage and optional tail pooling.

**Tail Pooling Control:**
```python
# Trong config.py:
TAIL_POOL_ENABLED = False  # True: bật, False: tắt
TAIL_POOL_START = 18       # MOB bắt đầu pooling

# Trong transitions.py, tail pooling chỉ chạy khi:
if TAIL_POOL_ENABLED and tail_pool_start is not None and tail_pool_start < max_mob:
    # Apply tail pooling...
```

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `df_trans` | DataFrame | Output từ prepare_transitions |
| `cfg` | dict | Config |
| `states` | list | Thứ tự trạng thái chuẩn |
| `segment_levels` | list[tuple] | [(name, cols), ...] |
| `max_mob` | int | MOB tối đa |
| `weight_mode` | str | "ead" hoặc "count" |
| `min_count` | int | Số lượng tối thiểu |
| `prior_strengths` | dict | {"coarse": τ1, "full": τ2} |
| `tail_pool_start` | int | MOB bắt đầu pooling |

**Output:**
- `transitions_dict`: Dict[(level, segment_key, mob)] → DataFrame (ma trận P)
- `transitions_long_df`: Long-form của tất cả probabilities
- `meta_df`: Metadata về sample sizes

**Algorithm / Thuật toán:**

```
For each MOB in [0, MAX_MOB-1]:
    
    ┌─────────────────────────────────────────────────────────┐
    │ LEVEL 1: GLOBAL (tất cả data)                           │
    │ ├── Tính counts matrix từ tất cả transitions            │
    │ ├── counts[i,j] = Σ weight(from=i, to=j)               │
    │ └── P_global = normalize(counts)                        │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ LEVEL 2: COARSE (per segment[0])                        │
    │ ├── Tính counts_seg cho mỗi segment                     │
    │ ├── posterior = counts_seg + τ_coarse × P_global        │
    │ └── P_coarse = normalize(posterior)                     │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ LEVEL 3: FULL (per all segment_cols)                    │
    │ ├── Tính counts_seg cho mỗi full segment                │
    │ ├── posterior = counts_seg + τ_full × P_parent          │
    │ │   (P_parent = P_coarse hoặc P_global)                 │
    │ └── P_full = normalize(posterior)                       │
    └─────────────────────────────────────────────────────────┘

After all MOBs processed:
    
    ┌─────────────────────────────────────────────────────────┐
    │ TAIL POOLING (nếu tail_pool_start != None)              │
    │ ├── Lấy matrices cho MOB >= tail_pool_start             │
    │ ├── P_pooled = mean(P[mob] for mob in tail_mobs)        │
    │ └── Replace tất cả tail matrices bằng P_pooled          │
    └─────────────────────────────────────────────────────────┘
```

**Hierarchical Shrinkage Formula:**

```
posterior_counts[i,j] = observed_counts[i,j] + τ × P_parent[i,j]

P[i,j] = posterior_counts[i,j] / Σ_k posterior_counts[i,k]
```

**Absorbing State Handling:**
```python
if from_state in absorbing_states:
    P[i, :] = 0
    P[i, i] = 1  # One-hot: chỉ ở lại trạng thái hiện tại
```

**Zero-row Handling:**
```python
if row_sum == 0:
    P[i, :] = 0
    P[i, i] = 1  # Identity: giữ nguyên trạng thái
```


---

## 6. Module: forecast.py

### 6.1 Function: `build_initial_vectors(df_snapshot, cfg, states, segment_cols, denom_level)`

**Mục đích / Purpose:**
Xây dựng vector EAD khởi tạo tại MOB=0 cho mỗi cohort-segment.

Build initial EAD vectors at MOB=0 for each cohort-segment.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `df_snapshot` | DataFrame | Dữ liệu snapshot gốc |
| `cfg` | dict | Config |
| `states` | list | Danh sách trạng thái |
| `segment_cols` | list | Cột phân khúc |
| `denom_level` | str | "cohort" hoặc "cohort_segment" |

**Output:**
- `df_init`: DataFrame với [cohort, segment_key, state, ead]
- `denom_map`: Dict[(cohort, segment_key)] → total_ead_at_mob0

**Algorithm:**
```
1. Parse cohort từ orig_date
2. Filter to MOB=0 only
3. Build segment_key = join(segment_cols, "|")
4. Aggregate: sum(ead) group by [cohort, segment_key, bucket]
5. Build denom_map:
   - If denom_level="cohort": denom = total EAD của cohort
   - If denom_level="cohort_segment": denom = total EAD của cohort+segment
```

**Ví dụ Output:**
```
cohort   | segment_key | state  | ead
---------|-------------|--------|--------
2023-01  | TOPUP       | DPD0   | 5000000
2023-01  | TOPUP       | DPD1+  | 200000
2023-01  | SALPIL      | DPD0   | 8000000
```

### 6.2 Function: `get_best_matrix(transitions_dict, mob, segment_key)`

**Mục đích / Purpose:**
Lấy ma trận chuyển đổi tốt nhất với fallback hierarchy.

Get best available transition matrix with hierarchy fallback.

**Fallback Order:**
```
1. FULL level (segment_key đầy đủ)
   ↓ không có
2. COARSE level (segment_key[0] - phần đầu)
   ↓ không có
3. GLOBAL level (tất cả data)
```

**Code Logic:**
```python
def get_best_matrix(transitions_dict, mob, segment_key):
    # Try FULL
    if ("FULL", segment_key, mob) in transitions_dict:
        return transitions_dict[("FULL", segment_key, mob)]
    
    # Try COARSE
    coarse_key = segment_key.split("|")[0]
    if ("COARSE", coarse_key, mob) in transitions_dict:
        return transitions_dict[("COARSE", coarse_key, mob)]
    
    # Fallback to GLOBAL
    return transitions_dict.get(("GLOBAL", "", mob))
```

### 6.3 Function: `forecast(df_init, transitions_dict, states, max_mob, actual_override_df=None)`

**Mục đích / Purpose:**
Dự báo phân bố EAD theo MOB sử dụng ma trận chuyển đổi.

Forecast EAD distribution over MOBs using transition matrices.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `df_init` | DataFrame | Vector khởi tạo từ build_initial_vectors |
| `transitions_dict` | dict | Ma trận từ estimate_transition_matrices |
| `states` | list | Danh sách trạng thái |
| `max_mob` | int | MOB tối đa để dự báo |
| `actual_override_df` | DataFrame | (Optional) Actual values để override |

**Output:**
- DataFrame với [cohort, segment_key, mob, state, ead]

**Algorithm / Thuật toán:**

```
For each (cohort, segment_key) group:
    
    ┌─────────────────────────────────────────────────────────┐
    │ Step 1: Build initial vector v(0)                       │
    │ v = [ead_DPD0, ead_DPD1+, ead_DPD30+, ..., ead_PREPAY]  │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 2: Record MOB=0                                    │
    │ records.append({mob=0, state=s, ead=v[s]} for s)        │
    └─────────────────────────────────────────────────────────┘
                              │
                              ▼
    ┌─────────────────────────────────────────────────────────┐
    │ Step 3: Forecast MOB 1 to max_mob                       │
    │ For mob in [1, max_mob]:                                │
    │   ├── Check actual_override (nếu có, dùng actual)       │
    │   ├── P = get_best_matrix(mob-1, segment_key)           │
    │   ├── v = v @ P  (matrix multiplication)                │
    │   └── records.append({mob, state, ead} for each state)  │
    └─────────────────────────────────────────────────────────┘
```

**Matrix Multiplication:**
```
v(t+1) = v(t) × P(t)

Trong đó:
- v(t) = [ead_state1, ead_state2, ..., ead_stateN] (1×N vector)
- P(t) = N×N transition matrix cho MOB t
- v(t+1) = EAD distribution tại MOB t+1
```

**Ví dụ / Example:**
```python
# v(0) = [100000, 0, 0, 0, 0, 0, 0]  (100% ở DPD0)
# P(0) = [[0.9, 0.08, 0.01, 0.005, 0.003, 0.001, 0.001], ...]

v(1) = v(0) @ P(0)
# v(1) = [90000, 8000, 1000, 500, 300, 100, 100]
```


---

## 7. Module: calibration.py

### 7.1 Function: `fit_del_curve_factors(actual_del_long, pred_del_long, max_mob, k_clip)`

**Mục đích / Purpose:**
Tính hệ số hiệu chỉnh k cho mỗi MOB để forecast khớp với actual.

Compute calibration factors k per MOB to match forecast to actual.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `actual_del_long` | DataFrame | DEL thực tế [cohort, mob, del_pct] |
| `pred_del_long` | DataFrame | DEL dự báo [cohort, mob, del_pct] |
| `max_mob` | int | MOB tối đa |
| `k_clip` | tuple | (k_min, k_max) để clip |

**Output:**
- DataFrame với [mob, k, n_cohorts_used, pred_mean, actual_mean]

**Formula:**
```
k[mob] = clip(actual_mean / pred_mean, k_min, k_max)

Trong đó:
- actual_mean = mean(actual_del_pct) tại MOB đó
- pred_mean = mean(pred_del_pct) tại MOB đó
- k_min, k_max = bounds để tránh extreme values (e.g., 0.5, 2.0)
```

**Ví dụ / Example:**
```
MOB | actual_mean | pred_mean | k (raw) | k (clipped)
----|-------------|-----------|---------|------------
0   | 0.01        | 0.01      | 1.00    | 1.00
6   | 0.05        | 0.04      | 1.25    | 1.25
12  | 0.12        | 0.08      | 1.50    | 1.50
18  | 0.20        | 0.10      | 2.00    | 2.00 (capped)
24  | 0.30        | 0.12      | 2.50    | 2.00 (capped)
```

### 7.2 Function: `apply_matrix_calibration(P, bad_states, k, absorbing_states)`

**Mục đích / Purpose:**
Áp dụng hệ số k vào ma trận chuyển đổi để tăng/giảm xác suất vào bad states.

Apply calibration factor k to transition matrix to increase/decrease probability to bad states.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `P` | DataFrame | Ma trận chuyển đổi (N×N) |
| `bad_states` | list | Trạng thái xấu (e.g., BUCKETS_30P) |
| `k` | float | Hệ số hiệu chỉnh |
| `absorbing_states` | list | Trạng thái hấp thụ (giữ nguyên) |

**Output:**
- DataFrame: Ma trận đã calibrate (row sums = 1)

**Algorithm:**
```
For each row (from_state):
    If from_state is absorbing:
        Keep one-hot (không thay đổi)
    Else:
        1. Scale bad probabilities: P[bad] *= k
        2. Cap total bad probability at 1.0
        3. Adjust good probabilities to maintain row sum = 1
        4. Renormalize if needed
```

**Ví dụ / Example:**
```
Before (k=1.5):
DPD0 → [0.85, 0.10, 0.02, 0.01, 0.01, 0.005, 0.005]
        good   good  bad   bad   bad   bad    good

After:
- bad_total = 0.02 + 0.01 + 0.01 + 0.005 = 0.045
- new_bad_total = 0.045 * 1.5 = 0.0675
- good_total = 0.85 + 0.10 + 0.005 = 0.955
- new_good_total = 1 - 0.0675 = 0.9325
- scale_good = 0.9325 / 0.955 = 0.976

DPD0 → [0.83, 0.098, 0.03, 0.015, 0.015, 0.0075, 0.0049]
```

### 7.3 Function: `apply_vector_calibration(v, bad_states, k)`

**Mục đích / Purpose:**
Áp dụng hệ số k vào vector EAD để scale bad EAD, giữ nguyên total.

Apply calibration factor k to EAD vector to scale bad EAD while preserving total.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `v` | Series | Vector EAD theo state |
| `bad_states` | list | Trạng thái xấu |
| `k` | float | Hệ số hiệu chỉnh |

**Output:**
- Series: Vector EAD đã calibrate (total EAD không đổi)

**Algorithm:**
```
1. bad_ead = sum(v[bad_states])
2. good_ead = sum(v[good_states])
3. total_ead = bad_ead + good_ead

4. new_bad_ead = min(bad_ead * k, total_ead)
5. new_good_ead = total_ead - new_bad_ead

6. Scale bad states: v[bad] *= (new_bad_ead / bad_ead)
7. Scale good states: v[good] *= (new_good_ead / good_ead)
```

**Ví dụ / Example:**
```
Before (k=1.5):
v = [80000, 10000, 5000, 3000, 2000]  # total = 100000
     good   good   bad   bad   bad

bad_ead = 5000 + 3000 + 2000 = 10000
good_ead = 80000 + 10000 = 90000

new_bad_ead = 10000 * 1.5 = 15000
new_good_ead = 100000 - 15000 = 85000

After:
v = [75556, 9444, 7500, 4500, 3000]  # total = 100000 (preserved)
```


---

## 8. Module: metrics.py

### 8.1 Function: `compute_del_from_snapshot(df_snapshot, cfg, bad_states, segment_cols, max_mob, denom_level)`

**Mục đích / Purpose:**
Tính DEL (Delinquency Rate) từ dữ liệu snapshot thực tế.

Compute DEL (Delinquency Rate) from actual snapshot data.

**DEL Formula:**
```
DEL30(mob) = Σ EAD[bad_states at mob] / Σ EAD[all states at MOB=0]

Trong đó:
- Numerator: Tổng EAD của các trạng thái xấu tại MOB hiện tại
- Denominator: Tổng EAD tại MOB=0 (cohort hoặc cohort+segment)
```

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `df_snapshot` | DataFrame | Dữ liệu snapshot |
| `cfg` | dict | Config |
| `bad_states` | list | Trạng thái xấu (e.g., BUCKETS_30P) |
| `segment_cols` | list | Cột phân khúc |
| `max_mob` | int | MOB tối đa |
| `denom_level` | str | "cohort" hoặc "cohort_segment" |

**Output:**
- `actual_del_long`: DataFrame [cohort, segment_key, mob, del_pct, denom_ead, numer_ead]
- `denom_map`: Dict[(cohort, segment_key)] → denom_ead

**Algorithm:**
```
1. Parse cohort từ orig_date
2. Build segment_key

3. Compute denominator (từ MOB=0):
   - If denom_level="cohort": denom = sum(ead) per cohort
   - If denom_level="cohort_segment": denom = sum(ead) per cohort+segment

4. Compute numerator (bad EAD) per cohort, segment, mob:
   numer = sum(ead where bucket in bad_states)

5. Compute DEL:
   del_pct = numer_ead / denom_ead
```

**Ví dụ / Example:**
```
Cohort 2023-01, Segment TOPUP:
- MOB=0: Total EAD = 10,000,000 (denominator)
- MOB=6: Bad EAD = 500,000
- DEL30(6) = 500,000 / 10,000,000 = 5%

Output row:
cohort=2023-01, segment_key=TOPUP, mob=6, del_pct=0.05, denom_ead=10000000, numer_ead=500000
```

### 8.2 Function: `compute_del_from_forecast(forecast_df, bad_states, denom_map)`

**Mục đích / Purpose:**
Tính DEL từ dữ liệu forecast.

Compute DEL from forecast data.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `forecast_df` | DataFrame | Output từ forecast() |
| `bad_states` | list | Trạng thái xấu |
| `denom_map` | dict | Denominator từ build_initial_vectors |

**Output:**
- DataFrame [cohort, segment_key, mob, del_pct, denom_ead, numer_ead]

**Algorithm:**
```
1. Compute numerator: sum(ead where state in bad_states) per cohort, segment, mob
2. Lookup denominator từ denom_map
3. del_pct = numer_ead / denom_ead
```

### 8.3 Function: `make_mixed_report(actual_del_long, pred_del_long, max_mob)`

**Mục đích / Purpose:**
Tạo báo cáo hỗn hợp: dùng actual nếu có, forecast nếu không.

Create mixed report: use actual where available, forecast otherwise.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `actual_del_long` | DataFrame | DEL thực tế |
| `pred_del_long` | DataFrame | DEL dự báo |
| `max_mob` | int | MOB tối đa |

**Output:**
- `mixed_wide`: Wide format [cohort, segment_key, MOB_0, MOB_1, ..., MOB_24]
- `flags_wide`: "ACTUAL" hoặc "FORECAST" cho mỗi cell
- `actual_wide`: Chỉ actual values
- `forecast_wide`: Chỉ forecast values

**Logic:**
```
For each (cohort, segment_key, mob):
    If actual exists:
        mixed_value = actual
        flag = "ACTUAL"
    Else:
        mixed_value = forecast
        flag = "FORECAST"
```

**Ví dụ Output (mixed_wide):**
```
cohort   | segment_key | MOB_0 | MOB_1 | ... | MOB_12 | ... | MOB_24
---------|-------------|-------|-------|-----|--------|-----|-------
2023-01  | TOPUP       | 0.00  | 0.01  | ... | 0.08   | ... | 0.15
2023-06  | TOPUP       | 0.00  | 0.01  | ... | 0.07   | ... | 0.14*

* = FORECAST (cohort mới chưa đủ data đến MOB_24)
```


---

## 9. Module: export.py

### 9.1 Function: `export_to_excel(path, ...)`

**Mục đích / Purpose:**
Xuất tất cả kết quả ra file Excel với sheet riêng cho mỗi product và sheet Portfolio tổng hợp.

Export all results to Excel workbook with separate sheets per product and Portfolio summary.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `path` | str | Đường dẫn file output |
| `transitions_long_df` | DataFrame | Ma trận dạng long |
| `mixed_wide` | DataFrame | Báo cáo hỗn hợp |
| `flags_wide` | DataFrame | Flags ACTUAL/FORECAST |
| `actual_wide` | DataFrame | DEL thực tế |
| `forecast_wide` | DataFrame | DEL dự báo |
| `factors_df` | DataFrame | (Optional) Calibration factors |
| `forecast_df` | DataFrame | (Optional) Forecast dạng long |
| `meta_df` | DataFrame | (Optional) Metadata |

**Output Sheets Structure:**

**Portfolio Sheets (Tổng hợp):**
| Sheet Name | Nội dung |
|------------|----------|
| `Portfolio_Mixed` | DEL tổng hợp (mean của tất cả products) |
| `Portfolio_Actual` | Actual DEL tổng hợp |
| `Portfolio_Forecast` | Forecast DEL tổng hợp |
| `Portfolio_Flags` | Flags (ACTUAL/FORECAST/MIXED) |

**Per-Product Sheets (Mỗi product riêng):**
| Sheet Pattern | Ví dụ | Nội dung |
|---------------|-------|----------|
| `{Product}_Mixed` | `TOPUP_Mixed` | DEL mixed cho product TOPUP |
| `{Product}_Actual` | `TOPUP_Actual` | Actual DEL cho product TOPUP |
| `{Product}_Forecast` | `TOPUP_Forecast` | Forecast DEL cho product TOPUP |
| `{Product}_Flags` | `TOPUP_Flags` | Flags cho product TOPUP |

**Metadata Sheets:**
| Sheet Name | Nội dung |
|------------|----------|
| `transitions_long` | Tất cả transition probabilities |
| `segment_meta` | Sample sizes per segment per MOB |
| `calibration_factors` | K-factors per MOB |
| `forecast_long` | Full forecast data |

**Helper Functions:**
- `_sanitize_sheet_name()`: Đảm bảo tên sheet ≤ 31 ký tự, không có ký tự đặc biệt
- `_compute_portfolio_del()`: Tính DEL Portfolio từ tất cả products
- `_split_by_segment()`: Chia DataFrame theo segment_key

---

## 10. Module: backtest.py

### 10.1 Function: `split_cohorts(df, cfg, train_ratio=0.7)`

**Mục đích / Purpose:**
Chia dữ liệu theo cohort thành train và test sets.

Split data by cohort into train and test sets.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `df` | DataFrame | Dữ liệu gốc |
| `cfg` | dict | Config |
| `train_ratio` | float | Tỷ lệ cohorts cho training (default 0.7) |

**Output:**
- `train_df`: DataFrame cho training
- `test_df`: DataFrame cho testing

**Algorithm:**
```
1. Parse cohort từ orig_date
2. Sort cohorts theo thời gian
3. n_train = int(len(cohorts) * train_ratio)
4. train_cohorts = cohorts[:n_train]  # Cohorts cũ
5. test_cohorts = cohorts[n_train:]   # Cohorts mới
```

**Ví dụ / Example:**
```
Cohorts: [2022-01, 2022-02, ..., 2023-06, 2023-07, ..., 2023-12]
train_ratio = 0.7

Train: 2022-01 → 2023-06 (70% cohorts đầu)
Test:  2023-07 → 2023-12 (30% cohorts cuối)
```

### 10.2 Function: `compute_backtest_metrics(actual_del_long, pred_del_long, max_mob)`

**Mục đích / Purpose:**
Tính metrics backtest (MAE, MAPE) theo MOB.

Compute backtest metrics (MAE, MAPE) by MOB.

**Input:**
| Parameter | Type | Mô tả |
|-----------|------|-------|
| `actual_del_long` | DataFrame | DEL thực tế |
| `pred_del_long` | DataFrame | DEL dự báo |
| `max_mob` | int | MOB tối đa |

**Output:**
- DataFrame [mob, mae, mape, n_obs]

**Metrics:**
```
MAE (Mean Absolute Error):
MAE[mob] = mean(|actual_del - pred_del|)

MAPE (Mean Absolute Percentage Error):
MAPE[mob] = mean(|actual_del - pred_del| / actual_del)
            (chỉ tính khi actual_del > 0)
```

### 10.3 Function: `run_backtest(...)`

**Mục đích / Purpose:**
Chạy full backtest pipeline với cohort split.

Run full backtest pipeline with cohort split.

**Pipeline:**
```
1. split_cohorts() → train_df, test_df
2. prepare_transitions(train_df) → df_trans
3. estimate_transition_matrices(df_trans) → transitions_dict
4. build_initial_vectors(test_df) → df_init, denom_map
5. forecast(df_init, transitions_dict) → forecast_df
6. compute_del_from_snapshot(test_df) → actual_del_long
7. compute_del_from_forecast(forecast_df) → pred_del_long
8. (Optional) fit_del_curve_factors() → factors_df
9. compute_backtest_metrics() → metrics_df
10. make_mixed_report() → mixed_wide, flags_wide, ...
```

**Output:**
Dict với tất cả kết quả: transitions_dict, forecast_df, metrics_df, etc.


---

## 11. Thuật ngữ / Glossary

| Thuật ngữ | Tiếng Việt | English | Định nghĩa |
|-----------|------------|---------|------------|
| **MOB** | Tháng trên sổ | Months on Book | Số tháng kể từ ngày giải ngân |
| **EAD** | Dư nợ tại thời điểm vỡ nợ | Exposure at Default | Số tiền còn nợ (PRINCIPLE_OUTSTANDING) |
| **DPD** | Ngày quá hạn | Days Past Due | Số ngày trễ thanh toán |
| **DEL** | Tỷ lệ nợ xấu | Delinquency Rate | % EAD ở trạng thái xấu / EAD ban đầu |
| **Cohort** | Nhóm theo thời gian | Cohort | Nhóm khoản vay theo tháng giải ngân |
| **Segment** | Phân khúc | Segment | Nhóm theo đặc điểm (product type, etc.) |
| **Transition Matrix** | Ma trận chuyển đổi | Transition Matrix | Ma trận xác suất chuyển trạng thái |
| **Absorbing State** | Trạng thái hấp thụ | Absorbing State | Trạng thái không thể thoát ra |
| **Shrinkage** | Co rút | Shrinkage | Kỹ thuật Bayesian mượn thông tin |
| **Tail Pooling** | Gộp đuôi | Tail Pooling | Gộp matrices ở MOB cao để giảm noise |
| **Calibration** | Hiệu chỉnh | Calibration | Điều chỉnh forecast khớp actual |
| **MAE** | Sai số tuyệt đối TB | Mean Absolute Error | Trung bình |actual - pred| |
| **MAPE** | Sai số % tuyệt đối TB | Mean Absolute % Error | Trung bình |actual - pred| / actual |

---

## Phụ lục: Công thức Toán học / Mathematical Formulas

### A. Markov Chain Forecast

```
v(t+1) = v(t) × P(t)

Trong đó:
- v(t) ∈ ℝ^N: Vector EAD tại MOB t
- P(t) ∈ ℝ^(N×N): Transition matrix cho MOB t → t+1
- N = số trạng thái (7 trong model này)
```

### B. Hierarchical Shrinkage

```
P_posterior = normalize(C_observed + τ × P_prior)

Trong đó:
- C_observed: Ma trận counts quan sát được
- τ: Prior strength (hyperparameter)
- P_prior: Ma trận từ level cao hơn
- normalize(): Chia mỗi hàng cho tổng hàng
```

### C. DEL Calculation

```
DEL(cohort, segment, mob) = Σ EAD_bad(mob) / Σ EAD_total(mob=0)

Trong đó:
- EAD_bad = EAD của các trạng thái trong bad_states
- EAD_total(mob=0) = Tổng EAD tại thời điểm giải ngân
```

### D. Calibration Factor

```
k(mob) = clip(actual_mean / pred_mean, k_min, k_max)

Matrix calibration:
P'[i,bad] = P[i,bad] × k
P'[i,good] = P[i,good] × (1 - Σ P'[i,bad]) / (1 - Σ P[i,bad])

Vector calibration:
v'[bad] = v[bad] × k × (total / (bad × k + good))
v'[good] = v[good] × (total - Σ v'[bad]) / good
```

---

## Phụ lục: Ví dụ End-to-End / End-to-End Example

### Input Data (Oct25.parquet)
```
AGREEMENT_ID | MOB | STATE_MODEL | PRINCIPLE_OUTSTANDING | DISBURSAL_DATE | PRODUCT_TYPE
-------------|-----|-------------|----------------------|----------------|-------------
A001         | 0   | DPD0        | 100000               | 2023-01-15     | TOPUP
A001         | 1   | DPD0        | 98000                | 2023-01-15     | TOPUP
A001         | 2   | DPD1+       | 96000                | 2023-01-15     | TOPUP
A002         | 0   | DPD0        | 200000               | 2023-01-20     | SALPIL
...
```

### Step 1: prepare_transitions
```
loan_id | mob | from_state | to_state | ead    | cohort  | PRODUCT_TYPE
--------|-----|------------|----------|--------|---------|-------------
A001    | 0   | DPD0       | DPD0     | 100000 | 2023-01 | TOPUP
A001    | 1   | DPD0       | DPD1+    | 98000  | 2023-01 | TOPUP
A002    | 0   | DPD0       | DPD0     | 200000 | 2023-01 | SALPIL
...
```

### Step 2: estimate_transition_matrices
```
Key: ("GLOBAL", "", 0)
Matrix P(0):
         DPD0   DPD1+  DPD30+ DPD60+ DPD90+ WRITEOFF PREPAY
DPD0     0.92   0.05   0.01   0.005  0.003  0.001    0.011
DPD1+    0.35   0.40   0.15   0.05   0.03   0.01     0.01
...
```

### Step 3: build_initial_vectors
```
cohort  | segment_key | state | ead
--------|-------------|-------|--------
2023-01 | TOPUP       | DPD0  | 5000000
2023-01 | TOPUP       | DPD1+ | 100000
2023-01 | SALPIL      | DPD0  | 8000000
...
```

### Step 4: forecast
```
cohort  | segment_key | mob | state  | ead
--------|-------------|-----|--------|--------
2023-01 | TOPUP       | 0   | DPD0   | 5000000
2023-01 | TOPUP       | 0   | DPD1+  | 100000
2023-01 | TOPUP       | 1   | DPD0   | 4600000  ← v(1) = v(0) × P(0)
2023-01 | TOPUP       | 1   | DPD1+  | 350000
...
2023-01 | TOPUP       | 24  | DPD0   | 2000000
2023-01 | TOPUP       | 24  | DPD30+ | 800000
...
```

### Step 5: compute_del
```
cohort  | segment_key | mob | del_pct | denom_ead | numer_ead
--------|-------------|-----|---------|-----------|----------
2023-01 | TOPUP       | 0   | 0.00    | 5100000   | 0
2023-01 | TOPUP       | 6   | 0.05    | 5100000   | 255000
2023-01 | TOPUP       | 12  | 0.12    | 5100000   | 612000
2023-01 | TOPUP       | 24  | 0.22    | 5100000   | 1122000
```

### Final Output (Excel)
- Sheet "del30_mixed": DEL curves với actual + forecast
- Sheet "del30_flags": ACTUAL/FORECAST markers
- Sheet "transitions_long": Tất cả transition probabilities
- Sheet "calibration_factors": K-factors per MOB

---

*Document generated for Credit Risk Markov-Chain Projection Model*
*Version: 1.0 | Date: 2026-01*
