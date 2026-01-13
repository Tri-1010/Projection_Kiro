# Hướng dẫn Sử dụng / User Guide

## 1. Cài đặt / Installation

```bash
pip install pandas numpy openpyxl
```

## 2. Cấu trúc Dữ liệu Đầu vào / Input Data Structure

File parquet cần có các cột sau / Required columns:
- `loan_id`: ID khoản vay
- `mob`: Months on Book (0, 1, 2, ...)
- `bucket`: Trạng thái nợ (CURRENT, DPD30, DPD60, DPD90+, CLOSED)
- `ead`: Exposure at Default
- `orig_date`: Ngày giải ngân (YYYY-MM-DD)
- `product`: Phân khúc sản phẩm (tùy chọn)
- `channel`: Kênh phân phối (tùy chọn)

## 3. Chạy Pipeline / Running the Pipeline

### CLI
```bash
python main.py --input data.parquet --output out/report.xlsx --calibrate
```

### Python
```python
from data_io import load_parquet
from transitions import prepare_transitions, estimate_transition_matrices
from forecast import build_initial_vectors, forecast
from metrics import compute_del_from_snapshot, compute_del_from_forecast, make_mixed_report
from export import export_to_excel
```

## 4. Cơ chế Mô hình / Model Mechanics

### 4.1 Chuỗi Markov không đồng nhất / Time-Inhomogeneous Markov Chain
- Một ma trận chuyển đổi cho mỗi MOB (0..23)
- Dự báo đến MOB=24

### 4.2 Shrinkage Phân cấp / Hierarchical Shrinkage
- GLOBAL: Ước lượng từ toàn bộ dữ liệu
- COARSE: Posterior = counts + τ_coarse × P_global
- FULL: Posterior = counts + τ_full × P_coarse

### 4.3 Tail Pooling
- Điều khiển bởi `TAIL_POOL_ENABLED` trong config.py (True/False)
- Với MOB >= TAIL_POOL_START: lấy trung bình ma trận
- Giảm nhiễu cho các MOB cao
- Mặc định: `TAIL_POOL_ENABLED = False`

### 4.4 Trạng thái Hấp thụ / Absorbing States
- DPD90+ và CLOSED là trạng thái hấp thụ
- Hàng ma trận là one-hot (tự chuyển về chính nó)

## 5. Đầu ra / Outputs

### Excel Report Structure:

**Portfolio Sheets (Tổng hợp tất cả products):**
1. `Portfolio_Mixed`: DEL30 tổng hợp (actual + forecast)
2. `Portfolio_Actual`: DEL30 thực tế tổng hợp
3. `Portfolio_Forecast`: DEL30 dự báo tổng hợp
4. `Portfolio_Flags`: Cờ ACTUAL/FORECAST/MIXED

**Per-Product Sheets (Mỗi product riêng biệt):**
- `{PRODUCT}_Mixed`: DEL30 hỗn hợp cho product
- `{PRODUCT}_Actual`: DEL30 thực tế cho product
- `{PRODUCT}_Forecast`: DEL30 dự báo cho product
- `{PRODUCT}_Flags`: Cờ ACTUAL/FORECAST cho product

Ví dụ: `TOPUP_Mixed`, `SALPIL_Mixed`, `XSELL_Mixed`, etc.

**Metadata Sheets:**
1. `transitions_long`: Ma trận chuyển đổi dạng dài
2. `segment_meta`: Metadata phân khúc
3. `calibration_factors`: Hệ số hiệu chỉnh (nếu có)
4. `forecast_long`: Dự báo dạng dài
