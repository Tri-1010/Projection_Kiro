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
- Với MOB >= TAIL_POOL_START: lấy trung bình ma trận
- Giảm nhiễu cho các MOB cao

### 4.4 Trạng thái Hấp thụ / Absorbing States
- DPD90+ và CLOSED là trạng thái hấp thụ
- Hàng ma trận là one-hot (tự chuyển về chính nó)

## 5. Đầu ra / Outputs

### Excel Report Sheets:
1. `transitions_long`: Ma trận chuyển đổi dạng dài
2. `segment_meta`: Metadata phân khúc
3. `del30_mixed`: DEL30 hỗn hợp (actual/forecast)
4. `del30_flags`: Cờ ACTUAL/FORECAST
5. `del30_actual`: DEL30 thực tế
6. `del30_forecast`: DEL30 dự báo
7. `calibration_factors`: Hệ số hiệu chỉnh (nếu có)
8. `forecast_long`: Dự báo dạng dài
