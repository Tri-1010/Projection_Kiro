# Hướng dẫn Vận hành / Operations Guide

## 1. Quy trình Phát hành / Release Checklist

### 1.1 Trước khi Phát hành / Pre-Release
- [ ] Chạy tất cả unit tests / Run all unit tests
- [ ] Chạy sanity check với dữ liệu thật / Run sanity check with real data
- [ ] Review code changes
- [ ] Cập nhật version number
- [ ] Cập nhật CHANGELOG

### 1.2 Kiểm tra Dữ liệu / Data Validation
- [ ] Schema validation passed
- [ ] No missing required columns
- [ ] Bucket values in BUCKETS_CANON
- [ ] MOB values non-negative
- [ ] EAD values non-negative

### 1.3 Kiểm tra Kết quả / Output Validation
- [ ] Excel file generated successfully
- [ ] All required sheets present
- [ ] DEL30 values in reasonable range (0-100%)
- [ ] No NaN in critical columns

## 2. Xử lý Sự cố / Troubleshooting

### 2.1 Lỗi Schema / Schema Errors
```
ValueError: Missing required columns: ['loan_id']
```
**Giải pháp / Solution**: Kiểm tra tên cột trong file parquet khớp với config.py

### 2.2 Lỗi Memory / Memory Errors
```
MemoryError: Unable to allocate array
```
**Giải pháp / Solution**: 
- Giảm số lượng segments
- Xử lý theo batch cohorts

### 2.3 Lỗi Transition Matrix / Transition Matrix Errors
```
Row sum != 1.0
```
**Giải pháp / Solution**: Kiểm tra dữ liệu đầu vào có đủ observations

## 3. Monitoring

### 3.1 Metrics to Track
- MAE by MOB
- MAPE by MOB
- Coverage (% segments with sufficient data)
- Runtime

### 3.2 Alerts
- MAE > 5% at any MOB
- MAPE > 50% at any MOB
- Coverage < 80%

## 4. Backup & Recovery

### 4.1 Backup
- Lưu file parquet đầu vào
- Lưu file Excel đầu ra
- Lưu config.py version

### 4.2 Recovery
- Restore từ backup
- Re-run pipeline với cùng config
