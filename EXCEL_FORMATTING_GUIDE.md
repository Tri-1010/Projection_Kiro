# Excel Formatting Guide - Hướng dẫn định dạng Excel

## Tổng quan / Overview

Hệ thống xuất Excel đã được cập nhật với các tính năng định dạng chuyên nghiệp theo yêu cầu:

The Excel export system has been updated with professional formatting features as requested:

## Các tính năng mới / New Features

### 1. **Title Format - Định dạng tiêu đề**
- **Vị trí**: Dòng 1 của mỗi sheet Mixed
- **Format**: `{SEGMENT_KEY}_{DEL_TYPE} Actual & Forecast`
- **Ví dụ**: `CDLPIL_DEL30 Actual & Forecast`
- **Style**: 
  - Font size: 20
  - Bold: ✅
  - Color: Dark Blue (#1F4E79)
  - Alignment: Center
  - Merged cells across all columns

### 2. **Header Format - Định dạng header**
- **Vị trí**: Từ dòng 3
- **Style**:
  - Bold: ✅
  - Background: Light Blue (#D9E1F2)
  - Alignment: Center
  - Border: Thin border

### 3. **Value Format - Định dạng giá trị**
- **MOB columns**: Format percentage với 2 decimal places (`0.00%`)
- **Ví dụ**: `0.0523` → `5.23%`
- **Non-MOB columns**: Giữ nguyên format gốc

### 4. **Color Scale - Thang màu**
- **Áp dụng**: Tất cả MOB columns (MOB_0 đến MOB_cuối)
- **Type**: Green-Yellow-Red gradient (xanh thấp, đỏ cao)
- **Colors**:
  - 🟢 Green (#63BE7B): Giá trị thấp nhất (tốt nhất - ít delinquency)
  - 🟡 Yellow (#FFEB9C): Giá trị trung bình (50th percentile)
  - 🔴 Red (#F8696B): Giá trị cao nhất (xấu nhất - nhiều delinquency)

### 5. **Border System - Hệ thống viền**
- **Standard Border**: Thin border cho tất cả cells
- **Special Border**: Thick red border (#FF0000) để phân biệt ACTUAL và FORECAST:
  - Đặt ở **cạnh phải** và **cạnh dưới** của cell ACTUAL cuối cùng (trước khi chuyển sang FORECAST)
  - Giúp nhận biết rõ ranh giới giữa dữ liệu thực tế và dự báo

### 6. **Grid Lines - Đường lưới**
- **Status**: ❌ Disabled (bỏ grid lines)
- **Lý do**: Tạo giao diện sạch sẽ, chuyên nghiệp

## Cấu trúc Sheet / Sheet Structure

### Mixed Sheets (Định dạng đặc biệt)
```
Row 1: [TITLE] SEGMENT_DEL Actual & Forecast
Row 2: [EMPTY]
Row 3: [HEADERS] cohort | segment_key | MOB_0 | MOB_1 | ...
Row 4+: [DATA] với formatting đầy đủ
```

### Other Sheets (Định dạng chuẩn)
```
Row 1: [TITLE] SEGMENT DEL_TYPE Type
Row 2: [EMPTY]  
Row 3: [HEADERS] với background color
Row 4+: [DATA] với percentage formatting
```

## Các file được cập nhật / Updated Files

### 1. `export.py`
- ✅ Added `_format_mixed_sheet()` - Format đặc biệt cho Mixed sheets
- ✅ Added `_format_standard_sheet()` - Format chuẩn cho các sheet khác
- ✅ Updated `export_all_del_to_excel()` - Áp dụng formatting cho DEL30/60/90
- ✅ Updated `export_to_excel()` - Áp dụng formatting cho single DEL
- ✅ Added openpyxl styling imports

### 2. `test_formatting.py`
- ✅ Test script để demo các tính năng formatting
- ✅ Tạo dữ liệu test với DEL30/60/90
- ✅ Xuất file `test_formatted_output.xlsx` để kiểm tra

## Cách sử dụng / Usage

### Trong Notebooks
```python
from export import export_all_del_to_excel

# Xuất với formatting đầy đủ
export_all_del_to_excel(
    'output.xlsx',
    transitions_long_df,
    del_results,  # dict với keys: 'del30', 'del60', 'del90'
    factors_df=factors_df,
    forecast_df=forecast_df,
    meta_df=meta_df
)
```

### Test Formatting
```bash
python test_formatting.py
```
Sẽ tạo file `test_formatted_output.xlsx` để xem preview formatting.

## Kết quả / Results

### ✅ Hoàn thành
- [x] Values format 2 decimal places với % (`0.00%`)
- [x] Sheet bỏ grid lines
- [x] Title dòng 1: `{SEGMENT}_{DEL} Actual & Forecast`, size 20, bold, Dark Blue
- [x] Headers từ dòng 3: bold, background color, center alignment
- [x] Color scale Green-Yellow-Red cho MOB columns (xanh thấp, đỏ cao)
- [x] Border đỏ dày ở cạnh phải và dưới của cell ACTUAL cuối cùng (ranh giới với FORECAST)
- [x] Border bình thường cho tất cả cells khác

### 📊 Sheets được format
- **Mixed Sheets**: Định dạng đặc biệt với title, color scale, borders
- **Actual/Forecast/Flags Sheets**: Định dạng chuẩn với title và percentage
- **Metadata Sheets**: Định dạng cơ bản với title

## Lưu ý kỹ thuật / Technical Notes

### Dependencies
```python
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.utils import get_column_letter
```

### Performance
- Formatting được áp dụng sau khi write data
- Color scale chỉ áp dụng cho MOB columns để tối ưu performance
- Border logic được tối ưu để tránh conflict

### Compatibility
- ✅ Compatible với existing notebooks
- ✅ Backward compatible với old export functions
- ✅ Works với cả DEL30/60/90 và single DEL exports

---

**Tác giả**: Kiro AI Assistant  
**Ngày cập nhật**: January 2026  
**Version**: 2.0 - Professional Excel Formatting