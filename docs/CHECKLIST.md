# Danh sách Kiểm tra Chất lượng / Quality Assurance Checklist

## 1. transitions.py

### 1.1 prepare_transitions
- [ ] **VI**: Logic next_state đúng (shift -1 theo loan_id)
- [ ] **EN**: Correct next_state logic (shift -1 by loan_id)

```python
# Verify: to_state should be next mob's bucket for same loan
df_trans.groupby('loan_id').apply(lambda g: g.sort_values('mob'))
```

- [ ] **VI**: Xử lý duplicate (loan_id, mob) - giữ bản ghi mới nhất
- [ ] **EN**: Handle duplicate (loan_id, mob) - keep latest record

- [ ] **VI**: Parse cohort đúng định dạng YYYY-MM
- [ ] **EN**: Correct cohort parsing to YYYY-MM format

### 1.2 Absorbing States
- [ ] **VI**: Hàng absorbing là one-hot (chỉ có 1 ở đường chéo)
- [ ] **EN**: Absorbing rows are one-hot (only 1 on diagonal)

```python
# Verify absorbing rows
for state in ['DPD90+', 'CLOSED']:
    row = P.loc[state]
    assert row[state] == 1.0 and row.sum() == 1.0
```

- [ ] **VI**: Auto-expand absorbing states (DPD120+, DPD180+ nếu có)
- [ ] **EN**: Auto-expand absorbing states (DPD120+, DPD180+ if present)

### 1.3 Hierarchical Shrinkage
- [ ] **VI**: GLOBAL → COARSE → FULL đúng thứ tự
- [ ] **EN**: GLOBAL → COARSE → FULL correct order

- [ ] **VI**: Posterior = counts + τ × P_parent
- [ ] **EN**: Posterior = counts + τ × P_parent

```python
# Verify shrinkage
assert tau_coarse == PRIOR_STRENGTH_COARSE
assert tau_full == PRIOR_STRENGTH_FULL
```

### 1.4 Tail Pooling
- [ ] **VI**: Ma trận MOB >= TAIL_POOL_START được pooled
- [ ] **EN**: Matrices for MOB >= TAIL_POOL_START are pooled

```python
# Verify tail pooling
for mob in range(TAIL_POOL_START, MAX_MOB):
    assert np.allclose(P[mob], P_pooled)
```

### 1.5 Row Sums
- [ ] **VI**: Tổng mỗi hàng = 1.0 (float safe)
- [ ] **EN**: Each row sums to 1.0 (float safe)

```python
for mob in range(MAX_MOB):
    P = transitions_dict[('GLOBAL', '', mob)]
    assert np.allclose(P.sum(axis=1), 1.0)
```

---

## 2. forecast.py

### 2.1 Totals Conservation
- [ ] **VI**: Tổng EAD không đổi qua các MOB (trừ khi có CLOSED)
- [ ] **EN**: Total EAD conserved across MOBs (unless CLOSED enabled)

```python
# Verify conservation
for cohort in forecast_df['cohort'].unique():
    totals = forecast_df[forecast_df['cohort']==cohort].groupby('mob')['ead'].sum()
    # Should be constant if no closure
```

### 2.2 Matrix Selection
- [ ] **VI**: Fallback đúng: FULL → COARSE → GLOBAL
- [ ] **EN**: Correct fallback: FULL → COARSE → GLOBAL

### 2.3 Actual Override
- [ ] **VI**: actual_override_df thay thế forecast đúng MOB
- [ ] **EN**: actual_override_df correctly replaces forecast at MOB

---

## 3. metrics.py

### 3.1 DEL30 Denominator
- [ ] **VI**: Mẫu số là EAD tại MOB=0
- [ ] **EN**: Denominator is EAD at MOB=0

```python
# Verify denominator
assert denom_map[(cohort, seg)] == df_mob0[df_mob0['cohort']==cohort]['ead'].sum()
```

### 3.2 Mixed Report
- [ ] **VI**: mixed_wide dùng actual nếu có, forecast nếu không
- [ ] **EN**: mixed_wide uses actual if available, forecast otherwise

### 3.3 Flags
- [ ] **VI**: flags_wide đúng "ACTUAL" hoặc "FORECAST"
- [ ] **EN**: flags_wide correctly shows "ACTUAL" or "FORECAST"

```python
# Verify flags
for col in mob_cols:
    mask_actual = flags_wide[col] == 'ACTUAL'
    assert actual_wide.loc[mask_actual, col].notna().all()
```

---

## 4. calibration.py

### 4.1 K-Factors
- [ ] **VI**: k[mob] được clip trong [k_min, k_max]
- [ ] **EN**: k[mob] clipped within [k_min, k_max]

```python
assert (factors_df['k'] >= K_CLIP[0]).all()
assert (factors_df['k'] <= K_CLIP[1]).all()
```

### 4.2 Safe Division
- [ ] **VI**: Xử lý pred_mean = 0 an toàn
- [ ] **EN**: Handle pred_mean = 0 safely

### 4.3 Matrix Calibration
- [ ] **VI**: Row sum = 1 sau calibration
- [ ] **EN**: Row sum = 1 after calibration

- [ ] **VI**: Absorbing rows không đổi
- [ ] **EN**: Absorbing rows unchanged

### 4.4 Vector Calibration
- [ ] **VI**: Tổng EAD không đổi
- [ ] **EN**: Total EAD preserved

```python
assert np.isclose(v2.sum(), v.sum())
```

---

## 5. export.py

### 5.1 Required Sheets
- [ ] **VI**: Tất cả sheets bắt buộc tồn tại
- [ ] **EN**: All required sheets exist

```python
required = ['transitions_long', 'del30_mixed', 'del30_flags', 'del30_actual', 'del30_forecast']
wb = openpyxl.load_workbook(path)
assert all(s in wb.sheetnames for s in required)
```

### 5.2 Sheet Names
- [ ] **VI**: Tên sheet <= 31 ký tự
- [ ] **EN**: Sheet names <= 31 characters

### 5.3 MOB Columns
- [ ] **VI**: Cột MOB_0 đến MOB_24 đầy đủ
- [ ] **EN**: Columns MOB_0 to MOB_24 complete

---

## 6. Test Commands

```bash
# Run sanity check
python scripts/run_sanity.py

# Run unit tests
pytest tests/ -v

# Run specific test
pytest tests/test_sanity.py -v
```
