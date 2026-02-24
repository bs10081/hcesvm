# Class1-First 策略測試總結

**執行時間**: 2026-02-24 02:07:19 - 03:48:35
**測試資料集**: Thyroid, Contraceptive

---

## 測試完成狀態

| 資料集 | 狀態 | Training Acc | Testing Acc | 報告 |
|--------|------|--------------|-------------|------|
| **Thyroid** | ✅ 完成 | 92.53% | 92.36% | `CLASS1_FIRST_THYROID_FULL_REPORT.md` |
| **Contraceptive** | ✅ 完成 | 39.81% | 36.27% | `CLASS1_FIRST_CONTRACEPTIVE_FULL_REPORT.md` |

---

## 關鍵發現

### Thyroid 資料集

**問題**: 平凡解（Trivial Solution）

- **資料分布**: 極度不平衡（Class 3: 92.5%）
- **H1 和 H2**: 兩個分類器都學到全零權重，intercept = -1
- **結果**: 所有樣本都被預測為 Class 3
- **Per-Class Accuracy**:
  - Class 1: 0.00%
  - Class 2: 0.00%
  - Class 3: 100.00%
- **原因**: 優化器發現犧牲少數類來最大化多數類準確率是最優的

### Contraceptive 資料集

**問題**: 優化未收斂 + H1 瓶頸

- **資料分布**: 相對平衡（Class 1: 42.7%, Class 2: 22.6%, Class 3: 34.7%）
- **H1 (Gap 74%)**: 只使用一個特徵 (w[8]=2.0)，無法有效識別 Class 1
- **H2 (Gap 62%)**: 使用兩個特徵，表現較好
- **Per-Class Accuracy (Testing)**:
  - Class 1: 10.32% ⚠️
  - Class 2: 62.69% ✅
  - Class 3: 50.98% ⚠️
- **總體**: Testing Accuracy 36.27%（略高於隨機猜測的 33%）

---

## 程式碼追蹤確認

### Class1-First 策略實作 ✅

檢查 `examples/run_class1_first_test.py` 確認：

1. **H1 分類規則** ✅
   ```python
   # H1: Class 1 (+1) vs {Class 2, 3} (-1)
   X_h1_pos = X1  # Class 1 → +1
   X_h1_neg = np.vstack([X2, X3])  # Class 2, 3 → -1
   ```

2. **H2 分類規則** ✅
   ```python
   # H2: Class 2 (+1) vs Class 3 (-1)
   X_h2_pos = X2  # Class 2 → +1
   X_h2_neg = X3  # Class 3 → -1
   ```

3. **預測邏輯** ✅
   ```python
   f1 = self.h1.decision_function(X)
   if f1 >= 0:
       return 1  # Class 1
   else:
       f2 = self.h2.decision_function(X)
       return 2 if f2 >= 0 else 3
   ```

**結論**: 分類規則實作正確，符合 Class1-First 策略定義。

---

## 主要問題分析

### 1. 資料不平衡導致平凡解（Thyroid）

**問題**：
- CE-SVM 的目標函數在極度不平衡時會偏向多數類
- 優化器發現全零權重 + 負 intercept 是數學上的最優解

**解決方案**：
- 使用 `class_weight='balanced'`
- 調整目標函數中準確率項的權重
- 使用 SMOTE 等過採樣技術

### 2. 優化未收斂（Contraceptive）

**問題**：
- 時間限制 1800 秒不足以收斂
- H1 Gap 74%, H2 Gap 62%

**解決方案**：
- 增加時間限制（3600 或 7200 秒）
- 調整 C 參數（增加準確率項權重）
- 使用 warm start

### 3. H1 分類器是瓶頸（Contraceptive）

**問題**：
- H1 只學到一個特徵，過於簡單
- 無法有效區分 Class 1 vs {Class 2, 3}
- 導致 Class 1 準確率僅 10%

**解決方案**：
- 調整 C 參數或 M 參數
- 考慮對 H1 使用更嚴格的準確率要求
- 使用特徵選擇或特徵工程

---

## 與其他策略的對比需求

根據測試結果，建議對比：

### 需要比較的策略

1. **Test2 策略**（動態 minority class）
   - 根據 minority class 位置動態調整分類規則
   - 移除 majority class 的準確率項

2. **Test3 策略**（sample-weighted）
   - 使用樣本數量倒數作為權重
   - 自動平衡不同類別的重要性

3. **Class1-First 策略**（固定規則）← 本次測試
   - 固定的分類順序
   - 不考慮資料分布

### 對比指標

- Per-class accuracy
- Total accuracy
- MIP Gap
- Training time
- 模型複雜度（特徵數量、權重大小）

---

## 生成的檔案

### 測試結果

1. **日誌檔案**:
   - `results/class1_first_thyroid_20260224_020719.log`
   - `results/class1_first_thyroid_run.log`
   - `results/class1_first_contraceptive_20260224_021834.log`
   - `results/class1_first_contraceptive_run.log`
   - `results/class1_first_contraceptive_full.log`

2. **詳細報告**:
   - `CLASS1_FIRST_THYROID_FULL_REPORT.md`
   - `CLASS1_FIRST_CONTRACEPTIVE_FULL_REPORT.md`

3. **總結報告**:
   - `CLASS1_FIRST_SUMMARY.md` (本檔案)

### 測試腳本

- `examples/run_class1_first_test.py`

---

## 建議的下一步

### 優先級 1: 對比測試（相同資料集）

使用 Thyroid 和 Contraceptive 測試：
- Test2 策略
- Test3 策略
- 比較三種策略在相同資料集上的表現

### 優先級 2: 改進 Class1-First 策略

1. 增加時間限制（3600 秒）
2. 調整 C 參數（C=10.0）
3. 使用 class_weight='balanced'

### 優先級 3: 擴展到其他資料集

測試 Class1-First 在其他資料集上的表現：
- Balance
- Hayes_Roth
- New_Thyroid
- TAE
- Wine
- Car_Evaluation

---

## MEMORY.md 更新狀態

**已更新**: Class1-First 策略的說明已加入 `MEMORY.md`

---

**生成時間**: 2026-02-24 03:50:00
**策略版本**: Class1-First v1.0
**測試完成**: 2/9 資料集
