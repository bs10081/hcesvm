# Class1-First 策略測試結果：Thyroid

**測試時間**: 2026-02-24 02:07:19
**策略**: Class1-First (固定分類規則)
**資料集**: Thyroid (`~/Developer/NSVORA/Archive/thyroid_split.xlsx`)

---

## 資料分布

### Training Set
- **Class 1**: 14 samples (2.4%)
- **Class 2**: 29 samples (5.0%)
- **Class 3**: 533 samples (92.5%)
- **Total**: 576 samples
- **Features**: 21

### Testing Set
- **Class 1**: 3 samples (2.1%)
- **Class 2**: 8 samples (5.6%)
- **Class 3**: 133 samples (92.4%)
- **Total**: 144 samples

**⚠️ 資料極度不平衡**：Class 3 佔超過 92%

---

## 分類策略

### H1 分類器: Class 1 (+1) vs {Class 2, 3} (-1)

**訓練資料**:
- Positive class (+1, Class 1): 14 samples
- Negative class (-1, Class 2&3): 562 samples

**優化結果**:
- **Status**: Optimal (Status code: 2)
- **Objective Value**: 27.0
- **MIP Gap**: 0.000000
- **Runtime**: 0.32 seconds
- **Explored Nodes**: 1
- **Simplex Iterations**: 3,291

**模型參數**:
```
Weights (w): [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Intercept (b): -1.0
```

**準確率下界**:
- Positive class accuracy lb: 0.0000
- Negative class accuracy lb: 1.0000

### H2 分類器: Class 2 (+1) vs Class 3 (-1)

**訓練資料**:
- Positive class (+1, Class 2): 29 samples
- Negative class (-1, Class 3): 533 samples

**優化結果**:
- **Status**: Optimal (Status code: 2)
- **Objective Value**: 57.0
- **MIP Gap**: 0.000000
- **Runtime**: 1.52 seconds
- **Explored Nodes**: 3,933
- **Simplex Iterations**: 228,789

**模型參數**:
```
Weights (w): [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]
Intercept (b): -1.0
```

**準確率下界**:
- Positive class accuracy lb: 0.0000
- Negative class accuracy lb: 1.0000

---

## 預測結果

### Training Set

| Metric | Value |
|--------|-------|
| **Class 1 Accuracy** | 0.0000 (0/14) |
| **Class 2 Accuracy** | 0.0000 (0/29) |
| **Class 3 Accuracy** | 1.0000 (533/533) |
| **Total Accuracy** | 0.9253 (533/576) |

### Testing Set

| Metric | Value |
|--------|-------|
| **Class 1 Accuracy** | 0.0000 (0/3) |
| **Class 2 Accuracy** | 0.0000 (0/8) |
| **Class 3 Accuracy** | 1.0000 (133/133) |
| **Total Accuracy** | 0.9236 (133/144) |

---

## 問題分析

### 平凡解問題 (Trivial Solution)

**現象**:
- 兩個分類器都學到了全零權重和 intercept = -1.0
- 所有樣本的決策函數值都是 f(x) = w·x + b = 0 + (-1) = -1 < 0
- 因此所有樣本都被預測為負類

**預測路徑**:
1. H1 預測所有樣本為 -1 → 進入 H2
2. H2 預測所有樣本為 -1 → 最終輸出 Class 3
3. 結果：所有樣本都被分類為 Class 3

**根本原因**:

由於資料極度不平衡（Class 3 佔 92.5%），優化器發現以下策略可以最小化目標函數：

```
目標函數: min Σ|w| + C·Σ(slack) - l_positive - l_negative
```

對於 H1:
- 設 w = 0, b = -1 → 所有樣本預測為 -1
- Negative class (Class 2&3) accuracy = 562/562 = 100%
- Positive class (Class 1) accuracy = 0/14 = 0%
- 由於 negative class 樣本數遠多於 positive class，這個解在目標函數中是有利的

對於 H2:
- 同樣的邏輯：w = 0, b = -1
- Negative class (Class 3) accuracy = 533/533 = 100%
- Positive class (Class 2) accuracy = 0/29 = 0%

**數學解釋**:

在極度不平衡的情況下，犧牲少數類的準確率來獲得多數類的完美準確率，對於標準的 CE-SVM 目標函數來說是最優的。

---

## 結論與建議

### 結論

1. ❌ **Class1-First 策略在 Thyroid 資料集上失敗**
2. ❌ **無法有效分類 Class 1 和 Class 2**
3. ✅ **優化器成功達到數學上的最優解**（但是平凡解）
4. ⚠️ **資料不平衡是主要問題**

### 建議

1. **使用 Class Weighting**:
   ```python
   cesvm_params = {
       'class_weight': 'balanced',  # 使用樣本數量倒數作為權重
       ...
   }
   ```

2. **使用 Test3 策略**:
   ```python
   model = HierarchicalCESVM(
       strategy='test3',  # 使用平衡權重
       ...
   )
   ```

3. **調整目標函數**:
   - 增加準確率項的權重
   - 對少數類使用更高的權重
   - 考慮使用 SMOTE 等過採樣技術

4. **資料預處理**:
   - 考慮對少數類進行過採樣
   - 或對多數類進行欠採樣
   - 使用 stratified sampling

---

## 參考資料

- 測試腳本: `examples/run_class1_first_test.py`
- 完整日誌: `results/class1_first_thyroid_20260224_020719.log`
- 策略說明: `CLAUDE.md` 和 `MEMORY.md`

---

**生成時間**: 2026-02-24
**策略版本**: Class1-First v1.0
