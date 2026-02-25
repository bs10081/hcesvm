# Class1-First 策略測試結果：Contraceptive

**測試時間**: 2026-02-24 02:18:34
**策略**: Class1-First (固定分類規則)
**資料集**: Contraceptive (`~/Developer/NSVORA/Archive/contraceptive_split.xlsx`)
**時間限制**: 1800 秒 (30 分鐘) per classifier

---

## 資料分布

### Training Set
- **Class 1**: 503 samples (42.7%)
- **Class 2**: 266 samples (22.6%)
- **Class 3**: 409 samples (34.7%)
- **Total**: 1,178 samples
- **Features**: 9

### Testing Set
- **Class 1**: 126 samples (42.7%)
- **Class 2**: 67 samples (22.7%)
- **Class 3**: 102 samples (34.6%)
- **Total**: 295 samples

**✅ 資料分布相對平衡**

---

## 分類策略

### H1 分類器: Class 1 (+1) vs {Class 2, 3} (-1)

**訓練資料**:
- Positive class (+1, Class 1): 503 samples
- Negative class (-1, Class 2&3): 675 samples

**優化結果**:
- **Status**: Time Limit Reached (Status code: 9)
- **Best Objective**: 1005.0
- **Best Bound**: 28.08
- **MIP Gap**: 74.12% ⚠️
- **Runtime**: 1800.01 seconds (達到時間限制)
- **Explored Nodes**: 476,756
- **Simplex Iterations**: 83,104,084

**模型參數**:
```
Weights (w): [0. 0. 0. 0. 0. 0. 0. 0. 2.]
Intercept (b): -1.0
```

**特徵使用**:
- Selected features: 9/9 (所有特徵)
- L1 norm: 2.0
- 只有第 9 個特徵有非零權重 (w[8] = 2.0)

**準確率下界**:
- Positive class (Class 1) accuracy lb: 數據未提供
- Negative class (Class 2&3) accuracy lb: 數據未提供

### H2 分類器: Class 2 (+1) vs Class 3 (-1)

**訓練資料**:
- Positive class (+1, Class 2): 266 samples
- Negative class (-1, Class 3): 409 samples

**優化結果**:
- **Status**: Time Limit Reached (Status code: 9)
- **Best Objective**: 536.74
- **Best Bound**: 201.01
- **MIP Gap**: 62.55% ⚠️
- **Runtime**: 1800.01 seconds (達到時間限制)
- **Explored Nodes**: 476,756
- **Simplex Iterations**: 83,104,084

**模型參數**:
```
Weights (w): [ 0.  2.  0.  0.  0.  0.  0.  0. -2.]
Intercept (b): -7.0
```

**特徵使用**:
- Selected features: 9/9 (所有特徵)
- L1 norm: 4.0
- 第 2 個特徵: w[1] = +2.0
- 第 9 個特徵: w[8] = -2.0

**準確率下界**:
- Positive class (Class 2) accuracy lb: 0.6203
- Negative class (Class 3) accuracy lb: 0.6430

---

## 預測結果

### Training Set

| Metric | Value | Correct/Total |
|--------|-------|---------------|
| **Class 1 Accuracy** | 0.1213 | 61/503 |
| **Class 2 Accuracy** | 0.6203 | 165/266 |
| **Class 3 Accuracy** | 0.5941 | 243/409 |
| **Total Accuracy** | **0.3981** | 469/1,178 |

### Testing Set

| Metric | Value | Correct/Total |
|--------|-------|---------------|
| **Class 1 Accuracy** | 0.1032 | 13/126 |
| **Class 2 Accuracy** | 0.6269 | 42/67 |
| **Class 3 Accuracy** | 0.5098 | 52/102 |
| **Total Accuracy** | **0.3627** | 107/295 |

---

## 結果分析

### 性能表現

1. **Class 1 分類表現極差** ⚠️
   - Training: 12.13%
   - Testing: 10.32%
   - **問題**：H1 分類器無法有效識別 Class 1

2. **Class 2 分類表現良好** ✅
   - Training: 62.03%
   - Testing: 62.69%
   - H2 分類器對 Class 2 有較好的識別能力

3. **Class 3 分類表現中等** ⚠️
   - Training: 59.41%
   - Testing: 50.98%
   - 有一定的過擬合現象

4. **總體準確率偏低** ❌
   - Training: 39.81%
   - Testing: 36.27%
   - 遠低於 baseline（隨機猜測約 33%，僅略高）

### 優化問題分析

#### H1 分類器問題

**MIP Gap 74.12% - 解的品質差**

1. **特徵選擇過於簡單**：
   - 只使用一個特徵（w[8] = 2.0）
   - 決策函數：f(x) = 2·x[8] - 1
   - 這意味著 H1 幾乎只依賴第 9 個特徵來分類

2. **分類邏輯**：
   - 當 x[8] >= 0.5 時，預測為 +1 (Class 1)
   - 當 x[8] < 0.5 時，預測為 -1 (進入 H2)

3. **為什麼 Class 1 準確率這麼低**：
   - H1 的簡單決策規則無法有效捕捉 Class 1 的特徵
   - 大部分 Class 1 樣本被誤分到 H2，進而被誤分為 Class 2 或 3

#### H2 分類器表現

**MIP Gap 62.55% - 解的品質中等**

1. **特徵使用較合理**：
   - 使用兩個特徵：w[1] = +2.0, w[8] = -2.0
   - 決策函數：f(x) = 2·x[1] - 2·x[8] - 7

2. **分類效果較好**：
   - Class 2 vs Class 3 的分類準確率都在 50-63%
   - 相較於 H1，H2 的性能明顯更好

### 時間限制的影響

**兩個分類器都達到時間限制但未收斂**：

1. **H1 Gap 74%**：
   - 離最優解還很遠
   - 可能需要數小時才能找到更好的解

2. **H2 Gap 62%**：
   - 同樣離最優解較遠
   - 但相對 H1 已經找到較好的解

3. **權衡考慮**：
   - 增加時間限制 → 可能找到更好的解
   - 但不保證能顯著改善（問題可能本質上就很難）

---

## 與 Thyroid 的對比

| 指標 | Thyroid | Contraceptive |
|------|---------|---------------|
| **資料平衡度** | 極度不平衡 (92.5% Class 3) | 相對平衡 (34-43%) |
| **H1 解的類型** | 平凡解 (w=0, b=-1) | 簡單解 (只用一個特徵) |
| **H2 解的類型** | 平凡解 (w=0, b=-1) | 合理解 (用兩個特徵) |
| **Training Acc** | 92.53% (只因為 Class 3 多) | 39.81% |
| **Testing Acc** | 92.36% (只因為 Class 3 多) | 36.27% |
| **Class 1 Acc** | 0.00% | 10.32% |
| **收斂狀態** | 最優解 (Gap 0%) | 未收斂 (Gap 62-74%) |

**關鍵觀察**：
- Thyroid 找到了（不好的）最優解
- Contraceptive 在時間限制內未找到最優解，但至少學到了一些特徵

---

## 結論

### 主要發現

1. ❌ **Class1-First 策略在 Contraceptive 資料集上表現不佳**
   - Testing accuracy 僅 36.27%
   - Class 1 幾乎無法被正確分類（10.32%）

2. ⚠️ **優化未收斂**
   - H1 Gap: 74%
   - H2 Gap: 62%
   - 需要更長的時間或更好的優化策略

3. 📊 **H1 是瓶頸**
   - H1 分類器過於簡單（只用一個特徵）
   - 導致 Class 1 無法被有效識別
   - 大量 Class 1 樣本流入 H2，被誤分為 Class 2 或 3

4. ✅ **H2 表現相對較好**
   - 使用兩個特徵
   - Class 2 和 Class 3 的準確率都在 50-63%

### 建議

#### 短期改進

1. **增加時間限制**：
   ```python
   --time-limit 3600  # 1 小時
   --time-limit 7200  # 2 小時
   ```

2. **調整 C 參數**：
   ```python
   --C 10.0  # 增加準確率項的權重
   ```

3. **使用 Warm Start**：
   - 使用啟發式方法初始化

#### 中期改進

1. **使用 class_weight='balanced'**：
   ```python
   cesvm_params = {
       'class_weight': 'balanced',
       ...
   }
   ```

2. **調整目標函數**：
   - 對 Class 1 使用更高的權重
   - 考慮使用 Test3 策略（sample-weighted）

#### 長期改進

1. **特徵工程**：
   - 分析哪些特徵對 Class 1 最有區分度
   - 考慮特徵組合或轉換

2. **嘗試不同的分類策略**：
   - Test2 策略（動態 minority class）
   - Test3 策略（balanced weighting）
   - One-vs-Rest (OvR)
   - One-vs-One (OvO)

3. **集成方法**：
   - 結合多個分類器的預測
   - Voting 或 Stacking

---

## 下一步行動

### 建議的測試順序

1. **✅ 完成 Thyroid 和 Contraceptive 測試** ← 已完成

2. **對比測試**（使用相同資料集）：
   - Test2 策略
   - Test3 策略
   - 比較三種策略的性能

3. **其他資料集測試**（Class1-First 策略）：
   - Balance
   - Hayes_Roth
   - New_Thyroid
   - TAE
   - Wine
   - Car_Evaluation

4. **優化參數調整**：
   - 增加時間限制
   - 調整 C 參數
   - 使用 class weighting

---

## 參考資料

- 測試腳本: `examples/run_class1_first_test.py`
- 完整日誌: `results/class1_first_contraceptive_20260224_021834.log`
- 輸出日誌: `results/class1_first_contraceptive_full.log`
- Thyroid 報告: `CLASS1_FIRST_THYROID_FULL_REPORT.md`

---

**生成時間**: 2026-02-24
**策略版本**: Class1-First v1.0
**時間限制**: 1800 秒/分類器
