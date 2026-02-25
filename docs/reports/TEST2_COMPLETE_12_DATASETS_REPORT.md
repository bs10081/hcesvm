# Test2 策略 - 完整準確率與權重報告（12 個資料集）

**生成時間**: 2026-02-04 07:36:46
**資料集數量**: 12

---

## 📊 總覽表

| # | Dataset | Test2 | Train Acc | Test Acc | H1 L1 Norm | H2 L1 Norm | Features |
|---|---------|-------|-----------|----------|------------|------------|----------|
| 1 | Abalone | ✅ | 0.9198 | 0.9199 | 0.0000 | 0.0000 | 9/9 |
| 2 | Balance | ❌ | 0.9320 | 0.8560 | 8.0000 | 8.0000 | 4/4 |
| 3 | Car_Evaluation | ❌ | 0.8712 | 0.8324 | 13.0000 | 10.0000 | 6/6 |
| 4 | Contraceptive | ❌ | 0.2725 | 0.2576 | 2.0000 | 0.0000 | 9/9 |
| 5 | Hayes_Roth | ✅ | 0.8000 | 0.6667 | 3.0000 | 6.0000 | 4/4 |
| 6 | New_Thyroid | ❌ | 0.9709 | 0.9767 | 4.0102 | 5.2325 | 5/5 |
| 7 | Squash_Stored | ❌ | 1.0000 | 0.4545 | 3.6699 | 3.6699 | 23/23 |
| 8 | Squash_Unstored | ✅ | 1.0000 | 0.8182 | N/A | N/A | N/A |
| 9 | TAE | ❌ | 0.4250 | 0.4839 | 0.0000 | 4.0000 | 5/5 |
| 10 | Thyroid | ❌ | 0.9253 | 0.9236 | 0.0000 | 0.0000 | 21/21 |
| 11 | Wine | ✅ | 0.9930 | 0.9722 | 3.5265 | 6.1538 | 13/13 |
| 12 | Wine_Quality | ✅ | 0.8249 | 0.8250 | 0.0000 | 0.0000 | 11/11 |

**說明**:
- **H1 L1 Norm**: 第一層分類器權重的 L1 範數 (||w1||₁)
- **H2 L1 Norm**: 第二層分類器權重的 L1 範數 (||w2||₁)
- **Features**: 選擇的特徵數量 / 總特徵數量
- L1 Norm 越大表示模型越依賴特徵權重，L1 Norm = 0 表示僅依賴截距項（bias）

---

## 📈 各資料集詳細結果

### 1. Abalone

**Class Roles**: Majority=Class 2, Medium=Class 3, Minority=Class 1
**Test2 Rule Applied**: ✅ Yes

**模型資訊**:

**H1 分類器**:
- Objective Value: 118.000000
- Weight L1 Norm (||w1||₁): 0.000000
- Positive Class Acc LB: 0.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 9/9

**H2 分類器**:
- Objective Value: 418.000000
- Weight L1 Norm (||w2||₁): 0.000000
- Positive Class Acc LB: 0.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 9/9

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.9198 | 0.0000 | 1.0000 | 0.0000 |
| **Testing** | 0.9199 | 0.0000 | 1.0000 | 0.0000 |

---

### 2. Balance

**Class Roles**: Majority=Class 3, Medium=Class 1, Minority=Class 2
**Test2 Rule Applied**: ❌ No

**模型資訊**:

**H1 分類器**:
- Objective Value: 46.081315
- Weight L1 Norm (||w1||₁): 8.000000
- Positive Class Acc LB: 0.9926
- Negative Class Acc LB: 0.9261
- Selected Features: 4/4

**H2 分類器**:
- Objective Value: 43.076087
- Weight L1 Norm (||w2||₁): 8.000000
- Positive Class Acc LB: 0.9351
- Negative Class Acc LB: 0.9888
- Selected Features: 4/4

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.9320 | 0.9261 | 0.9487 | 0.9351 |
| **Testing** | 0.8560 | 0.8793 | 0.8000 | 0.8421 |

---

### 3. Car_Evaluation

**Class Roles**: Majority=Class 1, Medium=Class 2, Minority=Class 3
**Test2 Rule Applied**: ❌ No

**模型資訊**:

**H1 分類器**:
- Objective Value: 77.179092
- Weight L1 Norm (||w1||₁): 13.000000
- Positive Class Acc LB: 0.8269
- Negative Class Acc LB: 0.9940
- Selected Features: 6/6

**H2 分類器**:
- Objective Value: 371.183782
- Weight L1 Norm (||w2||₁): 10.000000
- Positive Class Acc LB: 0.8792
- Negative Class Acc LB: 0.9370
- Selected Features: 6/6

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.8712 | 0.8946 | 0.8149 | 0.8269 |
| **Testing** | 0.8324 | 0.8636 | 0.7692 | 0.6923 |

---

### 4. Contraceptive

**Class Roles**: Majority=Class 1, Medium=Class 3, Minority=Class 2
**Test2 Rule Applied**: ❌ No

**模型資訊**:

**H1 分類器**:
- Objective Value: 936.917246
- Weight L1 Norm (||w1||₁): 2.000000
- Positive Class Acc LB: 0.9615
- Negative Class Acc LB: 0.1213
- Selected Features: 9/9

**H2 分類器**:
- Objective Value: 817.000000
- Weight L1 Norm (||w2||₁): 0.000000
- Positive Class Acc LB: 0.0000
- Negative Class Acc LB: 1.0000
- Selected Features: 9/9

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.2725 | 0.1213 | 0.9774 | 0.0000 |
| **Testing** | 0.2576 | 0.1032 | 0.9403 | 0.0000 |

---

### 5. Hayes_Roth

**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

**模型資訊**:

**H1 分類器**:
- Objective Value: 20.083333
- Weight L1 Norm (||w1||₁): 3.000000
- Positive Class Acc LB: 0.9167
- Negative Class Acc LB: 0.0000
- Selected Features: 4/4

**H2 分類器**:
- Objective Value: 45.349999
- Weight L1 Norm (||w2||₁): 6.000000
- Positive Class Acc LB: 0.0000
- Negative Class Acc LB: 0.6500
- Selected Features: 4/4

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.8000 | 0.6500 | 0.8780 | 0.9167 |
| **Testing** | 0.6667 | 0.3636 | 0.9000 | 0.8333 |

---

### 6. New_Thyroid

**Class Roles**: Majority=Class 1, Medium=Class 2, Minority=Class 3
**Test2 Rule Applied**: ❌ No

**模型資訊**:

**H1 分類器**:
- Objective Value: 3.010160
- Weight L1 Norm (||w1||₁): 4.010160
- Positive Class Acc LB: 1.0000
- Negative Class Acc LB: 1.0000
- Selected Features: 5/5

**H2 分類器**:
- Objective Value: 32.403055
- Weight L1 Norm (||w2||₁): 5.232542
- Positive Class Acc LB: 0.8462
- Negative Class Acc LB: 0.9833
- Selected Features: 5/5

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.9709 | 0.9750 | 0.9286 | 1.0000 |
| **Testing** | 0.9767 | 0.9667 | 1.0000 | 1.0000 |

---

### 7. Squash_Stored

**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

**模型資訊**:

**H1 分類器**:
- Objective Value: 1.669923
- Weight L1 Norm (||w1||₁): 3.669923
- Positive Class Acc LB: 1.0000
- Negative Class Acc LB: 1.0000
- Selected Features: 23/23

**H2 分類器**:
- Objective Value: 1.669923
- Weight L1 Norm (||w2||₁): 3.669923
- Positive Class Acc LB: 1.0000
- Negative Class Acc LB: 1.0000
- Selected Features: 23/23

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Testing** | 0.4545 | 0.0000 | 0.5000 | 0.6000 |

---

### 8. Squash_Unstored

**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

**模型資訊**:

**H1 分類器**:

**H2 分類器**:

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Testing** | 0.8182 | 1.0000 | 0.8000 | 0.0000 |

---

### 9. TAE

**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

**模型資訊**:

**H1 分類器**:
- Objective Value: 77.000000
- Weight L1 Norm (||w1||₁): 0.000000
- Positive Class Acc LB: 1.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 5/5

**H2 分類器**:
- Objective Value: 75.689410
- Weight L1 Norm (||w2||₁): 4.000000
- Positive Class Acc LB: 0.4878
- Negative Class Acc LB: 0.8228
- Selected Features: 5/5

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.4250 | 0.0000 | 0.7750 | 0.4878 |
| **Testing** | 0.4839 | 0.0000 | 0.8000 | 0.6364 |

---

### 10. Thyroid

**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

**模型資訊**:

**H1 分類器**:
- Objective Value: 27.000000
- Weight L1 Norm (||w1||₁): 0.000000
- Positive Class Acc LB: 1.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 21/21

**H2 分類器**:
- Objective Value: 85.000000
- Weight L1 Norm (||w2||₁): 0.000000
- Positive Class Acc LB: 1.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 21/21

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.9253 | 0.0000 | 0.0000 | 1.0000 |
| **Testing** | 0.9236 | 0.0000 | 0.0000 | 1.0000 |

---

### 11. Wine

**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

**模型資訊**:

**H1 分類器**:
- Objective Value: 3.526532
- Weight L1 Norm (||w1||₁): 3.526532
- Positive Class Acc LB: 1.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 13/13

**H2 分類器**:
- Objective Value: 5.153758
- Weight L1 Norm (||w2||₁): 6.153758
- Positive Class Acc LB: 0.0000
- Negative Class Acc LB: 1.0000
- Selected Features: 13/13

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.9930 | 1.0000 | 1.0000 | 0.9737 |
| **Testing** | 0.9722 | 1.0000 | 1.0000 | 0.9000 |

---

### 12. Wine_Quality

**Class Roles**: Majority=Class 2, Medium=Class 3, Minority=Class 1
**Test2 Rule Applied**: ✅ Yes

**模型資訊**:

**H1 分類器**:
- Objective Value: 100.000000
- Weight L1 Norm (||w1||₁): 0.000000
- Positive Class Acc LB: 0.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 11/11

**H2 分類器**:
- Objective Value: 348.000000
- Weight L1 Norm (||w2||₁): 0.000000
- Positive Class Acc LB: 0.0000
- Negative Class Acc LB: 0.0000
- Selected Features: 11/11

**準確率結果**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.8249 | 0.0000 | 1.0000 | 0.0000 |
| **Testing** | 0.8250 | 0.0000 | 1.0000 | 0.0000 |

---

## 📊 整體統計分析

**平均 H1 L1 Norm**: 3.382420
**H1 L1 Norm 範圍**: 0.000000 - 13.000000
**平均 H2 L1 Norm**: 3.914202
**H2 L1 Norm 範圍**: 0.000000 - 10.000000

**平均訓練準確率**: 0.8279 (12 個資料集)
**平均測試準確率**: 0.7489 (12 個資料集)

**Test2 規則應用**: 5/12 個資料集

**測試準確率分層**:
- 優秀 (≥90%): 4 個 - Abalone, New_Thyroid, Thyroid, Wine
- 良好 (70-90%): 4 個 - Balance, Car_Evaluation, Squash_Unstored, Wine_Quality
- 中等 (50-70%): 1 個 - Hayes_Roth
- 需改進 (<50%): 3 個 - Contraceptive, Squash_Stored, TAE

---

## 📝 技術說明

### 權重資訊

本報告中的 **L1 Norm** 表示分類器權重向量的 L1 範數：
- ||w||₁ = Σ|wᵢ| (所有權重絕對值的總和)
- L1 Norm 反映了模型對特徵的依賴程度
- L1 Norm = 0 表示模型僅使用截距項（bias/b），不依賴任何特徵
- L1 Norm > 0 表示模型使用特徵進行分類，數值越大表示特徵權重越分散或越大

### 分類器結構

每個分類器的決策函數形式為：
```
f(x) = w₁x₁ + w₂x₂ + ... + wₙxₙ + b
```
其中：
- w = [w₁, w₂, ..., wₙ] 是權重向量
- b 是截距項（bias）
- 分類決策：sign(f(x)) = +1 或 -1

**注意**: 由於日誌文件中未存儲完整的權重向量 w 和截距項 b 的具體數值，
本報告僅提供 L1 Norm 作為權重的整體度量。若需要完整的 w 和 b 值，
需要修改代碼以在訓練時輸出這些參數。

---

**報告生成時間**: 2026-02-04 07:36:46