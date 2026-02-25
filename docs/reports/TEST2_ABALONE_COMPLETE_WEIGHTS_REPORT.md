# Test2 策略 Abalone 資料集完整權重和截距報告

**執行時間**: 2026-02-04 11:06:26
**訓練時長**: 40分14秒
**資料集**: Abalone (Primary dataset)

---

## 資料集資訊

- **Class 1 (Young)**: 59 samples (1.8%) - MINORITY
- **Class 2 (Adult)**: 3073 samples (92.0%) - MAJORITY
- **Class 3 (Old)**: 209 samples (6.3%) - MEDIUM
- **特徵數量**: 9
- **總樣本數**: 3341 (Training)
- **不平衡比率**: 52.1:1

---

## Test2 策略配置

### 動態類別角色
- **Majority**: Class 2 (3073 samples)
- **Medium**: Class 3 (209 samples)
- **Minority**: Class 1 (59 samples)

### Test2 Rule 應用狀態
- **Test2 Rule Applied**: Yes (majority = Class 2，不在邊緣)
- **H1 Accuracy Mode**: `negative_only` (僅最大化 Class 1 的準確率)
- **H2 Accuracy Mode**: `positive_only` (僅最大化 Class 3 的準確率)

### 階層式分類器結構
- **H1**: Class {2, 3} (+1) vs Class 1 (-1)
  - 基於 ordinal 規則：Class 1 是最小類別，標記為 -1
  - Training samples: 3341 (Positive: 3282, Negative: 59)

- **H2**: Class 3 (+1) vs Class {1, 2} (-1)
  - 使用所有訓練資料
  - Training samples: 3341 (Positive: 209, Negative: 3132)

---

## 模型參數

| 參數 | 值 |
|------|-----|
| C_hyper | 1.0 |
| epsilon | 0.0001 |
| M | 1000.0 |
| time_limit | 1800 秒 (30分鐘) |
| mip_gap | 0.0001 |
| threads | 0 (自動) |
| enable_selection | True |
| feat_upper_bound | 1000 |
| feat_lower_bound | 1e-07 |

---

## H1 分類器完整權重和截距

### 描述
- **分類任務**: Class {2, 3} (+1) vs Class 1 (-1)
- **訓練時間**: ~10分鐘 (613.96 秒)
- **MIP 狀態**: Optimal solution found (gap: 0.0000%)

### 權重向量 (w)
```
w[0] = 0.0000000000
w[1] = 0.0000000000
w[2] = 0.0000000000
w[3] = 0.0000000000
w[4] = 0.0000000000
w[5] = 0.0000000000
w[6] = 0.0000000000
w[7] = 0.0000000000
w[8] = 0.0000000000
```

### 截距 (b)
```
b = 1.0000000000
```

### 模型統計
- **L1 Norm**: ||w||₁ = 0.0000000000
- **Selected Features**: 9/9 (所有特徵都被選擇，但權重為 0)
- **Objective Value**: 118.000000
- **Positive Class Accuracy LB**: 0.0000
- **Negative Class Accuracy LB**: 0.0000

---

## H2 分類器完整權重和截距

### 描述
- **分類任務**: Class 3 (+1) vs Class {1, 2} (-1)
- **訓練時間**: ~30分鐘 (1800.04 秒，達到時間限制)
- **MIP 狀態**: Time limit reached (gap: 32.59%)

### 權重向量 (w)
```
w[0] = 0.0000000000
w[1] = 0.0000000000
w[2] = 0.0000000000
w[3] = 0.0000000000
w[4] = 0.0000000000
w[5] = 0.0000000000
w[6] = 0.0000000000
w[7] = 0.0000000000
w[8] = 0.0000000000
```

### 截距 (b)
```
b = -1.0000000000
```

### 模型統計
- **L1 Norm**: ||w||₁ = 0.0000000000
- **Selected Features**: 9/9 (所有特徵都被選擇，但權重為 0)
- **Objective Value**: 418.000000
- **Positive Class Accuracy LB**: 0.0000
- **Negative Class Accuracy LB**: 0.0000

---

## 預測結果分析

### Training Set 準確率

| 指標 | 值 |
|------|-----|
| **Total Accuracy** | 91.98% |
| **Class 1 Accuracy** | 0.00% (0/59) |
| **Class 2 Accuracy** | 100.00% (3073/3073) |
| **Class 3 Accuracy** | 0.00% (0/209) |

#### Confusion Matrix (Training)
```
               Pred 1    Pred 2    Pred 3
True Class 1:      0        59         0
True Class 2:      0      3073         0
True Class 3:      0       209         0
```

### Test Set 準確率

| 指標 | 值 |
|------|-----|
| **Total Accuracy** | 91.99% |
| **Class 1 Accuracy** | 0.00% (0/15) |
| **Class 2 Accuracy** | 100.00% (769/769) |
| **Class 3 Accuracy** | 0.00% (0/52) |

#### Confusion Matrix (Test)
```
               Pred 1    Pred 2    Pred 3
True Class 1:      0        15         0
True Class 2:      0       769         0
True Class 3:      0        52         0
```

---

## 結果解釋

### 模型行為
這個模型實際上是一個 **常數預測器**（constant predictor），它將所有樣本都預測為 Class 2（多數類）。

### 決策函數
由於權重向量全為 0：

**H1 決策**:
- f₁(x) = w₁ᵀx + b₁ = 0 + 1.0 = **1.0** (常數)
- 由於 f₁(x) ≥ 0 恆成立，所有樣本都被分配到 Positive class {2, 3}

**H2 決策**:
- f₂(x) = w₂ᵀx + b₂ = 0 + (-1.0) = **-1.0** (常數)
- 由於 f₂(x) < 0 恆成立，所有樣本都被分配到 Negative class {1, 2}

**最終預測**:
- H1 預測 → {2, 3}
- H2 預測 → {1, 2}
- 交集 → **Class 2**

### 原因分析

1. **極端資料不平衡**
   - Class 2 占 92% 的樣本
   - 預測所有樣本為 Class 2 就能達到 92% 準確率

2. **L1 正則化效應**
   - 目標函數包含 L1 norm 項（最小化 ||w||₁）
   - 在不平衡資料集上，模型傾向於選擇最簡單的解（w = 0）

3. **Test2 Rule 的限制**
   - H1 只最大化 Class 1 準確率（negative_only）
   - H2 只最大化 Class 3 準確率（positive_only）
   - 但由於多數類 Class 2 的壓倒性優勢，模型無法有效學習

4. **MIP 求解狀態**
   - H1 達到最優解（gap = 0%）
   - H2 達到時間限制（gap = 32.59%），但解仍然是 w = 0

---

## 建議

### 針對極度不平衡資料集的改進方案

1. **調整 C_hyper 參數**
   - 增加 C_hyper 值，減少 L1 正則化強度
   - 建議嘗試: C_hyper ∈ {10, 100, 1000}

2. **使用類別權重**
   - 為 minority classes 增加權重
   - 補償樣本數量的不平衡

3. **資料採樣策略**
   - Over-sampling minority classes (Class 1, Class 3)
   - Under-sampling majority class (Class 2)
   - SMOTE 等合成採樣方法

4. **修改目標函數**
   - 使用 accuracy_mode = "both" 而非 Test2 rule
   - 或調整 Test2 rule 的實施方式

5. **特徵工程**
   - 檢查特徵的區分能力
   - 可能需要特徵轉換或新特徵創建

---

## 檔案位置

- **測試腳本**: `examples/run_test2_abalone_weights.py`
- **執行日誌**: `results/test2_abalone_weights_20260204_110626.log`
- **本報告**: `TEST2_ABALONE_COMPLETE_WEIGHTS_REPORT.md`
- **修改的模型檔案**: `src/hcesvm/models/hierarchical.py` (已添加 weights 和 intercept 輸出)

---

**報告生成時間**: 2026-02-04 11:46:26
