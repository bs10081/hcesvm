# Test2 策略 Gurobi 程式碼說明

## 概述

Test2 策略是一個針對極度不平衡資料集的階層式分類策略，透過動態調整 Gurobi 目標函數來避免模型過度偏向多數類別。

---

## 核心概念

### 1. 動態 Class Roles 分配

根據訓練資料的樣本數量，動態決定三個角色：

```python
def _determine_class_roles(self, X1, X2, X3) -> Dict:
    """
    根據樣本數量排序，決定：
    - Minority: 樣本數最少的類別
    - Medium: 樣本數中等的類別
    - Majority: 樣本數最多的類別
    """
    counts = [(1, len(X1), X1), (2, len(X2), X2), (3, len(X3), X3)]
    sorted_counts = sorted(counts, key=lambda x: x[1])

    return {
        'minority': sorted_counts[0][0],
        'medium': sorted_counts[1][0],
        'majority': sorted_counts[2][0],
        ...
    }
```

**範例**: Abalone 資料集
- Class 1: 59 samples → **Minority**
- Class 2: 3073 samples → **Majority**
- Class 3: 209 samples → **Medium**

---

## 2. Test2 Rule (核心創新)

### 規則描述

**只在 Majority = Class 2 (非邊緣類別) 時觸發**：

```python
def _determine_accuracy_mode(self, positive_classes, negative_classes) -> str:
    """
    Test2 Rule:
    - 如果 majority 在邊緣 (class 1 或 3): 使用標準 "both" 模式
    - 如果 majority = class 2 (中間): 應用 Test2 Rule
        - majority 在 +1 側 → accuracy_mode = "negative_only" (移除 -l_p)
        - majority 在 -1 側 → accuracy_mode = "positive_only" (移除 -l_n)
    """
    majority = self.class_roles['majority']

    # 檢查是否在邊緣
    if majority in [1, 3]:
        return "both"  # 標準目標函數

    # Majority = 2 (非邊緣)，觸發 Test2 Rule
    if majority in positive_classes:
        return "negative_only"  # 不最大化 +1 類的準確率
    elif majority in negative_classes:
        return "positive_only"  # 不最大化 -1 類的準確率
    else:
        return "both"
```

### 為什麼這樣做？

在極度不平衡的情況下：
- **標準目標函數**: `min ||w||₁ + C·Σ(α+β+ρ) - l_p - l_n`
  - 同時最大化 `l_p` (正類準確率下界) 和 `l_n` (負類準確率下界)
  - 問題：多數類會主導優化，導致模型退化為「全部預測多數類」

- **Test2 Rule**: 移除多數類所在側的準確率項
  - 如果 majority 在 +1 側 → 移除 `-l_p`，目標函數變為: `min ||w||₁ + C·Σ(α+β+ρ) - l_n`
  - 如果 majority 在 -1 側 → 移除 `-l_n`，目標函數變為: `min ||w||₁ + C·Σ(α+β+ρ) - l_p`
  - **效果**: 強制模型關注少數類的準確率，避免退化解

---

## 3. Gurobi 目標函數實現

### 位置: `src/hcesvm/models/binary_cesvm.py:131-143`

```python
def build_model(self, X: np.ndarray, y: np.ndarray) -> gp.Model:
    """建立 Gurobi 優化模型"""
    # ... (省略前面的變數定義)

    # === 目標函數 ===
    obj_expr = (
        gp.quicksum(w_plus[j] + w_minus[j] for j in range(d))  # ||w||₁
        + self.C_hyper * gp.quicksum(alpha[i] + beta[i] + rho[i] for i in range(n))  # 懲罰項
    )

    # 根據 accuracy_mode 動態添加準確率項
    if self.accuracy_mode in ("both", "negative_only"):
        obj_expr -= l_n  # 最大化負類準確率下界
    if self.accuracy_mode in ("both", "positive_only"):
        obj_expr -= l_p  # 最大化正類準確率下界

    model.setObjective(obj_expr, GRB.MINIMIZE)
```

### 三種模式

| accuracy_mode | 目標函數 | 說明 |
|--------------|---------|------|
| `"both"` | `min ||w||₁ + C·Σ(α+β+ρ) - l_p - l_n` | 標準：同時最大化兩類準確率 |
| `"positive_only"` | `min ||w||₁ + C·Σ(α+β+ρ) - l_p` | 只最大化正類 (+1) 準確率 |
| `"negative_only"` | `min ||w||₁ + C·Σ(α+β+ρ) - l_n` | 只最大化負類 (-1) 準確率 |

---

## 4. Test2 策略的階層式分類器配置

### H1 和 H2 的配置邏輯

根據 **minority 位置**決定如何分割：

#### Case 1: Minority = Class 1 (邊緣)

```python
if minority == 1:
    # H1: {2, 3} (+1) vs {1} (-1)
    h1_positive_classes = [2, 3]
    h1_negative_classes = [1]

    # H2: {3} (+1) vs {1, 2} (-1)  [使用全部訓練資料]
    h2_positive_classes = [3]
    h2_negative_classes = [1, 2]
```

**Test2 Rule 應用**:
- 如果 `majority == 2` (在 H1 的 +1 側) → H1 使用 `accuracy_mode="negative_only"`
- 如果 `majority == 2` (在 H2 的 -1 側) → H2 使用 `accuracy_mode="positive_only"`

#### Case 2: Minority = Class 3 (邊緣)

```python
elif minority == 3:
    # H1: {3} (+1) vs {1, 2} (-1)
    h1_positive_classes = [3]
    h1_negative_classes = [1, 2]

    # H2: {2, 3} (+1) vs {1} (-1)  [使用全部訓練資料]
    h2_positive_classes = [2, 3]
    h2_negative_classes = [1]
```

#### Case 3: Minority = Class 2 (中間)

```python
else:  # minority == 2
    # 與 Case 1 相同
    # H1: {2, 3} (+1) vs {1} (-1)
    # H2: {3} (+1) vs {1, 2} (-1)
```

---

## 5. 完整流程範例: Abalone 資料集

### 初始狀態
- Class 1: 59 samples (1.8%) → **Minority**
- Class 2: 3073 samples (92.0%) → **Majority**
- Class 3: 209 samples (6.3%) → **Medium**

### Step 1: 確定 Class Roles
```python
class_roles = {
    'minority': 1,
    'medium': 3,
    'majority': 2
}
```

### Step 2: H1 配置

**分類任務**: `{2, 3} vs {1}`
- Positive class (+1): Class 2, 3
- Negative class (-1): Class 1

**Test2 Rule 檢查**:
- `majority = 2` (非邊緣) ✓
- `majority in positive_classes` → `2 in [2, 3]` ✓
- **結論**: `accuracy_mode = "negative_only"`

**Gurobi 目標函數**:
```
min ||w||₁ + C·Σ(α+β+ρ) - l_n
```
(移除 `-l_p`，不最大化正類準確率，因為 Class 2 是多數類)

### Step 3: H2 配置

**分類任務**: `{3} vs {1, 2}` (使用全部訓練資料)
- Positive class (+1): Class 3
- Negative class (-1): Class 1, 2

**Test2 Rule 檢查**:
- `majority = 2` (非邊緣) ✓
- `majority in negative_classes` → `2 in [1, 2]` ✓
- **結論**: `accuracy_mode = "positive_only"`

**Gurobi 目標函數**:
```
min ||w||₁ + C·Σ(α+β+ρ) - l_p
```
(移除 `-l_n`，不最大化負類準確率，因為 Class 2 是多數類)

---

## 6. Gurobi 約束條件 (完整 MIP 模型)

### 位置: `src/hcesvm/models/binary_cesvm.py:146-200`

```python
# 1. SVM 分離約束
for i in range(n):
    model.addConstr(
        y[i] * (Σ(w_plus[j] - w_minus[j]) * X[i, j] + b) >= 1 - ksi[i],
        name=f"svm_sep_{i}"
    )

# 2. Big-M 約束 (三層指示器)
for i in range(n):
    model.addConstr(ksi[i] <= M * alpha[i])      # α=0 → ξ=0 (完美分類)
    model.addConstr(ksi[i] <= 1 + M * beta[i])   # β=0 → ξ≤1 (準確分類)
    model.addConstr(ksi[i] <= 2 + M * rho[i])    # ρ=0 → ξ≤2 (可容忍錯誤)

# 3. 階層約束
for i in range(n):
    model.addConstr(alpha[i] >= beta[i])
    model.addConstr(beta[i] >= rho[i])

# 4. 準確率下界約束
for i in range(n):
    model.addConstr(ksi[i] >= (1 + epsilon) * beta[i])

# 5. 正類準確率約束
model.addConstr(
    Σ((1 - beta[i]) * (1 + y[i])) >= l_p * Σ(1 + y[i]),
    name="pos_accuracy"
)

# 6. 負類準確率約束
model.addConstr(
    Σ((1 - beta[i]) * (1 - y[i])) >= l_n * Σ(1 - y[i]),
    name="neg_accuracy"
)

# 7. 特徵選擇約束
if enable_selection:
    for j in range(d):
        model.addConstr(w_plus[j] + w_minus[j] <= feat_upper_bound * v[j])
        model.addConstr(w_plus[j] + w_minus[j] >= feat_lower_bound * v[j])
```

---

## 7. 決策變數

| 變數 | 類型 | 說明 |
|------|------|------|
| `w_plus[j]` | 連續 | 權重正部分 (w = w⁺ - w⁻) |
| `w_minus[j]` | 連續 | 權重負部分 |
| `b` | 連續 | 截距項 |
| `ksi[i]` | 連續 | 鬆弛變數 (違反 margin 程度) |
| `alpha[i]` | 二元 | 指示器 1: ξ > 0 (分類錯誤) |
| `beta[i]` | 二元 | 指示器 2: ξ > 1 (嚴重錯誤) |
| `rho[i]` | 二元 | 指示器 3: ξ > 2 (極端錯誤) |
| `l_p` | 連續 [0,1] | 正類準確率下界 |
| `l_n` | 連續 [0,1] | 負類準確率下界 |
| `v[j]` | 二元 | 特徵選擇指示器 (可選) |

---

## 8. 關鍵程式碼位置

### 階層式分類器邏輯
- **檔案**: `src/hcesvm/models/hierarchical.py`
- **Class Roles 決定**: 第 90-123 行
- **Test2 Rule 判斷**: 第 125-156 行
- **H1 資料準備**: 第 158-227 行
- **H2 資料準備**: 第 229-299 行
- **訓練流程**: 第 301-460 行

### 二元 CE-SVM 模型
- **檔案**: `src/hcesvm/models/binary_cesvm.py`
- **參數初始化**: 第 34-81 行
- **Gurobi 模型建立**: 第 90-203 行
- **目標函數設定**: 第 131-143 行
- **約束條件**: 第 146-200 行

---

## 9. 實驗結果摘要

### 12 個資料集測試結果

| Test2 Rule 應用 | 資料集數 | 平均 Train Acc | 平均 Test Acc |
|----------------|---------|---------------|--------------|
| **Yes** (5個) | 5/12 | 85.95% | 73.77% |
| **No** (7個) | 7/12 | 79.89% | 75.47% |
| **全部** | 12/12 | 82.79% | 74.89% |

### Test2 Rule 觸發的 5 個資料集

1. **Abalone**: Maj=2, Train=91.98%, Test=91.99%
2. **Wine_Quality**: Maj=2, Train=82.49%, Test=82.50%
3. **Hayes_Roth**: Maj=2, Train=80.00%, Test=66.67%
4. **Squash_Unstored**: Maj=2, Train=100.00%, Test=81.82%
5. **Wine**: Maj=2, Train=99.30%, Test=97.22%

---

## 10. 限制與改進方向

### 觀察到的問題

1. **Trivial Solution**: Abalone 資料集模型退化為全預測多數類
   - 原因: 極度不平衡 (92% vs 1.8% vs 6.3%)
   - 權重全為 0，僅依靠截距項

2. **MIP Gap**: 大型資料集可能無法在 30 分鐘內收斂
   - Abalone H2: 32.53% gap

### 可能的改進

1. **調整參數**:
   - 增加 `C_hyper` 值，強化準確率權重
   - 調整 `time_limit`，給予更多求解時間
   - 修改 `M` 值，改善數值穩定性

2. **預處理**:
   - 使用 SMOTE 或其他過採樣技術
   - 類別權重調整

3. **模型增強**:
   - 添加 class-wise 權重到目標函數
   - 考慮引入 kernel 方法

---

## 參考資料

- **完整結果**: `results/20260204_test2/`
- **總結報告**: `results/20260204_test2/test2_all_datasets_summary_20260204_171233.md`
- **原始程式碼**:
  - `src/hcesvm/models/hierarchical.py`
  - `src/hcesvm/models/binary_cesvm.py`
