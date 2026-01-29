# Decision Variables Access in CE-SVM

本文檔說明如何存取和使用 CE-SVM 模型中的所有 decision variables。

## 概述

CE-SVM 模型訓練完成後，所有 decision variables 都會被保存在模型中，可以用於：
- 模型診斷和分析
- Support vector 識別
- 誤分類樣本分析
- 研究和論文撰寫

## 可用的 Decision Variables

### 1. 主要決策變數

| 變數 | 說明 | 存取方式 |
|------|------|----------|
| **w** | 權重向量 | `model.weights` |
| **b** | 截距 | `model.intercept` |
| **w⁺, w⁻** | 權重分解（L1正則化用） | `model.get_weight_decomposition()` |
| **v** | 特徵選擇指示變數 | `model.solution['v']` |

### 2. 樣本層級變數（per-sample）

| 變數 | 說明 | 範圍 | 存取方式 |
|------|------|------|----------|
| **ξ (ksi)** | Slack變數（違規程度） | [0, ∞) | `model.get_slack_variables()` |
| **α (alpha)** | 第一層指示變數 (ξ > 0) | {0, 1} | `model.get_indicator_variables()['alpha']` |
| **β (beta)** | 第二層指示變數 (ξ > 1) | {0, 1} | `model.get_indicator_variables()['beta']` |
| **ρ (rho)** | 第三層指示變數 (ξ > 2) | {0, 1} | `model.get_indicator_variables()['rho']` |

### 3. 準確率下界

| 變數 | 說明 | 存取方式 |
|------|------|----------|
| **l⁺** | 正類準確率下界 | `model.solution['l_p']` |
| **l⁻** | 負類準確率下界 | `model.solution['l_n']` |

## 使用方法

### 基本存取

```python
from hcesvm import BinaryCESVM
import numpy as np

# 訓練模型
model = BinaryCESVM(C_hyper=1.0, time_limit=60)
model.fit(X_train, y_train)

# 1. 獲取基本解摘要
summary = model.get_solution_summary()
print(f"Objective: {summary['objective_value']}")
print(f"Selected features: {summary['n_selected_features']}")
print(f"Support vectors: {summary['n_support_vectors']}")
```

### Slack Variables (ξ)

```python
# 獲取所有樣本的 slack 值
ksi = model.get_slack_variables()

# 分析 slack 分布
print(f"Min slack: {ksi.min()}")
print(f"Max slack: {ksi.max()}")
print(f"Mean slack: {ksi.mean()}")

# 找出 slack 最大的樣本（最難分類）
top_difficult = np.argsort(ksi)[-10:][::-1]
print(f"Most difficult samples: {top_difficult}")
```

### 指示變數 (α, β, ρ)

```python
# 獲取三層指示變數
indicators = model.get_indicator_variables()

alpha = indicators['alpha']  # ξ > 0
beta = indicators['beta']    # ξ > 1
rho = indicators['rho']      # ξ > 2

# 統計各層樣本數
print(f"Samples with α=1: {np.sum(alpha > 0.5)}")
print(f"Samples with β=1: {np.sum(beta > 0.5)}")
print(f"Samples with ρ=1: {np.sum(rho > 0.5)}")

# 驗證階層關係: α ≥ β ≥ ρ
assert np.all(alpha >= beta) and np.all(beta >= rho)
```

### Support Vectors

```python
# 獲取 support vector 遮罩
sv_mask = model.get_support_vectors_mask(threshold=1e-6)

# 找出所有 support vectors
sv_indices = np.where(sv_mask)[0]
print(f"Support vector count: {len(sv_indices)}")
print(f"Support vector indices: {sv_indices}")

# 獲取 support vectors 的 slack 值
sv_slacks = model.get_slack_variables()[sv_mask]
print(f"Support vector slacks: {sv_slacks}")
```

### Margin Errors

```python
# 獲取 margin error 遮罩（ξ > 1）
me_mask = model.get_margin_errors_mask(margin_threshold=1.0)

# 找出所有 margin errors
me_indices = np.where(me_mask)[0]
print(f"Margin error count: {len(me_indices)}")

# 檢查這些樣本是否被正確分類
y_pred = model.predict(X_train[me_indices])
y_true = y_train[me_indices]
misclassified = np.sum(y_pred != y_true)
print(f"Misclassified: {misclassified}/{len(me_indices)}")
```

### 權重分解

```python
# 獲取權重分解
w_decomp = model.get_weight_decomposition()

w_plus = w_decomp['w_plus']
w_minus = w_decomp['w_minus']
w = model.weights

# 驗證: w = w⁺ - w⁻
assert np.allclose(w, w_plus - w_minus)

# L1 norm 計算
l1_norm = np.sum(w_plus + w_minus)
print(f"L1 norm: {l1_norm}")
```

### 直接存取 Solution Dictionary

```python
# 所有變數都儲存在 solution 字典中
sol = model.solution

# 可用的 keys:
# - 'weights', 'w_plus', 'w_minus', 'intercept'
# - 'ksi', 'alpha', 'beta', 'rho'
# - 'v', 'selected_features'
# - 'l_p', 'l_n'
# - 'objective_value', 'n_selected_features'
# - 'n_support_vectors', 'n_margin_errors'

# 直接存取
ksi = sol['ksi']
alpha = sol['alpha']
w_plus = sol['w_plus']
```

## 應用範例

### 1. 識別困難樣本

```python
# 找出最難分類的 10 個樣本
ksi = model.get_slack_variables()
difficult_indices = np.argsort(ksi)[-10:][::-1]

print("Most difficult samples:")
for i in difficult_indices:
    print(f"Sample {i}: slack={ksi[i]:.4f}")
```

### 2. 分析分類品質分布

```python
indicators = model.get_indicator_variables()
ksi = model.get_slack_variables()

# 分類樣本到不同品質等級
perfect = np.sum(ksi < 1e-6)  # ξ = 0
on_margin = np.sum((ksi >= 1e-6) & (ksi <= 1.0))  # 0 < ξ ≤ 1
margin_error = np.sum((ksi > 1.0) & (ksi <= 2.0))  # 1 < ξ ≤ 2
severe_error = np.sum(ksi > 2.0)  # ξ > 2

print(f"Perfect classification: {perfect}")
print(f"On margin: {on_margin}")
print(f"Margin errors: {margin_error}")
print(f"Severe errors: {severe_error}")
```

### 3. 視覺化 Slack 分布

```python
import matplotlib.pyplot as plt

ksi = model.get_slack_variables()
y_train = ...  # 你的訓練標籤

# 按類別分組
ksi_pos = ksi[y_train == 1]
ksi_neg = ksi[y_train == -1]

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.hist(ksi_pos, bins=30, alpha=0.7, label='Positive')
plt.xlabel('Slack (ξ)')
plt.ylabel('Count')
plt.title('Positive Class Slack Distribution')

plt.subplot(1, 2, 2)
plt.hist(ksi_neg, bins=30, alpha=0.7, label='Negative')
plt.xlabel('Slack (ξ)')
plt.ylabel('Count')
plt.title('Negative Class Slack Distribution')

plt.tight_layout()
plt.show()
```

## 三層指示變數的意義

CE-SVM 使用三層指示變數來實現分段懲罰：

| 條件 | α | β | ρ | 意義 |
|------|---|---|---|------|
| ξ = 0 | 0 | 0 | 0 | 完美分類 |
| 0 < ξ ≤ 1 | 1 | 0 | 0 | 在 margin 內 |
| 1 < ξ ≤ 2 | 1 | 1 | 0 | Margin error |
| ξ > 2 | 1 | 1 | 1 | 嚴重誤分類 |

目標函數中的懲罰項：
```
C * Σ(α + β + ρ)
```

這意味著：
- 完美分類：懲罰 = 0
- Margin 內：懲罰 = C
- Margin error：懲罰 = 2C
- 嚴重誤分類：懲罰 = 3C

## 注意事項

1. **模型必須先訓練**: 所有 decision variables 只在 `fit()` 完成後才可用
2. **時間限制解**: 如果求解達到時間限制，變數仍會保存最佳解
3. **Big-M 值**: Slack 值可能達到 M (預設 1000)，表示嚴重違規
4. **數值精度**: 使用 threshold (如 1e-6) 來判斷變數是否為零

## 完整範例

完整的使用範例請參考：
- `examples/decision_variables_example.py` - 詳細的 decision variables 分析範例

## 參考

相關方法文檔：
- `BinaryCESVM.get_solution_summary()` - 獲取解摘要
- `BinaryCESVM.get_slack_variables()` - 獲取 slack 變數
- `BinaryCESVM.get_indicator_variables()` - 獲取指示變數
- `BinaryCESVM.get_support_vectors_mask()` - 獲取 support vector 遮罩
- `BinaryCESVM.get_margin_errors_mask()` - 獲取 margin error 遮罩
- `BinaryCESVM.get_weight_decomposition()` - 獲取權重分解
