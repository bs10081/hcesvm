# HCESVM - Hierarchical Cost-Effective SVM

**階層式成本效益支持向量機**，用於多類別有序分類問題。

## 概述

HCESVM 是一個基於 Gurobi 優化求解器的二元/多類別分類庫，實作了：

- **Binary CE-SVM**: 二元 Cost-Effective SVM 模型
  - L1 正則化（特徵稀疏性）
  - 特徵選擇（透過二元變數，無成本預算約束）
  - 三層準確率指標 (α, β, ρ)
  - 類別平衡優化

- **Hierarchical Classifier**: 層次分類器
  - 將 3 類有序分類分解為 2 個二元分類器的級聯結構
  - Classifier 1: Class 3 vs {Class 1, 2}
  - Classifier 2: Class 2 vs Class 1

## 安裝

### 系統需求

- Python >= 3.10
- Gurobi Optimizer (學術授權或商業授權)
- uv (Python 套件管理工具)

### 使用 uv 安裝

```bash
# Clone repository
git clone <repository-url>
cd hcesvm

# 安裝依賴（uv 會自動建立虛擬環境）
uv sync

# 執行測試
uv run python examples/run_parkinsons.py
```

### 手動安裝依賴

```bash
pip install numpy pandas openpyxl gurobipy
```

## 快速開始

### 二元分類範例

```python
from hcesvm import BinaryCESVM, get_default_params
import numpy as np

# 準備資料（y 標籤為 +1/-1）
X = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
y = np.array([1, 1, -1, -1])

# 建立模型
params = get_default_params()
model = BinaryCESVM(**params)

# 訓練
model.fit(X, y)

# 預測
predictions = model.predict(X)
print(predictions)  # [1, 1, -1, -1]

# 查看解的摘要
summary = model.get_solution_summary()
print(f"Selected features: {summary['n_selected_features']}")
print(f"Objective value: {summary['objective_value']}")
```

### 層次分類器範例

```python
from hcesvm import HierarchicalCESVM
import numpy as np

# 準備三類資料
X1 = np.array([[1, 2], [1.5, 2.5]])  # Class 1
X2 = np.array([[3, 4], [3.5, 4.5]])  # Class 2
X3 = np.array([[5, 6], [5.5, 6.5]])  # Class 3

# 建立層次分類器
hc = HierarchicalCESVM()

# 訓練
hc.fit(X1, X2, X3)

# 預測
X_test = np.array([[1, 2], [3, 4], [5, 6]])
predictions = hc.predict(X_test)
print(predictions)  # [1, 2, 3]
```

## 專案結構

```
hcesvm/
├── src/hcesvm/              # 核心套件
│   ├── models/              # 模型實作
│   │   ├── binary_cesvm.py  # 二元 CE-SVM
│   │   └── hierarchical.py  # 層次分類器
│   ├── utils/               # 工具函數
│   │   ├── data_loader.py   # 資料載入
│   │   └── evaluator.py     # 評估函數
│   └── config.py            # 配置參數
├── examples/                # 範例腳本
│   ├── run_parkinsons.py    # Parkinsons 二元分類
│   └── run_balance_hierarchical.py  # Balance 3-class
├── data/                    # 測試資料
├── docs/                    # 文檔
└── tests/                   # 單元測試
```

## 數學模型

### 二元 CE-SVM 目標函數

```
min  ||w||₁ + C·Σ(α + β + ρ) - l₊ - l₋
```

其中：
- `||w||₁`: L1 範數正則化
- `α, β, ρ`: 三層二元指標變數
- `l₊, l₋`: 正/負類準確率下界

### 主要約束

1. **SVM 分離**: `yᵢ·(w·xᵢ + b) >= 1 - ξᵢ`
2. **Big-M 三層**: `ξ <= M·α`, `ξ <= 1+M·β`, `ξ <= 2+M·ρ`
3. **層次關係**: `α >= β >= ρ`
4. **準確率下界**: 正負類準確率約束
5. **特徵選擇**: `w⁺ + w⁻ <= M·v`（無成本/預算約束）

詳細數學推導請參考 [`docs/CE_SVM_MATHEMATICAL_MODEL.md`](docs/CE_SVM_MATHEMATICAL_MODEL.md)

## 層次分類架構

```
輸入樣本 x
     │
     ▼
┌────────────────────────┐
│  Classifier 1 (H1)     │
│  Class 3 (+1) vs       │
│  {Class 1,2} (-1)      │
└────────────────────────┘
     │
     │ if f₁(x) < 0
     ▼
┌────────────────────────┐
│  Classifier 2 (H2)     │
│  Class 2 (+1) vs       │
│  Class 1 (-1)          │
└────────────────────────┘
     │
     ▼
最終類別: 1, 2, 或 3
```

**預測規則**：
- `f₁(x) = w₁·x + b₁ >= 0` → **Class 3**
- `f₁(x) < 0` 且 `f₂(x) = w₂·x + b₂ >= 0` → **Class 2**
- `f₁(x) < 0` 且 `f₂(x) < 0` → **Class 1**

## API 參考

### BinaryCESVM

**參數**:
- `C_hyper` (float): 鬆弛變數懲罰係數，預設 1.0
- `epsilon` (float): 準確率約束容差，預設 0.0001
- `M` (float): Big-M 常數，預設 1000
- `enable_selection` (bool): 啟用特徵選擇，預設 True
- `time_limit` (int): Gurobi 時間限制（秒），預設 600
- `mip_gap` (float): MIP 最優性差距，預設 1e-4
- `verbose` (bool): 顯示求解器輸出，預設 True

**方法**:
- `fit(X, y)`: 訓練模型
- `predict(X)`: 預測類別標籤
- `decision_function(X)`: 計算決策值
- `get_solution_summary()`: 取得解的摘要

### HierarchicalCESVM

**參數**:
- `cesvm_params` (dict, optional): 傳遞給二元 CE-SVM 的參數

**方法**:
- `fit(X1, X2, X3)`: 訓練層次分類器
- `predict(X)`: 預測類別標籤（1, 2, 或 3）
- `get_model_summary()`: 取得兩個分類器的摘要

## 範例

### 執行 Parkinsons 測試

```bash
cd ~/Developer/hcesvm
uv run python examples/run_parkinsons.py
```

### 執行 Balance 層次分類測試

```bash
uv run python examples/run_balance_hierarchical.py
```

## 與 NSVORA 整合

此 repository 設計為 NSVORA 專案的 git submodule：

```bash
cd ~/Developer/NSVORA
git submodule add ../hcesvm libs/hcesvm
git commit -m "Add hcesvm submodule"
```

## 授權

本專案用於學術研究，基於 Gurobi 學術授權。

## 引用

如果您在研究中使用本專案，請引用：

```
@software{hcesvm2025,
  title={HCESVM: Hierarchical Cost-Effective SVM for Ordinal Classification},
  author={Your Name},
  year={2025},
  url={https://github.com/yourusername/hcesvm}
}
```

## 相關專案

- [NSVORA](https://github.com/dreamaker0224/NSVORA): Non-parallel Support Vector Ordinal Regression with Accuracy constraints

## 參考文獻

- 原始 CE-SVM LINGO 模型：`CEAS_SVM1_SL_Par.lg4`
- Parkinsons 資料集：UCI Machine Learning Repository

## 聯絡方式

如有問題或建議，請透過 GitHub Issues 聯繫。
