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
  - 支援 3 類固定/動態策略與 N 類 `test3` 級聯結構
  - 目前有效策略：`single_filter`, `multiple_filter`, `inverted`, `test3`
  - `time_limit` 對 H1, H2, ... 每個 binary classifier 分別套用

- **SVOR / NPSVOR**: 直接以 Gurobi 建立 ordinal regression QP
  - `SVORImplicitQP`: 共用一組 `w` 與 `K-1` 個 thresholds
  - `NPSVORQP`: 每個 rank 一個非平行超平面
  - 支援任意有序類別數 `K`，不限 3 類
  - 支援 LINGO split workbook 與一般 `csv / xlsx / xlsm`

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
source .venv/bin/activate
python examples/run_parkinsons.py
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

# 建立層次分類器（使用 multiple_filter 策略）
hc = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800  # 1800s for each hyperplane
    },
    strategy='multiple_filter'  # 可選: single_filter, multiple_filter, inverted, test3
)

# 訓練
hc.fit(X1, X2, X3)

# 預測
X_test = np.array([[1, 2], [3, 4], [5, 6]])
predictions = hc.predict(X_test)
print(predictions)  # [1, 2, 3]
```

### 不平衡資料範例

```python
# 使用 test3 策略處理不平衡資料
hc_imbalanced = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800  # 1800s for each hyperplane
    },
    strategy='test3'  # 樣本加權，適合不平衡資料
)

# 訓練（test3 會使用 balanced class weighting）
hc_imbalanced.fit(X1_small, X2_large, X3_medium)
```

### SVOR / NPSVOR 多類別範例

```python
from hcesvm import NPSVORQP, SVORImplicitQP

X = [[-4.0], [-3.0], [0.0], [1.0], [4.0], [5.0], [8.0], [9.0]]
y = [10, 10, 20, 20, 30, 30, 40, 40]

svor = SVORImplicitQP(C=1.0, add_order_constraints=True)
svor.fit(X, y)
print(svor.predict(X))

npsvor = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.1, prediction_rule="min_distance")
npsvor.fit(X, y)
print(npsvor.predict(X))
```

### SVOR / NPSVOR CLI

平衡資料集的 LINGO workbook 已經整理在：

- `data/svor/SVOR_balance_split.xlsx`
- `data/npsvor/NPSVOR_balance_split.xlsx`

執行內建 demo：

```bash
source .venv/bin/activate
PYTHONPATH=src python examples/run_svor_npsvor_balance.py
PYTHONPATH=src python examples/run_ordinal_multiclass_demo.py
```

直接跑 LINGO workbook：

```bash
source .venv/bin/activate
PYTHONPATH=src python -m hcesvm.ordinal_cli \
  --model svor \
  --workbook data/svor/SVOR_balance_split.xlsx
```

直接跑一般 CSV：

```bash
source .venv/bin/activate
PYTHONPATH=src python -m hcesvm.ordinal_cli \
  --model npsvor \
  --train-file train.csv \
  --test-file test.csv \
  --target-column label
```

直接跑一般 Excel：

```bash
source .venv/bin/activate
PYTHONPATH=src python -m hcesvm.ordinal_cli \
  --model svor \
  --train-file ordinal.xlsx \
  --train-sheet Train \
  --test-sheet Test \
  --target-column target
```

## 分類策略 (Classification Strategies)

`HierarchicalCESVM(strategy=...)` 目前支援 **4 種策略**：

| 策略 | Scope | H1 分組 | H2 / Hk 分組 | 目標函數 |
|------|-------|---------|--------------|----------|
| **single_filter** | 3-class only | Class 3 vs {1,2} | Class 2 vs 1 | 標準 |
| **multiple_filter** | 3-class only | Class 1 vs {2,3} | {1,2} vs 3 | 標準 |
| **inverted** | 3-class only | Medium vs {Majority, Minority} | {Medium, Majority} vs Minority | 標準 |
| **test3** | N-class | Class 1 vs {2..N} | Hk: {1..k} vs {k+1..N} | `class_weight="balanced"` |

`single_filter`, `multiple_filter`, `inverted` 只支援 3 類資料。`test3`
支援 3 類與 N 類資料，會建立 `N-1` 個 binary classifiers，並使用
`accuracy_mode="both"` 與 `class_weight="balanced"`。

`class1_first` 與 `test2` 是歷史實驗名稱，不是目前
`HierarchicalCESVM` 的有效策略。舊報告或舊範例若仍保留這些名稱，應視為
historical reference，不作為新的建議入口。

### 策略選擇

| 場景 | 建議策略 | 理由 |
|------|----------|------|
| 3-class 原始基準比較 | `single_filter` | 保留原始 Class 3 first cascade |
| 3-class 標準固定分組 | `multiple_filter` | 固定、容易比較 |
| 3-class 依樣本分布調整 | `inverted` | 根據 majority / medium / minority 動態分組 |
| 3-class 或 N-class teaching-data | `test3` | N-class 支援與 balanced weighting |

### 目標函數

標準目標用於 `single_filter`, `multiple_filter`, `inverted`：

```text
min  sum_j(w+_j + w-_j) + C * sum_i(alpha_i + beta_i + rho_i) - l+ - l-
```

`test3` 使用樣本數倒數加權：

```text
min  sum_j(w+_j + w-_j) + C * sum_i(alpha_i + beta_i + rho_i)
     - (1 / s+) * l+ - (1 / s-) * l-
```

其中 `s+` 與 `s-` 是該 binary classifier 的正類與負類樣本數。

### 實作範例

3-class 固定策略：

```python
from hcesvm import HierarchicalCESVM

model = HierarchicalCESVM(
    cesvm_params={
        "C_hyper": 1.0,
        "M": 1000.0,
        "time_limit": 1800,
    },
    strategy="multiple_filter",
)

model.fit(X1_train, X2_train, X3_train)
predictions = model.predict(X_test)
```

N-class `test3`：

```python
from hcesvm import HierarchicalCESVM

model = HierarchicalCESVM(
    cesvm_params={
        "C_hyper": 1.0,
        "M": 1000.0,
        "time_limit": 1800,
    },
    strategy="test3",
    n_classes=4,
)

model.fit(X1_train, X2_train, X3_train, X4_train)
predictions = model.predict(X_test)
```

### `time_limit` 語意

- `BinaryCESVM.time_limit` 是單一 Gurobi solve 的秒數上限。
- `HierarchicalCESVM` 會對每個 hyperplane 各自建立一個 `BinaryCESVM`。
- `cesvm_params["time_limit"]` 是 **per classifier / per hyperplane** 的限制，不是整體總預算。
- 4-class `test3` 會訓練 3 個 hyperplanes；若 `time_limit=1800`，最壞情況約為
  `1800s * 3`，因為目前是串行訓練。

### Teaching-data Runner CLI

HCESVM teaching-data runners 都使用 **per-classifier** `time_limit`，不會在 runner
端除以 `n_classes - 1`。

```bash
source .venv/bin/activate

# Derived 1000-sample HCESVM(test3)
python examples/run_teaching_data_hcesvm_1000.py --time-limit 1800

# Full teaching-data HCESVM(test3); accepts --time-limit none
python examples/run_teaching_data_hcesvm_full.py --time-limit none

# Full runner Gurobi resource controls
python examples/run_teaching_data_hcesvm_full.py \
  --datasets cement_strength \
  --time-limit none \
  --threads 0 \
  --soft-mem-limit-gb 56 \
  --nodefile-start-gb 23.6 \
  --nodefile-dir auto

# Deadline-aware HCESVM(test3)
python examples/run_teaching_data_hcesvm_deadline.py --dataset skill --time-limit 1800

# Exact-split SVOR / NPSVOR baselines
python -m hcesvm.skill_1000_baselines_runner
```

輸出報告應保留 training/testing per-class accuracy、total accuracy、每個
classifier 的 `weights`、`b`、objective/gap 與 solve status。

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
├── results/                 # 執行結果
│   └── archive/             # 歷史日誌歸檔
├── docs/                    # 文檔
│   ├── strategies/          # 策略文檔
│   ├── reports/             # 測試報告歸檔
│   ├── CE_SVM_MATHEMATICAL_MODEL.md
│   └── DECISION_VARIABLES.md
├── scripts/                 # 維護腳本
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

以下以 `single_filter` 3-class cascade 為例：

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
- `time_limit` (int): Gurobi 時間限制（秒），預設 600；對 `HierarchicalCESVM` 而言是 each hyperplane 的限制
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
  - 其中 `cesvm_params["time_limit"]` 會原樣傳給每個 binary classifier，不會在模型內自動拆分成總預算
- `strategy` (str): `single_filter`, `multiple_filter`, `inverted`, `test3`
- `n_classes` (int, optional): 類別數；N-class 目前僅支援 `test3`

**方法**:
- `fit(X1, X2, ...)`: 依序傳入各 class 的訓練資料並訓練層次分類器
- `predict(X)`: 預測類別標籤
- `get_model_summary()`: 取得各 classifier 的摘要

## 範例

### 執行 Parkinsons 測試

```bash
cd ~/Developer/hcesvm
source .venv/bin/activate && python examples/run_parkinsons.py
```

### 執行 Balance 層次分類測試

```bash
source .venv/bin/activate && python examples/run_balance_hierarchical.py
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
