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

# 建立層次分類器（使用 class1_first 策略）
hc = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800
    },
    strategy='class1_first'  # 可選: single_filter, multiple_filter, inverted, test2, test3
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
        'time_limit': 1800
    },
    strategy='test3'  # 樣本加權，適合不平衡資料
)

# 訓練（自動檢測少數類並調整權重）
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
PYTHONPATH=src python examples/run_svor_npsvor_balance.py
PYTHONPATH=src python examples/run_ordinal_multiclass_demo.py
```

直接跑 LINGO workbook：

```bash
PYTHONPATH=src python -m hcesvm.ordinal_cli \
  --model svor \
  --workbook data/svor/SVOR_balance_split.xlsx
```

直接跑一般 CSV：

```bash
PYTHONPATH=src python -m hcesvm.ordinal_cli \
  --model npsvor \
  --train-file train.csv \
  --test-file test.csv \
  --target-column label
```

直接跑一般 Excel：

```bash
PYTHONPATH=src python -m hcesvm.ordinal_cli \
  --model svor \
  --train-file ordinal.xlsx \
  --train-sheet Train \
  --test-sheet Test \
  --target-column target
```

## 分類策略 (Classification Strategies)

HCESVM 支援 **6 種分類策略**，分為固定策略和動態策略兩類：

### 策略總覽

| 策略 | 類型 | H1 分組 | H2 分組 | 目標函數 | 適用場景 |
|------|------|---------|---------|----------|----------|
| **single_filter** | 固定 | Class 3 vs {1,2} | Class 2 vs 1 | 標準 | 原始基準策略 |
| **multiple_filter** | 固定 | Class 1 vs {2,3} | {1,2} vs 3 | 標準 | 標準多層過濾 |
| **class1_first** | 固定 | Class 1 vs {2,3} | Class 2 vs 3 | 標準 | 優先識別 Class 1 |
| **inverted** | 動態 | Medium vs {Maj,Min} | {Med,Maj} vs Min | 標準 | 動態分組適應 |
| **test2** | 動態 | 依 minority 位置 | 依 minority 位置 | 移除準確率項 | 激進不平衡處理 |
| **test3** | 動態 | 依 minority 位置 | 依 minority 位置 | 樣本加權 | 平衡不平衡處理 |

**圖例**:
- **Maj** = 多數類（最大樣本數）
- **Med** = 中位類（中等樣本數）
- **Min** = 少數類（最小樣本數）
- **s_p** = 正類 (+1) 樣本數
- **s_n** = 負類 (-1) 樣本數

---

### 固定策略 (Fixed Strategies)

**特性**:
- 類別分組預先定義
- 不受樣本分布影響
- 所有資料集行為一致
- 更容易理解和除錯

**適用時機**:
- 平衡資料集（各類別樣本數相近）
- 領域知識建議特定分類層次
- 優先考慮可重現性和一致性

**可用策略**:
1. **`single_filter`** - Class 3 優先分離
2. **`multiple_filter`** - Class 1 優先分離
3. **`class1_first`** - Class 1 優先且特定預測流程

---

### 動態策略 (Dynamic Strategies)

**特性**:
- 類別分組適應樣本分布
- 訓練時識別少數類
- 不同資料集可能有不同分組
- 更複雜但可能在不平衡資料上表現更好

**適用時機**:
- 不平衡資料集（樣本數差異顯著）
- 少數類檢測至關重要
- 需要在多樣資料集上達到最佳性能

**可用策略**:
1. **`inverted`** - 中位類分離，使用標準目標
2. **`test2`** - 激進地移除多數類準確率項
3. **`test3`** - 平衡的樣本加權目標函數

---

### 目標函數變化

#### 標準目標
```
min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - l⁺ - l⁻
```
使用於: `single_filter`, `multiple_filter`, `class1_first`, `inverted`

#### Test2 目標（準確率項移除）
```
當多數類為 Class 2 時:
  移除多數類的準確率項

結果: 更激進地優化少數類
```
使用於: `test2`

#### Test3 目標（樣本加權）
```
min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - (1/s⁺)·l⁺ - (1/s⁻)·l⁻
```
**效果**: 少數類通過樣本數倒數自動獲得更高權重。

使用於: `test3`

---

### 策略選擇指南

#### 決策樹

```
資料集有顯著類別不平衡？
├─ 否 → 使用固定策略
│   ├─ 領域建議 Class 1 優先？→ class1_first
│   ├─ 標準方法？→ multiple_filter
│   └─ 基準比較？→ single_filter
│
└─ 是 → 使用動態策略
    ├─ 激進少數類聚焦？→ test2
    ├─ 平衡加權？→ test3
    └─ 標準適應？→ inverted
```

#### 推薦預設值

| 場景 | 推薦策略 | 理由 |
|------|----------|------|
| 平衡資料集 | `multiple_filter` 或 `class1_first` | 一致、可解釋 |
| 中度不平衡 (1:2-1:3) | `test3` | 平衡的樣本加權 |
| 嚴重不平衡 (>1:3) | `test2` | 激進的少數類聚焦 |
| 探索性分析 | `inverted` | 標準目標的自適應 |
| 生產部署 | `class1_first` | 固定、可預測行為 |

---

### 實作範例

#### 固定策略範例
```python
from hcesvm import HierarchicalCESVM

model = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800
    },
    strategy='class1_first'  # 固定策略
)

model.fit(X1_train, X2_train, X3_train)
predictions = model.predict(X_test)
```

#### 動態策略範例
```python
from hcesvm import HierarchicalCESVM

# Test3 策略（樣本加權）
model = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800
    },
    strategy='test3'  # 動態策略，平衡加權
)

model.fit(X1_train, X2_train, X3_train)
predictions = model.predict(X_test)
```

---

### 性能考量

**計算成本**:
所有策略的計算複雜度相似，主要由 MIP 求解時間主導。

**典型求解時間（每個分類器）**: 1-30 分鐘（取決於資料集大小和 `time_limit`）

**記憶體使用**:
- **固定策略**: 最小開銷
- **動態策略**: 額外的樣本計數和類別檢測（~O(n)）

**準確率權衡**:
- **固定策略**: 一致但可能無法適應不平衡
- **動態策略**: 在不平衡資料上可能更好，但較不可預測

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
