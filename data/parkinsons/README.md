# Parkinsons CE-SVM Dataset

## 檔案說明

### 1. Parkinsons_CESVM.xlsx
**來源**: 原始 LINGO CE-SVM 專案  
**格式**: Excel 資料檔  
**內容**:
- **Sheet "train"**: 訓練資料集
  - Row 0-3: 元資料（LINGO 求解結果）
    - Row 0: 權重 (w)
    - Row 1: 權重正部分 (w+) 和特徵成本 (cost)
    - Row 2: 權重負部分 (w-)
    - Row 3: 特徵選擇變數 (v)
  - Row 4: 欄位標題
  - Row 5+: 訓練樣本
    - 22 個特徵（語音特徵）
    - `status` 欄位：類別標籤（+1 = Parkinson's, -1 = Healthy）
    - 其他欄位：LINGO 求解結果（ksi, ai, bi, ri, predict, etc.）

- **Sheet "工作表1"**: 測試資料集（60 個樣本）

**資料集資訊**:
- 總樣本數: 136 (train) + 60 (test) = 196
- 正類 (+1): 104 個樣本（Parkinson's 患者）
- 負類 (-1): 32 個樣本（健康對照組）
- 特徵數: 22 個語音特徵
- 來源: UCI Machine Learning Repository - Parkinsons Dataset

### 2. CEAS_SVM1_SL_Par.lg4
**來源**: 原始 LINGO 優化模型  
**格式**: LINGO 模型檔（RTF 格式）  
**內容**:
- CE-SVM (Cost-Effective SVM) 數學模型定義
- 使用 L1 範數正則化（透過 w+ - w- 分解）
- 包含特徵選擇機制（原始版本有 cost 和 budget 約束）
- 三層準確率指標（ai, bi, ri）

**模型特性**:
```lingo
Sets:
    set1/1..136/: y, ksi, predict, ai, bi, ri;
    set2/1..22/: w, w_plus, w_minus, cost, v;
    set3(set1, set2): x;
Endsets

Objective:
    min ||w||₁ + C*Σ(α+β+ρ) - l_p - l_n

Constraints:
    - SVM 分離: y(i) * (w·x + b) >= 1 - ksi(i)
    - Big-M 三層: ksi <= M*ai, ksi <= 1+M*bi, ksi <= 2+M*ri
    - 特徵選擇: Σcost·v <= budget
    - 準確率下界: l_p, l_n
```

## Python 實作差異

我們的 Python/Gurobi 實作與原始 LINGO 模型的差異：

| 特性 | LINGO 原始模型 | Python 實作 |
|------|---------------|-------------|
| **正則化** | L1 (w+ - w-) | ✅ 相同 |
| **特徵選擇** | 有（v 變數） | ✅ 保留 |
| **成本約束** | Σcost·v <= budget | ❌ 移除 |
| **三層指標** | ai, bi, ri | ✅ 相同（α, β, ρ） |
| **準確率下界** | l_p, l_n | ✅ 相同 |
| **預測規則** | 未明確定義 | 決策值符號（f(x) >= 0） |

**簡化原因**:
- **移除 cost/budget**: 特徵選擇由 L1 正則化自然驅動，無需外部成本限制
- **保留 v 變數**: 仍可透過 `w+ + w- <= M*v` 約束實現特徵稀疏性

## 使用範例

### 載入資料

```python
from hcesvm.utils import load_parkinsons_data

# 載入訓練資料
X, y, n_features = load_parkinsons_data(
    'data/parkinsons/Parkinsons_CESVM.xlsx',
    sheet_name='train',
    skiprows=4
)

print(f"Samples: {len(y)}, Features: {n_features}")
print(f"Positive (+1): {sum(y == 1)}, Negative (-1): {sum(y == -1)}")
```

### 訓練模型

```python
from hcesvm import BinaryCESVM, get_default_params

params = get_default_params()
model = BinaryCESVM(**params)
model.fit(X, y)

# 預測
predictions = model.predict(X)
accuracy = (predictions == y).mean()
print(f"Training Accuracy: {accuracy:.4f}")
```

### 執行測試腳本

```bash
cd ~/Developer/hcesvm
uv run python examples/run_parkinsons.py
```

## 參考資料

- **原始資料集**: [UCI Parkinsons Dataset](https://archive.ics.uci.edu/ml/datasets/parkinsons)
- **LINGO 求解器**: LINGO 16+ (Windows only)
- **Python 實作**: Gurobi 11+ (跨平台)

## 引用

如果使用此資料集，請引用：

```
@misc{parkinsons_uci,
  title={Parkinsons Dataset},
  author={Little, Max A and McSharry, Patrick E and Roberts, Stephen J and Costello, Declan AE and Moroz, Irene M},
  year={2007},
  institution={UCI Machine Learning Repository}
}
```
