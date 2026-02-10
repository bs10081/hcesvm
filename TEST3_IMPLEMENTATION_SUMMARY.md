# Test3 策略實現總結

**實現日期**: 2026-02-10
**實現者**: Claude Opus 4.6 via Happy

---

## 概述

Test3 策略已成功實現，使用樣本數量加權的目標函數來處理不平衡資料集。

## 核心改動

### 1. `src/hcesvm/models/binary_cesvm.py`

#### 新增參數
- `class_weight`: 控制準確率項的加權方式
  - `"none"` (預設): 等權重
  - `"balanced"`: 使用樣本數量倒數加權

#### 目標函數修改

**標準 (class_weight="none")**:
```
min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - l⁺ - l⁻
```

**Test3 (class_weight="balanced")**:
```
min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - (1/s⁺)·l⁺ - (1/s⁻)·l⁻
```

其中 `s⁺` 和 `s⁻` 分別為正類和負類的樣本數量。

### 2. `src/hcesvm/models/hierarchical.py`

#### 新增策略
- `strategy="test3"`: 新增的策略選項

#### 資料準備
- H1 和 H2 的資料準備邏輯與 Test2 相同
- 根據 minority class 位置動態分組

#### 參數配置
Test3 自動設定：
- `class_weight="balanced"`
- `accuracy_mode="both"`

與 Test2 的差異：
- Test2: 動態設定 `accuracy_mode` ("positive_only" 或 "negative_only")
- Test3: 固定使用 "both"，但透過加權處理不平衡

---

## 策略比較

| 策略 | 目標函數處理方式 | 準確率項 | 適用場景 |
|------|----------------|---------|---------|
| **Test2** | 動態移除準確率項 | `-l⁺` 或 `-l⁻` (擇一) | Majority 為 class 2 時 |
| **Test3** | 樣本數量加權 | `-(1/s⁺)·l⁺ - (1/s⁻)·l⁻` | 所有不平衡情況 |

### 權重效果

假設樣本分布：
- Class 1: 20 samples (minority)
- Class 2: 100 samples (majority)
- Class 3: 50 samples (medium)

**H1: Class {2,3} vs Class 1**
- 正類 (+1): 150 samples → weight = 1/150 = 0.0067
- 負類 (-1): 20 samples → weight = 1/20 = 0.05

**效果**: 負類 (minority) 的準確率權重提高 7.5 倍

---

## 驗證結果

### 語法檢查
✅ Import 成功
✅ Test2 策略不受影響
✅ Test3 策略可正常使用
✅ `class_weight` 參數驗證通過

### 功能測試
測試於不平衡資料集 (20/100/50 samples)：
- ✅ H1 正確使用 `class_weight="balanced"`
- ✅ H2 正確使用 `class_weight="balanced"`
- ✅ 準確率下界符合預期 (minority class 準確率優先)
- ✅ 預測功能正常

---

## 使用方式

### Python 程式碼

```python
from hcesvm import HierarchicalCESVM

# 建立 Test3 模型
model = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800,
        'mip_gap': 1e-4
    },
    strategy='test3'
)

# 訓練
model.fit(X1_train, X2_train, X3_train)

# 預測
predictions = model.predict(X_test)
```

### 比較測試

執行比較測試腳本：
```bash
source .venv/bin/activate
python examples/run_test3_comparison.py
```

---

## 檔案清單

| 檔案 | 修改類型 | 說明 |
|------|---------|------|
| `src/hcesvm/models/binary_cesvm.py` | 修改 | 新增 `class_weight` 參數和加權目標函數 |
| `src/hcesvm/models/hierarchical.py` | 修改 | 新增 "test3" 策略支援 |
| `CLAUDE.md` | 更新 | 新增 Test3 策略說明 |
| `examples/run_test3_comparison.py` | 新增 | Test2 vs Test3 比較測試腳本 |
| `TEST3_IMPLEMENTATION_SUMMARY.md` | 新增 | 本實現總結文檔 |

---

## 數學原理

### 為何使用樣本數量倒數加權？

在不平衡資料集中，若使用等權重 `-l⁺ - l⁻`：
- 優化器傾向最大化 majority class 準確率 (因為樣本數多)
- Minority class 準確率被犧牲

使用倒數加權 `-(1/s⁺)·l⁺ - (1/s⁻)·l⁻`：
- Minority class 的每個正確預測對目標函數的貢獻更大
- 迫使優化器平衡兩類準確率
- 自動適應任意不平衡比例

### 與 Test2 的互補性

- **Test2**: 激進策略，完全移除 majority 準確率項
  - 優點: 強制關注 minority
  - 缺點: 可能過度犧牲 majority 準確率

- **Test3**: 平衡策略，調整準確率項權重
  - 優點: 同時考慮兩類，更平滑的權衡
  - 缺點: 在極端不平衡時可能不夠激進

---

## 未來工作

1. **大規模測試**: 在 12 個資料集上進行完整評估
2. **參數調整**: 探索 `C_hyper` 與 `class_weight` 的交互作用
3. **權重變體**: 考慮其他加權方式 (如 sqrt(1/s) 或 log(1/s))
4. **混合策略**: 結合 Test2 和 Test3 的優點

---

## 提交資訊

準備建立 Git commit：
```bash
git add src/hcesvm/models/binary_cesvm.py
git add src/hcesvm/models/hierarchical.py
git add CLAUDE.md
git add examples/run_test3_comparison.py
git add TEST3_IMPLEMENTATION_SUMMARY.md

git commit -m "Implement Test3 strategy with sample-weighted objective function

Add class_weight parameter to BinaryCESVM for balanced accuracy optimization:
- class_weight='none': Equal weight (default)
- class_weight='balanced': Inverse sample count weighting

Add Test3 strategy to HierarchicalCESVM:
- Uses balanced class weighting for both H1 and H2
- Automatically handles imbalanced datasets
- Objective: min ||w||₁ + C·Σ(α+β+ρ) - (1/s⁺)·l⁺ - (1/s⁻)·l⁻

Differences from Test2:
- Test2: Dynamically removes -l⁺ or -l⁻ based on majority position
- Test3: Weights both terms by inverse sample count

Files modified:
- src/hcesvm/models/binary_cesvm.py
- src/hcesvm/models/hierarchical.py
- CLAUDE.md
- examples/run_test3_comparison.py (new)
- TEST3_IMPLEMENTATION_SUMMARY.md (new)

Generated with [Claude Code](https://claude.ai/code)
via [Happy](https://happy.engineering)

Co-Authored-By: Claude <noreply@anthropic.com>
Co-Authored-By: Happy <yesreply@happy.engineering>"
```

---

**實現完成！** ✅
