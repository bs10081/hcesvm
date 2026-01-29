# Multiple Filter 階層式分類器測試報告

## 概述

本報告記錄了使用 **Multiple Filter** 策略的 Hierarchical CE-SVM 在 Balance 和 Abalone 兩個資料集上的測試結果。

## 實作內容

### 1. 分類策略

**Multiple Filter (從下往上)**：
```
       Input X
          |
    [H1: Class 1 vs {2,3}]    <-- 先分離 Class 1
          |
    [H2: Class 2 vs Class 3]   <-- 再分離 Class 2 和 Class 3
```

**預測邏輯**：
| H1 結果 | H2 結果 | 最終預測 |
|---------|---------|----------|
| f1 >= 0 | - | Class 1 |
| f1 < 0 | f2 >= 0 | Class 2 |
| f1 < 0 | f2 < 0 | Class 3 |

### 2. 修改的檔案

1. **src/hcesvm/models/hierarchical.py**
   - 新增 `strategy` 參數支援 "single_filter" 和 "multiple_filter"
   - 修改 `_prepare_h1_data()` 根據策略決定正/負類
   - 修改 `_prepare_h2_data()` 根據策略決定正/負類，新增 X3 參數
   - 更新 `fit()` 顯示策略資訊
   - 更新 `predict()` 根據策略調整預測邏輯
   - 更新 `get_model_summary()` 包含策略資訊

2. **src/hcesvm/config.py**
   - 更新 `HIERARCHICAL_CONFIG` 記錄兩種策略的配置
   - 設定預設策略為 "multiple_filter"

3. **src/hcesvm/utils/data_loader.py**
   - 修正 `load_multiclass_data()` 排除 Balance dataset 的非數值欄位

4. **examples/run_abalone_hierarchical.py** 和 **examples/run_balance_hierarchical.py**
   - 明確指定 `strategy="multiple_filter"`
   - 更新輸出以顯示策略資訊

---

## 測試結果

### Test 1: Balance Dataset

#### Dataset 資訊
- **檔案**: `/home/bs10081/Developer/NSVORA/Archive/balance_split.xlsx`
- **特徵數**: 4
- **類別分佈**:
  - Class 1: 230 samples (46%)
  - Class 2: 39 samples (7.8%)
  - Class 3: 231 samples (46.2%)
- **總樣本數**: 500
- **不平衡比例**: 5.9:1 (Class 1+3 vs Class 2)

#### 模型參數
- C_hyper: 1.0
- epsilon: 0.0001
- M: 1000.0
- time_limit: 300 seconds per classifier
- mip_gap: 0.0001

#### 訓練結果

**Classifier 1 (H1): Class 1 vs {2,3}**
- Objective: 46.081318
- Selected Features: 4/4
- L1 Norm: 7.999998
- Training Time: ~18 seconds
- Status: Optimal solution found (gap 0.0000%)
- Positive accuracy lb: 0.9261
- Negative accuracy lb: 0.9926

**Classifier 2 (H2): Class 2 vs Class 3**
- Objective: 39.090576
- Selected Features: 4/4
- L1 Norm: 8.000000
- Training Time: ~6 seconds
- Status: Optimal solution found (gap 0.0000%)
- Positive accuracy lb: 0.9744
- Negative accuracy lb: 0.9351

**總訓練時間**: ~24 秒

#### 評估結果

**Training Set Performance**:
- **Total Accuracy**: 93.20%

**Per-Class Accuracy**:
| Class | Accuracy | Samples |
|-------|----------|---------|
| Class 1 | 92.61% | 230 |
| Class 2 | 94.87% | 39 |
| Class 3 | 93.51% | 231 |

**Confusion Matrix**:
```
            Predicted
            1     2     3
Actual 1:  213    15     2
       2:    1    37     1
       3:    1    14   216
```

**分析**:
- ✅ 模型在所有三個類別上都達到超過 92% 的準確率
- ✅ 對於少數類別 Class 2 (39 samples)，準確率達到 94.87%
- ✅ 兩個分類器都達到最優解
- ✅ 特徵選擇：使用所有 4 個特徵

---

### Test 2: Abalone Dataset

#### Dataset 資訊
- **檔案**: `/home/bs10081/Developer/NSVORA/datasets/primary/abalone/abalone_split.xlsx`
- **特徵數**: 9
- **類別分佈**:
  - Class 1 (Young): 59 samples (1.8%)
  - Class 2 (Adult): 3073 samples (91.9%)
  - Class 3 (Old): 209 samples (6.3%)
- **總樣本數**: 3341
- **不平衡比例**: 52.1:1 (極度不平衡)

#### 模型參數
- C_hyper: 1.0
- epsilon: 0.0001
- M: 1000.0
- time_limit: 600 seconds per classifier
- mip_gap: 0.0001

#### 訓練結果

**Classifier 1 (H1): Class 1 vs {2,3}**
- Objective: 117.000000
- Selected Features: 9/9
- L1 Norm: 0.000000
- Training Time: 600 seconds (time limit reached)
- Status: Time limit reached (gap: ~45%)
- Positive accuracy lb: 1.0000
- Negative accuracy lb: 0.0000

**Classifier 2 (H2): Class 2 vs Class 3**
- Objective: 417.000000
- Selected Features: 9/9
- L1 Norm: 0.000000
- Training Time: 600 seconds (time limit reached)
- Status: Time limit reached (gap: 39.81%)
- Positive accuracy lb: 1.0000
- Negative accuracy lb: 0.0000

**總訓練時間**: ~1200 秒 (20 分鐘)

#### 評估結果

**Training Set Performance**:
- **Total Accuracy**: 91.98%

**Per-Class Accuracy**:
| Class | Accuracy | Samples |
|-------|----------|---------|
| Class 1 | 0.00% ⚠️ | 59 |
| Class 2 | 100.00% | 3073 |
| Class 3 | 0.00% ⚠️ | 209 |

**Confusion Matrix**:
```
            Predicted
            1      2     3
Actual 1:   0     59     0
       2:   0   3073     0
       3:   0    209     0
```

**問題分析**:
- ❌ 模型預測所有樣本都是 Class 2（多數類別）
- ❌ 兩個分類器都未在時限內達到最優解
- ❌ L1 norm = 0.0 表示權重向量為零，模型未正確訓練
- ❌ 極度不平衡的資料集 (52.1:1) 導致優化問題變得非常困難

**根本原因**:
1. **資料極度不平衡**: Class 1 只有 59 個樣本 (1.8%)，相對於 Class 2+3 的 3282 個樣本
2. **優化難度高**: H1 分類器需要區分 59 vs 3282 個樣本，這在 CE-SVM 的混合整數規劃問題中非常困難
3. **Time limit 不足**: 600 秒對於這麼大且不平衡的資料集來說不夠，導致求解器未能找到合理的解

---

## 結論

### 成功的方面

1. ✅ **Multiple Filter 策略成功實作**
   - 支援可配置的分類策略（single_filter / multiple_filter）
   - 預測邏輯正確實現
   - 程式碼結構清晰，易於擴展

2. ✅ **Balance Dataset 表現優異**
   - 總準確率 93.20%
   - 所有類別準確率均超過 92%
   - 對少數類別 (Class 2, 7.8%) 也有良好表現
   - 訓練時間短 (~24 秒)

### 待改進的方面

1. ⚠️ **Abalone Dataset 的挑戰**
   - 極度不平衡導致訓練失敗
   - 需要更長的 time limit 或更好的初始解
   - 可能需要調整超參數（例如增大 C_hyper）
   - 考慮使用 class weights 或 resampling 技術

### 建議

#### 短期建議

1. **增加 Abalone 的 time limit**
   - 將 time_limit 從 600 秒增加到 1800-3600 秒
   - 或使用 warm start 提供更好的初始解

2. **調整超參數**
   - 增大 C_hyper 以增加對少數類別的懲罰
   - 調整 epsilon 和 M 參數

3. **資料預處理**
   - 考慮使用 SMOTE 等過採樣技術平衡資料
   - 或使用 under-sampling 減少多數類別樣本

#### 長期建議

1. **優化求解器設定**
   - 調整 Gurobi 的 heuristics 參數
   - 使用更好的 branching 策略
   - 探索 warm start 選項

2. **模型改進**
   - 實作 class-weighted CE-SVM
   - 探索其他不平衡資料處理方法
   - 考慮使用 kernel methods

---

## 檔案清單

### 修改的核心檔案
- `src/hcesvm/models/hierarchical.py`
- `src/hcesvm/config.py`
- `src/hcesvm/utils/data_loader.py`

### 測試腳本
- `examples/run_balance_hierarchical.py`
- `examples/run_abalone_hierarchical.py`

### 報告檔案
- `TEST_REPORT_MULTIPLE_FILTER.md`

---

**測試日期**: 2026-01-24
**測試環境**:
- OS: Ubuntu 24.04.3 LTS
- CPU: AMD Ryzen 9 9900X 12-Core Processor
- Gurobi Version: 13.0.1
- Python: 3.11
