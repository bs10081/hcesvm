# Multiple Filter 階層式分類器測試報告 (v2 - 30分鐘 Time Limit)

## 更新內容

相較於 v1 版本，此次測試將 time_limit 從 600 秒（10 分鐘）增加到 1800 秒（30 分鐘），以應對 Abalone 資料集的極度不平衡問題。

---

## Abalone Dataset 重測結果

### Dataset 資訊
- **檔案**: `/home/bs10081/Developer/NSVORA/datasets/primary/abalone/abalone_split.xlsx`
- **特徵數**: 9
- **類別分佈**:
  - Class 1 (Young): 59 samples (1.8%)
  - Class 2 (Adult): 3073 samples (91.9%)
  - Class 3 (Old): 209 samples (6.3%)
- **總樣本數**: 3341
- **不平衡比例**: 52.1:1 (極度不平衡)

### 模型參數
- C_hyper: 1.0
- epsilon: 0.0001
- M: 1000.0
- **time_limit: 1800 seconds (30 minutes)** ⬆️ 從 600 秒增加
- mip_gap: 0.0001

### 訓練結果

**Classifier 1 (H1): Class 1 vs {2,3}**
- Objective: 117.000000
- Selected Features: 9/9
- L1 Norm: 0.000000
- Training Time: ~1800 seconds
- **Status: ✅ Optimal solution found (gap: 0.0099%)**
- Best Objective: 117.0
- Best Bound: 116.988

**Classifier 2 (H2): Class 2 vs Class 3**
- Objective: 417.000000
- Selected Features: 9/9
- L1 Norm: 0.000000
- Training Time: 1800 seconds (time limit reached)
- **Status: ❌ Time limit reached (gap: 37.43%)**
- Best Objective: 417.0
- Best Bound: 260.933

**總訓練時間**: ~60 分鐘

### 評估結果

**Training Set Performance**:
- **Total Accuracy**: 91.98%

**Per-Class Accuracy**:
| Class | Accuracy | Samples | 狀態 |
|-------|----------|---------|------|
| Class 1 | 0.00% | 59 | ❌ |
| Class 2 | 100.00% | 3073 | ✅ |
| Class 3 | 0.00% | 209 | ❌ |

**Confusion Matrix**:
```
            Predicted
            1      2     3
Actual 1:   0     59     0
       2:   0   3073     0
       3:   0    209     0
```

### 問題分析

**改善部分**:
- ✅ H1 分類器（Class 1 vs {2,3}）成功達到最優解
- ✅ Time limit 增加後，H1 的 gap 從 ~45% 降至 0.0099%

**持續問題**:
- ❌ H2 分類器（Class 2 vs {3}）仍然無法在 30 分鐘內達到最優解
- ❌ Gap 37.43% 表示仍有很大的優化空間
- ❌ 模型仍然預測所有樣本為 Class 2
- ❌ L1 norm = 0.0 表示權重向量接近零

**根本原因**:

H2 分類器面對 **3073 vs 209** 的極度不平衡（14.7:1），這是整個階層式分類器中最困難的子問題：

1. **樣本數量差異巨大**: Class 2 的樣本數是 Class 3 的 14.7 倍
2. **CE-SVM 的目標函數**: 在極度不平衡下，優化器傾向於犧牲少數類別以最大化整體目標
3. **MIP 求解難度**: 混合整數規劃問題在不平衡資料上的求解空間非常大

---

## 對比：Balance Dataset 結果

為了對比，Balance dataset 在相同設定下表現優異：

### 訓練結果
- **Total Accuracy**: 93.20%
- **H1 (Class 1 vs {2,3})**: Optimal (gap 0.0%)
- **H2 (Class 2 vs Class 3)**: Optimal (gap 0.0%)
- **Training Time**: ~24 秒

### Per-Class Accuracy
| Class | Accuracy | Samples |
|-------|----------|---------|
| Class 1 | 92.61% | 230 |
| Class 2 | 94.87% | 39 |
| Class 3 | 93.51% | 231 |

**關鍵差異**:
- Balance 的不平衡比例僅 5.9:1
- H2 子問題（39 vs 231 = 5.9:1）遠比 Abalone 的 H2（3073 vs 209 = 14.7:1）容易求解

---

## 結論與建議

### v2 測試總結

**改善**:
- ✅ 增加 time_limit 後，H1 成功達到最優解
- ✅ Balance dataset 持續表現優異

**未解決問題**:
- ❌ Abalone 的 H2 分類器仍然失敗
- ❌ 30 分鐘的 time_limit 對於極度不平衡的子問題仍然不足

### 建議解決方案

#### 1. 調整超參數（推薦）

增大 `C_hyper` 以增加對少數類別的懲罰：

```python
params['C_hyper'] = 10.0  # 或 100.0
```

這會讓優化器更重視少數類別的正確分類。

#### 2. 資料預處理

使用 SMOTE 或其他過採樣技術平衡資料：

```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_h2_balanced, y_h2_balanced = smote.fit_resample(X_h2, y_h2)
```

#### 3. 修改損失函數

考慮在 CE-SVM 的目標函數中加入 class weights：

```python
# 在 binary_cesvm.py 中修改目標函數
weight_pos = len(y[y == -1]) / len(y[y == 1])
# 調整正類樣本的懲罰權重
```

#### 4. 使用其他策略

考慮使用 **One-vs-Rest** 或 **One-vs-One** 策略，而非階層式：

```
Class 1 vs Rest
Class 2 vs Rest
Class 3 vs Rest
```

#### 5. 增加更多 time limit（次選）

雖然可以繼續增加到 3600 秒（1 小時），但這只是治標不治本。

---

## 檔案更新清單

### v2 修改
- `src/hcesvm/config.py` - time_limit 改為 1800 秒
- `examples/run_abalone_hierarchical.py` - time_limit 更新註解
- `~/.claude/CLAUDE.md` - 新增專案資訊
- `~/.claude/skills/test-hcesvm.json` - 新增測試 skill

### 完整檔案清單
- `src/hcesvm/models/hierarchical.py` - 主要實作
- `src/hcesvm/config.py` - 配置檔案
- `src/hcesvm/utils/data_loader.py` - 資料載入
- `examples/run_balance_hierarchical.py` - Balance 測試
- `examples/run_abalone_hierarchical.py` - Abalone 測試
- `TEST_REPORT_MULTIPLE_FILTER.md` - v1 測試報告
- `TEST_REPORT_MULTIPLE_FILTER_V2.md` - v2 測試報告（本檔案）

---

**測試日期**: 2026-01-24
**版本**: v2.0 (30 minutes time limit)
**測試環境**:
- OS: Ubuntu 24.04.3 LTS
- CPU: AMD Ryzen 9 9900X 12-Core Processor
- Gurobi Version: 13.0.1
- Python: 3.11
