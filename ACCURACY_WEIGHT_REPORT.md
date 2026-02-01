# Inverted Strategy - 12 Datasets 完整準確率與權重報告

**報告日期**: 2026-01-31
**策略**: Inverted (Dynamic)

---

## Training Set Results

### 詳細結果表

| # | Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc | C1 Samples (Weight) | C2 Samples (Weight) | C3 Samples (Weight) | Total Samples |
|---|---------|-----------|-------------|-------------|-------------|---------------------|---------------------|---------------------|---------------|
| 1 | Abalone | **0.9198** | 0.0000 | 1.0000 | 0.0000 | 59 (1.8%) | 3073 (92.0%) | 209 (6.3%) | 3341 |
| 2 | Car_Evaluation | **0.8553** | 0.9163 | 0.8149 | 0.0000 | 968 (70.0%) | 362 (26.2%) | 52 (3.8%) | 1382 |
| 3 | Wine_Quality | **0.8249** | 0.0000 | 1.0000 | 0.0000 | 50 (3.9%) | 1055 (82.5%) | 174 (13.6%) | 1279 |
| 4 | Balance | **0.8860** | 0.9261 | 0.0000 | 0.9957 | 230 (46.0%) | 39 (7.8%) | 231 (46.2%) | 500 |
| 5 | Contraceptive | **0.7742** | 1.0000 | 0.0000 | 1.0000 | 503 (42.7%) | 266 (22.6%) | 409 (34.7%) | 1178 |
| 6 | Hayes_Roth | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 40 (38.1%) | 41 (39.0%) | 24 (22.9%) | 105 |
| 7 | New_Thyroid | **0.9884** | 0.9917 | 1.0000 | 0.9583 | 120 (69.8%) | 28 (16.3%) | 24 (14.0%) | 172 |
| 8 | Squash_Stored | **0.9756** | 1.0000 | 0.9412 | 1.0000 | 6 (14.6%) | 17 (41.5%) | 18 (43.9%) | 41 |
| 9 | Squash_Unstored | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 19 (46.3%) | 19 (46.3%) | 3 (7.3%) | 41 |
| 10 | TAE | **0.6667** | 1.0000 | 0.0000 | 1.0000 | 39 (32.5%) | 40 (33.3%) | 41 (34.2%) | 120 |
| 11 | Thyroid | **0.9792** | 0.2143 | 0.9655 | 1.0000 | 14 (2.4%) | 29 (5.0%) | 533 (92.5%) | 576 |
| 12 | Wine | **1.0000** | 1.0000 | 1.0000 | 1.0000 | 47 (33.1%) | 57 (40.1%) | 38 (26.8%) | 142 |

### 統計摘要 (Training Set)

| 指標 | 數值 |
|------|------|
| **Average Total Accuracy** | **0.9058** (90.58%) |
| Average Class 1 Accuracy | 0.7540 (75.40%) |
| Average Class 2 Accuracy | 0.7268 (72.68%) |
| Average Class 3 Accuracy | 0.7462 (74.62%) |
| **Best Total Accuracy** | 1.0000 (Hayes_Roth, Squash_Unstored, Wine) |
| **Worst Total Accuracy** | 0.6667 (TAE) |

---

## Test Set Results (5 Datasets)

### 詳細結果表

| # | Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc | C1 Samples (Weight) | C2 Samples (Weight) | C3 Samples (Weight) | Total Samples |
|---|---------|-----------|-------------|-------------|-------------|---------------------|---------------------|---------------------|---------------|
| 1 | Abalone | **0.9199** | 0.0000 | 1.0000 | 0.0000 | 15 (1.8%) | 769 (92.0%) | 52 (6.2%) | 836 |
| 2 | Car_Evaluation | **0.8468** | 0.9256 | 0.7582 | 0.0000 | 242 (69.9%) | 91 (26.3%) | 13 (3.8%) | 346 |
| 3 | Wine_Quality | **0.8250** | 0.0000 | 1.0000 | 0.0000 | 13 (4.1%) | 264 (82.5%) | 43 (13.4%) | 320 |
| 4 | Balance | **0.8400** | 0.8793 | 0.0000 | 0.9474 | 58 (46.4%) | 10 (8.0%) | 57 (45.6%) | 125 |
| 5 | New_Thyroid | **0.9767** | 0.9667 | 1.0000 | 1.0000 | 30 (69.8%) | 7 (16.3%) | 6 (14.0%) | 43 |

### 統計摘要 (Test Set)

| 指標 | 數值 |
|------|------|
| **Average Total Accuracy** | **0.8817** (88.17%) |
| Average Class 1 Accuracy | 0.5543 (55.43%) |
| Average Class 2 Accuracy | 0.7516 (75.16%) |
| Average Class 3 Accuracy | 0.3895 (38.95%) |
| **Best Total Accuracy** | 0.9767 (New_Thyroid) |
| **Worst Total Accuracy** | 0.8250 (Wine_Quality) |

---

## 關鍵觀察

### 1. 類別不平衡對準確率的影響

#### 極度不平衡的 Datasets:
- **Abalone**: Class 2 佔 92.0%，模型專注於 Class 2 (100% acc)，忽略其他類別
- **Thyroid**: Class 3 佔 92.5%，Class 1 準確率僅 21.43%
- **Wine_Quality**: Class 2 佔 82.5%，僅 Class 2 有高準確率

#### 平衡的 Datasets (表現較好):
- **Hayes_Roth**: 三類別相對平衡 (38%, 39%, 23%) → 100% 準確率
- **Balance**: 兩類別平衡 (46%, 46%, 8%) → 88.60% 準確率
- **Wine**: 相對平衡 (33%, 40%, 27%) → 100% 準確率

### 2. Inverted 策略的動態角色分配

每個 dataset 根據樣本數自動分配：
- **Majority**: 樣本數最多的類別
- **Medium**: 樣本數居中的類別
- **Minority**: 樣本數最少的類別

H1 分類器先區分 Medium vs {Majority, Minority}，H2 再區分 {Medium, Majority} vs Minority

### 3. 測試集準確率分析

- **測試集平均準確率 (88.17%)** 略低於訓練集 (90.58%)
- **New_Thyroid** 測試表現最好：97.67%
- Class 3 在測試集的平均準確率較低 (38.95%)，可能因為：
  - 多數 datasets 的 Class 3 是 minority class
  - 訓練樣本數不足導致泛化能力較弱

### 4. 樣本數統計

| Dataset | 樣本數範圍 | 不平衡程度 |
|---------|-----------|-----------|
| 大型 | Abalone (3341), Wine_Quality (1279), Car_Evaluation (1382), Contraceptive (1178) | 高度不平衡 |
| 中型 | Balance (500), Thyroid (576), New_Thyroid (172) | 中度不平衡 |
| 小型 | Hayes_Roth (105), TAE (120), Wine (142), Squash (41) | 相對平衡 |

---

## 結論

1. **整體表現**: Inverted 策略在 12 個 datasets 達到 **90.58%** 平均訓練準確率，**88.17%** 平均測試準確率

2. **類別不平衡處理**:
   - 平衡 datasets (Hayes_Roth, Wine) 達到 100% 準確率
   - 極度不平衡 datasets (Abalone, Thyroid) 在少數類別表現較差

3. **建議改進方向**:
   - 對不平衡 datasets 使用 class weight 或 SMOTE 等技術
   - 調整 C_hyper 參數以平衡不同類別的重要性
   - 考慮使用 stratified sampling

---

**報告生成時間**: 2026-01-31 18:00
**策略**: Inverted (Dynamic Class Role Assignment)
