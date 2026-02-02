# Test2 策略 - 完整準確率報告

**測試日期**: 2026-02-01
**執行時長**: 2:40:43
**成功率**: 12/12 (100%)

---

## 📊 整體統計

- **平均訓練準確率**: 95.99%
- **平均測試準確率**: 88.20% (5個有測試資料的資料集)
- **Test2 規則應用**: 5/12 資料集

---

## 📈 詳細結果

### 1. Abalone
**Class Roles**: Majority=Class 2, Medium=Class 3, Minority=Class 1
**Test2 Rule Applied**: ✅ Yes

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 91.98% | 0.00% | 100.00% | 0.00% |
| **Testing** | 91.99% | 0.00% | 100.00% | 0.00% |

**說明**: Test2 規則生效（majority=2），H1 最大化 minority class (Class 1)，H2 最大化 medium class (Class 3)。結果顯示模型將所有樣本都預測為 Class 2（majority class），達到約 92% 準確率。

---

### 2. Car_Evaluation
**Class Roles**: Majority=Class 1, Medium=Class 2, Minority=Class 3
**Test2 Rule Applied**: ❌ No

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 87.12% | 89.46% | 81.49% | 82.69% |
| **Testing** | 83.24% | 86.36% | 76.92% | 69.23% |

**說明**: Test2 規則未生效（majority=1），使用 `accuracy_mode='both'`。模型在三個 class 上都有合理的表現，測試準確率約 83%。

---

### 3. Wine_Quality
**Class Roles**: Majority=Class 2, Medium=Class 3, Minority=Class 1
**Test2 Rule Applied**: ✅ Yes

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 82.49% | 0.00% | 100.00% | 0.00% |
| **Testing** | 82.50% | 0.00% | 100.00% | 0.00% |

**說明**: Test2 規則生效（majority=2）。類似 Abalone，模型將所有樣本都預測為 Class 2，達到約 82% 準確率。

---

### 4. Balance
**Class Roles**: Majority=Class 3, Medium=Class 1, Minority=Class 2
**Test2 Rule Applied**: ❌ No

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 93.20% | 92.61% | 94.87% | 93.51% |
| **Testing** | 85.60% | 87.93% | 80.00% | 84.21% |

**說明**: Test2 規則未生效（majority=3）。模型在所有 class 上都有良好表現，訓練準確率 93%，測試準確率 86%。

---

### 5. Contraceptive
**Class Roles**: Majority=Class 1, Medium=Class 3, Minority=Class 2
**Test2 Rule Applied**: ❌ No

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Testing** | N/A | N/A | N/A | N/A |

**說明**: 完美的訓練準確率。無測試資料。

---

### 6. Hayes_Roth
**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Testing** | N/A | N/A | N/A | N/A |

**說明**: Test2 規則生效，完美的訓練準確率。無測試資料。

---

### 7. New_Thyroid
**Class Roles**: Majority=Class 1, Medium=Class 2, Minority=Class 3
**Test2 Rule Applied**: ❌ No

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 97.09% | 97.50% | 92.86% | 100.00% |
| **Testing** | 97.67% | 96.67% | 100.00% | 100.00% |

**說明**: 優秀的表現，訓練準確率 97%，測試準確率 98%。所有 class 都有高準確率。

---

### 8. Squash_Stored
**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Testing** | N/A | N/A | N/A | N/A |

**說明**: 完美的訓練準確率。無測試資料。

---

### 9. Squash_Unstored
**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Testing** | N/A | N/A | N/A | N/A |

**說明**: Test2 規則生效，完美的訓練準確率。無測試資料。

---

### 10. TAE
**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Testing** | N/A | N/A | N/A | N/A |

**說明**: 完美的訓練準確率。無測試資料。

---

### 11. Thyroid
**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Testing** | N/A | N/A | N/A | N/A |

**說明**: 完美的訓練準確率。無測試資料。

---

### 12. Wine
**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 100.00% | 100.00% | 100.00% | 100.00% |
| **Testing** | N/A | N/A | N/A | N/A |

**說明**: Test2 規則生效，完美的訓練準確率。無測試資料。

---

## 📋 完整總結表

| # | Dataset | Test2 | Train Total | Train C1 | Train C2 | Train C3 | Test Total | Test C1 | Test C2 | Test C3 |
|---|---------|-------|-------------|----------|----------|----------|------------|---------|---------|---------|
| 1 | Abalone | ✅ | 91.98% | 0.00% | 100.00% | 0.00% | 91.99% | 0.00% | 100.00% | 0.00% |
| 2 | Car_Evaluation | ❌ | 87.12% | 89.46% | 81.49% | 82.69% | 83.24% | 86.36% | 76.92% | 69.23% |
| 3 | Wine_Quality | ✅ | 82.49% | 0.00% | 100.00% | 0.00% | 82.50% | 0.00% | 100.00% | 0.00% |
| 4 | Balance | ❌ | 93.20% | 92.61% | 94.87% | 93.51% | 85.60% | 87.93% | 80.00% | 84.21% |
| 5 | Contraceptive | ❌ | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| 6 | Hayes_Roth | ✅ | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| 7 | New_Thyroid | ❌ | 97.09% | 97.50% | 92.86% | 100.00% | 97.67% | 96.67% | 100.00% | 100.00% |
| 8 | Squash_Stored | ❌ | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| 9 | Squash_Unstored | ✅ | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| 10 | TAE | ❌ | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| 11 | Thyroid | ❌ | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |
| 12 | Wine | ✅ | 100.00% | 100.00% | 100.00% | 100.00% | N/A | N/A | N/A | N/A |

---

## 🔍 關鍵觀察

### Test2 規則生效的資料集（5個）

當 **majority = Class 2** 時，Test2 規則生效：
- **Abalone**: 91.98% train, 91.99% test - 模型傾向預測 majority class
- **Wine_Quality**: 82.49% train, 82.50% test - 模型傾向預測 majority class
- **Hayes_Roth**: 100% train - 完美分類
- **Squash_Unstored**: 100% train - 完美分類
- **Wine**: 100% train - 完美分類

### Test2 規則未生效的資料集（7個）

當 **majority ∈ {1, 3}** 時，使用 `accuracy_mode='both'`：
- **Car_Evaluation**: 87.12% train, 83.24% test - 平衡的多類別表現
- **Balance**: 93.20% train, 85.60% test - 良好的泛化能力
- **New_Thyroid**: 97.09% train, 97.67% test - 優秀表現
- **其他 4 個小型資料集**: 100% train accuracy

### 潛在問題

在 Test2 規則生效的大型資料集（Abalone, Wine_Quality）中，模型顯示出：
- **Class 1 和 Class 3 的準確率為 0%**
- **Class 2 的準確率為 100%**
- 這表明模型傾向於將所有樣本預測為 majority class（Class 2）
- 雖然 total accuracy 看起來不錯（82-92%），但實際上缺乏 class 之間的區分能力

這可能是 Test2 策略的一個**權衡**：通過專注於特定 class 的準確率，可能犧牲了其他 class 的區分能力。

---

**報告生成時間**: 2026-02-02 13:58:12
