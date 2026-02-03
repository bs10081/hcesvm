# Test2 策略 - 詳細準確率與權重報告

**生成時間**: 2026-02-03 15:08:27
**資料集數量**: 7

---

## 📊 總覽表

| # | Dataset | Test2 | Train Total | Test Total | H1 L1 Norm | H2 L1 Norm | Features |
|---|---------|-------|-------------|------------|------------|------------|----------|
| 1 | Contraceptive | ❌ | 0.2725 | 0.2576 | 2.0000 | 0.0000 | 9/9 |
| 2 | Hayes_Roth | ✅ | 0.8000 | 0.6667 | 3.0000 | 6.0000 | 4/4 |
| 3 | Squash_Stored | ❌ | 1.0000 | 0.4545 | 0.4944 | 3.6699 | 23/23 |
| 4 | Squash_Unstored | ✅ | 1.0000 | 0.8182 | 0.0641 | 0.5827 | 22/22 |
| 5 | TAE | ❌ | 0.4250 | 0.4839 | 0.0000 | 4.0000 | 5/5 |
| 6 | Thyroid | ❌ | 0.9253 | 0.9236 | 0.0000 | 0.0000 | 21/21 |
| 7 | Wine | ✅ | 0.9930 | 0.9722 | 3.5265 | 6.1538 | 13/13 |

**註**: 
- **H1 L1 Norm**: 第一層分類器的權重 L1 範數
- **H2 L1 Norm**: 第二層分類器的權重 L1 範數
- **Features**: 選擇的特徵數/總特徵數

---

## 📈 各資料集詳細結果

### 1. Contraceptive

**Class Roles**: Majority=Class 1, Medium=Class 3, Minority=Class 2
**Test2 Rule Applied**: ❌ No

**模型權重**:
- H1 L1 Norm: 2.000000
- H2 L1 Norm: 0.000000
- Selected Features: 9/9

**準確率**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.2725 | 0.1213 | 0.9774 | 0.0000 |
| **Testing** | 0.2576 | 0.1032 | 0.9403 | 0.0000 |

---

### 2. Hayes_Roth

**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

**模型權重**:
- H1 L1 Norm: 3.000000
- H2 L1 Norm: 6.000000
- Selected Features: 4/4

**準確率**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.8000 | 0.6500 | 0.8780 | 0.9167 |
| **Testing** | 0.6667 | 0.3636 | 0.9000 | 0.8333 |

---

### 3. Squash_Stored

**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

**模型權重**:
- H1 L1 Norm: 0.494352
- H2 L1 Norm: 3.669923
- Selected Features: 23/23

**準確率**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Testing** | 0.4545 | 0.0000 | 0.5000 | 0.6000 |

---

### 4. Squash_Unstored

**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

**模型權重**:
- H1 L1 Norm: 0.064072
- H2 L1 Norm: 0.582740
- Selected Features: 22/22

**準確率**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| **Testing** | 0.8182 | 1.0000 | 0.8000 | 0.0000 |

---

### 5. TAE

**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

**模型權重**:
- H1 L1 Norm: 0.000000
- H2 L1 Norm: 4.000000
- Selected Features: 5/5

**準確率**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.4250 | 0.0000 | 0.7750 | 0.4878 |
| **Testing** | 0.4839 | 0.0000 | 0.8000 | 0.6364 |

---

### 6. Thyroid

**Class Roles**: Majority=Class 3, Medium=Class 2, Minority=Class 1
**Test2 Rule Applied**: ❌ No

**模型權重**:
- H1 L1 Norm: 0.000000
- H2 L1 Norm: 0.000000
- Selected Features: 21/21

**準確率**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.9253 | 0.0000 | 0.0000 | 1.0000 |
| **Testing** | 0.9236 | 0.0000 | 0.0000 | 1.0000 |

---

### 7. Wine

**Class Roles**: Majority=Class 2, Medium=Class 1, Minority=Class 3
**Test2 Rule Applied**: ✅ Yes

**模型權重**:
- H1 L1 Norm: 3.526532
- H2 L1 Norm: 6.153758
- Selected Features: 13/13

**準確率**:

| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |
|---------|-----------|-------------|-------------|-------------|
| **Training** | 0.9930 | 1.0000 | 1.0000 | 0.9737 |
| **Testing** | 0.9722 | 1.0000 | 1.0000 | 0.9000 |

---

## 📊 統計分析

**平均 H1 L1 Norm**: 1.297851
**平均 H2 L1 Norm**: 2.915203
**平均訓練準確率**: 0.7737
**平均測試準確率**: 0.6538

**Test2 規則應用**: 3/7 資料集

---

**報告生成時間**: 2026-02-03 15:08:27