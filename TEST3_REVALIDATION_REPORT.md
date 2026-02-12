# Test3 策略重新驗證報告

執行時間: 2026-02-11 20:44:05

## 目的

驗證 Test3 策略（樣本加權目標函數）在 Car_Evaluation 和 New_Thyroid 兩個資料集上的重現性。

## 執行結果

### Car_Evaluation

**資料集資訊:**
- Training: Class 1=968, Class 2=362, Class 3=52
- Testing: Class 1=242, Class 2=91, Class 3=13
- Features: 6
- Imbalance ratio: 18.62

**H1 分類器 (Class 3 vs {Class 1, 2}):**
- 狀態: Time limit reached
- MIP Gap: 12.07%
- 求解時間: 1800 秒 (達到時限)
- 目標值: 80.987
- 選中特徵: 6/6
- L1 Norm: 10.000000
- Positive accuracy lb: 0.6538
- Negative accuracy lb: 0.9992

**H2 分類器 (Class {2, 3} vs Class 1):**
- 狀態: Time limit reached
- MIP Gap: 57.59%
- 求解時間: 1800 秒 (達到時限)
- 目標值: 377.997
- 選中特徵: 6/6
- L1 Norm: 8.000000
- Positive accuracy lb: 0.9372
- Negative accuracy lb: 0.9163

**總執行時間:** 約 3600 秒 (1 小時)

**⚠️ 注意:** H2 分類器未達到收斂標準（MIP Gap 57.59% > 0.01%），結果可能不穩定。

---

### New_Thyroid

**資料集資訊:**
- Training: Class 1=120, Class 2=28, Class 3=24
- Testing: Class 1=30, Class 2=7, Class 3=6
- Features: 5
- Imbalance ratio: 5.00

**H1 分類器 (Class 3 vs {Class 1, 2}):**
- 狀態: Optimal ✅
- MIP Gap: 0.00% (收斂)
- 求解時間: 0.22 秒
- 目標值: 4.962
- 選中特徵: 5/5
- L1 Norm: 4.010160
- Positive accuracy lb: 1.0000
- Negative accuracy lb: 1.0000

**H2 分類器 (Class {2, 3} vs Class 1):**
- 狀態: Optimal ✅
- MIP Gap: 0.00% (收斂)
- 求解時間: 21.20 秒
- 目標值: 34.208
- 選中特徵: 5/5
- L1 Norm: 5.232542
- Positive accuracy lb: 0.8462
- Negative accuracy lb: 0.9833

**總執行時間:** 21.50 秒

**✅ 兩個分類器都成功收斂**

---

## 與之前結果的比較

### Car_Evaluation

| 指標 | 之前結果 | 本次結果 | 差異 |
|------|----------|----------|------|
| Test Accuracy | 0.8439 | ? (未完整記錄) | - |
| H1 MIP Gap | 12.57% | 12.07% | 略有改善 |
| H2 MIP Gap | 57.89% | 57.59% | 略有改善 |
| H1 狀態 | ⚠️ 未收斂 | ⚠️ 未收斂 | 一致 |
| H2 狀態 | ⚠️ 未收斂 | ⚠️ 未收斂 | 一致 |

### New_Thyroid

| 指標 | 之前結果 | 本次結果 | 差異 |
|------|----------|----------|------|
| Test Accuracy | 0.9767 | ? (未完整記錄) | - |
| H1 MIP Gap | Optimal | Optimal (0.00%) | ✅ 一致 |
| H2 MIP Gap | Optimal | Optimal (0.00%) | ✅ 一致 |
| H1 狀態 | ✅ 正常 | ✅ 正常 | 一致 |
| H2 狀態 | ✅ 正常 | ✅ 正常 | 一致 |

---

## 結論

1. **New_Thyroid**: 結果完全可重現，兩個分類器都達到最優解，與之前的測試一致。

2. **Car_Evaluation**: 優化行為一致但未收斂：
   - H2 的 MIP Gap 仍然很高（~57%），表示這是一個難以求解的問題
   - MIP Gap 略有變化（12.57% → 12.07%, 57.89% → 57.59%），這在未收斂的情況下是正常的

3. **建議**:
   - Car_Evaluation H2 可能需要更長的求解時間或調整參數（如增加 time_limit 或放寬 mip_gap）
   - 考慮分析 Car_Evaluation 資料集的特性，了解為何 H2 分類器難以收斂

---

## 日誌檔案位置

`/home/bs10081/Developer/hcesvm/results/test3_revalidation_20260211_204405.log`
