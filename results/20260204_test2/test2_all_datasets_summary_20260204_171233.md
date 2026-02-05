# Test2 Strategy - All Datasets Test Summary

**Test Date**: 2026-02-04 13:29:59

**Total Duration**: 3:42:33.993749

---

## Configuration

- **Strategy**: test2 (dynamic)
- **Time Limit**: 1800 seconds (30 minutes) per classifier
- **C_hyper**: 1.0
- **M**: 1000.0
- **MIP Gap**: 1e-4

### Test2 Rule

When majority class = Class 2:
- H1: `accuracy_mode='minority'` (maximize minority class accuracy)
- H2: `accuracy_mode='medium'` (maximize medium class accuracy)

Otherwise (majority ∈ {1, 3}):
- H1: `accuracy_mode='both'`
- H2: `accuracy_mode='both'`

---

## Results Summary

| # | Dataset | Status | Train Acc | Test Acc | Train MAE | Test MAE | Duration | Test2 Rule | Class Roles |
|---|---------|--------|-----------|----------|-----------|----------|----------|------------|-------------|
| 1 | Abalone | ✓ PASS | 0.9198 | 0.9199 | N/A | N/A | 0:40:00 | Yes | Maj:2, Med:3, Min:1 |
| 2 | Car_Evaluation | ✓ PASS | 0.8712 | 0.8324 | N/A | N/A | 1:00:00 | No | Maj:1, Med:2, Min:3 |
| 3 | Wine_Quality | ✓ PASS | 0.8249 | 0.8250 | N/A | N/A | 1:00:00 | Yes | Maj:2, Med:3, Min:1 |
| 4 | Balance | ✓ PASS | 0.9320 | 0.8560 | N/A | N/A | 0:00:40 | No | Maj:3, Med:1, Min:2 |
| 5 | Contraceptive | ✓ PASS | 0.2725 | 0.2576 | N/A | N/A | 1:00:00 | No | Maj:1, Med:3, Min:2 |
| 6 | Hayes_Roth | ✓ PASS | 0.8000 | 0.6667 | N/A | N/A | 0:00:00 | Yes | Maj:2, Med:1, Min:3 |
| 7 | New_Thyroid | ✓ PASS | 0.9709 | 0.9767 | N/A | N/A | 0:00:19 | No | Maj:1, Med:2, Min:3 |
| 8 | Squash_Stored | ✓ PASS | 1.0000 | 0.4545 | N/A | N/A | 0:00:00 | No | Maj:3, Med:2, Min:1 |
| 9 | Squash_Unstored | ✓ PASS | 1.0000 | 0.8182 | N/A | N/A | 0:00:00 | Yes | Maj:2, Med:1, Min:3 |
| 10 | TAE | ✓ PASS | 0.4250 | 0.4839 | N/A | N/A | 0:01:22 | No | Maj:3, Med:2, Min:1 |
| 11 | Thyroid | ✓ PASS | 0.9253 | 0.9236 | N/A | N/A | 0:00:07 | No | Maj:3, Med:2, Min:1 |
| 12 | Wine | ✓ PASS | 0.9930 | 0.9722 | N/A | N/A | 0:00:00 | Yes | Maj:2, Med:1, Min:3 |

---

## Detailed Results

### 1. Abalone

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9198
- **Test Accuracy**: 0.9199
- **Training Duration**: 0:40:00.291088
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 3
  - Minority: Class 1
- **Test2 Rule Applied**: Yes
- **Accuracy Modes**:
  - H1: negative_only
  - H2: positive_only
- **Log File**: `test2_Abalone_20260204_132959.log`

### 2. Car_Evaluation

- **Status**: ✓ PASS
- **Training Accuracy**: 0.8712
- **Test Accuracy**: 0.8324
- **Training Duration**: 1:00:00.188075
- **Class Roles**:
  - Majority: Class 1
  - Medium: Class 2
  - Minority: Class 3
- **Test2 Rule Applied**: No
- **Accuracy Modes**:
  - H1: both
  - H2: both
- **Log File**: `test2_Car_Evaluation_20260204_141000.log`

### 3. Wine_Quality

- **Status**: ✓ PASS
- **Training Accuracy**: 0.8249
- **Test Accuracy**: 0.8250
- **Training Duration**: 1:00:00.213056
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 3
  - Minority: Class 1
- **Test2 Rule Applied**: Yes
- **Accuracy Modes**:
  - H1: negative_only
  - H2: positive_only
- **Log File**: `test2_Wine_Quality_20260204_151001.log`

### 4. Balance

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9320
- **Test Accuracy**: 0.8560
- **Training Duration**: 0:00:40.647705
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 1
  - Minority: Class 2
- **Test2 Rule Applied**: No
- **Accuracy Modes**:
  - H1: both
  - H2: both
- **Log File**: `test2_Balance_20260204_161002.log`

### 5. Contraceptive

- **Status**: ✓ PASS
- **Training Accuracy**: 0.2725
- **Test Accuracy**: 0.2576
- **Training Duration**: 1:00:00.175338
- **Class Roles**:
  - Majority: Class 1
  - Medium: Class 3
  - Minority: Class 2
- **Test2 Rule Applied**: No
- **Accuracy Modes**:
  - H1: both
  - H2: both
- **Log File**: `test2_Contraceptive_20260204_161042.log`

### 6. Hayes_Roth

- **Status**: ✓ PASS
- **Training Accuracy**: 0.8000
- **Test Accuracy**: 0.6667
- **Training Duration**: 0:00:00.514701
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 1
  - Minority: Class 3
- **Test2 Rule Applied**: Yes
- **Accuracy Modes**:
  - H1: positive_only
  - H2: negative_only
- **Log File**: `test2_Hayes_Roth_20260204_171043.log`

### 7. New_Thyroid

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9709
- **Test Accuracy**: 0.9767
- **Training Duration**: 0:00:19.758840
- **Class Roles**:
  - Majority: Class 1
  - Medium: Class 2
  - Minority: Class 3
- **Test2 Rule Applied**: No
- **Accuracy Modes**:
  - H1: both
  - H2: both
- **Log File**: `test2_New_Thyroid_20260204_171043.log`

### 8. Squash_Stored

- **Status**: ✓ PASS
- **Training Accuracy**: 1.0000
- **Test Accuracy**: 0.4545
- **Training Duration**: 0:00:00.085813
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 2
  - Minority: Class 1
- **Test2 Rule Applied**: No
- **Accuracy Modes**:
  - H1: both
  - H2: both
- **Log File**: `test2_Squash_Stored_20260204_171103.log`

### 9. Squash_Unstored

- **Status**: ✓ PASS
- **Training Accuracy**: 1.0000
- **Test Accuracy**: 0.8182
- **Training Duration**: 0:00:00.021708
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 1
  - Minority: Class 3
- **Test2 Rule Applied**: Yes
- **Accuracy Modes**:
  - H1: positive_only
  - H2: negative_only
- **Log File**: `test2_Squash_Unstored_20260204_171103.log`

### 10. TAE

- **Status**: ✓ PASS
- **Training Accuracy**: 0.4250
- **Test Accuracy**: 0.4839
- **Training Duration**: 0:01:22.163584
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 2
  - Minority: Class 1
- **Test2 Rule Applied**: No
- **Accuracy Modes**:
  - H1: both
  - H2: both
- **Log File**: `test2_TAE_20260204_171103.log`

### 11. Thyroid

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9253
- **Test Accuracy**: 0.9236
- **Training Duration**: 0:00:07.403040
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 2
  - Minority: Class 1
- **Test2 Rule Applied**: No
- **Accuracy Modes**:
  - H1: both
  - H2: both
- **Log File**: `test2_Thyroid_20260204_171225.log`

### 12. Wine

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9930
- **Test Accuracy**: 0.9722
- **Training Duration**: 0:00:00.268891
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 1
  - Minority: Class 3
- **Test2 Rule Applied**: Yes
- **Accuracy Modes**:
  - H1: positive_only
  - H2: negative_only
- **Log File**: `test2_Wine_20260204_171233.log`

---

## Statistics

- **Total Datasets**: 12
- **Passed**: 12
- **Failed**: 0
- **Success Rate**: 100.0%

- **Average Training Accuracy**: 0.8279
- **Average Test Accuracy**: 0.7489
- **Datasets where Test2 Rule Applied**: 5/12

---

**Report Generated**: 2026-02-04 17:12:33
