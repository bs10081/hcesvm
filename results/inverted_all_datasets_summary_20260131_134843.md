# Inverted Strategy - All Datasets Test Summary

**Test Date**: 2026-01-31 10:01:24

**Total Duration**: 3:47:19.384462

---

## Configuration

- **Strategy**: inverted (dynamic)
- **Time Limit**: 1800 seconds (30 minutes) per classifier
- **C_hyper**: 1.0
- **M**: 1000.0
- **MIP Gap**: 1e-4

---

## Results Summary

| # | Dataset | Status | Train Acc | Test Acc | Duration | Class Roles |
|---|---------|--------|-----------|----------|----------|-------------|
| 1 | Abalone | ✓ PASS | 0.9198 | 0.9199 | 0:59:55 | Maj:2, Med:3, Min:1 |
| 2 | Car_Evaluation | ✓ PASS | 0.8553 | 0.8468 | 1:00:00 | Maj:1, Med:2, Min:3 |
| 3 | Wine_Quality | ✓ PASS | 0.8249 | 0.8250 | 1:00:00 | Maj:2, Med:3, Min:1 |
| 4 | Balance | ✓ PASS | 0.8860 | 0.8400 | 0:01:49 | Maj:3, Med:1, Min:2 |
| 5 | Contraceptive | ✓ PASS | 0.7742 | N/A | 0:30:00 | Maj:1, Med:3, Min:2 |
| 6 | Hayes_Roth | ✓ PASS | 1.0000 | N/A | 0:00:00 | Maj:2, Med:1, Min:3 |
| 7 | New_Thyroid | ✗ FAIL | N/A | N/A | N/A | N/A |
| 8 | Squash_Stored | ✗ FAIL | N/A | N/A | N/A | N/A |
| 9 | Squash_Unstored | ✗ FAIL | N/A | N/A | N/A | N/A |
| 10 | TAE | ✓ PASS | 0.6667 | N/A | 0:15:20 | Maj:3, Med:2, Min:1 |
| 11 | Thyroid | ✓ PASS | 0.9792 | N/A | 0:00:09 | Maj:3, Med:2, Min:1 |
| 12 | Wine | ✓ PASS | 1.0000 | N/A | 0:00:00 | Maj:2, Med:1, Min:3 |

---

## Individual Results

### 1. Abalone

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9198
- **Test Accuracy**: 0.9199
- **Training Duration**: 0:59:55.944043
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 3
  - Minority: Class 1
- **Log File**: `inverted_Abalone_20260131_100124.log`

### 2. Car_Evaluation

- **Status**: ✓ PASS
- **Training Accuracy**: 0.8553
- **Test Accuracy**: 0.8468
- **Training Duration**: 1:00:00.212157
- **Class Roles**:
  - Majority: Class 1
  - Medium: Class 2
  - Minority: Class 3
- **Log File**: `inverted_Car_Evaluation_20260131_110120.log`

### 3. Wine_Quality

- **Status**: ✓ PASS
- **Training Accuracy**: 0.8249
- **Test Accuracy**: 0.8250
- **Training Duration**: 1:00:00.227404
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 3
  - Minority: Class 1
- **Log File**: `inverted_Wine_Quality_20260131_120121.log`

### 4. Balance

- **Status**: ✓ PASS
- **Training Accuracy**: 0.8860
- **Test Accuracy**: 0.8400
- **Training Duration**: 0:01:49.456936
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 1
  - Minority: Class 2
- **Log File**: `inverted_Balance_20260131_130122.log`

### 5. Contraceptive

- **Status**: ✓ PASS
- **Training Accuracy**: 0.7742
- **Training Duration**: 0:30:00.329418
- **Class Roles**:
  - Majority: Class 1
  - Medium: Class 3
  - Minority: Class 2
- **Log File**: `inverted_Contraceptive_20260131_130311.log`

### 6. Hayes_Roth

- **Status**: ✓ PASS
- **Training Accuracy**: 1.0000
- **Training Duration**: 0:00:00.054000
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 1
  - Minority: Class 3
- **Log File**: `inverted_Hayes_Roth_20260131_133312.log`

### 7. New_Thyroid

- **Status**: ✗ FAIL
- **Error**: Failed to load training data: could not convert string to float: 'normal'
- **Log File**: `inverted_New_Thyroid_20260131_133312.log`

### 8. Squash_Stored

- **Status**: ✗ FAIL
- **Error**: Failed to load training data: could not convert string to float: 'LINC'
- **Log File**: `inverted_Squash_Stored_20260131_133312.log`

### 9. Squash_Unstored

- **Status**: ✗ FAIL
- **Error**: Failed to load training data: could not convert string to float: 'LINC'
- **Log File**: `inverted_Squash_Unstored_20260131_133312.log`

### 10. TAE

- **Status**: ✓ PASS
- **Training Accuracy**: 0.6667
- **Training Duration**: 0:15:20.838536
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 2
  - Minority: Class 1
- **Log File**: `inverted_TAE_20260131_133312.log`

### 11. Thyroid

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9792
- **Training Duration**: 0:00:09.910507
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 2
  - Minority: Class 1
- **Log File**: `inverted_Thyroid_20260131_134833.log`

### 12. Wine

- **Status**: ✓ PASS
- **Training Accuracy**: 1.0000
- **Training Duration**: 0:00:00.076361
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 1
  - Minority: Class 3
- **Log File**: `inverted_Wine_20260131_134843.log`

---

## Statistics

- **Total Datasets**: 12
- **Passed**: 9
- **Failed**: 3
- **Success Rate**: 75.0%

- **Average Training Accuracy**: 0.8784
- **Average Test Accuracy**: 0.8579

---

**Report Generated**: 2026-01-31 13:48:43
