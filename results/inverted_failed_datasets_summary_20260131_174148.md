# Inverted Strategy - Failed Datasets Retest Summary

**Test Date**: 2026-01-31 17:41:47

**Total Duration**: 0:00:00.499684

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
| 1 | New_Thyroid | ✓ PASS | 0.9884 | 0.9767 | 0:00:00 | Maj:1, Med:2, Min:3 |
| 2 | Squash_Stored | ✓ PASS | 0.9756 | N/A | 0:00:00 | Maj:3, Med:2, Min:1 |
| 3 | Squash_Unstored | ✓ PASS | 1.0000 | N/A | 0:00:00 | Maj:2, Med:1, Min:3 |

---

## Individual Results

### 1. New_Thyroid

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9884
- **Test Accuracy**: 0.9767
- **Training Duration**: 0:00:00.278216
- **Class Roles**:
  - Majority: Class 1
  - Medium: Class 2
  - Minority: Class 3
- **Log File**: `inverted_New_Thyroid_20260131_174147.log`

### 2. Squash_Stored

- **Status**: ✓ PASS
- **Training Accuracy**: 0.9756
- **Training Duration**: 0:00:00.119996
- **Class Roles**:
  - Majority: Class 3
  - Medium: Class 2
  - Minority: Class 1
- **Log File**: `inverted_Squash_Stored_20260131_174148.log`

### 3. Squash_Unstored

- **Status**: ✓ PASS
- **Training Accuracy**: 1.0000
- **Training Duration**: 0:00:00.022047
- **Class Roles**:
  - Majority: Class 2
  - Medium: Class 1
  - Minority: Class 3
- **Log File**: `inverted_Squash_Unstored_20260131_174148.log`

---

## Statistics

- **Total Datasets**: 3
- **Passed**: 3
- **Failed**: 0
- **Success Rate**: 100.0%

- **Average Training Accuracy**: 0.9880
- **Average Test Accuracy**: 0.9767

---

**Report Generated**: 2026-01-31 17:41:48
