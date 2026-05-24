# HCESVM(test3) Derived 1000-Sample Validation

Generated at: `2026-05-09 10:36:43 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/HCESVM_TEST3_1000_VALIDATION_20260509_103342.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260509_test3_teaching_data_1000/test3_teaching_data_1000_20260509_103342.log`

## skill_method3_4class_1000

- Source dataset: `skill`
- Split rule: reuse the existing stratified split, then downsample to `800/200` independently
- Train counts: `[649, 642, 497, 28] -> [286, 283, 219, 12]`
- Test counts: `[162, 161, 124, 7] -> [71, 71, 55, 3]`
- Per-classifier time limit: `180s`
- Final status: `stopped_early`
- Stop reason: `requested stop after 1 classifiers`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train |  | requested stop after 1 classifiers |
| Test |  | requested stop after 1 classifiers |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4} (-1) | 286/514 | time_limit_with_solution | 180.14 | 571.998054 | 0.957058 | -1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
