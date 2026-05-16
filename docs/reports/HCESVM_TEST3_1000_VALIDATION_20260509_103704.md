# HCESVM(test3) Derived 1000-Sample Validation

Generated at: `2026-05-09 11:46:52 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/HCESVM_TEST3_1000_VALIDATION_20260509_103704.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260509_test3_teaching_data_1000/test3_teaching_data_1000_20260509_103704.log`

## skill_method3_4class_1000

- Source dataset: `skill`
- Split rule: reuse the existing stratified split, then downsample to `800/200` independently
- Train counts: `[649, 642, 497, 28] -> [286, 283, 219, 12]`
- Test counts: `[162, 161, 124, 7] -> [71, 71, 55, 3]`
- Per-classifier time limit: `1800s`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.3538 | C1=0.0000 (n=286), C2=1.0000 (n=283), C3=0.0000 (n=219), C4=0.0000 (n=12) |
| Test | 0.3550 | C1=0.0000 (n=71), C2=1.0000 (n=71), C3=0.0000 (n=55), C4=0.0000 (n=3) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4} (-1) | 286/514 | time_limit_with_solution | 1800.45 | 571.998054 | 0.938704 | -1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| H2 | Class {1, 2} (+1) vs Class {3, 4} (-1) | 569/231 | time_limit_with_solution | 1802.01 | 461.998243 | 0.953831 | 1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| H3 | Class {1, 2, 3} (+1) vs Class {4} (-1) | 788/12 | optimal | 585.41 | 23.998731 | 0.000000 | 1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
