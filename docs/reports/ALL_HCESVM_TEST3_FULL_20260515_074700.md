# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-15 07:49:03 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/ALL_HCESVM_TEST3_FULL_20260515_074700.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260515_test3_no_time_limit_full/test3_all_20260515_074700.log`

## balance

- Source URL: `/home/bs10081/Developer/hcesvm/data/svor/SVOR_balance_split.xlsx`
- Split rule: `Preserve workbook Train/Test sheets`
- Train counts: `[230, 39, 231]`
- Test counts: `[58, 10, 57]`
- Per-classifier time limit: `none`
- Expected classifiers: `2`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.9320 | C1=0.9261 (n=230), C2=0.9487 (n=39), C3=0.9351 (n=231) |
| Test | 0.8560 | C1=0.8793 (n=58), C2=0.8000 (n=10), C3=0.8421 (n=57) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class 1 (+1) vs {Class 2, 3} (-1) | 230/270 | optimal | 58.92 | 47.992297 | 0.000000 | -1.000000 | [2.0, 2.0, -2.0, -2.0] |
| H2 | Class {1, 2} (+1) vs Class 3 (-1) | 269/231 | optimal | 21.65 | 44.992272 | 0.000000 | 1.000000 | [2.0, 2.0, -2.0, -2.0] |

## hayes_roth

- Source URL: `/home/bs10081/Developer/NSVORA/Archive/hayes_roth_split.xlsx`
- Split rule: `Preserve workbook Train/Test sheets`
- Train counts: `[40, 41, 24]`
- Test counts: `[11, 10, 6]`
- Per-classifier time limit: `none`
- Expected classifiers: `2`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.7810 | C1=0.6500 (n=40), C2=0.9512 (n=41), C3=0.7083 (n=24) |
| Test | 0.6296 | C1=0.3636 (n=11), C2=0.9000 (n=10), C3=0.6667 (n=6) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class 1 (+1) vs {Class 2, 3} (-1) | 40/65 | optimal | 0.30 | 45.968365 | 0.000000 | 9.000000 | [0.0, -2.0, -2.0, -2.0] |
| H2 | Class {1, 2} (+1) vs Class 3 (-1) | 81/24 | optimal | 0.43 | 20.949765 | 0.000000 | 7.000000 | [0.0, -1.0, -1.0, -1.0] |

## new_thyroid

- Source URL: `/home/bs10081/Developer/NSVORA/Archive/new-thyroid_split.xlsx`
- Split rule: `Preserve workbook Train/Test sheets`
- Train counts: `[120, 28, 24]`
- Test counts: `[30, 7, 6]`
- Per-classifier time limit: `none`
- Expected classifiers: `2`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.9360 | C1=0.9750 (n=120), C2=0.9286 (n=28), C3=0.7500 (n=24) |
| Test | 0.8837 | C1=0.9667 (n=30), C2=1.0000 (n=7), C3=0.3333 (n=6) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class 1 (+1) vs {Class 2, 3} (-1) | 120/52 | optimal | 41.23 | 34.208076 | 0.000000 | 6.132621 | [0.122717655, -0.9052879533, -2.2947273868, -1.6417266103, -0.2680828402] |
| H2 | Class {1, 2} (+1) vs Class 3 (-1) | 148/24 | optimal | 0.21 | 4.961736 | 0.000000 | -14.699081 | [0.0430575714, 2.9801644896, 0.0, 0.0, -0.9869375907] |
