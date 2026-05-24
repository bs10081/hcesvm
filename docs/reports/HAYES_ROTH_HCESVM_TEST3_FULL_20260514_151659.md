# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-14 15:17:00 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/HAYES_ROTH_HCESVM_TEST3_FULL_20260514_151659.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260514_test3_no_time_limit_full/test3_hayes_roth_20260514_151659.log`

## hayes_roth

- Source URL: `/home/bs10081/Developer/NSVORA/Archive/hayes_roth_split.xlsx`
- Split rule: stratified `80/20` train/test split
- Random state: `42`
- Train counts: `[40, 41, 24]`
- Test counts: `[11, 10, 6]`
- Per-classifier time limit: `none`
- Expected classifiers: `2`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.7524 | C1=0.5500 (n=40), C2=0.9512 (n=41), C3=0.7500 (n=24) |
| Test | 0.7407 | C1=0.7273 (n=11), C2=0.9000 (n=10), C3=0.5000 (n=6) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class 1 (+1) vs {Class 2, 3} (-1) | 40/65 | optimal | 0.70 | 56.970865 | 0.000000 | 9.000000 | [0.0, -2.0, -2.0, -2.0] |
| H2 | Class {1, 2} (+1) vs Class 3 (-1) | 81/24 | optimal | 0.35 | 19.948029 | 0.000000 | 7.000000 | [0.0, -1.0, -1.0, -1.0] |
