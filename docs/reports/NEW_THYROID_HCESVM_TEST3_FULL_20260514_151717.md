# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-14 15:18:01 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/NEW_THYROID_HCESVM_TEST3_FULL_20260514_151717.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260514_test3_no_time_limit_full/test3_new_thyroid_20260514_151717.log`

## new_thyroid

- Source URL: `/home/bs10081/Developer/NSVORA/Archive/new-thyroid_split.xlsx`
- Split rule: stratified `80/20` train/test split
- Random state: `42`
- Train counts: `[120, 28, 24]`
- Test counts: `[30, 7, 6]`
- Per-classifier time limit: `none`
- Expected classifiers: `2`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.9302 | C1=0.9667 (n=120), C2=0.9643 (n=28), C3=0.7083 (n=24) |
| Test | 0.9535 | C1=1.0000 (n=30), C2=0.8571 (n=7), C3=0.8333 (n=6) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class 1 (+1) vs {Class 2, 3} (-1) | 120/52 | optimal | 44.15 | 36.799396 | 0.000000 | 6.039029 | [0.1431052486, -1.0174328122, -2.3109807424, -2.0269543179, -0.3253200266] |
| H2 | Class {1, 2} (+1) vs Class 3 (-1) | 148/24 | optimal | 0.19 | 4.961736 | 0.000000 | -14.699081 | [0.0430575714, 2.9801644896, 0.0, 0.0, -0.9869375907] |
