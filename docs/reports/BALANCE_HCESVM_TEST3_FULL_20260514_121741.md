# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-14 12:20:22 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/BALANCE_HCESVM_TEST3_FULL_20260514_121741.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260514_test3_no_time_limit_full/test3_balance_20260514_121741.log`

## balance

- Source URL: `/home/bs10081/Developer/hcesvm/data/svor/SVOR_balance_split.xlsx`
- Split rule: stratified `80/20` train/test split
- Random state: `42`
- Train counts: `[230, 39, 231]`
- Test counts: `[58, 10, 57]`
- Per-classifier time limit: `none`
- Expected classifiers: `2`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.9220 | C1=0.9174 (n=230), C2=0.9231 (n=39), C3=0.9264 (n=231) |
| Test | 0.8960 | C1=0.9138 (n=58), C2=0.9000 (n=10), C3=0.8772 (n=57) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class 1 (+1) vs {Class 2, 3} (-1) | 230/270 | optimal | 83.48 | 54.992349 | 0.000000 | -1.000000 | [2.0, 2.0, -2.0, -2.0] |
| H2 | Class {1, 2} (+1) vs Class 3 (-1) | 269/231 | optimal | 78.24 | 53.992341 | 0.000000 | 1.000000 | [2.0, 2.0, -2.0, -2.0] |
