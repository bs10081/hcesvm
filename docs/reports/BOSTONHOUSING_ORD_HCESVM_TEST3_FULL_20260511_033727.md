# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-11 03:37:32 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/BOSTONHOUSING_ORD_HCESVM_TEST3_FULL_20260511_033727.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260511_test3_teaching_data_full/test3_bostonhousing_ord_20260511_033727.log`

## bostonhousing_ord

- Source URL: `https://raw.githubusercontent.com/gagolews/teaching-data/master/ordinal_regression/bostonhousing_ord.csv`
- Split rule: stratified `80/20` train/test split
- Random state: `42`
- Train counts: `[61, 191, 98, 29, 25]`
- Test counts: `[16, 48, 25, 7, 6]`
- Per-classifier time limit: `5s`
- Expected classifiers: `4`
- Final status: `stopped_early`
- Stop reason: `requested stop after 1 classifiers`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train |  | requested stop after 1 classifiers |
| Test |  | requested stop after 1 classifiers |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4, 5} (-1) | 61/343 | time_limit_with_solution | 5.05 | 74.305898 | 0.894752 | -43.324017 | [0.0696571149, 0.0, -0.3138466246, 0.0725231427, 0.0, 0.8968813343, 0.1126503674, -1.8527550624, -0.1801508413, 0.0166450251, 0.9709545946, -0.0165716233, 0.8202850276] |
