# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-11 08:08:38 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/BOSTONHOUSING_ORD_HCESVM_TEST3_FULL_20260511_033755.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260511_test3_teaching_data_full/test3_bostonhousing_ord_20260511_033755.log`

## bostonhousing_ord

- Source URL: `https://raw.githubusercontent.com/gagolews/teaching-data/master/ordinal_regression/bostonhousing_ord.csv`
- Split rule: stratified `80/20` train/test split
- Random state: `42`
- Train counts: `[61, 191, 98, 29, 25]`
- Test counts: `[16, 48, 25, 7, 6]`
- Per-classifier time limit: `5400s`
- Expected classifiers: `4`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.8366 | C1=0.8689 (n=61), C2=0.8796 (n=191), C3=0.7245 (n=98), C4=0.8621 (n=29), C5=0.8400 (n=25) |
| Test | 0.7451 | C1=0.8125 (n=16), C2=0.8125 (n=48), C3=0.5600 (n=25), C4=0.8571 (n=7), C5=0.6667 (n=6) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4, 5} (-1) | 61/343 | time_limit_with_solution | 5401.28 | 69.383966 | 0.416284 | -27.621092 | [0.4876769418, 0.0, -0.3346233079, 0.5547388467, 0.0, -3.696286814, 0.1674372565, -2.5700486712, -0.1154743638, 0.0029460196, 1.5729319061, -0.0193929643, 0.8794802848] |
| H2 | Class {1, 2} (+1) vs Class {3, 4, 5} (-1) | 252/152 | time_limit_with_solution | 5401.12 | 122.175924 | 0.695530 | 13.892166 | [0.6788976154, -0.0235187006, 0.0761173103, -0.545876854, 0.0, -7.2609427354, 0.0741643276, 0.8818552273, -0.4600252388, 0.0165734393, 0.8324559631, 0.0051979266, 0.3295429009] |
| H3 | Class {1, 2, 3} (+1) vs Class {4, 5} (-1) | 350/54 | time_limit_with_solution | 5401.06 | 41.448036 | 0.194441 | 45.709362 | [0.0, 0.0010550863, 0.147936248, -0.3728144104, 0.0, -6.8539667241, 0.0760109901, 0.6546156327, 0.2631649559, 0.0061870291, 0.472412772, -0.0486303457, 0.5701841949] |
| H4 | Class {1, 2, 3, 4} (+1) vs Class {5} (-1) | 379/25 | optimal | 38.89 | 19.318257 | 0.000000 | 2.433311 | [-1e-07, -0.0115173964, -0.3905372042, 0.8199266574, 0.0, -4.2080967208, 0.0633892875, 0.7130994581, 0.0382520091, 0.0174836071, 0.6915315818, 0.0185830853, 0.3836788484] |
