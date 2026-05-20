# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-18 22:42:25 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/CEMENT_STRENGTH_HCESVM_TEST3_FULL_20260517_142734.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260517_test3_no_time_limit_full/test3_cement_strength_20260517_142734.log`

## Run Configuration

- Threads: `0`
- SoftMemLimit: `21.0 GB`
- Per-classifier time limit: `none`
- NodeFileStart: `18.0`
- NodeFileDir: `/home/bs10081/hcesvm-gurobi-nodefiles/cement_strength_20260517_142734`

## cement_strength

- Source URL: `https://raw.githubusercontent.com/gagolews/teaching-data/master/ordinal_regression/cement_strength.csv`
- Split rule: `Stratified train_test_split with test_size=0.2`
- Train counts: `[157, 248, 195, 121, 77]`
- Test counts: `[39, 62, 49, 31, 19]`
- Per-classifier time limit: `none`
- Expected classifiers: `4`
- Final status: `completed`
- Random state: `42`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.6892 | C1=0.8471 (n=157), C2=0.7258 (n=248), C3=0.6103 (n=195), C4=0.5207 (n=121), C5=0.7143 (n=77) |
| Test | 0.6650 | C1=0.7949 (n=39), C2=0.7742 (n=62), C3=0.6122 (n=49), C4=0.4194 (n=31), C5=0.5789 (n=19) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | RAM Used (GB) | Peak RAM (GB) | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4, 5} (-1) | 157/641 | mem_limit_with_solution | 1976.72 | 127.088132 | 0.790270 | 18.871967 | 21.033299 | 1331.508573 | [-0.9137231812, -0.6824138021, -0.8221223045, -0.6907512915, 0.5538036274, -0.4595611594, -0.5040308287, -3.4686642835] |
| H2 | Class {1, 2} (+1) vs Class {3, 4, 5} (-1) | 405/393 | mem_limit_with_solution | 40323.38 | 295.968990 | 0.603941 | 19.290358 | 21.023978 | 520.004122 | [-0.7775879375, -0.5315149947, -0.4803302367, 0.4437920774, -2.9719819567, -0.1466516492, -0.1566078733, -1.465028296] |
| H3 | Class {1, 2, 3} (+1) vs Class {4, 5} (-1) | 600/198 | mem_limit_with_solution | 33024.19 | 330.260365 | 0.661314 | 17.939714 | 21.057315 | 611.482886 | [-0.4162615936, -0.4534352572, -0.3371329538, 0.0360779329, -2.1914686283, -0.2251331248, -0.2330797387, -0.373090449] |
| H4 | Class {1, 2, 3, 4} (+1) vs Class {5} (-1) | 721/77 | mem_limit_with_solution | 40765.34 | 107.486003 | 0.277260 | 19.475332 | 21.003473 | 308.055821 | [-0.5569715348, -0.5730242515, -0.5577314874, 1.1444825313, 2.122792361, -0.0999903943, -0.1462676762, -0.2953850888] |
