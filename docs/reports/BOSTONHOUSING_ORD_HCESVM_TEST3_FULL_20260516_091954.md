# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-17 08:30:48 UTC`
Workbook: `/home/bs10081/Developer/hcesvm-boston-nodefile-run-20260516_085304/docs/reports/BOSTONHOUSING_ORD_HCESVM_TEST3_FULL_20260516_091954.xlsx`
Log: `/home/bs10081/Developer/hcesvm-boston-nodefile-run-20260516_085304/results/archive/20260516_test3_no_time_limit_full/test3_bostonhousing_ord_20260516_091954.log`

## Run Configuration

- Threads: `0`
- SoftMemLimit: `27.0 GB`
- Per-classifier time limit: `none`
- NodeFileStart: `23.6`
- NodeFileDir: `/home/bs10081/hcesvm-gurobi-nodefiles/bostonhousing_ord_20260516_091952`

## bostonhousing_ord

- Source URL: `https://raw.githubusercontent.com/gagolews/teaching-data/master/ordinal_regression/bostonhousing_ord.csv`
- Split rule: `Stratified train_test_split with test_size=0.2`
- Train counts: `[61, 191, 98, 29, 25]`
- Test counts: `[16, 48, 25, 7, 6]`
- Per-classifier time limit: `none`
- Expected classifiers: `4`
- Final status: `completed`
- Random state: `42`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.8317 | C1=0.8197 (n=61), C2=0.9058 (n=191), C3=0.6939 (n=98), C4=0.8621 (n=29), C5=0.8000 (n=25) |
| Test | 0.7549 | C1=0.7500 (n=16), C2=0.8958 (n=48), C3=0.4800 (n=25), C4=0.8571 (n=7), C5=0.6667 (n=6) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | RAM Used (GB) | Peak RAM (GB) | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4, 5} (-1) | 61/343 | mem_limit_with_solution | 35969.49 | 71.096393 | 0.339593 | 24.896352 | 27.004447 | -101.933267 | [0.2580788422, 0.0, -0.5521288861, 0.5417373545, 0.0, 0.3824200607, 0.4228260569, -2.770443828, -0.5491182169, 0.0340239026, 2.0868674558, -0.0274203619, 1.4876162003] |
| H2 | Class {1, 2} (+1) vs Class {3, 4, 5} (-1) | 252/152 | mem_limit_with_solution | 25574.66 | 117.597800 | 0.618762 | 25.411106 | 27.057486 | -82.758072 | [0.0758714135, -0.0743189788, -0.2496714716, -0.2635215662, 0.0, -1.9568910305, 0.120880076, 2.382373453, -1.3301863832, 0.054023154, 2.4449825136, 0.0094151011, 2.6449878846] |
| H3 | Class {1, 2, 3} (+1) vs Class {4, 5} (-1) | 350/54 | optimal | 21876.04 | 41.448036 | 0.000000 | 0.051265 | 4.518083 | 45.709360 | [-1e-07, 0.0010550862, 0.1479362766, -0.3728144855, 0.0, -6.8539668012, 0.0760109928, 0.6546156752, 0.2631649563, 0.0061870294, 0.4724127881, -0.0486303425, 0.5701842415] |
| H4 | Class {1, 2, 3, 4} (+1) vs Class {5} (-1) | 379/25 | optimal | 33.87 | 19.318257 | 0.000000 | 0.017156 | 0.608291 | 2.433311 | [-1e-07, -0.0115173964, -0.3905372042, 0.8199266574, 0.0, -4.2080967208, 0.0633892875, 0.7130994581, 0.0382520091, 0.0174836071, 0.6915315818, 0.0185830853, 0.3836788484] |
