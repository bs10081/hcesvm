# HCESVM(test3) Full Teaching-Data Validation

Generated at: `2026-05-22 02:46:19 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/CEMENT_STRENGTH_HCESVM_TEST3_FULL_20260522_024218.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260522_test3_teaching_data_full/test3_cement_strength_20260522_024218.log`

## Run Configuration

- Threads: `0`
- SoftMemLimit: `56.0 GB`
- Per-classifier time limit: `60`
- MIPGap: `0.0001`
- MIPFocus: `None`
- NodeFileStart: `None`
- NodeFileDir: `None`

## cement_strength

- Source URL: `https://raw.githubusercontent.com/gagolews/teaching-data/master/ordinal_regression/cement_strength.csv`
- Split rule: `Stratified train_test_split with test_size=0.2`
- Train counts: `[157, 248, 195, 121, 77]`
- Test counts: `[39, 62, 49, 31, 19]`
- Per-classifier time limit: `60s`
- Expected classifiers: `4`
- Final status: `completed`
- Random state: `42`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.6792 | C1=0.8471 (n=157), C2=0.7460 (n=248), C3=0.5692 (n=195), C4=0.5207 (n=121), C5=0.6494 (n=77) |
| Test | 0.6450 | C1=0.8462 (n=39), C2=0.7742 (n=62), C3=0.5306 (n=49), C4=0.4194 (n=31), C5=0.4737 (n=19) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | RAM Used (GB) | Peak RAM (GB) | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4, 5} (-1) | 157/641 | time_limit_with_solution | 60.11 | 132.141778 | 0.898657 | 0.044304 | 0.400717 | 1268.019627 | [-1.006423064, -0.8080469509, -0.8569076692, -0.6839808424, -0.0114528021, -0.4067551846, -0.4590707031, -2.9160531748] |
| H2 | Class {1, 2} (+1) vs Class {3, 4, 5} (-1) | 405/393 | time_limit_with_solution | 60.13 | 313.525882 | 0.903321 | 0.072686 | 0.519900 | 90.577840 | [-0.3479187799, -0.2001225191, -0.1641367736, 0.5724167856, -1.1368706418, -0.0929573963, 0.0709751737, -0.9448516055] |
| H3 | Class {1, 2, 3} (+1) vs Class {4, 5} (-1) | 600/198 | time_limit_with_solution | 60.14 | 314.882472 | 0.888736 | 0.035577 | 0.583331 | 399.962413 | [-0.5919764766, -0.5611132545, -0.5830790506, 1.0052973568, 0.7714637251, -0.1445759732, -0.1439543401, -1.086034039] |
| H4 | Class {1, 2, 3, 4} (+1) vs Class {5} (-1) | 721/77 | time_limit_with_solution | 60.12 | 107.786428 | 0.744758 | 0.064083 | 0.391556 | 1113.798048 | [-0.8731947665, -0.9141973588, -0.8757960419, 0.5447176579, 1.4437534734, -0.4057638727, -0.4519026315, -0.2869084822] |
