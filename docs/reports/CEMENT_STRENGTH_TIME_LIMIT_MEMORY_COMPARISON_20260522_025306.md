# Cement Strength Time-Limit / Memory-Limit Comparison

Generated at: `2026-05-22 02:53:06 UTC`
Workbook: `docs/reports/CEMENT_STRENGTH_TIME_LIMIT_MEMORY_COMPARISON_20260522_025306.xlsx`
Generation log: `results/archive/20260522_cement_strength_comparison/cement_strength_comparison_20260522_025306.log`

## Longest Artifact

- Longest version: `Software Memory Limit 版`
- Fit seconds: `116089.661303`
- Fit hours: `32.2471`
- Workbook: `docs/reports/CEMENT_STRENGTH_HCESVM_TEST3_FULL_20260517_142734.xlsx`
- Log: `results/archive/20260517_test3_no_time_limit_full/test3_cement_strength_20260517_142734.log`

## Total Accuracy

| Version | Fit Seconds | Fit Hours | Train Total | Test Total | Status | Note |
| --- | ---: | ---: | ---: | ---: | --- | --- |
| 1 分鐘極限版 | 240.537563 | 0.0668 | 0.6791979950 | 0.6450000000 | completed | per-classifier TimeLimit=60s, Threads=0 |
| 30 分鐘版 | 1800.320380 | 0.5001 | 0.6904761905 | 0.6650000000 | success | fit_seconds=1800.32038; legacy teaching-data three-model artifact |
| Software Memory Limit 版 | 116089.661303 | 32.2471 | 0.6892230576 | 0.6650000000 | completed | no TimeLimit; ended each H with mem_limit_with_solution |

## Per-Class Accuracy

| Version | Split | Total | C1 | C2 | C3 | C4 | C5 |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 分鐘極限版 | train | 0.6792 | 0.8471 (n=157) | 0.7460 (n=248) | 0.5692 (n=195) | 0.5207 (n=121) | 0.6494 (n=77) |
| 1 分鐘極限版 | test | 0.6450 | 0.8462 (n=39) | 0.7742 (n=62) | 0.5306 (n=49) | 0.4194 (n=31) | 0.4737 (n=19) |
| 30 分鐘版 | train | 0.6905 | 0.8408 (n=157) | 0.7258 (n=248) | 0.6103 (n=195) | 0.5372 (n=121) | 0.7143 (n=77) |
| 30 分鐘版 | test | 0.6650 | 0.7949 (n=39) | 0.7742 (n=62) | 0.6122 (n=49) | 0.4194 (n=31) | 0.5789 (n=19) |
| Software Memory Limit 版 | train | 0.6892 | 0.8471 (n=157) | 0.7258 (n=248) | 0.6103 (n=195) | 0.5207 (n=121) | 0.7143 (n=77) |
| Software Memory Limit 版 | test | 0.6650 | 0.7949 (n=39) | 0.7742 (n=62) | 0.6122 (n=49) | 0.4194 (n=31) | 0.5789 (n=19) |

## Hyperplane OBJ / b / Weights

| Version | H | Status | Elapsed (s) | OBJ | mip_gap | b | weights |
| --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| 1 分鐘極限版 | H1 | time_limit_with_solution | 60.11 | 132.1417783806 | 0.8986574515 | 1268.0196269302 | `[-1.006423064, -0.8080469509, -0.8569076692, -0.6839808424, -0.0114528021, -0.4067551846, -0.4590707031, -2.9160531748]` |
| 1 分鐘極限版 | H2 | time_limit_with_solution | 60.13 | 313.5258821421 | 0.9033211203 | 90.5778403851 | `[-0.3479187799, -0.2001225191, -0.1641367736, 0.5724167856, -1.1368706418, -0.0929573963, 0.0709751737, -0.9448516055]` |
| 1 分鐘極限版 | H3 | time_limit_with_solution | 60.14 | 314.8824724043 | 0.8887361510 | 399.9624126080 | `[-0.5919764766, -0.5611132545, -0.5830790506, 1.0052973568, 0.7714637251, -0.1445759732, -0.1439543401, -1.086034039]` |
| 1 分鐘極限版 | H4 | time_limit_with_solution | 60.12 | 107.7864276628 | 0.7447577790 | 1113.7980483890 | `[-0.8731947665, -0.9141973588, -0.8757960419, 0.5447176579, 1.4437534734, -0.4057638727, -0.4519026315, -0.2869084822]` |
| 30 分鐘版 | H1 | not recorded |  | 127.0881511664 | 0.8471107289 | 1331.5085727859 | `[-0.9137231812, -0.6824138021, -0.8221223045, -0.6907512915, 0.5538036274, -0.4595611594, -0.5040308287, -3.4686642835]` |
| 30 分鐘版 | H2 | not recorded |  | 295.9689903079 | 0.7576440905 | 520.0041222553 | `[-0.7775879375, -0.5315149947, -0.4803302367, 0.4437920774, -2.9719819567, -0.1466516492, -0.1566078733, -1.465028296]` |
| 30 分鐘版 | H3 | not recorded |  | 330.2603915336 | 0.8092890333 | 611.4828856652 | `[-0.4162615936, -0.4534352572, -0.3371329538, 0.0360779329, -2.1914686283, -0.2251331248, -0.2330797387, -0.373090449]` |
| 30 分鐘版 | H4 | not recorded |  | 108.4254344162 | 0.4988347187 | 268.3513313814 | `[-0.5119714443, -0.5442280315, -0.5332890467, 1.114009239, 2.2264639775, -0.0904291474, -0.1303059691, -0.2857142857]` |
| Software Memory Limit 版 | H1 | mem_limit_with_solution | 1976.72 | 127.0881323844 | 0.7902698870 | 1331.5085727860 | `[-0.9137231812, -0.6824138021, -0.8221223045, -0.6907512915, 0.5538036274, -0.4595611594, -0.5040308287, -3.4686642835]` |
| Software Memory Limit 版 | H2 | mem_limit_with_solution | 40323.38 | 295.9689903079 | 0.6039406890 | 520.0041222553 | `[-0.7775879375, -0.5315149947, -0.4803302367, 0.4437920774, -2.9719819567, -0.1466516492, -0.1566078733, -1.465028296]` |
| Software Memory Limit 版 | H3 | mem_limit_with_solution | 33024.19 | 330.2603653319 | 0.6613140239 | 611.4828856652 | `[-0.4162615936, -0.4534352572, -0.3371329538, 0.0360779329, -2.1914686283, -0.2251331248, -0.2330797387, -0.373090449]` |
| Software Memory Limit 版 | H4 | mem_limit_with_solution | 40765.34 | 107.4860030852 | 0.2772598635 | 308.0558212496 | `[-0.5569715348, -0.5730242515, -0.5577314874, 1.1444825313, 2.122792361, -0.0999903943, -0.1462676762, -0.2953850888]` |

## Source Artifacts

| Version | Workbook | Report | Log |
| --- | --- | --- | --- |
| 1 分鐘極限版 | `docs/reports/CEMENT_STRENGTH_HCESVM_TEST3_FULL_20260522_024218.xlsx` | `docs/reports/CEMENT_STRENGTH_HCESVM_TEST3_FULL_20260522_024218.md` | `results/archive/20260522_test3_teaching_data_full/test3_cement_strength_20260522_024218.log` |
| 30 分鐘版 | `docs/reports/TEACHING_DATA_THREE_MODEL_COMPARISON_20260423_190157.xlsx` | `docs/reports/TEACHING_DATA_CLASS_ACCURACY_SUMMARY_20260423_190157.md` | `results/archive/20260423_three_model_ordinal_regression/three_model_ordinal_regression_all_20260423_190157.log` |
| Software Memory Limit 版 | `docs/reports/CEMENT_STRENGTH_HCESVM_TEST3_FULL_20260517_142734.xlsx` | `docs/reports/CEMENT_STRENGTH_HCESVM_TEST3_FULL_20260517_142734.md` | `results/archive/20260517_test3_no_time_limit_full/test3_cement_strength_20260517_142734.log` |
