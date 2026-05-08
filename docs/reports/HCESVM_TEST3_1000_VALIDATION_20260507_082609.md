# HCESVM(test3) Derived 1000-Sample Validation

Generated at: `2026-05-07 09:26:12 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/HCESVM_TEST3_1000_VALIDATION_20260507_082609.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260507_test3_teaching_data_1000/test3_teaching_data_1000_20260507_082609.log`

## skill_method3_4class_1000

- Source dataset: `skill`
- Split rule: reuse the existing stratified split, then downsample to `800/200` independently
- Train counts: `[649, 642, 497, 28] -> [286, 283, 219, 12]`
- Test counts: `[162, 161, 124, 7] -> [71, 71, 55, 3]`
- Per-classifier time limit: `600s`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.3538 | C1=0.0000 (n=286), C2=1.0000 (n=283), C3=0.0000 (n=219), C4=0.0000 (n=12) |
| Test | 0.3550 | C1=0.0000 (n=71), C2=1.0000 (n=71), C3=0.0000 (n=55), C4=0.0000 (n=3) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4} (-1) | 286/514 | time_limit_with_solution | 600.30 | 571.998054 | 0.945570 | -1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| H2 | Class {1, 2} (+1) vs Class {3, 4} (-1) | 569/231 | time_limit_with_solution | 600.58 | 461.998243 | 0.961944 | 1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| H3 | Class {1, 2, 3} (+1) vs Class {4} (-1) | 788/12 | time_limit_with_solution | 600.18 | 23.998731 | 0.097322 | 1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |

## californiahousing_1000

- Source dataset: `californiahousing`
- Split rule: reuse the existing stratified split, then downsample to `800/200` independently
- Train counts: `[2595, 5476, 4059, 2040, 1103, 1239] -> [126, 265, 197, 99, 53, 60]`
- Test counts: `[649, 1369, 1015, 509, 276, 310] -> [32, 66, 49, 25, 13, 15]`
- Per-classifier time limit: `360s`
- Final status: `completed`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.5138 | C1=0.0000 (n=126), C2=0.6453 (n=265), C3=0.7005 (n=197), C4=0.5859 (n=99), C5=0.2075 (n=53), C6=0.5500 (n=60) |
| Test | 0.5000 | C1=0.0000 (n=32), C2=0.5909 (n=66), C3=0.7959 (n=49), C4=0.3600 (n=25), C5=0.1538 (n=13), C6=0.7333 (n=15) |

| Classifier | Description | Samples (+/-) | Status | Elapsed (s) | Objective | mip_gap | b | weights |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | --- |
| H1 | Class {1} (+1) vs Class {2, 3, 4, 5, 6} (-1) | 126/674 | time_limit_with_solution | 360.19 | 251.998516 | 0.853492 | -1.000000 | [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] |
| H2 | Class {1, 2} (+1) vs Class {3, 4, 5, 6} (-1) | 391/409 | time_limit_with_solution | 360.54 | 457.662947 | 0.969448 | 492.297755 | [5.8778648624, 6.254725008, 0.0609348946, 0.0036262489, -0.0329655348, 0.0022468844, 0.0118162356, -4.4229339819] |
| H3 | Class {1, 2, 3} (+1) vs Class {4, 5, 6} (-1) | 588/212 | time_limit_with_solution | 360.46 | 262.896602 | 0.946531 | 298.346411 | [3.3760034109, 4.01751814, -0.2537773185, -0.00480741, 0.0024435514, 0.0197329809, -0.0334249807, -5.1939932686] |
| H4 | Class {1, 2, 3, 4} (+1) vs Class {5, 6} (-1) | 687/113 | time_limit_with_solution | 360.21 | 203.865947 | 0.787608 | 116.692344 | [0.9946906403, 1.6071738785, -0.322915497, -0.0064307881, 0.0560253326, 0.0356520283, -0.1123693879, -6.7364421914] |
| H5 | Class {1, 2, 3, 4, 5} (+1) vs Class {6} (-1) | 740/60 | time_limit_with_solution | 360.28 | 106.233526 | 0.856884 | 70.472027 | [0.8867924984, 1.9220368342, -0.3087206832, -0.002817864, -0.0129621571, 0.0124370056, -0.0018465489, -3.0972471592] |
