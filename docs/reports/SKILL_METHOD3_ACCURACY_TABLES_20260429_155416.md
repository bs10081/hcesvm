# Skill Method 3 Accuracy Tables

Generated at: `2026-04-29 15:54:16`

Sources:
- Previous teaching-data comparison workbook: `docs/reports/TEACHING_DATA_THREE_MODEL_COMPARISON_20260423_190157.xlsx`
- Previous class-accuracy summary: `docs/reports/TEACHING_DATA_CLASS_ACCURACY_SUMMARY_20260423_190157.md`
- Method 3 workbook: `docs/reports/SKILL_METHOD3_4CLASS_CEMENTSCALE_20260429_061944.xlsx`
- Method 3 log: `results/archive/20260429_method3_skill_4class/method3_skill_4class_cementscale_20260429_061944.log`

Adjusted `skill` definition for Method 3:
- Keep original classes `4, 5, 6, 7`
- Relabel to `C1=orig4`, `C2=orig5`, `C3=orig6`, `C4=orig7`
- Minority class: `C4`
- Train sample sizes: `C1=285`, `C2=282`, `C3=219`, `C4=12`
- Test sample sizes: `C1=71`, `C2=71`, `C3=55`, `C4=3`

## Table 1. HCESVM(test3) on Previous Datasets

| Dataset | Train Total | Test Total | Train Per-Class Accuracy | Test Per-Class Accuracy |
|---|---:|---:|---|---|
| `cement_strength` | 0.6905 | 0.6650 | `C1=0.8408, C2=0.7258, C3=0.6103, C4=0.5372, C5=0.7143` | `C1=0.7949, C2=0.7742, C3=0.6122, C4=0.4194, C5=0.5789` |
| `bostonhousing_ord` | 0.8366 | 0.7549 | `C1=0.8689, C2=0.8743, C3=0.7347, C4=0.8966, C5=0.8000` | `C1=0.8125, C2=0.8125, C3=0.6000, C4=0.8571, C5=0.6667` |
| `californiahousing` | 0.4523 | 0.4457 | `C1=0.0000, C2=0.6958, C3=0.6943, C4=0.4123, C5=0.0000, C6=0.0000` | `C1=0.0000, C2=0.6874, C3=0.6778, C4=0.4145, C5=0.0000, C6=0.0000` |
| `skill` | 0.2608 | 0.2455 | `C1=0.0000, C2=0.0000, C3=0.4887, C4=0.7396, C5=0.0000, C6=0.0000, C7=0.0000` | `C1=0.0000, C2=0.0000, C3=0.5315, C4=0.6481, C5=0.0000, C6=0.0000, C7=0.0000` |
| `stock_ord` | 0.8829 | 0.8053 | `C1=0.9524, C2=0.7637, C3=0.8807, C4=0.9515, C5=0.9130` | `C1=0.9688, C2=0.6889, C3=0.7037, C4=0.9286, C5=0.8235` |

## Table 2. HCESVM(test3) on Adjusted Skill (Method 3)

| Model | Split | Total Accuracy | Per-Class Accuracy | Sample Size |
|---|---|---:|---|---|
| `HCESVM(test3)` | Train | 0.3534 | `C1=0.0000, C2=1.0000, C3=0.0000, C4=0.0000` | `C1=285, C2=282, C3=219, C4=12 (minority)` |
| `HCESVM(test3)` | Test | 0.3550 | `C1=0.0000, C2=1.0000, C3=0.0000, C4=0.0000` | `C1=71, C2=71, C3=55, C4=3 (minority)` |

## Table 3. SVOR and NPSVOR on Adjusted Skill (Method 3)

| Model | Split | Total Accuracy | Per-Class Accuracy | Sample Size |
|---|---|---:|---|---|
| `SVOR` | Train | 0.5075 | `C1=0.4491, C2=0.6631, C3=0.4064, C4=0.0833` | `C1=285, C2=282, C3=219, C4=12 (minority)` |
| `SVOR` | Test | 0.5150 | `C1=0.4507, C2=0.6620, C3=0.4364, C4=0.0000` | `C1=71, C2=71, C3=55, C4=3 (minority)` |
| `NPSVOR` | Train | 0.5150 | `C1=0.4281, C2=0.7092, C3=0.4018, C4=0.0833` | `C1=285, C2=282, C3=219, C4=12 (minority)` |
| `NPSVOR` | Test | 0.5200 | `C1=0.4507, C2=0.6901, C3=0.4182, C4=0.0000` | `C1=71, C2=71, C3=55, C4=3 (minority)` |
