# SKILL Method3 4-Class 1000 Split Baselines

Generated at: `2026-05-09 07:39:20 UTC`
Workbook: `/home/bs10081/Developer/hcesvm/docs/reports/SKILL_METHOD3_4CLASS_1000_SVOR_NPSVOR_20260509_073920.xlsx`
Log: `/home/bs10081/Developer/hcesvm/results/archive/20260509_svor_npsvor_skill_1000/svor_npsvor_skill_method3_4class_1000_20260509_073920.log`
Manifest: `/home/bs10081/Developer/hcesvm/results/archive/20260509_test3_teaching_data_1000/skill_method3_4class_1000/skill_method3_4class_1000_manifest_20260509_055457.json`

- Source dataset: `skill`
- Source train CSV: `/home/bs10081/Developer/hcesvm/results/archive/20260509_test3_teaching_data_1000/skill_method3_4class_1000/skill_method3_4class_1000_train_20260509_055457.csv`
- Source test CSV: `/home/bs10081/Developer/hcesvm/results/archive/20260509_test3_teaching_data_1000/skill_method3_4class_1000/skill_method3_4class_1000_test_20260509_055457.csv`
- Relabel map: `4->1, 5->2, 6->3, 7->4`
- Train counts: `[649, 642, 497, 28] -> [286, 283, 219, 12]`
- Test counts: `[162, 161, 124, 7] -> [71, 71, 55, 3]`

## SVOR

- Final status: `completed`
- Fit seconds: `0.104`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.5375 | C1=0.5175 (n=286), C2=0.6784 (n=283), C3=0.4110 (n=219), C4=0.0000 (n=12) |
| Test | 0.5100 | C1=0.4789 (n=71), C2=0.6338 (n=71), C3=0.4182 (n=55), C4=0.0000 (n=3) |

| Component | Description | Status | Objective | b | weights |
| --- | --- | --- | ---: | --- | --- |
| GLOBAL | Shared hyperplane with ordered thresholds | optimal | 911.747012 | [-2.0748430409, -0.4138010926, 1.9883473612] | [-0.0012208341, 0.01603433, -2.193e-06, 0.0113154138, 0.0968945044, 0.0167677313, 0.1169640878, 0.0139530283, 0.0013973555, 0.0143796995, -0.0242983528, -0.024055001, -0.1738672084, -0.000323583, 0.0527965383, -0.0728812528, 0.0007746372, 0.0049525081] |

## NPSVOR

- Final status: `completed`
- Fit seconds: `0.342`

| Split | Total Accuracy | Per-Class Accuracy |
| --- | ---: | --- |
| Train | 0.5450 | C1=0.5315 (n=286), C2=0.6926 (n=283), C3=0.4018 (n=219), C4=0.0000 (n=12) |
| Test | 0.4900 | C1=0.4366 (n=71), C2=0.6338 (n=71), C3=0.4000 (n=55), C4=0.0000 (n=3) |

| Component | Description | Status | Objective | b | weights |
| --- | --- | --- | ---: | --- | --- |
| RANK_1 | Non-parallel hyperplane for class 1 | optimal | 184.475076 | 0.6713784946451973 | [0.0032956151, 0.0008880157, 0.0003454825, 0.0058170331, -0.0080194672, 0.0049122684, 0.054468295, 0.0072194529, -0.0013570706, 0.0130536507, -0.0052740326, -0.0075500504, -0.0439296832, -0.0004589444, 0.0317261398, -0.0391106989, 7.01387e-05, 0.003692047] |
| RANK_2 | Non-parallel hyperplane for class 2 | optimal | 361.793572 | 0.12229854319869252 | [0.0088695731, 0.0100668665, -1.5701e-06, 0.0099412153, 0.0604264114, 0.0083464357, 0.0664711263, 0.0080432537, -0.0049098767, 0.0085909507, -0.0115659776, -0.0123288673, -0.1295106375, -0.0018150399, 0.0384979663, -0.0506479835, -0.0010319909, 0.0011141419] |
| RANK_3 | Non-parallel hyperplane for class 3 | optimal | 172.975104 | -0.4243507260012307 | [-0.001964419, 0.0067359048, -1.1467e-06, 0.0059910058, 0.0555730071, 0.0114531655, 0.0308788244, 0.0049906187, -0.0012079339, 0.0064476983, -0.0111567374, -0.0045317102, -0.1281670976, 0.0014343287, 0.0150324033, -0.0351994811, 0.0014653156, -0.0002714252] |
| RANK_4 | Non-parallel hyperplane for class 4 | suboptimal_with_solution | 9.600237 | -0.9999549251780411 | [2.75e-06, 7.388e-07, 0.0, -1.291e-07, 0.0005218562, 0.001198146, 7.1088e-06, 0.0017104998, 0.0016183918, 0.0008779085, -9.84e-07, -2.1542e-06, -7.5049e-06, 8.467e-07, 0.0008150398, -6.6425e-06, -1.8147e-06, 0.0004531831] |

## Interpretation

- Exact split checked: train `[286, 283, 219, 12]` and test `[71, 71, 55, 3]`.
- SVOR C4 accuracy: train `0.0000`, test `0.0000`.
- NPSVOR C4 accuracy: train `0.0000`, test `0.0000`.
- Current HCESVM(test3) on the same split: train total `0.3538`, test total `0.3550`, C4 train/test `0.0000` / `0.0000`.
- Ordinal baselines that substantially outperform HCESVM on total accuracy: `SVOR, NPSVOR`.
- Minority-only explanation is incomplete: C4 still collapses for both baselines, but at least one ordinal baseline still improves total accuracy noticeably over HCESVM.
