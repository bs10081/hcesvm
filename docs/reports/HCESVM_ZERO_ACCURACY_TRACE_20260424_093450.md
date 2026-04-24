# HCESVM Zero-Accuracy Coefficient Trace

Generated: 2026-04-24 09:34:50 UTC
Source workbook: `/home/bs10081/Developer/hcesvm-multiclass-eval/docs/reports/TEACHING_DATA_THREE_MODEL_COMPARISON_20260423_190157.xlsx`
Source log: `/home/bs10081/Developer/hcesvm-multiclass-eval/results/archive/20260423_three_model_ordinal_regression/three_model_ordinal_regression_all_20260423_190157.log`

## Conclusion

In the latest 30m-budget run, zero-coefficient classifiers explain some, but not necessarily all, zero-accuracy HCESVM classes.
The strongest evidence comes from `californiahousing` and `skill`, where zero-accuracy classes coincide with all-zero `Hk` weights, constant `b`, and time-limited solves with very high MIP gaps.

## Zero-Accuracy Summary

| Dataset | Split | Class | Samples | Blocker Hk | Zero-Coeff Fraction | Diagnosis |
|---|---|---:|---:|---|---:|---|
| californiahousing | train | 1 | 2595 | H1 | 1.00 | all samples blocked by always-negative zero-coefficient classifier(s) |
| californiahousing | train | 5 | 1103 | H2, H3, H4 | 0.66 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| californiahousing | train | 6 | 1239 | H2, H3, H4 | 0.82 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| californiahousing | test | 1 | 649 | H1 | 1.00 | all samples blocked by always-negative zero-coefficient classifier(s) |
| californiahousing | test | 5 | 276 | H2, H3, H4 | 0.63 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| californiahousing | test | 6 | 310 | H2, H3, H4 | 0.82 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| skill | train | 1 | 134 | H1 | 1.00 | all samples blocked by always-negative zero-coefficient classifier(s) |
| skill | train | 2 | 277 | H2 | 1.00 | all samples blocked by always-negative zero-coefficient classifier(s) |
| skill | train | 5 | 642 | H3, H4 | 0.86 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| skill | train | 6 | 497 | H3, H4 | 0.95 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| skill | train | 7 | 28 | H4 | 1.00 | all samples trapped by always-positive zero-coefficient classifier(s) |
| skill | test | 1 | 33 | H1 | 1.00 | all samples blocked by always-negative zero-coefficient classifier(s) |
| skill | test | 2 | 70 | H2 | 1.00 | all samples blocked by always-negative zero-coefficient classifier(s) |
| skill | test | 5 | 161 | H3, H4 | 0.89 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| skill | test | 6 | 124 | H3, H4 | 0.96 | mixed blockers; zero-coefficient classifiers explain part of the failure |
| skill | test | 7 | 7 | H4 | 1.00 | all samples trapped by always-positive zero-coefficient classifier(s) |

## californiahousing

Zero-coefficient classifiers in this dataset:
- `H1`: `always_negative`, `b=-1.0000`, `mip_gap=0.9965`, `time_limit=360s`
- `H4`: `always_positive`, `b=1.0000`, `mip_gap=0.9961`, `time_limit=360s`
- `H5`: `always_positive`, `b=1.0000`, `mip_gap=0.9954`, `time_limit=360s`

- `test` class `1`: all `649` samples were blocked before reaching the correct class. Diagnosis: all samples blocked by always-negative zero-coefficient classifier(s).
  blocker `H1` (`target_negative`): `blocked=649`, `all_zero_weights=True`, `constant_decision=always_negative`, `predictions=class_2=631, class_3=16, class_4=2`
- `test` class `5`: all `276` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H2` (`upstream_positive`): `blocked=15`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_2=15`
  blocker `H3` (`upstream_positive`): `blocked=88`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=88`
  blocker `H4` (`upstream_positive`): `blocked=173`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=173`
- `test` class `6`: all `310` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H2` (`upstream_positive`): `blocked=11`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_2=11`
  blocker `H3` (`upstream_positive`): `blocked=45`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=45`
  blocker `H4` (`upstream_positive`): `blocked=254`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=254`
- `train` class `1`: all `2595` samples were blocked before reaching the correct class. Diagnosis: all samples blocked by always-negative zero-coefficient classifier(s).
  blocker `H1` (`target_negative`): `blocked=2595`, `all_zero_weights=True`, `constant_decision=always_negative`, `predictions=class_2=2511, class_3=80, class_4=4`
- `train` class `5`: all `1103` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H2` (`upstream_positive`): `blocked=42`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_2=42`
  blocker `H3` (`upstream_positive`): `blocked=329`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=329`
  blocker `H4` (`upstream_positive`): `blocked=732`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=732`
- `train` class `6`: all `1239` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H2` (`upstream_positive`): `blocked=36`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_2=36`
  blocker `H3` (`upstream_positive`): `blocked=184`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=184`
  blocker `H4` (`upstream_positive`): `blocked=1019`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=1019`

## skill

Zero-coefficient classifiers in this dataset:
- `H1`: `always_negative`, `b=-1.0000`, `mip_gap=0.9118`, `time_limit=300s`
- `H2`: `always_negative`, `b=-1.0000`, `mip_gap=0.9640`, `time_limit=300s`
- `H4`: `always_positive`, `b=0.0000`, `mip_gap=0.9627`, `time_limit=300s`
- `H5`: `always_positive`, `b=1.0000`, `mip_gap=0.9799`, `time_limit=300s`
- `H6`: `always_positive`, `b=1.0000`, `mip_gap=0.7585`, `time_limit=300s`

- `test` class `1`: all `33` samples were blocked before reaching the correct class. Diagnosis: all samples blocked by always-negative zero-coefficient classifier(s).
  blocker `H1` (`target_negative`): `blocked=33`, `all_zero_weights=True`, `constant_decision=always_negative`, `predictions=class_3=27, class_4=6`
- `test` class `2`: all `70` samples were blocked before reaching the correct class. Diagnosis: all samples blocked by always-negative zero-coefficient classifier(s).
  blocker `H2` (`target_negative`): `blocked=70`, `all_zero_weights=True`, `constant_decision=always_negative`, `predictions=class_3=48, class_4=22`
- `test` class `5`: all `161` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H3` (`upstream_positive`): `blocked=17`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=17`
  blocker `H4` (`upstream_positive`): `blocked=144`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=144`
- `test` class `6`: all `124` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H3` (`upstream_positive`): `blocked=5`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=5`
  blocker `H4` (`upstream_positive`): `blocked=119`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=119`
- `test` class `7`: all `7` samples were blocked before reaching the correct class. Diagnosis: all samples trapped by always-positive zero-coefficient classifier(s).
  blocker `H4` (`upstream_positive`): `blocked=7`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=7`
- `train` class `1`: all `134` samples were blocked before reaching the correct class. Diagnosis: all samples blocked by always-negative zero-coefficient classifier(s).
  blocker `H1` (`target_negative`): `blocked=134`, `all_zero_weights=True`, `constant_decision=always_negative`, `predictions=class_3=106, class_4=28`
- `train` class `2`: all `277` samples were blocked before reaching the correct class. Diagnosis: all samples blocked by always-negative zero-coefficient classifier(s).
  blocker `H2` (`target_negative`): `blocked=277`, `all_zero_weights=True`, `constant_decision=always_negative`, `predictions=class_3=188, class_4=89`
- `train` class `5`: all `642` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H3` (`upstream_positive`): `blocked=87`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=87`
  blocker `H4` (`upstream_positive`): `blocked=555`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=555`
- `train` class `6`: all `497` samples were blocked before reaching the correct class. Diagnosis: mixed blockers; zero-coefficient classifiers explain part of the failure.
  blocker `H3` (`upstream_positive`): `blocked=23`, `all_zero_weights=False`, `constant_decision=non_constant`, `predictions=class_3=23`
  blocker `H4` (`upstream_positive`): `blocked=474`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=474`
- `train` class `7`: all `28` samples were blocked before reaching the correct class. Diagnosis: all samples trapped by always-positive zero-coefficient classifier(s).
  blocker `H4` (`upstream_positive`): `blocked=28`, `all_zero_weights=True`, `constant_decision=always_positive`, `predictions=class_4=28`

## Interpretation

- `always_negative` zero-coefficient classifiers make the corresponding class unreachable because `x·w + b < 0` for every sample.
- `always_positive` zero-coefficient classifiers siphon all remaining samples into the classifier index class, which can wipe out later classes in the cascade.
- The observed high MIP gaps indicate the solver stopped at the allocated time budget with weak incumbent quality; this is consistent with, but does not mathematically prove, that the time budget was too short.

