# HCESVM Classification Strategies

This document describes the current `HierarchicalCESVM(strategy=...)` interface.
The implementation accepts five strategy names:

- `single_filter`
- `multiple_filter`
- `inverted`
- `test3`
- `test4`

`class1_first` and `test2` are historical experiment names. They may still
appear in old scripts or reports, but they are not valid current
`HierarchicalCESVM` strategy values.

## Current Strategy Table

| Strategy | Scope | H1 Grouping | H2 / Hk Grouping | Objective |
| --- | --- | --- | --- | --- |
| `single_filter` | 3-class only | Class 3 vs {1,2} | H2: Class 2 vs 1 | Standard |
| `multiple_filter` | 3-class only | Class 1 vs {2,3} | H2: {1,2} vs 3 | Standard |
| `inverted` | 3-class only | Medium vs {Majority, Minority} | H2: {Medium, Majority} vs Minority | Standard |
| `test3` | N-class | Class 1 vs {2..N} | Hk: {1..k} vs {k+1..N} | Balanced sample weighting |
| `test4` | N-class | Class 1 vs {2..N} | Hk: {1..k} vs {k+1..N} | Normalized indicator penalty |

Legend:

- Majority: class with the largest training sample count
- Medium: class with the middle training sample count
- Minority: class with the smallest training sample count
- `s+`: positive sample count for a binary classifier
- `s-`: negative sample count for a binary classifier

## 3-Class Strategies

### `single_filter`

`single_filter` is the original 3-class cascade:

```text
H1: Class 3 (+1) vs {Class 1, Class 2} (-1)
  if H1 predicts +1 -> Class 3
  otherwise run H2

H2: Class 2 (+1) vs Class 1 (-1)
  if H2 predicts +1 -> Class 2
  otherwise -> Class 1
```

### `multiple_filter`

`multiple_filter` is the standard fixed 3-class grouping:

```text
H1: Class 1 (+1) vs {Class 2, Class 3} (-1)
  if H1 predicts +1 -> Class 1
  otherwise run H2

H2: {Class 1, Class 2} (+1) vs Class 3 (-1)
  if H2 predicts +1 -> Class 2
  otherwise -> Class 3
```

### `inverted`

`inverted` is a 3-class dynamic strategy. It determines majority, medium, and
minority classes from training sample counts:

```text
H1: Medium (+1) vs {Majority, Minority} (-1)
H2: {Medium, Majority} (+1) vs Minority (-1)
```

The objective remains the standard CE-SVM objective. Only the class grouping is
data-dependent.

## N-Class Strategies

### `test3`

`test3` supports 3-class and N-class ordinal classification. For N classes it
uses `N-1` binary classifiers:

```text
H1: Class 1 (+1) vs {2, 3, ..., N} (-1)
H2: {1, 2} (+1) vs {3, 4, ..., N} (-1)
H3: {1, 2, 3} (+1) vs {4, 5, ..., N} (-1)
...
H(N-1): {1, 2, ..., N-1} (+1) vs Class N (-1)
```

Prediction rule:

```text
Hk predicts +1 -> Class k
Hk predicts -1 -> continue to H(k+1)
H(N-1) predicts -1 -> Class N
```

`test3` configures each `BinaryCESVM` with:

- `class_weight="balanced"`
- `accuracy_mode="both"`

The weighted objective is:

```text
min  sum_j(w+_j + w-_j) + C * sum_i(alpha_i + beta_i + rho_i)
     - (1 / s+) * l+ - (1 / s-) * l-
```

This keeps the original indicator penalty and applies inverse-sample-count
weights to the positive and negative accuracy terms.

### `test4`

`test4` uses the same cascade grouping and prediction rule as `test3`, but it
defines the fourth model revision with a normalized indicator penalty:

```text
min  sum_j(w+_j + w-_j) + (C / (s+ + s-)) * sum_i(alpha_i + beta_i + rho_i)
     - (1 / s+) * l+ - (1 / s-) * l-
```

It configures each `BinaryCESVM` with:

- `objective_variant="test4"`
- `class_weight="balanced"`
- `accuracy_mode="both"`

## Selection Guide

| Situation | Recommended Strategy | Notes |
| --- | --- | --- |
| 3-class baseline comparison | `single_filter` | Original Class 3 first cascade |
| 3-class fixed grouping | `multiple_filter` | Stable and easy to compare |
| 3-class sample-count-adaptive grouping | `inverted` | Uses majority / medium / minority roles |
| 3-class or N-class teaching-data runs | `test4` | test3 grouping with normalized indicator penalty |

## Examples

3-class fixed strategy:

```python
from hcesvm import HierarchicalCESVM

model = HierarchicalCESVM(
    cesvm_params={
        "C_hyper": 1.0,
        "M": 1000.0,
        "time_limit": 1800,
    },
    strategy="multiple_filter",
)

model.fit(X1_train, X2_train, X3_train)
predictions = model.predict(X_test)
```

N-class `test4`:

```python
from hcesvm import HierarchicalCESVM

model = HierarchicalCESVM(
    cesvm_params={
        "C_hyper": 1.0,
        "M": 1000.0,
        "time_limit": 1800,
    },
    strategy="test4",
    n_classes=4,
)

model.fit(X1_train, X2_train, X3_train, X4_train)
predictions = model.predict(X_test)
```

## Runtime Semantics

`time_limit` is per classifier, not a total cascade budget. A 4-class `test3`
or `test4` run trains three binary classifiers, so `time_limit=1800` can take roughly
`1800s * 3` in the worst case.

Teaching-data HCESVM runners preserve this per-classifier behavior:

```bash
source .venv/bin/activate
python examples/run_teaching_data_hcesvm_1000.py --time-limit 1800
python examples/run_teaching_data_hcesvm_full.py --time-limit none
python examples/run_teaching_data_hcesvm_deadline.py --dataset skill --time-limit 1800
```

The full runner also supports Gurobi resource controls:

```bash
python examples/run_teaching_data_hcesvm_full.py \
  --time-limit none \
  --threads 0 \
  --soft-mem-limit-gb 56 \
  --nodefile-start-gb 23.6 \
  --nodefile-dir auto
```

## Historical References

The old strategy documents below describe previous experiments and may not match
the current `HierarchicalCESVM` constructor:

- [class1_first.md](./class1_first.md) - historical
- [test2.md](./test2.md) - historical

Current strategy details:

- [single_filter.md](./single_filter.md)
- [multiple_filter.md](./multiple_filter.md)
- [inverted.md](./inverted.md)
- [test3.md](./test3.md)
- `test4` is documented in this README until a dedicated strategy page exists.

For mathematical details:

- [CE-SVM Mathematical Model](/docs/CE_SVM_MATHEMATICAL_MODEL.md)
- [Decision Variables](/docs/DECISION_VARIABLES.md)
