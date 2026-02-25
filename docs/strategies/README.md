# HCESVM Classification Strategies

## Overview

This document provides a comprehensive overview of all classification strategies implemented in the HCESVM project for three-class ordinal classification problems.

## Strategy Types

HCESVM supports **6 classification strategies**, categorized into two types:

### Fixed Strategies
Strategies with predetermined class groupings that do not change based on sample distribution.

### Dynamic Strategies
Strategies that adapt class groupings based on the minority/majority class distribution in the training data.

---

## Strategy Comparison Table

| Strategy | Type | H1 Grouping | H2 Grouping | Objective Function | Use Case |
|----------|------|-------------|-------------|-------------------|----------|
| **single_filter** | Fixed | Class 3 vs {1,2} | Class 2 vs 1 | Standard | Original baseline strategy |
| **multiple_filter** | Fixed | Class 1 vs {2,3} | {1,2} vs 3 | Standard | Standard multi-filter approach |
| **class1_first** | Fixed | Class 1 vs {2,3} | Class 2 vs 3 | Standard | Prioritize Class 1 identification |
| **inverted** | Dynamic | Medium vs {Maj,Min} | {Med,Maj} vs Min | Standard | Adaptive class grouping |
| **test2** | Dynamic | Based on minority position | Based on minority position | Removes accuracy term for majority (when Class 2) | Aggressive imbalance handling |
| **test3** | Dynamic | Based on minority position | Based on minority position | Sample-weighted: `-(1/s_p)·l_p - (1/s_n)·l_n` | Balanced imbalance handling |

**Legend:**
- **Maj** = Majority class (largest sample count)
- **Med** = Medium class (middle sample count)
- **Min** = Minority class (smallest sample count)
- **s_p** = Number of positive (+1) class samples
- **s_n** = Number of negative (-1) class samples

---

## Fixed vs Dynamic Strategies

### Fixed Strategies

**Characteristics:**
- Class groupings are predetermined
- Independent of sample distribution
- Consistent behavior across all datasets
- Simpler to understand and debug

**When to use:**
- Balanced datasets (sample counts similar across classes)
- When domain knowledge suggests a specific classification hierarchy
- When reproducibility and consistency are prioritized

**Available Fixed Strategies:**
1. `single_filter` - Class 3 separation first
2. `multiple_filter` - Class 1 separation first
3. `class1_first` - Class 1 priority with specific prediction flow

### Dynamic Strategies

**Characteristics:**
- Class groupings adapt to sample distribution
- Minority class is identified at training time
- Different groupings for different datasets
- More complex but potentially better for imbalanced data

**When to use:**
- Imbalanced datasets (significant differences in sample counts)
- When minority class detection is critical
- When optimal performance across diverse datasets is desired

**Available Dynamic Strategies:**
1. `inverted` - Medium class separation with standard objective
2. `test2` - Aggressive accuracy term removal for majority class
3. `test3` - Balanced sample-weighted objective function

---

## Hierarchical Classification Flow

All strategies use a two-level hierarchical approach:

```
Training Data (3 classes)
        ↓
   [Classifier H1]
        ↓
   Split into 2 groups
        ↓
   [Classifier H2]
        ↓
  Final 3-class prediction
```

### Prediction Logic

**Fixed Strategies (single_filter, multiple_filter):**
- H1 → Binary prediction separates one group from another
- H2 → Further classifies within remaining group

**Fixed Strategy (class1_first):**
```
H1: Class 1 (+1) vs {Class 2, 3} (-1)
  ├─ H1 = +1 → Class 1
  └─ H1 = -1 → H2: Class 2 (+1) vs Class 3 (-1)
      ├─ H2 = +1 → Class 2
      └─ H2 = -1 → Class 3
```

**Dynamic Strategies:**
- Minority class position is detected during training
- Class groupings are adjusted accordingly
- Prediction flow depends on detected configuration

---

## Objective Function Variations

### Standard Objective
```
min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - l⁺ - l⁻
```
Used by: `single_filter`, `multiple_filter`, `class1_first`, `inverted`

### Test2 Objective (Accuracy Term Removal)
```
When majority class is Class 2:
  Remove the accuracy term for the majority class

Result: More aggressive optimization for minority class
```
Used by: `test2`

### Test3 Objective (Sample-Weighted)
```
min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - (1/s⁺)·l⁺ - (1/s⁻)·l⁻
```
Where:
- `s⁺` = number of positive class samples
- `s⁻` = number of negative class samples

**Effect:** Minority class receives higher weight automatically through inverse sample counting.

Used by: `test3`

---

## Strategy Selection Guide

### Decision Tree

```
Do you have significant class imbalance?
├─ No → Use Fixed Strategy
│   ├─ Domain suggests Class 1 priority? → class1_first
│   ├─ Standard approach? → multiple_filter
│   └─ Baseline comparison? → single_filter
│
└─ Yes → Use Dynamic Strategy
    ├─ Aggressive minority focus? → test2
    ├─ Balanced weighting? → test3
    └─ Standard adaptation? → inverted
```

### Recommended Defaults

| Scenario | Recommended Strategy | Rationale |
|----------|---------------------|-----------|
| Balanced dataset | `multiple_filter` or `class1_first` | Consistent, interpretable |
| Moderate imbalance (1:2-1:3) | `test3` | Balanced sample weighting |
| Severe imbalance (>1:3) | `test2` | Aggressive minority focus |
| Exploratory analysis | `inverted` | Adaptive with standard objective |
| Production deployment | `class1_first` | Fixed, predictable behavior |

---

## Implementation Examples

### Fixed Strategy Example
```python
from hcesvm import HierarchicalCESVM

model = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800
    },
    strategy='class1_first'  # Fixed strategy
)

model.fit(X1_train, X2_train, X3_train)
predictions = model.predict(X_test)
```

### Dynamic Strategy Example
```python
from hcesvm import HierarchicalCESVM

# Test3 strategy (sample-weighted)
model = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800
    },
    strategy='test3'  # Dynamic strategy with balanced weighting
)

model.fit(X1_train, X2_train, X3_train)
predictions = model.predict(X_test)
```

---

## Performance Considerations

### Computational Cost
All strategies have similar computational complexity, dominated by MIP solving time.

**Typical solve time per classifier:** 1-30 minutes (depends on dataset size and `time_limit`)

### Memory Usage
- **Fixed strategies:** Minimal overhead
- **Dynamic strategies:** Additional sample counting and class detection (~O(n))

### Accuracy Tradeoffs
- **Fixed strategies:** Consistent but may not adapt to imbalance
- **Dynamic strategies:** Potentially better on imbalanced data but less predictable

---

## Further Reading

For detailed information on each strategy:
- [single_filter.md](./single_filter.md)
- [multiple_filter.md](./multiple_filter.md)
- [class1_first.md](./class1_first.md)
- [inverted.md](./inverted.md)
- [test2.md](./test2.md)
- [test3.md](./test3.md)

For mathematical details:
- [CE-SVM Mathematical Model](/docs/CE_SVM_MATHEMATICAL_MODEL.md)
- [Decision Variables](/docs/DECISION_VARIABLES.md)

---

## Version History

| Version | Date | Changes |
|---------|------|---------|
| 1.0 | 2026-02-24 | Initial comprehensive strategy documentation |
