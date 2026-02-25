# Coding Style Rules for HCESVM

## 🎯 Core Principles

### 1. Immutability (CRITICAL)
ALWAYS create new objects, NEVER mutate existing ones

### 2. File Organization
- 200-400 lines typical, **800 lines maximum**
- Extract utilities from large modules

### 3. Function Length
- Functions should be **< 50 lines**

## 📐 Python Style Guide

### PEP 8 Compliance
```python
# Line length: 88 characters (Black default)
# Indentation: 4 spaces

class BinaryCESVM:  # PascalCase for classes
    def fit_model(self, X, y):  # snake_case for functions
        time_limit = 1800  # snake_case for variables
        C_HYPER_DEFAULT = 1.0  # UPPERCASE for constants
```

### Type Hints
```python
from typing import Optional, Tuple
import numpy as np

def fit(
    self,
    X: np.ndarray,
    y: np.ndarray,
    time_limit: Optional[int] = None
) -> Tuple[np.ndarray, float]:
    """Fit the CE-SVM model."""
    pass
```

## 🧹 Code Quality Checklist

Before committing:
- [ ] Functions < 50 lines
- [ ] Files < 800 lines
- [ ] No deep nesting (>4 levels)
- [ ] No hardcoded values
- [ ] No mutation

## 🔧 Tools

```bash
# Black (formatter)
black src/ tests/ examples/

# isort (import sorting)
isort src/ tests/ examples/

# Flake8 (linter)
flake8 src/ --max-line-length=88
```

---

**Version**: 1.0  
**Last Updated**: 2026-02-25
