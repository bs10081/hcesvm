# Testing Rules for HCESVM

## 🎯 Minimum Test Coverage: 80%

### Required Test Types
1. **Unit Tests** - Individual functions, classes
2. **Integration Tests** - Model pipelines, data loading
3. **Strategy Tests** - All 6 classification strategies

## ✅ Test-Driven Development (TDD)

### MANDATORY Workflow:
1. **Write test first** (RED) - Test should FAIL
2. **Write minimal implementation** (GREEN) - Test should PASS
3. **Refactor** (IMPROVE) - Clean up code
4. **Verify coverage** - Ensure 80%+

### Example:
```python
# Step 1: Write failing test
def test_binary_cesvm_fit():
    model = BinaryCESVM()
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, -1])
    model.fit(X, y)
    assert model.is_fitted

# Step 2: Implement
def fit(self, X, y):
    self.is_fitted = True

# Step 3: Refactor
def fit(self, X, y):
    self._validate_input(X, y)
    self._build_model(X, y)
    self.is_fitted = True
```

## 📊 Running Tests

```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/hcesvm --cov-report=html

# Specific test
pytest tests/test_hierarchical.py::TestHierarchicalCESVMFixedStrategies -v
```

## ✅ Test Quality Standards

### Good Tests are:
1. **Fast** - Run in seconds
2. **Isolated** - No dependencies
3. **Repeatable** - Same results
4. **Self-validating** - Clear pass/fail
5. **Timely** - Written before code (TDD)

### Test Structure (AAA Pattern):
```python
def test_predict_returns_correct_shape():
    # Arrange
    model = BinaryCESVM()
    X_train = np.array([[1, 2], [3, 4]])
    y_train = np.array([1, -1])
    
    # Act
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    
    # Assert
    assert predictions.shape == (1,)
```

## 📋 Pre-Commit Checklist

- [ ] All tests pass (`pytest tests/`)
- [ ] Coverage ≥ 80% (`pytest --cov=src/hcesvm`)
- [ ] New code has corresponding tests

---

**Version**: 1.0  
**Last Updated**: 2026-02-25
