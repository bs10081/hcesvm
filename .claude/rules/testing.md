# Testing Rules for HCESVM

## 🎯 Minimum Test Coverage: 80%

### Required Test Types
1. **Unit Tests** - Individual functions, classes
2. **Integration Tests** - Model pipelines, data loading
3. **Strategy Tests** - All 4 classification strategies (single_filter, multiple_filter, inverted, test3)

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

## 🤖 Using Subagents for Testing

### When to Use Subagents

**Use subagents for**:
- Long-running validation tests (>5 minutes)
- Dataset validation tests across multiple datasets
- Strategy comparison tests
- Tests that generate extensive logs

**DON'T use subagents for**:
- Quick unit tests (use pytest directly)
- Tests that need immediate feedback
- Interactive debugging

### Available Test Agents

#### 1. general-purpose
**Use for**: Multi-dataset validation, strategy comparison

```python
Task(
    subagent_type="general-purpose",
    description="Validate Test3 strategy",
    prompt="""
    Execute Test3 strategy validation:
    1. Activate venv: source .venv/bin/activate
    2. Run: python examples/validate_test3_refactor.py
    3. Save output with timestamp
    4. Report: training/testing accuracy for each dataset
    """,
    run_in_background=False
)
```

#### 2. tdd-guide
**Use for**: Enforcing test-first development

```python
Task(
    subagent_type="tdd-guide",
    description="Add Test3 strategy tests",
    prompt="Create tests for Test3 fixed grouping strategy following TDD"
)
```

### Test Execution Pattern

```python
# Example: Run validation tests via subagent
Task(
    subagent_type="general-purpose",
    description="Dataset validation tests",
    prompt="""
    Run validation tests for contraceptive and thyroid datasets:
    - Use Test3 strategy (fixed grouping with balanced weighting)
    - Record training and testing accuracy
    - Save results to: results/test3_validation_{timestamp}.log
    - Report back with summary
    """
)
```

## 📊 Running Tests

### Unit Tests (Direct Execution)
```bash
# All tests
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src/hcesvm --cov-report=html

# Specific test
pytest tests/test_hierarchical.py::TestHierarchicalCESVMFixedStrategies -v
```

### Validation Tests (Via Subagent)
```python
# Long-running dataset validation
Task(
    subagent_type="general-purpose",
    description="Run dataset validation",
    prompt="Execute python examples/validate_test3_refactor.py"
)
```

## ✅ Test Quality Standards

### Good Tests are:
1. **Fast** - Run in seconds (unit tests)
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

- [ ] All unit tests pass (`pytest tests/`)
- [ ] Coverage ≥ 80% (`pytest --cov=src/hcesvm`)
- [ ] New code has corresponding tests
- [ ] Validation tests passed (if modifying strategies)

## 📁 Test Results Management

### Output Locations
- **Unit test results**: Terminal output
- **Validation test logs**: `results/test3_validation_YYYYMMDD_HHMMSS.log`
- **Coverage reports**: `htmlcov/`

### Archiving
Use `/archive-results` or `scripts/archive_results.py` to archive test logs:

```bash
python scripts/archive_results.py
```

Archived to: `results/archive/YYYYMMDD_test3_validation/`

---

**Version**: 2.0
**Last Updated**: 2026-02-26
