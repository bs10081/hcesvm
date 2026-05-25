# Testing Rules for HCESVM

## Coverage Target

Minimum target: 80% coverage for maintained source code.

Required test categories:

1. Unit tests for individual classes and helpers.
2. Integration tests for model pipelines, runners, and data loading.
3. Strategy tests for the current `HierarchicalCESVM` strategies:
   `single_filter`, `multiple_filter`, `inverted`, and `test3`.

## Current Strategy Expectations

- `single_filter`, `multiple_filter`, and `inverted` are 3-class only.
- `test3` supports 3-class and N-class runs.
- Invalid strategy names should raise clearly.
- N-class usage with non-`test3` strategies should fail.

Primary strategy test file:

```bash
source .venv/bin/activate
pytest tests/test_hierarchical.py -q
```

## Runtime-Semantics Tests

HCESVM runner `time_limit` values are per classifier, not total budgets. Changes
to teaching-data runners or runtime helpers must preserve that behavior.

```bash
source .venv/bin/activate
pytest tests/test_teaching_data_runtime.py tests/test_runner_time_limit_semantics.py -q
```

Full teaching-data runner resource controls are covered here:

```bash
source .venv/bin/activate
pytest tests/test_teaching_data_hcesvm_full_runner.py -q
```

## Test Execution

Run focused tests first, then broaden when changing shared behavior:

```bash
source .venv/bin/activate

# Focused strategy tests
pytest tests/test_hierarchical.py -q

# Runner semantics
pytest tests/test_teaching_data_runtime.py tests/test_runner_time_limit_semantics.py -q

# Full runner CLI/resource behavior
pytest tests/test_teaching_data_hcesvm_full_runner.py -q

# Broader suite when touching shared model behavior
pytest tests/ -q
```

## Experiment Output Requirements

Validation and formal experiment logs must include:

- training/testing per-class accuracy
- training/testing total accuracy
- each classifier's `weights`
- each classifier's `b`
- solve status / solver status label

Archive validation logs under `results/archive/YYYYMMDD_strategy_name/` with a
timestamped filename.

## Pre-Commit Checklist

- [ ] Relevant pytest targets pass under `.venv`.
- [ ] `git diff --check` passes.
- [ ] New or changed behavior has corresponding tests.
- [ ] Generated reports/logs/workbooks are intentionally included or left out.

**Version**: 2.1
**Last Updated**: 2026-05-25
