# HCESVM Agent Instructions

Follow `CLAUDE.md` for detailed project guidance. This file is the short entry
contract for coding agents working anywhere in this repository.

Hard rules:

- Preserve user changes. Do not revert unrelated work or generated artifacts.
- Run Python, pytest, and project scripts through `.venv`:
  `source .venv/bin/activate && ...`
- Timestamp every experiment run and archive logs under
  `results/archive/YYYYMMDD_strategy_name/`.
- Experiment outputs must include train/test per-class accuracy, train/test total
  accuracy, classifier `weights`, `b`, and solve status.
- HCESVM runner `time_limit` values are per classifier, not total budgets.
- Do not leave temporary logs, monitors, ad hoc test scripts, generated scripts,
  or backup files in the repo root.
- Do not stage unrelated generated outputs, reports, workbooks, logs, or CSVs.

Current `HierarchicalCESVM` strategies are `single_filter`, `multiple_filter`,
`inverted`, and `test3`. Treat `class1_first` and `test2` as historical names
only, unless the user explicitly asks to inspect legacy artifacts.
