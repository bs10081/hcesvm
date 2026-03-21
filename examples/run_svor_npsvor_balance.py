"""Run the balance workbook through both SVOR and NPSVOR implementations."""

from __future__ import annotations

import json
from pathlib import Path

from hcesvm.ordinal_cli import run_workbook


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    jobs = [
        ("svor", root / "data" / "svor" / "SVOR_balance_split.xlsx"),
        ("npsvor", root / "data" / "npsvor" / "NPSVOR_balance_split.xlsx"),
    ]

    for model_name, workbook in jobs:
        print(f"=== {model_name.upper()} ===")
        summary = run_workbook(
            workbook,
            model_name=model_name,
            prediction_rule="min_distance",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
