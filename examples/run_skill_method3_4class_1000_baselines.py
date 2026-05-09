#!/usr/bin/env python3
"""Wrapper for the exact archived skill 1000-sample baseline run."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from hcesvm.skill_1000_baselines_runner import main


if __name__ == "__main__":
    raise SystemExit(main())
