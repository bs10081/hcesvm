#!/usr/bin/env python3
"""Run Hierarchical CE-SVM tests on all datasets sequentially."""

import subprocess
import sys
from pathlib import Path

def main():
    examples_dir = Path(__file__).parent

    scripts = [
        ("Abalone", examples_dir / "run_abalone_hierarchical.py"),
        ("Wine Quality", examples_dir / "run_wine_quality_hierarchical.py"),
    ]

    for name, script_path in scripts:
        print("\n" + "=" * 80)
        print(f"Starting {name} Dataset Test")
        print("=" * 80 + "\n")

        result = subprocess.run([sys.executable, str(script_path)])

        if result.returncode != 0:
            print(f"\nError: {name} test failed with return code {result.returncode}")
            sys.exit(result.returncode)

        print(f"\n{name} Dataset Test Completed")

    print("\n" + "=" * 80)
    print("All Dataset Tests Completed Successfully!")
    print("=" * 80)

if __name__ == "__main__":
    main()
