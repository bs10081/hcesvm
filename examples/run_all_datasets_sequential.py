#!/usr/bin/env python3
"""Sequential execution of all NSVORA primary datasets tests."""

import sys
from pathlib import Path
import subprocess
from datetime import datetime

def run_test(script_name, dataset_name):
    """Run a single test script and capture results."""
    print("\n" + "=" * 80)
    print(f"Starting: {dataset_name}")
    print(f"Script: {script_name}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80 + "\n")

    script_path = Path(__file__).parent / script_name
    log_file = Path(__file__).parent.parent / f"{dataset_name.lower().replace(' ', '_')}_test_output.log"

    try:
        # Run the script and capture output
        with open(log_file, 'w') as f:
            result = subprocess.run(
                [sys.executable, str(script_path)],
                stdout=f,
                stderr=subprocess.STDOUT,
                text=True
            )

        # Display the log content
        with open(log_file, 'r') as f:
            print(f.read())

        if result.returncode == 0:
            print(f"\n✓ {dataset_name} completed successfully")
            print(f"  Output saved to: {log_file}")
            return True
        else:
            print(f"\n✗ {dataset_name} failed with exit code {result.returncode}")
            print(f"  Check log file: {log_file}")
            return False

    except Exception as e:
        print(f"\n✗ {dataset_name} failed with exception: {e}")
        return False

def main():
    """Run all tests sequentially."""
    print("=" * 80)
    print("Sequential Execution of All NSVORA Primary Datasets")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nTests to run:")
    print("  1. Abalone (multiple_filter) - ~60 min")
    print("  2. Solar Flare (single_filter) - ~60 min")
    print("  3. Wine Quality (multiple_filter) - ~60 min")
    print("\nEstimated total time: ~180 minutes (3 hours)")
    print("=" * 80)

    start_time = datetime.now()
    results = {}

    # Test 1: Abalone
    results['Abalone'] = run_test('run_abalone_hierarchical.py', 'Abalone')

    # Test 2: Solar Flare
    results['Solar Flare'] = run_test('run_solar_flare_hierarchical.py', 'Solar Flare')

    # Test 3: Wine Quality
    results['Wine Quality'] = run_test('run_wine_quality_hierarchical.py', 'Wine Quality')

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print("\n" + "=" * 80)
    print("All Tests Completed!")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print("\nResults Summary:")

    for dataset, success in results.items():
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status} - {dataset}")

    passed = sum(results.values())
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 80)

    # Return exit code based on results
    return 0 if all(results.values()) else 1

if __name__ == "__main__":
    sys.exit(main())
