#!/usr/bin/env python3
"""Quick validation: only check feature count matching."""

import sys
from pathlib import Path
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm.utils import load_multiclass_data


# Only the 7 problematic datasets
DATASETS = [
    ("Contraceptive", "~/Developer/NSVORA/Archive/contraceptive_split.xlsx"),
    ("Hayes_Roth", "~/Developer/NSVORA/Archive/hayes-roth_split.xlsx"),
    ("Squash_Stored", "~/Developer/NSVORA/Archive/squash-stored_split.xlsx"),
    ("Squash_Unstored", "~/Developer/NSVORA/Archive/squash-unstored_split.xlsx"),
    ("TAE", "~/Developer/NSVORA/Archive/tae_split.xlsx"),
    ("Thyroid", "~/Developer/NSVORA/Archive/thyroid_split.xlsx"),
    ("Wine", "~/Developer/NSVORA/Archive/wine_split.xlsx"),
]


def quick_validate(dataset_name: str, data_path: str) -> dict:
    """Quick validation: check if train and test feature counts match.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset Excel file

    Returns:
        dict with validation results
    """
    data_file = Path(data_path).expanduser()

    if not data_file.exists():
        return {'success': False, 'error': 'File not found'}

    try:
        # Load training data
        X_train_classes, _, n_train_features = load_multiclass_data(
            str(data_file),
            sheet_name='Train',
            skiprows=5
        )

        # Load test data
        X_test_classes, _, n_test_features = load_multiclass_data(
            str(data_file),
            sheet_name='Test',
            skiprows=4
        )

        # Check feature count match
        match = (n_train_features == n_test_features)

        return {
            'success': match,
            'train_features': n_train_features,
            'test_features': n_test_features,
            'match': match
        }

    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'train_features': None,
            'test_features': None,
            'match': False
        }


def main():
    """Quick validate all 7 problematic datasets."""
    print("="*70)
    print("Quick Validation: Feature Count Matching")
    print("="*70)
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    results = {}

    for dataset_name, data_path in DATASETS:
        print(f"Checking {dataset_name:20s}...", end=" ")
        result = quick_validate(dataset_name, data_path)
        results[dataset_name] = result

        if result['success'] and result['match']:
            print(f"✅ MATCH (Train={result['train_features']}, Test={result['test_features']})")
        elif not result['match']:
            print(f"❌ MISMATCH (Train={result['train_features']}, Test={result['test_features']})")
        else:
            print(f"❌ ERROR: {result.get('error', 'Unknown')}")

    # Summary
    print()
    print("="*70)
    print("Summary")
    print("="*70)

    passed = sum(1 for r in results.values() if r['success'] and r['match'])
    total = len(results)

    print(f"Total: {passed}/{total} datasets have matching feature counts")

    if passed == total:
        print("\n✅ All datasets fixed! Feature counts now match between Train and Test.")
        print("\nYou can now run the full Test2 tests to get testing accuracy.")
        return 0
    else:
        print("\n❌ Some datasets still have issues. Please check the results above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
