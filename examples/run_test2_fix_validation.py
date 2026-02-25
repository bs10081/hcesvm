#!/usr/bin/env python3
"""Validate Test2 fix for 7 problematic datasets."""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm import HierarchicalCESVM, get_default_params
from hcesvm.utils import load_multiclass_data, evaluate_hierarchical_model, print_evaluation_results


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


def validate_single_dataset(dataset_name: str, data_path: str) -> dict:
    """Validate that test data can be loaded and features match training data.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset Excel file

    Returns:
        dict with validation results
    """
    data_file = Path(data_path).expanduser()

    print(f"\n{'='*60}")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*60}")

    if not data_file.exists():
        print(f"❌ File not found: {data_file}")
        return {'success': False, 'error': 'File not found'}

    try:
        # Load training data
        print("Loading training data...")
        X_train_classes, n_train_classes, n_train_features = load_multiclass_data(
            str(data_file),
            sheet_name='Train',
            skiprows=5
        )
        X1_train, X2_train, X3_train = X_train_classes

        print(f"  Train features: {n_train_features}")
        print(f"  Class sizes: {len(X1_train)}, {len(X2_train)}, {len(X3_train)}")

        # Load test data
        print("Loading test data...")
        X_test_classes, n_test_classes, n_test_features = load_multiclass_data(
            str(data_file),
            sheet_name='Test',
            skiprows=4
        )
        X1_test, X2_test, X3_test = X_test_classes

        print(f"  Test features: {n_test_features}")
        print(f"  Class sizes: {len(X1_test)}, {len(X2_test)}, {len(X3_test)}")

        # Check feature count match
        if n_train_features == n_test_features:
            print(f"✅ Feature count matches: {n_train_features}")

            # Quick test: train and predict
            print("\nQuick validation: Training and predicting...")
            params = get_default_params()
            params['verbose'] = False
            params['time_limit'] = 300  # 5 minutes max

            hc = HierarchicalCESVM(cesvm_params=params, strategy="test2")
            hc.fit(X1_train, X2_train, X3_train)

            # Predict on test data
            X_test_all = np.vstack([X1_test, X2_test, X3_test])
            y_test_all = np.concatenate([
                np.ones(len(X1_test), dtype=int),
                2 * np.ones(len(X2_test), dtype=int),
                3 * np.ones(len(X3_test), dtype=int)
            ])

            y_test_pred = hc.predict(X_test_all)
            test_results = evaluate_hierarchical_model(y_test_all, y_test_pred)
            test_accuracy = test_results['total_accuracy']

            print(f"✅ Test accuracy: {test_accuracy:.4f}")

            return {
                'success': True,
                'train_features': n_train_features,
                'test_features': n_test_features,
                'test_accuracy': test_accuracy
            }
        else:
            print(f"❌ Feature mismatch: Train={n_train_features}, Test={n_test_features}")
            return {
                'success': False,
                'error': f'Feature mismatch: Train={n_train_features}, Test={n_test_features}',
                'train_features': n_train_features,
                'test_features': n_test_features
            }

    except Exception as e:
        print(f"❌ Error: {e}")
        return {'success': False, 'error': str(e)}


def main():
    """Validate all 7 problematic datasets."""
    print("="*60)
    print("Test2 Fix Validation")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\nValidating {len(DATASETS)} datasets...")

    results = {}

    for dataset_name, data_path in DATASETS:
        result = validate_single_dataset(dataset_name, data_path)
        results[dataset_name] = result

    # Summary
    print("\n" + "="*60)
    print("Validation Summary")
    print("="*60)

    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)

    print(f"\nResults:")
    for dataset_name, result in results.items():
        if result['success']:
            print(f"  ✅ {dataset_name:20s} - Test Accuracy: {result['test_accuracy']:.4f}")
        else:
            print(f"  ❌ {dataset_name:20s} - {result.get('error', 'Unknown error')}")

    print(f"\nTotal: {passed}/{total} datasets passed")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    return 0 if passed == total else 1


if __name__ == "__main__":
    sys.exit(main())
