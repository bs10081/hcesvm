#!/usr/bin/env python3
"""
Evaluate Test3 models on Car_Evaluation and New_Thyroid datasets.
"""

import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm import HierarchicalCESVM, get_default_params
from hcesvm.utils import load_multiclass_data, evaluate_hierarchical_model


def detect_test_skiprows(data_file: Path) -> int:
    """Detect skiprows for Test sheet (usually 4)."""
    return 4


def evaluate_dataset(dataset_name: str, data_path: str):
    """Evaluate Test3 strategy on a single dataset."""

    print("=" * 80)
    print(f"Evaluating {dataset_name}")
    print("=" * 80)

    # Expand path
    data_file = Path(data_path).expanduser()

    if not data_file.exists():
        print(f"Error: Dataset not found: {data_file}")
        return None

    # Load training data
    print("\nLoading training data...")
    X_classes_train, n_classes_train, n_features = load_multiclass_data(
        str(data_file), sheet_name='Train', skiprows=5
    )
    X1_train, X2_train, X3_train = X_classes_train

    # Load test data
    print("Loading test data...")
    skiprows_test = detect_test_skiprows(data_file)
    X_classes_test, n_classes_test, _ = load_multiclass_data(
        str(data_file), sheet_name='Test', skiprows=skiprows_test
    )
    X1_test, X2_test, X3_test = X_classes_test

    # Combine test data
    X_test_all = np.vstack([X1_test, X2_test, X3_test])
    y_test_all = np.hstack([
        np.ones(len(X1_test)),
        2 * np.ones(len(X2_test)),
        3 * np.ones(len(X3_test))
    ])

    X_train_all = np.vstack([X1_train, X2_train, X3_train])
    y_train_all = np.hstack([
        np.ones(len(X1_train)),
        2 * np.ones(len(X2_train)),
        3 * np.ones(len(X3_train))
    ])

    print(f"Training samples: Class 1={len(X1_train)}, Class 2={len(X2_train)}, Class 3={len(X3_train)}")
    print(f"Testing samples: Class 1={len(X1_test)}, Class 2={len(X2_test)}, Class 3={len(X3_test)}")

    # Get default parameters
    params = get_default_params()
    params['verbose'] = False
    params['time_limit'] = 1800
    params['mip_gap'] = 1e-4

    # Train model
    print("\nTraining Test3 model...")
    hc = HierarchicalCESVM(cesvm_params=params, strategy='test3')
    hc.fit(X1_train, X2_train, X3_train)

    # Evaluate on training set
    print("\n" + "-" * 60)
    print("Training Set:")
    y_train_pred = hc.predict(X_train_all)
    train_metrics = evaluate_hierarchical_model(y_train_all, y_train_pred)
    print(f"  Total Accuracy: {train_metrics['total_accuracy']:.4f}")
    print(f"  Class 1: {train_metrics['per_class_accuracy']['Class 1']:.4f}")
    print(f"  Class 2: {train_metrics['per_class_accuracy']['Class 2']:.4f}")
    print(f"  Class 3: {train_metrics['per_class_accuracy']['Class 3']:.4f}")

    # Evaluate on test set
    print("\n" + "-" * 60)
    print("Test Set:")
    y_test_pred = hc.predict(X_test_all)
    test_metrics = evaluate_hierarchical_model(y_test_all, y_test_pred)
    print(f"  Total Accuracy: {test_metrics['total_accuracy']:.4f}")
    print(f"  Class 1: {test_metrics['per_class_accuracy']['Class 1']:.4f}")
    print(f"  Class 2: {test_metrics['per_class_accuracy']['Class 2']:.4f}")
    print(f"  Class 3: {test_metrics['per_class_accuracy']['Class 3']:.4f}")

    return {
        'train_acc': train_metrics['total_accuracy'],
        'test_acc': test_metrics['total_accuracy'],
        'train_class_acc': train_metrics['per_class_accuracy'],
        'test_class_acc': test_metrics['per_class_accuracy']
    }


if __name__ == "__main__":
    datasets = [
        ('Car_Evaluation', "~/Developer/NSVORA/datasets/primary/car_evaluation/car_evaluation_split.xlsx"),
        ('New_Thyroid', "~/Developer/NSVORA/Archive/new-thyroid_split.xlsx")
    ]

    results = {}

    for dataset_name, dataset_path in datasets:
        try:
            result = evaluate_dataset(dataset_name, dataset_path)
            if result is not None:
                results[dataset_name] = result
        except Exception as e:
            print(f"\nError evaluating {dataset_name}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    for dataset_name, result in results.items():
        print(f"\n{dataset_name}:")
        print(f"  Training Accuracy: {result['train_acc']:.4f}")
        print(f"  Test Accuracy:     {result['test_acc']:.4f}")
