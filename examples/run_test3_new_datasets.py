#!/usr/bin/env python3
"""
Run Test3 Strategy on 5 New Ordinal Regression Datasets

Tests Test3 strategy with N-class support (5-7 classes) on:
1. bostonhousing_ord (5 classes)
2. cement_strength (5 classes)
3. stock_ord (5 classes)
4. skill (7 classes)
5. californiahousing (6 classes)
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from hcesvm.models.hierarchical import HierarchicalCESVM
from hcesvm.utils.data_loader import load_csv_ordinal_data
from hcesvm.utils.evaluator import evaluate_multiclass


def run_test3_on_dataset(dataset_name: str, filepath: str, params: dict, log_file=None):
    """Run Test3 strategy on a single dataset.

    Args:
        dataset_name: Name of dataset
        filepath: Path to CSV file
        params: CE-SVM parameters
        log_file: File handle for logging
    """
    print("\n" + "=" * 80)
    print(f"DATASET: {dataset_name}")
    print("=" * 80)

    if log_file:
        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write(f"DATASET: {dataset_name}\n")
        log_file.write("=" * 80 + "\n")

    # Load data
    train_classes, test_classes, X_test, y_test, n_features = load_csv_ordinal_data(
        filepath,
        target_col="response",
        test_size=0.2,
        random_state=42
    )

    n_classes = len(train_classes)

    # Create model
    model = HierarchicalCESVM(
        cesvm_params=params,
        strategy='test3',
        n_classes=n_classes
    )

    # Fit model
    print(f"\n{'='*80}")
    print(f"Training Test3 strategy on {dataset_name}...")
    print(f"{'='*80}")

    model.fit(*train_classes)

    # Evaluate on training data
    print("\n" + "=" * 80)
    print("TRAINING SET EVALUATION")
    print("=" * 80)

    X_train = np.vstack(train_classes)
    y_train = np.concatenate([
        np.full(len(train_classes[k]), k+1) for k in range(n_classes)
    ])

    y_train_pred = model.predict(X_train)
    train_metrics = evaluate_multiclass(y_train, y_train_pred, n_classes=n_classes)

    print(f"\nTraining Accuracy by Class:")
    for k in range(1, n_classes + 1):
        print(f"  Class {k}: {train_metrics[f'class_{k}_accuracy']:.4f}")
    print(f"Total Training Accuracy: {train_metrics['total_accuracy']:.4f}")

    if log_file:
        log_file.write(f"\nTraining Accuracy by Class:\n")
        for k in range(1, n_classes + 1):
            log_file.write(f"  Class {k}: {train_metrics[f'class_{k}_accuracy']:.4f}\n")
        log_file.write(f"Total Training Accuracy: {train_metrics['total_accuracy']:.4f}\n")

    # Evaluate on test data
    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)

    y_test_pred = model.predict(X_test)
    test_metrics = evaluate_multiclass(y_test, y_test_pred, n_classes=n_classes)

    print(f"\nTest Accuracy by Class:")
    for k in range(1, n_classes + 1):
        print(f"  Class {k}: {test_metrics[f'class_{k}_accuracy']:.4f}")
    print(f"Total Test Accuracy: {test_metrics['total_accuracy']:.4f}")

    if log_file:
        log_file.write(f"\nTest Accuracy by Class:\n")
        for k in range(1, n_classes + 1):
            log_file.write(f"  Class {k}: {test_metrics[f'class_{k}_accuracy']:.4f}\n")
        log_file.write(f"Total Test Accuracy: {test_metrics['total_accuracy']:.4f}\n")

    # Model summary
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)

    summary = model.get_model_summary()

    for k in range(1, n_classes):
        clf_key = f'h{k}'
        clf_summary = summary['classifiers'][clf_key]
        print(f"\n{clf_summary['description']}:")
        print(f"  Objective: {clf_summary['objective_value']:.6f}")
        print(f"  Selected features: {clf_summary['n_selected_features']}/{n_features}")
        print(f"  L1 norm: {clf_summary['l1_norm']:.6f}")
        print(f"  Positive accuracy lb: {clf_summary['positive_class_accuracy_lb']:.4f}")
        print(f"  Negative accuracy lb: {clf_summary['negative_class_accuracy_lb']:.4f}")

        if log_file:
            log_file.write(f"\n{clf_summary['description']}:\n")
            log_file.write(f"  Objective: {clf_summary['objective_value']:.6f}\n")
            log_file.write(f"  Selected features: {clf_summary['n_selected_features']}/{n_features}\n")
            log_file.write(f"  L1 norm: {clf_summary['l1_norm']:.6f}\n")
            log_file.write(f"  Positive accuracy lb: {clf_summary['positive_class_accuracy_lb']:.4f}\n")
            log_file.write(f"  Negative accuracy lb: {clf_summary['negative_class_accuracy_lb']:.4f}\n")

    return {
        'train_metrics': train_metrics,
        'test_metrics': test_metrics,
        'model_summary': summary
    }


def main():
    """Main execution."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filepath = f"results/test3_new_datasets_{timestamp}.log"

    # Create results directory if it doesn't exist
    os.makedirs("results", exist_ok=True)

    # Test parameters
    params = {
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800,  # 30 minutes per classifier
        'mip_gap': 1e-4,
        'verbose': True
    }

    # Dataset configurations
    base_path = os.path.expanduser("~/Developer/NSVORA/datasets/ordinal_regression_new")
    datasets = [
        ("bostonhousing_ord", f"{base_path}/bostonhousing_ord.csv", "5 classes"),
        ("cement_strength", f"{base_path}/cement_strength.csv", "5 classes"),
        ("stock_ord", f"{base_path}/stock_ord.csv", "5 classes"),
        ("skill", f"{base_path}/skill.csv", "7 classes"),
        ("californiahousing", f"{base_path}/californiahousing.csv", "6 classes"),
    ]

    print("\n" + "=" * 80)
    print("Test3 Strategy Validation on New Ordinal Regression Datasets")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Log file: {log_filepath}")
    print(f"\nDatasets to test:")
    for name, path, info in datasets:
        print(f"  - {name} ({info})")
    print("\nParameters:")
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Open log file
    with open(log_filepath, 'w') as log_file:
        log_file.write("=" * 80 + "\n")
        log_file.write("Test3 Strategy Validation on New Ordinal Regression Datasets\n")
        log_file.write("=" * 80 + "\n")
        log_file.write(f"Timestamp: {timestamp}\n")
        log_file.write(f"\nParameters:\n")
        for key, value in params.items():
            log_file.write(f"  {key}: {value}\n")

        results = {}

        # Run tests
        for dataset_name, filepath, info in datasets:
            try:
                results[dataset_name] = run_test3_on_dataset(
                    dataset_name,
                    filepath,
                    params,
                    log_file
                )
                log_file.flush()  # Ensure immediate write
            except Exception as e:
                error_msg = f"ERROR: Failed to process {dataset_name}: {str(e)}"
                print(f"\n{error_msg}")
                if log_file:
                    log_file.write(f"\n{error_msg}\n")
                log_file.flush()

        # Summary
        print("\n" + "=" * 80)
        print("FINAL SUMMARY")
        print("=" * 80)

        log_file.write("\n" + "=" * 80 + "\n")
        log_file.write("FINAL SUMMARY\n")
        log_file.write("=" * 80 + "\n")

        for dataset_name, result in results.items():
            test_acc = result['test_metrics']['total_accuracy']
            train_acc = result['train_metrics']['total_accuracy']

            summary_line = f"{dataset_name}: Train={train_acc:.4f}, Test={test_acc:.4f}"
            print(summary_line)
            log_file.write(summary_line + "\n")

    print(f"\nResults saved to: {log_filepath}")


if __name__ == "__main__":
    main()
