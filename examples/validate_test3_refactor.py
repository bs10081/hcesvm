#!/usr/bin/env python3
"""
Validate Test3 Strategy Refactor

Tests the refactored Test3 strategy (fixed grouping with balanced weighting)
on contraceptive and thyroid datasets.
"""

import sys
import os
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from hcesvm.models.hierarchical import HierarchicalCESVM
from hcesvm.utils.data_loader import load_split_data
from hcesvm.utils.evaluator import evaluate_hierarchical_model


def test_dataset(dataset_name: str, file_path: str):
    """Test Test3 strategy on a single dataset."""
    print("=" * 80)
    print(f"Testing Dataset: {dataset_name}")
    print("=" * 80)
    print(f"File: {file_path}")
    print(f"Strategy: test3 (Fixed grouping with balanced class weighting)")
    print()

    # Load data
    print("Loading data...")
    try:
        data = load_split_data(file_path)
    except Exception as e:
        print(f"❌ Error loading data: {e}")
        return None

    X1_train = data['X1_train']
    X2_train = data['X2_train']
    X3_train = data['X3_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"✓ Data loaded successfully")
    print(f"  Training: Class 1={len(X1_train)}, Class 2={len(X2_train)}, Class 3={len(X3_train)}")
    print(f"  Testing: {len(X_test)} samples")
    print()

    # Train model
    print("Training model with Test3 strategy...")
    print("-" * 80)

    model = HierarchicalCESVM(
        cesvm_params={
            'C_hyper': 1.0,
            'M': 1000.0,
            'time_limit': 1800,  # 30 minutes per classifier
            'mip_gap': 1e-4,
            'verbose': True
        },
        strategy='test3'
    )

    try:
        model.fit(X1_train, X2_train, X3_train)
        print("-" * 80)
        print("✓ Model trained successfully")
    except Exception as e:
        print(f"❌ Error training model: {e}")
        return None

    print()

    # Evaluate
    print("Evaluating model...")
    print("-" * 80)

    try:
        # Training accuracy
        X_train_all = np.vstack([X1_train, X2_train, X3_train])
        y_train_all = np.concatenate([
            np.ones(len(X1_train)),
            2 * np.ones(len(X2_train)),
            3 * np.ones(len(X3_train))
        ])

        train_predictions = model.predict(X_train_all)
        train_results = evaluate_hierarchical_model(y_train_all, train_predictions)

        # Testing accuracy
        test_predictions = model.predict(X_test)
        test_results = evaluate_hierarchical_model(y_test, test_predictions)

        print("TRAINING RESULTS:")
        print(f"  Overall Accuracy: {train_results['total_accuracy']:.4f}")
        for class_name, acc in train_results['per_class_accuracy'].items():
            class_num = int(class_name.split()[1])
            count = train_results['class_distribution'][class_name]
            print(f"  {class_name} Accuracy: {acc:.4f} ({count} samples)")
        print()

        print("TESTING RESULTS:")
        print(f"  Overall Accuracy: {test_results['total_accuracy']:.4f}")
        for class_name, acc in test_results['per_class_accuracy'].items():
            class_num = int(class_name.split()[1])
            count = test_results['class_distribution'][class_name]
            print(f"  {class_name} Accuracy: {acc:.4f} ({count} samples)")
        print()

        # Model summary
        summary = model.get_model_summary()
        print("MODEL SUMMARY:")
        print(f"  Strategy: {summary['strategy']}")
        print(f"  H1 Description: {summary['h1']['description']}")
        print(f"  H2 Description: {summary['h2']['description']}")
        print(f"  H1 Selected Features: {summary['h1']['n_selected_features']}/{summary['n_features']}")
        print(f"  H2 Selected Features: {summary['h2']['n_selected_features']}/{summary['n_features']}")
        print()

        print("H1 WEIGHTS:")
        print(f"  w = {model.h1.weights}")
        print(f"  b = {model.h1.intercept}")
        print()

        print("H2 WEIGHTS:")
        print(f"  w = {model.h2.weights}")
        print(f"  b = {model.h2.intercept}")

        print("-" * 80)
        print("✓ Evaluation completed")
        print()

        return {
            'dataset': dataset_name,
            'train_results': train_results,
            'test_results': test_results,
            'summary': summary
        }

    except Exception as e:
        print(f"❌ Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Run validation tests."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("\n" + "=" * 80)
    print("TEST3 STRATEGY REFACTOR VALIDATION")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Testing: Contraceptive, Thyroid")
    print(f"Strategy: test3 (Fixed grouping with balanced class weighting)")
    print("=" * 80)
    print()

    # Dataset paths
    datasets = {
        'Contraceptive': os.path.expanduser('~/Developer/NSVORA/Archive/contraceptive_split.xlsx'),
        'Thyroid': os.path.expanduser('~/Developer/NSVORA/Archive/thyroid_split.xlsx')
    }

    results = []

    for dataset_name, file_path in datasets.items():
        result = test_dataset(dataset_name, file_path)
        if result:
            results.append(result)
        print()

    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    if len(results) == 0:
        print("❌ No results - all tests failed")
        return

    print(f"Successful tests: {len(results)}/{len(datasets)}")
    print()

    for result in results:
        print(f"{result['dataset']}:")
        print(f"  Training Accuracy: {result['train_results']['total_accuracy']:.4f}")
        print(f"  Testing Accuracy:  {result['test_results']['total_accuracy']:.4f}")
        print(f"  Strategy: {result['summary']['strategy']}")
        print()

    print("=" * 80)
    print("✓ Validation completed")
    print("=" * 80)


if __name__ == '__main__':
    main()
