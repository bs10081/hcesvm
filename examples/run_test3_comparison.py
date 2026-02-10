#!/usr/bin/env python3
"""
Compare Test2 vs Test3 strategies on a sample dataset.

Test2: Dynamic removal of accuracy terms based on majority position
Test3: Sample-weighted accuracy terms using inverse of sample count
"""

import sys
import os
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.hcesvm import HierarchicalCESVM
from src.hcesvm.utils.data_loader import load_split_data
from src.hcesvm.utils.evaluator import evaluate_classifier


def run_comparison(dataset_path: str, dataset_name: str):
    """Compare Test2 vs Test3 strategies on a dataset."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("=" * 80)
    print(f"Test2 vs Test3 Comparison: {dataset_name}")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    data = load_split_data(dataset_path)

    X1_train, X2_train, X3_train = data['train']
    X1_test, X2_test, X3_test = data['test']

    print(f"Training samples: Class 1={len(X1_train)}, Class 2={len(X2_train)}, Class 3={len(X3_train)}")
    print(f"Testing samples: Class 1={len(X1_test)}, Class 2={len(X2_test)}, Class 3={len(X3_test)}")

    # Calculate imbalance ratio
    train_counts = [len(X1_train), len(X2_train), len(X3_train)]
    max_count = max(train_counts)
    min_count = min(train_counts)
    imbalance_ratio = max_count / min_count
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    # Parameters
    cesvm_params = {
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800,
        'mip_gap': 1e-4,
        'verbose': False
    }

    results = {}

    # Test both strategies
    for strategy in ['test2', 'test3']:
        print("\n" + "=" * 80)
        print(f"Strategy: {strategy.upper()}")
        print("=" * 80)

        # Train model
        model = HierarchicalCESVM(
            cesvm_params=cesvm_params,
            strategy=strategy
        )

        print("\nTraining...")
        model.fit(X1_train, X2_train, X3_train)

        # Get model summary
        summary = model.get_model_summary()

        # Evaluate on training set
        print("\n" + "-" * 60)
        print("Training Set Evaluation:")
        print("-" * 60)
        train_metrics = evaluate_classifier(
            model,
            X1_train, X2_train, X3_train,
            dataset_name="Training"
        )

        # Evaluate on test set
        print("\n" + "-" * 60)
        print("Test Set Evaluation:")
        print("-" * 60)
        test_metrics = evaluate_classifier(
            model,
            X1_test, X2_test, X3_test,
            dataset_name="Testing"
        )

        # Store results
        results[strategy] = {
            'train': train_metrics,
            'test': test_metrics,
            'summary': summary
        }

    # Compare results
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\nDataset: {dataset_name}")
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")

    print("\n" + "-" * 60)
    print("Training Set Accuracy:")
    print("-" * 60)
    for strategy in ['test2', 'test3']:
        metrics = results[strategy]['train']
        print(f"\n{strategy.upper()}:")
        print(f"  Class 1: {metrics['class1_accuracy']:.4f}")
        print(f"  Class 2: {metrics['class2_accuracy']:.4f}")
        print(f"  Class 3: {metrics['class3_accuracy']:.4f}")
        print(f"  Total:   {metrics['total_accuracy']:.4f}")

    print("\n" + "-" * 60)
    print("Test Set Accuracy:")
    print("-" * 60)
    for strategy in ['test2', 'test3']:
        metrics = results[strategy]['test']
        print(f"\n{strategy.upper()}:")
        print(f"  Class 1: {metrics['class1_accuracy']:.4f}")
        print(f"  Class 2: {metrics['class2_accuracy']:.4f}")
        print(f"  Class 3: {metrics['class3_accuracy']:.4f}")
        print(f"  Total:   {metrics['total_accuracy']:.4f}")

    print("\n" + "-" * 60)
    print("Model Complexity:")
    print("-" * 60)
    for strategy in ['test2', 'test3']:
        summary = results[strategy]['summary']
        h1 = summary['h1']
        h2 = summary['h2']
        print(f"\n{strategy.upper()}:")
        print(f"  H1 selected features: {h1['n_selected_features']}")
        print(f"  H1 L1 norm: {h1['l1_norm']:.6f}")
        print(f"  H2 selected features: {h2['n_selected_features']}")
        print(f"  H2 L1 norm: {h2['l1_norm']:.6f}")

    print("\n" + "=" * 80)
    print("Comparison Complete!")
    print("=" * 80)

    return results


if __name__ == "__main__":
    # Example: Abalone dataset
    abalone_path = os.path.expanduser("~/Developer/NSVORA/datasets/primary/abalone/abalone_split.xlsx")

    if os.path.exists(abalone_path):
        results = run_comparison(abalone_path, "Abalone")
    else:
        print(f"Dataset not found: {abalone_path}")
        print("Please update the dataset path in the script.")
