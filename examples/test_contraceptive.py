#!/usr/bin/env python3
"""
Test Contraceptive Dataset with Test3 Strategy
"""

import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from hcesvm.models.hierarchical import HierarchicalCESVM
from hcesvm.utils.data_loader import load_split_data
from hcesvm.utils.evaluator import evaluate_hierarchical_model, print_evaluation_results


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print("CONTRACEPTIVE DATASET TEST - TEST3 STRATEGY")
    print("=" * 80)
    print(f"Timestamp: {timestamp}")
    print(f"Dataset: Contraceptive")
    print(f"Strategy: test3 (Fixed grouping with balanced class weighting)")
    print(f"C_hyper: 1.0")
    print("=" * 80)
    print()

    # Load data
    file_path = os.path.expanduser('~/Developer/NSVORA/Archive/contraceptive_split.xlsx')
    print(f"Loading data from: {file_path}")
    data = load_split_data(file_path)

    X1_train = data['X1_train']
    X2_train = data['X2_train']
    X3_train = data['X3_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"\nTraining data:")
    print(f"  Class 1: {len(X1_train)} samples")
    print(f"  Class 2: {len(X2_train)} samples")
    print(f"  Class 3: {len(X3_train)} samples")
    print(f"  Features: {X1_train.shape[1]}")
    print(f"\nTesting data: {len(X_test)} samples")
    print()

    # Train model
    print("=" * 80)
    print("Training Test3 Model (C_hyper=1.0)")
    print("=" * 80)

    model = HierarchicalCESVM(
        cesvm_params={
            'C_hyper': 1.0,
            'M': 1000.0,
            'time_limit': 1800,
            'mip_gap': 1e-4,
            'verbose': True
        },
        strategy='test3'
    )

    model.fit(X1_train, X2_train, X3_train)

    print("\n" + "=" * 80)
    print("MODEL TRAINED SUCCESSFULLY")
    print("=" * 80)

    # Evaluate on training data
    X_train_all = np.vstack([X1_train, X2_train, X3_train])
    y_train_all = np.concatenate([
        np.ones(len(X1_train)),
        2 * np.ones(len(X2_train)),
        3 * np.ones(len(X3_train))
    ])

    train_predictions = model.predict(X_train_all)
    train_results = evaluate_hierarchical_model(y_train_all, train_predictions)

    print("\n" + "=" * 80)
    print("TRAINING SET EVALUATION")
    print("=" * 80)
    print_evaluation_results(train_results)

    # Evaluate on test data
    test_predictions = model.predict(X_test)
    test_results = evaluate_hierarchical_model(y_test, test_predictions)

    print("\n" + "=" * 80)
    print("TEST SET EVALUATION")
    print("=" * 80)
    print_evaluation_results(test_results)

    # Model summary
    summary = model.get_model_summary()
    print("\n" + "=" * 80)
    print("MODEL SUMMARY")
    print("=" * 80)
    print(f"Strategy: {summary['strategy']}")
    print(f"\nH1 Classifier:")
    print(f"  Description: {summary['h1']['description']}")
    print(f"  Weights: {model.h1.weights}")
    print(f"  Intercept: {model.h1.intercept}")
    print(f"  Selected Features: {summary['h1']['n_selected_features']}/{summary['n_features']}")
    print(f"  L1 Norm: {summary['h1']['l1_norm']:.6f}")
    print(f"  Positive Class Accuracy LB: {summary['h1']['positive_class_accuracy_lb']:.4f}")
    print(f"  Negative Class Accuracy LB: {summary['h1']['negative_class_accuracy_lb']:.4f}")

    print(f"\nH2 Classifier:")
    print(f"  Description: {summary['h2']['description']}")
    print(f"  Weights: {model.h2.weights}")
    print(f"  Intercept: {model.h2.intercept}")
    print(f"  Selected Features: {summary['h2']['n_selected_features']}/{summary['n_features']}")
    print(f"  L1 Norm: {summary['h2']['l1_norm']:.6f}")
    print(f"  Positive Class Accuracy LB: {summary['h2']['positive_class_accuracy_lb']:.4f}")
    print(f"  Negative Class Accuracy LB: {summary['h2']['negative_class_accuracy_lb']:.4f}")

    print("\n" + "=" * 80)
    print("TEST COMPLETED")
    print("=" * 80)
    print(f"Training Accuracy: {train_results['total_accuracy']:.4f}")
    print(f"Testing Accuracy: {test_results['total_accuracy']:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
