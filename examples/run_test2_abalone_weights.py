#!/usr/bin/env python3
"""Test2 Strategy on Abalone dataset - Output complete weights and bias."""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm import HierarchicalCESVM, get_default_params
from hcesvm.utils import load_multiclass_data, evaluate_hierarchical_model, print_evaluation_results

def main():
    # Load data from NSVORA project
    data_file = Path.home() / "Developer/NSVORA/datasets/primary/abalone/abalone_split.xlsx"

    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("Please ensure NSVORA project is available")
        return

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print("Hierarchical CE-SVM on Abalone Dataset")
    print("Strategy: Test2 (Ordinal-based with Test2 Rule)")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)
    print(f"\nLoading data from: {data_file}")
    print("\nDataset Info:")
    print("  - Extremely imbalanced (52.1:1 ratio)")
    print("  - Class 1 (Young): 59 samples (1.8%)")
    print("  - Class 2 (Adult): 3073 samples (92.0%)")
    print("  - Class 3 (Old): 209 samples (6.3%)")
    print("  - Features: 9")
    print("  - Total samples: 3341")
    print()

    # Load 3-class training data
    X_classes, n_classes_list, n_features = load_multiclass_data(str(data_file))
    X1, X2, X3 = X_classes

    print(f"Loaded training class sizes: [{len(X1)}, {len(X2)}, {len(X3)}]")
    print()

    # Get default parameters
    params = get_default_params()
    params['verbose'] = True
    params['time_limit'] = 1800  # 30 minutes per classifier (larger dataset)

    print("=" * 80)
    print("Model Parameters")
    print("=" * 80)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # Train hierarchical classifier with Test2 strategy
    print("=" * 80)
    print("Training Hierarchical Classifier - Test2 Strategy")
    print("=" * 80)
    print("\nTest2 Strategy:")
    print("  - Split based on minority position with ordinal labeling")
    print("  - If majority = Class 2 (not at edge):")
    print("    - H1: Use accuracy mode to avoid maximizing majority class")
    print("    - H2: Use accuracy mode to avoid maximizing majority class")
    print("  - Otherwise: Use standard 'both' mode")
    print()
    print("This may take a while due to dataset size and imbalance...")
    print()

    train_start = datetime.now()

    hc = HierarchicalCESVM(cesvm_params=params, strategy="test2")
    hc.fit(X1, X2, X3)

    train_end = datetime.now()
    train_duration = train_end - train_start

    print(f"\nTraining completed in: {train_duration}")
    print()

    # Get model summary
    summary = hc.get_model_summary()
    print("=" * 80)
    print("Model Summary")
    print("=" * 80)
    print(f"Status: {summary['status']}")
    print(f"Strategy: {summary['strategy']}")
    print(f"Features: {summary['n_features']}")

    if 'class_roles' in summary:
        print(f"\nClass Roles:")
        print(f"  Majority: Class {summary['class_roles']['majority']}")
        print(f"  Medium:   Class {summary['class_roles']['medium']}")
        print(f"  Minority: Class {summary['class_roles']['minority']}")

    # Extract accuracy mode settings
    h1_accuracy_mode = getattr(hc.h1, 'accuracy_mode', 'both')
    h2_accuracy_mode = getattr(hc.h2, 'accuracy_mode', 'both')
    test2_rule_applied = summary['class_roles']['majority'] == 2 if 'class_roles' in summary else False

    print(f"\nTest2 Rule Applied: {'Yes' if test2_rule_applied else 'No'}")
    print(f"Accuracy Modes:")
    print(f"  H1: {h1_accuracy_mode}")
    print(f"  H2: {h2_accuracy_mode}")

    print(f"\nClassifier 1 (H1): {summary['h1']['description']}")
    h1 = summary['h1']
    print(f"  Objective: {h1['objective_value']:.6f}")
    print(f"  Selected Features: {h1['n_selected_features']}/{summary['n_features']}")
    print(f"  L1 Norm: {h1['l1_norm']:.6f}")
    print(f"  Positive accuracy lb: {h1['positive_class_accuracy_lb']:.4f}")
    print(f"  Negative accuracy lb: {h1['negative_class_accuracy_lb']:.4f}")
    if 'mip_gap' in h1:
        print(f"  MIP Gap: {h1['mip_gap']:.6e}")

    print(f"\nClassifier 2 (H2): {summary['h2']['description']}")
    h2 = summary['h2']
    print(f"  Objective: {h2['objective_value']:.6f}")
    print(f"  Selected Features: {h2['n_selected_features']}/{summary['n_features']}")
    print(f"  L1 Norm: {h2['l1_norm']:.6f}")
    print(f"  Positive accuracy lb: {h2['positive_class_accuracy_lb']:.4f}")
    print(f"  Negative accuracy lb: {h2['negative_class_accuracy_lb']:.4f}")
    if 'mip_gap' in h2:
        print(f"  MIP Gap: {h2['mip_gap']:.6e}")

    # Output complete H1 and H2 Weights and Bias
    print("\n" + "=" * 80)
    print("H1 Complete Weights and Bias")
    print("=" * 80)
    print(f"Description: {summary['h1']['description']}")
    print(f"\nIntercept (b): {hc.h1.intercept:.10f}")
    print(f"\nWeight vector (w) - {len(hc.h1.weights)} features:")
    for i, w in enumerate(hc.h1.weights):
        print(f"  w[{i}] = {w:.10f}")
    print(f"\nL1 Norm: {np.sum(np.abs(hc.h1.weights)):.10f}")

    print("\n" + "=" * 80)
    print("H2 Complete Weights and Bias")
    print("=" * 80)
    print(f"Description: {summary['h2']['description']}")
    print(f"\nIntercept (b): {hc.h2.intercept:.10f}")
    print(f"\nWeight vector (w) - {len(hc.h2.weights)} features:")
    for i, w in enumerate(hc.h2.weights):
        print(f"  w[{i}] = {w:.10f}")
    print(f"\nL1 Norm: {np.sum(np.abs(hc.h2.weights)):.10f}")

    # Training Set Evaluation
    print("\n" + "=" * 80)
    print("Training Set Evaluation")
    print("=" * 80)

    X_train_all = np.vstack([X1, X2, X3])
    y_train_all = np.concatenate([
        np.ones(len(X1), dtype=int),
        2 * np.ones(len(X2), dtype=int),
        3 * np.ones(len(X3), dtype=int)
    ])

    y_train_pred = hc.predict(X_train_all)
    train_results = evaluate_hierarchical_model(y_train_all, y_train_pred)
    print_evaluation_results(train_results)

    # Test Set Evaluation
    print("\n" + "=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)

    try:
        # Load test data (Test sheet has different structure: skiprows=4)
        X_test_classes, _, _ = load_multiclass_data(str(data_file), sheet_name='Test', skiprows=4)
        X1_test, X2_test, X3_test = X_test_classes

        print(f"Loaded test class sizes: [{len(X1_test)}, {len(X2_test)}, {len(X3_test)}]")
        print()

        # Combine test data with labels
        X_test_all = np.vstack([X1_test, X2_test, X3_test])
        y_test_all = np.concatenate([
            np.ones(len(X1_test), dtype=int),
            2 * np.ones(len(X2_test), dtype=int),
            3 * np.ones(len(X3_test), dtype=int)
        ])

        y_test_pred = hc.predict(X_test_all)
        test_results = evaluate_hierarchical_model(y_test_all, y_test_pred)
        print_evaluation_results(test_results)

    except Exception as e:
        print(f"Warning: Could not load or evaluate test data: {e}")
        print("Skipping test set evaluation.")

    print("\n" + "=" * 80)
    print("Test Complete!")
    print("=" * 80)
    print(f"Training Duration: {train_duration}")
    print(f"Training Accuracy: {train_results['total_accuracy']:.4f}")
    if 'test_results' in locals():
        print(f"Test Accuracy: {test_results['total_accuracy']:.4f}")
    print("=" * 80)

if __name__ == "__main__":
    main()
