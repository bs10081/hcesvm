#!/usr/bin/env python3
"""Test Hierarchical CE-SVM on Abalone dataset with Multiple Filter strategy."""

import sys
from pathlib import Path

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

    print("=" * 60)
    print("Hierarchical CE-SVM on Abalone Dataset")
    print("Strategy: Multiple Filter (Minority Class First)")
    print("=" * 60)
    print(f"\nLoading data from: {data_file}")
    print("\nDataset Info:")
    print("  - Extremely imbalanced (52.1:1 ratio)")
    print("  - Class 1 (Young): 59 samples (1.8%) - MINORITY")
    print("  - Class 2 (Adult): 3073 samples (92.0%)")
    print("  - Class 3 (Old): 209 samples (6.3%)")
    print("  - Features: 9")
    print("  - Total samples: 3341")
    print()

    # Load 3-class data
    X_classes, n_classes_list, n_features = load_multiclass_data(str(data_file))
    X1, X2, X3 = X_classes

    print(f"Loaded class sizes: [{len(X1)}, {len(X2)}, {len(X3)}]")
    print()

    # Get default parameters
    params = get_default_params()
    params['verbose'] = True
    params['time_limit'] = 1800  # 30 minutes per classifier (larger dataset)

    print("\n" + "=" * 60)
    print("Model Parameters")
    print("=" * 60)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # Train hierarchical classifier with Multiple Filter strategy
    print("\n" + "=" * 60)
    print("Training Hierarchical Classifier - Multiple Filter")
    print("=" * 60)
    print("\nHierarchy Structure (Minority-First):")
    print("  H1: Class 1 (minority) vs {2,3} - Separate young abalone first")
    print("  H2: Class {1,2} vs Class 3 - Then separate old from others")
    print()
    print("This may take a while due to dataset size and imbalance...")
    print()

    hc = HierarchicalCESVM(cesvm_params=params, strategy="multiple_filter")
    hc.fit(X1, X2, X3)

    # Get model summary
    summary = hc.get_model_summary()
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Status: {summary['status']}")
    print(f"Strategy: {summary['strategy']}")
    print(f"Features: {summary['n_features']}")

    print(f"\nClassifier 1 (H1): {summary['h1']['description']}")
    h1 = summary['h1']
    print(f"  Objective: {h1['objective_value']:.6f}")
    print(f"  Selected Features: {h1['n_selected_features']}/{summary['n_features']}")
    print(f"  L1 Norm: {h1['l1_norm']:.6f}")
    if 'mip_gap' in h1:
        print(f"  MIP Gap: {h1['mip_gap']:.6e}")

    print(f"\nClassifier 2 (H2): {summary['h2']['description']}")
    h2 = summary['h2']
    print(f"  Objective: {h2['objective_value']:.6f}")
    print(f"  Selected Features: {h2['n_selected_features']}/{summary['n_features']}")
    print(f"  L1 Norm: {h2['l1_norm']:.6f}")
    if 'mip_gap' in h2:
        print(f"  MIP Gap: {h2['mip_gap']:.6e}")

    # Output H1 and H2 Weights
    print("\n" + "=" * 60)
    print("H1 Weights (Class 1 vs {2,3})")
    print("=" * 60)
    print(f"Intercept (b): {hc.h1.intercept:.6f}")
    print("Weight vector (w):")
    print(hc.h1.weights)

    print("\n" + "=" * 60)
    print("H2 Weights (Class {1,2} vs Class 3)")
    print("=" * 60)
    print(f"Intercept (b): {hc.h2.intercept:.6f}")
    print("Weight vector (w):")
    print(hc.h2.weights)

    # Predict on training data
    print("\n" + "=" * 60)
    print("Training Set Evaluation")
    print("=" * 60)

    # Combine all data with labels
    import numpy as np
    X_all = np.vstack([X1, X2, X3])
    y_all = np.concatenate([
        np.ones(len(X1), dtype=int),
        2 * np.ones(len(X2), dtype=int),
        3 * np.ones(len(X3), dtype=int)
    ])

    y_pred = hc.predict(X_all)
    results = evaluate_hierarchical_model(y_all, y_pred)
    print_evaluation_results(results)

    # Evaluate on test set
    print("\n" + "=" * 60)
    print("Test Set Evaluation")
    print("=" * 60)

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

    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
