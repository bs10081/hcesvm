#!/usr/bin/env python3
"""Test Hierarchical CE-SVM on Balance dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm import HierarchicalCESVM, get_default_params
from hcesvm.utils import load_multiclass_data, evaluate_hierarchical_model, print_evaluation_results

def main():
    # Load data from NSVORA project
    data_file = Path.home() / "Developer/NSVORA/Archive/balance_split.xlsx"
    
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        print("Please ensure NSVORA project is available")
        return
    
    print("=" * 60)
    print("Hierarchical CE-SVM on Balance Dataset")
    print("=" * 60)
    print(f"\nLoading data from: {data_file}")
    print()
    
    # Load 3-class data
    X_classes, n_classes_list, n_features = load_multiclass_data(str(data_file))
    X1, X2, X3 = X_classes
    
    # Get default parameters
    params = get_default_params()
    params['verbose'] = True
    params['time_limit'] = 300  # 5 minutes per classifier
    
    print("\n" + "=" * 60)
    print("Model Parameters")
    print("=" * 60)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()
    
    # Train hierarchical classifier
    hc = HierarchicalCESVM(cesvm_params=params)
    hc.fit(X1, X2, X3)
    
    # Get model summary
    summary = hc.get_model_summary()
    print("\n" + "=" * 60)
    print("Model Summary")
    print("=" * 60)
    print(f"Status: {summary['status']}")
    print(f"Features: {summary['n_features']}")
    
    print("\nClassifier 1 (H1):")
    h1 = summary['h1']
    print(f"  Objective: {h1['objective_value']:.6f}")
    print(f"  Selected Features: {h1['n_selected_features']}/{summary['n_features']}")
    print(f"  L1 Norm: {h1['l1_norm']:.6f}")
    
    print("\nClassifier 2 (H2):")
    h2 = summary['h2']
    print(f"  Objective: {h2['objective_value']:.6f}")
    print(f"  Selected Features: {h2['n_selected_features']}/{summary['n_features']}")
    print(f"  L2 Norm: {h2['l1_norm']:.6f}")
    
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
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
