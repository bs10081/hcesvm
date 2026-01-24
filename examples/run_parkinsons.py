#!/usr/bin/env python3
"""Test Binary CE-SVM on Parkinsons dataset."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm import BinaryCESVM, get_default_params
from hcesvm.utils import load_parkinsons_data, calculate_binary_metrics

def main():
    # Load data
    data_file = Path(__file__).parent.parent / "data/parkinsons/Parkinsons_CESVM.xlsx"
    
    print("=" * 60)
    print("Binary CE-SVM on Parkinsons Dataset")
    print("=" * 60)
    print(f"\nLoading data from: {data_file}")
    print()
    
    X, y, n_features = load_parkinsons_data(str(data_file))
    
    # Get default parameters
    params = get_default_params()
    params['verbose'] = True
    params['mip_gap'] = 0.05  # Allow 5% gap for faster convergence
    params['time_limit'] = 300  # Reduce to 5 minutes
    
    print("\n" + "=" * 60)
    print("Model Parameters")
    print("=" * 60)
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Train model
    print("\n" + "=" * 60)
    print("Training Binary CE-SVM")
    print("=" * 60)
    
    model = BinaryCESVM(**params)
    model.fit(X, y)
    
    # Get solution summary
    summary = model.get_solution_summary()
    print("\n" + "=" * 60)
    print("Solution Summary")
    print("=" * 60)
    print(f"Status: {summary['status']}")
    print(f"Objective Value: {summary['objective_value']:.6f}")
    print(f"L1 Norm: {summary['l1_norm']:.6f}")
    print(f"Selected Features: {summary['n_selected_features']}/{n_features}")
    print(f"Selected Feature Indices: {summary['selected_feature_indices']}")
    print(f"Positive Class Accuracy LB: {summary['positive_class_accuracy_lb']:.4f}")
    print(f"Negative Class Accuracy LB: {summary['negative_class_accuracy_lb']:.4f}")
    print(f"Intercept: {summary['intercept']:.6f}")
    
    # Predict on training data
    print("\n" + "=" * 60)
    print("Training Set Evaluation")
    print("=" * 60)
    
    y_pred = model.predict(X)
    metrics = calculate_binary_metrics(y, y_pred)
    
    print(f"Overall Accuracy: {metrics['accuracy']:.4f}")
    print(f"TPR (Sensitivity): {metrics['TPR']:.4f}")
    print(f"TNR (Specificity): {metrics['TNR']:.4f}")
    
    print("\n" + "=" * 60)
    print("Test Complete!")
    print("=" * 60)

if __name__ == "__main__":
    main()
