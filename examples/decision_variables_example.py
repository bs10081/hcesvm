#!/usr/bin/env python3
"""
Example: Accessing and Analyzing Decision Variables in CE-SVM

This script demonstrates how to access and analyze all decision variables
from a trained CE-SVM model, including slack variables, indicator variables,
and support vector information.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
from hcesvm.models.binary_cesvm import BinaryCESVM


def create_sample_data():
    """Create sample data with some overlap for demonstration."""
    np.random.seed(42)
    n_pos, n_neg = 50, 50
    n_features = 3

    # Create two classes with some overlap
    X_pos = np.random.randn(n_pos, n_features) + 1.5
    X_neg = np.random.randn(n_neg, n_features) - 1.5

    # Add noise to create overlap (making some samples hard to classify)
    X_pos[:8] -= 3.0  # Move some positive samples into negative region
    X_neg[:8] += 3.0  # Move some negative samples into positive region

    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(n_pos), -np.ones(n_neg)])

    # Shuffle
    indices = np.random.permutation(len(y))
    return X[indices], y[indices]


def main():
    print("=" * 70)
    print("CE-SVM Decision Variables Analysis Example")
    print("=" * 70)
    print()

    # Generate and train model
    X, y = create_sample_data()
    n_pos = np.sum(y == 1)
    n_neg = np.sum(y == -1)

    print(f"Dataset Info:")
    print(f"  Samples: {len(y)} (Positive: {n_pos}, Negative: {n_neg})")
    print(f"  Features: {X.shape[1]}")
    print()

    print("Training CE-SVM model...")
    model = BinaryCESVM(
        C_hyper=1.0,
        time_limit=60,
        verbose=False
    )
    model.fit(X, y)
    print("✓ Model training complete")
    print()

    # =========================================================================
    # 1. Basic Solution Summary
    # =========================================================================
    print("=" * 70)
    print("1. Basic Solution Summary")
    print("=" * 70)
    summary = model.get_solution_summary()
    print(f"Objective Value:        {summary['objective_value']:.6f}")
    print(f"L1 Norm (||w||_1):      {summary['l1_norm']:.6f}")
    print(f"Intercept (b):          {summary['intercept']:.6f}")
    print(f"Selected Features:      {summary['n_selected_features']}/{X.shape[1]}")
    print(f"Feature Indices:        {summary['selected_feature_indices']}")
    print(f"Pos Class Acc LB (l+):  {summary['positive_class_accuracy_lb']:.4f}")
    print(f"Neg Class Acc LB (l-):  {summary['negative_class_accuracy_lb']:.4f}")
    print(f"Support Vectors:        {summary['n_support_vectors']}")
    print(f"Margin Errors:          {summary['n_margin_errors']}")
    print()

    # =========================================================================
    # 2. Slack Variables (ξ)
    # =========================================================================
    print("=" * 70)
    print("2. Slack Variables (ξ)")
    print("=" * 70)
    ksi = model.get_slack_variables()
    print(f"Slack variables measure constraint violations for each sample.")
    print(f"  Total samples:          {len(ksi)}")
    print(f"  Min slack:              {ksi.min():.6f}")
    print(f"  Max slack:              {ksi.max():.6f}")
    print(f"  Mean slack:             {ksi.mean():.6f}")
    print(f"  Samples with ξ > 0:     {np.sum(ksi > 1e-6)} (support vectors)")
    print(f"  Samples with ξ > 1:     {np.sum(ksi > 1.0)} (margin errors)")
    print(f"  Samples with ξ > 2:     {np.sum(ksi > 2.0)} (severe errors)")
    print()

    # =========================================================================
    # 3. Three-Tier Indicator Variables (α, β, ρ)
    # =========================================================================
    print("=" * 70)
    print("3. Three-Tier Indicator Variables (α, β, ρ)")
    print("=" * 70)
    indicators = model.get_indicator_variables()
    n_alpha = np.sum(indicators['alpha'] > 0.5)
    n_beta = np.sum(indicators['beta'] > 0.5)
    n_rho = np.sum(indicators['rho'] > 0.5)

    print(f"Indicators track which accuracy tier each sample falls into:")
    print(f"  α = 1 if ξ > 0:         {n_alpha} samples")
    print(f"  β = 1 if ξ > 1:         {n_beta} samples")
    print(f"  ρ = 1 if ξ > 2:         {n_rho} samples")
    print(f"  Hierarchy (α ≥ β ≥ ρ): {n_alpha >= n_beta >= n_rho} ✓")
    print()

    # =========================================================================
    # 4. Support Vectors
    # =========================================================================
    print("=" * 70)
    print("4. Support Vectors Analysis")
    print("=" * 70)
    sv_mask = model.get_support_vectors_mask()
    sv_indices = np.where(sv_mask)[0]

    print(f"Support vectors are samples with ξ > 0 (on or inside margin):")
    print(f"  Total support vectors:  {len(sv_indices)}")
    if len(sv_indices) > 0:
        print(f"  First 10 SV indices:    {sv_indices[:10]}")
        print(f"  SV slack values:")
        for i in sv_indices[:5]:
            print(f"    Sample {i}: ξ = {ksi[i]:.6f}")
    print()

    # =========================================================================
    # 5. Margin Errors
    # =========================================================================
    print("=" * 70)
    print("5. Margin Errors Analysis")
    print("=" * 70)
    me_mask = model.get_margin_errors_mask()
    me_indices = np.where(me_mask)[0]

    print(f"Margin errors are samples with ξ > 1 (misclassified or far from margin):")
    print(f"  Total margin errors:    {len(me_indices)}")
    if len(me_indices) > 0:
        print(f"  Margin error indices:   {me_indices}")
        # Compare predictions with true labels
        y_pred = model.predict(X[me_indices])
        y_true = y[me_indices]
        misclassified = np.sum(y_pred != y_true)
        print(f"  Actually misclassified: {misclassified}/{len(me_indices)}")
    print()

    # =========================================================================
    # 6. Weight Decomposition
    # =========================================================================
    print("=" * 70)
    print("6. Weight Decomposition (w = w⁺ - w⁻)")
    print("=" * 70)
    w_decomp = model.get_weight_decomposition()
    print(f"Weight decomposition for L1 regularization:")
    print(f"  w⁺:  {w_decomp['w_plus']}")
    print(f"  w⁻:  {w_decomp['w_minus']}")
    print(f"  w:   {model.weights}")
    print(f"  Verification: w = w⁺ - w⁻? {np.allclose(model.weights, w_decomp['w_plus'] - w_decomp['w_minus'])} ✓")
    print()

    # =========================================================================
    # 7. Sample-Level Analysis
    # =========================================================================
    print("=" * 70)
    print("7. Sample-Level Detailed Analysis")
    print("=" * 70)
    print(f"Top 5 samples with highest slack (most difficult to classify):")
    print()

    top_indices = np.argsort(ksi)[-5:][::-1]
    y_pred = model.predict(X)
    decision_vals = model.decision_function(X)

    print(f"{'Rank':<6}{'Sample':<8}{'ξ':<10}{'α':<4}{'β':<4}{'ρ':<4}{'True':<6}{'Pred':<6}{'Decision':<10}{'Status':<15}")
    print("-" * 70)
    for rank, idx in enumerate(top_indices, 1):
        status = "Correct" if y_pred[idx] == y[idx] else "MISCLASSIFIED"
        print(f"{rank:<6}{idx:<8}{ksi[idx]:<10.4f}"
              f"{indicators['alpha'][idx]:<4.0f}"
              f"{indicators['beta'][idx]:<4.0f}"
              f"{indicators['rho'][idx]:<4.0f}"
              f"{y[idx]:+6.0f}"
              f"{y_pred[idx]:+6.0f}"
              f"{decision_vals[idx]:<10.4f}"
              f"{status:<15}")
    print()

    # =========================================================================
    # 8. Overall Model Performance
    # =========================================================================
    print("=" * 70)
    print("8. Overall Model Performance")
    print("=" * 70)
    accuracy = np.mean(y_pred == y)
    pos_acc = np.mean(y_pred[y == 1] == y[y == 1])
    neg_acc = np.mean(y_pred[y == -1] == y[y == -1])

    print(f"Training Accuracy:      {accuracy:.4f}")
    print(f"Positive Class Acc:     {pos_acc:.4f}")
    print(f"Negative Class Acc:     {neg_acc:.4f}")
    print()

    # =========================================================================
    # 9. Direct Solution Dictionary Access
    # =========================================================================
    print("=" * 70)
    print("9. Direct Access to All Variables")
    print("=" * 70)
    print(f"All decision variables are stored in model.solution dictionary:")
    print(f"Available keys:")
    for key in sorted(model.solution.keys()):
        value = model.solution[key]
        if isinstance(value, np.ndarray):
            print(f"  {key:<25} array shape {value.shape}")
        else:
            print(f"  {key:<25} {type(value).__name__}")
    print()

    print("=" * 70)
    print("✓ Example Complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
