#!/usr/bin/env python3
"""
Test script for Class 1 First classification strategy.

This script tests a new fixed hierarchical classification approach:
- H1: Class 1 (+1) vs {Class 2, 3} (-1)
- H2: Class 2 (+1) vs Class 3 (-1)

Prediction Logic:
- H1 predicts +1 → Class 1
- H1 predicts -1 → Go to H2
  - H2 predicts +1 → Class 2
  - H2 predicts -1 → Class 3

Usage:
    python examples/run_class1_first_test.py --dataset <dataset_name>

Example:
    python examples/run_class1_first_test.py --dataset thyroid
    python examples/run_class1_first_test.py --dataset contraceptive
"""

import sys
import os
from pathlib import Path
import numpy as np
import argparse
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hcesvm.utils.data_loader import load_split_data
from src.hcesvm.models.binary_cesvm import BinaryCESVM


class Class1FirstCESVM:
    """
    Fixed hierarchical classifier: Class 1 first, then Class 2 vs Class 3.

    This is a custom implementation that does NOT depend on sample distribution.
    Classification is always:
    - H1: Class 1 (+1) vs {Class 2, 3} (-1)
    - H2: Class 2 (+1) vs Class 3 (-1)
    """

    def __init__(self, cesvm_params: dict = None):
        """Initialize Class1First CE-SVM.

        Args:
            cesvm_params: Parameters for binary CE-SVM models (shared)
        """
        self.cesvm_params = cesvm_params or {}
        self.h1 = None
        self.h2 = None
        self.n_features = None

    def fit(self, X1: np.ndarray, X2: np.ndarray, X3: np.ndarray):
        """Fit the hierarchical classifier.

        Args:
            X1: Class 1 samples (n1, n_features)
            X2: Class 2 samples (n2, n_features)
            X3: Class 3 samples (n3, n_features)

        Returns:
            self
        """
        self.n_features = X1.shape[1]

        print("=" * 60)
        print("Training Class1-First Hierarchical CE-SVM")
        print("=" * 60)
        print(f"Class 1: {len(X1)} samples")
        print(f"Class 2: {len(X2)} samples")
        print(f"Class 3: {len(X3)} samples")
        print(f"Features: {self.n_features}")
        print()

        # ============================================================
        # Train H1: Class 1 (+1) vs {Class 2, 3} (-1)
        # ============================================================
        print("=" * 60)
        print("Training H1: Class 1 (+1) vs {Class 2, 3} (-1)")
        print("=" * 60)

        X_h1_pos = X1  # Class 1 is +1
        X_h1_neg = np.vstack([X2, X3])  # Class 2 and 3 are -1
        X_h1 = np.vstack([X_h1_pos, X_h1_neg])
        y_h1 = np.concatenate([
            np.ones(len(X1)),  # Class 1 → +1
            -np.ones(len(X2) + len(X3))  # Class 2, 3 → -1
        ])

        print(f"H1 Training samples: {len(X_h1)}")
        print(f"  Positive (+1, Class 1): {len(X1)} samples")
        print(f"  Negative (-1, Class 2+3): {len(X2) + len(X3)} samples")
        print()

        self.h1 = BinaryCESVM(**self.cesvm_params)
        self.h1.fit(X_h1, y_h1)

        h1_summary = self.h1.get_solution_summary()
        print(f"\nH1 Solution:")
        print(f"  Weights (w): {self.h1.weights}")
        print(f"  Intercept (b): {self.h1.intercept}")
        print(f"  Objective: {h1_summary['objective_value']:.6f}")
        print(f"  Selected features: {h1_summary['n_selected_features']}/{self.n_features}")
        print(f"  L1 norm: {h1_summary['l1_norm']:.6f}")
        print(f"  Positive accuracy lb: {h1_summary['positive_class_accuracy_lb']:.4f}")
        print(f"  Negative accuracy lb: {h1_summary['negative_class_accuracy_lb']:.4f}")

        # ============================================================
        # Train H2: Class 2 (+1) vs Class 3 (-1)
        # ============================================================
        print("\n" + "=" * 60)
        print("Training H2: Class 2 (+1) vs Class 3 (-1)")
        print("=" * 60)

        X_h2_pos = X2  # Class 2 is +1
        X_h2_neg = X3  # Class 3 is -1
        X_h2 = np.vstack([X_h2_pos, X_h2_neg])
        y_h2 = np.concatenate([
            np.ones(len(X2)),  # Class 2 → +1
            -np.ones(len(X3))  # Class 3 → -1
        ])

        print(f"H2 Training samples: {len(X_h2)}")
        print(f"  Positive (+1, Class 2): {len(X2)} samples")
        print(f"  Negative (-1, Class 3): {len(X3)} samples")
        print()

        self.h2 = BinaryCESVM(**self.cesvm_params)
        self.h2.fit(X_h2, y_h2)

        h2_summary = self.h2.get_solution_summary()
        print(f"\nH2 Solution:")
        print(f"  Weights (w): {self.h2.weights}")
        print(f"  Intercept (b): {self.h2.intercept}")
        print(f"  Objective: {h2_summary['objective_value']:.6f}")
        print(f"  Selected features: {h2_summary['n_selected_features']}/{self.n_features}")
        print(f"  L1 norm: {h2_summary['l1_norm']:.6f}")
        print(f"  Positive accuracy lb: {h2_summary['positive_class_accuracy_lb']:.4f}")
        print(f"  Negative accuracy lb: {h2_summary['negative_class_accuracy_lb']:.4f}")

        print("\n" + "=" * 60)
        print("Training Complete!")
        print("=" * 60)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using fixed hierarchical decision rule.

        Decision Rule:
        1. Compute f1(x) = w1*x + b1
        2. If f1(x) >= 0: predict Class 1
        3. Else:
            a. Compute f2(x) = w2*x + b2
            b. If f2(x) >= 0: predict Class 2
            c. Else: predict Class 3

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,), values in {1, 2, 3}
        """
        if self.h1 is None or self.h2 is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)

        # Step 1: H1 decision
        f1 = self.h1.decision_function(X)
        h1_pos_mask = f1 >= 0  # Class 1
        predictions[h1_pos_mask] = 1

        # Step 2: H2 decision on remaining samples
        h2_mask = ~h1_pos_mask
        if np.any(h2_mask):
            X_remaining = X[h2_mask]
            f2 = self.h2.decision_function(X_remaining)

            # f2 >= 0 → Class 2, f2 < 0 → Class 3
            h2_pos_mask = f2 >= 0
            predictions[h2_mask] = np.where(h2_pos_mask, 2, 3)

        return predictions


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description='Test Class1-First hierarchical classification strategy'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Dataset name (e.g., thyroid, contraceptive)'
    )
    parser.add_argument(
        '--time-limit',
        type=int,
        default=None,
        help='Time limit for each classifier in seconds (default: None, no limit)'
    )
    parser.add_argument(
        '--C',
        type=float,
        default=1.0,
        help='C hyperparameter (default: 1.0)'
    )
    parser.add_argument(
        '--M',
        type=float,
        default=1000.0,
        help='M parameter (default: 1000.0)'
    )

    args = parser.parse_args()

    # Dataset paths
    dataset_paths = {
        'thyroid': '~/Developer/NSVORA/Archive/thyroid_split.xlsx',
        'contraceptive': '~/Developer/NSVORA/Archive/contraceptive_split.xlsx',
        'car_evaluation': '~/Developer/NSVORA/datasets/primary/car_evaluation/car_evaluation_split.xlsx',
        'balance': '~/Developer/NSVORA/Archive/balance_split.xlsx',
        'hayes_roth': '~/Developer/NSVORA/Archive/hayes_roth_split.xlsx',
        'new_thyroid': '~/Developer/NSVORA/Archive/new_thyroid_split.xlsx',
        'tae': '~/Developer/NSVORA/Archive/tae_split.xlsx',
        'wine': '~/Developer/NSVORA/Archive/wine_split.xlsx',
    }

    dataset_name = args.dataset.lower()
    if dataset_name not in dataset_paths:
        print(f"Error: Unknown dataset '{args.dataset}'")
        print(f"Available datasets: {', '.join(dataset_paths.keys())}")
        sys.exit(1)

    data_path = Path(dataset_paths[dataset_name]).expanduser()

    if not data_path.exists():
        print(f"Error: Dataset file not found: {data_path}")
        sys.exit(1)

    # Create results directory
    results_dir = Path(__file__).resolve().parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Generate timestamp for this run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    print("=" * 80)
    print(f"Class1-First Strategy Test: {dataset_name.upper()}")
    print("=" * 80)
    print(f"Dataset: {data_path}")
    print(f"Timestamp: {timestamp}")
    time_limit_str = f"{args.time_limit}s per classifier" if args.time_limit else "No time limit (run to global optimum)"
    print(f"Time limit: {time_limit_str}")
    print(f"C: {args.C}, M: {args.M}")
    print("=" * 80)
    print()

    # Load data
    print("Loading data...")
    data = load_split_data(str(data_path))

    X1_train = data['X1_train']
    X2_train = data['X2_train']
    X3_train = data['X3_train']
    X_test = data['X_test']
    y_test = data['y_test']

    print(f"Training set sizes: Class 1: {len(X1_train)}, Class 2: {len(X2_train)}, Class 3: {len(X3_train)}")
    print(f"Test set size: {len(X_test)}")
    print()

    # Model parameters
    cesvm_params = {
        'C_hyper': args.C,
        'M': args.M,
        'time_limit': args.time_limit,
        'mip_gap': 1e-4,
    }

    # Train model
    model = Class1FirstCESVM(cesvm_params=cesvm_params)
    model.fit(X1_train, X2_train, X3_train)

    print("\n" + "=" * 80)
    print("EVALUATION")
    print("=" * 80)

    # Training set predictions
    X_train = np.vstack([X1_train, X2_train, X3_train])
    y_train = np.concatenate([
        np.ones(len(X1_train)),
        2 * np.ones(len(X2_train)),
        3 * np.ones(len(X3_train))
    ]).astype(int)

    y_train_pred = model.predict(X_train)

    # Test set predictions
    y_test_pred = model.predict(X_test)

    # Calculate accuracy by class
    def calculate_class_accuracy(y_true, y_pred):
        """Calculate per-class accuracy."""
        accuracies = {}
        for cls in [1, 2, 3]:
            mask = y_true == cls
            if np.sum(mask) > 0:
                accuracies[cls] = np.mean(y_pred[mask] == cls)
            else:
                accuracies[cls] = 0.0
        return accuracies

    train_class_acc = calculate_class_accuracy(y_train, y_train_pred)
    test_class_acc = calculate_class_accuracy(y_test, y_test_pred)

    train_total_acc = np.mean(y_train_pred == y_train)
    test_total_acc = np.mean(y_test_pred == y_test)

    # Print results
    print("\n" + "-" * 80)
    print("TRAINING SET RESULTS")
    print("-" * 80)
    print(f"Class 1 Accuracy: {train_class_acc[1]:.4f} ({np.sum((y_train == 1) & (y_train_pred == 1))}/{np.sum(y_train == 1)})")
    print(f"Class 2 Accuracy: {train_class_acc[2]:.4f} ({np.sum((y_train == 2) & (y_train_pred == 2))}/{np.sum(y_train == 2)})")
    print(f"Class 3 Accuracy: {train_class_acc[3]:.4f} ({np.sum((y_train == 3) & (y_train_pred == 3))}/{np.sum(y_train == 3)})")
    print(f"Total Accuracy: {train_total_acc:.4f} ({np.sum(y_train_pred == y_train)}/{len(y_train)})")

    print("\n" + "-" * 80)
    print("TEST SET RESULTS")
    print("-" * 80)
    print(f"Class 1 Accuracy: {test_class_acc[1]:.4f} ({np.sum((y_test == 1) & (y_test_pred == 1))}/{np.sum(y_test == 1)})")
    print(f"Class 2 Accuracy: {test_class_acc[2]:.4f} ({np.sum((y_test == 2) & (y_test_pred == 2))}/{np.sum(y_test == 2)})")
    print(f"Class 3 Accuracy: {test_class_acc[3]:.4f} ({np.sum((y_test == 3) & (y_test_pred == 3))}/{np.sum(y_test == 3)})")
    print(f"Total Accuracy: {test_total_acc:.4f} ({np.sum(y_test_pred == y_test)}/{len(y_test)})")

    print("\n" + "-" * 80)
    print("MODEL PARAMETERS")
    print("-" * 80)

    h1_summary = model.h1.get_solution_summary()
    h2_summary = model.h2.get_solution_summary()

    print("\nH1 (Class 1 vs {2,3}):")
    print(f"  Weights: {model.h1.weights}")
    print(f"  Intercept: {model.h1.intercept}")
    print(f"  MIP Gap: {h1_summary['mip_gap']:.6f}")
    print(f"  Status: {h1_summary['solver_status']}")

    print("\nH2 (Class 2 vs 3):")
    print(f"  Weights: {model.h2.weights}")
    print(f"  Intercept: {model.h2.intercept}")
    print(f"  MIP Gap: {h2_summary['mip_gap']:.6f}")
    print(f"  Status: {h2_summary['solver_status']}")

    # Save results to file
    result_file = results_dir / f'class1_first_{dataset_name}_{timestamp}.log'

    with open(result_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Class1-First Strategy Test: {dataset_name.upper()}\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Dataset: {data_path}\n")
        time_limit_str = f"{args.time_limit}s per classifier" if args.time_limit else "No time limit (run to global optimum)"
        f.write(f"Time limit: {time_limit_str}\n")
        f.write(f"C: {args.C}, M: {args.M}\n")
        f.write("\n")

        f.write("Training set sizes:\n")
        f.write(f"  Class 1: {len(X1_train)}\n")
        f.write(f"  Class 2: {len(X2_train)}\n")
        f.write(f"  Class 3: {len(X3_train)}\n")
        f.write(f"Test set size: {len(X_test)}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("TRAINING SET RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Class 1 Accuracy: {train_class_acc[1]:.4f}\n")
        f.write(f"Class 2 Accuracy: {train_class_acc[2]:.4f}\n")
        f.write(f"Class 3 Accuracy: {train_class_acc[3]:.4f}\n")
        f.write(f"Total Accuracy: {train_total_acc:.4f}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("TEST SET RESULTS\n")
        f.write("-" * 80 + "\n")
        f.write(f"Class 1 Accuracy: {test_class_acc[1]:.4f}\n")
        f.write(f"Class 2 Accuracy: {test_class_acc[2]:.4f}\n")
        f.write(f"Class 3 Accuracy: {test_class_acc[3]:.4f}\n")
        f.write(f"Total Accuracy: {test_total_acc:.4f}\n")
        f.write("\n")

        f.write("-" * 80 + "\n")
        f.write("MODEL PARAMETERS\n")
        f.write("-" * 80 + "\n")
        f.write("\nH1 (Class 1 vs {2,3}):\n")
        f.write(f"  Weights: {model.h1.weights}\n")
        f.write(f"  Intercept: {model.h1.intercept}\n")
        f.write(f"  MIP Gap: {h1_summary['mip_gap']:.6f}\n")
        f.write(f"  Status: {h1_summary['solver_status']}\n")
        f.write("\nH2 (Class 2 vs 3):\n")
        f.write(f"  Weights: {model.h2.weights}\n")
        f.write(f"  Intercept: {model.h2.intercept}\n")
        f.write(f"  MIP Gap: {h2_summary['mip_gap']:.6f}\n")
        f.write(f"  Status: {h2_summary['solver_status']}\n")

    print(f"\n\nResults saved to: {result_file}")
    print("=" * 80)


if __name__ == "__main__":
    main()
