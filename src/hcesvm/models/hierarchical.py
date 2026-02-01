#!/usr/bin/env python3
"""
Hierarchical Classifier for Multi-class Ordinal Classification

Implements a hierarchical (cascade) classifier using two binary CE-SVM models
to solve 3-class ordinal classification problems.

Three strategies are supported:

1. Single Filter (original):
    Input X
      |
      v
    [H1: Class 3 vs {1,2}]
      |
      | if f1(x) < 0
      v
    [H2: Class 2 vs Class 1]
      |
      v
    Final Class: 1, 2, or 3

2. Multiple Filter:
    Input X
      |
      v
    [H1: Class 1 vs {2,3}]
      |
      | if f1(x) < 0
      v
    [H2: Class {1,2} vs Class 3]
      |
      v
    Final Class: 1, 2, or 3

3. Inverted (dynamic):
    Input X
      |
      v
    [H1: medium vs {majority, minority}]
      |
      | if f1(x) < 0
      v
    [H2: {medium, majority} vs minority]
      |
      v
    Final Class: 1, 2, or 3

    Where majority/medium/minority are determined dynamically
    based on training sample counts.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .binary_cesvm import BinaryCESVM


class HierarchicalCESVM:
    """Hierarchical Cost-Effective SVM for 3-class ordinal classification."""

    def __init__(self, cesvm_params: Optional[Dict] = None, strategy: str = "multiple_filter"):
        """Initialize Hierarchical CE-SVM.

        Args:
            cesvm_params: Parameters for binary CE-SVM models (shared)
            strategy: Classification strategy
                - "single_filter": Class 3 vs {1,2}, then Class 2 vs Class 1 (original)
                - "multiple_filter": Class 1 vs {2,3}, then Class {1,2} vs Class 3 (new)
                - "inverted": medium vs {majority, minority}, then {medium, majority} vs minority (dynamic)
        """
        if strategy not in ["single_filter", "multiple_filter", "inverted"]:
            raise ValueError(f"Unknown strategy: {strategy}. "
                            f"Must be 'single_filter', 'multiple_filter', or 'inverted'")

        self.cesvm_params = cesvm_params or {}
        self.strategy = strategy

        # Two binary classifiers
        # Configuration depends on strategy
        self.h1 = None
        self.h2 = None

        # Training data info
        self.n_features = None

        # Class roles for inverted strategy
        self.class_roles = None

    def _determine_class_roles(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray
    ) -> Dict:
        """Determine class roles based on sample counts.

        Args:
            X1: Class 1 samples
            X2: Class 2 samples
            X3: Class 3 samples

        Returns:
            dict: {
                'majority': int (1, 2, or 3),
                'medium': int,
                'minority': int,
                'X_majority': ndarray,
                'X_medium': ndarray,
                'X_minority': ndarray
            }
        """
        counts = [(1, len(X1), X1), (2, len(X2), X2), (3, len(X3), X3)]
        sorted_counts = sorted(counts, key=lambda x: x[1])

        return {
            'minority': sorted_counts[0][0],
            'X_minority': sorted_counts[0][2],
            'medium': sorted_counts[1][0],
            'X_medium': sorted_counts[1][2],
            'majority': sorted_counts[2][0],
            'X_majority': sorted_counts[2][2],
        }

    def _prepare_h1_data(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for H1 classifier.

        Strategy determines which classes are positive/negative:
            - single_filter: Class 3 (+1) vs {Class 1, 2} (-1)
            - multiple_filter: Class 1 (+1) vs {Class 2, 3} (-1)
            - inverted: medium (+1) vs {majority, minority} (-1) - dynamic based on sample counts

        Args:
            X1: Class 1 samples
            X2: Class 2 samples
            X3: Class 3 samples

        Returns:
            X_h1: Combined feature matrix
            y_h1: Binary labels (+1 for positive class, -1 for negative class)
        """
        if self.strategy == "single_filter":
            # Original: Class 3 (+1) vs {1,2} (-1)
            X_pos = X3
            X_neg = np.vstack([X1, X2])
            y_pos = np.ones(len(X3))
            y_neg = -np.ones(len(X1) + len(X2))
        elif self.strategy == "multiple_filter":
            # New: Class 1 (+1) vs {2,3} (-1)
            X_pos = X1
            X_neg = np.vstack([X2, X3])
            y_pos = np.ones(len(X1))
            y_neg = -np.ones(len(X2) + len(X3))
        else:  # inverted
            # Dynamic: medium (+1) vs {majority, minority} (-1)
            roles = self._determine_class_roles(X1, X2, X3)
            X_pos = roles['X_medium']
            X_neg = np.vstack([roles['X_majority'], roles['X_minority']])
            y_pos = np.ones(len(roles['X_medium']))
            y_neg = -np.ones(len(roles['X_majority']) + len(roles['X_minority']))

        X_h1 = np.vstack([X_pos, X_neg])
        y_h1 = np.concatenate([y_pos, y_neg])

        return X_h1, y_h1
    
    def _prepare_h2_data(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for H2 classifier.

        Strategy determines which classes are positive/negative:
            - single_filter: Class 2 (+1) vs Class 1 (-1)
            - multiple_filter: Class {1,2} (+1) vs Class 3 (-1)
            - inverted: {medium, majority} (+1) vs minority (-1) - dynamic based on sample counts

        Args:
            X1: Class 1 samples
            X2: Class 2 samples
            X3: Class 3 samples

        Returns:
            X_h2: Combined feature matrix
            y_h2: Binary labels (+1 for positive class, -1 for negative class)
        """
        if self.strategy == "single_filter":
            # Original: Class 2 (+1) vs Class 1 (-1)
            X_pos = X2
            X_neg = X1
            y_pos = np.ones(len(X2))
            y_neg = -np.ones(len(X1))
        elif self.strategy == "multiple_filter":
            # New: Class {1,2} (+1) vs Class 3 (-1)
            X_pos = np.vstack([X1, X2])
            X_neg = X3
            y_pos = np.ones(len(X1) + len(X2))
            y_neg = -np.ones(len(X3))
        else:  # inverted
            # Dynamic: {medium, majority} (+1) vs minority (-1)
            # Order: medium first, then majority
            roles = self._determine_class_roles(X1, X2, X3)
            X_pos = np.vstack([roles['X_medium'], roles['X_majority']])
            X_neg = roles['X_minority']
            y_pos = np.ones(len(roles['X_medium']) + len(roles['X_majority']))
            y_neg = -np.ones(len(roles['X_minority']))

        X_h2 = np.vstack([X_pos, X_neg])
        y_h2 = np.concatenate([y_pos, y_neg])

        return X_h2, y_h2
    
    def fit(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray
    ) -> 'HierarchicalCESVM':
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
        print(f"Training Hierarchical CE-SVM (Strategy: {self.strategy})")
        print("=" * 60)
        print(f"Class 1: {len(X1)} samples")
        print(f"Class 2: {len(X2)} samples")
        print(f"Class 3: {len(X3)} samples")
        print(f"Features: {self.n_features}")

        # For inverted strategy, determine and store class roles
        if self.strategy == "inverted":
            self.class_roles = self._determine_class_roles(X1, X2, X3)
            print(f"\nDynamic Class Roles:")
            print(f"  Majority:  Class {self.class_roles['majority']} ({len(self.class_roles['X_majority'])} samples)")
            print(f"  Medium:    Class {self.class_roles['medium']} ({len(self.class_roles['X_medium'])} samples)")
            print(f"  Minority:  Class {self.class_roles['minority']} ({len(self.class_roles['X_minority'])} samples)")

        print()

        # Determine H1 description based on strategy
        if self.strategy == "single_filter":
            h1_desc = "Class 3 (+1) vs {Class 1, 2} (-1)"
        elif self.strategy == "multiple_filter":
            h1_desc = "Class 1 (+1) vs {Class 2, 3} (-1)"
        else:  # inverted
            h1_desc = f"Class {self.class_roles['medium']} (+1) vs {{Class {self.class_roles['majority']}, {self.class_roles['minority']}}} (-1)"

        # Train H1
        print("=" * 60)
        print(f"Training H1: {h1_desc}")
        print("=" * 60)
        X_h1, y_h1 = self._prepare_h1_data(X1, X2, X3)
        print(f"H1 Training samples: {len(X_h1)}")
        print(f"  Positive (+1): {np.sum(y_h1 == 1)} samples")
        print(f"  Negative (-1): {np.sum(y_h1 == -1)} samples")
        print()

        self.h1 = BinaryCESVM(**self.cesvm_params)
        self.h1.fit(X_h1, y_h1)

        print(f"\nH1 Solution:")
        h1_summary = self.h1.get_solution_summary()
        print(f"  Objective: {h1_summary['objective_value']:.6f}")
        print(f"  Selected features: {h1_summary['n_selected_features']}/{self.n_features}")
        print(f"  L1 norm: {h1_summary['l1_norm']:.6f}")
        print(f"  Positive accuracy lb: {h1_summary['positive_class_accuracy_lb']:.4f}")
        print(f"  Negative accuracy lb: {h1_summary['negative_class_accuracy_lb']:.4f}")

        # Determine H2 description based on strategy
        if self.strategy == "single_filter":
            h2_desc = "Class 2 (+1) vs Class 1 (-1)"
        elif self.strategy == "multiple_filter":
            h2_desc = "Class {1,2} (+1) vs Class 3 (-1)"
        else:  # inverted
            h2_desc = f"Class {{Class {self.class_roles['medium']}, {self.class_roles['majority']}}} (+1) vs Class {self.class_roles['minority']} (-1)"

        # Train H2
        print("\n" + "=" * 60)
        print(f"Training H2: {h2_desc}")
        print("=" * 60)
        X_h2, y_h2 = self._prepare_h2_data(X1, X2, X3)
        print(f"H2 Training samples: {len(X_h2)}")
        print(f"  Positive (+1): {np.sum(y_h2 == 1)} samples")
        print(f"  Negative (-1): {np.sum(y_h2 == -1)} samples")
        print()

        self.h2 = BinaryCESVM(**self.cesvm_params)
        self.h2.fit(X_h2, y_h2)

        print(f"\nH2 Solution:")
        h2_summary = self.h2.get_solution_summary()
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
        """Predict class labels using hierarchical decision rule.

        Decision Rule depends on strategy:

        single_filter:
            1. Compute f1(x) = w1*x + b1
            2. If f1(x) >= 0: predict Class 3
            3. Else:
                a. Compute f2(x) = w2*x + b2
                b. If f2(x) >= 0: predict Class 2
                c. Else: predict Class 1

        multiple_filter:
            1. Compute f1(x) = w1*x + b1
            2. If f1(x) >= 0: predict Class 1
            3. Else:
                a. Compute f2(x) = w2*x + b2
                b. If f2(x) >= 0: predict Class 2
                c. Else: predict Class 3

        inverted:
            1. Compute f1(x) = w1*x + b1
            2. If f1(x) >= 0: predict medium class
            3. Else:
                a. Compute f2(x) = w2*x + b2
                b. If f2(x) >= 0: predict majority class
                c. Else: predict minority class

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

        if self.strategy == "single_filter":
            # f1 >= 0 --> Class 3
            # f1 < 0  --> proceed to H2
            first_class = 3
            remaining_classes = (2, 1)  # (positive_class, negative_class)
        elif self.strategy == "multiple_filter":
            # f1 >= 0 --> Class 1
            # f1 < 0  --> proceed to H2
            first_class = 1
            remaining_classes = (2, 3)  # (positive_class, negative_class)
        else:  # inverted
            # f1 >= 0 --> medium class
            # f1 < 0  --> proceed to H2
            first_class = self.class_roles['medium']
            remaining_classes = (self.class_roles['majority'], self.class_roles['minority'])

        # Samples classified by H1
        h1_pos_mask = f1 >= 0
        predictions[h1_pos_mask] = first_class

        # Remaining samples go to H2
        h2_mask = ~h1_pos_mask

        if np.any(h2_mask):
            # Step 2: H2 decision on remaining samples
            X_remaining = X[h2_mask]
            f2 = self.h2.decision_function(X_remaining)

            # f2 >= 0 --> remaining_classes[0] (positive class in H2)
            # f2 < 0  --> remaining_classes[1] (negative class in H2)
            h2_pos_mask = f2 >= 0
            predictions[h2_mask] = np.where(
                h2_pos_mask,
                remaining_classes[0],
                remaining_classes[1]
            )

        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get summary of both classifiers.

        Returns:
            Dictionary with model information
        """
        if self.h1 is None or self.h2 is None:
            return {"status": "not_fitted"}

        # Determine classifier descriptions based on strategy
        if self.strategy == "single_filter":
            h1_desc = "Class 3 vs {1,2}"
            h2_desc = "Class 2 vs Class 1"
        elif self.strategy == "multiple_filter":
            h1_desc = "Class 1 vs {2,3}"
            h2_desc = "Class {1,2} vs Class 3"
        else:  # inverted
            h1_desc = f"Class {self.class_roles['medium']} vs {{Class {self.class_roles['majority']}, {self.class_roles['minority']}}}"
            h2_desc = f"Class {{Class {self.class_roles['medium']}, {self.class_roles['majority']}}} vs Class {self.class_roles['minority']}"

        summary = {
            "status": "fitted",
            "strategy": self.strategy,
            "n_features": self.n_features,
            "h1": {
                "description": h1_desc,
                **self.h1.get_solution_summary()
            },
            "h2": {
                "description": h2_desc,
                **self.h2.get_solution_summary()
            },
        }

        # Add class roles info for inverted strategy
        if self.strategy == "inverted" and self.class_roles is not None:
            summary["class_roles"] = {
                "majority": self.class_roles['majority'],
                "medium": self.class_roles['medium'],
                "minority": self.class_roles['minority']
            }

        return summary
