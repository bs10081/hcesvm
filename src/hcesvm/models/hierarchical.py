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
                - "test2": Split based on minority position with ordinal labeling and Test2 rule
        """
        if strategy not in ["single_filter", "multiple_filter", "inverted", "test2"]:
            raise ValueError(f"Unknown strategy: {strategy}. "
                            f"Must be 'single_filter', 'multiple_filter', 'inverted', or 'test2'")

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

    def _determine_accuracy_mode(
        self,
        positive_classes: list,
        negative_classes: list
    ) -> str:
        """Determine accuracy_mode based on majority position (Test2 Rule).

        Only applies when majority is NOT at the edge (majority = class 2).
        - If majority in +1 class: return "negative_only" (remove -l_p)
        - If majority in -1 class: return "positive_only" (remove -l_n)
        - If majority at edge (class 1 or 3): return "both"

        Args:
            positive_classes: List of classes mapped to +1 label
            negative_classes: List of classes mapped to -1 label

        Returns:
            accuracy_mode: "both", "positive_only", or "negative_only"
        """
        majority = self.class_roles['majority']

        # Check if majority is at the edge
        if majority in [1, 3]:
            return "both"  # Normal objective

        # Majority = 2 (not at edge), apply test2 rule
        if majority in positive_classes:
            return "negative_only"  # Remove -l_p (don't maximize +1 accuracy)
        elif majority in negative_classes:
            return "positive_only"  # Remove -l_n (don't maximize -1 accuracy)
        else:
            return "both"

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
            - test2: Split based on minority position with ordinal labeling

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
        elif self.strategy == "test2":
            # Test2: Split based on minority position, follow ordinal rule
            roles = self._determine_class_roles(X1, X2, X3)
            minority = roles['minority']

            if minority == 1:
                # Minority at Class 1 (edge) -> {1} vs {2, 3}
                X_neg = X1  # Lower ordinal = -1
                X_pos = np.vstack([X2, X3])  # Higher ordinal = +1
                y_neg = -np.ones(len(X1))
                y_pos = np.ones(len(X2) + len(X3))
            elif minority == 3:
                # Minority at Class 3 (edge) -> {1, 2} vs {3}
                X_neg = np.vstack([X1, X2])  # Lower ordinal = -1
                X_pos = X3  # Higher ordinal = +1
                y_neg = -np.ones(len(X1) + len(X2))
                y_pos = np.ones(len(X3))
            else:  # minority == 2
                # Minority at middle -> default {1} vs {2, 3}
                X_neg = X1  # Lower ordinal = -1
                X_pos = np.vstack([X2, X3])  # Higher ordinal = +1
                y_neg = -np.ones(len(X1))
                y_pos = np.ones(len(X2) + len(X3))
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
            - test2: Split based on minority position with ordinal labeling (uses ALL training data)

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
        elif self.strategy == "test2":
            # Test2: H2 uses ALL training data
            roles = self._determine_class_roles(X1, X2, X3)
            minority = roles['minority']

            if minority == 1:
                # H2 split {1,2} vs {3} (using all training data)
                X_neg = np.vstack([X1, X2])  # Lower ordinal = -1
                X_pos = X3  # Higher ordinal = +1
                y_neg = -np.ones(len(X1) + len(X2))
                y_pos = np.ones(len(X3))
            elif minority == 3:
                # H2 split {1} vs {2,3} (using all training data)
                X_neg = X1  # Lower ordinal = -1
                X_pos = np.vstack([X2, X3])  # Higher ordinal = +1
                y_neg = -np.ones(len(X1))
                y_pos = np.ones(len(X2) + len(X3))
            else:  # minority == 2
                # H2 split {1,2} vs {3} (using all training data)
                X_neg = np.vstack([X1, X2])  # Lower ordinal = -1
                X_pos = X3  # Higher ordinal = +1
                y_neg = -np.ones(len(X1) + len(X2))
                y_pos = np.ones(len(X3))
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

        # For inverted and test2 strategies, determine and store class roles
        if self.strategy in ["inverted", "test2"]:
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
        elif self.strategy == "test2":
            minority = self.class_roles['minority']
            if minority == 1:
                h1_desc = "Class {2, 3} (+1) vs Class 1 (-1)"
            elif minority == 3:
                h1_desc = "Class 3 (+1) vs {Class 1, 2} (-1)"
            else:  # minority == 2
                h1_desc = "Class {2, 3} (+1) vs Class 1 (-1)"
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

        # Determine accuracy_mode for H1 (test2 strategy only)
        h1_params = self.cesvm_params.copy()
        if self.strategy == "test2":
            minority = self.class_roles['minority']
            if minority == 1:
                h1_positive_classes = [2, 3]
                h1_negative_classes = [1]
            elif minority == 3:
                h1_positive_classes = [3]
                h1_negative_classes = [1, 2]
            else:  # minority == 2
                h1_positive_classes = [2, 3]
                h1_negative_classes = [1]
            h1_accuracy_mode = self._determine_accuracy_mode(
                h1_positive_classes, h1_negative_classes
            )
            h1_params['accuracy_mode'] = h1_accuracy_mode
            print(f"  Accuracy mode: {h1_accuracy_mode}")
        print()

        self.h1 = BinaryCESVM(**h1_params)
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
        elif self.strategy == "test2":
            minority = self.class_roles['minority']
            if minority == 1:
                h2_desc = "Class 3 (+1) vs Class {1, 2} (-1)"
            elif minority == 3:
                h2_desc = "Class {2, 3} (+1) vs Class 1 (-1)"
            else:  # minority == 2
                h2_desc = "Class 3 (+1) vs Class {1, 2} (-1)"
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

        # Determine accuracy_mode for H2 (test2 strategy only)
        h2_params = self.cesvm_params.copy()
        if self.strategy == "test2":
            minority = self.class_roles['minority']
            if minority == 1:
                # H2: {1,2} vs {3}
                h2_positive_classes = [3]
                h2_negative_classes = [1, 2]
            elif minority == 3:
                # H2: {1} vs {2,3}
                h2_positive_classes = [2, 3]
                h2_negative_classes = [1]
            else:  # minority == 2
                # H2: {1,2} vs {3}
                h2_positive_classes = [3]
                h2_negative_classes = [1, 2]
            h2_accuracy_mode = self._determine_accuracy_mode(
                h2_positive_classes, h2_negative_classes
            )
            h2_params['accuracy_mode'] = h2_accuracy_mode
            print(f"  Accuracy mode: {h2_accuracy_mode}")
        print()

        self.h2 = BinaryCESVM(**h2_params)
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
        elif self.strategy == "test2":
            # Test2: Decision depends on minority position
            minority = self.class_roles['minority']
            if minority == 1:
                # H1: {1} vs {2,3}  -> f1 < 0 = Class 1, f1 >= 0 -> H2
                # H2: {1,2} vs {3}  -> f2 < 0 = Class 2, f2 >= 0 = Class 3
                first_class = -1  # Special marker: negative means use H2 for f1 >= 0
                remaining_classes = (3, 2)  # For H2: f2 >= 0 -> Class 3, f2 < 0 -> Class 2
                negative_first_class = 1  # For f1 < 0 -> Class 1
            elif minority == 3:
                # H1: {1,2} vs {3} -> f1 >= 0 = Class 3, f1 < 0 -> H2
                # H2: {1} vs {2,3} -> f2 < 0 = Class 1, f2 >= 0 = Class 2 or 3
                first_class = 3  # f1 >= 0 -> Class 3
                remaining_classes = (2, 1)  # For H2: f2 >= 0 -> Class 2, f2 < 0 -> Class 1
            else:  # minority == 2
                # Same as minority == 1
                first_class = -1
                remaining_classes = (3, 2)
                negative_first_class = 1
        else:  # inverted
            # f1 >= 0 --> medium class
            # f1 < 0  --> proceed to H2
            first_class = self.class_roles['medium']
            remaining_classes = (self.class_roles['majority'], self.class_roles['minority'])

        # Samples classified by H1
        h1_pos_mask = f1 >= 0

        # Special handling for test2 when first_class is negative marker
        if self.strategy == "test2" and first_class == -1:
            # For test2 minority=1 or minority=2: f1 < 0 -> negative_first_class
            h1_neg_mask = f1 < 0
            predictions[h1_neg_mask] = negative_first_class
            # f1 >= 0 -> proceed to H2
            h2_mask = h1_pos_mask
        else:
            # Standard behavior
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
        elif self.strategy == "test2":
            minority = self.class_roles['minority']
            if minority == 1:
                h1_desc = "Class {2, 3} vs Class 1"
                h2_desc = "Class 3 vs Class {1, 2}"
            elif minority == 3:
                h1_desc = "Class 3 vs Class {1, 2}"
                h2_desc = "Class {2, 3} vs Class 1"
            else:  # minority == 2
                h1_desc = "Class {2, 3} vs Class 1"
                h2_desc = "Class 3 vs Class {1, 2}"
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

        # Add class roles info for inverted and test2 strategies
        if self.strategy in ["inverted", "test2"] and self.class_roles is not None:
            summary["class_roles"] = {
                "majority": self.class_roles['majority'],
                "medium": self.class_roles['medium'],
                "minority": self.class_roles['minority']
            }

        return summary
