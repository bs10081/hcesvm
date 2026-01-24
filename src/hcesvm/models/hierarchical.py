#!/usr/bin/env python3
"""
Hierarchical Classifier for Multi-class Ordinal Classification

Implements a hierarchical (cascade) classifier using two binary CE-SVM models
to solve 3-class ordinal classification problems.

Architecture:
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
"""

import numpy as np
from typing import Dict, Optional, Tuple
from .binary_cesvm import BinaryCESVM


class HierarchicalCESVM:
    """Hierarchical Cost-Effective SVM for 3-class ordinal classification."""

    def __init__(self, cesvm_params: Optional[Dict] = None):
        """Initialize Hierarchical CE-SVM.
        
        Args:
            cesvm_params: Parameters for binary CE-SVM models (shared)
        """
        self.cesvm_params = cesvm_params or {}
        
        # Two binary classifiers
        self.h1 = None  # Class 3 (+1) vs {1,2} (-1)
        self.h2 = None  # Class 2 (+1) vs Class 1 (-1)
        
        # Training data info
        self.n_features = None
        
    def _prepare_h1_data(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for H1 (Class 3 vs {1,2}).
        
        Args:
            X1: Class 1 samples
            X2: Class 2 samples
            X3: Class 3 samples
            
        Returns:
            X_h1: Combined feature matrix
            y_h1: Binary labels (+1 for Class 3, -1 for {1,2})
        """
        X_pos = X3  # Positive class: Class 3
        X_neg = np.vstack([X1, X2])  # Negative class: Class 1 + Class 2
        
        y_pos = np.ones(len(X3))
        y_neg = -np.ones(len(X1) + len(X2))
        
        X_h1 = np.vstack([X_pos, X_neg])
        y_h1 = np.concatenate([y_pos, y_neg])
        
        return X_h1, y_h1
    
    def _prepare_h2_data(
        self,
        X1: np.ndarray,
        X2: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for H2 (Class 2 vs Class 1).
        
        Args:
            X1: Class 1 samples
            X2: Class 2 samples
            
        Returns:
            X_h2: Combined feature matrix
            y_h2: Binary labels (+1 for Class 2, -1 for Class 1)
        """
        X_pos = X2  # Positive class: Class 2
        X_neg = X1  # Negative class: Class 1
        
        y_pos = np.ones(len(X2))
        y_neg = -np.ones(len(X1))
        
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
        print("Training Hierarchical CE-SVM")
        print("=" * 60)
        print(f"Class 1: {len(X1)} samples")
        print(f"Class 2: {len(X2)} samples")
        print(f"Class 3: {len(X3)} samples")
        print(f"Features: {self.n_features}")
        print()
        
        # Train H1: Class 3 vs {1,2}
        print("=" * 60)
        print("Training H1: Class 3 (+1) vs {Class 1, 2} (-1)")
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
        
        # Train H2: Class 2 vs Class 1
        print("\n" + "=" * 60)
        print("Training H2: Class 2 (+1) vs Class 1 (-1)")
        print("=" * 60)
        X_h2, y_h2 = self._prepare_h2_data(X1, X2)
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
        
        Decision Rule:
            1. Compute f1(x) = w1·x + b1
            2. If f1(x) >= 0: predict Class 3
            3. Else:
                a. Compute f2(x) = w2·x + b2
                b. If f2(x) >= 0: predict Class 2
                c. Else: predict Class 1
        
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
        
        # Class 3 samples (f1 >= 0)
        class3_mask = f1 >= 0
        predictions[class3_mask] = 3
        
        # {Class 1, 2} samples (f1 < 0)
        class12_mask = ~class3_mask
        
        if np.any(class12_mask):
            # Step 2: H2 decision on remaining samples
            X_remaining = X[class12_mask]
            f2 = self.h2.decision_function(X_remaining)
            
            # Class 2 samples (f2 >= 0)
            class2_mask = f2 >= 0
            predictions[class12_mask] = np.where(class2_mask, 2, 1)
        
        return predictions
    
    def get_model_summary(self) -> Dict:
        """Get summary of both classifiers.
        
        Returns:
            Dictionary with model information
        """
        if self.h1 is None or self.h2 is None:
            return {"status": "not_fitted"}
        
        return {
            "status": "fitted",
            "n_features": self.n_features,
            "h1": self.h1.get_solution_summary(),
            "h2": self.h2.get_solution_summary(),
        }
