#!/usr/bin/env python3
"""
Hierarchical Classifier for Multi-class Ordinal Classification

Implements a hierarchical (cascade) classifier using N-1 binary CE-SVM models
to solve N-class ordinal classification problems.

Four strategies are supported:

1. Single Filter (original, 3-class only):
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

2. Multiple Filter (3-class only):
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

3. Inverted (dynamic, 3-class only):
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

4. Test3 (fixed with balanced class weighting, supports N classes):
    For N classes, uses N-1 binary classifiers:

    H1: Class 1 (+1) vs {2,3,...,N} (-1)
    H2: {1,2} (+1) vs {3,4,...,N} (-1)
    H3: {1,2,3} (+1) vs {4,5,...,N} (-1)
    ...
    H(N-1): {1,2,...,N-1} (+1) vs Class N (-1)

    Prediction Logic:
    - Hk = +1 → Class k
    - Hk = -1 → Continue to H(k+1)
    - H(N-1) = -1 → Class N

    Uses fixed classification rule with balanced class weighting:
    - Objective: min ... - (1/s_p)*l_p - (1/s_n)*l_n
    - Implementation: class_weight="balanced"
"""

from datetime import datetime, timezone
import numpy as np
from typing import Any, Callable, Dict, Optional, Tuple, List
from .binary_cesvm import BinaryCESVM


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


class HierarchicalCESVM:
    """Hierarchical Cost-Effective SVM for N-class ordinal classification."""

    def __init__(self, cesvm_params: Optional[Dict] = None, strategy: str = "multiple_filter", n_classes: Optional[int] = None):
        """Initialize Hierarchical CE-SVM.

        Args:
            cesvm_params: Parameters for binary CE-SVM models (shared)
            strategy: Classification strategy
                - "single_filter": Class 3 vs {1,2}, then Class 2 vs Class 1 (3-class only)
                - "multiple_filter": Class 1 vs {2,3}, then Class {1,2} vs Class 3 (3-class only)
                - "inverted": medium vs {majority, minority}, then {medium, majority} vs minority (3-class only)
                - "test3": Class 1 vs {2,...,N}, then {1,2} vs {3,...,N}, ... (supports N classes)
            n_classes: Number of classes (None = auto-detect from training data)
        """
        if strategy not in ["single_filter", "multiple_filter", "inverted", "test3"]:
            raise ValueError(f"Unknown strategy: {strategy}. "
                            f"Must be 'single_filter', 'multiple_filter', 'inverted', or 'test3'")

        self.cesvm_params = cesvm_params or {}
        self.strategy = strategy
        self.n_classes = n_classes

        # Binary classifiers (for N classes, need N-1 classifiers)
        # For 3-class strategies: h1, h2
        # For N-class strategies: classifiers = {'h1': ..., 'h2': ..., ..., 'h{N-1}': ...}
        self.h1 = None
        self.h2 = None
        self.classifiers = {}  # Used for N > 3

        # Training data info
        self.n_features = None

        # Class roles for inverted strategy
        self.class_roles = None

        # Training progress state
        self.expected_classifier_count = None
        self.completed_classifier_count = 0
        self.fit_stopped_early = False
        self.fit_stop_reason = None

    def _reset_training_state(self) -> None:
        """Reset trained classifier state before starting a new fit."""
        self.h1 = None
        self.h2 = None
        self.classifiers = {}
        self.n_features = None
        self.class_roles = None
        self.expected_classifier_count = None
        self.completed_classifier_count = 0
        self.fit_stopped_early = False
        self.fit_stop_reason = None

    def _prepare_fit(self, X_classes: Tuple[np.ndarray, ...]) -> None:
        """Validate inputs and initialize shared fit state."""
        detected_n_classes = len(X_classes)

        if self.n_classes is None:
            self.n_classes = detected_n_classes
        elif self.n_classes != detected_n_classes:
            raise ValueError(f"Expected {self.n_classes} classes, got {detected_n_classes}")

        if self.strategy in ["single_filter", "multiple_filter", "inverted"] and self.n_classes != 3:
            raise ValueError(f"Strategy '{self.strategy}' only supports 3 classes, got {self.n_classes}")

        self._reset_training_state()
        self.n_features = X_classes[0].shape[1]
        self.expected_classifier_count = 2 if self.n_classes == 3 else self.n_classes - 1

        print("=" * 60)
        print(f"Training Hierarchical CE-SVM (Strategy: {self.strategy})")
        print("=" * 60)
        for i, X_k in enumerate(X_classes, 1):
            print(f"Class {i}: {len(X_k)} samples")
        print(f"Features: {self.n_features}")
        print()

    def is_fully_fitted(self) -> bool:
        """Return whether all expected binary classifiers have been trained."""
        return (
            self.expected_classifier_count is not None
            and self.completed_classifier_count == self.expected_classifier_count
            and not self.fit_stopped_early
        )

    def _build_classifier_progress(
        self,
        *,
        hk: int,
        description: str,
        classifier: BinaryCESVM,
        classifier_started_at: datetime,
        fit_started_at: datetime,
        positive_sample_count: int,
        negative_sample_count: int,
    ) -> Dict[str, Any]:
        """Build callback progress payload for a completed classifier."""
        finished_at = _utc_now()
        summary = classifier.get_solution_summary()

        return {
            "component": f"h{hk}",
            "hk": hk,
            "n_classifiers": self.expected_classifier_count,
            "description": description,
            "started_at_utc": classifier_started_at,
            "finished_at_utc": finished_at,
            "elapsed_seconds": (finished_at - classifier_started_at).total_seconds(),
            "cumulative_elapsed_seconds": (finished_at - fit_started_at).total_seconds(),
            "weights": None if classifier.weights is None else np.asarray(classifier.weights, dtype=float).copy(),
            "b": None if classifier.intercept is None else float(classifier.intercept),
            "objective_value": summary.get("objective_value"),
            "positive_class_accuracy_lb": summary.get("positive_class_accuracy_lb"),
            "negative_class_accuracy_lb": summary.get("negative_class_accuracy_lb"),
            "mip_gap": summary.get("mip_gap"),
            "positive_sample_count": positive_sample_count,
            "negative_sample_count": negative_sample_count,
        }

    def _should_continue_after_classifier(
        self,
        after_classifier: Callable[[Dict[str, Any]], bool | None] | None,
        progress: Dict[str, Any],
    ) -> bool:
        """Invoke the incremental callback and normalize the return value."""
        if after_classifier is None:
            return True

        callback_result = after_classifier(progress)
        return callback_result is not False

    def _prepare_hk_data(
        self,
        k: int,
        X_classes: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare data for Hk classifier (test3 strategy, N-class).

        For test3 strategy with N classes:
        - Hk: {1,2,...,k} (+1) vs {k+1,...,N} (-1)

        Args:
            k: Classifier index (1 to N-1)
            X_classes: List of feature matrices [X1, X2, ..., XN]

        Returns:
            X_hk: Combined feature matrix
            y_hk: Binary labels (+1 for classes 1...k, -1 for classes k+1...N)
        """
        N = len(X_classes)

        # Positive class: {1, 2, ..., k}
        X_pos_list = [X_classes[i] for i in range(k)]
        X_pos = np.vstack(X_pos_list) if len(X_pos_list) > 0 else np.empty((0, X_classes[0].shape[1]))

        # Negative class: {k+1, k+2, ..., N}
        X_neg_list = [X_classes[i] for i in range(k, N)]
        X_neg = np.vstack(X_neg_list) if len(X_neg_list) > 0 else np.empty((0, X_classes[0].shape[1]))

        y_pos = np.ones(len(X_pos))
        y_neg = -np.ones(len(X_neg))

        X_hk = np.vstack([X_pos, X_neg])
        y_hk = np.concatenate([y_pos, y_neg])

        return X_hk, y_hk

    def _determine_class_roles(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray
    ) -> Dict:
        """Determine class roles based on sample counts (3-class only).

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
            - test3: Class 1 (+1) vs {Class 2, 3} (-1) - fixed with balanced class weighting

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
        elif self.strategy == "test3":
            # Test3: Fixed rule - Class 1 (+1) vs {Class 2, 3} (-1)
            X_pos = X1  # Class 1 = +1
            X_neg = np.vstack([X2, X3])  # Class 2, 3 = -1
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
            - test3: {Class 1, 2} (+1) vs Class 3 (-1) - fixed with balanced class weighting

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
        elif self.strategy == "test3":
            # Test3: Fixed rule - {Class 1, 2} (+1) vs Class 3 (-1)
            X_pos = np.vstack([X1, X2])  # Class 1, 2 = +1
            X_neg = X3  # Class 3 = -1
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
        *X_classes: np.ndarray
    ) -> 'HierarchicalCESVM':
        """Fit the hierarchical classifier.

        Args:
            *X_classes: Variable number of class samples
                - For 3-class: X1, X2, X3
                - For N-class: X1, X2, ..., XN

        Returns:
            self
        """
        return self.fit_incremental(*X_classes)

    def fit_incremental(
        self,
        *X_classes: np.ndarray,
        after_classifier: Callable[[Dict[str, Any]], bool | None] | None = None,
    ) -> 'HierarchicalCESVM':
        """Fit the hierarchical classifier and emit progress after each classifier."""
        self._prepare_fit(X_classes)
        fit_started_at = _utc_now()

        if self.n_classes == 3:
            X1, X2, X3 = X_classes
            self._fit_3class(
                X1,
                X2,
                X3,
                after_classifier=after_classifier,
                fit_started_at=fit_started_at,
            )
        else:
            if self.strategy != "test3":
                raise ValueError(f"N-class (N={self.n_classes}) only supported for test3 strategy")
            self._fit_nclass(
                X_classes,
                after_classifier=after_classifier,
                fit_started_at=fit_started_at,
            )

        print("\n" + "=" * 60)
        if self.is_fully_fitted():
            print("Training Complete!")
        else:
            print("Training Stopped Early!")
            print(
                f"Completed classifiers: {self.completed_classifier_count}/{self.expected_classifier_count}"
            )
            if self.fit_stop_reason is not None:
                print(f"Stop reason: {self.fit_stop_reason}")
        print("=" * 60)

        return self

    def _fit_3class(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        X3: np.ndarray,
        *,
        after_classifier: Callable[[Dict[str, Any]], bool | None] | None = None,
        fit_started_at: datetime,
    ) -> None:
        """Fit 3-class hierarchical classifier (original logic).

        Args:
            X1: Class 1 samples
            X2: Class 2 samples
            X3: Class 3 samples
        """
        # For inverted strategy, determine and store class roles
        if self.strategy == "inverted":
            self.class_roles = self._determine_class_roles(X1, X2, X3)
            print(f"Dynamic Class Roles:")
            print(f"  Majority:  Class {self.class_roles['majority']} ({len(self.class_roles['X_majority'])} samples)")
            print(f"  Medium:    Class {self.class_roles['medium']} ({len(self.class_roles['X_medium'])} samples)")
            print(f"  Minority:  Class {self.class_roles['minority']} ({len(self.class_roles['X_minority'])} samples)")
            print()

        # Determine H1 description based on strategy
        if self.strategy == "single_filter":
            h1_desc = "Class 3 (+1) vs {Class 1, 2} (-1)"
        elif self.strategy == "multiple_filter":
            h1_desc = "Class 1 (+1) vs {Class 2, 3} (-1)"
        elif self.strategy == "test3":
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

        # Determine accuracy_mode for H1 (test3 strategy only)
        h1_params = self.cesvm_params.copy()
        if self.strategy == "test3":
            # Test3: Use balanced class weighting, always use "both" accuracy mode
            h1_params['class_weight'] = "balanced"
            h1_params['accuracy_mode'] = "both"
            print(f"  Class weight: balanced")
            print(f"  Accuracy mode: both")
        print()

        self.h1 = BinaryCESVM(**h1_params)
        h1_started_at = _utc_now()
        self.h1.fit(X_h1, y_h1)
        self.completed_classifier_count = 1

        print(f"\nH1 Solution:")
        h1_summary = self.h1.get_solution_summary()
        print(f"  Weights (w): {self.h1.weights}")
        print(f"  Intercept (b): {self.h1.intercept}")
        print(f"  Objective: {h1_summary['objective_value']:.6f}")
        print(f"  Selected features: {h1_summary['n_selected_features']}/{self.n_features}")
        print(f"  L1 norm: {h1_summary['l1_norm']:.6f}")
        print(f"  Positive accuracy lb: {h1_summary['positive_class_accuracy_lb']:.4f}")
        print(f"  Negative accuracy lb: {h1_summary['negative_class_accuracy_lb']:.4f}")

        h1_progress = self._build_classifier_progress(
            hk=1,
            description=h1_desc,
            classifier=self.h1,
            classifier_started_at=h1_started_at,
            fit_started_at=fit_started_at,
            positive_sample_count=int(np.sum(y_h1 == 1)),
            negative_sample_count=int(np.sum(y_h1 == -1)),
        )

        if not self._should_continue_after_classifier(after_classifier, h1_progress):
            self.fit_stopped_early = True
            self.fit_stop_reason = "after_classifier callback requested stop after h1"
            print("\nStopping after H1 due to after_classifier callback.")
            return

        # Determine H2 description based on strategy
        if self.strategy == "single_filter":
            h2_desc = "Class 2 (+1) vs Class 1 (-1)"
        elif self.strategy == "multiple_filter":
            h2_desc = "Class {1,2} (+1) vs Class 3 (-1)"
        elif self.strategy == "test3":
            h2_desc = "Class {1, 2} (+1) vs Class 3 (-1)"
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

        # Determine accuracy_mode for H2 (test3 strategy only)
        h2_params = self.cesvm_params.copy()
        if self.strategy == "test3":
            # Test3: Use balanced class weighting, always use "both" accuracy mode
            h2_params['class_weight'] = "balanced"
            h2_params['accuracy_mode'] = "both"
            print(f"  Class weight: balanced")
            print(f"  Accuracy mode: both")
        print()

        self.h2 = BinaryCESVM(**h2_params)
        h2_started_at = _utc_now()
        self.h2.fit(X_h2, y_h2)
        self.completed_classifier_count = 2

        print(f"\nH2 Solution:")
        h2_summary = self.h2.get_solution_summary()
        print(f"  Weights (w): {self.h2.weights}")
        print(f"  Intercept (b): {self.h2.intercept}")
        print(f"  Objective: {h2_summary['objective_value']:.6f}")
        print(f"  Selected features: {h2_summary['n_selected_features']}/{self.n_features}")
        print(f"  L1 norm: {h2_summary['l1_norm']:.6f}")
        print(f"  Positive accuracy lb: {h2_summary['positive_class_accuracy_lb']:.4f}")
        print(f"  Negative accuracy lb: {h2_summary['negative_class_accuracy_lb']:.4f}")

        h2_progress = self._build_classifier_progress(
            hk=2,
            description=h2_desc,
            classifier=self.h2,
            classifier_started_at=h2_started_at,
            fit_started_at=fit_started_at,
            positive_sample_count=int(np.sum(y_h2 == 1)),
            negative_sample_count=int(np.sum(y_h2 == -1)),
        )

        if (
            not self._should_continue_after_classifier(after_classifier, h2_progress)
            and self.completed_classifier_count < self.expected_classifier_count
        ):
            self.fit_stopped_early = True
            self.fit_stop_reason = "after_classifier callback requested stop after h2"

    def _fit_nclass(
        self,
        X_classes: Tuple[np.ndarray, ...],
        *,
        after_classifier: Callable[[Dict[str, Any]], bool | None] | None = None,
        fit_started_at: datetime,
    ) -> None:
        """Fit N-class hierarchical classifier (test3 strategy only).

        Args:
            X_classes: Tuple of class samples (X1, X2, ..., XN)
        """
        N = len(X_classes)

        # Train N-1 binary classifiers
        for k in range(1, N):
            # Hk: {1,...,k} vs {k+1,...,N}
            pos_classes = list(range(1, k+1))
            neg_classes = list(range(k+1, N+1))

            h_desc = f"Class {{{', '.join(map(str, pos_classes))}}} (+1) vs Class {{{', '.join(map(str, neg_classes))}}} (-1)"

            print("=" * 60)
            print(f"Training H{k}: {h_desc}")
            print("=" * 60)

            X_hk, y_hk = self._prepare_hk_data(k, X_classes)
            print(f"H{k} Training samples: {len(X_hk)}")
            print(f"  Positive (+1): {np.sum(y_hk == 1)} samples")
            print(f"  Negative (-1): {np.sum(y_hk == -1)} samples")

            # Test3: Use balanced class weighting, always use "both" accuracy mode
            hk_params = self.cesvm_params.copy()
            hk_params['class_weight'] = "balanced"
            hk_params['accuracy_mode'] = "both"
            print(f"  Class weight: balanced")
            print(f"  Accuracy mode: both")
            print()

            classifier = BinaryCESVM(**hk_params)
            classifier_started_at = _utc_now()
            classifier.fit(X_hk, y_hk)

            self.classifiers[f'h{k}'] = classifier
            self.completed_classifier_count = len(self.classifiers)

            print(f"\nH{k} Solution:")
            hk_summary = classifier.get_solution_summary()
            print(f"  Weights (w): {classifier.weights}")
            print(f"  Intercept (b): {classifier.intercept}")
            print(f"  Objective: {hk_summary['objective_value']:.6f}")
            print(f"  Selected features: {hk_summary['n_selected_features']}/{self.n_features}")
            print(f"  L1 norm: {hk_summary['l1_norm']:.6f}")
            print(f"  Positive accuracy lb: {hk_summary['positive_class_accuracy_lb']:.4f}")
            print(f"  Negative accuracy lb: {hk_summary['negative_class_accuracy_lb']:.4f}")
            print()

            progress = self._build_classifier_progress(
                hk=k,
                description=h_desc,
                classifier=classifier,
                classifier_started_at=classifier_started_at,
                fit_started_at=fit_started_at,
                positive_sample_count=int(np.sum(y_hk == 1)),
                negative_sample_count=int(np.sum(y_hk == -1)),
            )

            should_continue = self._should_continue_after_classifier(after_classifier, progress)
            if not should_continue and self.completed_classifier_count < self.expected_classifier_count:
                self.fit_stopped_early = True
                self.fit_stop_reason = f"after_classifier callback requested stop after h{k}"
                print(f"Stopping after H{k} due to after_classifier callback.")
                break

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using hierarchical decision rule.

        Decision Rule depends on strategy and number of classes:

        For 3-class strategies (single_filter, multiple_filter, inverted):
            [Same as before]

        For N-class strategy (test3):
            1. Compute f1(x) = w1*x + b1
            2. If f1(x) >= 0: predict Class 1
            3. Else:
                a. Compute f2(x) = w2*x + b2
                b. If f2(x) >= 0: predict Class 2
                c. Else: continue to H3, ..., H(N-1)
            4. If H(N-1)(x) >= 0: predict Class N-1
            5. Else: predict Class N

        Args:
            X: Feature matrix (n_samples, n_features)

        Returns:
            Predicted labels (n_samples,), values in {1, 2, ..., N}
        """
        if not self.is_fully_fitted():
            if self.completed_classifier_count == 0:
                raise RuntimeError("Model not fitted. Call fit() first.")
            raise RuntimeError(
                f"Model not fully fitted. Completed "
                f"{self.completed_classifier_count}/{self.expected_classifier_count} classifiers."
            )

        if self.n_classes == 3:
            # Use original 3-class prediction logic
            return self._predict_3class(X)
        else:
            # Use N-class prediction logic
            return self._predict_nclass(X)

    def _predict_3class(self, X: np.ndarray) -> np.ndarray:
        """Predict for 3-class strategies (original logic).

        Args:
            X: Feature matrix

        Returns:
            Predicted labels in {1, 2, 3}
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
        elif self.strategy == "test3":
            # Test3: Fixed prediction rule
            # H1: f1 >= 0 -> Class 1, f1 < 0 -> H2
            # H2: f2 >= 0 -> Class 2, f2 < 0 -> Class 3
            first_class = 1  # f1 >= 0 -> Class 1
            remaining_classes = (2, 3)  # For H2: f2 >= 0 -> Class 2, f2 < 0 -> Class 3
        else:  # inverted
            # f1 >= 0 --> medium class
            # f1 < 0  --> proceed to H2
            first_class = self.class_roles['medium']
            remaining_classes = (self.class_roles['majority'], self.class_roles['minority'])

        # Samples classified by H1
        h1_pos_mask = f1 >= 0

        # Standard behavior for all strategies
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

    def _predict_nclass(self, X: np.ndarray) -> np.ndarray:
        """Predict for N-class test3 strategy.

        Prediction logic:
        - H1 >= 0 → Class 1
        - H1 < 0, H2 >= 0 → Class 2
        - H1 < 0, H2 < 0, H3 >= 0 → Class 3
        - ...
        - H1 < 0, ..., H(N-1) < 0 → Class N

        Args:
            X: Feature matrix

        Returns:
            Predicted labels in {1, 2, ..., N}
        """
        if len(self.classifiers) == 0:
            raise RuntimeError("Model not fitted. Call fit() first.")

        n_samples = X.shape[0]
        predictions = np.zeros(n_samples, dtype=int)
        remaining_mask = np.ones(n_samples, dtype=bool)

        # Iterate through classifiers H1, H2, ..., H(N-1)
        for k in range(1, self.n_classes):
            classifier = self.classifiers[f'h{k}']

            # Only evaluate remaining samples
            if np.any(remaining_mask):
                X_remaining = X[remaining_mask]
                fk = classifier.decision_function(X_remaining)

                # fk >= 0 → Class k
                hk_pos_mask = fk >= 0

                # Update predictions for samples classified as Class k
                remaining_indices = np.where(remaining_mask)[0]
                classified_indices = remaining_indices[hk_pos_mask]
                predictions[classified_indices] = k

                # Update remaining_mask
                remaining_mask[classified_indices] = False

        # Remaining samples → Class N
        predictions[remaining_mask] = self.n_classes

        return predictions

    
    def get_model_summary(self) -> Dict:
        """Get summary of all classifiers.

        Returns:
            Dictionary with model information
        """
        if self.expected_classifier_count is None or self.completed_classifier_count == 0:
            return {"status": "not_fitted"}

        status = "fitted" if self.is_fully_fitted() else "partially_fitted"

        if self.n_classes == 3:
            # Use original 3-class summary logic
            # Determine classifier descriptions based on strategy
            if self.strategy == "single_filter":
                h1_desc = "Class 3 vs {1,2}"
                h2_desc = "Class 2 vs Class 1"
            elif self.strategy == "multiple_filter":
                h1_desc = "Class 1 vs {2,3}"
                h2_desc = "Class {1,2} vs Class 3"
            elif self.strategy == "test3":
                h1_desc = "Class 1 vs {2,3}"
                h2_desc = "Class {1,2} vs Class 3"
            else:  # inverted
                h1_desc = f"Class {self.class_roles['medium']} vs {{Class {self.class_roles['majority']}, {self.class_roles['minority']}}}"
                h2_desc = f"Class {{Class {self.class_roles['medium']}, {self.class_roles['majority']}}} vs Class {self.class_roles['minority']}"

            summary = {
                "status": status,
                "strategy": self.strategy,
                "n_classes": self.n_classes,
                "n_features": self.n_features,
                "completed_classifier_count": self.completed_classifier_count,
                "expected_classifier_count": self.expected_classifier_count,
                "stop_reason": self.fit_stop_reason,
            }

            if self.h1 is not None:
                summary["h1"] = {
                    "description": h1_desc,
                    **self.h1.get_solution_summary()
                }
            if self.h2 is not None:
                summary["h2"] = {
                    "description": h2_desc,
                    **self.h2.get_solution_summary()
                }

            # Add class roles info for inverted strategy only
            if self.strategy == "inverted" and self.class_roles is not None:
                summary["class_roles"] = {
                    "majority": self.class_roles['majority'],
                    "medium": self.class_roles['medium'],
                    "minority": self.class_roles['minority']
                }

            return summary

        else:
            # N-class summary
            summary = {
                "status": status,
                "strategy": self.strategy,
                "n_classes": self.n_classes,
                "n_features": self.n_features,
                "completed_classifier_count": self.completed_classifier_count,
                "expected_classifier_count": self.expected_classifier_count,
                "stop_reason": self.fit_stop_reason,
                "classifiers": {}
            }

            for k in range(1, self.completed_classifier_count + 1):
                pos_classes = list(range(1, k+1))
                neg_classes = list(range(k+1, self.n_classes+1))
                h_desc = f"Class {{{', '.join(map(str, pos_classes))}}} vs Class {{{', '.join(map(str, neg_classes))}}}"

                summary["classifiers"][f"h{k}"] = {
                    "description": h_desc,
                    **self.classifiers[f'h{k}'].get_solution_summary()
                }

            return summary
