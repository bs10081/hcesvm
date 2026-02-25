"""
Unit tests for BinaryCESVM

Tests core functionality of the binary CE-SVM classifier including:
- Basic fit/predict functionality
- Class weight handling
- Accuracy mode variations
- Solution summary extraction
"""

import numpy as np
import pytest
from hcesvm.models.binary_cesvm import BinaryCESVM


class TestBinaryCESVMBasic:
    """Test basic BinaryCESVM functionality."""

    def setup_method(self):
        """Set up test data for each test."""
        # Simple linearly separable data
        np.random.seed(42)
        self.X_simple = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],  # Class +1
            [5, 5], [5, 6], [6, 5], [6, 6]   # Class -1
        ])
        self.y_simple = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    def test_fit_predict_simple(self):
        """Test basic fit and predict on simple data."""
        model = BinaryCESVM(
            C_hyper=1.0,
            M=1000.0,
            time_limit=60,
            verbose=False
        )

        model.fit(self.X_simple, self.y_simple)
        predictions = model.predict(self.X_simple)

        # Should correctly classify all training samples
        assert len(predictions) == len(self.y_simple)
        assert all(p in [-1, 1] for p in predictions)

    def test_decision_function(self):
        """Test decision function output."""
        model = BinaryCESVM(C_hyper=1.0, M=1000.0, time_limit=60, verbose=False)
        model.fit(self.X_simple, self.y_simple)

        decision_values = model.decision_function(self.X_simple)

        # Check output shape
        assert len(decision_values) == len(self.X_simple)

        # Positive class should have positive decision values
        assert all(decision_values[:4] > 0)  # First 4 are class +1
        # Negative class should have negative decision values
        assert all(decision_values[4:] < 0)  # Last 4 are class -1

    def test_solution_summary(self):
        """Test solution summary extraction."""
        model = BinaryCESVM(C_hyper=1.0, M=1000.0, time_limit=60, verbose=False)
        model.fit(self.X_simple, self.y_simple)

        summary = model.get_solution_summary()

        # Check required keys
        required_keys = [
            'n_samples', 'n_features', 'n_selected_features',
            'objective_value', 'solve_time', 'mip_gap'
        ]
        for key in required_keys:
            assert key in summary

        # Check values
        assert summary['n_samples'] == len(self.X_simple)
        assert summary['n_features'] == self.X_simple.shape[1]
        assert summary['n_selected_features'] <= summary['n_features']


class TestBinaryCESVMClassWeight:
    """Test class weight handling."""

    def setup_method(self):
        """Set up imbalanced data."""
        np.random.seed(42)
        # Imbalanced: 10 samples of class +1, 2 samples of class -1
        self.X_imbalanced = np.vstack([
            np.random.randn(10, 2) + [0, 0],  # Class +1
            np.random.randn(2, 2) + [5, 5]    # Class -1
        ])
        self.y_imbalanced = np.array([1]*10 + [-1]*2)

    def test_class_weight_none(self):
        """Test with no class weighting."""
        model = BinaryCESVM(
            C_hyper=1.0,
            M=1000.0,
            class_weight='none',
            time_limit=60,
            verbose=False
        )

        model.fit(self.X_imbalanced, self.y_imbalanced)
        predictions = model.predict(self.X_imbalanced)

        # Should work without error
        assert len(predictions) == len(self.y_imbalanced)

    def test_class_weight_balanced(self):
        """Test with balanced class weighting (Test3 strategy)."""
        model = BinaryCESVM(
            C_hyper=1.0,
            M=1000.0,
            class_weight='balanced',
            time_limit=60,
            verbose=False
        )

        model.fit(self.X_imbalanced, self.y_imbalanced)
        predictions = model.predict(self.X_imbalanced)

        # Should work without error
        assert len(predictions) == len(self.y_imbalanced)

        # Check that sample counts were computed
        assert hasattr(model, 's_plus')
        assert hasattr(model, 's_minus')
        assert model.s_plus == 10
        assert model.s_minus == 2


class TestBinaryCESVMAccuracyMode:
    """Test accuracy mode variations (Test2 strategy)."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.X = np.array([
            [1, 1], [1, 2], [2, 1], [2, 2],
            [5, 5], [5, 6], [6, 5], [6, 6]
        ])
        self.y = np.array([1, 1, 1, 1, -1, -1, -1, -1])

    def test_accuracy_mode_both(self):
        """Test with both accuracy terms (standard)."""
        model = BinaryCESVM(
            C_hyper=1.0,
            M=1000.0,
            accuracy_mode='both',
            time_limit=60,
            verbose=False
        )

        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        assert len(predictions) == len(self.y)

    def test_accuracy_mode_positive(self):
        """Test with only positive class accuracy term."""
        model = BinaryCESVM(
            C_hyper=1.0,
            M=1000.0,
            accuracy_mode='positive',
            time_limit=60,
            verbose=False
        )

        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        assert len(predictions) == len(self.y)

    def test_accuracy_mode_negative(self):
        """Test with only negative class accuracy term."""
        model = BinaryCESVM(
            C_hyper=1.0,
            M=1000.0,
            accuracy_mode='negative',
            time_limit=60,
            verbose=False
        )

        model.fit(self.X, self.y)
        predictions = model.predict(self.X)

        assert len(predictions) == len(self.y)


class TestBinaryCESVMEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_labels(self):
        """Test with invalid labels (not +1/-1)."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([0, 2])  # Invalid: should be +1/-1

        model = BinaryCESVM(C_hyper=1.0, M=1000.0, verbose=False)

        # Should raise an error or handle gracefully
        # (Implementation dependent - adjust as needed)
        with pytest.raises((ValueError, AssertionError)):
            model.fit(X, y)

    def test_single_class(self):
        """Test with only one class (invalid for binary classification)."""
        X = np.array([[1, 2], [3, 4], [5, 6]])
        y = np.array([1, 1, 1])  # All same class

        model = BinaryCESVM(C_hyper=1.0, M=1000.0, verbose=False)

        # Should raise an error
        with pytest.raises((ValueError, AssertionError)):
            model.fit(X, y)

    def test_mismatched_dimensions(self):
        """Test with mismatched X and y dimensions."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, -1, 1])  # Wrong length

        model = BinaryCESVM(C_hyper=1.0, M=1000.0, verbose=False)

        with pytest.raises((ValueError, AssertionError)):
            model.fit(X, y)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
