"""
Unit tests for HierarchicalCESVM

Tests the hierarchical three-class classifier including:
- All 6 classification strategies
- Fixed strategies (single_filter, multiple_filter, class1_first)
- Dynamic strategies (inverted, test2, test3)
- Prediction logic and model summaries
"""

import numpy as np
import pytest
from hcesvm.models.hierarchical import HierarchicalCESVM


class TestHierarchicalCESVMFixedStrategies:
    """Test fixed classification strategies."""

    def setup_method(self):
        """Set up test data for three classes."""
        np.random.seed(42)
        # Three well-separated classes
        self.X1 = np.random.randn(10, 2) + [0, 0]   # Class 1
        self.X2 = np.random.randn(10, 2) + [5, 0]   # Class 2
        self.X3 = np.random.randn(10, 2) + [10, 0]  # Class 3

        # Test data (one sample from each class)
        self.X_test = np.array([
            [0, 0],   # Should be Class 1
            [5, 0],   # Should be Class 2
            [10, 0]   # Should be Class 3
        ])
        self.y_test = np.array([1, 2, 3])

    def test_single_filter_strategy(self):
        """Test single_filter strategy: Class 3 vs {1,2} -> Class 2 vs 1."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='single_filter'
        )

        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        # Check output
        assert len(predictions) == len(self.y_test)
        assert all(p in [1, 2, 3] for p in predictions)

    def test_multiple_filter_strategy(self):
        """Test multiple_filter strategy: Class 1 vs {2,3} -> {1,2} vs 3."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='multiple_filter'
        )

        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        # Check output
        assert len(predictions) == len(self.y_test)
        assert all(p in [1, 2, 3] for p in predictions)

    def test_class1_first_strategy(self):
        """Test class1_first strategy: Class 1 vs {2,3} -> Class 2 vs 3."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='class1_first'
        )

        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        # Check output
        assert len(predictions) == len(self.y_test)
        assert all(p in [1, 2, 3] for p in predictions)

        # Check prediction logic
        # H1 should separate Class 1 from {2, 3}
        # H2 should separate Class 2 from Class 3


class TestHierarchicalCESVMDynamicStrategies:
    """Test dynamic classification strategies."""

    def setup_method(self):
        """Set up imbalanced test data."""
        np.random.seed(42)
        # Imbalanced: Class 1 (minority), Class 2 (majority), Class 3 (medium)
        self.X1 = np.random.randn(5, 2) + [0, 0]    # Class 1 (minority)
        self.X2 = np.random.randn(20, 2) + [5, 0]   # Class 2 (majority)
        self.X3 = np.random.randn(10, 2) + [10, 0]  # Class 3 (medium)

        self.X_test = np.array([
            [0, 0],   # Class 1
            [5, 0],   # Class 2
            [10, 0]   # Class 3
        ])
        self.y_test = np.array([1, 2, 3])

    def test_inverted_strategy(self):
        """Test inverted strategy: Medium vs {Majority, Minority} -> {Med,Maj} vs Min."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='inverted'
        )

        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        # Check output
        assert len(predictions) == len(self.y_test)
        assert all(p in [1, 2, 3] for p in predictions)

        # Check class detection
        assert hasattr(model, 'minority_class')
        assert hasattr(model, 'majority_class')
        assert model.minority_class == 1  # Smallest count
        assert model.majority_class == 2  # Largest count

    def test_test2_strategy(self):
        """Test test2 strategy: Dynamic grouping with accuracy term removal."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test2'
        )

        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        # Check output
        assert len(predictions) == len(self.y_test)
        assert all(p in [1, 2, 3] for p in predictions)

        # Check class detection
        assert hasattr(model, 'minority_class')
        assert hasattr(model, 'majority_class')

        # Check accuracy mode (Test2 Rule: remove majority accuracy when Class 2)
        # Since Class 2 is majority here, one of the classifiers should have
        # accuracy_mode != 'both'

    def test_test3_strategy(self):
        """Test test3 strategy: Dynamic grouping with balanced class weighting."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test3'
        )

        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        # Check output
        assert len(predictions) == len(self.y_test)
        assert all(p in [1, 2, 3] for p in predictions)

        # Check class weight is set to balanced
        assert model.h1_clf.class_weight == 'balanced'
        assert model.h2_clf.class_weight == 'balanced'


class TestHierarchicalCESVMPredictionLogic:
    """Test prediction logic for different strategies."""

    def setup_method(self):
        """Set up simple test data."""
        np.random.seed(42)
        self.X1 = np.array([[0, 0], [0.5, 0.5]])
        self.X2 = np.array([[5, 5], [5.5, 5.5]])
        self.X3 = np.array([[10, 10], [10.5, 10.5]])

    def test_class1_first_prediction_flow(self):
        """Test class1_first prediction flow."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='class1_first'
        )

        model.fit(self.X1, self.X2, self.X3)

        # Test each class
        pred_1 = model.predict(self.X1[:1])
        pred_2 = model.predict(self.X2[:1])
        pred_3 = model.predict(self.X3[:1])

        assert pred_1[0] == 1  # Should predict Class 1
        assert pred_2[0] == 2  # Should predict Class 2
        assert pred_3[0] == 3  # Should predict Class 3

    def test_model_summary(self):
        """Test model summary extraction."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='class1_first'
        )

        model.fit(self.X1, self.X2, self.X3)
        summary = model.get_model_summary()

        # Check structure
        assert 'strategy' in summary
        assert 'h1_summary' in summary
        assert 'h2_summary' in summary

        # Check H1 and H2 summaries
        assert 'objective_value' in summary['h1_summary']
        assert 'objective_value' in summary['h2_summary']


class TestHierarchicalCESVMStrategyComparison:
    """Compare behavior across strategies."""

    def setup_method(self):
        """Set up balanced test data."""
        np.random.seed(42)
        # Balanced: equal samples for all classes
        self.X1 = np.random.randn(10, 2) + [0, 0]
        self.X2 = np.random.randn(10, 2) + [5, 0]
        self.X3 = np.random.randn(10, 2) + [10, 0]

        self.X_test = np.array([[0, 0], [5, 0], [10, 0]])

    @pytest.mark.parametrize('strategy', [
        'single_filter',
        'multiple_filter',
        'class1_first',
        'inverted',
        'test2',
        'test3'
    ])
    def test_all_strategies_work(self, strategy):
        """Test that all strategies can fit and predict."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy=strategy
        )

        # Should work without error
        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.X_test)
        assert all(p in [1, 2, 3] for p in predictions)


class TestHierarchicalCESVMEdgeCases:
    """Test edge cases and error handling."""

    def test_invalid_strategy(self):
        """Test with invalid strategy name."""
        with pytest.raises((ValueError, KeyError)):
            HierarchicalCESVM(strategy='invalid_strategy')

    def test_mismatched_dimensions(self):
        """Test with mismatched feature dimensions."""
        X1 = np.array([[1, 2]])     # 2 features
        X2 = np.array([[3, 4, 5]])  # 3 features (wrong!)
        X3 = np.array([[6, 7]])     # 2 features

        model = HierarchicalCESVM(strategy='class1_first')

        with pytest.raises((ValueError, AssertionError)):
            model.fit(X1, X2, X3)

    def test_empty_class(self):
        """Test with empty class (invalid)."""
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([]).reshape(0, 2)  # Empty!
        X3 = np.array([[5, 6], [7, 8]])

        model = HierarchicalCESVM(strategy='class1_first')

        with pytest.raises((ValueError, AssertionError)):
            model.fit(X1, X2, X3)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
