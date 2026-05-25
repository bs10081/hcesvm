"""
Unit tests for HierarchicalCESVM

Tests the hierarchical multi-class classifier including:
- 3-class strategies (single_filter, multiple_filter, inverted, test3, test4)
- N-class test3/test4 strategies
- Prediction logic and model summaries
"""

import numpy as np
import pytest
from hcesvm.models.binary_cesvm import BinaryCESVM
from hcesvm.models.hierarchical import HierarchicalCESVM


class TestHierarchicalCESVMFixedStrategies:
    """Test fixed classification strategies (3-class)."""

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


class TestHierarchicalCESVMDynamicStrategies:
    """Test dynamic classification strategies (3-class)."""

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
        assert hasattr(model, 'class_roles')
        assert model.class_roles['minority'] == 1  # Smallest count
        assert model.class_roles['majority'] == 2  # Largest count

    def test_test3_strategy(self):
        """Test test3 strategy: Fixed grouping with balanced class weighting."""
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

        # Test3 uses fixed grouping, no dynamic class detection
        assert not hasattr(model, 'class_roles') or model.class_roles is None
        assert model.h1.objective_variant == "standard"
        assert model.h2.objective_variant == "standard"
        assert model.h1.class_weight == "balanced"
        assert model.h2.class_weight == "balanced"

    def test_test4_strategy(self):
        """Test test4 strategy: test3 grouping with normalized indicator penalty."""
        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test4'
        )

        model.fit(self.X1, self.X2, self.X3)
        predictions = model.predict(self.X_test)

        assert len(predictions) == len(self.y_test)
        assert all(p in [1, 2, 3] for p in predictions)
        assert not hasattr(model, 'class_roles') or model.class_roles is None
        assert model.h1.objective_variant == "test4"
        assert model.h2.objective_variant == "test4"
        assert model.h1.class_weight == "balanced"
        assert model.h2.class_weight == "balanced"


class TestHierarchicalCESVMNClass:
    """Test N-class test3/test4 strategies (N > 3)."""

    def test_5class_test3_strategy(self):
        """Test test3 strategy with 5 classes."""
        np.random.seed(42)
        # Create 5 well-separated classes
        X1 = np.random.randn(8, 2) + [0, 0]
        X2 = np.random.randn(8, 2) + [5, 0]
        X3 = np.random.randn(8, 2) + [10, 0]
        X4 = np.random.randn(8, 2) + [15, 0]
        X5 = np.random.randn(8, 2) + [20, 0]

        # Test data
        X_test = np.array([
            [0, 0],   # Class 1
            [5, 0],   # Class 2
            [10, 0],  # Class 3
            [15, 0],  # Class 4
            [20, 0]   # Class 5
        ])

        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test3',
            n_classes=5
        )

        model.fit(X1, X2, X3, X4, X5)

        # Check model properties
        assert model.n_classes == 5
        assert len(model.classifiers) == 4  # N-1 classifiers

        # Check predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [1, 2, 3, 4, 5] for p in predictions)

        # Check model summary
        summary = model.get_model_summary()
        assert summary['status'] == 'fitted'
        assert summary['n_classes'] == 5
        assert 'classifiers' in summary
        assert len(summary['classifiers']) == 4
        assert all(classifier.objective_variant == "standard" for classifier in model.classifiers.values())

    def test_5class_test4_strategy(self):
        """Test test4 strategy with 5 classes."""
        np.random.seed(42)
        X1 = np.random.randn(8, 2) + [0, 0]
        X2 = np.random.randn(8, 2) + [5, 0]
        X3 = np.random.randn(8, 2) + [10, 0]
        X4 = np.random.randn(8, 2) + [15, 0]
        X5 = np.random.randn(8, 2) + [20, 0]

        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test4',
            n_classes=5
        )

        model.fit(X1, X2, X3, X4, X5)

        assert model.n_classes == 5
        assert len(model.classifiers) == 4
        assert all(classifier.objective_variant == "test4" for classifier in model.classifiers.values())

    def test_7class_test3_strategy(self):
        """Test test3 strategy with 7 classes."""
        np.random.seed(42)
        # Create 7 well-separated classes
        X1 = np.random.randn(6, 2) + [0, 0]
        X2 = np.random.randn(6, 2) + [5, 0]
        X3 = np.random.randn(6, 2) + [10, 0]
        X4 = np.random.randn(6, 2) + [15, 0]
        X5 = np.random.randn(6, 2) + [20, 0]
        X6 = np.random.randn(6, 2) + [25, 0]
        X7 = np.random.randn(6, 2) + [30, 0]

        # Test data
        X_test = np.array([
            [0, 0],   # Class 1
            [5, 0],   # Class 2
            [10, 0],  # Class 3
            [15, 0],  # Class 4
            [20, 0],  # Class 5
            [25, 0],  # Class 6
            [30, 0]   # Class 7
        ])

        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test3',
            n_classes=7
        )

        model.fit(X1, X2, X3, X4, X5, X6, X7)

        # Check model properties
        assert model.n_classes == 7
        assert len(model.classifiers) == 6  # N-1 classifiers

        # Check predictions
        predictions = model.predict(X_test)
        assert len(predictions) == len(X_test)
        assert all(p in [1, 2, 3, 4, 5, 6, 7] for p in predictions)

        # Check model summary
        summary = model.get_model_summary()
        assert summary['status'] == 'fitted'
        assert summary['n_classes'] == 7
        assert 'classifiers' in summary
        assert len(summary['classifiers']) == 6


class TestHierarchicalCESVMPredictionLogic:
    """Test prediction logic for different strategies."""

    def setup_method(self):
        """Set up simple test data."""
        np.random.seed(42)
        self.X1 = np.array([[0, 0], [0.5, 0.5]])
        self.X2 = np.array([[5, 5], [5.5, 5.5]])
        self.X3 = np.array([[10, 10], [10.5, 10.5]])

    def test_model_summary(self):
        """Test model summary extraction."""
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
        summary = model.get_model_summary()

        # Check structure
        assert 'strategy' in summary
        assert 'h1' in summary
        assert 'h2' in summary

        # Check H1 and H2 summaries
        assert 'objective_value' in summary['h1']
        assert 'objective_value' in summary['h2']


class TestHierarchicalCESVMIncrementalFit:
    """Test incremental training behavior for N-class test3/test4."""

    def setup_method(self):
        """Create 5-class data for incremental-fit tests."""
        np.random.seed(7)
        self.X_classes = tuple(
            np.random.randn(4, 2) + [offset, 0]
            for offset in [0, 5, 10, 15, 20]
        )
        self.X_test = np.array([[0.0, 0.0], [20.0, 0.0]])

    def _patch_binary_fit(self, monkeypatch):
        """Replace BinaryCESVM.fit with a deterministic fake solver."""
        call_counter = {'count': 0}

        def fake_fit(self, X, y):
            call_counter['count'] += 1
            hk = call_counter['count']
            n_features = X.shape[1]
            self.weights = np.full(n_features, float(hk))
            self.intercept = float(hk)
            self.selected_features = np.ones(n_features, dtype=bool)
            self.solution = {
                'objective_value': float(hk),
                'n_selected_features': n_features,
                'l_p': 0.7,
                'l_n': 0.8,
                'n_support_vectors': 0,
                'n_margin_errors': 0,
                'mip_gap': 0.0,
                'solver_status': 2,
            }
            return self

        monkeypatch.setattr(BinaryCESVM, 'fit', fake_fit)
        return call_counter

    def test_fit_incremental_trains_all_classifiers_when_callback_continues(self, monkeypatch):
        """fit_incremental should emit progress for each Hk and finish normally."""
        self._patch_binary_fit(monkeypatch)
        progress_rows = []

        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test3',
            n_classes=5
        )

        model.fit_incremental(
            *self.X_classes,
            after_classifier=lambda progress: progress_rows.append(progress) or True,
        )

        assert model.is_fully_fitted() is True
        assert model.completed_classifier_count == 4
        assert len(model.classifiers) == 4
        assert [row['hk'] for row in progress_rows] == [1, 2, 3, 4]
        assert all(row['n_classifiers'] == 4 for row in progress_rows)
        assert all(row['elapsed_seconds'] >= 0 for row in progress_rows)

        summary = model.get_model_summary()
        assert summary['status'] == 'fitted'
        assert summary['completed_classifier_count'] == 4
        assert len(summary['classifiers']) == 4

    def test_fit_incremental_stops_cleanly_after_callback_requests_stop(self, monkeypatch):
        """Stopping after an Hk should leave a partial fit that cannot predict."""
        self._patch_binary_fit(monkeypatch)
        progress_rows = []

        def stop_after_second(progress):
            progress_rows.append(progress)
            return progress['hk'] < 2

        model = HierarchicalCESVM(
            cesvm_params={
                'C_hyper': 1.0,
                'M': 1000.0,
                'time_limit': 120,
                'verbose': False
            },
            strategy='test3',
            n_classes=5
        )

        model.fit_incremental(*self.X_classes, after_classifier=stop_after_second)

        assert model.is_fully_fitted() is False
        assert model.fit_stopped_early is True
        assert model.completed_classifier_count == 2
        assert len(model.classifiers) == 2
        assert [row['hk'] for row in progress_rows] == [1, 2]

        summary = model.get_model_summary()
        assert summary['status'] == 'partially_fitted'
        assert summary['completed_classifier_count'] == 2
        assert set(summary['classifiers']) == {'h1', 'h2'}

        with pytest.raises(RuntimeError, match="Model not fully fitted"):
            model.predict(self.X_test)


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
        'inverted',
        'test3',
        'test4'
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

        model = HierarchicalCESVM(strategy='test3')

        with pytest.raises((ValueError, AssertionError)):
            model.fit(X1, X2, X3)

    def test_empty_class(self):
        """Test with empty class (should work but may produce unexpected results)."""
        X1 = np.array([[1, 2], [3, 4]])
        X2 = np.array([]).reshape(0, 2)  # Empty!
        X3 = np.array([[5, 6], [7, 8]])

        model = HierarchicalCESVM(strategy='test3')

        # Should not raise error, but predictions may be meaningless
        # This is a degenerate case where one class has no samples
        model.fit(X1, X2, X3)

        # Model should still be able to predict
        predictions = model.predict(X1)
        assert len(predictions) == len(X1)
        assert all(p in [1, 2, 3] for p in predictions)

    def test_nclass_with_wrong_strategy(self):
        """Test N-class (N>3) with a 3-class-only strategy should fail."""
        np.random.seed(42)
        X1 = np.random.randn(5, 2)
        X2 = np.random.randn(5, 2)
        X3 = np.random.randn(5, 2)
        X4 = np.random.randn(5, 2)
        X5 = np.random.randn(5, 2)

        model = HierarchicalCESVM(strategy='multiple_filter', n_classes=5)

        with pytest.raises(ValueError, match="only supports 3 classes"):
            model.fit(X1, X2, X3, X4, X5)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
