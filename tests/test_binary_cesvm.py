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
from gurobipy import GRB
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

    def objective_coefficients(
        self,
        *,
        class_weight: str,
        C_hyper: float,
        objective_variant: str = "standard",
    ) -> dict[str, float]:
        model = BinaryCESVM(
            C_hyper=C_hyper,
            M=1000.0,
            class_weight=class_weight,
            objective_variant=objective_variant,
            time_limit=60,
            verbose=False,
            release_solver_resources_after_fit=False,
        )
        try:
            gurobi_model = model.build_model(self.X_imbalanced, self.y_imbalanced)
            gurobi_model.update()
            objective = gurobi_model.getObjective()
            return {
                objective.getVar(index).VarName: objective.getCoeff(index)
                for index in range(objective.size())
            }
        finally:
            model.release_solver_resources()

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

    def test_class_weight_none_keeps_indicator_penalty_unscaled(self):
        """Standard strategies keep C as the alpha/beta/rho coefficient."""
        coeffs = self.objective_coefficients(class_weight='none', C_hyper=6.0)

        for var_name in ("alpha[0]", "beta[0]", "rho[0]"):
            assert coeffs[var_name] == pytest.approx(6.0)

    def test_class_weight_balanced_keeps_standard_indicator_penalty_unscaled(self):
        """Balanced class weighting alone does not activate the test4 objective."""
        coeffs = self.objective_coefficients(class_weight='balanced', C_hyper=6.0)

        for var_name in ("alpha[0]", "beta[0]", "rho[0]"):
            assert coeffs[var_name] == pytest.approx(6.0)

    def test_test4_objective_normalizes_indicator_penalty_by_sample_count(self):
        """The test4 model uses C / n_samples as the alpha/beta/rho coefficient."""
        coeffs = self.objective_coefficients(
            class_weight='balanced',
            C_hyper=6.0,
            objective_variant='test4',
        )

        for var_name in ("alpha[0]", "beta[0]", "rho[0]"):
            assert coeffs[var_name] == pytest.approx(0.5)


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

    def test_invalid_objective_variant(self):
        """Unknown objective variants should be rejected at construction time."""
        with pytest.raises(ValueError, match="objective_variant"):
            BinaryCESVM(objective_variant="patched_test3")

    def test_mismatched_dimensions(self):
        """Test with mismatched X and y dimensions."""
        X = np.array([[1, 2], [3, 4]])
        y = np.array([1, -1, 1])  # Wrong length

        model = BinaryCESVM(C_hyper=1.0, M=1000.0, verbose=False)

        with pytest.raises((ValueError, AssertionError)):
            model.fit(X, y)


class TestBinaryCESVMHeartbeat:
    """Test solver heartbeat configuration and formatting."""

    def test_mip_focus_must_be_gurobi_supported_value(self):
        model = BinaryCESVM(verbose=False, mip_focus=3)
        assert model.mip_focus == 3

        with pytest.raises(ValueError, match="mip_focus"):
            BinaryCESVM(verbose=False, mip_focus=4)

    def test_heartbeat_interval_must_be_positive(self):
        with pytest.raises(ValueError, match="heartbeat_interval_seconds"):
            BinaryCESVM(verbose=False, heartbeat_interval_seconds=0)

    def test_heartbeat_callback_emits_throttled_progress(self):
        class FakeCallbackModel:
            def __init__(self, values):
                self.values = values

            def cbGet(self, key):
                return self.values[key]

        messages = []
        model = BinaryCESVM(verbose=False, heartbeat_interval_seconds=10.0)
        model.heartbeat_label = "H3"
        callback = model._build_heartbeat_callback(printer=messages.append)

        fake_model = FakeCallbackModel(
            {
                GRB.Callback.RUNTIME: 5.0,
                GRB.Callback.MIP_NODCNT: 12.0,
                GRB.Callback.MIP_SOLCNT: 0.0,
                GRB.Callback.MIP_OBJBST: GRB.INFINITY,
                GRB.Callback.MIP_OBJBND: 41.25,
            }
        )
        callback(fake_model, GRB.Callback.MIP)

        assert len(messages) == 1
        assert "H3 heartbeat: stage=mip" in messages[0]
        assert "runtime=5.0s" in messages[0]
        assert "nodes=12" in messages[0]
        assert "solutions=0" in messages[0]
        assert "gap=<none>" in messages[0]

        fake_model.values[GRB.Callback.RUNTIME] = 12.0
        fake_model.values[GRB.Callback.MIP_NODCNT] = 18.0
        callback(fake_model, GRB.Callback.MIP)

        assert len(messages) == 1

        fake_model.values[GRB.Callback.RUNTIME] = 16.0
        fake_model.values[GRB.Callback.MIP_NODCNT] = 22.0
        fake_model.values[GRB.Callback.MIP_SOLCNT] = 1.0
        fake_model.values[GRB.Callback.MIP_OBJBST] = 50.0
        fake_model.values[GRB.Callback.MIP_OBJBND] = 45.0
        callback(fake_model, GRB.Callback.MIP)

        assert len(messages) == 2
        assert "runtime=16.0s" in messages[1]
        assert "solutions=1" in messages[1]
        assert "gap=10.00%" in messages[1]

    def test_heartbeat_callback_reports_presolve_stage(self):
        class FakeCallbackModel:
            def __init__(self, values):
                self.values = values

            def cbGet(self, key):
                return self.values[key]

        messages = []
        model = BinaryCESVM(verbose=False, heartbeat_interval_seconds=30.0)
        model.heartbeat_label = "H1"
        callback = model._build_heartbeat_callback(printer=messages.append)

        fake_model = FakeCallbackModel(
            {
                GRB.Callback.RUNTIME: 30.0,
                GRB.Callback.PRE_ROWDEL: 123.0,
                GRB.Callback.PRE_COLDEL: 7.0,
            }
        )
        callback(fake_model, GRB.Callback.PRESOLVE)

        assert len(messages) == 1
        assert "H1 heartbeat: stage=presolve" in messages[0]
        assert "rows_deleted=123" in messages[0]
        assert "columns_deleted=7" in messages[0]


class TestBinaryCESVMMemorySafety:
    """Test resource cleanup and lightweight solution retention."""

    def test_fit_releases_solver_resources_after_success(self, monkeypatch):
        """fit() should dispose model/env once extracted state is enough."""

        class FakeResource:
            def __init__(self):
                self.disposed = False

            def dispose(self):
                self.disposed = True

        def fake_build_model(self, X, y):
            self.model = FakeResource()
            self.env = FakeResource()
            self.n_samples = len(X)
            self.n_features = X.shape[1]
            self.s_plus = int(np.sum(y == 1))
            self.s_minus = int(np.sum(y == -1))
            return self.model

        def fake_solve(self):
            self.weights = np.array([1.0, -1.0])
            self.intercept = 0.5
            self.selected_features = np.array([True, True])
            self.solve_time = 0.25
            self.solution = {
                'weights': self.weights,
                'w_plus': np.array([1.0, 0.0]),
                'w_minus': np.array([0.0, 1.0]),
                'intercept': self.intercept,
                'selected_features': self.selected_features,
                'v': np.array([1.0, 1.0]),
                'l_p': 0.7,
                'l_n': 0.8,
                'objective_value': 1.5,
                'n_selected_features': 2,
                'n_support_vectors': 0,
                'n_margin_errors': 0,
                'n_samples': self.n_samples,
                'n_features': self.n_features,
                'solve_time': self.solve_time,
                'mip_gap': 0.0,
                'solver_status': GRB.OPTIMAL,
            }
            return True

        monkeypatch.setattr(BinaryCESVM, 'build_model', fake_build_model)
        monkeypatch.setattr(BinaryCESVM, 'solve', fake_solve)

        model = BinaryCESVM(verbose=False, release_solver_resources_after_fit=True)
        model.fit(np.array([[0.0, 0.0], [1.0, 1.0]]), np.array([1, -1]))

        assert model.model is None
        assert model.env is None
        assert model.predict(np.array([[2.0, 1.0]])).shape == (1,)

    def test_extract_solution_can_skip_large_raw_arrays(self):
        """retain_raw_solution_arrays=False should avoid storing per-sample arrays."""

        class FakeVar:
            def __init__(self, name, value):
                self.VarName = name
                self.X = value

        class FakeModel:
            def __init__(self):
                self.status = GRB.OPTIMAL
                self.Status = GRB.OPTIMAL
                self.ObjVal = 3.14
                self.MIPGap = 0.02
                self.MemUsed = 1.25
                self.MaxMemUsed = 2.5
                self._vars = {
                    "w_plus[0]": FakeVar("w_plus[0]", 1.0),
                    "w_plus[1]": FakeVar("w_plus[1]", 0.5),
                    "w_minus[0]": FakeVar("w_minus[0]", 0.0),
                    "w_minus[1]": FakeVar("w_minus[1]", 0.25),
                    "b": FakeVar("b", -0.75),
                    "l_p": FakeVar("l_p", 0.6),
                    "l_n": FakeVar("l_n", 0.7),
                    "v[0]": FakeVar("v[0]", 1.0),
                    "v[1]": FakeVar("v[1]", 1.0),
                    "ksi[0]": FakeVar("ksi[0]", 0.0),
                    "ksi[1]": FakeVar("ksi[1]", 1.5),
                    "alpha[0]": FakeVar("alpha[0]", 0.0),
                    "alpha[1]": FakeVar("alpha[1]", 1.0),
                    "beta[0]": FakeVar("beta[0]", 0.0),
                    "beta[1]": FakeVar("beta[1]", 1.0),
                    "rho[0]": FakeVar("rho[0]", 0.0),
                    "rho[1]": FakeVar("rho[1]", 0.0),
                }

            def getVars(self):
                return list(self._vars.values())

            def getVarByName(self, name):
                return self._vars[name]

        model = BinaryCESVM(verbose=False, retain_raw_solution_arrays=False, release_solver_resources_after_fit=False)
        model.model = FakeModel()
        model.n_samples = 2
        model.n_features = 2
        model._extract_solution()

        assert 'ksi' not in model.solution
        assert model.get_solution_summary()['max_mem_used_gb'] == 2.5
        assert np.allclose(model.get_weight_decomposition()['w_plus'], np.array([1.0, 0.5]))

        with pytest.raises(RuntimeError, match="retain_raw_solution_arrays=True"):
            model.get_slack_variables()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
