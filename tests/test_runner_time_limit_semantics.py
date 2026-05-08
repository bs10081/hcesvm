"""Tests for runner time-limit semantics."""

from __future__ import annotations

import io
import sys

import pytest

from examples import run_teaching_data_hcesvm_1000 as run_1000
from examples import run_teaching_data_three_models as three_models


class FakeClassifier:
    def __init__(self) -> None:
        self.weights = [0.25, -0.5]
        self.intercept = 1.5


class FakeHierarchicalCESVM:
    last_init: dict[str, object] | None = None

    def __init__(self, *, cesvm_params, strategy, n_classes) -> None:
        type(self).last_init = {
            "cesvm_params": dict(cesvm_params),
            "strategy": strategy,
            "n_classes": n_classes,
        }
        self.classifiers = {}

    def fit(self, *train_classes) -> None:
        self.classifiers = {
            "h1": FakeClassifier(),
            "h2": FakeClassifier(),
            "h3": FakeClassifier(),
        }

    def predict(self, X):
        return [1] * len(X)

    def get_model_summary(self):
        return {
            "classifiers": {
                "h1": {
                    "description": "Class {1} (+1) vs Class {2, 3, 4} (-1)",
                    "objective_value": 1.0,
                    "positive_class_accuracy_lb": 0.9,
                    "negative_class_accuracy_lb": 0.8,
                    "mip_gap": 0.0,
                },
                "h2": {
                    "description": "Class {1, 2} (+1) vs Class {3, 4} (-1)",
                    "objective_value": 2.0,
                    "positive_class_accuracy_lb": 0.85,
                    "negative_class_accuracy_lb": 0.75,
                    "mip_gap": 0.0,
                },
                "h3": {
                    "description": "Class {1, 2, 3} (+1) vs Class {4} (-1)",
                    "objective_value": 3.0,
                    "positive_class_accuracy_lb": 0.8,
                    "negative_class_accuracy_lb": 0.7,
                    "mip_gap": 0.0,
                },
            }
        }


def test_three_model_runner_preserves_explicit_per_classifier_time_limit(monkeypatch):
    monkeypatch.setattr(three_models, "HierarchicalCESVM", FakeHierarchicalCESVM)
    monkeypatch.setattr(
        three_models,
        "evaluate_multiclass",
        lambda y_true, y_pred, *, n_classes: {
            "total_accuracy": 1.0,
            **{f"class_{index}_accuracy": 1.0 for index in range(1, n_classes + 1)},
            **{f"class_{index}_count": 1 for index in range(1, n_classes + 1)},
        },
    )

    bundle = three_models.DatasetBundle(
        name="synthetic_ord",
        source_url="https://example.test/synthetic.csv",
        n_classes=4,
        n_features=2,
        train_classes=[[1], [2], [3], [4]],
        test_classes=[[1], [2], [3], [4]],
        X_train=[0, 1, 2, 3],
        y_train=[1, 2, 3, 4],
        X_test=[0, 1, 2, 3],
        y_test=[1, 2, 3, 4],
    )
    tee_buffer = io.StringIO()
    tee = three_models.TeeStream(tee_buffer)

    result = three_models.run_hcesvm(
        bundle,
        tee,
        {
            "C_hyper": 1.0,
            "M": 1000.0,
            "time_limit": 1800,
            "mip_gap": 1e-4,
            "threads": 2,
            "soft_mem_limit_gb": 18.0,
            "retain_raw_solution_arrays": False,
            "release_solver_resources_after_fit": True,
            "verbose": False,
        },
    )

    assert FakeHierarchicalCESVM.last_init is not None
    assert FakeHierarchicalCESVM.last_init["cesvm_params"]["time_limit"] == 1800
    assert result.status == "success"
    assert "HCESVM per-classifier time limit: 1800s" in tee_buffer.getvalue()
    assert "Total HCESVM time budget:" not in tee_buffer.getvalue()


def test_hcesvm_1000_runner_uses_time_limit_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_teaching_data_hcesvm_1000.py", "--time-limit", "900"])
    args = run_1000.parse_arguments()
    assert args.time_limit == 900


def test_hcesvm_1000_runner_rejects_legacy_total_time_limit_flag(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run_teaching_data_hcesvm_1000.py", "--total-time-limit", "900"])
    with pytest.raises(SystemExit):
        run_1000.parse_arguments()
