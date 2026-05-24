"""Tests for the derived 1000-sample teaching-data HCESVM helpers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from gurobipy import GRB

from hcesvm.utils.teaching_data_1000 import (
    CALIFORNIAHOUSING_1000_RECIPE,
    SKILL_METHOD3_4CLASS_1000_RECIPE,
    build_classifier_diagnostics_row,
    calculate_proportional_class_targets,
    derive_dataset_split,
    filter_and_relabel_classes,
)


def test_filter_and_relabel_classes_remaps_skill_method3_classes():
    X = np.arange(40, dtype=float).reshape(10, 4)
    y = np.asarray([1, 2, 3, 4, 5, 6, 7, 4, 6, 7], dtype=int)

    filtered_X, filtered_y = filter_and_relabel_classes(
        X,
        y,
        class_map=SKILL_METHOD3_4CLASS_1000_RECIPE.class_map,
    )

    np.testing.assert_array_equal(filtered_X, X[[3, 4, 5, 6, 7, 8, 9]])
    np.testing.assert_array_equal(filtered_y, np.asarray([1, 2, 3, 4, 1, 3, 4], dtype=int))


def test_recipe_target_counts_match_skill_plan():
    train_counts = calculate_proportional_class_targets([649, 642, 497, 28], 800)
    test_counts = calculate_proportional_class_targets([162, 161, 124, 7], 200)

    assert train_counts == list(SKILL_METHOD3_4CLASS_1000_RECIPE.expected_train_counts)
    assert test_counts == list(SKILL_METHOD3_4CLASS_1000_RECIPE.expected_test_counts)
    assert sum(train_counts) == 800
    assert sum(test_counts) == 200


def test_recipe_target_counts_match_californiahousing_plan():
    train_counts = calculate_proportional_class_targets(
        [2595, 5476, 4059, 2040, 1103, 1239],
        800,
    )
    test_counts = calculate_proportional_class_targets(
        [649, 1369, 1015, 509, 276, 310],
        200,
    )

    assert train_counts == list(CALIFORNIAHOUSING_1000_RECIPE.expected_train_counts)
    assert test_counts == list(CALIFORNIAHOUSING_1000_RECIPE.expected_test_counts)
    assert sum(train_counts) == 800
    assert sum(test_counts) == 200


def test_derive_dataset_split_is_reproducible_for_same_seed():
    X_train = np.arange(240, dtype=float).reshape(60, 4)
    y_train = np.asarray([1] * 18 + [2] * 16 + [3] * 14 + [4] * 12, dtype=int)
    X_test = np.arange(160, dtype=float).reshape(40, 4)
    y_test = np.asarray([1] * 12 + [2] * 10 + [3] * 10 + [4] * 8, dtype=int)

    recipe = SKILL_METHOD3_4CLASS_1000_RECIPE.with_overrides(
        train_target_size=20,
        test_target_size=12,
        class_map=None,
        expected_train_counts=(6, 5, 5, 4),
        expected_test_counts=(4, 3, 3, 2),
    )

    first = derive_dataset_split(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        source_url="https://example.test/synthetic.csv",
        recipe=recipe,
    )
    second = derive_dataset_split(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        source_url="https://example.test/synthetic.csv",
        recipe=recipe,
    )

    np.testing.assert_array_equal(first.X_train, second.X_train)
    np.testing.assert_array_equal(first.y_train, second.y_train)
    np.testing.assert_array_equal(first.X_test, second.X_test)
    np.testing.assert_array_equal(first.y_test, second.y_test)
    assert first.train_sampled_counts == [6, 5, 5, 4]
    assert first.test_sampled_counts == [4, 3, 3, 2]


@dataclass(slots=True)
class FakeClassifier:
    weights: np.ndarray
    intercept: float
    solution: dict[str, object]


def test_classifier_diagnostics_row_includes_solver_status_fields():
    progress = {
        "hk": 2,
        "description": "Class {1, 2} (+1) vs Class {3, 4} (-1)",
        "positive_sample_count": 120,
        "negative_sample_count": 80,
        "elapsed_seconds": 33.5,
        "cumulative_elapsed_seconds": 91.0,
    }
    classifier = FakeClassifier(
        weights=np.asarray([0.5, -1.25], dtype=float),
        intercept=0.75,
        solution={
            "objective_value": 12.34,
            "mip_gap": 0.015,
            "solver_status": GRB.TIME_LIMIT,
        },
    )

    row = build_classifier_diagnostics_row(
        dataset_name="skill_method3_4class_1000",
        progress=progress,
        classifier=classifier,
    )

    assert row["solver_status_code"] == int(GRB.TIME_LIMIT)
    assert row["solver_status_label"] == "time_limit_with_solution"
    assert row["classifier"] == "H2"
    assert row["weights"] == "[0.5, -1.25]"
    assert row["b"] == 0.75
