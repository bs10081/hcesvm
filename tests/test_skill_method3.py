"""Tests for the derived Method 3 skill split helpers."""

from __future__ import annotations

import numpy as np

from hcesvm.utils.skill_method3 import (
    calculate_proportional_class_targets,
    class_counts,
    derive_skill_method3_split,
    filter_and_relabel_classes,
    sample_by_class_targets,
)


def test_filter_and_relabel_classes_keeps_only_skill_classes_4_to_7():
    X = np.arange(21, dtype=float).reshape(7, 3)
    y = np.asarray([1, 2, 3, 4, 5, 6, 7], dtype=int)

    filtered_X, filtered_y = filter_and_relabel_classes(X, y)

    np.testing.assert_array_equal(filtered_X, X[3:])
    np.testing.assert_array_equal(filtered_y, np.asarray([1, 2, 3, 4], dtype=int))


def test_calculate_proportional_class_targets_matches_cementscale_plan():
    assert calculate_proportional_class_targets([649, 642, 497, 28], 798) == [285, 282, 219, 12]
    assert calculate_proportional_class_targets([162, 161, 124, 7], 200) == [71, 71, 55, 3]


def test_sample_by_class_targets_is_reproducible():
    X = np.arange(68, dtype=float).reshape(17, 4)
    y = np.asarray([1] * 6 + [2] * 5 + [3] * 4 + [4] * 2, dtype=int)
    targets = [3, 2, 1, 1]

    first_X, first_y = sample_by_class_targets(X, y, class_targets=targets, random_state=42)
    second_X, second_y = sample_by_class_targets(X, y, class_targets=targets, random_state=42)

    np.testing.assert_array_equal(first_X, second_X)
    np.testing.assert_array_equal(first_y, second_y)
    assert class_counts(first_y, n_classes=4) == targets


def test_derive_skill_method3_split_relabels_and_downsamples_both_sides():
    X_train = np.arange(33, dtype=float).reshape(11, 3)
    y_train = np.asarray([1, 2, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=int)
    X_test = np.arange(30, dtype=float).reshape(10, 3)
    y_test = np.asarray([1, 3, 4, 4, 5, 5, 6, 6, 7, 7], dtype=int)

    split = derive_skill_method3_split(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        source_url="https://example.test/skill.csv",
        train_target_size=6,
        test_target_size=6,
        random_state=42,
    )

    assert split.n_classes == 4
    assert split.n_features == 3
    assert split.train_original_counts == [2, 2, 2, 2]
    assert split.test_original_counts == [2, 2, 2, 2]
    assert split.train_sampled_counts == [2, 2, 1, 1]
    assert split.test_sampled_counts == [2, 2, 1, 1]
    assert class_counts(split.y_train, n_classes=4) == [2, 2, 1, 1]
    assert class_counts(split.y_test, n_classes=4) == [2, 2, 1, 1]
