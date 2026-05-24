"""Tests for multiclass SVOR/NPSVOR models."""

from __future__ import annotations

import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np

from hcesvm.models.npsvor import NPSVORQP
from hcesvm.models.svor import SVORImplicitQP
from hcesvm.utils.evaluator import calculate_accuracy
from hcesvm.utils.ordinal_data import (
    load_lingo_split_workbook,
    load_tabular_dataset_split,
    subset_samples_per_class,
)
from hcesvm.utils.ordinal_synthetic import make_staircase_ordinal_data


ROOT = Path(__file__).resolve().parents[1]
SVOR_BALANCE = ROOT / "data" / "svor" / "SVOR_balance_split.xlsx"
NPSVOR_BALANCE = ROOT / "data" / "npsvor" / "NPSVOR_balance_split.xlsx"
HAS_GUROBI = importlib.util.find_spec("gurobipy") is not None


def _accuracy(y_true, y_pred) -> float:
    return float(calculate_accuracy(np.asarray(y_true), np.asarray(y_pred)))


def _fit_or_skip_size_limited(model, X, y) -> None:
    try:
        model.fit(X, y)
    except RuntimeError as exc:
        if "size-limited gurobi license" in str(exc).lower():
            raise unittest.SkipTest(str(exc)) from exc
        raise


@unittest.skipUnless(HAS_GUROBI, "gurobipy is required for optimization tests.")
class WorkbookModelTests(unittest.TestCase):
    def test_svor_matches_reported_balance_accuracy(self) -> None:
        dataset = load_lingo_split_workbook(SVOR_BALANCE)
        model = SVORImplicitQP(C=1.0)

        _fit_or_skip_size_limited(model, dataset.X_train, dataset.y_train)

        train_accuracy = _accuracy(dataset.y_train, model.predict(dataset.X_train))
        test_accuracy = _accuracy(dataset.y_test, model.predict(dataset.X_test))

        self.assertEqual(len(model.classes_), 3)
        self.assertEqual(len(model.thresholds_), 2)
        self.assertAlmostEqual(train_accuracy, dataset.reported_train_accuracy or 0.0, places=3)
        self.assertAlmostEqual(test_accuracy, dataset.reported_test_accuracy or 0.0, places=3)

    def test_npsvor_matches_reported_balance_accuracy(self) -> None:
        dataset = load_lingo_split_workbook(NPSVOR_BALANCE)
        model = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.2, prediction_rule="min_distance")

        _fit_or_skip_size_limited(model, dataset.X_train, dataset.y_train)

        train_accuracy = _accuracy(dataset.y_train, model.predict(dataset.X_train))
        test_accuracy = _accuracy(dataset.y_test, model.predict(dataset.X_test))

        self.assertEqual(len(model.classes_), 3)
        self.assertEqual(len(model.hyperplanes_), 3)
        self.assertAlmostEqual(train_accuracy, dataset.reported_train_accuracy or 0.0, places=3)
        self.assertAlmostEqual(test_accuracy, dataset.reported_test_accuracy or 0.0, places=3)

    def test_balance_subsets_still_solve_end_to_end(self) -> None:
        svor_dataset = subset_samples_per_class(
            load_lingo_split_workbook(SVOR_BALANCE),
            train_limit=10,
            test_limit=10,
        )
        npsvor_dataset = subset_samples_per_class(
            load_lingo_split_workbook(NPSVOR_BALANCE),
            train_limit=10,
            test_limit=10,
        )

        svor = SVORImplicitQP(C=1.0)
        svor.fit(svor_dataset.X_train, svor_dataset.y_train)
        self.assertEqual(len(svor.predict(svor_dataset.X_test)), len(svor_dataset.X_test))

        npsvor = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.2)
        npsvor.fit(npsvor_dataset.X_train, npsvor_dataset.y_train)
        self.assertEqual(len(npsvor.predict(npsvor_dataset.X_test)), len(npsvor_dataset.X_test))

    def test_models_handle_four_classes(self) -> None:
        X = [[-6.0], [-5.0], [-4.0], [-1.0], [0.0], [1.0], [4.0], [5.0], [6.0], [9.0], [10.0], [11.0]]
        y = [1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4]

        svor = SVORImplicitQP(C=1.0)
        svor.fit(X, y)
        svor_predictions = svor.predict(X)

        npsvor = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.1)
        npsvor.fit(X, y)
        npsvor_predictions = npsvor.predict(X)

        self.assertEqual(len(svor.classes_), 4)
        self.assertEqual(len(svor.thresholds_), 3)
        self.assertGreaterEqual(_accuracy(y, svor_predictions), 0.75)

        self.assertEqual(len(npsvor.classes_), 4)
        self.assertEqual(len(npsvor.hyperplanes_), 4)
        self.assertGreaterEqual(_accuracy(y, npsvor_predictions), 0.75)

    def test_models_handle_five_nonconsecutive_labels(self) -> None:
        X, y = make_staircase_ordinal_data(
            num_classes=5,
            samples_per_class=5,
            labels=[10, 20, 30, 40, 50],
            two_dimensional=True,
        )

        svor = SVORImplicitQP(C=1.0, add_order_constraints=True)
        svor.fit(X, y)
        svor_predictions = svor.predict(X)

        self.assertEqual(svor.classes_, [10, 20, 30, 40, 50])
        self.assertEqual(len(svor.thresholds_), 4)
        self.assertEqual(sorted(svor.thresholds_), svor.prediction_thresholds_)
        self.assertGreaterEqual(_accuracy(y, svor_predictions), 0.99)

        npsvor_min = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.1, prediction_rule="min_distance")
        npsvor_min.fit(X, y)
        self.assertEqual(npsvor_min.classes_, [10, 20, 30, 40, 50])
        self.assertEqual(len(npsvor_min.hyperplanes_), 5)
        self.assertGreaterEqual(_accuracy(y, npsvor_min.predict(X)), 0.99)

        npsvor_ordered = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.1, prediction_rule="ordered_sum")
        npsvor_ordered.fit(X, y)
        self.assertEqual(npsvor_ordered.classes_, [10, 20, 30, 40, 50])
        self.assertEqual(len(npsvor_ordered.hyperplanes_), 5)
        self.assertGreaterEqual(_accuracy(y, npsvor_ordered.predict(X)), 0.99)

    def test_models_train_from_generic_csv_split(self) -> None:
        X_train, y_train = make_staircase_ordinal_data(
            num_classes=5,
            samples_per_class=3,
            labels=[10, 20, 30, 40, 50],
            two_dimensional=True,
        )
        X_test, y_test = make_staircase_ordinal_data(
            num_classes=5,
            samples_per_class=2,
            labels=[10, 20, 30, 40, 50],
            two_dimensional=True,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            train_path = Path(temp_dir) / "train.csv"
            test_path = Path(temp_dir) / "test.csv"

            for path, X_rows, y_rows in [(train_path, X_train, y_train), (test_path, X_test, y_test)]:
                with path.open("w", encoding="utf-8", newline="") as handle:
                    writer = csv.writer(handle)
                    writer.writerow(["f1", "f2", "label"])
                    for row, label in zip(X_rows, y_rows):
                        writer.writerow([row[0], row[1], label])

            dataset = load_tabular_dataset_split(train_path, test_path=test_path, target_column="label")

            svor = SVORImplicitQP(C=1.0, add_order_constraints=True)
            svor.fit(dataset.X_train, dataset.y_train)
            self.assertGreaterEqual(_accuracy(dataset.y_test, svor.predict(dataset.X_test)), 0.99)

            npsvor = NPSVORQP(C1=1.0, C2=1.0, epsilon=0.1, prediction_rule="ordered_sum")
            npsvor.fit(dataset.X_train, dataset.y_train)
            self.assertGreaterEqual(_accuracy(dataset.y_test, npsvor.predict(dataset.X_test)), 0.99)


if __name__ == "__main__":
    unittest.main()
