"""CLI helpers for running SVOR and NPSVOR on ordinal datasets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from .models.npsvor import NPSVORQP
from .models.svor import SVORImplicitQP
from .utils.evaluator import calculate_accuracy
from .utils.ordinal_data import (
    OrdinalDatasetSplit,
    load_lingo_split_workbook,
    load_tabular_dataset_split,
    subset_samples_per_class,
)


def _summarize_model(model: Any, dataset: OrdinalDatasetSplit, model_name: str) -> dict[str, Any]:
    train_predictions = model.predict(dataset.X_train)
    test_predictions = model.predict(dataset.X_test)

    summary: dict[str, Any] = {
        "model": model_name,
        "data_source": str(dataset.workbook_path),
        "workbook": str(dataset.workbook_path),
        "classes": list(model.classes_),
        "feature_names": dataset.feature_names,
        "train_samples": len(dataset.X_train),
        "test_samples": len(dataset.X_test),
        "train_accuracy": float(calculate_accuracy(np.asarray(dataset.y_train), np.asarray(train_predictions))),
        "test_accuracy": float(calculate_accuracy(np.asarray(dataset.y_test), np.asarray(test_predictions))),
        "reported_train_accuracy": dataset.reported_train_accuracy,
        "reported_test_accuracy": dataset.reported_test_accuracy,
    }

    if isinstance(model, SVORImplicitQP):
        summary["weights"] = model.weights_
        summary["thresholds"] = model.thresholds_
    else:
        summary["hyperplanes"] = model.hyperplanes_
        summary["biases"] = model.biases_
        summary["prediction_rule"] = model.prediction_rule

    return summary


def run_workbook(
    workbook_path: str | Path,
    *,
    model_name: str,
    C: float = 1.0,
    C1: float = 1.0,
    C2: float = 1.0,
    epsilon: float = 0.2,
    prediction_rule: str = "min_distance",
    add_order_constraints: bool = False,
    max_samples_per_class: int | None = None,
    solver_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = load_lingo_split_workbook(workbook_path)
    if max_samples_per_class is not None:
        dataset = subset_samples_per_class(
            dataset,
            train_limit=max_samples_per_class,
            test_limit=max_samples_per_class,
        )

    if model_name == "svor":
        model = SVORImplicitQP(
            C=C,
            add_order_constraints=add_order_constraints,
            solver_params=solver_params,
        )
    elif model_name == "npsvor":
        model = NPSVORQP(
            C1=C1,
            C2=C2,
            epsilon=epsilon,
            prediction_rule=prediction_rule,
            solver_params=solver_params,
        )
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}.")

    model.fit(dataset.X_train, dataset.y_train)
    return _summarize_model(model, dataset, model_name)


def run_tabular(
    train_path: str | Path,
    *,
    model_name: str,
    test_path: str | Path | None = None,
    target_column: str | None = None,
    feature_columns: list[str] | None = None,
    train_sheet: str | None = None,
    test_sheet: str | None = None,
    delimiter: str | None = None,
    C: float = 1.0,
    C1: float = 1.0,
    C2: float = 1.0,
    epsilon: float = 0.2,
    prediction_rule: str = "min_distance",
    add_order_constraints: bool = False,
    max_samples_per_class: int | None = None,
    solver_params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    dataset = load_tabular_dataset_split(
        train_path,
        test_path=test_path,
        target_column=target_column,
        feature_columns=feature_columns,
        train_sheet=train_sheet,
        test_sheet=test_sheet,
        delimiter=delimiter,
    )
    if max_samples_per_class is not None:
        dataset = subset_samples_per_class(
            dataset,
            train_limit=max_samples_per_class,
            test_limit=max_samples_per_class,
        )

    if model_name == "svor":
        model = SVORImplicitQP(
            C=C,
            add_order_constraints=add_order_constraints,
            solver_params=solver_params,
        )
    elif model_name == "npsvor":
        model = NPSVORQP(
            C1=C1,
            C2=C2,
            epsilon=epsilon,
            prediction_rule=prediction_rule,
            solver_params=solver_params,
        )
    else:
        raise ValueError(f"Unsupported model_name={model_name!r}.")

    model.fit(dataset.X_train, dataset.y_train)
    return _summarize_model(model, dataset, model_name)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Gurobi SVOR/NPSVOR on ordinal datasets.")
    parser.add_argument("--model", choices=["svor", "npsvor"], required=True)
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument("--workbook", help="Path to the LINGO-generated Train/Test workbook.")
    source_group.add_argument("--train-file", help="Path to a generic CSV/XLSX/XLSM training file.")
    parser.add_argument("--test-file", help="Optional generic CSV/XLSX/XLSM test file.")
    parser.add_argument("--train-sheet", help="Optional sheet name for generic Excel training data.")
    parser.add_argument("--test-sheet", help="Optional sheet name for generic Excel test data.")
    parser.add_argument("--target-column", help="Target column name for generic CSV/Excel input.")
    parser.add_argument(
        "--feature-columns",
        help="Comma-separated feature column names for generic CSV/Excel input. Defaults to all numeric columns except the target.",
    )
    parser.add_argument("--delimiter", help="Optional CSV delimiter override, for example ';' or '\\t'.")
    parser.add_argument("--json-out", help="Optional path for a JSON summary.")
    parser.add_argument("--C", type=float, default=1.0, help="SVOR penalty parameter.")
    parser.add_argument("--C1", type=float, default=1.0, help="NPSVOR in-class tube penalty.")
    parser.add_argument("--C2", type=float, default=1.0, help="NPSVOR out-of-class margin penalty.")
    parser.add_argument("--epsilon", type=float, default=0.2, help="NPSVOR epsilon tube width.")
    parser.add_argument(
        "--prediction-rule",
        choices=["min_distance", "ordered_sum"],
        default="min_distance",
        help="NPSVOR prediction rule. min_distance matches the provided workbook behavior.",
    )
    parser.add_argument(
        "--add-order-constraints",
        action="store_true",
        help="Add explicit monotone threshold constraints to the SVOR model.",
    )
    parser.add_argument(
        "--max-samples-per-class",
        type=int,
        help="Optional per-class cap for smoke testing under a size-limited license.",
    )

    args = parser.parse_args(argv)
    feature_columns = (
        [part.strip() for part in args.feature_columns.split(",") if part.strip()]
        if args.feature_columns
        else None
    )

    try:
        if args.workbook is not None:
            summary = run_workbook(
                args.workbook,
                model_name=args.model,
                C=args.C,
                C1=args.C1,
                C2=args.C2,
                epsilon=args.epsilon,
                prediction_rule=args.prediction_rule,
                add_order_constraints=args.add_order_constraints,
                max_samples_per_class=args.max_samples_per_class,
            )
        else:
            summary = run_tabular(
                args.train_file,
                model_name=args.model,
                test_path=args.test_file,
                target_column=args.target_column,
                feature_columns=feature_columns,
                train_sheet=args.train_sheet,
                test_sheet=args.test_sheet,
                delimiter=args.delimiter,
                C=args.C,
                C1=args.C1,
                C2=args.C2,
                epsilon=args.epsilon,
                prediction_rule=args.prediction_rule,
                add_order_constraints=args.add_order_constraints,
                max_samples_per_class=args.max_samples_per_class,
            )
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    json_text = json.dumps(summary, indent=2, ensure_ascii=False)
    print(json_text)

    if args.json_out:
        output_path = Path(args.json_out).expanduser().resolve()
        output_path.write_text(json_text + "\n", encoding="utf-8")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
