"""Shared helpers for ordinal Gurobi models."""

from __future__ import annotations

from typing import Any, Sequence

import gurobipy as gp
from gurobipy import GRB


def _as_matrix(X: Sequence[Sequence[float]]) -> list[list[float]]:
    """Convert array-like input into a validated dense matrix."""
    matrix = [[float(value) for value in row] for row in X]
    if len(matrix) == 0:
        raise ValueError("X must contain at least one row.")

    n_features = len(matrix[0])
    if n_features == 0:
        raise ValueError("X must contain at least one feature.")

    for row in matrix:
        if len(row) != n_features:
            raise ValueError("All rows in X must have the same number of features.")

    return matrix


def _ordered_unique(values: Sequence[Any]) -> list[Any]:
    try:
        return sorted(set(values))
    except TypeError:
        ordered: list[Any] = []
        seen: set[Any] = set()
        for value in values:
            if value not in seen:
                seen.add(value)
                ordered.append(value)
        return ordered


def _encode_labels(y: Sequence[Any]) -> tuple[list[Any], list[int]]:
    labels = list(y)
    if len(labels) == 0:
        raise ValueError("y must contain at least one label.")

    classes = _ordered_unique(labels)
    if len(classes) < 2:
        raise ValueError("At least two ordinal classes are required.")

    lookup = {label: index for index, label in enumerate(classes)}
    return classes, [lookup[label] for label in labels]


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    return sum(l_value * r_value for l_value, r_value in zip(left, right))


def _configure_model(model: gp.Model, solver_params: dict[str, Any] | None) -> None:
    model.Params.OutputFlag = 0
    if not solver_params:
        return

    for name, value in solver_params.items():
        setattr(model.Params, name, value)


def _ensure_optimal(model: gp.Model) -> None:
    if model.Status not in {GRB.OPTIMAL, GRB.SUBOPTIMAL}:
        raise RuntimeError(f"Gurobi did not solve the model successfully. Status={model.Status}.")


def _optimize_model(model: gp.Model) -> None:
    try:
        model.optimize()
    except gp.GurobiError as exc:
        message = str(exc)
        if "size-limited license" in message.lower():
            raise RuntimeError(
                "The model is larger than the currently active size-limited Gurobi license. "
                "Activate your full license with grbgetkey and rerun, or solve a smaller per-class subset for smoke tests."
            ) from exc
        raise

    _ensure_optimal(model)
