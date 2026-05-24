"""SVOR models implemented with Gurobi."""

from __future__ import annotations

from typing import Any, Sequence

import gurobipy as gp
from gurobipy import GRB

from ._ordinal_common import (
    _as_matrix,
    _configure_model,
    _dot,
    _encode_labels,
    _optimize_model,
)


class SVORImplicitQP:
    """Implicit-threshold linear SVOR that mirrors the provided LINGO workbook."""

    def __init__(
        self,
        C: float = 1.0,
        *,
        add_order_constraints: bool = False,
        solver_params: dict[str, Any] | None = None,
    ) -> None:
        self.C = float(C)
        self.add_order_constraints = add_order_constraints
        self.solver_params = solver_params or {}

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> "SVORImplicitQP":
        X_matrix = _as_matrix(X)
        classes, y_index = _encode_labels(y)
        n_samples = len(X_matrix)
        n_features = len(X_matrix[0])
        n_thresholds = len(classes) - 1

        model = gp.Model("svor_implicit_multiclass")
        _configure_model(model, self.solver_params)

        weights = [model.addVar(lb=-GRB.INFINITY, name=f"w[{j}]") for j in range(n_features)]
        thresholds = [model.addVar(lb=-GRB.INFINITY, name=f"b[{t}]") for t in range(n_thresholds)]
        slacks = {
            (i, t): model.addVar(lb=0.0, name=f"xi[{i},{t}]")
            for i in range(n_samples)
            for t in range(n_thresholds)
        }

        regularizer = 0.5 * gp.quicksum(weight * weight for weight in weights)
        hinge_loss = self.C * gp.quicksum(slacks.values())
        model.setObjective(regularizer + hinge_loss, GRB.MINIMIZE)

        for i, row in enumerate(X_matrix):
            score = gp.quicksum(weights[j] * row[j] for j in range(n_features))
            for threshold_index in range(n_thresholds):
                if y_index[i] <= threshold_index:
                    model.addConstr(
                        score - thresholds[threshold_index] <= -1.0 + slacks[i, threshold_index],
                        name=f"lower[{i},{threshold_index}]",
                    )
                else:
                    model.addConstr(
                        score - thresholds[threshold_index] >= 1.0 - slacks[i, threshold_index],
                        name=f"upper[{i},{threshold_index}]",
                    )

        if self.add_order_constraints:
            for threshold_index in range(n_thresholds - 1):
                model.addConstr(
                    thresholds[threshold_index] <= thresholds[threshold_index + 1],
                    name=f"threshold_order[{threshold_index}]",
                )

        _optimize_model(model)

        self.classes_ = classes
        self.weights_ = [float(weight.X) for weight in weights]
        self.thresholds_ = [float(threshold.X) for threshold in thresholds]
        self.prediction_thresholds_ = sorted(self.thresholds_)
        self.objective_value_ = float(model.ObjVal)
        self.model_status_ = int(model.Status)
        return self

    def decision_function(self, X: Sequence[Sequence[float]]) -> list[float]:
        if not hasattr(self, "weights_"):
            raise RuntimeError("The model must be fitted before calling decision_function().")
        return [_dot(self.weights_, row) for row in _as_matrix(X)]

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        scores = self.decision_function(X)
        predictions: list[Any] = []
        for score in scores:
            class_index = sum(1 for threshold in self.prediction_thresholds_ if score >= threshold)
            predictions.append(self.classes_[class_index])
        return predictions
