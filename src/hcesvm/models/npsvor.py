"""NPSVOR models implemented with Gurobi."""

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


class NPSVORQP:
    """Nonparallel SVOR that mirrors the provided LINGO workbook."""

    def __init__(
        self,
        C1: float = 1.0,
        C2: float = 1.0,
        epsilon: float = 0.2,
        *,
        prediction_rule: str = "min_distance",
        solver_params: dict[str, Any] | None = None,
    ) -> None:
        self.C1 = float(C1)
        self.C2 = float(C2)
        self.epsilon = float(epsilon)
        self.prediction_rule = prediction_rule
        self.solver_params = solver_params or {}

    def fit(self, X: Sequence[Sequence[float]], y: Sequence[Any]) -> "NPSVORQP":
        X_matrix = _as_matrix(X)
        classes, y_index = _encode_labels(y)
        n_features = len(X_matrix[0])
        n_classes = len(classes)

        hyperplanes: list[list[float]] = []
        biases: list[float] = []
        objective_values: list[float] = []
        statuses: list[int] = []

        for rank in range(n_classes):
            rank_model = gp.Model(f"npsvor_rank_{rank}")
            _configure_model(rank_model, self.solver_params)

            weights = [rank_model.addVar(lb=-GRB.INFINITY, name=f"w[{rank},{j}]") for j in range(n_features)]
            bias = rank_model.addVar(lb=-GRB.INFINITY, name=f"b[{rank}]")

            same_rank = [index for index, label in enumerate(y_index) if label == rank]
            lower_rank = [index for index, label in enumerate(y_index) if label < rank]
            upper_rank = [index for index, label in enumerate(y_index) if label > rank]

            xi_plus = {index: rank_model.addVar(lb=0.0, name=f"xi_plus[{rank},{index}]") for index in same_rank}
            xi_minus = {index: rank_model.addVar(lb=0.0, name=f"xi_minus[{rank},{index}]") for index in same_rank}
            eta_lower = {index: rank_model.addVar(lb=0.0, name=f"eta_lower[{rank},{index}]") for index in lower_rank}
            eta_upper = {index: rank_model.addVar(lb=0.0, name=f"eta_upper[{rank},{index}]") for index in upper_rank}

            regularizer = 0.5 * gp.quicksum(weight * weight for weight in weights)
            tube_loss = self.C1 * gp.quicksum(xi_plus[index] + xi_minus[index] for index in same_rank)
            margin_loss = self.C2 * (
                gp.quicksum(eta_lower[index] for index in lower_rank)
                + gp.quicksum(eta_upper[index] for index in upper_rank)
            )
            rank_model.setObjective(regularizer + tube_loss + margin_loss, GRB.MINIMIZE)

            for index in same_rank:
                score = gp.quicksum(weights[j] * X_matrix[index][j] for j in range(n_features)) + bias
                rank_model.addConstr(-self.epsilon - xi_minus[index] <= score, name=f"tube_low[{rank},{index}]")
                rank_model.addConstr(score <= self.epsilon + xi_plus[index], name=f"tube_high[{rank},{index}]")

            for index in lower_rank:
                score = gp.quicksum(weights[j] * X_matrix[index][j] for j in range(n_features)) + bias
                rank_model.addConstr(score <= -1.0 + eta_lower[index], name=f"lower_margin[{rank},{index}]")

            for index in upper_rank:
                score = gp.quicksum(weights[j] * X_matrix[index][j] for j in range(n_features)) + bias
                rank_model.addConstr(score >= 1.0 - eta_upper[index], name=f"upper_margin[{rank},{index}]")

            _optimize_model(rank_model)

            hyperplanes.append([float(weight.X) for weight in weights])
            biases.append(float(bias.X))
            objective_values.append(float(rank_model.ObjVal))
            statuses.append(int(rank_model.Status))

        self.classes_ = classes
        self.hyperplanes_ = hyperplanes
        self.biases_ = biases
        self.objective_values_ = objective_values
        self.model_statuses_ = statuses
        return self

    def decision_function(self, X: Sequence[Sequence[float]]) -> list[list[float]]:
        if not hasattr(self, "hyperplanes_"):
            raise RuntimeError("The model must be fitted before calling decision_function().")

        X_matrix = _as_matrix(X)
        return [
            [_dot(self.hyperplanes_[rank], row) + self.biases_[rank] for rank in range(len(self.classes_))]
            for row in X_matrix
        ]

    def predict(self, X: Sequence[Sequence[float]]) -> list[Any]:
        scores = self.decision_function(X)
        predictions: list[Any] = []

        for row in scores:
            if self.prediction_rule == "min_distance":
                class_index = min(range(len(row)), key=lambda index: abs(row[index]))
            elif self.prediction_rule == "ordered_sum":
                class_index = sum(1 for index in range(len(row) - 1) if row[index] + row[index + 1] > 0)
            else:
                raise ValueError(
                    f"Unsupported prediction_rule={self.prediction_rule!r}. "
                    "Use 'min_distance' or 'ordered_sum'."
                )

            predictions.append(self.classes_[class_index])

        return predictions
