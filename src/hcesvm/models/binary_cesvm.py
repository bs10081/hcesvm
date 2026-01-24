#!/usr/bin/env python3
"""
Binary CE-SVM (Cost-Effective Support Vector Machine) Model

Implements a binary classification CE-SVM using Gurobi optimizer.
Features:
- L1 norm regularization (feature sparsity)
- Feature selection (no cost/budget constraints)
- Three-tier accuracy indicators
- Class-balanced accuracy optimization

Based on: CEAS_SVM1_SL_Par.lg4 LINGO model (simplified)
"""

import numpy as np
import gurobipy as gp
from gurobipy import GRB
from typing import Dict, Optional


class BinaryCESVM:
    """Binary Cost-Effective SVM with feature selection.
    
    Mathematical Model:
        min  ||w||_1 + C * Σ(α + β + ρ) - l_+ - l_-
        s.t. y_i * (w·x_i + b) >= 1 - ξ_i
             ξ_i <= M * α_i
             ξ_i <= 1 + M * β_i
             ξ_i <= 2 + M * ρ_i
             α_i >= β_i >= ρ_i
             w = w+ - w-, w+, w- >= 0
             (feature activation constraints, no cost/budget)
    """

    def __init__(
        self,
        C_hyper: float = 1.0,
        epsilon: float = 0.0001,
        M: float = 1000.0,
        enable_selection: bool = True,
        feat_upper_bound: float = 1000,
        feat_lower_bound: float = 0.0000001,
        time_limit: int = 600,
        mip_gap: float = 1e-4,
        threads: int = 0,
        verbose: bool = True
    ):
        """Initialize Binary CE-SVM model.
        
        Args:
            C_hyper: Slack penalty coefficient
            epsilon: Accuracy constraint tolerance
            M: Big-M constant for indicator constraints
            enable_selection: Enable feature selection
            feat_upper_bound: Upper bound for feature activation
            feat_lower_bound: Lower bound for feature activation
            time_limit: Gurobi solver time limit (seconds)
            mip_gap: Gurobi MIP gap tolerance
            threads: Number of threads (0 = all available)
            verbose: Whether to print solver output
        """
        self.C_hyper = C_hyper
        self.epsilon = epsilon
        self.M = M
        self.enable_selection = enable_selection
        self.feat_upper_bound = feat_upper_bound
        self.feat_lower_bound = feat_lower_bound
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.verbose = verbose

        # Solution storage
        self.weights = None
        self.intercept = None
        self.selected_features = None
        self.model = None
        self.solution = None

    def build_model(self, X: np.ndarray, y: np.ndarray) -> gp.Model:
        """Build Gurobi optimization model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,), values in {+1, -1}
            
        Returns:
            Gurobi model object
        """
        n, d = X.shape

        # Count positive and negative samples
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)

        if n_pos == 0 or n_neg == 0:
            raise ValueError("Both positive and negative samples are required")

        # Create Gurobi model
        model = gp.Model("Binary_CE_SVM")
        model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.mip_gap)
        model.setParam('OutputFlag', 1 if self.verbose else 0)
        model.setParam('Threads', self.threads)

        # === Decision Variables ===
        w_plus = model.addVars(d, lb=0, name="w_plus")
        w_minus = model.addVars(d, lb=0, name="w_minus")
        b = model.addVar(lb=-GRB.INFINITY, name="b")
        ksi = model.addVars(n, lb=0, name="ksi")
        alpha = model.addVars(n, vtype=GRB.BINARY, name="alpha")
        beta = model.addVars(n, vtype=GRB.BINARY, name="beta")
        rho = model.addVars(n, vtype=GRB.BINARY, name="rho")
        l_p = model.addVar(lb=0, ub=1, name="l_p")
        l_n = model.addVar(lb=0, ub=1, name="l_n")

        # Feature selection variables (optional)
        if self.enable_selection:
            v = model.addVars(d, vtype=GRB.BINARY, name="v")

        # === Objective Function ===
        obj_expr = (
            gp.quicksum(w_plus[j] + w_minus[j] for j in range(d))
            + self.C_hyper * gp.quicksum(alpha[i] + beta[i] + rho[i] for i in range(n))
            - l_p - l_n
        )
        model.setObjective(obj_expr, GRB.MINIMIZE)

        # === Constraints ===

        # 1. SVM separation constraints
        for i in range(n):
            model.addConstr(
                y[i] * (gp.quicksum((w_plus[j] - w_minus[j]) * X[i, j] for j in range(d)) + b)
                >= 1 - ksi[i],
                name=f"svm_sep_{i}"
            )

        # 2. Big-M constraints (three-tier)
        for i in range(n):
            model.addConstr(ksi[i] <= self.M * alpha[i], name=f"bigM1_{i}")
            model.addConstr(ksi[i] <= 1 + self.M * beta[i], name=f"bigM2_{i}")
            model.addConstr(ksi[i] <= 2 + self.M * rho[i], name=f"bigM3_{i}")

        # 3. Step Loss Constraint (Tier hierarchy)
        for i in range(n):
            model.addConstr(alpha[i] >= beta[i], name=f"tier1_{i}")
            model.addConstr(beta[i] >= rho[i], name=f"tier2_{i}")

        # 4. Accuracy lower bound constraints
        for i in range(n):
            model.addConstr(ksi[i] >= (1 + self.epsilon) * beta[i], name=f"acc_lb_{i}")

        # 5. Positive class accuracy constraint
        # LINGO: sum((1-beta[i])*(1+y[i])) >= l_p * sum(1+y[i])
        # (1+y[i]) = 2 for y=+1, 0 for y=-1 (selects positive class)
        model.addConstr(
            gp.quicksum((1 - beta[i]) * (1 + y[i]) for i in range(n))
            >= l_p * gp.quicksum(1 + y[i] for i in range(n)),
            name="pos_accuracy"
        )

        # 6. Negative class accuracy constraint
        # LINGO: sum((1-beta[i])*(1-y[i])) >= l_n * sum(1-y[i])
        # (1-y[i]) = 2 for y=-1, 0 for y=+1 (selects negative class)
        model.addConstr(
            gp.quicksum((1 - beta[i]) * (1 - y[i]) for i in range(n))
            >= l_n * gp.quicksum(1 - y[i] for i in range(n)),
            name="neg_accuracy"
        )

        # 7. Feature activation constraints (no cost/budget)
        if self.enable_selection:
            for j in range(d):
                model.addConstr(
                    w_plus[j] + w_minus[j] <= self.feat_upper_bound * v[j],
                    name=f"feat_upper_{j}"
                )
                model.addConstr(
                    w_plus[j] + w_minus[j] >= self.feat_lower_bound * v[j],
                    name=f"feat_lower_{j}"
                )

        model.update()
        self.model = model
        return model

    def solve(self) -> bool:
        """Solve the CE-SVM optimization model.
        
        Returns:
            True if optimal solution found, False otherwise
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        self.model.optimize()

        if self.model.status == GRB.OPTIMAL:
            self._extract_solution()
            return True
        elif self.model.status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            # Accept best solution found within time limit
            print(f"Time limit reached. Using best solution found (gap: {self.model.MIPGap:.2%})")
            self._extract_solution()
            return True
        elif self.model.status == GRB.INFEASIBLE:
            print("Model is infeasible. Computing IIS...")
            self.model.computeIIS()
            self.model.write("cesvm_infeasible.ilp")
            print("IIS written to cesvm_infeasible.ilp")
            return False
        else:
            print(f"Optimization ended with status {self.model.status}")
            return False

    def _extract_solution(self):
        """Extract solution from solved Gurobi model."""
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT]:
            return

        d = len([v for v in self.model.getVars() if v.VarName.startswith("w_plus")])

        w_plus_vals = np.array([self.model.getVarByName(f"w_plus[{j}]").X for j in range(d)])
        w_minus_vals = np.array([self.model.getVarByName(f"w_minus[{j}]").X for j in range(d)])
        self.weights = w_plus_vals - w_minus_vals
        self.intercept = self.model.getVarByName("b").X

        if self.enable_selection:
            v_vals = np.array([self.model.getVarByName(f"v[{j}]").X for j in range(d)])
            self.selected_features = v_vals > 0.5
        else:
            self.selected_features = np.ones(d, dtype=bool)

        self.solution = {
            'weights': self.weights,
            'intercept': self.intercept,
            'selected_features': self.selected_features,
            'objective_value': self.model.ObjVal,
            'l_p': self.model.getVarByName("l_p").X,
            'l_n': self.model.getVarByName("l_n").X,
            'n_selected_features': int(np.sum(self.selected_features)),
        }

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BinaryCESVM':
        """Fit the Binary CE-SVM model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,), values in {+1, -1}
            
        Returns:
            self
        """
        self.build_model(X, y)
        success = self.solve()
        if not success:
            raise RuntimeError("CE-SVM optimization failed")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict class labels using decision value sign.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Predicted labels (n_samples,), values in {+1, -1}
        """
        if self.weights is None or self.intercept is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        decision_values = X @ self.weights + self.intercept
        predictions = np.where(decision_values >= 0, 1, -1)
        return predictions

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        """Compute decision function values.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Decision values (n_samples,)
        """
        if self.weights is None or self.intercept is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return X @ self.weights + self.intercept

    def get_solution_summary(self) -> Dict:
        """Get a summary of the solution.
        
        Returns:
            Dictionary with solution information
        """
        if self.solution is None:
            return {"status": "not_solved"}
        return {
            "status": "optimal",
            "objective_value": self.solution['objective_value'],
            "n_selected_features": self.solution['n_selected_features'],
            "selected_feature_indices": np.where(self.selected_features)[0].tolist(),
            "l1_norm": np.sum(np.abs(self.weights)),
            "positive_class_accuracy_lb": self.solution['l_p'],
            "negative_class_accuracy_lb": self.solution['l_n'],
            "intercept": self.intercept,
        }
