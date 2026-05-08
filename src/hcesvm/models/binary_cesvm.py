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

from datetime import datetime, timezone
from time import perf_counter
from typing import Callable, Dict, Optional

import gurobipy as gp
import numpy as np
from gurobipy import GRB


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
        time_limit: Optional[int] = 600,
        mip_gap: float = 1e-4,
        threads: int = 0,
        soft_mem_limit_gb: Optional[float] = None,
        verbose: bool = True,
        accuracy_mode: str = "both",
        class_weight: str = "none",
        heartbeat_interval_seconds: Optional[float] = None,
        retain_raw_solution_arrays: bool = True,
        release_solver_resources_after_fit: bool = True,
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
            soft_mem_limit_gb: Gurobi SoftMemLimit in GB (None = unlimited)
            verbose: Whether to print solver output
            accuracy_mode: Which accuracy bounds to include in objective
                          ("both", "positive_only", "negative_only")
            class_weight: Class weighting for accuracy terms
                         ("none": equal weight (default), "balanced": inverse of sample count)
            heartbeat_interval_seconds: Emit solver heartbeats this often while optimize()
                                        is still running (None = disabled)
            retain_raw_solution_arrays: Whether to keep per-sample raw solution arrays
            release_solver_resources_after_fit: Whether to dispose Gurobi model/env after fit
        """
        accuracy_mode_aliases = {
            "positive": "positive_only",
            "negative": "negative_only",
        }
        accuracy_mode = accuracy_mode_aliases.get(accuracy_mode, accuracy_mode)

        self.C_hyper = C_hyper
        self.epsilon = epsilon
        self.M = M
        self.enable_selection = enable_selection
        self.feat_upper_bound = feat_upper_bound
        self.feat_lower_bound = feat_lower_bound
        self.time_limit = time_limit
        self.mip_gap = mip_gap
        self.threads = threads
        self.soft_mem_limit_gb = soft_mem_limit_gb
        self.verbose = verbose
        self.accuracy_mode = accuracy_mode
        self.class_weight = class_weight
        self.retain_raw_solution_arrays = retain_raw_solution_arrays
        self.release_solver_resources_after_fit = release_solver_resources_after_fit

        # Validate accuracy_mode
        if accuracy_mode not in ["both", "positive_only", "negative_only"]:
            raise ValueError(
                f"accuracy_mode must be 'both', 'positive_only', or 'negative_only', "
                f"got '{accuracy_mode}'"
            )

        # Validate class_weight
        if class_weight not in ["none", "balanced"]:
            raise ValueError(
                f"class_weight must be 'none' or 'balanced', got '{class_weight}'"
            )

        if heartbeat_interval_seconds is not None:
            heartbeat_interval_seconds = float(heartbeat_interval_seconds)
            if (
                not np.isfinite(heartbeat_interval_seconds)
                or heartbeat_interval_seconds <= 0
            ):
                raise ValueError(
                    "heartbeat_interval_seconds must be a positive number or None"
                )

        # Solution storage
        self.weights = None
        self.intercept = None
        self.selected_features = None
        self.env = None
        self.model = None
        self.solution = None
        self.n_samples = None
        self.n_features = None
        self.s_plus = None
        self.s_minus = None
        self.solve_time = None
        self.heartbeat_interval_seconds = heartbeat_interval_seconds
        self.heartbeat_label = None

    def build_model(self, X: np.ndarray, y: np.ndarray) -> gp.Model:
        """Build Gurobi optimization model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,), values in {+1, -1}
            
        Returns:
            Gurobi model object
        """
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        if X.ndim != 2:
            raise ValueError(f"X must be a 2D array, got shape {X.shape}")
        if y.ndim != 1:
            raise ValueError(f"y must be a 1D array, got shape {y.shape}")
        if X.shape[0] != y.shape[0]:
            raise ValueError(
                f"X and y must have the same number of samples, got {X.shape[0]} and {y.shape[0]}"
            )

        n, d = X.shape
        self.n_samples = int(n)
        self.n_features = int(d)

        # Count positive and negative samples
        n_pos = np.sum(y == 1)
        n_neg = np.sum(y == -1)
        self.s_plus = int(n_pos)
        self.s_minus = int(n_neg)

        if n_pos == 0 or n_neg == 0:
            raise ValueError("Both positive and negative samples are required")

        # Create Gurobi model
        self.env = gp.Env()
        model = gp.Model("Binary_CE_SVM", env=self.env)
        if self.time_limit is not None:
            model.setParam('TimeLimit', self.time_limit)
        model.setParam('MIPGap', self.mip_gap)
        model.setParam('OutputFlag', 1 if self.verbose else 0)
        model.setParam('Threads', self.threads)
        if self.soft_mem_limit_gb is not None:
            model.setParam('SoftMemLimit', float(self.soft_mem_limit_gb))

        # === Decision Variables ===
        # w⁺ⱼ ∈ ℝ⁺, j = 1,...,d  (positive part of weight vector)
        w_plus = model.addVars(d, lb=0, name="w_plus")

        # w⁻ⱼ ∈ ℝ⁺, j = 1,...,d  (negative part of weight vector)
        # Note: w = w⁺ - w⁻, and ||w||₁ = Σⱼ(w⁺ⱼ + w⁻ⱼ)
        w_minus = model.addVars(d, lb=0, name="w_minus")

        # b ∈ ℝ  (intercept/bias term)
        b = model.addVar(lb=-GRB.INFINITY, name="b")

        # ξᵢ ∈ ℝ⁺, i = 1,...,n  (slack variables for margin violations)
        ksi = model.addVars(n, lb=0, name="ksi")

        # αᵢ ∈ {0,1}  (tier-1 indicator: αᵢ=1 if ξᵢ > 0)
        alpha = model.addVars(n, vtype=GRB.BINARY, name="alpha")

        # βᵢ ∈ {0,1}  (tier-2 indicator: βᵢ=1 if ξᵢ > 1, i.e., misclassified)
        beta = model.addVars(n, vtype=GRB.BINARY, name="beta")

        # ρᵢ ∈ {0,1}  (tier-3 indicator: ρᵢ=1 if ξᵢ > 2, i.e., severely misclassified)
        rho = model.addVars(n, vtype=GRB.BINARY, name="rho")

        # l⁺ ∈ [0,1]  (lower bound on positive class accuracy)
        l_p = model.addVar(lb=0, ub=1, name="l_p")

        # l⁻ ∈ [0,1]  (lower bound on negative class accuracy)
        l_n = model.addVar(lb=0, ub=1, name="l_n")

        # vⱼ ∈ {0,1}  (feature selection indicator: vⱼ=1 if feature j is selected)
        if self.enable_selection:
            v = model.addVars(d, vtype=GRB.BINARY, name="v")

        # === Objective Function ===
        #
        # Standard (accuracy_mode="both", class_weight="none"):
        #   min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - l⁺ - l⁻
        #        \_________/     \__________________/   \_____/
        #         ||w||₁          misclassification     accuracy
        #        (sparsity)         penalty            maximization
        #
        # Test2 Rule variants (class_weight="none"):
        #   accuracy_mode="positive_only":  ... - l⁺        (remove -l⁻)
        #   accuracy_mode="negative_only":  ... - l⁻        (remove -l⁺)
        #
        # Test3 (class_weight="balanced"):
        #   min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - (1/s⁺)·l⁺ - (1/s⁻)·l⁻
        #   Where s⁺ = |{i: yᵢ=+1}|, s⁻ = |{i: yᵢ=-1}|
        #   Gives higher weight to accuracy of minority class
        #
        obj_expr = (
            # Term 1: ||w||₁ = Σⱼ(w⁺ⱼ + w⁻ⱼ)  (L1 regularization for sparsity)
            gp.quicksum(w_plus[j] + w_minus[j] for j in range(d))
            # Term 2: C·Σᵢ(αᵢ + βᵢ + ρᵢ)  (three-tier misclassification penalty)
            + self.C_hyper * gp.quicksum(alpha[i] + beta[i] + rho[i] for i in range(n))
        )

        # Determine accuracy term weights based on class_weight parameter
        if self.class_weight == "balanced":
            # Test3: Use inverse of sample count as weight
            weight_pos = 1.0 / n_pos
            weight_neg = 1.0 / n_neg
        else:
            # Standard or Test2: Equal weights
            weight_pos = 1.0
            weight_neg = 1.0

        # Term 3: -weight_neg·l⁻  (maximize negative class accuracy lower bound)
        if self.accuracy_mode in ("both", "negative_only"):
            obj_expr -= weight_neg * l_n

        # Term 4: -weight_pos·l⁺  (maximize positive class accuracy lower bound)
        if self.accuracy_mode in ("both", "positive_only"):
            obj_expr -= weight_pos * l_p

        model.setObjective(obj_expr, GRB.MINIMIZE)

        # === Constraints ===

        # Constraint 1: SVM Separation (margin constraint)
        # yᵢ·(w·xᵢ + b) ≥ 1 - ξᵢ,  ∀i = 1,...,n
        # where w·xᵢ = Σⱼ(w⁺ⱼ - w⁻ⱼ)·xᵢⱼ
        for i in range(n):
            model.addConstr(
                y[i] * (gp.quicksum((w_plus[j] - w_minus[j]) * X[i, j] for j in range(d)) + b)
                >= 1 - ksi[i],
                name=f"svm_sep_{i}"
            )

        # Constraint 2: Big-M Constraints (three-tier indicator activation)
        # These constraints link slack variables ξᵢ to binary indicators αᵢ, βᵢ, ρᵢ
        #
        # Tier 1: ξᵢ ≤ M·αᵢ         (αᵢ=0 → ξᵢ=0, perfect classification)
        # Tier 2: ξᵢ ≤ 1 + M·βᵢ     (βᵢ=0 → ξᵢ≤1, within margin)
        # Tier 3: ξᵢ ≤ 2 + M·ρᵢ     (ρᵢ=0 → ξᵢ≤2, acceptable error)
        for i in range(n):
            model.addConstr(ksi[i] <= self.M * alpha[i], name=f"bigM1_{i}")
            model.addConstr(ksi[i] <= 1 + self.M * beta[i], name=f"bigM2_{i}")
            model.addConstr(ksi[i] <= 2 + self.M * rho[i], name=f"bigM3_{i}")

        # Constraint 3: Tier Hierarchy
        # αᵢ ≥ βᵢ ≥ ρᵢ,  ∀i
        # Ensures: if ξᵢ > 2 (ρᵢ=1), then ξᵢ > 1 (βᵢ=1), then ξᵢ > 0 (αᵢ=1)
        for i in range(n):
            model.addConstr(alpha[i] >= beta[i], name=f"tier1_{i}")
            model.addConstr(beta[i] >= rho[i], name=f"tier2_{i}")

        # Constraint 4: Accuracy Lower Bound Trigger
        # ξᵢ ≥ (1 + ε)·βᵢ,  ∀i
        # If βᵢ=1 (misclassified), then ξᵢ ≥ 1+ε (ensures ξᵢ > 1)
        for i in range(n):
            model.addConstr(ksi[i] >= (1 + self.epsilon) * beta[i], name=f"acc_lb_{i}")

        # Constraint 5: Positive Class Accuracy Lower Bound
        # Σᵢ[(1-βᵢ)·𝟙{yᵢ=+1}] ≥ l⁺·|{i: yᵢ=+1}|
        #
        # Using indicator trick: (1+yᵢ)/2 = 1 if yᵢ=+1, 0 if yᵢ=-1
        # Rewritten: Σᵢ[(1-βᵢ)·(1+yᵢ)] ≥ l⁺·Σᵢ(1+yᵢ)
        # LINGO: sum((1-beta[i])*(1+y[i])) >= l_p * sum(1+y[i])
        # (1+y[i]) = 2 for y=+1, 0 for y=-1 (selects positive class)
        model.addConstr(
            gp.quicksum((1 - beta[i]) * (1 + y[i]) for i in range(n))
            >= l_p * gp.quicksum(1 + y[i] for i in range(n)),
            name="pos_accuracy"
        )

        # Constraint 6: Negative Class Accuracy Lower Bound
        # Σᵢ[(1-βᵢ)·𝟙{yᵢ=-1}] ≥ l⁻·|{i: yᵢ=-1}|
        #
        # Using indicator trick: (1-yᵢ)/2 = 1 if yᵢ=-1, 0 if yᵢ=+1
        # Rewritten: Σᵢ[(1-βᵢ)·(1-yᵢ)] ≥ l⁻·Σᵢ(1-yᵢ)
        # LINGO: sum((1-beta[i])*(1-y[i])) >= l_n * sum(1-y[i])
        # (1-y[i]) = 2 for y=-1, 0 for y=+1 (selects negative class)
        model.addConstr(
            gp.quicksum((1 - beta[i]) * (1 - y[i]) for i in range(n))
            >= l_n * gp.quicksum(1 - y[i] for i in range(n)),
            name="neg_accuracy"
        )

        # Constraint 7: Feature Selection Bounds
        # If vⱼ = 0 (feature not selected): w⁺ⱼ + w⁻ⱼ = 0
        # If vⱼ = 1 (feature selected):     L ≤ w⁺ⱼ + w⁻ⱼ ≤ U
        #
        # Upper bound: w⁺ⱼ + w⁻ⱼ ≤ U·vⱼ
        # Lower bound: w⁺ⱼ + w⁻ⱼ ≥ L·vⱼ  (forces non-zero weight if selected)
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

    @staticmethod
    def _format_heartbeat_metric(
        value: float | int | None,
        *,
        decimals: int | None = None,
    ) -> str:
        """Format optional heartbeat metrics without leaking inf/nan."""
        if value is None:
            return "<none>"

        numeric_value = float(value)
        if (
            not np.isfinite(numeric_value)
            or abs(numeric_value) >= GRB.INFINITY / 10
        ):
            return "<none>"

        if decimals is None:
            if numeric_value.is_integer():
                return str(int(numeric_value))
            return f"{numeric_value:.6f}"

        return f"{numeric_value:.{decimals}f}"

    def _format_heartbeat_message(
        self,
        *,
        stage: str,
        runtime_seconds: float,
        node_count: float | None = None,
        solution_count: float | None = None,
        incumbent_objective: float | None = None,
        best_bound: float | None = None,
        rows_deleted: float | None = None,
        columns_deleted: float | None = None,
    ) -> str:
        """Build a compact heartbeat line for long-running solves."""
        timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        label = self.heartbeat_label or "BinaryCESVM"
        parts = [
            f"[{timestamp}] {label} heartbeat: stage={stage}",
            f"runtime={runtime_seconds:.1f}s",
        ]

        if stage == "presolve":
            parts.append(
                f"rows_deleted={self._format_heartbeat_metric(rows_deleted)}"
            )
            parts.append(
                f"columns_deleted={self._format_heartbeat_metric(columns_deleted)}"
            )
            return ", ".join(parts)

        parts.append(f"nodes={self._format_heartbeat_metric(node_count)}")
        parts.append(f"solutions={self._format_heartbeat_metric(solution_count)}")
        parts.append(
            f"incumbent={self._format_heartbeat_metric(incumbent_objective, decimals=6)}"
        )
        parts.append(
            f"best_bound={self._format_heartbeat_metric(best_bound, decimals=6)}"
        )

        gap = None
        if (
            solution_count is not None
            and float(solution_count) > 0
            and incumbent_objective is not None
            and best_bound is not None
            and np.isfinite(float(incumbent_objective))
            and np.isfinite(float(best_bound))
            and abs(float(incumbent_objective)) < GRB.INFINITY / 10
            and abs(float(best_bound)) < GRB.INFINITY / 10
        ):
            denominator = max(abs(float(incumbent_objective)), 1e-10)
            gap = abs(float(incumbent_objective) - float(best_bound)) / denominator

        if gap is None:
            parts.append("gap=<none>")
        else:
            parts.append(f"gap={gap:.2%}")

        return ", ".join(parts)

    def _build_heartbeat_callback(
        self,
        printer: Callable[[str], None] | None = None,
    ) -> Callable[[gp.Model, int], None] | None:
        """Create a throttled Gurobi callback for long-running solve heartbeats."""
        if self.heartbeat_interval_seconds is None:
            return None

        interval_seconds = float(self.heartbeat_interval_seconds)
        emit = print if printer is None else printer
        last_reported_runtime = -interval_seconds

        def callback(callback_model: gp.Model, where: int) -> None:
            nonlocal last_reported_runtime

            if where == GRB.Callback.PRESOLVE:
                runtime_seconds = float(callback_model.cbGet(GRB.Callback.RUNTIME))
                if runtime_seconds - last_reported_runtime < interval_seconds:
                    return
                last_reported_runtime = runtime_seconds
                emit(
                    self._format_heartbeat_message(
                        stage="presolve",
                        runtime_seconds=runtime_seconds,
                        rows_deleted=callback_model.cbGet(GRB.Callback.PRE_ROWDEL),
                        columns_deleted=callback_model.cbGet(GRB.Callback.PRE_COLDEL),
                    )
                )
                return

            if where != GRB.Callback.MIP:
                return

            runtime_seconds = float(callback_model.cbGet(GRB.Callback.RUNTIME))
            if runtime_seconds - last_reported_runtime < interval_seconds:
                return

            last_reported_runtime = runtime_seconds
            emit(
                self._format_heartbeat_message(
                    stage="mip",
                    runtime_seconds=runtime_seconds,
                    node_count=callback_model.cbGet(GRB.Callback.MIP_NODCNT),
                    solution_count=callback_model.cbGet(GRB.Callback.MIP_SOLCNT),
                    incumbent_objective=callback_model.cbGet(GRB.Callback.MIP_OBJBST),
                    best_bound=callback_model.cbGet(GRB.Callback.MIP_OBJBND),
                )
            )

        return callback

    def solve(self) -> bool:
        """Solve the CE-SVM optimization model.
        
        Returns:
            True if optimal solution found, False otherwise
        """
        if self.model is None:
            raise RuntimeError("Model not built. Call build_model() first.")

        solve_started_at = perf_counter()
        heartbeat_callback = self._build_heartbeat_callback()
        if heartbeat_callback is None:
            self.model.optimize()
        else:
            self.model.optimize(heartbeat_callback)
        self.solve_time = perf_counter() - solve_started_at

        if self.model.status == GRB.OPTIMAL:
            self._extract_solution()
            return True
        elif self.model.status == GRB.TIME_LIMIT and self.model.SolCount > 0:
            # Accept best solution found within time limit
            print(f"Time limit reached. Using best solution found (gap: {self.model.MIPGap:.2%})")
            self._extract_solution()
            return True
        elif self.model.status == GRB.MEM_LIMIT and self.model.SolCount > 0:
            print(f"Soft memory limit reached. Using best solution found (gap: {self.model.MIPGap:.2%})")
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
        if self.model.status not in [GRB.OPTIMAL, GRB.TIME_LIMIT, GRB.MEM_LIMIT]:
            return

        d = len([v for v in self.model.getVars() if v.VarName.startswith("w_plus")])
        n = len([v for v in self.model.getVars() if v.VarName.startswith("ksi")])

        # Extract weight variables
        w_plus_vals = np.array([self.model.getVarByName(f"w_plus[{j}]").X for j in range(d)])
        w_minus_vals = np.array([self.model.getVarByName(f"w_minus[{j}]").X for j in range(d)])
        self.weights = w_plus_vals - w_minus_vals
        self.intercept = self.model.getVarByName("b").X

        # Extract feature selection variables
        if self.enable_selection:
            v_vals = np.array([self.model.getVarByName(f"v[{j}]").X for j in range(d)])
            self.selected_features = v_vals > 0.5
        else:
            v_vals = np.ones(d)
            self.selected_features = np.ones(d, dtype=bool)

        # Extract accuracy lower bounds
        l_p_val = self.model.getVarByName("l_p").X
        l_n_val = self.model.getVarByName("l_n").X

        raw_solution = {}
        if self.retain_raw_solution_arrays:
            ksi_vals = np.array([self.model.getVarByName(f"ksi[{i}]").X for i in range(n)])
            alpha_vals = np.array([self.model.getVarByName(f"alpha[{i}]").X for i in range(n)])
            beta_vals = np.array([self.model.getVarByName(f"beta[{i}]").X for i in range(n)])
            rho_vals = np.array([self.model.getVarByName(f"rho[{i}]").X for i in range(n)])
            n_support_vectors = int(np.sum(ksi_vals > 1e-6))
            n_margin_errors = int(np.sum(ksi_vals > 1.0))
            raw_solution = {
                'ksi': ksi_vals,
                'alpha': alpha_vals,
                'beta': beta_vals,
                'rho': rho_vals,
            }
        else:
            n_support_vectors = 0
            n_margin_errors = 0
            for i in range(n):
                ksi_val = self.model.getVarByName(f"ksi[{i}]").X
                if ksi_val > 1e-6:
                    n_support_vectors += 1
                if ksi_val > 1.0:
                    n_margin_errors += 1

        self.solution = {
            # Primary decision variables
            'weights': self.weights,
            'w_plus': w_plus_vals,
            'w_minus': w_minus_vals,
            'intercept': self.intercept,
            'selected_features': self.selected_features,
            'v': v_vals,

            # Accuracy lower bounds
            'l_p': l_p_val,
            'l_n': l_n_val,

            # Summary statistics
            'objective_value': self.model.ObjVal,
            'n_selected_features': int(np.sum(self.selected_features)),
            'n_support_vectors': n_support_vectors,
            'n_margin_errors': n_margin_errors,
            'n_samples': self.n_samples,
            'n_features': self.n_features,
            'solve_time': self.solve_time,

            # Solver information
            'mip_gap': self.model.MIPGap if hasattr(self.model, 'MIPGap') else 0.0,
            'solver_status': self.model.Status,
            'mem_used_gb': self.model.MemUsed if hasattr(self.model, 'MemUsed') else None,
            'max_mem_used_gb': self.model.MaxMemUsed if hasattr(self.model, 'MaxMemUsed') else None,
        }
        self.solution.update(raw_solution)

    def release_solver_resources(self) -> None:
        """Free Gurobi resources once the extracted solution is sufficient."""
        if self.model is not None:
            self.model.dispose()
            self.model = None
        if self.env is not None:
            self.env.dispose()
            self.env = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'BinaryCESVM':
        """Fit the Binary CE-SVM model.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            y: Label vector (n_samples,), values in {+1, -1}
            
        Returns:
            self
        """
        self.build_model(X, y)
        try:
            success = self.solve()
            if not success:
                raise RuntimeError("CE-SVM optimization failed")
            return self
        finally:
            if self.release_solver_resources_after_fit:
                self.release_solver_resources()

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
            "n_samples": self.solution.get('n_samples'),
            "n_features": self.solution.get('n_features'),
            "objective_value": self.solution['objective_value'],
            "n_selected_features": self.solution['n_selected_features'],
            "selected_feature_indices": np.where(self.selected_features)[0].tolist(),
            "l1_norm": np.sum(np.abs(self.weights)),
            "positive_class_accuracy_lb": self.solution['l_p'],
            "negative_class_accuracy_lb": self.solution['l_n'],
            "n_support_vectors": self.solution['n_support_vectors'],
            "n_margin_errors": self.solution['n_margin_errors'],
            "solve_time": self.solution.get('solve_time'),
            "intercept": self.intercept,
            "mip_gap": self.solution.get('mip_gap', 0.0),
            "solver_status": self.solution.get('solver_status', 'unknown'),
            "mem_used_gb": self.solution.get('mem_used_gb'),
            "max_mem_used_gb": self.solution.get('max_mem_used_gb'),
        }

    def _require_raw_solution_array(self, key: str) -> np.ndarray:
        """Return a retained raw solution array or raise a clear error."""
        if self.solution is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        if key not in self.solution:
            raise RuntimeError(
                f"Raw solution array '{key}' was not retained. "
                f"Initialize BinaryCESVM with retain_raw_solution_arrays=True."
            )
        return self.solution[key]

    def get_slack_variables(self) -> np.ndarray:
        """Get slack variables (ksi) for all training samples.

        Returns:
            Array of slack values (n_samples,)
        """
        return self._require_raw_solution_array('ksi')

    def get_indicator_variables(self) -> Dict[str, np.ndarray]:
        """Get three-tier indicator variables for all training samples.

        Returns:
            Dictionary with keys 'alpha', 'beta', 'rho' containing binary arrays
        """
        return {
            'alpha': self._require_raw_solution_array('alpha'),
            'beta': self._require_raw_solution_array('beta'),
            'rho': self._require_raw_solution_array('rho'),
        }

    def get_support_vectors_mask(self, threshold: float = 1e-6) -> np.ndarray:
        """Get boolean mask indicating which training samples are support vectors.

        Args:
            threshold: Minimum slack value to consider as support vector

        Returns:
            Boolean array (n_samples,)
        """
        return self._require_raw_solution_array('ksi') > threshold

    def get_margin_errors_mask(self, margin_threshold: float = 1.0) -> np.ndarray:
        """Get boolean mask indicating samples with margin errors.

        Args:
            margin_threshold: Slack threshold for margin errors (default: 1.0)

        Returns:
            Boolean array (n_samples,)
        """
        return self._require_raw_solution_array('ksi') > margin_threshold

    def get_weight_decomposition(self) -> Dict[str, np.ndarray]:
        """Get decomposition of weights into positive and negative parts.

        Returns:
            Dictionary with keys 'w_plus' and 'w_minus'
        """
        if self.solution is None:
            raise RuntimeError("Model not fitted. Call fit() first.")
        return {
            'w_plus': self.solution['w_plus'],
            'w_minus': self.solution['w_minus'],
        }
