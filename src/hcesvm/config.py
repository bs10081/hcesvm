"""Configuration parameters for HCESVM models."""

from typing import Dict, Any


# Default CE-SVM hyperparameters
DEFAULT_CESVM_PARAMS: Dict[str, Any] = {
    # Regularization and penalty
    "C_hyper": 1.0,          # Slack penalty coefficient
    "epsilon": 0.0001,       # Accuracy constraint tolerance
    "M": 1000.0,             # Big-M constant

    # Gurobi solver parameters
    "time_limit": 600,       # Time limit in seconds
    "mip_gap": 1e-4,         # MIP optimality gap
    "threads": 0,            # 0 = use all available threads
    "verbose": True,         # Print solver output
}


# Feature selection parameters (no cost/budget)
FEATURE_SELECTION_PARAMS: Dict[str, Any] = {
    "enable_selection": True,      # Enable feature selection via binary variables
    "feat_upper_bound": 1000,      # Upper bound for feature activation
    "feat_lower_bound": 0.0000001, # Lower bound for feature activation
}


# Hierarchical classifier configuration
HIERARCHICAL_CONFIG: Dict[str, Any] = {
    # Classifier 1: Class 3 (+1) vs {Class 1, 2} (-1)
    "classifier_1": {
        "positive_class": 3,
        "negative_classes": [1, 2],
        "description": "Class 3 vs {1,2}"
    },

    # Classifier 2: Class 2 (+1) vs Class 1 (-1)
    "classifier_2": {
        "positive_class": 2,
        "negative_class": 1,
        "description": "Class 2 vs Class 1"
    },
}


def get_default_params() -> Dict[str, Any]:
    """Get default CE-SVM parameters."""
    return {
        **DEFAULT_CESVM_PARAMS,
        **FEATURE_SELECTION_PARAMS,
    }
