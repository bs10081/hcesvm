"""Configuration parameters for HCESVM models."""

from typing import Dict, Any


# Default CE-SVM hyperparameters
DEFAULT_CESVM_PARAMS: Dict[str, Any] = {
    # Regularization and penalty
    "C_hyper": 1.0,          # Slack penalty coefficient
    "epsilon": 0.0001,       # Accuracy constraint tolerance
    "M": 1000.0,             # Big-M constant

    # Gurobi solver parameters
    "time_limit": 1800,      # Time limit in seconds (30 minutes)
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
    "default_strategy": "multiple_filter",  # Default strategy

    # Single Filter Strategy (original)
    "single_filter": {
        "classifier_1": {
            "positive_class": 3,
            "negative_classes": [1, 2],
            "description": "Class 3 vs {1,2}"
        },
        "classifier_2": {
            "positive_class": 2,
            "negative_class": 1,
            "description": "Class 2 vs Class 1"
        },
    },

    # Multiple Filter Strategy (new)
    "multiple_filter": {
        "classifier_1": {
            "positive_class": 1,
            "negative_classes": [2, 3],
            "description": "Class 1 vs {2,3}"
        },
        "classifier_2": {
            "positive_class": 2,
            "negative_class": 3,
            "description": "Class 2 vs Class 3"
        },
    },
}


def get_default_params() -> Dict[str, Any]:
    """Get default CE-SVM parameters."""
    return {
        **DEFAULT_CESVM_PARAMS,
        **FEATURE_SELECTION_PARAMS,
    }
