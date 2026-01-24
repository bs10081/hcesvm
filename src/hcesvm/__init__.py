"""HCESVM: Hierarchical Cost-Effective SVM for ordinal classification."""

__version__ = "0.1.0"

from .models import BinaryCESVM, HierarchicalCESVM
from .config import (
    DEFAULT_CESVM_PARAMS,
    FEATURE_SELECTION_PARAMS,
    HIERARCHICAL_CONFIG,
    get_default_params,
)

__all__ = [
    'BinaryCESVM',
    'HierarchicalCESVM',
    'DEFAULT_CESVM_PARAMS',
    'FEATURE_SELECTION_PARAMS',
    'HIERARCHICAL_CONFIG',
    'get_default_params',
]
