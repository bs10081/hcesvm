#!/usr/bin/env python3
"""Data loading utilities for HCESVM."""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional


def load_parkinsons_data(
    excel_file: str,
    sheet_name: str = 'train',
    skiprows: int = 4
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Load Parkinsons CE-SVM data from Excel file.
    
    Excel format:
        Row 0-3: Metadata (w, w+, w-, v)
        Row 4: Column headers
        Row 5+: Training data
    
    Args:
        excel_file: Path to Excel file
        sheet_name: Sheet name to read
        skiprows: Number of rows to skip (metadata)
        
    Returns:
        X: Feature matrix (n_samples, n_features)
        y: Label vector (n_samples,), values in {+1, -1}
        n_features: Number of features
    """
    # Load data
    df = pd.read_excel(excel_file, sheet_name=sheet_name, header=skiprows)
    
    # Extract y labels (status column)
    if 'status' not in df.columns:
        raise ValueError("'status' column not found in Excel file")
    
    y = df['status'].values
    
    # Extract feature columns (exclude metadata columns)
    metadata_cols = [
        'Unnamed: 0', 'status', 'ksi', 'ai', 'bi', 'ri',
        'predict value', 'predic_class', 'correct', 'accuracy',
        'TP', 'FN', 'TN', 'FP', 'TPR', 'TNR'
    ]
    feature_cols = [
        c for c in df.columns
        if c not in metadata_cols and not c.startswith('Unnamed')
    ]
    
    X = df[feature_cols].values
    n_features = X.shape[1]
    
    # Filter out NaN samples
    valid_mask = ~np.isnan(y)
    X = X[valid_mask]
    y = y[valid_mask].astype(int)
    
    print(f"Loaded {len(y)} samples, {n_features} features")
    print(f"Positive class (+1): {np.sum(y == 1)}")
    print(f"Negative class (-1): {np.sum(y == -1)}")
    
    return X, y, n_features


def load_multiclass_data(
    excel_file: str,
    sheet_name: str = 'Train',
    skiprows: int = 5
) -> Tuple[List[np.ndarray], List[int], int]:
    """Load 3-class ordinal classification data (NSVORA format).
    
    Args:
        excel_file: Path to Excel file
        sheet_name: Sheet name to read
        skiprows: Number of rows to skip
        
    Returns:
        X_classes: List of feature matrices [X1, X2, X3]
        n_classes_list: List of class sizes [n1, n2, n3]
        n_features: Number of features
    """
    df = pd.read_excel(excel_file, sheet_name=sheet_name, skiprows=skiprows)
    
    # Find label column
    label_col = None
    for col in ['Actual.1', 'Actual', 'Class', 'class']:
        if col in df.columns:
            label_col = col
            break
    
    if label_col is None:
        raise ValueError("Label column not found")
    
    # Extract labels
    y = df[label_col].values
    
    # Extract feature columns
    metadata_cols = [
        'Unnamed: 0', label_col, 'Actual', 'predict', 'correct'
    ]
    feature_cols = [
        c for c in df.columns
        if c not in metadata_cols and not c.startswith('Unnamed')
    ]
    
    X = df[feature_cols].values
    n_features = X.shape[1]
    
    # Split by class
    X_classes = []
    n_classes_list = []
    
    for k in [1, 2, 3]:
        mask = y == k
        X_k = X[mask]
        X_classes.append(X_k)
        n_classes_list.append(len(X_k))
    
    print(f"Loaded multi-class data:")
    print(f"  Class 1: {n_classes_list[0]} samples")
    print(f"  Class 2: {n_classes_list[1]} samples")
    print(f"  Class 3: {n_classes_list[2]} samples")
    print(f"  Features: {n_features}")
    
    return X_classes, n_classes_list, n_features


def relabel_for_binary(y: np.ndarray, positive_class: int) -> np.ndarray:
    """Relabel multi-class labels for binary classification.
    
    Args:
        y: Original labels (1, 2, 3, ...)
        positive_class: Which class to label as +1
        
    Returns:
        Binary labels (+1 for positive_class, -1 for others)
    """
    return np.where(y == positive_class, 1, -1)
