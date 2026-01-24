#!/usr/bin/env python3
"""Evaluation utilities for HCESVM."""

import numpy as np
from typing import Dict


def calculate_accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate overall accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Accuracy (0-1)
    """
    return np.mean(y_true == y_pred)


def calculate_per_class_accuracy(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    classes: list = [1, 2, 3]
) -> Dict[str, float]:
    """Calculate per-class accuracy.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        classes: List of class labels
        
    Returns:
        Dictionary mapping class to accuracy
    """
    results = {}
    for k in classes:
        mask = y_true == k
        if np.sum(mask) > 0:
            results[f'Class {k}'] = np.mean(y_pred[mask] == k)
        else:
            results[f'Class {k}'] = np.nan
    return results


def calculate_binary_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, float]:
    """Calculate binary classification metrics.
    
    Args:
        y_true: True labels (+1, -1)
        y_pred: Predicted labels (+1, -1)
        
    Returns:
        Dictionary with TPR, TNR, accuracy
    """
    # Positive class (+1)
    pos_mask = y_true == 1
    TPR = np.mean(y_pred[pos_mask] == 1) if np.sum(pos_mask) > 0 else np.nan
    
    # Negative class (-1)
    neg_mask = y_true == -1
    TNR = np.mean(y_pred[neg_mask] == -1) if np.sum(neg_mask) > 0 else np.nan
    
    accuracy = np.mean(y_true == y_pred)
    
    return {
        'TPR': TPR,
        'TNR': TNR,
        'accuracy': accuracy,
    }


def evaluate_hierarchical_model(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict:
    """Evaluate hierarchical classifier performance.
    
    Args:
        y_true: True labels (1, 2, 3)
        y_pred: Predicted labels (1, 2, 3)
        
    Returns:
        Dictionary with various metrics
    """
    total_acc = calculate_accuracy(y_true, y_pred)
    per_class_acc = calculate_per_class_accuracy(y_true, y_pred)
    
    # Confusion matrix
    confusion = np.zeros((3, 3), dtype=int)
    for true_class in [1, 2, 3]:
        for pred_class in [1, 2, 3]:
            mask = (y_true == true_class) & (y_pred == pred_class)
            confusion[true_class-1, pred_class-1] = np.sum(mask)
    
    return {
        'total_accuracy': total_acc,
        'per_class_accuracy': per_class_acc,
        'confusion_matrix': confusion,
        'n_samples': len(y_true),
        'class_distribution': {
            'Class 1': np.sum(y_true == 1),
            'Class 2': np.sum(y_true == 2),
            'Class 3': np.sum(y_true == 3),
        }
    }


def print_evaluation_results(results: Dict):
    """Print evaluation results in a formatted manner.
    
    Args:
        results: Dictionary from evaluate_hierarchical_model
    """
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    
    print(f"\nTotal Accuracy: {results['total_accuracy']:.4f}")
    
    print("\nPer-Class Accuracy:")
    for class_name, acc in results['per_class_accuracy'].items():
        print(f"  {class_name}: {acc:.4f}")
    
    print("\nClass Distribution:")
    for class_name, count in results['class_distribution'].items():
        print(f"  {class_name}: {count} samples")
    
    print("\nConfusion Matrix:")
    print("     Pred 1  Pred 2  Pred 3")
    cm = results['confusion_matrix']
    for i, row in enumerate(cm):
        print(f"True {i+1}:  {row[0]:4d}    {row[1]:4d}    {row[2]:4d}")
    
    print("=" * 60)
