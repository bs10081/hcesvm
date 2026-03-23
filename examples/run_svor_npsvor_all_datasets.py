#!/usr/bin/env python3
"""
Sequential execution of SVOR and NPSVOR models on all valid datasets.

This script tests both SVOR (Support Vector Ordinal Regression) and
NPSVOR (Non-Parallel SVOR) models on all 9 valid datasets in the project.

Models:
- SVOR: Implicit threshold ordinal regression model
- NPSVOR: Non-parallel ordinal regression with multiple hyperplanes

Datasets (9 total):
1. Car_Evaluation
2. Wine_Quality (TBD path)
3. Balance
4. Contraceptive
5. Hayes_Roth
6. New_Thyroid
7. TAE
8. Thyroid
9. Wine

Usage:
    source .venv/bin/activate
    python examples/run_svor_npsvor_all_datasets.py
"""

import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict, Any

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hcesvm.utils.data_loader import load_split_data
from src.hcesvm.models.svor import SVORImplicitQP
from src.hcesvm.models.npsvor import NPSVORQP


# Dataset configurations
DATASETS = {
    'Car_Evaluation': {
        'path': '~/Developer/NSVORA/datasets/primary/car_evaluation/car_evaluation_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'Balance': {
        'path': '~/Developer/NSVORA/Archive/balance_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'Contraceptive': {
        'path': '~/Developer/NSVORA/Archive/contraceptive_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'Hayes_Roth': {
        'path': '~/Developer/NSVORA/Archive/hayes-roth_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'New_Thyroid': {
        'path': '~/Developer/NSVORA/Archive/new-thyroid_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'TAE': {
        'path': '~/Developer/NSVORA/Archive/tae_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'Thyroid': {
        'path': '~/Developer/NSVORA/Archive/thyroid_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'Wine': {
        'path': '~/Developer/NSVORA/Archive/wine_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
}


def convert_to_ordinal_format(
    X_classes: List[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert multi-class data to ordinal format (X, y).

    Args:
        X_classes: List of feature matrices [X1, X2, X3]

    Returns:
        X: Combined feature matrix
        y: Ordinal labels (1, 2, 3)
    """
    X_combined = []
    y_combined = []

    for class_idx, X_class in enumerate(X_classes, start=1):
        X_combined.append(X_class)
        y_combined.extend([class_idx] * len(X_class))

    X = np.vstack(X_combined)
    y = np.array(y_combined)

    return X, y


def calculate_accuracy(y_true: np.ndarray, y_pred: List[Any]) -> Dict[str, float]:
    """Calculate per-class and total accuracy.

    Args:
        y_true: True labels
        y_pred: Predicted labels

    Returns:
        Dictionary with accuracy metrics
    """
    y_pred_array = np.array(y_pred)

    # Total accuracy
    total_acc = np.mean(y_true == y_pred_array)

    # Per-class accuracy
    class_accs = {}
    for class_label in [1, 2, 3]:
        mask = y_true == class_label
        if mask.sum() > 0:
            class_acc = np.mean(y_true[mask] == y_pred_array[mask])
            class_accs[f'class_{class_label}'] = class_acc
        else:
            class_accs[f'class_{class_label}'] = 0.0

    return {
        'total': total_acc,
        **class_accs
    }


def test_svor(
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    log_file
) -> Dict[str, Any]:
    """Test SVOR model on a dataset.

    Args:
        dataset_name: Name of the dataset
        X_train, y_train: Training data
        X_test, y_test: Testing data
        log_file: File handle for logging

    Returns:
        Results dictionary
    """
    print(f"\n{'=' * 80}")
    print(f"Testing SVOR on {dataset_name}")
    print(f"{'=' * 80}")
    log_file.write(f"\n{'=' * 80}\n")
    log_file.write(f"Testing SVOR on {dataset_name}\n")
    log_file.write(f"{'=' * 80}\n")

    # Initialize model
    model = SVORImplicitQP(
        C=1.0,
        add_order_constraints=True,
        solver_params={'TimeLimit': 1800, 'MIPGap': 1e-4}
    )

    # Train
    print(f"Training SVOR...")
    log_file.write(f"Training SVOR...\n")
    start_time = datetime.now()

    try:
        model.fit(X_train.tolist(), y_train.tolist())
        train_time = (datetime.now() - start_time).total_seconds()

        # Predict
        print(f"Predicting on training set...")
        y_train_pred = model.predict(X_train.tolist())
        train_acc = calculate_accuracy(y_train, y_train_pred)

        print(f"Predicting on test set...")
        y_test_pred = model.predict(X_test.tolist())
        test_acc = calculate_accuracy(y_test, y_test_pred)

        # Log results
        print(f"\nSVOR Results:")
        print(f"  Training time: {train_time:.2f} seconds")
        print(f"  Training accuracy: {train_acc['total']:.4f}")
        print(f"    Class 1: {train_acc['class_1']:.4f}")
        print(f"    Class 2: {train_acc['class_2']:.4f}")
        print(f"    Class 3: {train_acc['class_3']:.4f}")
        print(f"  Testing accuracy: {test_acc['total']:.4f}")
        print(f"    Class 1: {test_acc['class_1']:.4f}")
        print(f"    Class 2: {test_acc['class_2']:.4f}")
        print(f"    Class 3: {test_acc['class_3']:.4f}")
        print(f"  Weights: {model.weights_}")
        print(f"  Thresholds: {model.thresholds_}")

        log_file.write(f"\nSVOR Results:\n")
        log_file.write(f"  Training time: {train_time:.2f} seconds\n")
        log_file.write(f"  Training accuracy: {train_acc['total']:.4f}\n")
        log_file.write(f"    Class 1: {train_acc['class_1']:.4f}\n")
        log_file.write(f"    Class 2: {train_acc['class_2']:.4f}\n")
        log_file.write(f"    Class 3: {train_acc['class_3']:.4f}\n")
        log_file.write(f"  Testing accuracy: {test_acc['total']:.4f}\n")
        log_file.write(f"    Class 1: {test_acc['class_1']:.4f}\n")
        log_file.write(f"    Class 2: {test_acc['class_2']:.4f}\n")
        log_file.write(f"    Class 3: {test_acc['class_3']:.4f}\n")
        log_file.write(f"  Weights: {model.weights_}\n")
        log_file.write(f"  Thresholds: {model.thresholds_}\n")

        return {
            'status': 'success',
            'train_time': train_time,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'weights': model.weights_,
            'thresholds': model.thresholds_
        }

    except Exception as e:
        error_msg = f"SVOR failed: {str(e)}"
        print(f"\n{error_msg}")
        log_file.write(f"\n{error_msg}\n")

        return {
            'status': 'failed',
            'error': str(e)
        }


def test_npsvor(
    dataset_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    log_file
) -> Dict[str, Any]:
    """Test NPSVOR model on a dataset.

    Args:
        dataset_name: Name of the dataset
        X_train, y_train: Training data
        X_test, y_test: Testing data
        log_file: File handle for logging

    Returns:
        Results dictionary
    """
    print(f"\n{'=' * 80}")
    print(f"Testing NPSVOR on {dataset_name}")
    print(f"{'=' * 80}")
    log_file.write(f"\n{'=' * 80}\n")
    log_file.write(f"Testing NPSVOR on {dataset_name}\n")
    log_file.write(f"{'=' * 80}\n")

    # Initialize model
    model = NPSVORQP(
        C1=1.0,
        C2=1.0,
        epsilon=0.2,
        prediction_rule='min_distance',
        solver_params={'TimeLimit': 1800, 'MIPGap': 1e-4}
    )

    # Train
    print(f"Training NPSVOR...")
    log_file.write(f"Training NPSVOR...\n")
    start_time = datetime.now()

    try:
        model.fit(X_train.tolist(), y_train.tolist())
        train_time = (datetime.now() - start_time).total_seconds()

        # Predict
        print(f"Predicting on training set...")
        y_train_pred = model.predict(X_train.tolist())
        train_acc = calculate_accuracy(y_train, y_train_pred)

        print(f"Predicting on test set...")
        y_test_pred = model.predict(X_test.tolist())
        test_acc = calculate_accuracy(y_test, y_test_pred)

        # Log results
        print(f"\nNPSVOR Results:")
        print(f"  Training time: {train_time:.2f} seconds")
        print(f"  Training accuracy: {train_acc['total']:.4f}")
        print(f"    Class 1: {train_acc['class_1']:.4f}")
        print(f"    Class 2: {train_acc['class_2']:.4f}")
        print(f"    Class 3: {train_acc['class_3']:.4f}")
        print(f"  Testing accuracy: {test_acc['total']:.4f}")
        print(f"    Class 1: {test_acc['class_1']:.4f}")
        print(f"    Class 2: {test_acc['class_2']:.4f}")
        print(f"    Class 3: {test_acc['class_3']:.4f}")
        print(f"  Hyperplanes (3 total):")
        for i, (hp, b) in enumerate(zip(model.hyperplanes_, model.biases_), 1):
            print(f"    Class {i}: w={hp}, b={b:.4f}")

        log_file.write(f"\nNPSVOR Results:\n")
        log_file.write(f"  Training time: {train_time:.2f} seconds\n")
        log_file.write(f"  Training accuracy: {train_acc['total']:.4f}\n")
        log_file.write(f"    Class 1: {train_acc['class_1']:.4f}\n")
        log_file.write(f"    Class 2: {train_acc['class_2']:.4f}\n")
        log_file.write(f"    Class 3: {train_acc['class_3']:.4f}\n")
        log_file.write(f"  Testing accuracy: {test_acc['total']:.4f}\n")
        log_file.write(f"    Class 1: {test_acc['class_1']:.4f}\n")
        log_file.write(f"    Class 2: {test_acc['class_2']:.4f}\n")
        log_file.write(f"    Class 3: {test_acc['class_3']:.4f}\n")
        log_file.write(f"  Hyperplanes (3 total):\n")
        for i, (hp, b) in enumerate(zip(model.hyperplanes_, model.biases_), 1):
            log_file.write(f"    Class {i}: w={hp}, b={b:.4f}\n")

        return {
            'status': 'success',
            'train_time': train_time,
            'train_acc': train_acc,
            'test_acc': test_acc,
            'hyperplanes': model.hyperplanes_,
            'biases': model.biases_
        }

    except Exception as e:
        error_msg = f"NPSVOR failed: {str(e)}"
        print(f"\n{error_msg}")
        log_file.write(f"\n{error_msg}\n")

        return {
            'status': 'failed',
            'error': str(e)
        }


def test_dataset(dataset_name: str, config: Dict[str, Any], log_file) -> Dict[str, Any]:
    """Test both SVOR and NPSVOR on a single dataset.

    Args:
        dataset_name: Name of the dataset
        config: Dataset configuration
        log_file: File handle for logging

    Returns:
        Results dictionary for both models
    """
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}")
    log_file.write(f"\n{'=' * 80}\n")
    log_file.write(f"Dataset: {dataset_name}\n")
    log_file.write(f"{'=' * 80}\n")

    # Expand path
    excel_path = os.path.expanduser(config['path'])

    # Check if file exists
    if not os.path.exists(excel_path):
        error_msg = f"Dataset file not found: {excel_path}"
        print(f"\n{error_msg}")
        log_file.write(f"\n{error_msg}\n")
        return {
            'svor': {'status': 'skipped', 'error': 'File not found'},
            'npsvor': {'status': 'skipped', 'error': 'File not found'}
        }

    try:
        # Load data
        print(f"Loading data from: {excel_path}")
        log_file.write(f"Loading data from: {excel_path}\n")

        data = load_split_data(
            excel_path,
            train_skiprows=config['train_skiprows'],
            test_skiprows=config['test_skiprows']
        )

        X1_train = data['X1_train']
        X2_train = data['X2_train']
        X3_train = data['X3_train']
        X1_test = data['X1_test']
        X2_test = data['X2_test']
        X3_test = data['X3_test']
        n_features = data['n_features']

        n1, n2, n3 = len(X1_train), len(X2_train), len(X3_train)
        n1_test, n2_test, n3_test = len(X1_test), len(X2_test), len(X3_test)

        print(f"\nDataset Info:")
        print(f"  Training: Class1={n1}, Class2={n2}, Class3={n3}")
        print(f"  Testing: Class1={n1_test}, Class2={n2_test}, Class3={n3_test}")
        print(f"  Features: {n_features}")

        log_file.write(f"\nDataset Info:\n")
        log_file.write(f"  Training: Class1={n1}, Class2={n2}, Class3={n3}\n")
        log_file.write(f"  Testing: Class1={n1_test}, Class2={n2_test}, Class3={n3_test}\n")
        log_file.write(f"  Features: {n_features}\n")

        # Convert to ordinal format
        X_train, y_train = convert_to_ordinal_format([X1_train, X2_train, X3_train])
        X_test, y_test = convert_to_ordinal_format([X1_test, X2_test, X3_test])

        # Test SVOR
        svor_results = test_svor(dataset_name, X_train, y_train, X_test, y_test, log_file)

        # Test NPSVOR
        npsvor_results = test_npsvor(dataset_name, X_train, y_train, X_test, y_test, log_file)

        return {
            'svor': svor_results,
            'npsvor': npsvor_results
        }

    except Exception as e:
        error_msg = f"Failed to process dataset: {str(e)}"
        print(f"\n{error_msg}")
        log_file.write(f"\n{error_msg}\n")

        import traceback
        traceback.print_exc()
        log_file.write(traceback.format_exc())

        return {
            'svor': {'status': 'failed', 'error': str(e)},
            'npsvor': {'status': 'failed', 'error': str(e)}
        }


def main():
    """Main function to run all tests."""
    # Setup
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    log_path = results_dir / f'svor_npsvor_all_datasets_{timestamp}.log'

    print(f"{'=' * 80}")
    print(f"SVOR and NPSVOR Sequential Testing")
    print(f"{'=' * 80}")
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Log file: {log_path}")
    print(f"\nDatasets to test: {len(DATASETS)}")
    for name in DATASETS.keys():
        print(f"  - {name}")
    print(f"{'=' * 80}\n")

    # Run tests
    start_time = datetime.now()
    all_results = {}

    with open(log_path, 'w') as log_file:
        log_file.write(f"{'=' * 80}\n")
        log_file.write(f"SVOR and NPSVOR Sequential Testing\n")
        log_file.write(f"{'=' * 80}\n")
        log_file.write(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"\nDatasets to test: {len(DATASETS)}\n")
        for name in DATASETS.keys():
            log_file.write(f"  - {name}\n")
        log_file.write(f"{'=' * 80}\n\n")

        for dataset_name, config in DATASETS.items():
            results = test_dataset(dataset_name, config, log_file)
            all_results[dataset_name] = results

    # Summary
    end_time = datetime.now()
    duration = end_time - start_time

    print(f"\n{'=' * 80}")
    print(f"All Tests Completed!")
    print(f"{'=' * 80}")
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {duration}")
    print(f"\n{'=' * 80}")
    print(f"Results Summary")
    print(f"{'=' * 80}\n")

    # SVOR Summary
    print("SVOR Results:")
    print(f"{'Dataset':<20} {'Status':<10} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 80)
    for dataset_name, results in all_results.items():
        svor = results['svor']
        if svor['status'] == 'success':
            train_acc = f"{svor['train_acc']['total']:.4f}"
            test_acc = f"{svor['test_acc']['total']:.4f}"
        else:
            train_acc = "N/A"
            test_acc = "N/A"
        print(f"{dataset_name:<20} {svor['status']:<10} {train_acc:<12} {test_acc:<12}")

    # NPSVOR Summary
    print("\nNPSVOR Results:")
    print(f"{'Dataset':<20} {'Status':<10} {'Train Acc':<12} {'Test Acc':<12}")
    print("-" * 80)
    for dataset_name, results in all_results.items():
        npsvor = results['npsvor']
        if npsvor['status'] == 'success':
            train_acc = f"{npsvor['train_acc']['total']:.4f}"
            test_acc = f"{npsvor['test_acc']['total']:.4f}"
        else:
            train_acc = "N/A"
            test_acc = "N/A"
        print(f"{dataset_name:<20} {npsvor['status']:<10} {train_acc:<12} {test_acc:<12}")

    print(f"\n{'=' * 80}")
    print(f"Log saved to: {log_path}")
    print(f"{'=' * 80}")

    # Save summary to log
    with open(log_path, 'a') as log_file:
        log_file.write(f"\n{'=' * 80}\n")
        log_file.write(f"All Tests Completed!\n")
        log_file.write(f"{'=' * 80}\n")
        log_file.write(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        log_file.write(f"Total duration: {duration}\n")
        log_file.write(f"\n{'=' * 80}\n")
        log_file.write(f"Results Summary\n")
        log_file.write(f"{'=' * 80}\n\n")

        log_file.write("SVOR Results:\n")
        log_file.write(f"{'Dataset':<20} {'Status':<10} {'Train Acc':<12} {'Test Acc':<12}\n")
        log_file.write("-" * 80 + "\n")
        for dataset_name, results in all_results.items():
            svor = results['svor']
            if svor['status'] == 'success':
                train_acc = f"{svor['train_acc']['total']:.4f}"
                test_acc = f"{svor['test_acc']['total']:.4f}"
            else:
                train_acc = "N/A"
                test_acc = "N/A"
            log_file.write(f"{dataset_name:<20} {svor['status']:<10} {train_acc:<12} {test_acc:<12}\n")

        log_file.write("\nNPSVOR Results:\n")
        log_file.write(f"{'Dataset':<20} {'Status':<10} {'Train Acc':<12} {'Test Acc':<12}\n")
        log_file.write("-" * 80 + "\n")
        for dataset_name, results in all_results.items():
            npsvor = results['npsvor']
            if npsvor['status'] == 'success':
                train_acc = f"{npsvor['train_acc']['total']:.4f}"
                test_acc = f"{npsvor['test_acc']['total']:.4f}"
            else:
                train_acc = "N/A"
                test_acc = "N/A"
            log_file.write(f"{dataset_name:<20} {npsvor['status']:<10} {train_acc:<12} {test_acc:<12}\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
