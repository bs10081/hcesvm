#!/usr/bin/env python3
"""
Revalidate Test3 strategy on Car_Evaluation and New_Thyroid datasets.

Purpose: Verify reproducibility of Test3 results by re-running tests
on two specific datasets.
"""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm import HierarchicalCESVM, get_default_params
from hcesvm.utils import load_multiclass_data, evaluate_hierarchical_model, print_evaluation_results


class TeeOutput:
    """Output class that writes to both file and console simultaneously."""

    def __init__(self, file, stream):
        self.file = file
        self.stream = stream

    def write(self, data):
        self.file.write(data)
        self.stream.write(data)

    def flush(self):
        self.file.flush()
        self.stream.flush()


def detect_test_skiprows(data_file: Path) -> int:
    """Detect skiprows for Test sheet (usually 4)."""
    return 4


def run_test3_validation(dataset_name: str, data_path: str, results_dir: Path, log_file_handle, original_stdout):
    """Run Test3 strategy on a single dataset."""

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    print("=" * 80)
    print(f"Test3 Strategy Revalidation: {dataset_name}")
    print(f"Timestamp: {timestamp}")
    print("=" * 80)
    print()

    # Expand path
    data_file = Path(data_path).expanduser()

    if not data_file.exists():
        print(f"Error: Dataset not found: {data_file}")
        return None

    # Load training data
    print("Loading training data...")
    try:
        X_classes_train, n_classes_train, n_features = load_multiclass_data(
            str(data_file), sheet_name='Train', skiprows=5
        )
        X1_train, X2_train, X3_train = X_classes_train
    except Exception as e:
        print(f"Error loading training data: {e}")
        return None

    # Load test data
    print("Loading test data...")
    try:
        skiprows_test = detect_test_skiprows(data_file)
        X_classes_test, n_classes_test, _ = load_multiclass_data(
            str(data_file), sheet_name='Test', skiprows=skiprows_test
        )
        X1_test, X2_test, X3_test = X_classes_test

        # Combine test data
        X_test_all = np.vstack([X1_test, X2_test, X3_test])
        y_test_all = np.hstack([
            np.ones(len(X1_test)),
            2 * np.ones(len(X2_test)),
            3 * np.ones(len(X3_test))
        ])
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None

    print()
    print(f"Training samples: Class 1={len(X1_train)}, Class 2={len(X2_train)}, Class 3={len(X3_train)}")
    print(f"Testing samples: Class 1={len(X1_test)}, Class 2={len(X2_test)}, Class 3={len(X3_test)}")
    print(f"Features: {n_features}")

    # Calculate imbalance ratio
    train_counts = [len(X1_train), len(X2_train), len(X3_train)]
    max_count = max(train_counts)
    min_count = min(train_counts)
    imbalance_ratio = max_count / min_count
    print(f"Imbalance ratio: {imbalance_ratio:.2f}")
    print()

    # Get default parameters
    params = get_default_params()
    params['verbose'] = True
    params['time_limit'] = 1800  # 30 minutes per classifier
    params['mip_gap'] = 1e-4

    print("=" * 80)
    print("Model Parameters")
    print("=" * 80)
    for key, value in params.items():
        print(f"  {key}: {value}")
    print()

    # Train model
    print("=" * 80)
    print("Training Hierarchical Classifier - TEST3 Strategy")
    print("=" * 80)
    print()
    print("Test3 Strategy:")
    print("  H1: {minority, medium} vs majority")
    print("  H2: minority vs medium")
    print("  Sample-weighted objective: -(1/s+)·l+ - (1/s-)·l-")
    print()

    train_start = datetime.now()

    try:
        hc = HierarchicalCESVM(cesvm_params=params, strategy='test3')
        hc.fit(X1_train, X2_train, X3_train)
    except Exception as e:
        print(f"Error during training: {e}")
        return None

    train_end = datetime.now()
    train_duration = train_end - train_start

    print()
    print(f"Training completed in {train_duration.total_seconds():.2f} seconds")
    print()

    # Get model summary
    summary = hc.get_model_summary()

    # Print MIP Gap information
    print("=" * 80)
    print("Optimization Results")
    print("=" * 80)
    print()
    print("H1 (First classifier):")
    print(f"  Status: {summary['h1']['status']}")
    print(f"  MIP Gap: {summary['h1']['mip_gap']:.4%}")
    print(f"  Solve Time: {summary['h1']['solve_time']:.2f}s")
    print(f"  Selected Features: {summary['h1']['n_selected_features']}")
    print(f"  L1 Norm: {summary['h1']['l1_norm']:.6f}")
    print()
    print("H2 (Second classifier):")
    print(f"  Status: {summary['h2']['status']}")
    print(f"  MIP Gap: {summary['h2']['mip_gap']:.4%}")
    print(f"  Solve Time: {summary['h2']['solve_time']:.2f}s")
    print(f"  Selected Features: {summary['h2']['n_selected_features']}")
    print(f"  L1 Norm: {summary['h2']['l1_norm']:.6f}")
    print()

    # Evaluate on training set
    print("=" * 80)
    print("Training Set Evaluation")
    print("=" * 80)
    print()

    X_train_all = np.vstack([X1_train, X2_train, X3_train])
    y_train_all = np.hstack([
        np.ones(len(X1_train)),
        2 * np.ones(len(X2_train)),
        3 * np.ones(len(X3_train))
    ])

    train_metrics = evaluate_hierarchical_model(hc, X_train_all, y_train_all)
    print_evaluation_results(train_metrics, "Training Set")
    print()

    # Evaluate on test set
    print("=" * 80)
    print("Test Set Evaluation")
    print("=" * 80)
    print()

    test_metrics = evaluate_hierarchical_model(hc, X_test_all, y_test_all)
    print_evaluation_results(test_metrics, "Test Set")
    print()

    print("=" * 80)
    print("Validation Complete!")
    print("=" * 80)
    print()

    return {
        'train': train_metrics,
        'test': test_metrics,
        'summary': summary,
        'elapsed_time': train_duration.total_seconds()
    }


if __name__ == "__main__":
    # Define datasets
    datasets = [
        ('Car_Evaluation', "~/Developer/NSVORA/datasets/primary/car_evaluation/car_evaluation_split.xlsx"),
        ('New_Thyroid', "~/Developer/NSVORA/Archive/new-thyroid_split.xlsx")
    ]

    # Create results directory
    results_dir = Path(__file__).parent.parent / 'results'
    results_dir.mkdir(exist_ok=True)

    # Open log file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = results_dir / f"test3_revalidation_{timestamp}.log"

    all_results = {}
    original_stdout = sys.stdout

    with open(log_filename, 'w') as log_file:
        sys.stdout = TeeOutput(log_file, original_stdout)

        print("=" * 80)
        print("Test3 Strategy Revalidation")
        print(f"Timestamp: {timestamp}")
        print("Datasets: Car_Evaluation, New_Thyroid")
        print("=" * 80)
        print()

        for dataset_name, dataset_path in datasets:
            print(f"\nProcessing {dataset_name}...")
            print()

            try:
                results = run_test3_validation(dataset_name, dataset_path, results_dir, log_file, original_stdout)
                if results is not None:
                    all_results[dataset_name] = results
            except Exception as e:
                print(f"\nError processing {dataset_name}: {str(e)}")
                import traceback
                traceback.print_exc()
                print()

        # Summary comparison
        print()
        print("=" * 80)
        print("SUMMARY: Test3 Revalidation Results")
        print("=" * 80)
        print()

        for dataset_name, results in all_results.items():
            print(f"{dataset_name}:")
            print(f"  Test Accuracy: {results['test']['total_accuracy']:.4f}")
            print(f"    Class 1: {results['test']['class_accuracies'][0]:.4f}")
            print(f"    Class 2: {results['test']['class_accuracies'][1]:.4f}")
            print(f"    Class 3: {results['test']['class_accuracies'][2]:.4f}")
            print(f"  H1 MIP Gap: {results['summary']['h1']['mip_gap']:.4%}")
            print(f"  H2 MIP Gap: {results['summary']['h2']['mip_gap']:.4%}")
            print(f"  Elapsed Time: {results['elapsed_time']:.2f}s")
            print(f"  H1 Status: {results['summary']['h1']['status']}")
            print(f"  H2 Status: {results['summary']['h2']['status']}")
            print()

        sys.stdout = original_stdout

    print(f"\nLog file saved to: {log_filename}")
