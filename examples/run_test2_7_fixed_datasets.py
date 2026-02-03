#!/usr/bin/env python3
"""Run Test2 strategy tests on the 7 fixed datasets to get testing accuracy."""

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


# Only the 7 fixed datasets
DATASETS = [
    ("Contraceptive", "~/Developer/NSVORA/Archive/contraceptive_split.xlsx"),
    ("Hayes_Roth", "~/Developer/NSVORA/Archive/hayes-roth_split.xlsx"),
    ("Squash_Stored", "~/Developer/NSVORA/Archive/squash-stored_split.xlsx"),
    ("Squash_Unstored", "~/Developer/NSVORA/Archive/squash-unstored_split.xlsx"),
    ("TAE", "~/Developer/NSVORA/Archive/tae_split.xlsx"),
    ("Thyroid", "~/Developer/NSVORA/Archive/thyroid_split.xlsx"),
    ("Wine", "~/Developer/NSVORA/Archive/wine_split.xlsx"),
]


def test_single_dataset(dataset_name: str, data_path: str, results_dir: Path) -> dict:
    """Test a single dataset with Test2 strategy.

    Returns:
        dict with test results and status
    """
    data_file = Path(data_path).expanduser()
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = results_dir / f"test2_fixed_{dataset_name}_{timestamp}.log"

    original_stdout = sys.stdout

    try:
        with open(log_file, 'w') as f:
            sys.stdout = TeeOutput(f, original_stdout)

            print("=" * 80)
            print(f"Hierarchical CE-SVM - Test2 Strategy (Fixed Datasets)")
            print(f"Dataset: {dataset_name}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()

            # Check if file exists
            if not data_file.exists():
                print(f"Error: {data_file} not found")
                sys.stdout = original_stdout
                return {'success': False, 'error': 'File not found', 'log_file': log_file}

            # Load training data
            print(f"Loading training data from: {data_file}")
            print()

            try:
                X_classes, n_classes_list, n_features = load_multiclass_data(
                    str(data_file), sheet_name='Train', skiprows=5
                )
            except Exception as e:
                print(f"Error loading training data: {e}")
                sys.stdout = original_stdout
                return {'success': False, 'error': f'Failed to load training data: {e}', 'log_file': log_file}

            X1, X2, X3 = X_classes

            print(f"Training data loaded:")
            print(f"  Class 1: {len(X1)} samples")
            print(f"  Class 2: {len(X2)} samples")
            print(f"  Class 3: {len(X3)} samples")
            print(f"  Features: {n_features}")
            print()

            # Get default parameters
            params = get_default_params()
            params['verbose'] = True
            params['time_limit'] = 1800  # 30 minutes per classifier

            print("=" * 80)
            print("Model Parameters")
            print("=" * 80)
            for key, value in params.items():
                print(f"  {key}: {value}")
            print()

            # Train hierarchical classifier with Test2 strategy
            print("=" * 80)
            print("Training Hierarchical Classifier - Test2 Strategy")
            print("=" * 80)
            print()

            train_start = datetime.now()

            try:
                hc = HierarchicalCESVM(cesvm_params=params, strategy="test2")
                hc.fit(X1, X2, X3)
            except Exception as e:
                print(f"Error during training: {e}")
                sys.stdout = original_stdout
                return {'success': False, 'error': f'Training failed: {e}', 'log_file': log_file}

            train_end = datetime.now()
            train_duration = train_end - train_start

            print(f"\nTraining completed in: {train_duration}")
            print()

            # Training Set Evaluation
            print("=" * 80)
            print("Training Set Evaluation")
            print("=" * 80)
            print()

            X_train_all = np.vstack([X1, X2, X3])
            y_train_all = np.concatenate([
                np.ones(len(X1), dtype=int),
                2 * np.ones(len(X2), dtype=int),
                3 * np.ones(len(X3), dtype=int)
            ])

            y_train_pred = hc.predict(X_train_all)
            train_results = evaluate_hierarchical_model(y_train_all, y_train_pred)
            print_evaluation_results(train_results)

            train_accuracy = train_results['total_accuracy']

            # Test Set Evaluation
            print("\n" + "=" * 80)
            print("Test Set Evaluation")
            print("=" * 80)
            print()

            test_accuracy = None

            try:
                print(f"Loading test data...")
                X_test_classes, _, _ = load_multiclass_data(
                    str(data_file), sheet_name='Test', skiprows=4
                )
                X1_test, X2_test, X3_test = X_test_classes

                print(f"Test data loaded:")
                print(f"  Class 1: {len(X1_test)} samples")
                print(f"  Class 2: {len(X2_test)} samples")
                print(f"  Class 3: {len(X3_test)} samples")
                print()

                X_test_all = np.vstack([X1_test, X2_test, X3_test])
                y_test_all = np.concatenate([
                    np.ones(len(X1_test), dtype=int),
                    2 * np.ones(len(X2_test), dtype=int),
                    3 * np.ones(len(X3_test), dtype=int)
                ])

                y_test_pred = hc.predict(X_test_all)
                test_results = evaluate_hierarchical_model(y_test_all, y_test_pred)
                print_evaluation_results(test_results)

                test_accuracy = test_results['total_accuracy']

            except Exception as e:
                print(f"Warning: Could not load or evaluate test data: {e}")

            print("\n" + "=" * 80)
            print("Test Complete!")
            print("=" * 80)
            print(f"Training Duration: {train_duration}")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            if test_accuracy is not None:
                print(f"Test Accuracy: {test_accuracy:.4f}")
            print(f"Log saved to: {log_file}")
            print("=" * 80)

        sys.stdout = original_stdout

        return {
            'success': True,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_duration': train_duration,
            'log_file': log_file
        }

    except Exception as e:
        sys.stdout = original_stdout
        print(f"Error testing {dataset_name}: {e}")
        return {'success': False, 'error': str(e), 'log_file': log_file}


def main():
    """Run all 7 fixed datasets."""
    print("=" * 80)
    print("Test2 Strategy - 7 Fixed Datasets")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Datasets to test:")
    for i, (name, _) in enumerate(DATASETS, 1):
        print(f"  {i}. {name}")
    print()
    print("Estimated time: 1-2 hours")
    print("=" * 80)
    print()

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)

    start_time = datetime.now()
    results = {}

    # Run tests sequentially
    for i, (dataset_name, data_path) in enumerate(DATASETS, 1):
        print("\n" + "=" * 80)
        print(f"[{i}/{len(DATASETS)}] Testing: {dataset_name}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 80)

        result = test_single_dataset(dataset_name, data_path, results_dir)
        results[dataset_name] = result

        if result['success']:
            print(f"✓ {dataset_name} completed successfully")
            print(f"  Train Accuracy: {result['train_accuracy']:.4f}")
            if result['test_accuracy'] is not None:
                print(f"  Test Accuracy:  {result['test_accuracy']:.4f}")
            print(f"  Duration: {result['train_duration']}")
        else:
            print(f"✗ {dataset_name} failed: {result.get('error', 'Unknown error')}")

    # Generate summary
    end_time = datetime.now()
    total_duration = end_time - start_time

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = results_dir / f"test2_fixed_datasets_summary_{timestamp}.txt"

    with open(summary_file, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Test2 Strategy - 7 Fixed Datasets Summary\n")
        f.write("=" * 80 + "\n")
        f.write(f"Test Date: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Total Duration: {total_duration}\n\n")

        f.write("Results:\n")
        f.write("-" * 80 + "\n")
        for dataset_name, result in results.items():
            if result['success']:
                f.write(f"{dataset_name:20s} - Train: {result['train_accuracy']:.4f}")
                if result['test_accuracy'] is not None:
                    f.write(f", Test: {result['test_accuracy']:.4f}")
                f.write(f", Duration: {result['train_duration']}\n")
            else:
                f.write(f"{dataset_name:20s} - FAILED: {result.get('error', 'Unknown')}\n")

        f.write("\n" + "-" * 80 + "\n")
        passed = sum(1 for r in results.values() if r['success'])
        f.write(f"Total: {passed}/{len(DATASETS)} tests passed\n")

    print("\n" + "=" * 80)
    print("All Tests Completed!")
    print("=" * 80)
    print(f"Total duration: {total_duration}")
    print(f"Summary saved to: {summary_file}")
    print("=" * 80)

    return 0 if all(r['success'] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
