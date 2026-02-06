#!/usr/bin/env python3
"""Sequential execution of Test2 strategy tests on all datasets."""

import sys
from pathlib import Path
from datetime import datetime
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


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

from hcesvm import HierarchicalCESVM, get_default_params
from hcesvm.utils import load_multiclass_data, evaluate_hierarchical_model, print_evaluation_results


# Define all datasets
DATASETS = [
    # Primary datasets
    ("Abalone", "~/Developer/NSVORA/datasets/primary/abalone/abalone_split.xlsx"),
    ("Car_Evaluation", "~/Developer/NSVORA/datasets/primary/car_evaluation/car_evaluation_split.xlsx"),
    ("Wine_Quality", "~/Developer/NSVORA/datasets/primary/wine_quality/wine_quality_split.xlsx"),
    # Archive datasets
    ("Balance", "~/Developer/NSVORA/Archive/balance_split.xlsx"),
    ("Contraceptive", "~/Developer/NSVORA/Archive/contraceptive_split.xlsx"),
    ("Hayes_Roth", "~/Developer/NSVORA/Archive/hayes-roth_split.xlsx"),
    ("New_Thyroid", "~/Developer/NSVORA/Archive/new-thyroid_split.xlsx"),
    ("Squash_Stored", "~/Developer/NSVORA/Archive/squash-stored_split.xlsx"),
    ("Squash_Unstored", "~/Developer/NSVORA/Archive/squash-unstored_split.xlsx"),
    ("TAE", "~/Developer/NSVORA/Archive/tae_split.xlsx"),
    ("Thyroid", "~/Developer/NSVORA/Archive/thyroid_split.xlsx"),
    ("Wine", "~/Developer/NSVORA/Archive/wine_split.xlsx"),
]


def detect_test_skiprows(data_file: Path) -> int:
    """Detect skiprows for Test sheet.

    Test sheets typically have skiprows=4 (one less than Train).
    Returns the appropriate skiprows value.
    """
    # Most datasets use skiprows=4 for Test sheet
    return 4


def test_single_dataset(dataset_name: str, data_path: str, results_dir: Path) -> dict:
    """Test a single dataset with Test2 strategy.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset Excel file
        results_dir: Directory to save results

    Returns:
        dict with test results and status
    """
    # Expand home directory
    data_file = Path(data_path).expanduser()

    # Prepare log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = results_dir / f"test2_{dataset_name}_{timestamp}.log"

    # Redirect stdout to log file
    original_stdout = sys.stdout

    try:
        with open(log_file, 'w') as f:
            sys.stdout = TeeOutput(f, original_stdout)

            print("=" * 80)
            print(f"Hierarchical CE-SVM - Test2 Strategy")
            print(f"Dataset: {dataset_name}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()

            # Check if file exists
            if not data_file.exists():
                print(f"Error: {data_file} not found")
                sys.stdout = original_stdout
                return {
                    'success': False,
                    'error': 'File not found',
                    'log_file': log_file
                }

            print(f"Loading training data from: {data_file}")
            print()

            # Load training data
            try:
                X_classes, n_classes_list, n_features = load_multiclass_data(
                    str(data_file),
                    sheet_name='Train',
                    skiprows=5
                )
            except Exception as e:
                print(f"Error loading training data: {e}")
                sys.stdout = original_stdout
                return {
                    'success': False,
                    'error': f'Failed to load training data: {e}',
                    'log_file': log_file
                }

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
            print("Test2 Strategy:")
            print("  H1: {minority, medium} vs majority")
            print("  H2: minority vs medium")
            print("  (Class roles determined dynamically based on sample counts)")
            print()

            train_start = datetime.now()

            try:
                hc = HierarchicalCESVM(cesvm_params=params, strategy="test2")
                hc.fit(X1, X2, X3)
            except Exception as e:
                print(f"Error during training: {e}")
                sys.stdout = original_stdout
                return {
                    'success': False,
                    'error': f'Training failed: {e}',
                    'log_file': log_file
                }

            train_end = datetime.now()
            train_duration = train_end - train_start

            print(f"\nTraining completed in: {train_duration}")
            print()

            # Get model summary
            summary = hc.get_model_summary()
            print("=" * 80)
            print("Model Summary")
            print("=" * 80)
            print(f"Status: {summary['status']}")
            print(f"Strategy: {summary['strategy']}")
            print(f"Features: {summary['n_features']}")

            if 'class_roles' in summary:
                print(f"\nClass Roles:")
                print(f"  Majority: Class {summary['class_roles']['majority']}")
                print(f"  Medium:   Class {summary['class_roles']['medium']}")
                print(f"  Minority: Class {summary['class_roles']['minority']}")

            # Extract accuracy mode settings from classifiers
            h1_accuracy_mode = None
            h2_accuracy_mode = None
            test2_rule_applied = False

            if hasattr(hc, 'h1') and hc.h1 is not None:
                h1_accuracy_mode = getattr(hc.h1, 'accuracy_mode', 'both')
            if hasattr(hc, 'h2') and hc.h2 is not None:
                h2_accuracy_mode = getattr(hc.h2, 'accuracy_mode', 'both')

            # Check if Test2 Rule is applied (majority == 2)
            if 'class_roles' in summary:
                test2_rule_applied = (summary['class_roles']['majority'] == 2)

            print(f"\nTest2 Rule Applied: {'Yes' if test2_rule_applied else 'No'} (Majority = Class {summary['class_roles'].get('majority', 'N/A')})")
            print(f"Accuracy Modes:")
            print(f"  H1: {h1_accuracy_mode}")
            print(f"  H2: {h2_accuracy_mode}")

            print(f"\nClassifier 1 (H1): {summary['h1']['description']}")
            h1 = summary['h1']
            print(f"  Objective: {h1['objective_value']:.6f}")
            print(f"  Selected Features: {h1['n_selected_features']}/{summary['n_features']}")
            print(f"  L1 Norm: {h1['l1_norm']:.6f}")
            if 'mip_gap' in h1:
                print(f"  MIP Gap: {h1['mip_gap']:.6e}")

            print(f"\nClassifier 2 (H2): {summary['h2']['description']}")
            h2 = summary['h2']
            print(f"  Objective: {h2['objective_value']:.6f}")
            print(f"  Selected Features: {h2['n_selected_features']}/{summary['n_features']}")
            print(f"  L1 Norm: {h2['l1_norm']:.6f}")
            if 'mip_gap' in h2:
                print(f"  MIP Gap: {h2['mip_gap']:.6e}")

            # Output complete H1 and H2 Weights and Bias
            print("\n" + "=" * 80)
            print("H1 Complete Weights and Bias")
            print("=" * 80)
            print(f"Description: {summary['h1']['description']}")
            print(f"\nIntercept (b): {hc.h1.intercept:.10f}")
            print(f"\nWeight vector (w) - {len(hc.h1.weights)} features:")
            for i, w in enumerate(hc.h1.weights):
                print(f"  w[{i}] = {w:.10f}")
            print(f"\nL1 Norm: {np.sum(np.abs(hc.h1.weights)):.10f}")

            print("\n" + "=" * 80)
            print("H2 Complete Weights and Bias")
            print("=" * 80)
            print(f"Description: {summary['h2']['description']}")
            print(f"\nIntercept (b): {hc.h2.intercept:.10f}")
            print(f"\nWeight vector (w) - {len(hc.h2.weights)} features:")
            for i, w in enumerate(hc.h2.weights):
                print(f"  w[{i}] = {w:.10f}")
            print(f"\nL1 Norm: {np.sum(np.abs(hc.h2.weights)):.10f}")

            # Training Set Evaluation
            print("\n" + "=" * 80)
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
            train_mae = train_results.get('mae', None)

            # Test Set Evaluation
            print("\n" + "=" * 80)
            print("Test Set Evaluation")
            print("=" * 80)
            print()

            test_accuracy = None
            test_mae = None

            try:
                # Try to load test data with skiprows=4
                skiprows_test = detect_test_skiprows(data_file)
                print(f"Loading test data (skiprows={skiprows_test})...")

                X_test_classes, _, _ = load_multiclass_data(
                    str(data_file),
                    sheet_name='Test',
                    skiprows=skiprows_test
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
                test_mae = test_results.get('mae', None)

            except Exception as e:
                print(f"Warning: Could not load or evaluate test data: {e}")
                print("Skipping test set evaluation.")

            print("\n" + "=" * 80)
            print("Test Complete!")
            print("=" * 80)
            print(f"Training Duration: {train_duration}")
            print(f"Training Accuracy: {train_accuracy:.4f}")
            if train_mae is not None:
                print(f"Training MAE: {train_mae:.4f}")
            if test_accuracy is not None:
                print(f"Test Accuracy: {test_accuracy:.4f}")
            if test_mae is not None:
                print(f"Test MAE: {test_mae:.4f}")
            print(f"Log saved to: {log_file}")
            print("=" * 80)

        # Restore stdout
        sys.stdout = original_stdout

        return {
            'success': True,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_mae': train_mae,
            'test_mae': test_mae,
            'train_duration': train_duration,
            'log_file': log_file,
            'class_roles': summary.get('class_roles', None),
            'test2_rule_applied': test2_rule_applied,
            'h1_accuracy_mode': h1_accuracy_mode,
            'h2_accuracy_mode': h2_accuracy_mode
        }

    except Exception as e:
        # Restore stdout if error occurs
        sys.stdout = original_stdout
        print(f"Error testing {dataset_name}: {e}")
        return {
            'success': False,
            'error': str(e),
            'log_file': log_file
        }


def main():
    """Run all datasets sequentially."""
    print("=" * 80)
    print("Sequential Execution of Test2 Strategy Tests")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Datasets to test:")
    for i, (name, path) in enumerate(DATASETS, 1):
        print(f"  {i:2d}. {name}")
    print()
    print("Strategy: test2 (dynamic)")
    print("Time limit: 1800 seconds (30 minutes) per classifier")
    print("Estimated total time: 12-24 hours")
    print()
    print("Test2 Rule:")
    print("  - If majority == 2:")
    print("    - H1: maximize minority class accuracy")
    print("    - H2: maximize medium class accuracy")
    print("  - Otherwise:")
    print("    - Both classifiers: maximize both classes (accuracy_mode='both')")
    print("=" * 80)
    print()

    # Create results directory
    results_dir = Path(__file__).parent.parent / "results"
    results_dir.mkdir(exist_ok=True)
    print(f"Results directory: {results_dir}")
    print()

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
            print(f"  Test2 Rule: {'Applied' if result['test2_rule_applied'] else 'Not Applied'}")
        else:
            print(f"✗ {dataset_name} failed: {result.get('error', 'Unknown error')}")

        print(f"  Log file: {result['log_file']}")

    # Generate summary report
    end_time = datetime.now()
    total_duration = end_time - start_time

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    summary_file = results_dir / f"test2_all_datasets_summary_{timestamp}.md"

    with open(summary_file, 'w') as f:
        f.write("# Test2 Strategy - All Datasets Test Summary\n\n")
        f.write(f"**Test Date**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**Total Duration**: {total_duration}\n\n")
        f.write("---\n\n")

        f.write("## Configuration\n\n")
        f.write("- **Strategy**: test2 (dynamic)\n")
        f.write("- **Time Limit**: 1800 seconds (30 minutes) per classifier\n")
        f.write("- **C_hyper**: 1.0\n")
        f.write("- **M**: 1000.0\n")
        f.write("- **MIP Gap**: 1e-4\n\n")
        f.write("### Test2 Rule\n\n")
        f.write("When majority class = Class 2:\n")
        f.write("- H1: `accuracy_mode='minority'` (maximize minority class accuracy)\n")
        f.write("- H2: `accuracy_mode='medium'` (maximize medium class accuracy)\n\n")
        f.write("Otherwise (majority ∈ {1, 3}):\n")
        f.write("- H1: `accuracy_mode='both'`\n")
        f.write("- H2: `accuracy_mode='both'`\n\n")
        f.write("---\n\n")

        f.write("## Results Summary\n\n")
        f.write("| # | Dataset | Status | Train Acc | Test Acc | Train MAE | Test MAE | Duration | Test2 Rule | Class Roles |\n")
        f.write("|---|---------|--------|-----------|----------|-----------|----------|----------|------------|-------------|\n")

        for i, (dataset_name, _) in enumerate(DATASETS, 1):
            result = results[dataset_name]
            status = "✓ PASS" if result['success'] else "✗ FAIL"

            if result['success']:
                train_acc = f"{result['train_accuracy']:.4f}"
                test_acc = f"{result['test_accuracy']:.4f}" if result['test_accuracy'] is not None else "N/A"
                train_mae = f"{result['train_mae']:.4f}" if result['train_mae'] is not None else "N/A"
                test_mae = f"{result['test_mae']:.4f}" if result['test_mae'] is not None else "N/A"
                duration = str(result['train_duration']).split('.')[0]  # Remove microseconds
                test2_rule = "Yes" if result['test2_rule_applied'] else "No"

                # Format class roles
                if result['class_roles']:
                    roles = result['class_roles']
                    class_roles = f"Maj:{roles['majority']}, Med:{roles['medium']}, Min:{roles['minority']}"
                else:
                    class_roles = "N/A"
            else:
                train_acc = "N/A"
                test_acc = "N/A"
                train_mae = "N/A"
                test_mae = "N/A"
                duration = "N/A"
                test2_rule = "N/A"
                class_roles = "N/A"

            f.write(f"| {i} | {dataset_name} | {status} | {train_acc} | {test_acc} | {train_mae} | {test_mae} | {duration} | {test2_rule} | {class_roles} |\n")

        f.write("\n---\n\n")
        f.write("## Detailed Results\n\n")

        for i, (dataset_name, _) in enumerate(DATASETS, 1):
            result = results[dataset_name]
            f.write(f"### {i}. {dataset_name}\n\n")

            if result['success']:
                f.write(f"- **Status**: ✓ PASS\n")
                f.write(f"- **Training Accuracy**: {result['train_accuracy']:.4f}\n")
                if result['test_accuracy'] is not None:
                    f.write(f"- **Test Accuracy**: {result['test_accuracy']:.4f}\n")
                if result['train_mae'] is not None:
                    f.write(f"- **Training MAE**: {result['train_mae']:.4f}\n")
                if result['test_mae'] is not None:
                    f.write(f"- **Test MAE**: {result['test_mae']:.4f}\n")
                f.write(f"- **Training Duration**: {result['train_duration']}\n")

                if result['class_roles']:
                    roles = result['class_roles']
                    f.write(f"- **Class Roles**:\n")
                    f.write(f"  - Majority: Class {roles['majority']}\n")
                    f.write(f"  - Medium: Class {roles['medium']}\n")
                    f.write(f"  - Minority: Class {roles['minority']}\n")

                f.write(f"- **Test2 Rule Applied**: {'Yes' if result['test2_rule_applied'] else 'No'}\n")
                f.write(f"- **Accuracy Modes**:\n")
                f.write(f"  - H1: {result['h1_accuracy_mode']}\n")
                f.write(f"  - H2: {result['h2_accuracy_mode']}\n")

                f.write(f"- **Log File**: `{result['log_file'].name}`\n\n")
            else:
                f.write(f"- **Status**: ✗ FAIL\n")
                f.write(f"- **Error**: {result.get('error', 'Unknown error')}\n")
                f.write(f"- **Log File**: `{result['log_file'].name}`\n\n")

        f.write("---\n\n")
        f.write("## Statistics\n\n")

        passed = sum(1 for r in results.values() if r['success'])
        total = len(results)
        success_rate = (passed / total) * 100 if total > 0 else 0

        f.write(f"- **Total Datasets**: {total}\n")
        f.write(f"- **Passed**: {passed}\n")
        f.write(f"- **Failed**: {total - passed}\n")
        f.write(f"- **Success Rate**: {success_rate:.1f}%\n\n")

        if passed > 0:
            train_accuracies = [r['train_accuracy'] for r in results.values() if r['success']]
            test_accuracies = [r['test_accuracy'] for r in results.values() if r['success'] and r['test_accuracy'] is not None]

            f.write(f"- **Average Training Accuracy**: {np.mean(train_accuracies):.4f}\n")
            if test_accuracies:
                f.write(f"- **Average Test Accuracy**: {np.mean(test_accuracies):.4f}\n")

            # Test2 Rule Statistics
            test2_applied_count = sum(1 for r in results.values() if r['success'] and r['test2_rule_applied'])
            f.write(f"- **Datasets where Test2 Rule Applied**: {test2_applied_count}/{passed}\n")

        f.write("\n---\n\n")
        f.write(f"**Report Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Print final summary to console
    print("\n" + "=" * 80)
    print("All Tests Completed!")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration}")
    print()
    print("Results Summary:")

    for dataset_name, result in results.items():
        if result['success']:
            status = f"✓ PASS - Train: {result['train_accuracy']:.4f}"
            if result['test_accuracy'] is not None:
                status += f", Test: {result['test_accuracy']:.4f}"
            status += f" [Test2: {'Yes' if result['test2_rule_applied'] else 'No'}]"
        else:
            status = f"✗ FAIL - {result.get('error', 'Unknown error')}"
        print(f"  {dataset_name:20s} - {status}")

    print()
    passed = sum(1 for r in results.values() if r['success'])
    total = len(results)
    print(f"Total: {passed}/{total} tests passed")
    print()
    print(f"Summary report saved to: {summary_file}")
    print("=" * 80)

    # Return exit code based on results
    return 0 if all(r['success'] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
