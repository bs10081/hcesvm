#!/usr/bin/env python3
"""Sequential execution comparing Test2 vs Test3 strategies on all datasets."""

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


def test_single_strategy(dataset_name: str, data_path: str, strategy: str,
                         X1_train, X2_train, X3_train, X_test_all, y_test_all,
                         results_dir: Path) -> dict:
    """Test a single dataset with a specific strategy.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset Excel file
        strategy: "test2" or "test3"
        X1_train, X2_train, X3_train: Training data
        X_test_all, y_test_all: Test data
        results_dir: Directory to save results

    Returns:
        dict with test results and status
    """
    # Prepare log file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = results_dir / f"{strategy}_{dataset_name}_{timestamp}.log"

    # Redirect stdout to log file
    original_stdout = sys.stdout

    try:
        with open(log_file, 'w') as f:
            sys.stdout = TeeOutput(f, original_stdout)

            print("=" * 80)
            print(f"Hierarchical CE-SVM - {strategy.upper()} Strategy")
            print(f"Dataset: {dataset_name}")
            print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print("=" * 80)
            print()

            # Print training data info
            print(f"Training data:")
            print(f"  Class 1: {len(X1_train)} samples")
            print(f"  Class 2: {len(X2_train)} samples")
            print(f"  Class 3: {len(X3_train)} samples")
            print(f"  Features: {X1_train.shape[1]}")
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

            # Train hierarchical classifier
            print("=" * 80)
            print(f"Training Hierarchical Classifier - {strategy.upper()} Strategy")
            print("=" * 80)
            print()

            if strategy == "test2":
                print("Test2 Strategy:")
                print("  H1: {minority, medium} vs majority")
                print("  H2: minority vs medium")
                print("  Objective function modified based on class roles")
                print()
            elif strategy == "test3":
                print("Test3 Strategy:")
                print("  H1: {minority, medium} vs majority")
                print("  H2: minority vs medium")
                print("  Sample-weighted objective: -(1/s+)·l+ - (1/s-)·l-")
                print()

            train_start = datetime.now()

            try:
                hc = HierarchicalCESVM(cesvm_params=params, strategy=strategy)
                hc.fit(X1_train, X2_train, X3_train)
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

            X_train_all = np.vstack([X1_train, X2_train, X3_train])
            y_train_all = np.concatenate([
                np.ones(len(X1_train), dtype=int),
                2 * np.ones(len(X2_train), dtype=int),
                3 * np.ones(len(X3_train), dtype=int)
            ])

            y_train_pred = hc.predict(X_train_all)
            train_results = evaluate_hierarchical_model(y_train_all, y_train_pred)
            print_evaluation_results(train_results)

            # Test Set Evaluation
            print("\n" + "=" * 80)
            print("Test Set Evaluation")
            print("=" * 80)
            print()

            y_test_pred = hc.predict(X_test_all)
            test_results = evaluate_hierarchical_model(y_test_all, y_test_pred)
            print_evaluation_results(test_results)

            print("\n" + "=" * 80)
            print("Test Complete!")
            print("=" * 80)
            print(f"Training Duration: {train_duration}")
            print(f"Training Accuracy: {train_results['total_accuracy']:.4f}")
            print(f"Test Accuracy: {test_results['total_accuracy']:.4f}")
            print(f"Log saved to: {log_file}")
            print("=" * 80)

        # Restore stdout
        sys.stdout = original_stdout

        return {
            'success': True,
            'train_results': train_results,
            'test_results': test_results,
            'train_duration': train_duration,
            'log_file': log_file,
            'class_roles': summary.get('class_roles', None)
        }

    except Exception as e:
        # Restore stdout if error occurs
        sys.stdout = original_stdout
        print(f"Error testing {dataset_name} with {strategy}: {e}")
        return {
            'success': False,
            'error': str(e),
            'log_file': log_file
        }


def test_dataset_comparison(dataset_name: str, data_path: str, results_dir: Path) -> dict:
    """Test a single dataset with both Test2 and Test3 strategies.

    Args:
        dataset_name: Name of the dataset
        data_path: Path to the dataset Excel file
        results_dir: Directory to save results

    Returns:
        dict with comparison results
    """
    # Expand home directory
    data_file = Path(data_path).expanduser()

    print("=" * 80)
    print(f"Loading Dataset: {dataset_name}")
    print("=" * 80)
    print(f"File: {data_file}")
    print()

    # Check if file exists
    if not data_file.exists():
        print(f"Error: {data_file} not found")
        return {
            'success': False,
            'error': 'File not found'
        }

    # Load training data
    try:
        X_classes, n_classes_list, n_features = load_multiclass_data(
            str(data_file),
            sheet_name='Train',
            skiprows=5
        )
    except Exception as e:
        print(f"Error loading training data: {e}")
        return {
            'success': False,
            'error': f'Failed to load training data: {e}'
        }

    X1_train, X2_train, X3_train = X_classes

    print(f"Training data loaded:")
    print(f"  Class 1: {len(X1_train)} samples")
    print(f"  Class 2: {len(X2_train)} samples")
    print(f"  Class 3: {len(X3_train)} samples")
    print(f"  Features: {n_features}")
    print()

    # Load test data
    try:
        skiprows_test = detect_test_skiprows(data_file)
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

    except Exception as e:
        print(f"Error loading test data: {e}")
        return {
            'success': False,
            'error': f'Failed to load test data: {e}'
        }

    # Test with Test2 strategy
    print("\n" + "=" * 80)
    print("Testing with Test2 Strategy")
    print("=" * 80)
    test2_result = test_single_strategy(
        dataset_name, data_path, "test2",
        X1_train, X2_train, X3_train, X_test_all, y_test_all,
        results_dir
    )

    # Test with Test3 strategy
    print("\n" + "=" * 80)
    print("Testing with Test3 Strategy")
    print("=" * 80)
    test3_result = test_single_strategy(
        dataset_name, data_path, "test3",
        X1_train, X2_train, X3_train, X_test_all, y_test_all,
        results_dir
    )

    return {
        'success': test2_result['success'] and test3_result['success'],
        'test2': test2_result,
        'test3': test3_result
    }


def main():
    """Run comparison tests on all datasets sequentially."""
    print("=" * 80)
    print("Test2 vs Test3 Strategy Comparison")
    print("=" * 80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Datasets to test:")
    for i, (name, path) in enumerate(DATASETS, 1):
        print(f"  {i:2d}. {name}")
    print()
    print("Strategies: test2, test3")
    print("Time limit: 1800 seconds (30 minutes) per classifier")
    print("Estimated total time: 24-48 hours")
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

        result = test_dataset_comparison(dataset_name, data_path, results_dir)
        results[dataset_name] = result

        if result['success']:
            print(f"✓ {dataset_name} completed successfully")
            print(f"  Test2 - Train: {result['test2']['train_results']['total_accuracy']:.4f}, Test: {result['test2']['test_results']['total_accuracy']:.4f}")
            print(f"  Test3 - Train: {result['test3']['train_results']['total_accuracy']:.4f}, Test: {result['test3']['test_results']['total_accuracy']:.4f}")
        else:
            print(f"✗ {dataset_name} failed")

    # Generate comparison report
    end_time = datetime.now()
    total_duration = end_time - start_time

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    report_file = results_dir / f"TEST2_VS_TEST3_COMPARISON_{timestamp}.md"

    with open(report_file, 'w') as f:
        f.write("# Test2 vs Test3 策略比較報告\n\n")
        f.write(f"**測試日期**: {start_time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write(f"**總執行時間**: {total_duration}\n\n")
        f.write("---\n\n")

        f.write("## 總覽表\n\n")
        f.write("| Dataset | Test2 Train | Test2 Test | Test3 Train | Test3 Test | Winner (Test) |\n")
        f.write("|---------|-------------|------------|-------------|------------|---------------|\n")

        for dataset_name, result in results.items():
            if result['success']:
                t2_train = result['test2']['train_results']['total_accuracy']
                t2_test = result['test2']['test_results']['total_accuracy']
                t3_train = result['test3']['train_results']['total_accuracy']
                t3_test = result['test3']['test_results']['total_accuracy']

                if t2_test > t3_test:
                    winner = "Test2"
                elif t3_test > t2_test:
                    winner = "Test3"
                else:
                    winner = "Tie"

                f.write(f"| {dataset_name} | {t2_train:.4f} | {t2_test:.4f} | {t3_train:.4f} | {t3_test:.4f} | {winner} |\n")
            else:
                f.write(f"| {dataset_name} | FAIL | FAIL | FAIL | FAIL | N/A |\n")

        f.write("\n---\n\n")

        f.write("## 各 Class 準確率比較\n\n")
        f.write("| Dataset | Strategy | C1 Train | C2 Train | C3 Train | C1 Test | C2 Test | C3 Test |\n")
        f.write("|---------|----------|----------|----------|----------|---------|---------|----------|\n")

        for dataset_name, result in results.items():
            if result['success']:
                # Test2
                t2_train = result['test2']['train_results']['per_class_accuracy']
                t2_test = result['test2']['test_results']['per_class_accuracy']
                f.write(f"| {dataset_name} | Test2 | {t2_train.get(1, 0):.4f} | {t2_train.get(2, 0):.4f} | {t2_train.get(3, 0):.4f} | {t2_test.get(1, 0):.4f} | {t2_test.get(2, 0):.4f} | {t2_test.get(3, 0):.4f} |\n")

                # Test3
                t3_train = result['test3']['train_results']['per_class_accuracy']
                t3_test = result['test3']['test_results']['per_class_accuracy']
                f.write(f"| {dataset_name} | Test3 | {t3_train.get(1, 0):.4f} | {t3_train.get(2, 0):.4f} | {t3_train.get(3, 0):.4f} | {t3_test.get(1, 0):.4f} | {t3_test.get(2, 0):.4f} | {t3_test.get(3, 0):.4f} |\n")

        f.write("\n---\n\n")

        f.write("## 統計摘要\n\n")

        # Count winners
        test2_wins = 0
        test3_wins = 0
        ties = 0

        for dataset_name, result in results.items():
            if result['success']:
                t2_test = result['test2']['test_results']['total_accuracy']
                t3_test = result['test3']['test_results']['total_accuracy']

                if t2_test > t3_test:
                    test2_wins += 1
                elif t3_test > t2_test:
                    test3_wins += 1
                else:
                    ties += 1

        total_successful = sum(1 for r in results.values() if r['success'])

        f.write(f"- **總資料集數量**: {len(DATASETS)}\n")
        f.write(f"- **成功測試**: {total_successful}\n")
        f.write(f"- **Test3 勝出**: {test3_wins}/{total_successful}\n")
        f.write(f"- **Test2 勝出**: {test2_wins}/{total_successful}\n")
        f.write(f"- **平手**: {ties}/{total_successful}\n\n")

        if total_successful > 0:
            t2_train_accs = [r['test2']['train_results']['total_accuracy'] for r in results.values() if r['success']]
            t2_test_accs = [r['test2']['test_results']['total_accuracy'] for r in results.values() if r['success']]
            t3_train_accs = [r['test3']['train_results']['total_accuracy'] for r in results.values() if r['success']]
            t3_test_accs = [r['test3']['test_results']['total_accuracy'] for r in results.values() if r['success']]

            f.write(f"### 平均準確率\n\n")
            f.write(f"- **Test2 平均 Training Accuracy**: {np.mean(t2_train_accs):.4f}\n")
            f.write(f"- **Test2 平均 Test Accuracy**: {np.mean(t2_test_accs):.4f}\n")
            f.write(f"- **Test3 平均 Training Accuracy**: {np.mean(t3_train_accs):.4f}\n")
            f.write(f"- **Test3 平均 Test Accuracy**: {np.mean(t3_test_accs):.4f}\n\n")

        f.write("\n---\n\n")
        f.write(f"**報告產生時間**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    # Print final summary to console
    print("\n" + "=" * 80)
    print("All Comparison Tests Completed!")
    print("=" * 80)
    print(f"Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"End time: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total duration: {total_duration}")
    print()
    print("Results Summary:")

    for dataset_name, result in results.items():
        if result['success']:
            t2_test = result['test2']['test_results']['total_accuracy']
            t3_test = result['test3']['test_results']['total_accuracy']
            winner = "Test3" if t3_test > t2_test else ("Test2" if t2_test > t3_test else "Tie")
            print(f"  {dataset_name:20s} - Test2: {t2_test:.4f}, Test3: {t3_test:.4f} [{winner}]")
        else:
            print(f"  {dataset_name:20s} - FAILED")

    print()
    total_successful = sum(1 for r in results.values() if r['success'])
    print(f"Total: {total_successful}/{len(DATASETS)} tests passed")
    print()
    print(f"Comparison report saved to: {report_file}")
    print("=" * 80)

    # Return exit code based on results
    return 0 if all(r['success'] for r in results.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
