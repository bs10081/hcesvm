#!/usr/bin/env python3
"""Extract detailed accuracy report from test2 log files."""

import re
from pathlib import Path
from datetime import datetime

# Define dataset order
DATASETS = [
    "Abalone",
    "Car_Evaluation",
    "Wine_Quality",
    "Balance",
    "Contraceptive",
    "Hayes_Roth",
    "New_Thyroid",
    "Squash_Stored",
    "Squash_Unstored",
    "TAE",
    "Thyroid",
    "Wine",
]


def extract_accuracy_from_log(log_file):
    """Extract training and testing accuracy from a log file."""
    with open(log_file, 'r') as f:
        content = f.read()

    result = {
        'dataset': None,
        'class_roles': {},
        'test2_applied': False,
        'train': {'total': None, 'class1': None, 'class2': None, 'class3': None, 'mae': None},
        'test': {'total': None, 'class1': None, 'class2': None, 'class3': None, 'mae': None}
    }

    # Extract dataset name
    match = re.search(r'Dataset: (.+)', content)
    if match:
        result['dataset'] = match.group(1)

    # Extract class roles
    maj_match = re.search(r'Majority:\s+Class (\d+)', content)
    med_match = re.search(r'Medium:\s+Class (\d+)', content)
    min_match = re.search(r'Minority:\s+Class (\d+)', content)

    if maj_match and med_match and min_match:
        result['class_roles'] = {
            'majority': int(maj_match.group(1)),
            'medium': int(med_match.group(1)),
            'minority': int(min_match.group(1))
        }
        result['test2_applied'] = (result['class_roles']['majority'] == 2)

    # Extract Training Set Evaluation
    train_section = re.search(
        r'Training Set Evaluation\n={80}\n\n(.+?)(?=\n={80}\nTest Set Evaluation|\n={80}\n\n\n={80}\n|$)',
        content,
        re.DOTALL
    )

    if train_section:
        train_text = train_section.group(1)

        # Find Evaluation Results section
        eval_section = re.search(r'Evaluation Results\n={60}\n\n(.+?)(?=Confusion Matrix|$)', train_text, re.DOTALL)
        if eval_section:
            eval_text = eval_section.group(1)

            # Total accuracy
            total_match = re.search(r'Total Accuracy:\s+([\d.]+)', eval_text)
            if total_match:
                result['train']['total'] = float(total_match.group(1))

            # Per-class accuracy - find the section after "Per-Class Accuracy:"
            per_class_section = re.search(r'Per-Class Accuracy:\s*\n(.+?)(?=\n\n|$)', eval_text, re.DOTALL)
            if per_class_section:
                per_class_text = per_class_section.group(1)
                class1_match = re.search(r'Class 1:\s+([\d.]+)', per_class_text)
                class2_match = re.search(r'Class 2:\s+([\d.]+)', per_class_text)
                class3_match = re.search(r'Class 3:\s+([\d.]+)', per_class_text)

                if class1_match:
                    result['train']['class1'] = float(class1_match.group(1))
                if class2_match:
                    result['train']['class2'] = float(class2_match.group(1))
                if class3_match:
                    result['train']['class3'] = float(class3_match.group(1))

            # MAE
            mae_match = re.search(r'MAE:\s+([\d.]+)', eval_text)
            if mae_match:
                result['train']['mae'] = float(mae_match.group(1))

    # Extract Test Set Evaluation
    test_section = re.search(
        r'Test Set Evaluation\n={80}\n\n(.+?)(?=\n={80}\nTest Complete|\n={80}\n\n\n={80}\n|$)',
        content,
        re.DOTALL
    )

    if test_section:
        test_text = test_section.group(1)

        # Check if test was skipped
        if 'Skipping test set evaluation' in test_text or 'Could not load or evaluate test data' in test_text:
            result['test'] = None
        else:
            # Find Evaluation Results section
            eval_section = re.search(r'Evaluation Results\n={60}\n\n(.+?)(?=Confusion Matrix|$)', test_text, re.DOTALL)
            if eval_section:
                eval_text = eval_section.group(1)

                # Total accuracy
                total_match = re.search(r'Total Accuracy:\s+([\d.]+)', eval_text)
                if total_match:
                    result['test']['total'] = float(total_match.group(1))

                # Per-class accuracy - find the section after "Per-Class Accuracy:"
                per_class_section = re.search(r'Per-Class Accuracy:\s*\n(.+?)(?=\n\n|$)', eval_text, re.DOTALL)
                if per_class_section:
                    per_class_text = per_class_section.group(1)
                    class1_match = re.search(r'Class 1:\s+([\d.]+)', per_class_text)
                    class2_match = re.search(r'Class 2:\s+([\d.]+)', per_class_text)
                    class3_match = re.search(r'Class 3:\s+([\d.]+)', per_class_text)

                    if class1_match:
                        result['test']['class1'] = float(class1_match.group(1))
                    if class2_match:
                        result['test']['class2'] = float(class2_match.group(1))
                    if class3_match:
                        result['test']['class3'] = float(class3_match.group(1))

                # MAE
                mae_match = re.search(r'MAE:\s+([\d.]+)', eval_text)
                if mae_match:
                    result['test']['mae'] = float(mae_match.group(1))

    return result


def main():
    """Extract and format accuracy reports."""
    results_dir = Path(__file__).parent / "results"

    # Find all test2 log files
    log_files = sorted(results_dir.glob("test2_*.log"))

    print("=" * 120)
    print("Test2 Strategy - Detailed Accuracy Report")
    print("=" * 120)
    print()

    all_results = []

    for log_file in log_files:
        result = extract_accuracy_from_log(log_file)
        if result['dataset']:
            all_results.append(result)

    # Sort by dataset order
    dataset_order = {name: i for i, name in enumerate(DATASETS)}
    all_results.sort(key=lambda x: dataset_order.get(x['dataset'], 999))

    # Print detailed results
    for i, result in enumerate(all_results, 1):
        print(f"\n{'=' * 120}")
        print(f"{i}. {result['dataset']}")
        print('=' * 120)

        # Class roles
        if result['class_roles']:
            roles = result['class_roles']
            print(f"Class Roles: Majority=Class {roles['majority']}, Medium=Class {roles['medium']}, Minority=Class {roles['minority']}")
            print(f"Test2 Rule Applied: {'Yes' if result['test2_applied'] else 'No'}")
        print()

        # Training results
        print("TRAINING SET:")
        train = result['train']
        if train['total'] is not None:
            print(f"  Total Accuracy:  {train['total']:.4f} ({train['total']*100:.2f}%)")
            print(f"  Per-class Accuracy:")
            if train['class1'] is not None:
                print(f"    Class 1: {train['class1']:.4f} ({train['class1']*100:.2f}%)")
            if train['class2'] is not None:
                print(f"    Class 2: {train['class2']:.4f} ({train['class2']*100:.2f}%)")
            if train['class3'] is not None:
                print(f"    Class 3: {train['class3']:.4f} ({train['class3']*100:.2f}%)")
            if train['mae'] is not None:
                print(f"  MAE: {train['mae']:.4f}")

        print()

        # Testing results
        if result['test'] is not None:
            test = result['test']
            if test['total'] is not None:
                print("TESTING SET:")
                print(f"  Total Accuracy:  {test['total']:.4f} ({test['total']*100:.2f}%)")
                print(f"  Per-class Accuracy:")
                if test['class1'] is not None:
                    print(f"    Class 1: {test['class1']:.4f} ({test['class1']*100:.2f}%)")
                if test['class2'] is not None:
                    print(f"    Class 2: {test['class2']:.4f} ({test['class2']*100:.2f}%)")
                if test['class3'] is not None:
                    print(f"    Class 3: {test['class3']:.4f} ({test['class3']*100:.2f}%)")
                if test['mae'] is not None:
                    print(f"  MAE: {test['mae']:.4f}")
            else:
                print("TESTING SET: N/A")
        else:
            print("TESTING SET: N/A")

    # Summary table
    print("\n\n" + "=" * 120)
    print("SUMMARY TABLE")
    print("=" * 120)
    print()

    # Header
    print(f"{'#':<3} {'Dataset':<20} {'Test2':<7} {'Train Total':<12} {'Train C1':<10} {'Train C2':<10} {'Train C3':<10} {'Test Total':<12} {'Test C1':<10} {'Test C2':<10} {'Test C3':<10}")
    print("-" * 120)

    for i, result in enumerate(all_results, 1):
        dataset = result['dataset']
        test2 = 'Yes' if result['test2_applied'] else 'No'

        train = result['train']
        train_total = f"{train['total']:.4f}" if train['total'] is not None else "N/A"
        train_c1 = f"{train['class1']:.4f}" if train['class1'] is not None else "N/A"
        train_c2 = f"{train['class2']:.4f}" if train['class2'] is not None else "N/A"
        train_c3 = f"{train['class3']:.4f}" if train['class3'] is not None else "N/A"

        if result['test'] is not None:
            test = result['test']
            test_total = f"{test['total']:.4f}" if test['total'] is not None else "N/A"
            test_c1 = f"{test['class1']:.4f}" if test['class1'] is not None else "N/A"
            test_c2 = f"{test['class2']:.4f}" if test['class2'] is not None else "N/A"
            test_c3 = f"{test['class3']:.4f}" if test['class3'] is not None else "N/A"
        else:
            test_total = test_c1 = test_c2 = test_c3 = "N/A"

        print(f"{i:<3} {dataset:<20} {test2:<7} {train_total:<12} {train_c1:<10} {train_c2:<10} {train_c3:<10} {test_total:<12} {test_c1:<10} {test_c2:<10} {test_c3:<10}")

    print()
    print("=" * 120)
    print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 120)


if __name__ == "__main__":
    main()
