#!/usr/bin/env python3
"""Extract detailed accuracy and weight information from Test2 logs."""

import re
from pathlib import Path
from datetime import datetime


def parse_log_file(log_path):
    """Parse a single log file to extract detailed metrics."""
    with open(log_path, 'r') as f:
        content = f.read()

    result = {
        'dataset': None,
        'train_total_acc': None,
        'train_class_accs': {},
        'test_total_acc': None,
        'test_class_accs': {},
        'h1_l1_norm': None,
        'h2_l1_norm': None,
        'h1_selected_features': None,
        'h2_selected_features': None,
        'total_features': None,
        'class_roles': {},
        'test2_rule_applied': False
    }

    # Extract dataset name
    dataset_match = re.search(r'Dataset: (\w+)', content)
    if dataset_match:
        result['dataset'] = dataset_match.group(1)

    # Extract total features
    features_match = re.search(r'Features: (\d+)', content)
    if features_match:
        result['total_features'] = int(features_match.group(1))

    # Extract class roles
    majority_match = re.search(r'Majority:\s+Class (\d+)', content)
    medium_match = re.search(r'Medium:\s+Class (\d+)', content)
    minority_match = re.search(r'Minority:\s+Class (\d+)', content)
    if majority_match:
        result['class_roles']['majority'] = int(majority_match.group(1))
        result['class_roles']['medium'] = int(medium_match.group(1))
        result['class_roles']['minority'] = int(minority_match.group(1))
        result['test2_rule_applied'] = (result['class_roles']['majority'] == 2)

    # Extract H1 solution
    h1_match = re.search(r'H1 Solution:.*?Selected features: (\d+)/(\d+).*?L1 norm: ([\d.]+)', content, re.DOTALL)
    if h1_match:
        result['h1_selected_features'] = int(h1_match.group(1))
        result['h1_l1_norm'] = float(h1_match.group(3))

    # Extract H2 solution
    h2_match = re.search(r'H2 Solution:.*?Selected features: (\d+)/(\d+).*?L1 norm: ([\d.]+)', content, re.DOTALL)
    if h2_match:
        result['h2_selected_features'] = int(h2_match.group(1))
        result['h2_l1_norm'] = float(h2_match.group(3))

    # Extract training set evaluation
    train_section = re.search(r'Training Set Evaluation.*?Total Accuracy: ([\d.]+).*?Per-Class Accuracy:(.*?)Class Distribution:', content, re.DOTALL)
    if train_section:
        result['train_total_acc'] = float(train_section.group(1))
        class_accs = re.findall(r'Class (\d+): ([\d.]+)', train_section.group(2))
        for class_num, acc in class_accs:
            result['train_class_accs'][int(class_num)] = float(acc)

    # Extract test set evaluation
    test_section = re.search(r'Test Set Evaluation.*?Total Accuracy: ([\d.]+).*?Per-Class Accuracy:(.*?)Class Distribution:', content, re.DOTALL)
    if test_section:
        result['test_total_acc'] = float(test_section.group(1))
        class_accs = re.findall(r'Class (\d+): ([\d.]+)', test_section.group(2))
        for class_num, acc in class_accs:
            result['test_class_accs'][int(class_num)] = float(acc)

    return result


def generate_detailed_report():
    """Generate detailed report for all 7 fixed datasets."""

    # Find all test2_fixed log files
    results_dir = Path('results')
    log_files = sorted(results_dir.glob('test2_fixed_*_202602*.log'))

    if not log_files:
        print("No log files found!")
        return

    all_results = []
    for log_file in log_files:
        result = parse_log_file(log_file)
        if result['dataset']:
            all_results.append(result)

    # Sort by dataset name
    all_results.sort(key=lambda x: x['dataset'])

    # Generate markdown report
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    report = []
    report.append("# Test2 策略 - 詳細準確率與權重報告")
    report.append("")
    report.append(f"**生成時間**: {timestamp}")
    report.append(f"**資料集數量**: {len(all_results)}")
    report.append("")
    report.append("---")
    report.append("")

    # Summary table
    report.append("## 📊 總覽表")
    report.append("")
    report.append("| # | Dataset | Test2 | Train Total | Test Total | H1 L1 Norm | H2 L1 Norm | Features |")
    report.append("|---|---------|-------|-------------|------------|------------|------------|----------|")

    for i, r in enumerate(all_results, 1):
        test2_mark = "✅" if r['test2_rule_applied'] else "❌"
        train_acc = f"{r['train_total_acc']:.4f}" if r['train_total_acc'] is not None else "N/A"
        test_acc = f"{r['test_total_acc']:.4f}" if r['test_total_acc'] is not None else "N/A"
        h1_norm = f"{r['h1_l1_norm']:.4f}" if r['h1_l1_norm'] is not None else "N/A"
        h2_norm = f"{r['h2_l1_norm']:.4f}" if r['h2_l1_norm'] is not None else "N/A"
        features = f"{r['h1_selected_features']}/{r['total_features']}" if r['h1_selected_features'] is not None else "N/A"

        report.append(f"| {i} | {r['dataset']} | {test2_mark} | {train_acc} | {test_acc} | {h1_norm} | {h2_norm} | {features} |")

    report.append("")
    report.append("**註**: ")
    report.append("- **H1 L1 Norm**: 第一層分類器的權重 L1 範數")
    report.append("- **H2 L1 Norm**: 第二層分類器的權重 L1 範數")
    report.append("- **Features**: 選擇的特徵數/總特徵數")
    report.append("")
    report.append("---")
    report.append("")

    # Detailed per-dataset reports
    report.append("## 📈 各資料集詳細結果")
    report.append("")

    for i, r in enumerate(all_results, 1):
        report.append(f"### {i}. {r['dataset']}")
        report.append("")

        # Class roles
        if r['class_roles']:
            test2_status = "✅ Yes" if r['test2_rule_applied'] else "❌ No"
            report.append(f"**Class Roles**: Majority=Class {r['class_roles']['majority']}, "
                         f"Medium=Class {r['class_roles']['medium']}, "
                         f"Minority=Class {r['class_roles']['minority']}")
            report.append(f"**Test2 Rule Applied**: {test2_status}")
            report.append("")

        # Model weights
        report.append("**模型權重**:")
        report.append(f"- H1 L1 Norm: {r['h1_l1_norm']:.6f}" if r['h1_l1_norm'] is not None else "- H1 L1 Norm: N/A")
        report.append(f"- H2 L1 Norm: {r['h2_l1_norm']:.6f}" if r['h2_l1_norm'] is not None else "- H2 L1 Norm: N/A")
        report.append(f"- Selected Features: {r['h1_selected_features']}/{r['total_features']}"
                     if r['h1_selected_features'] is not None else "- Selected Features: N/A")
        report.append("")

        # Accuracy table
        report.append("**準確率**:")
        report.append("")
        report.append("| Dataset | Total Acc | Class 1 Acc | Class 2 Acc | Class 3 Acc |")
        report.append("|---------|-----------|-------------|-------------|-------------|")

        # Training row
        train_total = f"{r['train_total_acc']:.4f}" if r['train_total_acc'] else "N/A"
        train_c1 = f"{r['train_class_accs'].get(1, 0):.4f}" if 1 in r['train_class_accs'] else "N/A"
        train_c2 = f"{r['train_class_accs'].get(2, 0):.4f}" if 2 in r['train_class_accs'] else "N/A"
        train_c3 = f"{r['train_class_accs'].get(3, 0):.4f}" if 3 in r['train_class_accs'] else "N/A"
        report.append(f"| **Training** | {train_total} | {train_c1} | {train_c2} | {train_c3} |")

        # Testing row
        test_total = f"{r['test_total_acc']:.4f}" if r['test_total_acc'] else "N/A"
        test_c1 = f"{r['test_class_accs'].get(1, 0):.4f}" if 1 in r['test_class_accs'] else "N/A"
        test_c2 = f"{r['test_class_accs'].get(2, 0):.4f}" if 2 in r['test_class_accs'] else "N/A"
        test_c3 = f"{r['test_class_accs'].get(3, 0):.4f}" if 3 in r['test_class_accs'] else "N/A"
        report.append(f"| **Testing** | {test_total} | {test_c1} | {test_c2} | {test_c3} |")

        report.append("")
        report.append("---")
        report.append("")

    # Statistics
    report.append("## 📊 統計分析")
    report.append("")

    # Average norms
    h1_norms = [r['h1_l1_norm'] for r in all_results if r['h1_l1_norm'] is not None]
    h2_norms = [r['h2_l1_norm'] for r in all_results if r['h2_l1_norm'] is not None]

    if h1_norms:
        report.append(f"**平均 H1 L1 Norm**: {sum(h1_norms)/len(h1_norms):.6f}")
    if h2_norms:
        report.append(f"**平均 H2 L1 Norm**: {sum(h2_norms)/len(h2_norms):.6f}")

    # Average accuracies
    train_accs = [r['train_total_acc'] for r in all_results if r['train_total_acc']]
    test_accs = [r['test_total_acc'] for r in all_results if r['test_total_acc']]

    if train_accs:
        report.append(f"**平均訓練準確率**: {sum(train_accs)/len(train_accs):.4f}")
    if test_accs:
        report.append(f"**平均測試準確率**: {sum(test_accs)/len(test_accs):.4f}")

    report.append("")

    # Test2 rule statistics
    test2_applied = sum(1 for r in all_results if r['test2_rule_applied'])
    report.append(f"**Test2 規則應用**: {test2_applied}/{len(all_results)} 資料集")
    report.append("")

    report.append("---")
    report.append("")
    report.append(f"**報告生成時間**: {timestamp}")

    # Write to file
    output_file = Path('TEST2_DETAILED_ACCURACY_WEIGHT_REPORT.md')
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(report))

    print(f"✅ 詳細報告已生成: {output_file}")
    print(f"📊 處理了 {len(all_results)} 個資料集")

    # Also print to console
    print("\n" + "="*80)
    print("報告預覽:")
    print("="*80)
    for line in report[:50]:  # Print first 50 lines
        print(line)
    print("...")
    print(f"\n完整報告請查看: {output_file}")


if __name__ == "__main__":
    generate_detailed_report()
