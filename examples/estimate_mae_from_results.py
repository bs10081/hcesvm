#!/usr/bin/env python3
"""
從已知的各類別準確率估算 MAE (Mean Absolute Error)

由於沒有完整混淆矩陣，我們使用保守估計：
- 正確分類：誤差為 0
- 錯誤分類：假設均勻分布到其他類別
"""

import numpy as np

def estimate_mae_from_accuracy(class_accuracies, class_samples):
    """
    從各類別準確率估算 MAE

    Args:
        class_accuracies: dict {1: acc1, 2: acc2, 3: acc3}
        class_samples: dict {1: n1, 2: n2, 3: n3}

    Returns:
        estimated_mae: float
    """
    total_error = 0.0
    total_samples = sum(class_samples.values())

    for true_class, accuracy in class_accuracies.items():
        n_samples = class_samples[true_class]
        n_correct = n_samples * accuracy
        n_wrong = n_samples * (1 - accuracy)

        # 正確分類的誤差為 0
        # 錯誤分類的誤差：假設均勻分布到其他類別
        # 對於類別 1：錯誤時預測為 2 或 3（誤差 1 或 2）
        # 對於類別 2：錯誤時預測為 1 或 3（誤差 1 或 1）
        # 對於類別 3：錯誤時預測為 1 或 2（誤差 2 或 1）

        if true_class == 1:
            # 假設錯誤均分給類別 2 和 3
            avg_error = (1 + 2) / 2  # 平均誤差 1.5
        elif true_class == 2:
            # 假設錯誤均分給類別 1 和 3
            avg_error = (1 + 1) / 2  # 平均誤差 1.0
        else:  # class 3
            # 假設錯誤均分給類別 1 和 2
            avg_error = (2 + 1) / 2  # 平均誤差 1.5

        total_error += n_wrong * avg_error

    return total_error / total_samples


# 已知的測試結果數據
results = {
    'Car_Evaluation': {
        'samples': {1: 242, 2: 91, 3: 13},
        'SVOR': {1: 0.9215, 2: 0.6813, 3: 0.5385},
        'NPSVOR': {1: 0.9298, 2: 0.6923, 3: 0.6154},
        'HCESVM_Test2': {1: 0.8636, 2: 0.7692, 3: 0.6923},
    },
    'Balance': {
        'samples': {1: 58, 2: 10, 3: 57},
        'SVOR': {1: 0.8793, 2: 0.8000, 3: 0.8421},
        'NPSVOR': {1: 0.8793, 2: 0.8000, 3: 0.8421},
        'HCESVM_Test2': {1: 0.8793, 2: 0.8000, 3: 0.8421},
    },
    'Contraceptive': {
        'samples': {1: 126, 2: 67, 3: 102},
        'SVOR': {1: 0.3968, 2: 0.8806, 3: 0.0294},
        'NPSVOR': {1: 0.4048, 2: 0.9254, 3: 0.0196},
        'HCESVM_Test2': {1: 0.1032, 2: 0.9403, 3: 0.0000},
        'HCESVM_Class1first': {1: 0.1032, 2: 0.6269, 3: 0.5098},
    },
    'Hayes_Roth': {
        'samples': {1: 11, 2: 10, 3: 6},
        'SVOR': {1: 0.3636, 2: 0.9000, 3: 0.6667},
        'NPSVOR': {1: 0.3636, 2: 0.9000, 3: 0.6667},
        'HCESVM_Test2': {1: 0.3636, 2: 0.9000, 3: 0.8333},
    },
    'New_Thyroid': {
        'samples': {1: 30, 2: 7, 3: 6},  # 估算值
        'SVOR': {1: 1.0000, 2: 0.4286, 3: 0.5000},
        'NPSVOR': {1: 1.0000, 2: 0.7143, 3: 0.5000},
        'HCESVM_Test2': {1: 0.9667, 2: 1.0000, 3: 1.0000},
    },
    'TAE': {
        'samples': {1: 5, 2: 3, 3: 11},
        'SVOR': {1: 0.0000, 2: 1.0000, 3: 0.3636},
        'NPSVOR': {1: 0.4000, 2: 0.5000, 3: 0.3636},
        'HCESVM_Test2': {1: 0.0000, 2: 0.8000, 3: 0.6364},
    },
    'Thyroid': {
        'samples': {1: 3, 2: 8, 3: 133},
        'SVOR': {1: 0.0000, 2: 0.0000, 3: 1.0000},
        'NPSVOR': {1: 0.0000, 2: 0.0000, 3: 1.0000},
        'HCESVM_Test2': {1: 0.0000, 2: 0.0000, 3: 1.0000},
    },
    'Wine': {
        'samples': {1: 12, 2: 14, 3: 10},
        'SVOR': {1: 1.0000, 2: 0.9286, 3: 0.9000},
        'NPSVOR': {1: 1.0000, 2: 0.9286, 3: 1.0000},
        'HCESVM_Test2': {1: 1.0000, 2: 1.0000, 3: 0.9000},
    },
}

print("="*80)
print("MAE (Mean Absolute Error) 估算報告")
print("="*80)
print("\nMAE 公式: MAE = (1/N) * Σ|yi - yi_pred|")
print("\n注意: 由於沒有完整混淆矩陣，此為基於各類別準確率的估算值")
print("="*80)
print()

# 收集所有結果
mae_results = {}

for dataset_name, dataset_info in results.items():
    print(f"\n{'='*80}")
    print(f"資料集: {dataset_name}")
    print(f"{'='*80}")

    samples = dataset_info['samples']
    total_samples = sum(samples.values())
    print(f"測試樣本: Class 1: {samples[1]}, Class 2: {samples[2]}, Class 3: {samples[3]} (總計: {total_samples})")
    print()

    mae_results[dataset_name] = {}

    for model_name, accuracies in dataset_info.items():
        if model_name == 'samples':
            continue

        mae = estimate_mae_from_accuracy(accuracies, samples)
        total_acc = sum(accuracies[c] * samples[c] for c in [1,2,3]) / total_samples

        mae_results[dataset_name][model_name] = {
            'accuracy': total_acc,
            'mae': mae
        }

        print(f"{model_name:<20} - Accuracy: {total_acc:.4f}, 估算 MAE: {mae:.4f}")

    print()

# 總結表
print("\n" + "="*80)
print("總結表 - 所有資料集比較")
print("="*80)
print(f"\n{'資料集':<20} {'模型':<20} {'準確率':>10} {'估算 MAE':>12}")
print("-"*80)

for dataset_name, models in mae_results.items():
    for model_name, metrics in models.items():
        print(f"{dataset_name:<20} {model_name:<20} {metrics['accuracy']:>10.4f} {metrics['mae']:>12.4f}")

# 平均值
print("\n" + "="*80)
print("平均值（所有資料集）")
print("="*80)

model_names = set()
for dataset_results in mae_results.values():
    model_names.update(dataset_results.keys())

for model_name in sorted(model_names):
    accs = []
    maes = []
    for dataset_results in mae_results.values():
        if model_name in dataset_results:
            accs.append(dataset_results[model_name]['accuracy'])
            maes.append(dataset_results[model_name]['mae'])

    avg_acc = np.mean(accs)
    avg_mae = np.mean(maes)

    print(f"{model_name:<20} - 平均準確率: {avg_acc:.4f}, 平均估算 MAE: {avg_mae:.4f}")

# 最佳模型統計
print("\n" + "="*80)
print("最佳模型統計（按 MAE 最低）")
print("="*80)

for dataset_name, models in mae_results.items():
    best_model = min(models.items(), key=lambda x: x[1]['mae'])
    print(f"{dataset_name:<20} - 最佳: {best_model[0]:<20} (MAE: {best_model[1]['mae']:.4f})")

print("\n" + "="*80)
print("結論")
print("="*80)
print("\nMAE 越低越好（0 = 完美預測）")
print("- MAE < 0.2: 優秀")
print("- MAE 0.2-0.5: 良好")
print("- MAE 0.5-1.0: 中等")
print("- MAE > 1.0: 需改進")
