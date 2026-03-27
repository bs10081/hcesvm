#!/usr/bin/env python3
"""
計算 SVOR, NPSVOR, HCESVM 的 MAE (Mean Absolute Error)
並生成比較報告

MAE = (1/N) * Σ|yi - yi_pred|

對於序數迴歸問題，MAE 反映預測與實際類別的平均誤差
"""

import sys
import os
from pathlib import Path
import numpy as np
from datetime import datetime
from typing import Tuple, List, Dict

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.hcesvm.utils.data_loader import load_split_data
from src.hcesvm.models.svor import SVORImplicitQP
from src.hcesvm.models.npsvor import NPSVORQP
from src.hcesvm.models.hierarchical import HierarchicalCESVM


def calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """計算 Mean Absolute Error"""
    return np.mean(np.abs(y_true - y_pred))


def get_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 3):
    """計算混淆矩陣"""
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for true, pred in zip(y_true, y_pred):
        cm[true-1, pred-1] += 1
    return cm


def calculate_mae_from_confusion_matrix(cm: np.ndarray) -> float:
    """從混淆矩陣計算 MAE"""
    n_classes = cm.shape[0]
    total_error = 0
    total_samples = 0

    for i in range(n_classes):
        for j in range(n_classes):
            error = abs(i - j)
            count = cm[i, j]
            total_error += error * count
            total_samples += count

    return total_error / total_samples if total_samples > 0 else 0.0


# Dataset configurations - 只測試部分資料集以節省時間
DATASETS = {
    'Wine': {
        'path': '~/Developer/NSVORA/Archive/wine_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'Contraceptive': {
        'path': '~/Developer/NSVORA/Archive/contraceptive_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
    'New_Thyroid': {
        'path': '~/Developer/NSVORA/Archive/new-thyroid_split.xlsx',
        'train_skiprows': 5,
        'test_skiprows': 4,
    },
}


def convert_to_ordinal_format(X_classes: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
    """Convert multi-class data to ordinal format"""
    X_combined = []
    y_combined = []

    for label, X in enumerate(X_classes, start=1):
        X_combined.append(X)
        y_combined.extend([label] * len(X))

    return np.vstack(X_combined), np.array(y_combined)


def main():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/mae_comparison_{timestamp}.log"

    results = {}

    print("="*80)
    print("MAE Comparison: SVOR vs NPSVOR vs HCESVM")
    print("="*80)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    with open(output_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MAE (Mean Absolute Error) Comparison\n")
        f.write("="*80 + "\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Formula: MAE = (1/N) * Σ|yi - yi_pred|\n")
        f.write("="*80 + "\n\n")

        for dataset_name, config in DATASETS.items():
            print(f"\n{'='*80}")
            print(f"Dataset: {dataset_name}")
            print(f"{'='*80}")

            f.write(f"\n{'='*80}\n")
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"{'='*80}\n")

            try:
                # Load data
                path = os.path.expanduser(config['path'])
                data = load_split_data(
                    path,
                    train_skiprows=config['train_skiprows'],
                    test_skiprows=config['test_skiprows']
                )
                X_train_classes = data['train']
                X_test_classes = data['test']

                X_train, y_train = convert_to_ordinal_format(X_train_classes)
                X_test, y_test = convert_to_ordinal_format(X_test_classes)

                print(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}")
                f.write(f"Training samples: {len(y_train)}, Testing samples: {len(y_test)}\n\n")

                # SVOR
                print("\nTraining SVOR...")
                svor = SVORImplicitQP()
                svor.fit(X_train, y_train)
                y_pred_svor = svor.predict(X_test)
                mae_svor = calculate_mae(y_test, y_pred_svor)
                cm_svor = get_confusion_matrix(y_test, y_pred_svor)
                acc_svor = np.mean(y_test == y_pred_svor)

                print(f"SVOR - Accuracy: {acc_svor:.4f}, MAE: {mae_svor:.4f}")
                f.write(f"SVOR:\n")
                f.write(f"  Accuracy: {acc_svor:.4f}\n")
                f.write(f"  MAE: {mae_svor:.4f}\n")
                f.write(f"  Confusion Matrix:\n")
                for i, row in enumerate(cm_svor, 1):
                    f.write(f"    True {i}: {row}\n")
                f.write("\n")

                # NPSVOR
                print("Training NPSVOR...")
                npsvor = NPSVORQP()
                npsvor.fit(X_train, y_train)
                y_pred_npsvor = npsvor.predict(X_test)
                mae_npsvor = calculate_mae(y_test, y_pred_npsvor)
                cm_npsvor = get_confusion_matrix(y_test, y_pred_npsvor)
                acc_npsvor = np.mean(y_test == y_pred_npsvor)

                print(f"NPSVOR - Accuracy: {acc_npsvor:.4f}, MAE: {mae_npsvor:.4f}")
                f.write(f"NPSVOR:\n")
                f.write(f"  Accuracy: {acc_npsvor:.4f}\n")
                f.write(f"  MAE: {mae_npsvor:.4f}\n")
                f.write(f"  Confusion Matrix:\n")
                for i, row in enumerate(cm_npsvor, 1):
                    f.write(f"    True {i}: {row}\n")
                f.write("\n")

                # HCESVM Test2
                print("Training HCESVM Test2...")
                hcesvm = HierarchicalCESVM(
                    cesvm_params={'C_hyper': 1.0, 'M': 1000.0, 'time_limit': 300},
                    strategy='test2'
                )
                hcesvm.fit(*X_train_classes)
                X_test_combined = np.vstack(X_test_classes)
                y_pred_hcesvm = hcesvm.predict(X_test_combined)
                mae_hcesvm = calculate_mae(y_test, y_pred_hcesvm)
                cm_hcesvm = get_confusion_matrix(y_test, y_pred_hcesvm)
                acc_hcesvm = np.mean(y_test == y_pred_hcesvm)

                print(f"HCESVM Test2 - Accuracy: {acc_hcesvm:.4f}, MAE: {mae_hcesvm:.4f}")
                f.write(f"HCESVM Test2:\n")
                f.write(f"  Accuracy: {acc_hcesvm:.4f}\n")
                f.write(f"  MAE: {mae_hcesvm:.4f}\n")
                f.write(f"  Confusion Matrix:\n")
                for i, row in enumerate(cm_hcesvm, 1):
                    f.write(f"    True {i}: {row}\n")
                f.write("\n")

                # Store results
                results[dataset_name] = {
                    'SVOR': {'acc': acc_svor, 'mae': mae_svor},
                    'NPSVOR': {'acc': acc_npsvor, 'mae': mae_npsvor},
                    'HCESVM': {'acc': acc_hcesvm, 'mae': mae_hcesvm}
                }

                print(f"✓ {dataset_name} completed")

            except Exception as e:
                print(f"✗ {dataset_name} failed: {e}")
                f.write(f"ERROR: {e}\n\n")

        # Summary table
        print("\n" + "="*80)
        print("SUMMARY TABLE")
        print("="*80)
        f.write("\n" + "="*80 + "\n")
        f.write("SUMMARY TABLE\n")
        f.write("="*80 + "\n\n")

        # Header
        header = f"{'Dataset':<20} {'Model':<12} {'Accuracy':>10} {'MAE':>10}"
        print(header)
        print("-"*80)
        f.write(header + "\n")
        f.write("-"*80 + "\n")

        for dataset_name, dataset_results in results.items():
            for model_name, metrics in dataset_results.items():
                line = f"{dataset_name:<20} {model_name:<12} {metrics['acc']:>10.4f} {metrics['mae']:>10.4f}"
                print(line)
                f.write(line + "\n")

        print("="*80)
        f.write("="*80 + "\n")

        # Average
        print("\nAVERAGE METRICS:")
        f.write("\nAVERAGE METRICS:\n")

        for model_name in ['SVOR', 'NPSVOR', 'HCESVM']:
            avg_acc = np.mean([r[model_name]['acc'] for r in results.values()])
            avg_mae = np.mean([r[model_name]['mae'] for r in results.values()])
            line = f"{model_name:<12} Avg Accuracy: {avg_acc:.4f}, Avg MAE: {avg_mae:.4f}"
            print(line)
            f.write(line + "\n")

    print(f"\n✓ Results saved to: {output_file}")
    print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")


if __name__ == "__main__":
    main()
