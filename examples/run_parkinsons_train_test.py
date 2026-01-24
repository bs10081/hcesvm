#!/usr/bin/env python3
"""Test Binary CE-SVM on Parkinsons dataset with train/test split."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from hcesvm import BinaryCESVM, get_default_params
from hcesvm.utils import load_parkinsons_data, calculate_binary_metrics

def main():
    data_file = Path(__file__).parent.parent / "data/parkinsons/Parkinsons_CESVM.xlsx"

    print("=" * 70)
    print("Binary CE-SVM on Parkinsons Dataset (Train/Test Split)")
    print("=" * 70)

    # Load training data
    print(f"\n載入訓練資料...")
    X_train, y_train, n_features = load_parkinsons_data(
        str(data_file),
        sheet_name='train',
        skiprows=4
    )

    # Load test data
    print(f"\n載入測試資料...")
    X_test, y_test, _ = load_parkinsons_data(
        str(data_file),
        sheet_name='工作表1',
        skiprows=0
    )

    # Get parameters
    params = get_default_params()
    params['verbose'] = True
    params['mip_gap'] = 0.05      # 5% gap tolerance
    params['time_limit'] = 1800   # 30 minutes

    print("\n" + "=" * 70)
    print("模型參數")
    print("=" * 70)
    for key, value in params.items():
        print(f"  {key}: {value}")

    # Train model
    print("\n" + "=" * 70)
    print("訓練 Binary CE-SVM (Time Limit: 30 分鐘)")
    print("=" * 70)

    model = BinaryCESVM(**params)
    model.fit(X_train, y_train)

    # Get solution summary
    summary = model.get_solution_summary()
    print("\n" + "=" * 70)
    print("模型摘要")
    print("=" * 70)
    print(f"Status: {summary['status']}")
    print(f"Objective Value: {summary['objective_value']:.6f}")
    print(f"L1 Norm: {summary['l1_norm']:.6f}")
    print(f"Selected Features: {summary['n_selected_features']}/{n_features}")
    print(f"Selected Feature Indices: {summary['selected_feature_indices']}")
    print(f"Positive Class Accuracy LB: {summary['positive_class_accuracy_lb']:.4f}")
    print(f"Negative Class Accuracy LB: {summary['negative_class_accuracy_lb']:.4f}")
    print(f"Intercept: {summary['intercept']:.6f}")

    # Evaluate on training set
    print("\n" + "=" * 70)
    print("訓練集評估 (Training Accuracy)")
    print("=" * 70)

    y_train_pred = model.predict(X_train)
    train_metrics = calculate_binary_metrics(y_train, y_train_pred)

    print(f"Overall Accuracy: {train_metrics['accuracy']:.4f} ({train_metrics['accuracy']*100:.2f}%)")
    print(f"TPR (Sensitivity): {train_metrics['TPR']:.4f} ({train_metrics['TPR']*100:.2f}%)")
    print(f"TNR (Specificity): {train_metrics['TNR']:.4f} ({train_metrics['TNR']*100:.2f}%)")

    # Evaluate on test set
    print("\n" + "=" * 70)
    print("測試集評估 (Testing Accuracy - 泛化性能)")
    print("=" * 70)

    y_test_pred = model.predict(X_test)
    test_metrics = calculate_binary_metrics(y_test, y_test_pred)

    print(f"Overall Accuracy: {test_metrics['accuracy']:.4f} ({test_metrics['accuracy']*100:.2f}%)")
    print(f"TPR (Sensitivity): {test_metrics['TPR']:.4f} ({test_metrics['TPR']*100:.2f}%)")
    print(f"TNR (Specificity): {test_metrics['TNR']:.4f} ({test_metrics['TNR']*100:.2f}%)")

    # Performance comparison
    print("\n" + "=" * 70)
    print("Train vs Test 性能比較")
    print("=" * 70)
    accuracy_diff = train_metrics['accuracy'] - test_metrics['accuracy']
    tpr_diff = train_metrics['TPR'] - test_metrics['TPR']
    tnr_diff = train_metrics['TNR'] - test_metrics['TNR']

    print(f"Accuracy 差異: {accuracy_diff:+.4f} ({accuracy_diff*100:+.2f}%)")
    print(f"TPR 差異: {tpr_diff:+.4f} ({tpr_diff*100:+.2f}%)")
    print(f"TNR 差異: {tnr_diff:+.4f} ({tnr_diff*100:+.2f}%)")

    if abs(accuracy_diff) < 0.05:
        print("\n泛化性能: ✅ 良好（差異 < 5%）")
    elif abs(accuracy_diff) < 0.10:
        print("\n泛化性能: ⚠️ 中等（差異 5-10%）")
    else:
        print("\n泛化性能: ❌ 過擬合（差異 > 10%）")

    print("\n" + "=" * 70)
    print("測試完成！")
    print("=" * 70)

if __name__ == "__main__":
    main()
