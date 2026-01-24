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
    
    print("=" * 60)
    print("Binary CE-SVM on Parkinsons Dataset (Train/Test Split)")
    print("=" * 60)
    
    # Load training data
    print(f"\n載入訓練資料: {data_file}")
    X_train, y_train, n_features = load_parkinsons_data(str(data_file), sheet_name='train')
    print(f"訓練集: {len(X_train)} 樣本, {n_features} 特徵")
    print(f"  正類 (+1): {(y_train == 1).sum()}")
    print(f"  負類 (-1): {(y_train == -1).sum()}")
    
    # Load test data
    print(f"\n載入測試資料: {data_file}")
    X_test, y_test, _ = load_parkinsons_data(str(data_file), sheet_name='工作表1', skiprows=0)
    print(f"測試集: {len(X_test)} 樣本, {n_features} 特徵")
    print(f"  正類 (+1): {(y_test == 1).sum()}")
    print(f"  負類 (-1): {(y_test == -1).sum()}")
    
    # Get parameters
    params = get_default_params()
    params['verbose'] = True
    params['mip_gap'] = 0.05
    params['time_limit'] = 1800  # 30 minutes for better convergence
    
    print("\n" + "=" * 60)
    print("模型參數")
    print("=" * 60)
    for key, value in params.items():
        print(f"  {key}: {value}")
    
    # Train model
    print("\n" + "=" * 60)
    print("訓練 Binary CE-SVM")
    print("=" * 60)
    
    model = BinaryCESVM(**params)
    model.fit(X_train, y_train)
    
    # Get solution summary
    summary = model.get_solution_summary()
    print("\n" + "=" * 60)
    print("模型摘要")
    print("=" * 60)
    print(f"Status: {summary['status']}")
    print(f"Objective Value: {summary['objective_value']:.6f}")
    print(f"L1 Norm: {summary['l1_norm']:.6f}")
    print(f"Selected Features: {summary['n_selected_features']}/{n_features}")
    print(f"Intercept: {summary['intercept']:.6f}")
    
    # Evaluate on training set
    print("\n" + "=" * 60)
    print("訓練集評估")
    print("=" * 60)
    
    y_train_pred = model.predict(X_train)
    train_metrics = calculate_binary_metrics(y_train, y_train_pred)
    
    print(f"Overall Accuracy: {train_metrics['accuracy']:.4f}")
    print(f"TPR (Sensitivity): {train_metrics['TPR']:.4f}")
    print(f"TNR (Specificity): {train_metrics['TNR']:.4f}")
    
    # Evaluate on test set
    print("\n" + "=" * 60)
    print("測試集評估（泛化性能）")
    print("=" * 60)
    
    y_test_pred = model.predict(X_test)
    test_metrics = calculate_binary_metrics(y_test, y_test_pred)
    
    print(f"Overall Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"TPR (Sensitivity): {test_metrics['TPR']:.4f}")
    print(f"TNR (Specificity): {test_metrics['TNR']:.4f}")
    
    print("\n" + "=" * 60)
    print("測試完成！")
    print("=" * 60)

if __name__ == "__main__":
    main()
