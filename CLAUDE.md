1. 記得所有執行結果都要記錄下來，並且用時間戳作為區隔
2. 執行結果需要包含 training 和 testing 每個 class 的 accuracy 和 total accuracy，以及 weight 和 b
3. 每次執行的時候要使用 .venv 啟用虛擬環境

## Test3 策略說明

**新增日期**: 2026-02-10

### 策略概述
Test3 策略使用樣本數量加權的目標函數，透過樣本數量的倒數作為準確率項的加權係數。

### 與 Test2 的差異

| 策略 | 目標函數處理方式 | 特點 |
|------|----------------|------|
| **Test2** | 根據 majority class 位置，動態移除 `-l_p` 或 `-l_n` | 當 majority 為 class 2 時，移除 majority 類別的準確率項 |
| **Test3** | 使用 `-(1/s_p)·l_p - (1/s_n)·l_n` | 使用樣本數量倒數加權，少數類權重自動提高 |

### 目標函數

```
min  Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - (1/s⁺)·l⁺ - (1/s⁻)·l⁻
```

其中：
- `s⁺` = 正類 (+1) 的樣本數量
- `s⁻` = 負類 (-1) 的樣本數量

### 參數設定

在 `BinaryCESVM` 中新增 `class_weight` 參數：
- `class_weight="none"` (預設): 等權重，標準目標函數
- `class_weight="balanced"`: 使用樣本數量倒數加權

在 `HierarchicalCESVM` 中：
- `strategy="test3"`: 自動使用 `class_weight="balanced"` 和 `accuracy_mode="both"`

### 使用範例

```python
from hcesvm import HierarchicalCESVM

# Test3 策略
model = HierarchicalCESVM(
    cesvm_params={
        'C_hyper': 1.0,
        'M': 1000.0,
        'time_limit': 1800
    },
    strategy='test3'
)

model.fit(X1_train, X2_train, X3_train)
predictions = model.predict(X_test)
```

### 比較測試腳本

使用 `examples/run_test3_comparison.py` 比較 Test2 vs Test3：

```bash
source .venv/bin/activate
python examples/run_test3_comparison.py
```
