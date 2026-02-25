1. 記得所有執行結果都要記錄下來，並且用時間戳作為區隔
2. 執行結果需要包含 training 和 testing 每個 class 的 accuracy 和 total accuracy，以及 weight 和 b
3. 每次執行的時候要使用 .venv 啟用虛擬環境

## 專案維護指南

### 結果歸檔規則

**日誌歸檔路徑**: `results/archive/`

所有測試日誌應按日期和策略分類歸檔：

```
results/archive/
├── YYYYMMDD_strategy_name/
│   ├── strategy_dataset_YYYYMMDD_HHMMSS.log
│   └── ...
```

**命名規範**:
- 使用日期前綴：`YYYYMMDD_`
- 策略名稱：`inverted`, `test2`, `test3`, `class1_first` 等
- 完整日誌名：`strategy_dataset_YYYYMMDD_HHMMSS.log`

**自動歸檔**: 使用 `scripts/archive_results.py` 或 `/archive-results` skill

### 臨時檔案清理

**不要留在根目錄的檔案**:
- `*.log` (臨時日誌)
- `monitor_*.sh` (監控腳本)
- `test_*.py` (臨時測試腳本)
- `generate_*.py` (臨時生成腳本)
- `*.backup`, `*.bak` (備份檔案)

**清理工具**: 使用 `scripts/cleanup_temp_files.py` 或 `/project-cleanup` skill

### 報告文檔管理

**測試報告位置**: `docs/reports/`
- 所有測試報告從根目錄移至此處
- 保持根目錄簡潔（只保留 README.md, CLAUDE.md, FINAL_TEST_RESULTS.md, CLASS1_FIRST_SUMMARY.md）

**策略文檔位置**: `docs/strategies/`
- 策略總覽：`docs/strategies/README.md`
- 各策略詳細文檔：`docs/strategies/{strategy_name}.md`

## 分類策略完整說明

HCESVM 支援 **6 種分類策略**，分為固定策略和動態策略兩大類：

### 固定策略 (Fixed Strategies)

預先定義類別分組，不隨樣本分布改變。

#### 1. single_filter (原始策略)
- **H1**: Class 3 (+1) vs {1,2} (-1)
- **H2**: Class 2 (+1) vs Class 1 (-1)
- **目標函數**: 標準
- **適用**: 基準比較

#### 2. multiple_filter (標準策略)
- **H1**: Class 1 (+1) vs {2,3} (-1)
- **H2**: {1,2} (+1) vs Class 3 (-1)
- **目標函數**: 標準
- **適用**: 一般平衡資料

#### 3. class1_first (Class 1 優先策略)
- **H1**: Class 1 (+1) vs {2,3} (-1)
- **H2**: Class 2 (+1) vs Class 3 (-1)
- **預測邏輯**:
  - H1 = +1 → Class 1
  - H1 = -1 → 進入 H2
    - H2 = +1 → Class 2
    - H2 = -1 → Class 3
- **目標函數**: 標準
- **適用**: 優先識別 Class 1，固定行為

### 動態策略 (Dynamic Strategies)

根據訓練資料的少數/多數類分布自動調整類別分組。

#### 4. inverted (動態分組策略)
- **類別檢測**: 識別 majority, medium, minority 類
- **H1**: Medium (+1) vs {Majority, Minority} (-1)
- **H2**: {Medium, Majority} (+1) vs Minority (-1)
- **目標函數**: 標準
- **適用**: 動態適應樣本分布

#### 5. test2 (激進不平衡策略)
- **類別檢測**: 識別 minority/majority 類
- **類別分組**: 根據 minority 位置動態調整
  - minority = 1: H1 為 1 vs {2,3}, H2 為 2 vs 3
  - minority = 2: H1 為 2 vs {1,3}, H2 為 1 vs 3
  - minority = 3: H1 為 3 vs {1,2}, H2 為 1 vs 2
- **目標函數修改 (Test2 Rule)**:
  - 當 majority 為 Class 2 時，移除 majority 類別的準確率項
  - 結果: `min ... - l_minority` (僅保留 minority 準確率項)
- **適用**: 嚴重不平衡 (>1:3)，少數類檢測關鍵

#### 6. test3 (平衡不平衡策略)
- **類別檢測**: 同 test2
- **類別分組**: 同 test2
- **目標函數**: 樣本加權
  ```
  min Σⱼ(w⁺ⱼ + w⁻ⱼ) + C·Σᵢ(αᵢ + βᵢ + ρᵢ) - (1/s⁺)·l⁺ - (1/s⁻)·l⁻
  ```
  - `s⁺` = 正類樣本數
  - `s⁻` = 負類樣本數
- **實作**: `class_weight="balanced"`
- **適用**: 中度不平衡 (1:2-1:3)，平衡性能

### 策略比較表

| 策略 | 類型 | H1 分組 | H2 分組 | 目標函數 | 適用場景 |
|------|------|---------|---------|----------|----------|
| single_filter | 固定 | Class 3 vs {1,2} | Class 2 vs 1 | 標準 | 基準策略 |
| multiple_filter | 固定 | Class 1 vs {2,3} | {1,2} vs 3 | 標準 | 平衡資料 |
| class1_first | 固定 | Class 1 vs {2,3} | Class 2 vs 3 | 標準 | Class 1 優先 |
| inverted | 動態 | Medium vs {Maj,Min} | {Med,Maj} vs Min | 標準 | 動態適應 |
| test2 | 動態 | 依 minority 位置 | 依 minority 位置 | 移除準確率項 | 嚴重不平衡 |
| test3 | 動態 | 依 minority 位置 | 依 minority 位置 | 樣本加權 | 中度不平衡 |

### 策略選擇建議

```
資料集有顯著類別不平衡？
├─ 否 → 使用固定策略
│   ├─ 需要 Class 1 優先？→ class1_first
│   ├─ 標準方法？→ multiple_filter
│   └─ 基準比較？→ single_filter
│
└─ 是 → 使用動態策略
    ├─ 激進少數類聚焦？→ test2
    ├─ 平衡加權？→ test3
    └─ 標準適應？→ inverted
```

## Test3 策略詳細說明

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

## Token 優化指南

### 核心原則
**200k context is not infinite** - 啟用太多工具時可能縮減到 70k

### 策略
1. **遠程執行** - 長時間運行的測試使用 tmux：
   ```bash
   tmux new -s hcesvm-test "python examples/run_test3_comparison.py"
   tmux attach -t hcesvm-test
   ```

2. **平行工作流** - 使用 git worktree 進行並行開發：
   ```bash
   git worktree add ../hcesvm-test3 -b feature/test3
   ```

3. **手動壓縮** - 必要時執行：
   ```bash
   /compact
   ```

### 避免上下文消耗
- ❌ 一次讀取大量日誌檔案
- ❌ 重複讀取相同檔案
- ✅ 使用 Grep 搜尋而非 Read 整個檔案
- ✅ 使用 `head`/`tail` 限制輸出

## Subagent 使用指南

### 可用 Agents

| Agent | 用途 | 何時使用 |
|-------|------|---------|
| **code-reviewer** | 代碼審查 | 寫完程式碼後 PROACTIVELY |
| **security-reviewer** | 安全審查 | 處理輸入/認證時 |
| **test-runner** | 測試執行 | 修改程式碼後 |

### 使用範例

```bash
# 代碼審查
/code-review src/hcesvm/models/hierarchical.py

# 安全審查
/security-review src/hcesvm/utils/data_loader.py

# 運行測試
/test-runner
```

### Agent 委託原則
- **Foreground** - 需要結果才能繼續時使用
- **Background** - 獨立工作時使用（尚未實作）
- **Isolation** - 使用 worktree 隔離環境（進階）

## Memory Persistence

### 跨會話狀態保存
使用 `.claude/memory/` 目錄（未實作）或：

1. **Git 歷史** - 提交訊息作為決策日誌
2. **CLAUDE.md** - 專案特定指令和規則
3. **結果日誌** - 保留測試結果供參考

### 檢查點策略
```bash
# 重要里程碑前建立 git tag
git tag -a v1.0-test3 -m "Test3 strategy validated"

# 使用分支追蹤實驗
git checkout -b experiment/test4-strategy
```

## Rules 系統

### 模組化規則
HCESVM 規則位於 `.claude/rules/`：

- **security.md** - 安全規則（無硬編碼密鑰、輸入驗證）
- **coding-style.md** - 編碼風格（不可變性、檔案大小）
- **testing.md** - 測試要求（TDD、80% 覆蓋率）
- **git-workflow.md** - Git 慣例（conventional commits）

### 規則優先級
1. **專案規則** (`.claude/rules/`) - 最高優先級
2. **全域規則** (`~/.claude/rules/`) - 次要優先級
3. **預設行為** - 最低優先級

## 命令速查

### 維護命令
```bash
/cleanup          # 清理臨時檔案
/archive          # 歸檔測試結果
/test-strategies  # 運行策略測試
```

### 測試命令
```bash
pytest tests/ -v                    # 所有測試
pytest tests/test_hierarchical.py   # 特定檔案
pytest --cov=src/hcesvm             # 含覆蓋率
```

### Git 命令
```bash
git status                          # 檢查狀態
git diff                            # 查看變更
git add src/hcesvm/models/          # 階段化變更
git commit -m "feat: add feature"   # 提交
```

## 最佳實踐摘要

### 開發工作流
1. **寫測試** (TDD) → 2. **寫程式碼** → 3. **代碼審查** → 4. **安全審查** → 5. **提交**

### 代碼品質
- ✅ 函數 < 50 行
- ✅ 檔案 < 800 行
- ✅ 覆蓋率 ≥ 80%
- ✅ 無硬編碼密鑰
- ✅ 類型提示
- ✅ Docstrings

### 提交前檢查
```bash
# 1. 運行測試
pytest tests/ -v

# 2. 檢查覆蓋率
pytest --cov=src/hcesvm

# 3. 格式化
black src/ tests/ examples/

# 4. Lint
flake8 src/ --max-line-length=88

# 5. 類型檢查
mypy src/hcesvm

# 6. 檢查安全性
grep -r "password" src/
```

---

**Last Updated**: 2026-02-25
**Version**: 2.0 (Enhanced with everything-claude-code principles)
