# HCESVM 專案指引

本文件保留目前仍有效的專案規則。更完整的策略說明請看 `README.md` 與
`docs/strategies/README.md`。

## 必守規則

1. 每次執行 Python、pytest、runner 或分析腳本前，都先啟用虛擬環境：
   ```bash
   source .venv/bin/activate
   ```
2. 所有實驗執行都要用時間戳區隔，並保留可追溯的 log、workbook 或報告。
3. 實驗輸出必須包含：
   - training/testing 每個 class 的 accuracy
   - training/testing total accuracy
   - 每個 classifier 的 `weights` 與 `b`
   - solver status / solve status label
4. 不要把臨時 log、監控腳本、臨時測試腳本或備份檔留在 repo 根目錄。
5. 不要 stage 或提交與目前任務無關的 generated outputs。

## 結果歸檔

日誌歸檔路徑固定為 `results/archive/`：

```text
results/archive/
└── YYYYMMDD_strategy_name/
    └── strategy_dataset_YYYYMMDD_HHMMSS.log
```

命名規範：

- 目錄使用日期前綴：`YYYYMMDD_`
- 策略名稱使用目前 runner 或實驗名稱，例如 `single_filter`, `multiple_filter`,
  `inverted`, `test3`, `test4`, `test3_teaching_data_1000`, `test3_no_time_limit_full`
- log 檔名包含策略、資料集與完整時間戳：`strategy_dataset_YYYYMMDD_HHMMSS.log`

歸檔工具：

```bash
source .venv/bin/activate
python scripts/archive_results.py
python scripts/archive_results.py --execute
```

臨時檔清理工具：

```bash
source .venv/bin/activate
python scripts/cleanup_temp_files.py
python scripts/cleanup_temp_files.py --execute
```

## 目前支援的 HCESVM 策略

`HierarchicalCESVM(strategy=...)` 目前接受 5 種策略：

| Strategy | Scope | Grouping | Objective |
| --- | --- | --- | --- |
| `single_filter` | 3-class only | H1: Class 3 vs {1,2}; H2: Class 2 vs 1 | standard |
| `multiple_filter` | 3-class only | H1: Class 1 vs {2,3}; H2: {1,2} vs 3 | standard |
| `inverted` | 3-class only | H1: medium vs {majority, minority}; H2: {medium, majority} vs minority | standard |
| `test3` | N-class | Hk: {1..k} vs {k+1..N} | `class_weight="balanced"`, `accuracy_mode="both"` |
| `test4` | N-class | Hk: {1..k} vs {k+1..N} | `objective_variant="test4"`, `class_weight="balanced"`, `accuracy_mode="both"` |

`class1_first` 與 `test2` 是歷史實驗名稱，不是目前
`HierarchicalCESVM` 的有效策略。若舊腳本或報告仍提到它們，應標成 historical，
不要作為新的建議入口。

## Runner 與 time limit 語意

HCESVM 的 `time_limit` 是 **per classifier / per hyperplane**，不是總預算。

- `BinaryCESVM.time_limit` 會直接傳給單一 Gurobi solve。
- `HierarchicalCESVM` 會為 H1, H2, ... 分別建立 `BinaryCESVM`。
- 4-class `test3` / `test4` 會訓練 3 個 classifiers；若 `time_limit=1800`，最壞情況約為
  `1800s * 3`，因為目前是串行訓練。
- teaching-data runners 不應把 user 指定的 time limit 拆分或覆寫成總預算。
- legacy `--total-time-limit` 不是目前的 runner 介面。

主要 runner：

```bash
source .venv/bin/activate

# 1000-sample HCESVM(test3)
python examples/run_teaching_data_hcesvm_1000.py --time-limit 1800

# Full teaching-data HCESVM(test3)
python examples/run_teaching_data_hcesvm_full.py --time-limit none

# Deadline-aware HCESVM(test3)
python examples/run_teaching_data_hcesvm_deadline.py --dataset skill --time-limit 1800

# Exact-split SVOR / NPSVOR baseline runner
python -m hcesvm.skill_1000_baselines_runner
```

Full runner 的重要資源參數：

```bash
python examples/run_teaching_data_hcesvm_full.py \
  --datasets cement_strength \
  --time-limit none \
  --threads 0 \
  --soft-mem-limit-gb 56 \
  --nodefile-start-gb 23.6 \
  --nodefile-dir auto
```

- `--threads`: 傳給 Gurobi `Threads`；`0` 表示使用所有可用 threads。
- `--soft-mem-limit-gb`: 傳給 Gurobi `SoftMemLimit`。
- `--nodefile-start-gb`: 傳給 Gurobi `NodeFileStart`，啟用 nodefile spill。
- `--nodefile-dir`: 傳給 Gurobi `NodeFileDir`；`auto` 會建立
  `$HOME/hcesvm-gurobi-nodefiles/<dataset>_<timestamp>`。
- `BinaryCESVM` 支援 solve 後釋放 Gurobi model/env resources；長時間 full runs
  應保留此行為，避免 classifiers 之間累積記憶體。

大型 Gurobi run 前先確認授權與資源：

```bash
source .venv/bin/activate
GRB_LICENSE_FILE=/home/bs10081/gurobi.lic python -c "import gurobipy as gp; print(gp.gurobi.version())"
```

## 測試與驗證

常用測試：

```bash
source .venv/bin/activate
pytest tests/test_hierarchical.py -q
pytest tests/test_teaching_data_runtime.py tests/test_runner_time_limit_semantics.py -q
pytest tests/test_teaching_data_hcesvm_full_runner.py -q
```

文件或 runner 語意變更後，至少跑：

```bash
source .venv/bin/activate
pytest tests/test_hierarchical.py tests/test_teaching_data_runtime.py tests/test_runner_time_limit_semantics.py tests/test_teaching_data_hcesvm_full_runner.py -q
git diff --check
```

## 專案維護

- 測試報告放在 `docs/reports/`。
- 策略文件放在 `docs/strategies/`。
- 根目錄保持精簡，不新增一次性 log、臨時 Python 腳本或備份檔。
- 若 worktree 已有使用者變更，不要 revert；只修改本任務需要的文件。
- 提交時 stage specific files，避免把不相關 reports/workbooks/logs 一起放入 commit。

## Git 指引

```bash
git status --short
git diff
git add CLAUDE.md AGENTS.md README.md docs/strategies/README.md .claude/...
git commit -S -m "docs: refresh agent and strategy guidance"
```

如果 1Password SSH signing 需要環境變數，通常使用：

```bash
SSH_AUTH_SOCK="$HOME/.1password/agent.sock" git commit -S -m "docs: refresh guidance"
```

**Last Updated**: 2026-05-25
