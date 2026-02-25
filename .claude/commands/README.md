# HCESVM Claude Code Commands

本目錄包含 HCESVM 專案的 Claude Code 自訂命令定義。

## 可用命令

### `/cleanup` - 專案清理
清理根目錄的臨時檔案。

**功能**:
- 掃描臨時日誌 (`*.log`)
- 識別監控腳本 (`monitor_*.sh`)
- 檢測臨時測試腳本 (`test_*.py`, `generate_*.py`)
- 顯示詳細報告並執行清理

**使用**:
```
/cleanup          # 預覽（dry-run 模式）
```

然後手動執行:
```bash
python scripts/cleanup_temp_files.py --execute
```

---

### `/archive` - 結果歸檔
將測試結果日誌歸檔到按日期分類的子目錄。

**功能**:
- 自動識別日誌的日期和策略
- 建立歸檔目錄結構 (`results/archive/YYYYMMDD_strategy/`)
- 智能檢測比較測試（test2 vs test3）
- 生成歸檔摘要

**使用**:
```
/archive          # 預覽歸檔計畫
```

然後手動執行:
```bash
python scripts/archive_results.py --execute
```

---

### `/test-strategies` - 策略測試
運行所有 6 種分類策略的測試套件。

**功能**:
- 測試固定策略（single_filter, multiple_filter, class1_first）
- 測試動態策略（inverted, test2, test3）
- 驗證預測邏輯
- 檢查邊界情況

**使用**:
```
/test-strategies  # 運行完整測試套件
```

---

## 命令架構

每個命令定義為一個 JSON 檔案：

```json
{
  "name": "command-name",
  "description": "簡短描述",
  "instructions": "詳細說明和使用範例",
  "handler": {
    "type": "command",
    "command": "要執行的腳本"
  }
}
```

## 添加新命令

1. 在 `.claude/commands/` 建立新的 JSON 檔案
2. 遵循 Claude Code settings JSON schema
3. 重啟 Claude Code 或重新載入設定

## 相關腳本

所有命令對應的腳本位於 `scripts/` 目錄：
- `cleanup_temp_files.py` - 專案清理
- `archive_results.py` - 結果歸檔

## 參考

- [Claude Code 文檔](https://docs.anthropic.com/claude/claude-code)
- [everything-claude-code](https://github.com/affaan-m/everything-claude-code) - 最佳實踐參考
