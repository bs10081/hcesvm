# Inverted 策略 12 Datasets 測試完整報告

**測試日期**: 2026-01-31  
**最終狀態**: ✅ **12/12 全部通過**

---

## 測試結果摘要

| Dataset | Train Acc | Test Acc | 狀態 |
|---------|-----------|----------|------|
| Abalone | 0.9198 | 0.9199 | ✅ |
| Car_Evaluation | 0.8553 | 0.8468 | ✅ |
| Wine_Quality | 0.8249 | 0.8250 | ✅ |
| Balance | 0.8860 | 0.8400 | ✅ |
| Contraceptive | 0.7742 | N/A | ✅ |
| Hayes_Roth | 1.0000 | N/A | ✅ |
| New_Thyroid | 0.9884 | 0.9767 | ✅ |
| Squash_Stored | 0.9756 | N/A | ✅ |
| Squash_Unstored | 1.0000 | N/A | ✅ |
| TAE | 0.6667 | N/A | ✅ |
| Thyroid | 0.9792 | N/A | ✅ |
| Wine | 1.0000 | N/A | ✅ |

**統計**:
- 平均 Training Accuracy: 0.8975
- 平均 Test Accuracy: 0.8835 (7 個有測試資料的 datasets)

---

## Bug 修復記錄

### Bug 1: 資料載入失敗 ✅ 已修復
- **檔案**: `src/hcesvm/utils/data_loader.py` (Line 111-116)
- **問題**: Pandas 新版使用 `str` dtype，原本 `dtype == 'object'` 失效
- **解決**: 改用 `dtype.kind == 'O'` 檢測所有字串類型

### Bug 2: 無法即時監控進度 ✅ 已修復
- **檔案**: `examples/run_inverted_all_datasets.py` (Line 13-26, 84)
- **解決**: 實現 TeeOutput class 同時輸出到 console 和日誌

---

## 詳細報告位置

- **完整總結**: `results/INVERTED_COMPLETE_SUMMARY.md`
- **第一輪測試**: `results/inverted_all_datasets_summary_20260131_134843.md`
- **失敗重測**: `results/inverted_failed_datasets_summary_20260131_174148.md`
- **個別日誌**: `results/inverted_*_*.log`

---

**報告生成時間**: 2026-01-31 17:43
