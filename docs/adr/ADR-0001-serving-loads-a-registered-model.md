# ADR-0001：服務端載入註冊過的模型，不在請求路徑上訓練

- **狀態**：已接受
- **日期**：2026-09-01
- **決策者**：TerryHC（本輪由 Claude Code 提案並實作）

## Context

原本 `api.py` 在 module import 時就 `SignalWeave()`，也就是每次啟動都現場 fit 一個 ranker、現場跑一次 frozen-window 評估。

這在單機、資料只有 2,176 筆事件的情況下**跑得動**，所以它不是效能問題。它是歸屬問題：

畫面上顯示的 NDCG、AUC、guardrail 結果，來自一次「剛好執行同一份程式碼」的計算。它們與正在回答請求的那組係數之間，沒有任何東西保證是同一個。在這個規模上兩者確實相同；但「因為規模小所以碰巧相同」不是一個可以寫進 README 的性質，而這個專案的全部主張就是**每個數字都能追回產生它的計算**。

拉扯的兩股力量：
- 這個專案刻意避免基礎設施表演（README 明寫「不做 infrastructure theater」）。模型註冊表聽起來就很像表演。
- 但「服務端不訓練」是 MLOps 裡少數在任何規模都成立的性質，而且它是可解釋性主張的前提。

## 選項

| 選項 | 好處 | 代價 | 可逆性 |
|---|---|---|---|
| **A（採用）** 檔案式註冊表：train 產生 bundle + manifest，服務端載入 | 指標可歸屬於特定版本；可回滾；模型二進位與程式碼版本可對帳；啟動 4s → 0.24s | 多兩個模組；cold checkout 要 bootstrap；bundle 與程式碼可能不相容，要處理 | 中——移除要同時改 api、tests、UI |
| B 維持啟動時 fit | 零新增概念；一個指令就跑得起來 | 指標與服務模型只是「碰巧相同」；無法回滾；無法回答「線上跑的是哪一版」 | 高 |
| C 直接上 MLflow／模型伺服器 | 業界標準，功能完整 | 為 80 個項目引入一整套服務；違反最小實作階梯 | 低——依賴難拔 |
| D 什麼都不做 | — | 上述歸屬問題留著，而專案的核心主張就站不住 | — |

「什麼都不做」在這裡真的被評估過：如果這個專案不宣稱可追溯，D 是對的。但它宣稱了。

## 決定

我們選 A。取得「服務端載入具名、可查核的成品」這個性質，只需要一個目錄加一個 JSON 指標；A 之上的所有東西都是包裝。

`SignalWeave(bundle=...)` 接收已註冊的 scaler、係數、fit matrix 與**當時的 frozen-window 報告**。API 走這條路徑，永不呼叫 `_fit_ranker`。

## 後果

- **好的**：指標讀自 bundle，所以它們必然屬於服務中的模型（由 `serving.metrics_belong_to_the_served_model` 守住）。回滾是 `--promote <VERSION>`。manifest 記下 data／code／model 三個摘要，二進位被換掉會被 `binary_matches_its_manifest` 抓到。啟動時間 4s → 0.24s【已確認：2026-08-31 本機同一 process 內計時】。
- **壞的**：cold checkout 沒有模型可載，所以第一次啟動會 bootstrap 訓練一個——這讓「服務端不訓練」多了一個例外，寫在 `load_or_bootstrap` 的 docstring 裡。另外多了 bundle 與程式碼不相容的狀態要處理（`IncompatibleBundle`）。
- **中性但要知道的**：`artifacts/models/*/model.joblib` 進 `.gitignore`，manifest 不進。這代表 clone 下來的 repo 有完整的模型履歷但沒有模型，要 `train` 一次。這是刻意的——能重建的東西不該進版控。

## 重評觸發

- 當模型大到 `joblib` 載入超過 2 秒，或 bundle 超過 50 MB —— 該換成外部儲存 + 指標，而不是繼續塞進 repo 結構。
- 當出現第二個服務實例 —— 檔案式 champion 指標會有競態，該換成有原子性的儲存。
- 當需要同時服務多個版本（真 A/B，不是 shadow）—— 目前的單一 champion 模型不夠。

## 證據

| 主張 | 證據等級 | 來源 |
|---|---|---|
| 載入比 fit 快一個數量級（0.24s vs ~4s） | 已確認 | 2026-08-31 本機，同一 process 內 `time.time()` 前後量測 |
| 載入後產生的 slate 與原 engine 完全相同 | 已確認 | `test_registry_round_trip_serves_an_identical_slate` |
| 不相容的特徵契約會被拒絕而非默默載入 | 已確認 | `test_registry_refuses_a_bundle_with_a_different_feature_contract` |
| 回滾可用 | 已確認 | 2026-08-31 本機，seed 7 版本升為 champion 後 `--promote` 回 seed 42 版本 |
| 這個設計在更大規模仍成立 | 推論 | 未在多實例或大模型下驗證；見「重評觸發」 |
