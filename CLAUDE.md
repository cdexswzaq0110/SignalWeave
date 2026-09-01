# CLAUDE.md — SignalWeave

> **描述：** 多目標推薦系統，每一個顯示出來的數字都能追回產生它的計算。
> **階段：** 雛型（可解釋性與 MLOps 骨架已完成，資料仍是模擬）
> **語言：** Python 3.11 · scikit-learn · FastAPI · 無框架前端

## 開發流程

沒有寫死的命令序列。能力按需載入（開發配置留在本機，不隨本 repo 散布）。

預設節奏：確認分支 → 爬最小實作階梯 → 做出最小可動的東西 → 跑起來看 → 留 Lesson。

ML 相關的改動走 `se-ml-lifecycle`：split 與 leakage 排在模型與調參之前，Gate 沒過不往下。

## 共享語言

見 [CONTEXT.md](CONTEXT.md)。詞彙有衝突時以那份為準。

## 這個專案的實際指令

| 用途 | 指令 |
|---|---|
| 安裝 | `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` |
| 訓練並註冊 | `python -m signalweave.train` |
| 服務 | `python -m signalweave`（<http://127.0.0.1:8010>） |
| 測試 | `python -m pytest -q` |
| 檢查套件 | `python -m signalweave.validation` |
| 列出模型版本 | `python -m signalweave.train --list` |
| 回滾 | `python -m signalweave.train --promote <VERSION>` |

沒有 lint 或型別檢查設定。**不要為了補齊這張表而加一個。**

## 這個專案特有的約束

**1. 服務端不 fit 模型。**
`api.py` 載入註冊過的 bundle。要改係數只有一條路：`python -m signalweave.train`，它會產生新版本。在請求路徑上訓練會讓「畫面顯示的指標」與「回答請求的模型」脫鉤，而那正是這個專案存在要解決的問題。理據見 [ADR-0001](docs/adr/ADR-0001-serving-loads-a-registered-model.md)。

**2. 動到指標就要重新產生 artifacts。**
`artifacts/evaluation.json` 是 canonical build 的紀錄。改了會影響指標的東西卻沒重跑 `train`，`artifact_matches_runtime` 會失敗，CI 會掛。這是刻意的。

**3. 檢查判定有三值，不要把 `known` 改成 `pass`。**
`known` 表示主張不成立、而且原因是刻意的權衡。把它改綠等於把已知的限制藏起來。要新增 `known` 就同時寫清楚代價。理據見 [ADR-0002](docs/adr/ADR-0002-three-valued-check-verdicts.md)。

**4. 不寫沒跑過的數字。**
README、docs 與 UI 上每一個數值都必須來自本機實際執行。這個 repo 的前身就是被這件事害到的——見 [L0001](docs/lessons/0001-generated-docs-endorse-generated-code.md)。要寫數字就先跑一次，或標 `未驗證`。

**5. `web/` 三個檔案，不要引入建置步驟。**
沒有框架、沒有打包、沒有 CDN。前端刻意保持能直接讀懂的狀態。
