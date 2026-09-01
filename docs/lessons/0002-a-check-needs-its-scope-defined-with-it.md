---
id: L0002
date: 2026-09-01
outcome: useful
tags: [gate, 驗證, ml, 確定性]
anchors:
  - src/signalweave/validation.py
  - src/signalweave/train.py
supersedes:
hits: 0
---

# 檢查的適用範圍要跟檢查一起定義，否則它會擋住正當的變體

## 觸發情境

把一條「倉庫內容要與程式碼產出一致」的檢查，接到一個同時也管實驗的 Gate 上。

## 領悟

**一條對 canonical build 正確的檢查，套到正當的變體上會變成假的紅燈。**

本專案的 `artifact_matches_runtime` 比對 `artifacts/evaluation.json` 與現場重算的結果，用來擋「改了會動指標的東西卻沒重新產生證據」。這條是對的。

然後訓練管線加上「任何 `fail` 就拒絕註冊」的 Gate。結果跑 `train --seed 7` 做 seed sweep 時被擋下【已確認：2026-08-31 本機，`refusing to register: 1 failing check(s): artifact_matches_runtime`】——因為 `artifacts/` 記錄的**本來就是** seed 42，seed 7 的結果跟它不同是理所當然的。

Gate 沒有壞，它忠實執行了。**壞的是那條檢查沒有把「我只對 seed 42 有意義」寫進自己的定義裡**，於是這個知識留在寫檢查的人腦裡，而 Gate 讀不到腦。

修法是把範圍寫進檢查本身：非 canonical seed 時回傳 `known` 並說明理由，同時 `train()` 也不再覆寫 `artifacts/`。兩邊一起改——只改 Gate 會讓實驗污染 canonical 紀錄，只改檢查會讓實驗覆寫證據。

## 為什麼會撞到

檢查是為了「守住倉庫一致性」寫的，Gate 是為了「守住模型品質」寫的。兩者都對，但它們的**適用母體不同**：倉庫一致性只對預設參數成立，模型品質對所有參數成立。把前者接到後者上，等於宣稱所有訓練跑都應該產生 canonical 結果。

這個錯誤在寫的當下看不出來，因為第一次跑一定是 seed 42。**是自己的預設值把 bug 藏起來了。**

## 下次怎麼做

1. **每寫一條會比對「committed 紀錄」的檢查，就問一次：哪些正當的執行不會符合這份紀錄？** 答案不是「沒有」的話，範圍就要寫進檢查。
2. **檢查分兩類，接 Gate 前先分好**：對「這次產出」成立的（leakage、範圍、可重現性）可以無條件擋；對「倉庫狀態」成立的（artifact 是否最新、文件是否同步）只在預設路徑上擋。
3. **用非預設參數跑一次新 Gate。** 第一次跑用預設值等於沒測 Gate 的邊界。「沒看過紅燈就不算裝好」還有另一面：**只看過該紅的紅燈，沒確認過不該紅的地方不會紅**。
4. **拒絕註冊的訊息要能自我解釋。** 本例的訊息直接印出失敗的 check id，所以三十秒就定位到問題。如果只印「checks failed」，會先去懷疑 seed 7 的資料。

## 失效條件

- 若未來 `artifacts/` 改成每個 seed 各存一份，這條檢查就能無條件成立，第 1、2 點在本專案失效（但通則仍成立）。
- 若 Gate 改成分級（blocking / advisory），第 2 點該被那個機制取代。
- 專案若不再有「canonical build」的概念，整則失效。
