---
id: L0001
date: 2026-09-01
outcome: useful
tags: [ml, 驗證, 繼承程式碼, 證據等級]
anchors:
  - src/signalweave/validation.py
  - src/signalweave/recommender.py
  - docs/architecture.md
supersedes:
hits: 0
---

# 同一次生成的文件、命名與程式碼會互相背書，對帳要靠第三方

## 觸發情境

接手一份文件與程式碼在同一輪產生的專案（AI 一次生成、或人在同一個 session 裡邊寫邊記），而且要判斷它的宣稱是否成立。

## 領悟

**三個互相符合的東西，如果來自同一次生成，只證明了生成當下的自我一致，沒有證明任何外部事實。**

這個 repo 的前身有一條召回路叫 trending。三份證據互相支持：

- 變數名叫 `recency`
- `docs/architecture.md` 寫「recency-weighted positive event popularity」
- 程式碼是 `recency[item_pos] += weight * (1.0 + item_pos / (len(items) * 8))`

讀起來完全自洽。但那行程式用的是**商品在陣列裡的索引**，整段從沒碰過任何一個 timestamp。實際量出來 trending 與 popularity 的相關係數是 **0.9921**【已確認：2026-08-31 本機 `np.corrcoef`】——第四條召回路是第一條的複製品。改成用事件時間做半衰期衰減後降到 **0.7947**【已確認：同上】。

同一個模式在這份程式碼裡出現了三次：

| 宣稱 | 實際 | 怎麼抓到的 |
|---|---|---|
| trending 是 recency-weighted | 用陣列索引，與時間無關 | 算 trending 與 popularity 的相關係數 |
| ranker 輸出是 calibrated probability | `class_weight="balanced"` 保證它不是；平均 0.4706 對真實正例率 0.2340 | 拿 5-fold held-out 預測值的平均去比正例率 |
| UI 的 accuracy policy 有對應指標 | 從沒被評估過，前端顯示的是 content baseline 的數字 | 比對 `evaluate()` 跑過的 policy 集合與 API 能服務的集合 |

三次都不是「程式碼寫錯」那麼單純。**是宣稱與計算從來沒有被放在一起檢驗過**——沒有任何機制逼它們對帳，因為它們出生在同一個句子裡。

## 為什麼會撞到

生成式流程會同時產出命名、文件與實作。三者一致是**生成過程的必然結果**，不是正確性的證據。而人類 review 的直覺剛好相反：命名、註解與程式碼互相吻合時，我們會降低警戒。

這是 `rules/evidence-grades.md` 第 1 條在繼承程式碼場景的具體形態——**推論被寫得像事實**，然後被三份文件引用。

## 下次怎麼做

1. **對「X 是用 Y 算的」這類宣稱，設計一個「如果不是 Y 就會不同」的統計量，然後跑它。** 不要讀程式碼確認——讀程式碼會被同一組命名說服。trending 那條的判準是「它應該與 popularity 不同」，一行 `corrcoef` 就抓到了。
2. **先問「這個宣稱如果是假的，哪個數字會不一樣」**，問不出來就代表這個宣稱目前不可驗證，該標 `未驗證` 而不是當成已知。
3. **把對帳寫成常駐檢查，不是一次性的 review。** 本專案的做法是 `validation.py`：每個會出現在 README 或 UI 的數字都有一項檢查在每次啟動時重算它。一次性 review 的結論會過期，檢查不會。
4. **特別懷疑「第 N 條路 / 第 N 個特徵」這種湊數的元件。** 它們最容易在生成時被寫成前一條的變體，而且因為貢獻小，指標上看不出來——trending 修好前後 balanced 的 NDCG 幾乎沒變【已確認：0.2131 → 0.2131】，靠指標是抓不到的。

## 失效條件

- 專案改為文件由程式碼自動生成（docstring → docs），第 1 點仍成立但第 3 點的實作方式要改。
- 若未來有工具能對「文件宣稱 ↔ 實作」做靜態對帳，第 1、2 點該被那個機制取代。
- 本則的三個具體案例已全部修復並被檢查守住；再次出現代表檢查被移除，那是更嚴重的問題。
