# SignalWeave 共享語言

> 這份檔是這個專案的**共享記憶體區段**。人與 Agent 讀同一份定義。
> **不要一次寫完。** 詞彙在命名衝突、解釋超過兩句、歧義被戳破時才長出來。

## Language

**Slate**：
一次請求回傳的、已排定順序的推薦組合。它是被當成一個整體最佳化的單位——多樣性、新穎度與創作者上限都定義在 slate 這一層，不在單一項目上。
_避免_：推薦清單、結果列表、feed

**Candidate**：
通過召回、進入排序階段的項目。目錄裡的東西叫 item，被某條召回路提出來的才叫 candidate。
_避免_：候選清單、item（在排序脈絡下）

**Ranker score**：
邏輯迴歸對單一 candidate 的輸出（欄位 `relevance`）。它**只用來排序**，不是機率——`class_weight="balanced"` 讓它的平均值遠高於真實正例率。
_避免_：機率、信心分數、engagement probability

**Slate utility**：
決定一個 candidate 排在第幾位的最終加權值（欄位 `score`）＝ relevance、diversity、novelty、freshness 四項各自乘上 policy 權重的總和。
_避免_：分數、score（單獨使用時）

**Policy**：
slate 最佳化的權重組合與創作者上限（accuracy／balanced／discovery）。**它不是模型**——同一組係數可以套任何一個 policy。
_避免_：模式、模型、策略模型

**Champion／Shadow**：
Champion 是預設服務的 policy；Shadow 是在每次 champion 請求背後照跑、但**永不回傳**的挑戰者。
_避免_：A/B、主要政策、備用政策

**Bundle／Version**：
一次訓練產生的、可載入的成品（scaler、係數、fit matrix、當時的 frozen-window 報告），由 `<UTC 時戳>-<模型摘要前 8 碼>` 命名。「模型」這個詞在對話裡太滑，指定某一個時用 version。
_避免_：模型檔、artifact（artifact 在本專案專指 `artifacts/` 裡的 JSON 證據）

**Check verdict**：
三值判定。`pass` 主張成立；`fail` 有東西壞了；`known` 主張不成立、而且原因是刻意的權衡。
_避免_：警告、warning、yellow

## Relationships

- 一個 **Slate** 由多個 **Candidate** 依 **Slate utility** 排序而成
- 一個 **Candidate** 帶一個 **Ranker score**，但可能由多條召回路同時提出
- 一個 **Version** 可被任何 **Policy** 使用；換 policy 不換 version
- **Champion** 與 **Shadow** 都是 **Policy**，差別只在誰的結果會回傳
- 一次 **訓練** 產生一個 **Version**；只有 `train` 會改係數

## Flagged ambiguities

- 「分數」曾同時指 ranker 的 sigmoid 輸出與最終排序值 —— 已解決：前者是 **Ranker score**（`relevance`），後者是 **Slate utility**（`score`）。API 欄位名刻意不一致於口語，以欄位為準。
- 「artifact」曾同時指模型二進位與 `artifacts/` 下的 JSON —— 已解決：模型叫 **Bundle**，`artifacts/` 專指證據 JSON。
- 「trending」曾被寫成「recency-weighted」但實作用的是目錄索引 —— 已解決：現在指**事件時間的半衰期衰減**，並由 `retrieval.sources_are_not_redundant` 守住。

## Out of scope

- **CTR、轉換率、留存**：資料是模擬的，這些詞會誘導出因果宣稱。
- **Embedding、向量檢索**：目前是 TF-IDF；等真的需要再定義，先定義會讓討論漂到還沒做的東西上。
- **使用者分群 / segment**：評估目前只在全體平均，沒有分群結論可講。
