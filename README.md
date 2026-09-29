## 安裝 Ollama 並拉取模型
```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull nomic-embed-text
ollama pull qwen2.5:7b
```

## 安裝 Python 依賴
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 啟動
```bash
streamlit run src/app.py
```

## 測試案例與預期答案
1. 純 RAG 檢索測試
* 輸入問題：「入職滿四年的員工，有幾天特休假？」
* 預期答案：「根據知識庫規定，入職滿三年但未滿五年的員工，每年享有 14 天特休假。因此，入職滿四年的員工有 14 天特休假。」
* 🔍 Debug 觀察重點：
    * 成功：Retrieval 成功召回包含「特休假規定」的 Chunk，且模型精準提取了「滿三年未滿五年：14 天」的條件。
    * 失敗排查：如果模型回答 10 天或 16 天，請檢查檢索到的 Chunk。這通常代表 chunk_size 太小，導致「條件（滿三年未滿五年）」與「結論（14天）」被切分到了不同的 Chunk 中，模型只看到了其中一半。

2. Agent 工具呼叫與邏輯推理測試
* 輸入問題：「我是金牌客戶，想買 LARA 企業版，今天是 9月26日，請問最後要付多少錢？」
* 預期答案：「最後需要支付 NT$ 96,000 元。」
* 🔍 Debug 觀察重點 (觀察 Agent 的 Execution Trace)：
    1. Agent 應先呼叫 knowledge_base_search 查詢企業版原價（得到 120,000 元）。
    2. Agent 應查詢折扣政策，得知金牌客戶有 15% 折扣，且 9月26日有額外 5% 促銷。
    3. Agent 應理解「不疊加，擇優計算」的規則，得出最大折扣為 20% (15% + 5%)。
    4. Agent 應呼叫 calculate_discount 工具，且參數必須是純數字：{"price": 120000, "discount_percent": 20}。
    5. 失敗排查：本地模型 (如 Qwen2.5 7B) 有時會錯誤地將參數生成為字串（如 "120,000" 或 "120000元"），導致工具呼叫失敗。如果發生這種情況，這正是我們需要在 Tool 的 docstring 中強調「輸入必須是純數字」的原因。

3. 幻覺邊界測試 (Hallucination Test)
* 輸入問題：「LARA Tech 有提供員工無薪假嗎？」
* 預期答案：「根據現有知識庫，我無法回答此問題。」（或：「知識庫中沒有關於員工無薪假的相關規定。」）
* 🔍 Debug 觀察重點：
    * 成功：模型嚴格遵守了 Prompt 中的邊界限制，承認知識庫中沒有此資訊。
    * 失敗排查：如果模型開始「瞎掰」（例如：「有的，員工可以依規定申請無薪假」），請立即開啟 Debug Mode。你會發現，Retrieval 很可能召回了「特休假」或「遠距工作」等不相關但主題相近的 Chunk。這完美印證了你的核心理念：問題出在 Retrieval 召回了干擾資訊，而不是 Prompt 寫得不夠好。解法應是優化檢索策略（如加入 Reranker），而非一味地對模型說「請不要瞎掰」。