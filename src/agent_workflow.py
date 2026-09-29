from typing import List

from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent


@tool
def calculate_discount(price: float, discount_percent: float) -> float:
    """計算單一折扣後的價格。輸入必須是純數字，例如 price=100, discount_percent=20。"""
    return price * (1 - discount_percent / 100)


@tool
def calculate_best_price(price: float, discount_percents: List[float]) -> str:
    """當同時有多個折扣（例如客戶等級折扣與促銷折扣）且「不可疊加、擇優計算」時使用。
    輸入原價與所有候選折扣百分比，工具會自動取最大折扣（不會相加）並算出最終價格。
    例如 price=120000, discount_percents=[15, 5]。"""
    best = max(discount_percents)
    final_price = price * (1 - best / 100)
    return f"採用最大折扣 {best}%（不疊加），最終價格為 NT$ {final_price:,.0f}"


def create_agent(vectorstore):
    # 使用支援 Tool Calling 的本地模型
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)

    # 改動 1：工具直接回傳「原始 chunk」，不再經過內層 LLM，避免二次轉述造成錯誤
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    @tool
    def knowledge_base_search(query: str) -> str:
        """【第一步必用工具】當使用者詢問產品價格、折扣規則、促銷活動、公司規定、產品規格或文件內容時使用。
        輸入應為具體的關鍵字，例如：'LARA 企業版 原價' 或 '客戶等級折扣 促銷活動 疊加'。
        回傳的是知識庫的原始文件片段，請自行閱讀並判斷。"""
        docs = retriever.invoke(query)
        return "\n\n---\n\n".join(d.page_content for d in docs)

    tools = [knowledge_base_search, calculate_discount, calculate_best_price]

    # 改動 2、3：加入折扣規則與擇優範例
    system_prompt = """你是自動化助理。你有三個工具：
'knowledge_base_search'（查文件原文）、'calculate_discount'（單一折扣計算）、'calculate_best_price'（多個折扣擇優計算）。

【規則】
1. 涉及價格、折扣、促銷時，必須先呼叫 knowledge_base_search 查詢，不可憑記憶回答。
2. 工具回傳的是文件原文，請自己仔細閱讀，特別注意「不與...疊加」、「擇優」、「日期區間」等條件。
3. 折扣預設「不可相加」。若文件寫明不疊加、擇優，就只取最大的那一個折扣。
4. 促銷活動要先確認使用者提到的日期是否落在活動期間內，不在期間內就不適用。
5. 最終金額必須呼叫計算工具得到，不可自己心算。
6. 若有多個候選折扣，使用 calculate_best_price；只有單一折扣才用 calculate_discount。
7. 文件中找不到答案時，回答「根據現有知識庫，我無法回答此問題。」

【範例 1】：
使用者：我是金牌客戶，想買 LARA 企業版，請問最後要付多少錢？（今天不在任何促銷期間）
思考：需要查企業版原價與金牌客戶折扣。
行動：呼叫 knowledge_base_search(query='LARA 企業版 原價 金牌客戶 折扣')
觀察：原價 120,000 元，金牌客戶 15% 折扣。
行動：呼叫 calculate_discount(price=120000, discount_percent=15)
觀察：102000
回答：最終價格為 NT$ 102,000 元。

【範例 2】：
使用者：我是金牌客戶，今天是促銷期間內，買 LARA 企業版要付多少？
思考：要查原價、等級折扣、促銷折扣，以及兩者是否可疊加。
行動：呼叫 knowledge_base_search(query='LARA 企業版 原價 客戶等級折扣 促銷 疊加')
觀察：原價 120,000；金牌 15%；促銷額外 5%，但不與客戶等級折扣疊加，系統自動擇優。
思考：不可疊加，所以在 15% 和 5% 之間取最大者。
行動：呼叫 calculate_best_price(price=120000, discount_percents=[15, 5])
觀察：採用最大折扣 15%（不疊加），最終價格為 NT$ 102,000
回答：因為促銷不與等級折扣疊加，採擇優計算，最終價格為 NT$ 102,000 元。

【現在請處理使用者的問題】：
"""

    # 使用 LangGraph 的 create_react_agent，並注入 System Prompt
    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=system_prompt
    )

    return agent_executor