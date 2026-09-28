from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent
from src.rag_pipeline import create_rag_chain


@tool
def calculate_discount(price: float, discount_percent: float) -> float:
    """計算打折後的價格。輸入必須是純數字，例如 price=100, discount_percent=20。"""
    return price * (1 - discount_percent / 100)


def create_agent(vectorstore):
    # 使用支援 Tool Calling 的本地模型
    llm = ChatOllama(model="qwen2.5:7b", temperature=0)

    rag_chain, _ = create_rag_chain(vectorstore)

    @tool
    def knowledge_base_search(query: str) -> str:
        """【第一步必用工具】當使用者詢問產品價格、折扣規則、公司規定、產品規格或文件內容時使用。輸入應為具體的關鍵字，例如：'LARA 企業版 原價 折扣' 或 '金牌客戶 折扣規則'。"""
        return rag_chain.invoke(query)

    tools = [knowledge_base_search, calculate_discount]

    # 定義嚴格的 System Prompt，強制 Step-by-Step 思考
    system_prompt = """你是自動化助理。你有兩個工具：'knowledge_base_search' 和 'calculate_discount'。

    【範例】：
    使用者：我是金牌客戶，想買 LARA 企業版，請問最後要付多少錢？
    思考：我需要先查詢 LARA 企業版的原價和金牌客戶的折扣規則。
    行動：呼叫 knowledge_base_search(query='LARA 企業版 原價 金牌客戶 折扣')
    觀察：原價 120,000 元，金牌客戶 15% 折扣。
    行動：呼叫 calculate_discount(price=120000, discount_percent=15)
    觀察：102,000 元。
    回答：最終價格為 NT$ 102,000 元。

    【現在請處理使用者的問題】：
    """

    # 使用 LangGraph 的 create_react_agent，並注入 System Prompt
    agent_executor = create_react_agent(
        llm,
        tools,
        prompt=system_prompt  # 在 LangGraph 0.2.x 中直接傳入 prompt
    )

    return agent_executor