from langchain_core.tools import tool
from langchain_ollama import ChatOllama
from langgraph.prebuilt import create_react_agent  # 現代化且穩定的 Agent 創建方式
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
        """當使用者詢問公司規定、產品規格或文件內容時，使用此工具查詢知識庫。輸入應為關鍵字或完整問題。"""
        return rag_chain.invoke(query)

    tools = [knowledge_base_search, calculate_discount]

    # 使用 LangGraph 的 create_react_agent
    # 它自動處理了 ReAct 循環、錯誤恢復和最大迭代次數限制，比舊版 AgentExecutor 穩定得多
    agent_executor = create_react_agent(llm, tools)

    return agent_executor