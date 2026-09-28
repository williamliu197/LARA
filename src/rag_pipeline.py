from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser


def create_rag_chain(vectorstore):
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    # 使用本地 Qwen 模型，temperature 設低一點以減少幻覺
    llm = ChatOllama(model="qwen2.5:7b", temperature=0.1)

    # 本地模型需要更明確的邊界設定
    prompt = ChatPromptTemplate.from_template("""
    你是一個專業的 AI 助手與邏輯分析師。請嚴格根據以下提供的【上下文】回答使用者的【問題】。
    
    規則：
    1. 只能使用上下文中的資訊回答。
    2. 【重要】如果問題涉及數字區間（例如年資、年齡、金額），請先判斷該數字是否落在上下文提到的區間內（例如：4年落在滿3年未滿5年的區間）。
    3. 如果上下文中真的沒有答案或無法推導，請明確回答：「根據現有知識庫，我無法回答此問題。」切勿捏造。
    
    【上下文】:
    {context}
    
    【問題】: {question}
    
    請先在心裡簡短分析條件，然後給出最終【回答】:
    """)

    def format_docs(docs):
        return "\n\n---\n\n".join([f"[來源: {doc.metadata.get('source')}]\n{doc.page_content}" for doc in docs])

    rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
    )
    return rag_chain, retriever