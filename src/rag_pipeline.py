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
    你是一個專業的 AI 助手。請嚴格根據以下提供的【上下文】回答使用者的【問題】。
    規則：
    1. 只能使用上下文中的資訊回答。
    2. 如果上下文中沒有答案，請明確回答：「根據現有知識庫，我無法回答此問題。」
    3. 不要捏造事實。

    【上下文】:
    {context}

    【問題】: {question}

    【回答】:
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