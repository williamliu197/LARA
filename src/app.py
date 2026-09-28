import sys
import os

# 動態獲取專案根目錄 (LARA/) 並加入 sys.path
# __file__ 是當前檔案 (app.py) 的絕對路徑，os.path.dirname 往上兩層就是專案根目錄
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.insert(0, project_root)

# --- 以下才是原本的 import ---
import streamlit as st
import os
from src.vector_store import build_vector_store
from src.rag_pipeline import create_rag_chain
from src.agent_workflow import create_agent

st.set_page_config(page_title="Local AI Retrieval Agent", layout="wide")
st.markdown(
    "<h1 style='text-align: center;'>Local AI Retrieval Agent</h1>",
    unsafe_allow_html=True
)

with st.sidebar:
    st.header("⚙️ 系統設定")
    debug_mode = st.toggle("開啟除錯模式 (Debug Mode)", value=True)
    mode = st.radio("選擇模式", ("純 RAG 問答", "AI Agent 自動化流程"))
    st.info("💡 本地模型推理需要時間，請耐心等待。開啟除錯模式可查看檢索細節。")


@st.cache_resource
def init_system():
    os.makedirs("data", exist_ok=True)
    sample_file = "data/sample.txt"
    if not os.path.exists(sample_file):
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(
                "本公司於 2024 年推出新一代 AI 平台。\n核心功能包含 RAG 知識庫建置與 Agent 自動化工作流。\n若遇到回答不準確，應從資料品質、Chunk 切分、Retrieval 策略逐層排查，而非僅修改 Prompt。\n產品 A 的原價是 1000 元。")
    return build_vector_store(sample_file)


vectorstore = init_system()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("請輸入您的問題..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("本地模型思考中..."):
            if mode == "純 RAG 問答":
                rag_chain, retriever = create_rag_chain(vectorstore)

                if debug_mode:
                    st.markdown("### 🔍 除錯日誌 (Retrieval)")
                    docs = retriever.invoke(prompt)
                    for i, doc in enumerate(docs):
                        st.markdown(
                            f"**Chunk {i + 1}** (來源: {doc.metadata.get('source')})\n> {doc.page_content[:150]}...")
                    st.markdown("---")

                # 使用 stream 提升本地模型體驗
                response = ""
                response_placeholder = st.empty()
                for chunk in rag_chain.stream(prompt):
                    response += chunk
                    response_placeholder.markdown(response + "▌")
                response_placeholder.markdown(response)

            else:

                # Agent 模式 (LangGraph)

                agent_executor = create_agent(vectorstore)

                if debug_mode:
                    st.markdown("### 🤖 Agent 思考與工具呼叫日誌 (LangGraph)")

                # LangGraph 的標準輸入格式

                input_dict = {"messages": [{"role": "user", "content": prompt}]}

                # 執行 Agent

                result = agent_executor.invoke(input_dict)

                # 提取最後一個訊息作為最終回答

                response = result["messages"][-1].content

                if debug_mode:

                    st.markdown("#### 執行軌跡 (Execution Trace):")

                    # LangGraph 會返回所有中間訊息，我們可以過濾出 Tool 呼叫

                    for msg in result["messages"]:

                        if hasattr(msg, 'tool_calls') and msg.tool_calls:

                            for tc in msg.tool_calls:
                                st.markdown(f"- 🔧 **呼叫工具**: `{tc['name']}`")

                                st.markdown(f"  - **參數**: `{tc['args']}`")

                        elif hasattr(msg, 'tool_call_id'):

                            st.markdown(f"- 👁️ **觀察結果**: {str(msg.content)[:200]}...")

                    st.markdown("---")

                st.markdown(response)

        st.session_state.messages.append({"role": "assistant", "content": response})