import os
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from src.data_loader import load_and_chunk_data


def build_vector_store(file_path: str, persist_directory: str = "./chroma_db_ollama"):
    # 使用本地 Ollama Embedding 模型
    embeddings = OllamaEmbeddings(model="nomic-embed-text")

    if os.path.exists(persist_directory):
        print("✅ 載入現有的 Vector Store...")
        vectorstore = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    else:
        print("🔄 正在建置新的 Vector Store (使用本地模型向量化，可能需要幾秒鐘)...")
        chunks = load_and_chunk_data(file_path)
        vectorstore = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=persist_directory
        )
        print("✅ Vector Store 建置完成。")

    return vectorstore