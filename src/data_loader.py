import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


# 改動：chunk_size 500 -> 800、overlap 50 -> 150，
# 避免「特別促銷活動」與「不與客戶等級折扣疊加」被切到不同 chunk
def load_and_chunk_data(file_path: str, chunk_size: int = 800, chunk_overlap: int = 150):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()

    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)
        doc.metadata["doc_type"] = "local_kb"

    # 針對中文優化切分符號
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", " ", ""]
    )
    return text_splitter.split_documents(documents)