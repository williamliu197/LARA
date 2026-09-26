import os
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_and_chunk_data(file_path: str, chunk_size: int = 500, chunk_overlap: int = 50):
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