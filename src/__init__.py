"""
LARA (Local AI Retrieval Agent) 核心模組初始化
此檔案用於封裝 src 目錄下的核心功能，簡化外部匯入路徑。
"""

# 匯入各模組的核心函數
from .data_loader import load_and_chunk_data
from .vector_store import build_vector_store
from .rag_pipeline import create_rag_chain
from .agent_workflow import create_agent

# 定義公開 API (Public API)
# 這能確保當別人使用 `from src import *` 時，只會匯入這些核心函數
# 展現良好的封裝與模組化設計
__all__ = [
    "load_and_chunk_data",
    "build_vector_store",
    "create_rag_chain",
    "create_agent",
]

# 可選：定義套件版本號
__version__ = "1.0.0"