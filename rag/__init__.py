"""
rag/__init__.py
───────────────
Public API for the RAG pipeline package.
"""

from rag.embedder import get_embedder, reset_embedder_cache
from rag.ingestion import ingest_file, is_ocr_result
from rag.vector_store import VectorStoreManager

__all__ = [
    "VectorStoreManager",
    "get_embedder",
    "reset_embedder_cache",
    "ingest_file",
    "is_ocr_result",
]
