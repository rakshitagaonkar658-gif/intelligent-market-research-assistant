"""
rag/chunker.py
──────────────
Text chunking for the RAG pipeline.

Uses LangChain's RecursiveCharacterTextSplitter to split documents into
overlapping chunks that fit within the embedding model's context window.

Default parameters (tuned for enterprise document RAG):
  - chunk_size   = 512  tokens (~400 words) — good balance of context vs precision
  - chunk_overlap = 64  characters — preserves sentence continuity across boundaries
  - separators   = paragraph → newline → sentence → word boundaries
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    source_name: str = "",
) -> list[dict[str, Any]]:
    """
    Split ``text`` into overlapping chunks and return them as a list of dicts.

    Each dict contains:
        - ``text``   : the chunk string
        - ``source`` : the originating document name
        - ``index``  : zero-based chunk position within the document

    Parameters
    ----------
    text : str
        The full document text to split.
    chunk_size : int
        Approximate number of characters per chunk (default 512).
    chunk_overlap : int
        Number of characters to overlap between adjacent chunks (default 64).
    source_name : str
        Label for the originating document (used in retrieval citations).

    Returns
    -------
    list[dict]
        List of chunk dicts ready to be embedded and stored.
    """
    if not text or not text.strip():
        logger.warning("chunk_text received empty text for source '%s'", source_name)
        return []

    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter  # noqa: PLC0415
    except ImportError:
        try:
            from langchain.text_splitter import RecursiveCharacterTextSplitter  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "LangChain text splitter not found. "
                "Run: pip install langchain-text-splitters"
            )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""],
    )

    raw_chunks: list[str] = splitter.split_text(text)

    chunks = [
        {
            "text": chunk.strip(),
            "source": source_name,
            "index": idx,
        }
        for idx, chunk in enumerate(raw_chunks)
        if chunk.strip()
    ]

    logger.info(
        "Chunked '%s': %d chars → %d chunks (size=%d, overlap=%d)",
        source_name or "unnamed",
        len(text),
        len(chunks),
        chunk_size,
        chunk_overlap,
    )
    return chunks


def chunks_to_langchain_docs(chunks: list[dict[str, Any]]) -> list[Any]:
    """
    Convert chunk dicts to LangChain Document objects for use with Chroma.

    Parameters
    ----------
    chunks : list[dict]
        Output from ``chunk_text()``.

    Returns
    -------
    list[langchain_core.documents.Document]
    """
    try:
        from langchain_core.documents import Document  # noqa: PLC0415
    except ImportError:
        from langchain.schema import Document  # type: ignore[no-redef]  # noqa: PLC0415

    return [
        Document(
            page_content=chunk["text"],
            metadata={
                "source": chunk.get("source", ""),
                "chunk_index": chunk.get("index", 0),
            },
        )
        for chunk in chunks
    ]
