"""
rag/vector_store.py
───────────────────
ChromaDB-backed vector store manager for the RAG pipeline.

Responsibilities:
  - Initialise a persistent ChromaDB client at data/chroma_db/
  - Add documents (ingest → chunk → embed → store)
  - Similarity search with source metadata
  - Bootstrap sample reports on first run
  - Expose collection statistics
"""

from __future__ import annotations

import logging
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Collection name — single collection for all ingested documents
_COLLECTION_NAME = "market_research_docs"


class VectorStoreManager:
    """
    Manages the ChromaDB vector store for the RAG pipeline.

    Usage
    -----
        vsm = VectorStoreManager()
        vsm.add_documents("my_report.pdf", text)
        results = vsm.similarity_search("What is the market size?", k=5)
        stats = vsm.get_collection_stats()
    """

    def __init__(self) -> None:
        from utils.config import config  # deferred

        self._chroma_dir = config.chroma_dir
        self._chroma_dir.mkdir(parents=True, exist_ok=True)

        self._embedder = None  # lazy-loaded
        self._vectorstore: Any = None  # lazy-loaded
        self._client: Any = None

        logger.info("VectorStoreManager initialised (persist_dir=%s)", self._chroma_dir)

    # ── Private helpers ───────────────────────────────────────────────────────

    def _get_embedder(self) -> Any:
        if self._embedder is None:
            from rag.embedder import get_embedder  # noqa: PLC0415
            self._embedder = get_embedder()
        return self._embedder

    def _get_vectorstore(self) -> Any:
        """Lazy-load the Chroma vectorstore wrapper."""
        if self._vectorstore is not None:
            return self._vectorstore

        embedder = self._get_embedder()

        try:
            from langchain_chroma import Chroma  # noqa: PLC0415
        except ImportError:
            try:
                from langchain_community.vectorstores import Chroma  # type: ignore[no-redef]  # noqa: PLC0415
            except ImportError:
                raise ImportError(
                    "chromadb and langchain-chroma are required. "
                    "Run: pip install chromadb langchain-chroma"
                )

        self._vectorstore = Chroma(
            collection_name=_COLLECTION_NAME,
            embedding_function=embedder,
            persist_directory=str(self._chroma_dir),
        )
        logger.debug("Chroma vectorstore loaded from %s", self._chroma_dir)
        return self._vectorstore

    # ── Public API ────────────────────────────────────────────────────────────

    def add_documents(self, source_name: str, text: str) -> int:
        """
        Ingest text into the vector store.

        Chunks the text, embeds each chunk, and stores them in ChromaDB
        with the source_name as metadata.

        Parameters
        ----------
        source_name : str
            Human-readable label for the document (e.g. filename).
        text : str
            Full extracted text to index.

        Returns
        -------
        int
            Number of chunks added to the store.
        """
        from rag.chunker import chunk_text, chunks_to_langchain_docs  # noqa: PLC0415

        if not text or not text.strip():
            logger.warning("add_documents: empty text for '%s' — nothing added", source_name)
            return 0

        chunks = chunk_text(text, source_name=source_name)
        if not chunks:
            logger.warning("add_documents: no chunks produced for '%s'", source_name)
            return 0

        docs = chunks_to_langchain_docs(chunks)
        vs = self._get_vectorstore()

        # Add with unique IDs to avoid duplicate key conflicts on re-ingestion
        ids = [f"{source_name}_{i}_{uuid.uuid4().hex[:8]}" for i in range(len(docs))]
        vs.add_documents(docs, ids=ids)

        logger.info("Added %d chunks from '%s' to vector store", len(docs), source_name)
        return len(docs)

    def similarity_search(
        self,
        query: str,
        k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Retrieve the top-k most relevant document chunks for a query.

        Parameters
        ----------
        query : str
            The user's question or search string.
        k : int
            Number of results to return (default 5).

        Returns
        -------
        list[dict]
            Each dict has keys: ``text``, ``source``, ``score``, ``chunk_index``.
        """
        if not query.strip():
            return []

        vs = self._get_vectorstore()
        try:
            results = vs.similarity_search_with_relevance_scores(query, k=k)
        except Exception as exc:  # noqa: BLE001
            logger.warning("similarity_search failed (%s) — falling back to basic search", exc)
            try:
                basic = vs.similarity_search(query, k=k)
                return [
                    {
                        "text": doc.page_content,
                        "source": doc.metadata.get("source", "unknown"),
                        "score": None,
                        "chunk_index": doc.metadata.get("chunk_index", 0),
                    }
                    for doc in basic
                ]
            except Exception as exc2:  # noqa: BLE001
                logger.error("similarity_search completely failed: %s", exc2)
                return []

        return [
            {
                "text": doc.page_content,
                "source": doc.metadata.get("source", "unknown"),
                "score": round(float(score), 4) if score is not None else None,
                "chunk_index": doc.metadata.get("chunk_index", 0),
            }
            for doc, score in results
        ]

    def get_collection_stats(self) -> dict[str, Any]:
        """
        Return basic statistics about the current collection.

        Returns
        -------
        dict
            Keys: ``total_chunks``, ``sources`` (list of unique source names).
        """
        try:
            vs = self._get_vectorstore()
            # Access the underlying chromadb collection for efficient counting
            collection = vs._collection  # type: ignore[attr-defined]
            count = collection.count()
            # Get all sources from metadata
            all_meta = collection.get(include=["metadatas"])["metadatas"]
            sources = sorted(set(m.get("source", "") for m in all_meta if m))
            return {
                "total_chunks": count,
                "sources": sources,
            }
        except Exception as exc:  # noqa: BLE001
            logger.warning("get_collection_stats failed: %s", exc)
            return {"total_chunks": 0, "sources": []}

    def bootstrap_sample_reports(self) -> bool:
        """
        Ingest the bundled sample report TXT files if the collection is empty.

        Returns True if bootstrapping was performed, False if already populated.
        """
        stats = self.get_collection_stats()
        if stats["total_chunks"] > 0:
            logger.info(
                "Vector store already populated (%d chunks from %d sources) — skipping bootstrap",
                stats["total_chunks"],
                len(stats["sources"]),
            )
            return False

        from utils.data_loader import load_sample_report_texts  # noqa: PLC0415

        reports = load_sample_report_texts()
        total = 0
        for report in reports:
            added = self.add_documents(report["filename"], report["text"])
            total += added
            logger.info("Bootstrapped '%s': %d chunks", report["filename"], added)

        logger.info("Bootstrap complete — %d total chunks from %d sample reports", total, len(reports))
        return True

    def document_exists(self, source_name: str) -> bool:
        """Check whether a document with the given source_name is already indexed."""
        stats = self.get_collection_stats()
        return source_name in stats.get("sources", [])
