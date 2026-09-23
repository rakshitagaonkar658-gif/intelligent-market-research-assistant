"""
rag/embedder.py
───────────────
Embedding model factory for the RAG pipeline.

Priority order:
  1. WatsonxEmbeddings (ibm/slate-30m-english-rtrvr) — when live credentials present
  2. HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2) — demo / fallback

The factory is cached at module level so the model is only loaded once per process.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_embedder_instance: Any = None


def get_embedder(force_refresh: bool = False) -> Any:
    """
    Return the application embedding model (cached after first call).

    Returns a LangChain-compatible Embeddings object that implements:
        embed_documents(texts: list[str]) -> list[list[float]]
        embed_query(text: str) -> list[float]

    Parameters
    ----------
    force_refresh : bool
        Discard the cached instance and create a new one.
    """
    global _embedder_instance

    if _embedder_instance is not None and not force_refresh:
        return _embedder_instance

    from utils.config import config  # deferred import

    if not config.is_demo():
        embedder = _try_watsonx_embedder(config)
        if embedder is not None:
            _embedder_instance = embedder
            return _embedder_instance
        logger.warning(
            "WatsonxEmbeddings initialisation failed — falling back to "
            "sentence-transformers. Set DEMO_MODE=true to suppress this warning."
        )

    _embedder_instance = _get_huggingface_embedder()
    return _embedder_instance


def _try_watsonx_embedder(config: Any) -> Any | None:
    """Attempt to create a WatsonxEmbeddings instance. Returns None on failure."""
    try:
        from langchain_ibm import WatsonxEmbeddings  # noqa: PLC0415
    except ImportError:
        logger.debug("langchain-ibm not installed — skipping WatsonxEmbeddings")
        return None

    try:
        embedder = WatsonxEmbeddings(
            model_id="ibm/slate-30m-english-rtrvr",
            url=config.watsonx_url,
            apikey=config.watsonx_api_key,
            project_id=config.watsonx_project_id,
        )
        logger.info("WatsonxEmbeddings initialised (ibm/slate-30m-english-rtrvr)")
        return embedder
    except Exception as exc:  # noqa: BLE001
        logger.warning("WatsonxEmbeddings failed: %s", exc)
        return None


def _get_huggingface_embedder() -> Any:
    """Return a HuggingFaceEmbeddings instance using all-MiniLM-L6-v2."""
    try:
        from langchain_huggingface import HuggingFaceEmbeddings  # noqa: PLC0415
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings  # type: ignore[no-redef]  # noqa: PLC0415
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for demo-mode embeddings. "
                "Run: pip install sentence-transformers langchain-huggingface"
            )

    logger.info(
        "Using HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2) for embeddings"
    )
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def reset_embedder_cache() -> None:
    """Clear the cached embedder (used in tests)."""
    global _embedder_instance
    _embedder_instance = None
