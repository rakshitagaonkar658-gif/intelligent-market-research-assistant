"""
tools/rag_tool.py
─────────────────
RAG document retrieval tool.

Retrieves the top-k most relevant chunks from the ChromaDB vector store,
builds a context-augmented prompt, asks the LLM to answer, and returns
the answer with cited source excerpts.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

_TOOL_NAME = "document_retrieval"
_TOOL_DESCRIPTION = (
    "Retrieves information from uploaded market research reports and indexed documents. "
    "Use this tool when the user asks questions about specific documents, reports, or "
    "when you need to find information from previously uploaded files. "
    "Input: a plain question or search query string."
)


def _run_rag(query: str, llm: Any, vector_store: Any) -> str:
    """Core RAG logic: retrieve → augment → generate."""
    if not query or not query.strip():
        return "Please provide a question to search the document store."

    # ── Retrieve ──────────────────────────────────────────────────────────────
    results = vector_store.similarity_search(query.strip(), k=5)

    if not results:
        return (
            "No relevant documents found in the vector store for this query.\n\n"
            "You can upload PDF, DOCX, or TXT files using the sidebar uploader, "
            "or ask the AI assistant directly without document context."
        )

    # ── Build context ─────────────────────────────────────────────────────────
    context_parts: list[str] = []
    citations: list[str] = []

    for i, chunk in enumerate(results, start=1):
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")
        score = chunk.get("score")
        score_str = f" (relevance: {score:.2f})" if score is not None else ""

        context_parts.append(f"[Source {i}: {source}{score_str}]\n{text}")
        citations.append(f"  {i}. **{source}** — \"{text[:120]}{'...' if len(text) > 120 else ''}\"")

    context = "\n\n---\n\n".join(context_parts)

    prompt = f"""You are an expert market research analyst.
Using ONLY the retrieved document excerpts below, answer the following question.
If the excerpts do not contain enough information to fully answer, say so clearly.
Always cite which source(s) your answer is drawn from.

RETRIEVED DOCUMENT EXCERPTS:
{context}

QUESTION: {query}

Provide a thorough, well-structured answer based on the retrieved content."""

    # ── Generate ──────────────────────────────────────────────────────────────
    try:
        answer = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("RAG LLM call failed: %s", exc)
        answer = f"LLM generation failed: {exc}"

    # ── Format output with citations ──────────────────────────────────────────
    citation_block = "\n".join(citations)
    return (
        f"📄 **Retrieved Evidence** — {len(results)} document chunks retrieved\n\n"
        f"{answer}\n\n"
        f"---\n"
        f"**Sources consulted:**\n{citation_block}"
    )


def get_rag_tool(llm: Any, vector_store: Any) -> Tool:
    """Return a configured RAG document retrieval Tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        func=lambda query: _run_rag(query, llm, vector_store),
    )
