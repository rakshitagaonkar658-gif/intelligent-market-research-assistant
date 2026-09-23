"""
tools/__init__.py
─────────────────
Tool registry — assembles all seven agent tools and returns them as a list.

Usage
-----
    from tools import get_all_tools
    from models import get_llm
    from rag.vector_store import VectorStoreManager

    tools = get_all_tools(get_llm(), VectorStoreManager())
"""

from __future__ import annotations

from typing import Any


def get_all_tools(llm: Any, vector_store: Any) -> list:
    """
    Instantiate and return all seven LangChain tools.

    Parameters
    ----------
    llm : BaseLLM
        The application LLM (Granite or MockLLM).
    vector_store : VectorStoreManager
        The initialised ChromaDB vector store manager.

    Returns
    -------
    list[Tool]
        All tools ready for use with a LangChain AgentExecutor.
    """
    from tools.rag_tool import get_rag_tool
    from tools.news_tool import get_news_tool
    from tools.sentiment_tool import get_sentiment_tool
    from tools.competitor_tool import get_competitor_tool
    from tools.trend_tool import get_trend_tool
    from tools.demand_tool import get_demand_tool
    from tools.report_tool import get_report_tool

    return [
        get_rag_tool(llm, vector_store),
        get_news_tool(llm),
        get_sentiment_tool(llm),
        get_competitor_tool(llm),
        get_trend_tool(llm),
        get_demand_tool(llm),
        get_report_tool(llm),
    ]


__all__ = ["get_all_tools"]
