"""
tools/news_tool.py
──────────────────
News research tool.

Priority:
  1. NewsAPI.org (live) — if NEWSAPI_KEY is configured
  2. Bundled sample_news.json — fallback for demo mode or missing key
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

_TOOL_NAME = "news_research"
_TOOL_DESCRIPTION = (
    "Searches for recent news articles and market intelligence on a given topic. "
    "Use this tool when the user asks about recent events, market news, headlines, "
    "or current developments in a specific industry or about specific companies. "
    "Input: a topic, keyword, or question about recent news."
)


def _fetch_newsapi(query: str, api_key: str) -> list[dict]:
    """Fetch articles from NewsAPI.org. Returns empty list on failure."""
    try:
        from newsapi import NewsApiClient  # noqa: PLC0415
        client = NewsApiClient(api_key=api_key)
        response = client.get_everything(
            q=query,
            language="en",
            sort_by="relevancy",
            page_size=10,
        )
        articles = response.get("articles", [])
        return [
            {
                "title": a.get("title", ""),
                "source": a.get("source", {}).get("name", "Unknown"),
                "published_at": (a.get("publishedAt") or "")[:10],
                "description": a.get("description") or a.get("content") or "",
                "url": a.get("url", ""),
            }
            for a in articles
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as exc:  # noqa: BLE001
        logger.warning("NewsAPI call failed: %s — using sample data", exc)
        return []


def _filter_sample_news(query: str) -> list[dict]:
    """Filter bundled sample news by keyword relevance."""
    from utils.data_loader import load_news  # noqa: PLC0415
    all_news = load_news()
    query_words = set(query.lower().split())

    def relevance(article: dict) -> float:
        text = (
            article.get("title", "") + " " +
            article.get("description", "") + " " +
            article.get("category", "")
        ).lower()
        matches = sum(1 for w in query_words if len(w) > 3 and w in text)
        return matches + article.get("relevance_score", 0.5)

    sorted_news = sorted(all_news, key=relevance, reverse=True)
    return sorted_news[:10]


def _run_news(query: str, llm: Any) -> str:
    """Fetch news articles and generate a summary using the LLM."""
    from utils.config import config  # noqa: PLC0415

    # Fetch articles
    articles: list[dict] = []
    source_label = "📰 Sample data (demo mode)"

    if config.has_newsapi():
        articles = _fetch_newsapi(query, config.newsapi_key)
        if articles:
            source_label = "📰 Live NewsAPI.org"

    if not articles:
        articles = _filter_sample_news(query)
        source_label = "📰 Sample data (bundled)"

    if not articles:
        return "No news articles found for this query."

    top5 = articles[:5]

    # Format for LLM
    articles_text = "\n\n".join(
        f"{i+1}. **{a['title']}** ({a['source']}, {a['published_at']})\n   {a['description'][:250]}"
        for i, a in enumerate(top5)
    )

    prompt = f"""You are a market intelligence analyst.
Analyse the following news articles about '{query}' and provide:
1. A concise executive summary (2-3 sentences)
2. The 3 most important market signals or trends emerging from these articles
3. Potential business implications

NEWS ARTICLES:
{articles_text}

Provide a structured, insightful analysis."""

    try:
        analysis = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("News LLM call failed: %s", exc)
        analysis = "\n".join(
            f"• {a['title']} — {a['description'][:150]}" for a in top5
        )

    return (
        f"{source_label} — {len(top5)} articles\n\n"
        f"**Query:** {query}\n\n"
        f"{analysis}\n\n"
        f"---\n"
        f"**Articles referenced:**\n" +
        "\n".join(f"  {i+1}. [{a['title']}]({a.get('url', '#')}) — {a['source']}" for i, a in enumerate(top5))
    )


def get_news_tool(llm: Any) -> Tool:
    """Return a configured news research Tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        func=lambda query: _run_news(query, llm),
    )
