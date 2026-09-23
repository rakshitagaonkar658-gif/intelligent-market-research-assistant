"""
tools/competitor_tool.py
─────────────────────────
Competitor comparison tool.

Loads the bundled competitor profiles, formats a comparison table,
and asks the LLM to provide strategic insights.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

_TOOL_NAME = "competitor_comparison"
_TOOL_DESCRIPTION = (
    "Compares competitors in the market including market share, strengths, weaknesses, "
    "pricing, and recent strategic activities. Use this tool when the user asks about "
    "competitive landscape, competitor analysis, market share, rival companies, or "
    "how companies compare against each other. "
    "Input: a company name, market segment, or comparison question."
)


def _run_competitor(query: str, llm: Any) -> str:
    """Build competitor comparison and generate LLM strategic insights."""
    from utils.data_loader import load_competitors  # noqa: PLC0415

    competitors = load_competitors()
    if not competitors:
        return "No competitor data available."

    # Sort by market share
    ranked = sorted(competitors, key=lambda c: c.get("market_share", 0), reverse=True)

    # Build markdown comparison table
    table_header = (
        "| Company | Share | Prev | Change | Revenue | Employees | HQ |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    table_rows = "\n".join(
        f"| {c['name']} | {c['market_share']}% | {c['market_share_prev']}% | "
        f"{c['market_share'] - c['market_share_prev']:+.1f}pp | "
        f"${c['revenue_bn']}B | {c['employees']:,} | {c['hq']} |"
        for c in ranked
    )
    table = table_header + table_rows

    # Build detailed profiles
    profiles = []
    for c in ranked:
        profiles.append(
            f"**{c['name']}** (Market share: {c['market_share']}%)\n"
            f"- Strengths: {'; '.join(c['strengths'][:3])}\n"
            f"- Weaknesses: {'; '.join(c['weaknesses'][:2])}\n"
            f"- Recent: {c['recent_activities'][0]}\n"
            f"- Strategy: {c['growth_strategy']}\n"
            f"- Entry price: {c['pricing'].get('entry_plan', 'N/A')}"
        )
    profiles_text = "\n\n".join(profiles)

    prompt = f"""You are a competitive intelligence analyst.
The user asks: "{query}"

Using the competitor data below, provide:
1. Direct answer to the user's question
2. Key competitive dynamics in this market
3. Who has momentum and why
4. Strategic recommendations based on competitive positioning

MARKET SHARE TABLE:
{table}

COMPETITOR PROFILES:
{profiles_text}

Be specific and reference actual company data in your analysis."""

    try:
        analysis = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Competitor LLM call failed: %s", exc)
        leader = ranked[0]
        analysis = f"Market leader: {leader['name']} with {leader['market_share']}% share."

    return (
        f"🏢 **Competitor Analysis** — {len(ranked)} companies tracked\n\n"
        f"{table}\n\n"
        f"{analysis}"
    )


def get_competitor_tool(llm: Any) -> Tool:
    """Return a configured competitor comparison Tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        func=lambda query: _run_competitor(query, llm),
    )
