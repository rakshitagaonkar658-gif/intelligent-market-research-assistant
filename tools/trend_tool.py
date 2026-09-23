"""
tools/trend_tool.py
────────────────────
Market trend analysis tool.

Loads 12-month keyword volume data, computes MoM growth rates,
identifies notable spikes/events, and asks the LLM to produce a
trend narrative with forward-looking interpretation.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

_TOOL_NAME = "trend_analysis"
_TOOL_DESCRIPTION = (
    "Analyses market trends, keyword search volumes, and growth patterns over time. "
    "Use this tool when the user asks about market trends, what is growing, "
    "trend data, keyword volumes, industry momentum, or what topics are rising. "
    "Input: a keyword, industry, or trend question."
)


def _run_trend(query: str, llm: Any) -> str:
    """Compute trend statistics and generate LLM narrative."""
    from utils.data_loader import load_trends  # noqa: PLC0415

    trends = load_trends()
    if not trends:
        return "No trend data available."

    # Group by keyword
    kw_data: dict[str, list[dict]] = defaultdict(list)
    for row in trends:
        kw_data[row["keyword"]].append(row)

    # Check if query references a specific keyword
    query_lower = query.lower()
    relevant_kws = [
        kw for kw in kw_data
        if kw.lower() in query_lower or any(w in kw.lower() for w in query_lower.split() if len(w) > 3)
    ]
    # Use all keywords if no specific match
    analysis_kws = relevant_kws if relevant_kws else list(kw_data.keys())

    # Build per-keyword statistics
    kw_stats: list[dict] = []
    for kw in analysis_kws:
        rows = sorted(kw_data[kw], key=lambda r: r["month"])
        vol_jan = rows[0]["volume"]
        vol_dec = rows[-1]["volume"]
        ytd_growth = round((vol_dec - vol_jan) / vol_jan * 100, 1)
        avg_monthly = round(sum(r["growth_pct"] for r in rows) / len(rows), 2)
        peak = max(rows, key=lambda r: r["growth_pct"])
        latest_vol = rows[-1]["volume"]
        kw_stats.append({
            "keyword": kw,
            "jan_vol": vol_jan,
            "dec_vol": vol_dec,
            "ytd_growth": ytd_growth,
            "avg_monthly_growth": avg_monthly,
            "peak_month": peak["month"],
            "peak_growth": peak["growth_pct"],
            "latest_vol": latest_vol,
            "monthly_data": rows,
        })

    # Sort by YTD growth
    kw_stats.sort(key=lambda s: s["ytd_growth"], reverse=True)

    # Build table
    table_header = (
        "| Keyword | Jan Vol | Dec Vol | YTD Growth | Avg MoM | Peak Month |\n"
        "|---|---|---|---|---|---|\n"
    )
    table_rows = "\n".join(
        f"| {s['keyword']} | {s['jan_vol']:,} | {s['dec_vol']:,} | "
        f"+{s['ytd_growth']}% | {s['avg_monthly_growth']}% | "
        f"{s['peak_month']} (+{s['peak_growth']}%) |"
        for s in kw_stats
    )
    table = table_header + table_rows

    # Notable events (MoM spikes > 15%)
    spikes = [
        f"  - **{row['keyword']} in {row['month']}**: +{row['growth_pct']}% MoM spike"
        for kw in analysis_kws
        for row in kw_data[kw]
        if row["growth_pct"] > 15
    ]
    spike_text = "\n".join(spikes) if spikes else "  No significant spikes detected."

    prompt = f"""You are a market trend analyst.
The user asks: "{query}"

Using the keyword trend data below, provide:
1. Direct answer to the user's trend question
2. Which keywords/topics show the strongest momentum and why
3. Notable trend events or inflection points
4. A forward-looking interpretation of where these trends are heading

TREND DATA TABLE:
{table}

NOTABLE SPIKE EVENTS:
{spike_text}

Be specific, cite the data, and clearly label any forward-looking statements as projections."""

    try:
        analysis = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Trend LLM call failed: %s", exc)
        top = kw_stats[0] if kw_stats else {}
        analysis = f"Top trend: {top.get('keyword', 'N/A')} with {top.get('ytd_growth', 'N/A')}% YTD growth."

    return (
        f"📈 **Trend Analysis** — {len(analysis_kws)} keyword(s) analysed\n\n"
        f"{table}\n\n"
        f"{analysis}"
    )


def get_trend_tool(llm: Any) -> Tool:
    """Return a configured trend analysis Tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        func=lambda query: _run_trend(query, llm),
    )
