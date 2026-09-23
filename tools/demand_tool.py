"""
tools/demand_tool.py
─────────────────────
Market demand analysis tool.

Loads regional demand index data, computes summaries by category and region,
and asks the LLM to interpret demand signals and highlight hotspots.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

_TOOL_NAME = "demand_analysis"
_TOOL_DESCRIPTION = (
    "Analyses market demand indicators, demand indexes, and regional demand patterns. "
    "Use this tool when the user asks about market demand, demand forecasts, "
    "which markets are growing, regional demand, or demand signals for a product category. "
    "Input: a product category, region, or demand-related question."
)


def _run_demand(query: str, llm: Any) -> str:
    """Compute demand statistics and generate LLM interpretation."""
    from utils.data_loader import load_demand  # noqa: PLC0415

    demand = load_demand()
    if not demand:
        return "No demand data available."

    # Group by category and region
    by_cat_region: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in demand:
        by_cat_region[row["category"]][row["region"]].append(row)

    # Determine if query targets a specific category or region
    query_lower = query.lower()
    focus_cats = [
        cat for cat in by_cat_region
        if any(w in cat.lower() for w in query_lower.split() if len(w) > 3)
    ]
    focus_regions = [
        r for r in ["North America", "Europe", "Asia-Pacific"]
        if r.lower() in query_lower
    ]
    use_cats = focus_cats if focus_cats else list(by_cat_region.keys())

    # Build Q4 2024 summary table
    q4_rows: list[dict] = []
    for cat in use_cats:
        for region, rows in by_cat_region[cat].items():
            q4 = [r for r in rows if r.get("quarter") == "Q4 2024"]
            if q4:
                q4_rows.append({
                    "category": cat,
                    "region": region,
                    "demand_index": q4[0]["demand_index"],
                    "yoy_change": q4[0]["yoy_change"],
                })

    q4_rows.sort(key=lambda r: r["demand_index"], reverse=True)

    table_header = (
        "| Category | Region | Demand Index (Q4 2024) | YoY Change |\n"
        "|---|---|---|---|\n"
    )
    table_rows = "\n".join(
        f"| {r['category']} | {r['region']} | {r['demand_index']} | +{r['yoy_change']}% |"
        for r in q4_rows
    )
    table = table_header + table_rows

    # YoY trend: Q1→Q4 for highest-demand category
    top_cat = q4_rows[0]["category"] if q4_rows else use_cats[0]
    quarterly_lines = []
    for region, rows in by_cat_region[top_cat].items():
        for r in sorted(rows, key=lambda x: x["quarter"]):
            quarterly_lines.append(
                f"  {top_cat} / {region} / {r['quarter']}: "
                f"index={r['demand_index']}, yoy=+{r['yoy_change']}%"
            )
    quarterly_text = "\n".join(quarterly_lines[:12])  # cap output length

    prompt = f"""You are a market demand analyst.
The user asks: "{query}"

Using the demand index data below, provide:
1. Direct answer to the user's demand question
2. Which categories and regions show the strongest demand growth
3. Notable demand patterns or seasonal effects
4. Strategic implications for market entry or investment

DEMAND INDEX TABLE (Q4 2024):
{table}

QUARTERLY TREND FOR '{top_cat}':
{quarterly_text}

Clearly label any demand projections as forecasts, not guaranteed outcomes."""

    try:
        analysis = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Demand LLM call failed: %s", exc)
        top = q4_rows[0] if q4_rows else {}
        analysis = (
            f"Highest demand: {top.get('category', 'N/A')} in "
            f"{top.get('region', 'N/A')} "
            f"(index: {top.get('demand_index', 'N/A')}, "
            f"YoY: +{top.get('yoy_change', 'N/A')}%)"
        )

    return (
        f"📉 **Demand Analysis** — {len(q4_rows)} category/region combinations\n\n"
        f"{table}\n\n"
        f"{analysis}"
    )


def get_demand_tool(llm: Any) -> Tool:
    """Return a configured demand analysis Tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        func=lambda query: _run_demand(query, llm),
    )
