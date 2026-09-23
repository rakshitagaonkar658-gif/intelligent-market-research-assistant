"""
utils/data_loader.py
────────────────────
Typed loader functions for bundled sample data files.

All functions return plain Python lists of dicts so they can be consumed
directly by tools, dashboard modules, and the data-driven MockLLM without
any pandas dependency at the data-loading layer.
"""

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Resolve the data directory relative to this file ─────────────────────────
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_json(filename: str) -> list[dict[str, Any]]:
    """Load a JSON file from the data directory and return it as a list."""
    path = _DATA_DIR / filename
    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            raise ValueError(f"Expected a JSON array in {filename}, got {type(data).__name__}")
        return data
    except FileNotFoundError:
        logger.error("Sample data file not found: %s", path)
        return []
    except json.JSONDecodeError as exc:
        logger.error("JSON parse error in %s: %s", path, exc)
        return []


# ── Public loader functions ───────────────────────────────────────────────────

def load_news() -> list[dict[str, Any]]:
    """
    Return a list of news article dicts.

    Each dict has keys: id, title, source, published_at, description, url,
    category, sentiment, relevance_score.
    """
    return _load_json("sample_news.json")


def load_competitors() -> list[dict[str, Any]]:
    """
    Return a list of competitor profile dicts.

    Each dict has keys: id, name, market_share, market_share_prev, founded,
    hq, revenue_bn, employees, strengths, weaknesses, recent_activities,
    pricing, key_products, growth_strategy.
    """
    return _load_json("sample_competitors.json")


def load_reviews() -> list[dict[str, Any]]:
    """
    Return a list of customer review dicts.

    Each dict has keys: id, product, rating, date, review_text, sentiment,
    reviewer.
    """
    return _load_json("sample_reviews.json")


def load_trends() -> list[dict[str, Any]]:
    """
    Return a list of monthly trend data points.

    Each dict has keys: keyword, month (YYYY-MM), volume, growth_pct.
    Keywords covered: AI, Cloud, Fintech, E-commerce, Cybersecurity.
    """
    return _load_json("sample_trends.json")


def load_demand() -> list[dict[str, Any]]:
    """
    Return a list of demand indicator records.

    Each dict has keys: category, region, quarter, demand_index, yoy_change.
    """
    return _load_json("sample_demand.json")


def load_sample_report_texts() -> list[dict[str, str]]:
    """
    Return a list of dicts with 'filename' and 'text' for each sample report.
    """
    reports_dir = _DATA_DIR / "sample_reports"
    results = []
    for path in sorted(reports_dir.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
            results.append({"filename": path.name, "text": text})
        except OSError as exc:
            logger.error("Could not read sample report %s: %s", path.name, exc)
    return results


# ── Convenience helpers used by the MockLLM ───────────────────────────────────

def get_top_competitor() -> dict[str, Any]:
    """Return the competitor with the highest current market share."""
    competitors = load_competitors()
    if not competitors:
        return {}
    return max(competitors, key=lambda c: c.get("market_share", 0))


def get_fastest_growing_keyword() -> dict[str, Any]:
    """Return the keyword with the highest average monthly growth in 2024."""
    trends = load_trends()
    if not trends:
        return {}
    # Group by keyword and compute mean growth_pct
    from collections import defaultdict
    totals: dict[str, list[float]] = defaultdict(list)
    for row in trends:
        totals[row["keyword"]].append(row.get("growth_pct", 0.0))
    avg_growth = {kw: sum(vals) / len(vals) for kw, vals in totals.items()}
    best_kw = max(avg_growth, key=lambda k: avg_growth[k])
    return {"keyword": best_kw, "avg_growth_pct": round(avg_growth[best_kw], 2)}


def get_sentiment_summary() -> dict[str, Any]:
    """Return aggregated sentiment statistics from customer reviews."""
    reviews = load_reviews()
    if not reviews:
        return {}
    total = len(reviews)
    positive = sum(1 for r in reviews if r.get("sentiment") == "positive")
    negative = sum(1 for r in reviews if r.get("sentiment") == "negative")
    neutral = total - positive - negative
    avg_rating = sum(r.get("rating", 3) for r in reviews) / total
    return {
        "total_reviews": total,
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "positive_pct": round(positive / total * 100, 1),
        "negative_pct": round(negative / total * 100, 1),
        "neutral_pct": round(neutral / total * 100, 1),
        "avg_rating": round(avg_rating, 2),
    }


def get_demand_summary() -> dict[str, Any]:
    """Return the highest-demand category and region for Q4 2024."""
    demand = load_demand()
    q4 = [d for d in demand if d.get("quarter") == "Q4 2024"]
    if not q4:
        return {}
    top = max(q4, key=lambda d: d.get("demand_index", 0))
    return {
        "top_category": top.get("category"),
        "top_region": top.get("region"),
        "demand_index": top.get("demand_index"),
        "yoy_change": top.get("yoy_change"),
    }
