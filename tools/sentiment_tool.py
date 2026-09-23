"""
tools/sentiment_tool.py
────────────────────────
Customer sentiment analysis tool.

Uses TextBlob for quantitative per-review polarity scoring, then passes
aggregated statistics and representative reviews to the LLM for a
qualitative narrative summary.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

_TOOL_NAME = "sentiment_analysis"
_TOOL_DESCRIPTION = (
    "Analyses customer sentiment from product reviews and feedback data. "
    "Use this tool when the user asks about customer opinions, product ratings, "
    "overall sentiment, user satisfaction, or what customers think about products. "
    "Input: a product name, company name, or topic to analyse sentiment for."
)


def _compute_textblob_sentiment(reviews: list[dict]) -> list[dict]:
    """Add TextBlob polarity score to each review dict."""
    try:
        from textblob import TextBlob  # noqa: PLC0415
    except ImportError:
        logger.warning("textblob not installed — using pre-labeled sentiment")
        for r in reviews:
            r["polarity"] = {"positive": 0.6, "neutral": 0.0, "negative": -0.5}.get(
                r.get("sentiment", "neutral"), 0.0
            )
        return reviews

    for review in reviews:
        text = review.get("review_text", "")
        try:
            blob = TextBlob(text)
            review["polarity"] = round(blob.sentiment.polarity, 3)
            review["subjectivity"] = round(blob.sentiment.subjectivity, 3)
        except Exception:  # noqa: BLE001
            review["polarity"] = 0.0
            review["subjectivity"] = 0.5
    return reviews


def _run_sentiment(query: str, llm: Any) -> str:
    """Run sentiment analysis and return a formatted report."""
    from utils.data_loader import load_reviews  # noqa: PLC0415

    all_reviews = load_reviews()
    if not all_reviews:
        return "No review data available for sentiment analysis."

    # Filter by query keyword if specific product/company mentioned
    query_lower = query.lower()
    filtered = [
        r for r in all_reviews
        if any(
            kw in (r.get("product", "") + " " + r.get("review_text", "")).lower()
            for kw in query_lower.split()
            if len(kw) > 3
        )
    ]
    # Fall back to all reviews if filter yields too few
    reviews = filtered if len(filtered) >= 5 else all_reviews

    # Compute TextBlob scores
    reviews = _compute_textblob_sentiment(reviews)

    # Aggregate statistics
    total = len(reviews)
    positive = [r for r in reviews if r.get("sentiment") == "positive"]
    negative = [r for r in reviews if r.get("sentiment") == "negative"]
    neutral = [r for r in reviews if r.get("sentiment") == "neutral"]

    avg_rating = sum(r.get("rating", 3) for r in reviews) / total
    avg_polarity = sum(r.get("polarity", 0.0) for r in reviews) / total

    # Product breakdown
    from collections import defaultdict
    prod_ratings: dict[str, list[int]] = defaultdict(list)
    for r in reviews:
        prod_ratings[r.get("product", "Unknown")].append(r.get("rating", 3))
    prod_avg = {p: round(sum(v) / len(v), 2) for p, v in prod_ratings.items()}
    top_product = max(prod_avg, key=lambda p: prod_avg[p])
    low_product = min(prod_avg, key=lambda p: prod_avg[p])

    # Sample reviews for LLM context
    sample_pos = positive[:3]
    sample_neg = negative[:2]

    stats_summary = f"""
SENTIMENT STATISTICS:
- Total reviews analysed: {total}
- Positive: {len(positive)} ({len(positive)/total*100:.1f}%)
- Neutral: {len(neutral)} ({len(neutral)/total*100:.1f}%)
- Negative: {len(negative)} ({len(negative)/total*100:.1f}%)
- Average star rating: {avg_rating:.2f}/5.0
- Average TextBlob polarity: {avg_polarity:.3f} (range: -1.0 to +1.0)
- Best rated product: {top_product} ({prod_avg[top_product]}/5.0)
- Lowest rated product: {low_product} ({prod_avg[low_product]}/5.0)

SAMPLE POSITIVE REVIEWS:
{chr(10).join(f'★{r["rating"]} - {r["product"]}: "{r["review_text"][:180]}"' for r in sample_pos)}

SAMPLE NEGATIVE REVIEWS:
{chr(10).join(f'★{r["rating"]} - {r["product"]}: "{r["review_text"][:180]}"' for r in sample_neg)}
"""

    prompt = f"""You are a customer insights analyst.
Based on the following sentiment statistics and customer reviews for '{query}', provide:
1. An overall sentiment assessment
2. Key themes from positive reviews
3. Key themes from negative reviews  
4. Actionable recommendations for product/service improvement

{stats_summary}

Write a professional, structured sentiment analysis report."""

    try:
        analysis = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Sentiment LLM call failed: %s", exc)
        analysis = f"Sentiment summary: {len(positive)/total*100:.0f}% positive, avg rating {avg_rating:.1f}/5.0"

    return (
        f"😊 **Sentiment Analysis** — {total} reviews analysed\n\n"
        f"**Quick Stats:** {len(positive)/total*100:.0f}% positive | "
        f"{len(neutral)/total*100:.0f}% neutral | "
        f"{len(negative)/total*100:.0f}% negative | "
        f"Avg ★{avg_rating:.1f}\n\n"
        f"{analysis}"
    )


def get_sentiment_tool(llm: Any) -> Tool:
    """Return a configured sentiment analysis Tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        func=lambda query: _run_sentiment(query, llm),
    )
