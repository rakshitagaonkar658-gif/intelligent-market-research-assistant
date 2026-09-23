"""
dashboard/alerts.py
────────────────────
Alerts tab — rules-based market intelligence alerts computed from sample data.

Alert rules:
  - Trend spike: keyword MoM growth > 20%
  - Competitor gaining: share increase > 1.5pp
  - Competitor losing: share decrease > 1.5pp
  - High demand: demand index > 100 in any segment
  - Negative sentiment: negative review pct > 25%
  - High YoY demand: any segment YoY > 25%
"""

from __future__ import annotations

from typing import Any


def _compute_alerts(competitors: list, trends: list, demand: list, sentiment: dict) -> list[dict]:
    """Run all alert rules and return a list of alert dicts."""
    alerts: list[dict] = []

    # ── Competitor alerts ─────────────────────────────────────────────────────
    for c in competitors:
        delta = c["market_share"] - c["market_share_prev"]
        if delta >= 1.5:
            alerts.append({
                "level": "warning",
                "category": "Competitor",
                "icon": "🚀",
                "title": f"{c['name']} gaining market share",
                "detail": (
                    f"Share increased from {c['market_share_prev']}% to {c['market_share']}% "
                    f"(+{delta:.1f}pp). Recent driver: {c['recent_activities'][0]}"
                ),
            })
        elif delta <= -1.5:
            alerts.append({
                "level": "info",
                "category": "Competitor",
                "icon": "📉",
                "title": f"{c['name']} losing market share",
                "detail": (
                    f"Share declined from {c['market_share_prev']}% to {c['market_share']}% "
                    f"({delta:.1f}pp). Key weakness: {c['weaknesses'][0]}"
                ),
            })

    # ── Trend alerts ──────────────────────────────────────────────────────────
    for row in trends:
        if row.get("growth_pct", 0) > 20:
            alerts.append({
                "level": "warning",
                "category": "Trend",
                "icon": "📈",
                "title": f"{row['keyword']} keyword spike",
                "detail": (
                    f"+{row['growth_pct']}% MoM growth in {row['month']} "
                    f"(volume: {row['volume']:,}). Significant above-average acceleration."
                ),
            })

    # ── Demand alerts ─────────────────────────────────────────────────────────
    for row in demand:
        if row.get("demand_index", 0) > 100:
            alerts.append({
                "level": "success",
                "category": "Demand",
                "icon": "🔥",
                "title": f"High demand: {row['category']} in {row['region']}",
                "detail": (
                    f"Demand index reached {row['demand_index']} in {row['quarter']} "
                    f"(YoY: +{row['yoy_change']}%). Indicates strong market opportunity."
                ),
            })
        if row.get("yoy_change", 0) > 25:
            alerts.append({
                "level": "success",
                "category": "Demand",
                "icon": "⚡",
                "title": f"Exceptional growth: {row['category']} in {row['region']}",
                "detail": (
                    f"+{row['yoy_change']}% YoY growth in {row['quarter']}. "
                    f"Demand index: {row['demand_index']}. High-priority market opportunity."
                ),
            })

    # ── Sentiment alerts ──────────────────────────────────────────────────────
    if sentiment.get("negative_pct", 0) > 25:
        alerts.append({
            "level": "error",
            "category": "Sentiment",
            "icon": "😞",
            "title": "Elevated negative sentiment",
            "detail": (
                f"{sentiment['negative_pct']}% of reviews are negative "
                f"(threshold: 25%). Immediate attention to customer pain points recommended."
            ),
        })

    # Deduplicate demand alerts (index > 100 check creates many — keep top 5 by index)
    demand_alerts = [a for a in alerts if a["category"] == "Demand"]
    if len(demand_alerts) > 5:
        demand_alerts_sorted = sorted(demand_alerts, key=lambda a: -float(a["detail"].split("demand index reached ")[-1].split(" ")[0]) if "Demand index" not in a["detail"] else 0, reverse=False)
        non_demand = [a for a in alerts if a["category"] != "Demand"]
        alerts = non_demand + demand_alerts[:5]

    return alerts


def render_alerts(st: Any) -> None:
    """Render the Alerts dashboard tab."""
    from utils.data_loader import (
        load_competitors, load_trends, load_demand, get_sentiment_summary
    )

    st.markdown("## 🔔 Market Intelligence Alerts")
    st.markdown("*Rules-based alerts computed from live market signals — refreshed on page load*")
    st.markdown("---")

    competitors = load_competitors()
    trends = load_trends()
    demand = load_demand()
    sentiment = get_sentiment_summary()

    # Filter demand to Q4 only to avoid duplicate alerts
    demand_q4 = [d for d in demand if d.get("quarter") == "Q4 2024"]

    alerts = _compute_alerts(competitors, trends, demand_q4, sentiment)

    if not alerts:
        st.success("✅ No significant market alerts at this time.")
        return

    # ── Summary badges ────────────────────────────────────────────────────────
    from collections import Counter
    level_counts = Counter(a["level"] for a in alerts)
    cat_counts = Counter(a["category"] for a in alerts)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Alerts", len(alerts))
    with col2:
        st.metric("🚨 High Priority", level_counts.get("warning", 0) + level_counts.get("error", 0))
    with col3:
        st.metric("✅ Opportunities", level_counts.get("success", 0))
    with col4:
        st.metric("ℹ️ Informational", level_counts.get("info", 0))

    st.markdown("---")

    # ── Category filter ───────────────────────────────────────────────────────
    categories = ["All"] + sorted(cat_counts.keys())
    selected_cat = st.selectbox("Filter by category", categories, key="alert_filter")

    filtered = alerts if selected_cat == "All" else [a for a in alerts if a["category"] == selected_cat]

    st.markdown(f"**Showing {len(filtered)} alert(s)**")
    st.markdown("")

    # ── Alert cards ───────────────────────────────────────────────────────────
    LEVEL_RENDERERS = {
        "error": "error",
        "warning": "warning",
        "success": "success",
        "info": "info",
    }

    for alert in filtered:
        render_fn = getattr(st, LEVEL_RENDERERS.get(alert["level"], "info"))
        render_fn(
            f"{alert['icon']} **[{alert['category']}] {alert['title']}**\n\n"
            f"{alert['detail']}"
        )

    st.markdown("---")
    st.markdown("### ⚙️ Alert Rules Active")
    with st.expander("View alert rule definitions"):
        st.markdown("""
| Rule | Threshold | Category |
|---|---|---|
| Competitor gaining share | ≥ +1.5 percentage points | Competitor |
| Competitor losing share | ≤ -1.5 percentage points | Competitor |
| Keyword search spike | > +20% MoM growth | Trend |
| High demand index | > 100 (any segment) | Demand |
| Exceptional YoY growth | > +25% (any segment) | Demand |
| Negative sentiment elevated | > 25% negative reviews | Sentiment |
        """)

    st.caption("📄 *Alerts are computed from bundled sample data on each page load. In a production deployment, these would refresh from live data sources.*")
