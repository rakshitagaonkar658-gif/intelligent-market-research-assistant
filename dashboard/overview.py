"""
dashboard/overview.py
──────────────────────
Market Overview tab — KPI metrics, top competitor card, sentiment snapshot,
trending keyword, and a summary data table.
"""

from __future__ import annotations

from typing import Any


def render_overview(st: Any) -> None:
    """Render the Market Overview dashboard tab."""
    import pandas as pd
    import plotly.graph_objects as go
    from utils.data_loader import (
        load_competitors, load_trends, get_sentiment_summary,
        get_top_competitor, get_fastest_growing_keyword, get_demand_summary,
    )

    st.markdown("## 📊 Market Overview")
    st.markdown("*Real-time snapshot of the enterprise technology market — Q4 2024*")
    st.markdown("---")

    # ── KPI Metrics Row ───────────────────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    tc = get_top_competitor()
    ss = get_sentiment_summary()
    fgk = get_fastest_growing_keyword()
    ds = get_demand_summary()
    competitors = load_competitors()
    total_market = sum(c.get("revenue_bn", 0) for c in competitors)

    with col1:
        st.metric(
            label="🏦 Total Market (tracked)",
            value=f"${total_market:.1f}B",
            delta="12.4% YoY",
        )
    with col2:
        st.metric(
            label="🥇 Market Leader",
            value=tc.get("name", "N/A"),
            delta=f"{tc.get('market_share', 0):.1f}% share",
        )
    with col3:
        st.metric(
            label="😊 Avg Sentiment",
            value=f"★ {ss.get('avg_rating', 0):.1f}/5",
            delta=f"{ss.get('positive_pct', 0):.0f}% positive",
        )
    with col4:
        st.metric(
            label="📈 Top Trend",
            value=fgk.get("keyword", "N/A"),
            delta=f"+{fgk.get('avg_growth_pct', 0):.1f}% avg MoM",
        )
    with col5:
        st.metric(
            label="🔥 Top Demand",
            value=ds.get("top_category", "N/A").split()[0] if ds.get("top_category") else "N/A",
            delta=f"+{ds.get('yoy_change', 0):.1f}% YoY",
        )

    st.markdown("---")

    # ── Market Share Donut Chart ──────────────────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("### 🥧 Market Share Distribution")
        comp_sorted = sorted(competitors, key=lambda c: c.get("market_share", 0), reverse=True)
        labels = [c["name"] for c in comp_sorted]
        values = [c["market_share"] for c in comp_sorted]

        fig_donut = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.45,
            textinfo="label+percent",
            marker=dict(colors=["#2e5dad", "#3b82d4", "#5ba0e8", "#8ec0f8", "#c0deff"]),
        )])
        fig_donut.update_layout(
            showlegend=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_donut, use_container_width=True)

    with col_right:
        st.markdown("### 📊 Revenue by Company ($B)")
        rev_sorted = sorted(competitors, key=lambda c: c.get("revenue_bn", 0), reverse=True)
        fig_bar = go.Figure(data=[go.Bar(
            x=[c["name"] for c in rev_sorted],
            y=[c["revenue_bn"] for c in rev_sorted],
            marker_color=["#2e5dad", "#3b82d4", "#5ba0e8", "#8ec0f8", "#c0deff"],
            text=[f"${c['revenue_bn']}B" for c in rev_sorted],
            textposition="outside",
        )])
        fig_bar.update_layout(
            showlegend=False,
            yaxis_title="Revenue ($B)",
            margin=dict(t=10, b=10, l=10, r=10),
            height=280,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#e5e7eb"),
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── Competitor Summary Table ──────────────────────────────────────────────
    st.markdown("### 🏢 Competitor Summary")
    comp_df = pd.DataFrame([
        {
            "Company": c["name"],
            "Market Share": f"{c['market_share']}%",
            "Share Change": f"{c['market_share'] - c['market_share_prev']:+.1f}pp",
            "Revenue": f"${c['revenue_bn']}B",
            "Employees": f"{c['employees']:,}",
            "HQ": c["hq"],
            "Entry Price": c["pricing"].get("entry_plan", "N/A"),
        }
        for c in comp_sorted
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    # ── Sentiment Quick Stats ─────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 😊 Sentiment Snapshot")
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.metric("Total Reviews", ss.get("total_reviews", 0))
    with s2:
        st.metric("Positive", f"{ss.get('positive_pct', 0):.0f}%", delta="↑ healthy")
    with s3:
        st.metric("Neutral", f"{ss.get('neutral_pct', 0):.0f}%")
    with s4:
        st.metric("Negative", f"{ss.get('negative_pct', 0):.0f}%", delta_color="inverse")

    st.caption("📄 *Data source: bundled sample data (18 news articles, 5 competitor profiles, 32 reviews, 60 trend data points)*")
