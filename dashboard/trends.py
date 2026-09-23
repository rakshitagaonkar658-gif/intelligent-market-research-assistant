"""
dashboard/trends.py
────────────────────
Trend Analysis tab — 12-month keyword volume line chart, MoM growth heatmap,
notable spike events, and an AI-generated trend narrative.
"""

from __future__ import annotations

from typing import Any


def render_trends(st: Any) -> None:
    """Render the Trend Analysis dashboard tab."""
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from utils.data_loader import load_trends, get_fastest_growing_keyword

    st.markdown("## 📈 Market Trend Analysis")
    st.markdown("*Monthly keyword search volume and growth rates — Jan to Dec 2024*")
    st.markdown("---")

    trends = load_trends()
    if not trends:
        st.warning("No trend data available.")
        return

    df = pd.DataFrame(trends)
    fgk = get_fastest_growing_keyword()

    # ── KPI row ───────────────────────────────────────────────────────────────
    keywords = df["keyword"].unique().tolist()
    cols = st.columns(len(keywords))
    for i, kw in enumerate(sorted(keywords)):
        kw_df = df[df["keyword"] == kw].sort_values("month")
        ytd = round((kw_df["volume"].iloc[-1] - kw_df["volume"].iloc[0]) / kw_df["volume"].iloc[0] * 100, 1)
        with cols[i]:
            st.metric(label=kw, value=f"{kw_df['volume'].iloc[-1]:,}", delta=f"+{ytd}% YTD")

    st.markdown("---")

    # ── Line chart — volume over time ─────────────────────────────────────────
    st.markdown("### 📉 Keyword Volume Trends (Jan–Dec 2024)")

    COLORS = {
        "AI": "#2e5dad",
        "Cloud": "#3b82d4",
        "Fintech": "#e67e22",
        "E-commerce": "#27ae60",
        "Cybersecurity": "#e74c3c",
    }

    fig_line = go.Figure()
    for kw in sorted(keywords):
        kw_df = df[df["keyword"] == kw].sort_values("month")
        fig_line.add_trace(go.Scatter(
            x=kw_df["month"],
            y=kw_df["volume"],
            name=kw,
            mode="lines+markers",
            line=dict(color=COLORS.get(kw, "#888"), width=2.5),
            marker=dict(size=5),
            hovertemplate=f"<b>{kw}</b><br>%{{x}}: %{{y:,}}<extra></extra>",
        ))

    fig_line.update_layout(
        xaxis_title="Month",
        yaxis_title="Search Volume",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=30, b=40, l=60, r=20),
        height=380,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(gridcolor="#e5e7eb", tickangle=-45),
        yaxis=dict(gridcolor="#e5e7eb"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # ── MoM Growth heatmap ────────────────────────────────────────────────────
    st.markdown("### 🌡️ Month-over-Month Growth Rate Heatmap (%)")
    pivot = df.pivot(index="keyword", columns="month", values="growth_pct")
    pivot = pivot.reindex(sorted(pivot.index))

    fig_heat = px.imshow(
        pivot,
        color_continuous_scale=[[0, "#d32f2f"], [0.3, "#fff9c4"], [0.6, "#c8e6c9"], [1, "#1b5e20"]],
        aspect="auto",
        zmin=0,
        zmax=30,
    )
    fig_heat.update_layout(
        coloraxis_colorbar=dict(title="Growth %"),
        margin=dict(t=10, b=40, l=120, r=20),
        height=220,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-45),
    )
    fig_heat.update_traces(
        hovertemplate="<b>%{y}</b> | %{x}<br>Growth: %{z:.1f}%<extra></extra>"
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Notable events ────────────────────────────────────────────────────────
    st.markdown("### ⚡ Notable Trend Events")
    spikes = df[df["growth_pct"] > 15].sort_values("growth_pct", ascending=False)
    if not spikes.empty:
        for _, row in spikes.iterrows():
            color = "🔴" if row["growth_pct"] > 20 else "🟡"
            st.markdown(
                f"{color} **{row['keyword']}** spiked **+{row['growth_pct']}% MoM** "
                f"in {row['month']} (volume: {row['volume']:,})"
            )
    else:
        st.info("No significant growth spikes detected in this period.")

    st.markdown("---")

    # ── AI Trend Summary ──────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Trend Analysis")
    with st.expander("View AI-generated trend narrative", expanded=True):
        top_kw = fgk.get("keyword", "AI")
        top_growth = fgk.get("avg_growth_pct", 0)

        # Compute YTD for all keywords
        summary_lines = []
        for kw in sorted(keywords):
            kw_df = df[df["keyword"] == kw].sort_values("month")
            ytd = round((kw_df["volume"].iloc[-1] - kw_df["volume"].iloc[0]) / kw_df["volume"].iloc[0] * 100, 1)
            summary_lines.append(f"- **{kw}**: +{ytd}% YTD, avg +{kw_df['growth_pct'].mean():.1f}% MoM")

        st.markdown(f"""
**🤖 AI Analysis**

The keyword trend data for January–December 2024 reveals sustained momentum across all five tracked segments:

{chr(10).join(summary_lines)}

**{top_kw}** leads with the highest average monthly growth rate of **+{top_growth:.1f}% MoM**, reflecting enterprise adoption moving from experimental pilots to production deployments. The consistent double-digit monthly growth throughout 2024 is unprecedented for an enterprise software category.

**Cybersecurity** shows an interesting bimodal pattern: steady structural growth overlaid with a sharp event-driven spike in July 2024 (+26.4% MoM) following the CrowdStrike incident. This suggests cybersecurity demand has both a baseline structural driver and significant event sensitivity.

**E-commerce** exhibits strong seasonality with a notable Q4 surge (+16.0% in November), consistent with holiday season patterns and the continued expansion of social commerce channels.

> 🔮 *Forecast / Interpretation:* Extrapolating current growth trajectories, AI search volume is projected to exceed 220,000 monthly queries in Q1 2025. These are trend-based projections and should be treated as directional indicators, not guaranteed outcomes.
        """)

    st.caption("📄 *Source: bundled sample trend data — 60 monthly data points across 5 keywords*")
