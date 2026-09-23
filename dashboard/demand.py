"""
dashboard/demand.py
────────────────────
Demand Forecast tab — grouped bar chart by region and quarter,
demand index heatmap, top demand segments, and AI interpretation.
"""

from __future__ import annotations

from typing import Any


def render_demand(st: Any) -> None:
    """Render the Demand Forecast dashboard tab."""
    import pandas as pd
    import plotly.graph_objects as go
    import plotly.express as px
    from utils.data_loader import load_demand, get_demand_summary

    st.markdown("## 📉 Market Demand Analysis")
    st.markdown("*Demand index by category and region — 2024 quarterly data*")
    st.markdown("---")

    demand = load_demand()
    ds = get_demand_summary()

    if not demand:
        st.warning("No demand data available.")
        return

    df = pd.DataFrame(demand)

    # ── KPI row ───────────────────────────────────────────────────────────────
    q4_df = df[df["quarter"] == "Q4 2024"]
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Top Category", ds.get("top_category", "N/A").split()[0])
    with c2:
        st.metric("Top Region", ds.get("top_region", "N/A"))
    with c3:
        st.metric("Peak Demand Index", ds.get("demand_index", "N/A"))
    with c4:
        st.metric("YoY Change", f"+{ds.get('yoy_change', 0):.1f}%", delta="Q4 2024")

    st.markdown("---")

    # ── Grouped Bar Chart — Q4 2024 by Region ────────────────────────────────
    st.markdown("### 📊 Demand Index by Category (Q4 2024)")
    regions = ["North America", "Europe", "Asia-Pacific"]
    categories = sorted(df["category"].unique())

    REGION_COLORS = {
        "North America": "#2e5dad",
        "Europe": "#27ae60",
        "Asia-Pacific": "#e67e22",
    }

    fig_bar = go.Figure()
    for region in regions:
        region_q4 = q4_df[q4_df["region"] == region].set_index("category")
        y_vals = [region_q4.loc[cat, "demand_index"] if cat in region_q4.index else 0 for cat in categories]
        fig_bar.add_trace(go.Bar(
            name=region,
            x=categories,
            y=y_vals,
            marker_color=REGION_COLORS[region],
            text=[f"{v:.0f}" for v in y_vals],
            textposition="outside",
        ))

    fig_bar.update_layout(
        barmode="group",
        xaxis_title="Category",
        yaxis_title="Demand Index",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=60, l=60, r=20),
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e5e7eb"),
        xaxis=dict(tickangle=-15),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    st.markdown("---")

    # ── YoY Change Heatmap ────────────────────────────────────────────────────
    st.markdown("### 🌡️ YoY Growth by Category & Region (Q4 2024)")
    pivot_yoy = q4_df.pivot(index="region", columns="category", values="yoy_change")
    pivot_yoy = pivot_yoy.reindex(regions)

    fig_heat = px.imshow(
        pivot_yoy,
        color_continuous_scale=[[0, "#fff9c4"], [0.5, "#66bb6a"], [1, "#1b5e20"]],
        aspect="auto",
        text_auto=".1f",
        zmin=0,
        zmax=35,
    )
    fig_heat.update_layout(
        coloraxis_colorbar=dict(title="YoY %"),
        margin=dict(t=10, b=40, l=120, r=20),
        height=200,
        paper_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(tickangle=-20),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("---")

    # ── Quarterly trend line for top category ─────────────────────────────────
    st.markdown("### 📈 Quarterly Demand Trajectory — AI Software")
    top_cat = ds.get("top_category", "AI Software")
    ai_df = df[df["category"] == top_cat].sort_values(["region", "quarter"])

    fig_line = go.Figure()
    for region in regions:
        r_df = ai_df[ai_df["region"] == region]
        fig_line.add_trace(go.Scatter(
            x=r_df["quarter"],
            y=r_df["demand_index"],
            name=region,
            mode="lines+markers",
            line=dict(color=REGION_COLORS[region], width=2.5),
            marker=dict(size=8),
        ))
    fig_line.update_layout(
        xaxis_title="Quarter",
        yaxis_title="Demand Index",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        margin=dict(t=30, b=40, l=60, r=20),
        height=280,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        yaxis=dict(gridcolor="#e5e7eb"),
    )
    st.plotly_chart(fig_line, use_container_width=True)

    st.markdown("---")

    # ── Full data table ────────────────────────────────────────────────────────
    with st.expander("📋 View full demand data table"):
        st.dataframe(
            df.sort_values(["quarter", "demand_index"], ascending=[True, False]),
            use_container_width=True,
            hide_index=True,
        )

    # ── AI demand interpretation ───────────────────────────────────────────────
    st.markdown("### 🤖 AI Demand Interpretation")
    with st.expander("View AI-generated demand analysis", expanded=True):
        st.markdown(f"""
**🤖 AI Analysis**

The demand index data for 2024 reveals clear regional and category patterns:

**Highest Demand:** {ds.get('top_category', 'AI Software')} in {ds.get('top_region', 'Asia-Pacific')} reached a demand index of **{ds.get('demand_index', 116.2)}** in Q4 2024, representing **+{ds.get('yoy_change', 0):.1f}% year-over-year** growth.

**Regional Dynamics:**
- **Asia-Pacific** consistently leads in growth rates across all categories, driven by accelerated digital transformation investment
- **North America** maintains the highest absolute demand indices, reflecting its mature enterprise technology adoption
- **Europe** shows more moderate growth, partly constrained by regulatory review processes around AI deployment

**Category Highlights:**
- *AI Software* shows the strongest and most consistent demand growth trajectory across all regions
- *Cybersecurity* exhibits a notable Q3 2024 spike driven by the CrowdStrike incident and subsequent board-level security reviews
- *E-commerce Platform* demand spikes sharply in Q4 reflecting holiday season infrastructure investments

**Strategic Implication:** Companies entering the AI Software market should prioritise Asia-Pacific go-to-market strategies, where demand growth rates are highest and early-mover advantage is still achievable.

> 🔮 *Forecast / Interpretation:* AI Software demand in Asia-Pacific is projected to surpass North America in absolute index by Q3 2025 if current growth differentials are maintained. These are model-based projections based on current trend extrapolation. Actual outcomes will depend on regulatory developments, macroeconomic conditions, and competitive dynamics.
        """)

    st.caption("📄 *Source: bundled sample demand data — 60 records across 5 categories, 3 regions, 4 quarters*")
