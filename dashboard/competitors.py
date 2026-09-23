"""
dashboard/competitors.py
─────────────────────────
Competitor Comparison tab — market share bar chart, share change tracker,
detailed comparison table, and LLM strategic insights.
"""

from __future__ import annotations

from typing import Any


def render_competitors(st: Any) -> None:
    """Render the Competitor Comparison dashboard tab."""
    import pandas as pd
    import plotly.graph_objects as go
    from utils.data_loader import load_competitors

    st.markdown("## 🏢 Competitor Comparison")
    st.markdown("*Detailed competitive landscape analysis — Q4 2024*")
    st.markdown("---")

    competitors = load_competitors()
    if not competitors:
        st.warning("No competitor data available.")
        return

    ranked = sorted(competitors, key=lambda c: c.get("market_share", 0), reverse=True)

    # ── Market Share Bar Chart ─────────────────────────────────────────────────
    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("### 📊 Current vs Previous Market Share")
        names = [c["name"] for c in ranked]
        current = [c["market_share"] for c in ranked]
        previous = [c["market_share_prev"] for c in ranked]

        fig = go.Figure()
        fig.add_trace(go.Bar(
            name="Current",
            x=names,
            y=current,
            marker_color="#2e5dad",
            text=[f"{v}%" for v in current],
            textposition="outside",
        ))
        fig.add_trace(go.Bar(
            name="Previous",
            x=names,
            y=previous,
            marker_color="#8ec0f8",
            text=[f"{v}%" for v in previous],
            textposition="outside",
        ))
        fig.update_layout(
            barmode="group",
            yaxis_title="Market Share (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02),
            margin=dict(t=30, b=40, l=50, r=20),
            height=320,
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            yaxis=dict(gridcolor="#e5e7eb"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("### 📈 Share Change Tracker")
        for c in ranked:
            delta = c["market_share"] - c["market_share_prev"]
            arrow = "▲" if delta > 0 else ("▼" if delta < 0 else "►")
            color_class = "🟢" if delta > 0 else ("🔴" if delta < 0 else "⚪")
            st.markdown(
                f"{color_class} **{c['name']}**  \n"
                f"&nbsp;&nbsp;&nbsp;{arrow} {delta:+.1f}pp → **{c['market_share']}%**"
            )
            st.markdown("")

    st.markdown("---")

    # ── Detailed Comparison Table ─────────────────────────────────────────────
    st.markdown("### 📋 Full Competitive Profile")
    comp_df = pd.DataFrame([
        {
            "Company": c["name"],
            "Share": f"{c['market_share']}%",
            "Δ Share": f"{c['market_share'] - c['market_share_prev']:+.1f}pp",
            "Revenue": f"${c['revenue_bn']}B",
            "Staff": f"{c['employees']:,}",
            "Entry Price": c["pricing"].get("entry_plan", "N/A"),
            "Strategy": c["growth_strategy"][:60] + "..." if len(c["growth_strategy"]) > 60 else c["growth_strategy"],
        }
        for c in ranked
    ])
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── Expandable Profiles ───────────────────────────────────────────────────
    st.markdown("### 🔍 Detailed Company Profiles")
    tabs = st.tabs([c["name"] for c in ranked])
    for tab, comp in zip(tabs, ranked):
        with tab:
            t1, t2 = tab.columns(2)
            with t1:
                tab.markdown(f"**Founded:** {comp['founded']} | **HQ:** {comp['hq']}")
                tab.markdown(f"**Revenue:** ${comp['revenue_bn']}B | **Employees:** {comp['employees']:,}")
                tab.markdown("**✅ Strengths:**")
                for s in comp["strengths"]:
                    tab.markdown(f"  - {s}")
                tab.markdown("**⚠️ Weaknesses:**")
                for w in comp["weaknesses"]:
                    tab.markdown(f"  - {w}")
            with t2:
                tab.markdown("**🚀 Recent Activities:**")
                for a in comp["recent_activities"]:
                    tab.markdown(f"  - {a}")
                tab.markdown("**💰 Pricing:**")
                for tier, price in comp["pricing"].items():
                    tab.markdown(f"  - {tier.replace('_', ' ').title()}: {price}")
                tab.markdown(f"**📌 Key Products:** {', '.join(comp['key_products'])}")

    st.markdown("---")

    # ── Strategic Insights ────────────────────────────────────────────────────
    st.markdown("### 🤖 AI Strategic Insights")
    with st.expander("View AI-generated competitive analysis", expanded=True):
        leader = ranked[0]
        challenger = ranked[1]
        fastest_gainer = max(ranked, key=lambda c: c["market_share"] - c["market_share_prev"])
        losing = min(ranked, key=lambda c: c["market_share"] - c["market_share_prev"])

        st.markdown(f"""
**🤖 AI Analysis**

**Market Structure:** The enterprise technology market shows a clear two-tier structure. **{leader['name']}** leads with {leader['market_share']}% share, benefitting from {leader['strengths'][0].lower()}. **{challenger['name']}** is the primary challenger at {challenger['market_share']}%, growing through {challenger['growth_strategy'].lower()}.

**Momentum Watch:** **{fastest_gainer['name']}** is the fastest-gaining player, adding {fastest_gainer['market_share'] - fastest_gainer['market_share_prev']:+.1f}pp of share. This growth is driven by: {fastest_gainer['recent_activities'][0].lower()}.

**Concern:** **{losing['name']}** lost {abs(losing['market_share'] - losing['market_share_prev']):.1f}pp — a warning signal. Key vulnerabilities: {losing['weaknesses'][0].lower()}.

**Pricing Dynamics:** Entry pricing ranges from ${ranked[-1]['pricing']['entry_plan']} (lowest) to ${leader['pricing']['entry_plan']} (highest), creating a significant price ladder. Cloud-native challengers are undercutting incumbents by 40–60% at comparable workload levels.

**Strategic Imperatives:**
- Enterprises should re-evaluate incumbent contracts against challenger pricing — potential 35–50% cost savings available
- Developer experience is the #1 differentiator for mid-market adoption
- Compliance and vertical specialisation remain defensible moats in regulated industries

> 🔮 *Forecast / Interpretation:* {fastest_gainer['name']} is on track to surpass the current #3 ranked competitor within 2–3 quarters based on current trajectory. This is a projection, not a guaranteed outcome.
        """)

    st.caption("📄 *Source: bundled sample competitor data — 5 company profiles with historical share data*")
