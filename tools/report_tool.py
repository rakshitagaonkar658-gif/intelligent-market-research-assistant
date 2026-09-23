"""
tools/report_tool.py
─────────────────────
Market research report generation tool.

Orchestrates all data sources into a comprehensive Markdown report using the LLM,
then converts it to a downloadable PDF via reportlab.

PDF is saved to data/exports/<timestamp>_market_report.pdf.
The tool returns both the Markdown string and the pdf_path so the agent can
pass it to the UI for a st.download_button.
"""

from __future__ import annotations

import logging
import re
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any

from langchain_core.tools import Tool

logger = logging.getLogger(__name__)

_TOOL_NAME = "report_generation"
_TOOL_DESCRIPTION = (
    "Generates a comprehensive market research report covering market overview, "
    "competitor analysis, customer sentiment, trends, demand forecasts, and "
    "actionable recommendations. The report is returned as Markdown and also "
    "exported as a downloadable PDF. "
    "Use this tool when the user asks to generate, create, or write a market report, "
    "research summary, or comprehensive analysis. "
    "Input: a topic, market segment, or 'full report'."
)

# Module-level storage for the latest pdf_path so the agent can retrieve it
_latest_pdf_path: Path | None = None


def get_latest_pdf_path() -> Path | None:
    """Return the path of the most recently generated PDF report."""
    return _latest_pdf_path


def _gather_data_summary() -> dict[str, Any]:
    """Collect summary statistics from all data sources."""
    from utils.data_loader import (  # noqa: PLC0415
        load_competitors, load_trends, get_sentiment_summary,
        get_demand_summary, get_top_competitor, get_fastest_growing_keyword,
        load_news,
    )
    from collections import defaultdict

    competitors = load_competitors()
    ranked_comp = sorted(competitors, key=lambda c: c.get("market_share", 0), reverse=True)

    trends = load_trends()
    kw_data: dict[str, list] = defaultdict(list)
    for row in trends:
        kw_data[row["keyword"]].append(row)

    kw_ytd: dict[str, float] = {}
    for kw, rows in kw_data.items():
        rows_s = sorted(rows, key=lambda r: r["month"])
        if rows_s:
            kw_ytd[kw] = round((rows_s[-1]["volume"] - rows_s[0]["volume"]) / rows_s[0]["volume"] * 100, 1)

    news = load_news()
    top_news = sorted(news, key=lambda n: n.get("relevance_score", 0), reverse=True)[:5]

    return {
        "competitors": ranked_comp,
        "sentiment": get_sentiment_summary(),
        "demand": get_demand_summary(),
        "top_competitor": get_top_competitor(),
        "fastest_kw": get_fastest_growing_keyword(),
        "kw_ytd": kw_ytd,
        "top_news": top_news,
    }


def _build_report_prompt(query: str, data: dict[str, Any]) -> str:
    """Build the full report generation prompt."""
    tc = data["top_competitor"]
    fgk = data["fastest_kw"]
    ss = data["sentiment"]
    ds = data["demand"]
    comps = data["competitors"]
    kw_ytd = data["kw_ytd"]
    top_news = data["top_news"]

    comp_table = "\n".join(
        f"| {c['name']} | {c['market_share']}% | "
        f"{c['market_share'] - c['market_share_prev']:+.1f}pp | "
        f"${c['revenue_bn']}B | {c['pricing'].get('entry_plan', 'N/A')} |"
        for c in comps
    )

    kw_table = "\n".join(
        f"| {kw} | +{growth}% YTD |"
        for kw, growth in sorted(kw_ytd.items(), key=lambda x: x[1], reverse=True)
    )

    news_items = "\n".join(
        f"  - {a['title']} ({a['source']}, {a['published_at']})"
        for a in top_news
    )

    return textwrap.dedent(f"""
    You are a senior market research analyst producing a comprehensive report for: "{query}"

    Generate a full professional market research report in Markdown with these exact sections:
    # Market Research Report
    ## Executive Summary
    ## 1. Market Overview
    ## 2. Competitive Landscape
    ## 3. Customer Sentiment Analysis
    ## 4. Market Trend Analysis
    ## 5. Demand Forecast
    ## 6. Key Market Events & News
    ## 7. Actionable Recommendations
    ## 8. Disclaimer

    DATA TO INCORPORATE:

    COMPETITORS (sorted by share):
    | Company | Share | Change | Revenue | Entry Price |
    |---|---|---|---|---|
    {comp_table}

    SENTIMENT: {ss.get('positive_pct', 'N/A')}% positive, avg {ss.get('avg_rating', 'N/A')}/5 from {ss.get('total_reviews', 0)} reviews

    DEMAND: Highest demand — {ds.get('top_category', 'N/A')} in {ds.get('top_region', 'N/A')} (index: {ds.get('demand_index', 'N/A')}, YoY: +{ds.get('yoy_change', 'N/A')}%)

    KEYWORD TRENDS:
    | Keyword | YTD Growth |
    |---|---|
    {kw_table}

    RECENT NEWS:
    {news_items}

    Requirements:
    - Use all the data above — reference specific numbers
    - Section 8 Disclaimer must clearly state that forecasts are projections, not guarantees
    - Write professionally, ~800-1200 words total
    - Mark any forward-looking statements with [Forecast]
    """).strip()


def _markdown_to_pdf(markdown_text: str, output_path: Path) -> Path:
    """
    Convert a Markdown string to a PDF file using reportlab.

    Supports: headings (# ## ###), bold (**), bullet lists (- •), tables (|),
    and plain paragraphs. Returns the path to the written PDF.
    """
    try:
        from reportlab.lib.pagesizes import A4  # noqa: PLC0415
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle  # noqa: PLC0415
        from reportlab.lib.units import cm  # noqa: PLC0415
        from reportlab.lib import colors  # noqa: PLC0415
        from reportlab.platypus import (  # noqa: PLC0415
            SimpleDocTemplate, Paragraph, Spacer, HRFlowable, Table, TableStyle,
        )
    except ImportError:
        raise ImportError("reportlab is required for PDF generation. Run: pip install reportlab")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=2 * cm,
        leftMargin=2 * cm,
        topMargin=2 * cm,
        bottomMargin=2 * cm,
    )

    styles = getSampleStyleSheet()
    # Custom styles
    h1_style = ParagraphStyle("H1", parent=styles["Heading1"], fontSize=20, spaceAfter=12,
                               textColor=colors.HexColor("#1a3a6b"))
    h2_style = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=14, spaceAfter=8,
                               textColor=colors.HexColor("#2e5dad"), spaceBefore=14)
    h3_style = ParagraphStyle("H3", parent=styles["Heading3"], fontSize=12, spaceAfter=6,
                               textColor=colors.HexColor("#3b6fcc"), spaceBefore=10)
    body_style = ParagraphStyle("Body", parent=styles["Normal"], fontSize=10,
                                 spaceAfter=6, leading=16)
    bullet_style = ParagraphStyle("Bullet", parent=styles["Normal"], fontSize=10,
                                   spaceAfter=4, leftIndent=20, leading=14)
    meta_style = ParagraphStyle("Meta", parent=styles["Normal"], fontSize=8,
                                 textColor=colors.grey, spaceAfter=4)

    def _escape(text: str) -> str:
        """Escape XML special characters for ReportLab Paragraph."""
        return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def _md_inline(text: str) -> str:
        """Convert inline Markdown (**bold**, *italic*) to ReportLab XML."""
        text = _escape(text)
        text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
        text = re.sub(r"\*(.+?)\*", r"<i>\1</i>", text)
        return text

    story: list = []
    lines = markdown_text.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i].rstrip()

        # Blank line
        if not line:
            story.append(Spacer(1, 4))
            i += 1
            continue

        # H1
        if line.startswith("# "):
            story.append(Paragraph(_md_inline(line[2:]), h1_style))
            story.append(HRFlowable(width="100%", thickness=1,
                                     color=colors.HexColor("#1a3a6b"), spaceAfter=6))
            i += 1
            continue

        # H2
        if line.startswith("## "):
            story.append(Paragraph(_md_inline(line[3:]), h2_style))
            i += 1
            continue

        # H3
        if line.startswith("### "):
            story.append(Paragraph(_md_inline(line[4:]), h3_style))
            i += 1
            continue

        # Horizontal rule
        if line.startswith("---"):
            story.append(HRFlowable(width="100%", thickness=0.5,
                                     color=colors.lightgrey, spaceAfter=4))
            i += 1
            continue

        # Table (lines starting with |)
        if line.startswith("|"):
            table_lines = []
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i].strip())
                i += 1
            # Parse table — skip separator rows (|---|)
            rows = []
            for tl in table_lines:
                if re.match(r"^\|[-| ]+\|$", tl):
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                rows.append(cells)

            if rows:
                # Normalize column count
                max_cols = max(len(r) for r in rows)
                norm_rows = [r + [""] * (max_cols - len(r)) for r in rows]
                # Build reportlab table data with escaped text
                tbl_data = [
                    [Paragraph(_md_inline(cell), body_style) for cell in row]
                    for row in norm_rows
                ]
                tbl = Table(tbl_data, repeatRows=1, hAlign="LEFT")
                tbl.setStyle(TableStyle([
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2e5dad")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("ROWBACKGROUNDS", (0, 1), (-1, -1),
                     [colors.white, colors.HexColor("#f0f4fb")]),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#d0d7e8")),
                    ("TOPPADDING", (0, 0), (-1, -1), 4),
                    ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                    ("LEFTPADDING", (0, 0), (-1, -1), 6),
                ]))
                story.append(tbl)
                story.append(Spacer(1, 8))
            continue

        # Bullet point
        if line.startswith(("- ", "• ", "* ")):
            story.append(Paragraph("• " + _md_inline(line[2:]), bullet_style))
            i += 1
            continue

        # Numbered list
        if re.match(r"^\d+\. ", line):
            story.append(Paragraph(_md_inline(line), bullet_style))
            i += 1
            continue

        # Blockquote
        if line.startswith("> "):
            quote_style = ParagraphStyle(
                "Quote", parent=body_style, leftIndent=15,
                textColor=colors.HexColor("#555555"), borderPadding=4,
            )
            story.append(Paragraph(_md_inline(line[2:]), quote_style))
            i += 1
            continue

        # Regular paragraph
        story.append(Paragraph(_md_inline(line), body_style))
        i += 1

    # Footer
    story.append(Spacer(1, 20))
    story.append(HRFlowable(width="100%", thickness=0.5, color=colors.grey))
    story.append(Paragraph(
        f"Generated by Intelligent Market Research Assistant — IBM Granite Powered | "
        f"{datetime.now().strftime('%Y-%m-%d %H:%M')}",
        meta_style,
    ))

    doc.build(story)
    logger.info("PDF report written to %s", output_path)
    return output_path


def _run_report(query: str, llm: Any) -> str:
    """Generate a full market research report and export to PDF."""
    global _latest_pdf_path

    data = _gather_data_summary()
    prompt = _build_report_prompt(query, data)

    try:
        markdown_report = llm.invoke(prompt)
    except Exception as exc:  # noqa: BLE001
        logger.error("Report LLM call failed: %s", exc)
        markdown_report = "# Market Research Report\n\nReport generation failed. Please try again."

    # Generate PDF
    from utils.config import config  # noqa: PLC0415
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    pdf_path = config.exports_dir / f"{timestamp}_market_report.pdf"

    try:
        _markdown_to_pdf(markdown_report, pdf_path)
        _latest_pdf_path = pdf_path
        pdf_note = f"\n\n---\n📥 **PDF Report saved:** `{pdf_path.name}` — download via the button below."
    except Exception as exc:  # noqa: BLE001
        logger.error("PDF generation failed: %s", exc)
        _latest_pdf_path = None
        pdf_note = f"\n\n---\n⚠️ PDF generation failed: {exc}"

    return markdown_report + pdf_note


def get_report_tool(llm: Any) -> Tool:
    """Return a configured report generation Tool."""
    return Tool(
        name=_TOOL_NAME,
        description=_TOOL_DESCRIPTION,
        func=lambda query: _run_report(query, llm),
    )
