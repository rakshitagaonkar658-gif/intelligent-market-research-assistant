"""
utils/explainability.py
────────────────────────
Provenance labelling and rendering for all AI-generated responses.

Every insight shown to users is tagged as one of:
  📄 Retrieved Evidence   — text pulled from indexed documents via RAG
  🤖 AI Analysis          — LLM-generated interpretation of evidence/data
  🔮 Forecast/Interpretation — forward-looking projections (not guaranteed facts)

Usage
-----
    from utils.explainability import format_response, render_explainability_card

    response = format_response(answer_text, tool_calls=["sentiment_analysis"])
    render_explainability_card(st, response)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional


# ── Tag enum ─────────────────────────────────────────────────────────────────

class ExplainabilityTag(str, Enum):
    RETRIEVED = "retrieved"
    AI_ANALYSIS = "ai_analysis"
    FORECAST = "forecast"


# Icons and display labels
_TAG_META = {
    ExplainabilityTag.RETRIEVED: {
        "icon": "📄",
        "label": "Retrieved Evidence",
        "color": "blue",
        "streamlit_fn": "info",
    },
    ExplainabilityTag.AI_ANALYSIS: {
        "icon": "🤖",
        "label": "AI Analysis",
        "color": "green",
        "streamlit_fn": "success",
    },
    ExplainabilityTag.FORECAST: {
        "icon": "🔮",
        "label": "Forecast / Interpretation",
        "color": "orange",
        "streamlit_fn": "warning",
    },
}

# Tool name → primary tag mapping
_TOOL_TAGS: dict[str, ExplainabilityTag] = {
    "document_retrieval": ExplainabilityTag.RETRIEVED,
    "news_research": ExplainabilityTag.AI_ANALYSIS,
    "sentiment_analysis": ExplainabilityTag.AI_ANALYSIS,
    "competitor_comparison": ExplainabilityTag.AI_ANALYSIS,
    "trend_analysis": ExplainabilityTag.AI_ANALYSIS,
    "demand_analysis": ExplainabilityTag.FORECAST,
    "report_generation": ExplainabilityTag.AI_ANALYSIS,
}


# ── Data structures ───────────────────────────────────────────────────────────

def tag_text(text: str, tag: ExplainabilityTag) -> dict[str, Any]:
    """Wrap a text string with a provenance tag."""
    meta = _TAG_META[tag]
    return {
        "text": text,
        "tag": tag,
        "icon": meta["icon"],
        "label": meta["label"],
    }


@dataclass
class ExplainableResponse:
    """Structured response with provenance metadata for UI rendering."""

    answer: str
    """The main answer text (may contain Markdown)."""

    tool_calls: list[str] = field(default_factory=list)
    """Names of tools invoked to produce this answer."""

    sources: list[str] = field(default_factory=list)
    """Document/data source names referenced in the answer."""

    primary_tag: ExplainabilityTag = ExplainabilityTag.AI_ANALYSIS
    """The dominant provenance tag for this response."""

    has_forecast: bool = False
    """True if the response contains forward-looking projections."""

    has_retrieved: bool = False
    """True if the response is grounded in retrieved document evidence."""

    pdf_path: Optional[Path] = None
    """Path to a generated PDF report, if applicable."""

    error: Optional[str] = None
    """Error message if the agent failed."""

    @property
    def is_error(self) -> bool:
        return self.error is not None

    @property
    def tag_icon(self) -> str:
        return _TAG_META[self.primary_tag]["icon"]

    @property
    def tag_label(self) -> str:
        return _TAG_META[self.primary_tag]["label"]


# ── Factory ───────────────────────────────────────────────────────────────────

def format_response(
    answer: str,
    tool_calls: Optional[list[str]] = None,
    sources: Optional[list[str]] = None,
    pdf_path: Optional[Path] = None,
    error: Optional[str] = None,
) -> ExplainableResponse:
    """
    Build an ExplainableResponse from raw agent output.

    Parameters
    ----------
    answer : str
        The agent's final answer text.
    tool_calls : list[str]
        Names of tools that were invoked.
    sources : list[str]
        Document source names (extracted from RAG results).
    pdf_path : Path | None
        Path to generated PDF if report tool was used.
    error : str | None
        Error message if generation failed.
    """
    tool_calls = tool_calls or []
    sources = sources or []

    # Determine primary tag from tools used
    primary_tag = ExplainabilityTag.AI_ANALYSIS
    if "document_retrieval" in tool_calls:
        primary_tag = ExplainabilityTag.RETRIEVED
    elif "demand_analysis" in tool_calls:
        primary_tag = ExplainabilityTag.FORECAST

    # Detect forecast content via text patterns
    forecast_patterns = [
        r"forecast", r"projection", r"projected", r"expected to",
        r"anticipated", r"estimated to", r"\[forecast\]", r"not guaranteed",
        r"trend-based", r"directional indicator",
    ]
    has_forecast = any(
        re.search(pat, answer, re.IGNORECASE) for pat in forecast_patterns
    )

    has_retrieved = (
        "document_retrieval" in tool_calls
        or bool(sources)
        or "retrieved" in answer.lower()[:200]
    )

    # Extract sources from answer text if not provided explicitly
    if not sources:
        # Try to find "Source X: filename" patterns from RAG output
        found = re.findall(r"\[Source \d+: ([^\]]+)\]", answer)
        sources = list(dict.fromkeys(found))  # deduplicate preserving order

    return ExplainableResponse(
        answer=answer,
        tool_calls=tool_calls,
        sources=sources,
        primary_tag=primary_tag,
        has_forecast=has_forecast,
        has_retrieved=has_retrieved,
        pdf_path=pdf_path,
        error=error,
    )


# ── Streamlit renderer ────────────────────────────────────────────────────────

def render_explainability_card(st_module: Any, response: ExplainableResponse) -> None:
    """
    Render an ExplainableResponse as a structured, colour-coded card in Streamlit.

    Parameters
    ----------
    st_module : streamlit module
        Pass ``st`` directly — avoids a top-level streamlit import.
    response : ExplainableResponse
        The response to render.
    """
    st = st_module

    # ── Error state ───────────────────────────────────────────────────────────
    if response.is_error:
        st.error(f"⚠️ Agent Error\n\n{response.error}")
        return

    # ── Main answer ───────────────────────────────────────────────────────────
    st.markdown(response.answer)

    # ── Provenance badges ─────────────────────────────────────────────────────
    if response.tool_calls or response.has_retrieved or response.has_forecast:
        with st.expander("🔍 How was this answer generated?", expanded=False):

            # Tool calls made
            if response.tool_calls:
                tool_badges = " · ".join(
                    f"`{t}`" for t in response.tool_calls
                )
                st.markdown(f"**Tools used:** {tool_badges}")

            # Evidence labels
            labels_shown: list[str] = []
            if response.has_retrieved:
                labels_shown.append("📄 **Retrieved Evidence** — answer is grounded in indexed documents")
            if response.has_forecast:
                labels_shown.append("🔮 **Forecast / Interpretation** — contains forward-looking projections (not guaranteed facts)")
            if not response.has_retrieved:
                labels_shown.append("🤖 **AI Analysis** — generated from structured data and LLM reasoning")

            for label in labels_shown:
                st.markdown(f"- {label}")

            # Sources
            if response.sources:
                st.markdown("**Sources consulted:**")
                for src in response.sources:
                    st.markdown(f"  - 📄 `{src}`")

    # ── Forecast disclaimer ───────────────────────────────────────────────────
    if response.has_forecast:
        st.warning(
            "🔮 **Forecast Disclaimer:** This response contains projections or trend-based interpretations. "
            "These are analytical estimates and should not be treated as guaranteed outcomes."
        )

    # ── Retrieved evidence highlight ─────────────────────────────────────────
    if response.has_retrieved and not response.has_forecast:
        st.info(
            "📄 **Source Grounded:** This answer is based on retrieved document evidence "
            "from your indexed reports."
        )

    # ── PDF download button ───────────────────────────────────────────────────
    if response.pdf_path and response.pdf_path.exists():
        try:
            pdf_bytes = response.pdf_path.read_bytes()
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=response.pdf_path.name,
                mime="application/pdf",
                key=f"pdf_dl_{response.pdf_path.stem}",
            )
        except OSError:
            st.caption("⚠️ PDF file could not be read for download.")
