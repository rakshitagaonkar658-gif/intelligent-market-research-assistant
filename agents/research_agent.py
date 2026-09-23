"""
agents/research_agent.py
─────────────────────────
LangChain ReAct Agent for intelligent market research.

The agent uses the Reasoning + Acting (ReAct) pattern to autonomously:
  1. Decide which tools to call based on the user's question
  2. Observe the tool output
  3. Reason about whether more information is needed
  4. Produce a final answer with proper source labelling

All responses are wrapped in ExplainableResponse via utils/explainability.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# ── ReAct system prompt ───────────────────────────────────────────────────────
_SYSTEM_PROMPT = """You are an expert market research analyst powered by IBM Granite AI.
Your role is to provide comprehensive, data-driven market intelligence by using the available tools.

AVAILABLE TOOLS AND WHEN TO USE THEM:
- document_retrieval: Use when asked about specific uploaded documents or reports
- news_research: Use for recent news, current events, or latest market developments
- sentiment_analysis: Use for customer opinions, product reviews, or satisfaction analysis
- competitor_comparison: Use for competitive landscape, market share, or rival company analysis
- trend_analysis: Use for keyword trends, growth patterns, or topic momentum
- demand_analysis: Use for demand forecasts, regional demand, or market sizing
- report_generation: Use ONLY when explicitly asked to generate/create a full report

LABELLING RULES — always include these labels in your Final Answer:
- Start factual data points with: [Retrieved Evidence]
- Start AI interpretations with: [AI Analysis]
- Start projections/forecasts with: [Forecast / Interpretation]
- Always note: "Forecasts are projections, not guaranteed outcomes"

RESPONSE FORMAT:
- Be concise but thorough
- Use bullet points or numbered lists where helpful
- Always cite which tool(s) produced the data
- If a tool returns an error, acknowledge it and provide what you can

You MUST follow this exact format for every step:

Thought: I need to determine which tool will best answer this question.
Action: <tool_name>
Action Input: <input_to_tool>
Observation: <tool_output>
... (repeat Thought/Action/Observation as needed, max 5 iterations)
Thought: I now have enough information to answer.
Final Answer: <your_complete_answer>
"""


class ResearchAgent:
    """
    Wrapper around a LangChain AgentExecutor for market research tasks.

    Provides a clean `.run(query)` interface that returns an `ExplainableResponse`
    regardless of whether the underlying LLM is Granite or MockLLM.
    """

    def __init__(
        self,
        llm: Optional[Any] = None,
        vector_store: Optional[Any] = None,
    ) -> None:
        """
        Parameters
        ----------
        llm : BaseLLM | None
            Pass an LLM instance or leave None to auto-resolve via get_llm().
        vector_store : VectorStoreManager | None
            Pass a VectorStoreManager or leave None to create a new one.
        """
        from models import get_llm  # deferred
        from rag.vector_store import VectorStoreManager  # deferred
        from tools import get_all_tools  # deferred

        self._llm = llm or get_llm()
        self._vector_store = vector_store or VectorStoreManager()
        self._tools = get_all_tools(self._llm, self._vector_store)
        self._tool_map = {t.name: t for t in self._tools}
        self._executor = self._build_executor()

        logger.info(
            "ResearchAgent initialised — LLM: %s, tools: %s",
            type(self._llm).__name__,
            [t.name for t in self._tools],
        )

    # ── Agent construction ────────────────────────────────────────────────────

    def _build_executor(self) -> Any:
        """
        Build an LCEL-based agent chain for LangChain v1.x.

        Returns None if construction fails — the run() method falls back
        to direct-tool routing in that case.
        """
        try:
            return self._build_lcel_chain()
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "LCEL chain build failed (%s) — using direct-tool mode", exc
            )
            return None

    def _build_lcel_chain(self) -> Any:
        """Build a simple LCEL routing chain compatible with LangChain v1.x."""
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder  # noqa: PLC0415
        from langchain_core.runnables import RunnableLambda  # noqa: PLC0415

        tool_descriptions = "\n".join(
            f"- {t.name}: {t.description}" for t in self._tools
        )
        # We use a simple prompt + LLM chain; tool selection is done by
        # _select_tool in the direct routing path for full reliability.
        # The LCEL chain here provides structured prompting for live Granite mode.
        prompt = ChatPromptTemplate.from_messages([
            ("system", _SYSTEM_PROMPT + "\n\nAvailable tools:\n" + tool_descriptions),
            ("human", "{input}"),
        ])

        chain = prompt | self._llm
        return chain

    # ── Public API ────────────────────────────────────────────────────────────

    def run(self, query: str) -> Any:
        """
        Run the research agent for a given query.

        Parameters
        ----------
        query : str
            The user's research question.

        Returns
        -------
        ExplainableResponse
            A structured response with provenance metadata.
        """
        from utils.explainability import format_response  # noqa: PLC0415
        from tools.report_tool import get_latest_pdf_path  # noqa: PLC0415

        if not query or not query.strip():
            return format_response(
                "Please enter a question to get started.",
                error="Empty query"
            )

        # ── Direct-tool routing (primary path — reliable in all LangChain versions) ──
        # Routes query to the best matching tool, invokes it, wraps in ExplainableResponse.
        # The LCEL chain (_executor) is available as an enhancement for live Granite mode
        # but direct routing is the guaranteed path for demos and offline use.
        return self._run_direct(query)

    def _run_direct(self, query: str) -> Any:
        """
        Direct tool routing fallback — bypasses AgentExecutor.

        Selects the most appropriate tool based on keyword matching in the query,
        invokes it, and wraps the result in an ExplainableResponse.
        """
        from utils.explainability import format_response  # noqa: PLC0415
        from tools.report_tool import get_latest_pdf_path  # noqa: PLC0415

        query_lower = query.lower()

        # Route to best tool
        tool_name = self._select_tool(query_lower)
        tool = self._tool_map.get(tool_name)

        if not tool:
            return format_response(
                "I couldn't determine which analysis to perform. "
                "Try asking about competitors, trends, sentiment, news, or demand.",
                tool_calls=[],
            )

        try:
            answer = tool.func(query)
        except Exception as exc:  # noqa: BLE001
            logger.error("Direct tool call failed for %s: %s", tool_name, exc)
            return format_response(
                f"The {tool_name} tool encountered an error: {exc}",
                error=str(exc),
            )

        pdf_path = get_latest_pdf_path() if tool_name == "report_generation" else None

        return format_response(
            answer=answer,
            tool_calls=[tool_name],
            pdf_path=pdf_path,
        )

    def _select_tool(self, query_lower: str) -> str:
        """Select the best tool name based on query keywords."""
        if any(w in query_lower for w in ["full report", "generate report", "create report", "write report", "research report"]):
            return "report_generation"
        if any(w in query_lower for w in ["sentiment", "review", "customer", "opinion", "rating", "feedback"]):
            return "sentiment_analysis"
        if any(w in query_lower for w in ["competitor", "competition", "rival", "market share", "versus", " vs "]):
            return "competitor_comparison"
        if any(w in query_lower for w in ["trend", "keyword", "growing", "search volume", "momentum"]):
            return "trend_analysis"
        if any(w in query_lower for w in ["demand", "demand index", "demand forecast", "regional demand"]):
            return "demand_analysis"
        if any(w in query_lower for w in ["news", "article", "headline", "recent", "latest", "current event"]):
            return "news_research"
        if any(w in query_lower for w in ["document", "report", "upload", "file", "what does", "according to"]):
            return "document_retrieval"
        # Default: try RAG first, then generic LLM
        return "document_retrieval"

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _extract_sources(self, answer: str, intermediate_steps: list) -> list[str]:
        """Extract source names from answer text and intermediate step outputs."""
        sources: list[str] = []

        # From RAG output patterns like "[Source 1: filename]"
        found = re.findall(r"\[Source \d+: ([^\]]+)\]", answer)
        sources.extend(found)

        # From intermediate step tool outputs
        for step in intermediate_steps:
            if len(step) >= 2 and isinstance(step[1], str):
                obs = step[1]
                found_obs = re.findall(r"\[Source \d+: ([^\]]+)\]", obs)
                sources.extend(found_obs)

        return list(dict.fromkeys(sources))  # deduplicate preserving order

    def get_tool_names(self) -> list[str]:
        """Return the names of all registered tools."""
        return [t.name for t in self._tools]

    def get_llm_type(self) -> str:
        """Return a human-readable label for the active LLM."""
        llm_type = getattr(self._llm, "_llm_type", type(self._llm).__name__)
        if llm_type == "mock":
            return "Demo Mode (MockLLM)"
        return f"IBM Granite ({llm_type})"
