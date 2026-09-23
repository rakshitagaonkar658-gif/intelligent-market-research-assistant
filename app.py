"""
app.py
───────
Intelligent Market Research Assistant — Streamlit entry point.

Run with:
    streamlit run app.py

The application auto-detects demo mode when WATSONX_API_KEY is absent.
See .env.example for all configuration options.
"""

from __future__ import annotations

import logging
import sys

# ── Patch for ChromaDB on Python 3.14+ (pysqlite3 not required on newer builds)
try:
    import pysqlite3  # type: ignore
    sys.modules["sqlite3"] = pysqlite3
except ImportError:
    pass  # pysqlite3 not installed — use built-in sqlite3

import streamlit as st

# ── Page config must be FIRST Streamlit call ──────────────────────────────────
st.set_page_config(
    page_title="Intelligent Market Research Assistant",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "About": (
            "# Intelligent Market Research Assistant\n"
            "Agentic AI powered by IBM Granite via watsonx.ai · "
            "RAG with ChromaDB · Streamlit UI\n\n"
            "Problem Statement No. 6 — College Project"
        )
    },
)

# ── Logging setup ─────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Custom CSS theming ────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stApp"] {
    background-color: #f7f8fa;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0a1628 0%, #1a3a6b 60%, #2e5dad 100%);
    border-right: 1px solid #2e5dad;
}
[data-testid="stSidebar"] * {
    color: #e8edf5 !important;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #ffffff !important;
}
[data-testid="stSidebar"] .stMetric label {
    color: #a0b4d0 !important;
}
[data-testid="stSidebar"] .stMetric [data-testid="stMetricValue"] {
    color: #ffffff !important;
    font-size: 1.1rem !important;
}

/* ── Header banner ── */
.main-header {
    background: linear-gradient(135deg, #1a3a6b 0%, #2e5dad 50%, #3b82d4 100%);
    padding: 18px 24px 14px 24px;
    border-radius: 12px;
    margin-bottom: 16px;
    color: white;
    box-shadow: 0 2px 12px rgba(46, 93, 173, 0.3);
}
.main-header h1 { color: white !important; margin: 0 0 4px 0; font-size: 1.6rem; }
.main-header p  { color: #c8d8f0 !important; margin: 0; font-size: 0.9rem; }

/* ── KPI metric cards ── */
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #e5e7eb;
    border-radius: 10px;
    padding: 12px 16px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
}
[data-testid="stMetricValue"]  { color: #1a3a6b; font-weight: 700; }
[data-testid="stMetricDelta"]  { font-size: 0.8rem; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: white;
    border-radius: 10px 10px 0 0;
    border-bottom: 2px solid #2e5dad;
    gap: 4px;
    padding: 4px 4px 0 4px;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px 8px 0 0;
    padding: 8px 18px;
    font-weight: 500;
    color: #57606a;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: #2e5dad !important;
    color: white !important;
}

/* ── Dataframes ── */
[data-testid="stDataFrame"] {
    border-radius: 8px;
    border: 1px solid #e5e7eb;
    overflow: hidden;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    border: 1px solid #e5e7eb;
    border-radius: 8px;
    background: white;
}

/* ── Chat messages ── */
[data-testid="stChatMessage"] {
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid #e5e7eb;
}

/* ── Buttons ── */
.stButton > button {
    border-radius: 8px;
    border: 1px solid #2e5dad;
    color: #2e5dad;
    font-weight: 500;
    transition: all 0.15s;
}
.stButton > button:hover {
    background: #2e5dad;
    color: white;
}

/* ── Success/Info/Warning/Error boxes ── */
[data-testid="stAlert"] { border-radius: 8px; }

/* ── Download button ── */
[data-testid="stDownloadButton"] > button {
    background: #27ae60;
    color: white;
    border: none;
    border-radius: 8px;
    font-weight: 600;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #1e8449;
}

/* ── Demo mode badge ── */
.demo-badge {
    background: #f39c12;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 4px;
}
.live-badge {
    background: #27ae60;
    color: white;
    padding: 4px 10px;
    border-radius: 12px;
    font-size: 0.78rem;
    font-weight: 700;
    display: inline-block;
    margin-top: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Session state initialisation ──────────────────────────────────────────────
def _init_session_state() -> None:
    defaults = {
        "agent": None,
        "vsm": None,
        "bootstrapped": False,
        "chat_messages": [],
        "uploaded_docs": [],
        "chat_input_prefill": "",
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


# ── Agent + vector store initialisation (cached) ─────────────────────────────
@st.cache_resource(show_spinner=False)
def _load_agent_and_vsm():
    """
    Load the ResearchAgent and VectorStoreManager once per process.
    Cached by st.cache_resource so subsequent page reloads reuse the instances.
    """
    from rag.vector_store import VectorStoreManager  # noqa: PLC0415
    from agents.research_agent import ResearchAgent  # noqa: PLC0415

    vsm = VectorStoreManager()
    vsm.bootstrap_sample_reports()
    agent = ResearchAgent(vector_store=vsm)
    return agent, vsm


# ── Sidebar ───────────────────────────────────────────────────────────────────
def _render_sidebar(config, agent, vsm) -> None:
    with st.sidebar:
        # App branding
        st.markdown("""
<div style="text-align:center; padding: 8px 0 16px 0;">
  <div style="font-size:2.8rem;">📊</div>
  <div style="font-size:1.05rem; font-weight:700; letter-spacing:0.5px;">Market Research</div>
  <div style="font-size:0.82rem; opacity:0.75;">Intelligent Assistant</div>
</div>
""", unsafe_allow_html=True)

        # Mode badge
        if config.is_demo():
            st.markdown('<span class="demo-badge">⚠️ DEMO MODE</span>', unsafe_allow_html=True)
            st.caption("Set WATSONX_API_KEY in .env for live mode")
        else:
            st.markdown('<span class="live-badge">✅ LIVE MODE</span>', unsafe_allow_html=True)

        st.markdown("---")

        # Model info
        st.markdown("#### 🤖 AI Configuration")
        st.caption(f"**Model:** `{config.granite_model_id}`")
        st.caption(f"**LLM:** {agent.get_llm_type()}")
        st.caption(f"**Tools:** {len(agent.get_tool_names())}")

        st.markdown("---")

        # Vector store stats
        st.markdown("#### 📚 Knowledge Base")
        try:
            stats = vsm.get_collection_stats()
            st.metric("Indexed Chunks", stats["total_chunks"])
            if stats["sources"]:
                st.caption("**Indexed sources:**")
                for src in stats["sources"]:
                    st.caption(f"  📄 {src}")
        except Exception:  # noqa: BLE001
            st.caption("Knowledge base loading...")

        st.markdown("---")

        # Quick file upload in sidebar
        st.markdown("#### 📎 Quick Upload")
        from rag.ingestion import ingest_file, is_ocr_result  # noqa: PLC0415
        uploaded = st.file_uploader(
            "Upload a document",
            type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
            key="sidebar_uploader",
            help="PDF, DOCX, TXT — full text extraction; PNG/JPG — OCR text only",
        )
        if uploaded and uploaded.name not in st.session_state.get("uploaded_docs", []):
            with st.spinner(f"Processing {uploaded.name}..."):
                try:
                    text = ingest_file(uploaded.name, uploaded.read())
                    if is_ocr_result(text):
                        st.info("🔍 OCR extraction — text only, not vision analysis.")
                    n = vsm.add_documents(uploaded.name, text)
                    st.session_state.uploaded_docs.append(uploaded.name)
                    st.success(f"✅ {uploaded.name} — {n} chunks indexed")
                except Exception as exc:  # noqa: BLE001
                    st.error(f"⚠️ {exc}")

        st.markdown("---")

        # Session info
        st.markdown("#### 📈 Session Stats")
        st.caption(f"Chat messages: {len(st.session_state.get('chat_messages', []))}")
        st.caption(f"Uploaded docs: {len(st.session_state.get('uploaded_docs', []))}")

        st.markdown("---")
        st.caption("IBM Granite · LangChain · ChromaDB · Streamlit")
        st.caption("Problem Statement No. 6")


# ── Main header ───────────────────────────────────────────────────────────────
def _render_header(config) -> None:
    mode_str = "🟡 Demo Mode" if config.is_demo() else "🟢 Live — IBM Granite"
    st.markdown(f"""
<div class="main-header">
  <h1>📊 Intelligent Market Research Assistant</h1>
  <p>
    Agentic AI · IBM Granite · RAG (ChromaDB) · LangChain · Streamlit &nbsp;|&nbsp;
    {mode_str}
  </p>
</div>
""", unsafe_allow_html=True)


# ── Main app ──────────────────────────────────────────────────────────────────
def main() -> None:
    _init_session_state()

    from utils.config import config  # noqa: PLC0415

    # Demo mode banner
    if config.is_demo():
        st.warning(
            "⚠️ **Demo Mode Active** — Running with MockLLM and bundled sample data. "
            "Add `WATSONX_API_KEY` and `WATSONX_PROJECT_ID` to `.env` for live IBM Granite access.",
            icon="⚠️",
        )

    # Load agent + VSM (cached — only runs once per process)
    with st.spinner("🚀 Initialising AI Research Agent..."):
        try:
            agent, vsm = _load_agent_and_vsm()
            st.session_state.agent = agent
            st.session_state.vsm = vsm
        except Exception as exc:  # noqa: BLE001
            st.error(f"❌ Failed to initialise agent: {exc}")
            st.stop()

    _render_header(config)
    _render_sidebar(config, agent, vsm)

    # ── 7-tab dashboard ───────────────────────────────────────────────────────
    from dashboard.overview import render_overview  # noqa: PLC0415
    from dashboard.trends import render_trends  # noqa: PLC0415
    from dashboard.competitors import render_competitors  # noqa: PLC0415
    from dashboard.sentiment import render_sentiment  # noqa: PLC0415
    from dashboard.demand import render_demand  # noqa: PLC0415
    from dashboard.alerts import render_alerts  # noqa: PLC0415
    from dashboard.chat import render_chat  # noqa: PLC0415

    tabs = st.tabs([
        "📊 Market Overview",
        "📈 Trend Analysis",
        "🏢 Competitor Comparison",
        "😊 Sentiment Analysis",
        "📉 Demand Forecast",
        "🔔 Alerts",
        "💬 AI Research Assistant",
    ])

    with tabs[0]:
        render_overview(st)
    with tabs[1]:
        render_trends(st)
    with tabs[2]:
        render_competitors(st)
    with tabs[3]:
        render_sentiment(st)
    with tabs[4]:
        render_demand(st)
    with tabs[5]:
        render_alerts(st)
    with tabs[6]:
        render_chat(st, agent, vsm)


if __name__ == "__main__":
    main()
