"""
dashboard/chat.py
──────────────────
AI Research Assistant chat tab.

Features:
  - Full message history via st.session_state
  - File upload (PDF, DOCX, TXT, PNG/JPG) → ingestion into ChromaDB
  - OCR disclaimer banner for image uploads
  - Voice input via Web Speech API (Chrome/Edge)
  - Agent routing with ExplainableResponse rendering
  - PDF download button when report_generation tool produces a PDF
"""

from __future__ import annotations

from typing import Any


# ── Suggested queries shown as quick-start buttons ───────────────────────────
_SUGGESTED_QUERIES = [
    "Analyse customer sentiment across all products",
    "Compare competitor market share and strategy",
    "What are the top market trends in 2024?",
    "Analyse demand by region and category",
    "Summarise recent market news",
    "Generate a full market research report",
]


def _init_session(st: Any) -> None:
    """Initialise session state keys for the chat tab."""
    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []
    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = []
    if "chat_input_prefill" not in st.session_state:
        st.session_state.chat_input_prefill = ""


def _render_file_uploader(st: Any, vsm: Any) -> None:
    """Render the document upload section and ingest uploaded files."""
    from rag.ingestion import ingest_file, is_ocr_result  # noqa: PLC0415

    st.markdown("#### 📎 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload market reports, PDFs, or images",
        type=["pdf", "docx", "txt", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="chat_file_uploader",
        help="Supported: PDF, DOCX, TXT (text extraction), PNG/JPG (OCR text extraction only)",
    )

    if uploaded_files:
        for uploaded_file in uploaded_files:
            fname = uploaded_file.name
            if fname not in st.session_state.uploaded_docs:
                with st.spinner(f"Processing {fname}..."):
                    try:
                        file_bytes = uploaded_file.read()
                        text = ingest_file(fname, file_bytes)

                        # Show OCR disclaimer prominently for images
                        if is_ocr_result(text):
                            st.info(
                                "🔍 **Image processed via OCR** — text-only extraction. "
                                "The system cannot interpret charts, diagrams, or visual layouts. "
                                "Only text visible in the image was extracted."
                            )

                        chunks_added = vsm.add_documents(fname, text)
                        st.session_state.uploaded_docs.append(fname)
                        st.success(f"✅ **{fname}** indexed — {chunks_added} chunks added to knowledge base")

                    except ValueError as e:
                        st.error(f"⚠️ Unsupported file type: {e}")
                    except Exception as e:  # noqa: BLE001
                        st.error(f"⚠️ Failed to process {fname}: {e}")

    if st.session_state.uploaded_docs:
        st.caption(f"📚 Indexed documents: {', '.join(st.session_state.uploaded_docs)}")


def _render_voice_input(st: Any) -> str | None:
    """Render voice input component and return transcript if available."""
    from utils.voice_input import render_voice_input  # noqa: PLC0415
    return render_voice_input(st)


def _render_message_history(st: Any) -> None:
    """Render all previous chat messages."""
    for msg in st.session_state.chat_messages:
        role = msg["role"]
        content = msg["content"]
        with st.chat_message(role):
            st.markdown(content)

            # Render explainability card if present
            if role == "assistant" and "response_obj" in msg:
                resp = msg["response_obj"]
                _render_response_extras(st, resp)


def _render_response_extras(st: Any, resp: Any) -> None:
    """Render tool usage, provenance badges, and PDF download for a response."""
    from utils.explainability import render_explainability_card  # noqa: PLC0415

    # Render provenance expander and forecast disclaimer
    if resp.tool_calls or resp.has_retrieved or resp.has_forecast:
        with st.expander("🔍 How was this answer generated?", expanded=False):
            if resp.tool_calls:
                tool_badges = " · ".join(f"`{t}`" for t in resp.tool_calls)
                st.markdown(f"**Tools used:** {tool_badges}")

            labels: list[str] = []
            if resp.has_retrieved:
                labels.append("📄 **Retrieved Evidence** — grounded in indexed documents")
            if resp.has_forecast:
                labels.append("🔮 **Forecast / Interpretation** — contains projections (not guaranteed)")
            if not resp.has_retrieved:
                labels.append("🤖 **AI Analysis** — generated from structured data + LLM reasoning")
            for lbl in labels:
                st.markdown(f"- {lbl}")

            if resp.sources:
                st.markdown("**Sources:**")
                for src in resp.sources:
                    st.markdown(f"  - 📄 `{src}`")

    if resp.has_forecast:
        st.warning(
            "🔮 **Forecast Disclaimer:** Contains trend-based projections. "
            "Not guaranteed outcomes."
        )

    # PDF download button
    if resp.pdf_path and resp.pdf_path.exists():
        try:
            pdf_bytes = resp.pdf_path.read_bytes()
            st.download_button(
                label="📥 Download PDF Report",
                data=pdf_bytes,
                file_name=resp.pdf_path.name,
                mime="application/pdf",
                key=f"pdf_{resp.pdf_path.stem}",
            )
        except OSError:
            st.caption("⚠️ PDF could not be read.")


def render_chat(st: Any, agent: Any, vsm: Any) -> None:
    """
    Render the full AI Research Assistant chat interface.

    Parameters
    ----------
    st : streamlit module
    agent : ResearchAgent instance
    vsm : VectorStoreManager instance
    """
    _init_session(st)

    st.markdown("## 💬 AI Research Assistant")
    st.markdown("*Ask any market research question — the agent will select the best tools automatically*")
    st.markdown("---")

    # ── Sidebar-like controls ─────────────────────────────────────────────────
    with st.expander("📎 Upload Documents & Tools", expanded=False):
        _render_file_uploader(st, vsm)

        st.markdown("---")
        st.markdown("#### 🎤 Voice Input")
        voice_transcript = _render_voice_input(st)
        if voice_transcript:
            st.session_state.chat_input_prefill = voice_transcript
            st.success(f"🎤 Transcript: *{voice_transcript}*")

    # ── Suggested queries ─────────────────────────────────────────────────────
    if not st.session_state.chat_messages:
        st.markdown("#### 💡 Quick Start — click a question below:")
        cols = st.columns(3)
        for i, suggestion in enumerate(_SUGGESTED_QUERIES):
            with cols[i % 3]:
                if st.button(suggestion, key=f"suggest_{i}", use_container_width=True):
                    st.session_state.chat_input_prefill = suggestion

    # ── Message history ───────────────────────────────────────────────────────
    _render_message_history(st)

    # ── Chat input ────────────────────────────────────────────────────────────
    prefill = st.session_state.get("chat_input_prefill", "")
    user_input = st.chat_input(
        "Ask a market research question... (e.g. 'Compare competitor strategies')",
        key="chat_main_input",
    )

    # Handle prefill from voice/quick-start buttons
    if prefill and not user_input:
        user_input = prefill
        st.session_state.chat_input_prefill = ""

    if user_input:
        user_input = user_input.strip()
        if not user_input:
            return

        # Add user message to history
        st.session_state.chat_messages.append({
            "role": "user",
            "content": user_input,
        })

        # Display user message
        with st.chat_message("user"):
            st.markdown(user_input)

        # ── Run agent ─────────────────────────────────────────────────────────
        with st.chat_message("assistant"):
            with st.spinner("🔍 Analysing market data..."):
                try:
                    response = agent.run(user_input)
                except Exception as exc:  # noqa: BLE001
                    from utils.explainability import format_response  # noqa: PLC0415
                    response = format_response(
                        f"An error occurred: {exc}\n\nPlease try again or rephrase your question.",
                        error=str(exc),
                    )

            # Render answer
            if response.is_error:
                st.error(f"⚠️ {response.answer}")
            else:
                st.markdown(response.answer)
                _render_response_extras(st, response)

        # Save assistant message with response object
        st.session_state.chat_messages.append({
            "role": "assistant",
            "content": response.answer,
            "response_obj": response,
        })

    # ── Chat controls ─────────────────────────────────────────────────────────
    if st.session_state.chat_messages:
        st.markdown("---")
        col_a, col_b = st.columns([1, 4])
        with col_a:
            if st.button("🗑️ Clear chat", key="clear_chat"):
                st.session_state.chat_messages = []
                st.rerun()

    st.markdown("---")
    st.caption(
        "🤖 Powered by **IBM Granite** (watsonx.ai) · "
        "📄 RAG via **ChromaDB** · "
        "🔍 Data: News, Competitors, Reviews, Trends · "
        "🎤 Voice: Web Speech API (Chrome/Edge only)"
    )
