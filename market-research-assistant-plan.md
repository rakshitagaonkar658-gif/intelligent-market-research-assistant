# Intelligent Market Research Assistant — Implementation Plan
<!-- confirmed: 2025-07 — all modifications applied, ready for implementation -->

## Top-Level Overview

Build a **production-style agentic AI application** that acts as an intelligent market research assistant.
The system fuses multiple market-data sources (news, uploaded industry reports, competitor data, customer reviews, trend data),
processes them through a RAG pipeline backed by ChromaDB, and exposes a LangChain agent powered by IBM Granite
(`ibm/granite-3-8b-instruct`) via watsonx.ai. A professional Streamlit dashboard surfaces seven analysis views plus
an AI Research Assistant chat.

**Key constraints:**
- IBM Granite is the only LLM; credentials come entirely from environment variables.
- A `MockLLM` fallback lets graders run the full UI with sample data when no credentials are present.
- All API keys (watsonx, NewsAPI) are read from `.env`; never hard-coded.
- ChromaDB runs fully local — zero extra infrastructure.
- Embeddings use `ibm/slate-30m-english-rtrvr` via watsonx; fall back to `sentence-transformers/all-MiniLM-L6-v2` when credentials are absent.
- Bundled sample JSON/PDF files guarantee a demo even without external APIs.

---

## Folder Structure

```
intelligent-market-research-assistant/
├── app.py                         # Streamlit entry-point
├── .env.example                   # Template for environment variables
├── requirements.txt
├── README.md
│
├── agents/
│   ├── __init__.py
│   └── research_agent.py          # LangChain AgentExecutor + tool wiring
│
├── tools/
│   ├── __init__.py
│   ├── rag_tool.py                # RAG document retrieval tool
│   ├── news_tool.py               # NewsAPI + sample fallback
│   ├── sentiment_tool.py          # TextBlob + Granite sentiment analysis
│   ├── competitor_tool.py         # Competitor comparison tool
│   ├── trend_tool.py              # Trend analysis tool
│   ├── demand_tool.py             # Market demand analysis tool
│   └── report_tool.py             # Report generation tool
│
├── rag/
│   ├── __init__.py
│   ├── ingestion.py               # Document parsing (PDF, DOCX, TXT, image OCR)
│   ├── chunker.py                 # Text splitting
│   ├── embedder.py                # Embedding factory (watsonx / sentence-transformers)
│   └── vector_store.py            # ChromaDB wrapper (init, add, retrieve)
│
├── models/
│   ├── __init__.py
│   ├── granite_llm.py             # WatsonxLLM wrapper + env config
│   └── mock_llm.py                # Deterministic mock for demo/testing
│
├── dashboard/
│   ├── __init__.py
│   ├── overview.py                # Market Overview tab
│   ├── trends.py                  # Trend Analysis tab
│   ├── competitors.py             # Competitor Comparison tab
│   ├── sentiment.py               # Sentiment Analysis tab
│   ├── demand.py                  # Demand Forecast tab
│   ├── alerts.py                  # Alerts tab
│   └── chat.py                    # AI Research Assistant chat tab
│
├── utils/
│   ├── __init__.py
│   ├── config.py                  # Env-var loading + validation
│   ├── data_loader.py             # Loader for sample JSON data files
│   ├── explainability.py          # Tag and format retrieved vs AI-generated content
│   └── voice_input.py             # Streamlit component for Web Speech API
│
└── data/
    ├── sample_news.json
    ├── sample_competitors.json
    ├── sample_reviews.json
    ├── sample_trends.json
    ├── sample_demand.json
    └── sample_reports/
        ├── market_overview_report.txt
        └── tech_industry_report.txt
```

---

## Sub-Tasks

---

### Sub-Task 1 — Project Scaffold & Configuration

**Intent:** Establish the complete folder skeleton, environment configuration, dependencies list, and README before any logic is written. Every subsequent sub-task builds on this foundation.

**Expected Outcomes:**
- All folders and `__init__.py` files exist.
- `requirements.txt` lists every dependency with pinned or minimum versions.
- `.env.example` documents every environment variable.
- `README.md` describes setup, running, and demo mode.
- `utils/config.py` loads and validates env vars at startup with clear error messages.

**Todo List:**
1. Create all directories: `agents/`, `tools/`, `rag/`, `models/`, `dashboard/`, `utils/`, `data/`, `data/sample_reports/`.
2. Add `__init__.py` to each Python package directory.
3. Write `requirements.txt` with the following dependencies:
   - `streamlit>=1.32`
   - `langchain>=0.2`
   - `langchain-community>=0.2`
   - `langchain-ibm>=0.1`
   - `ibm-watsonx-ai>=0.2`
   - `chromadb>=0.5`
   - `sentence-transformers>=2.7`
   - `pypdf>=4.0`
   - `python-docx>=1.1`
   - `Pillow>=10.0`
   - `pytesseract>=0.3` (OCR for image inputs — text extraction only, not vision understanding)
   - `reportlab>=4.0`
   - `textblob>=0.18`
   - `plotly>=5.20`
   - `pandas>=2.0`
   - `python-dotenv>=1.0`
   - `requests>=2.31`
   - `newsapi-python>=0.2.7`
4. Write `.env.example` with these variables (all optional for demo mode):
   - `WATSONX_API_KEY`
   - `WATSONX_PROJECT_ID`
   - `WATSONX_URL` (default `https://us-south.ml.cloud.ibm.com`)
   - `GRANITE_MODEL_ID` (default `ibm/granite-3-8b-instruct`)
   - `NEWSAPI_KEY`
   - `DEMO_MODE` (`true` / `false`)
5. Write `utils/config.py`:
   - Load `.env` with `python-dotenv`.
   - Expose a `Config` dataclass or simple namespace.
   - Auto-detect demo mode: if `WATSONX_API_KEY` is missing, set `demo_mode=True` and log a warning.
   - Expose `is_demo()` helper used throughout the app.
6. Write `README.md` covering: prerequisites, `pip install -r requirements.txt`, `.env` setup, `streamlit run app.py`, and demo mode instructions.

**Relevant Context:** No existing code — greenfield.

**Status:** [x] done

---

### Sub-Task 2 — Sample / Demo Data

**Intent:** Create realistic bundled data files so every feature can be demonstrated without any external API call. These are the fallback for all tools.

**Expected Outcomes:**
- `data/sample_news.json` — 15+ news articles with `title`, `source`, `published_at`, `description`, `url` fields.
- `data/sample_competitors.json` — 5 competitor profiles with `name`, `market_share`, `strengths`, `weaknesses`, `recent_activities`, `pricing`.
- `data/sample_reviews.json` — 30+ customer reviews with `product`, `rating`, `review_text`, `date`, `sentiment` fields.
- `data/sample_trends.json` — Monthly trend data for 5 keywords over 12 months: `keyword`, `month`, `volume`, `growth_pct`.
- `data/sample_demand.json` — Demand indicators: `category`, `region`, `quarter`, `demand_index`, `yoy_change`.
- `data/sample_reports/market_overview_report.txt` — 500-word synthetic market overview.
- `data/sample_reports/tech_industry_report.txt` — 500-word synthetic tech industry summary.
- `utils/data_loader.py` — Functions: `load_news()`, `load_competitors()`, `load_reviews()`, `load_trends()`, `load_demand()`.

**Todo List:**
1. Write `data/sample_news.json` with 15 varied news items spanning AI, fintech, and retail industries.
2. Write `data/sample_competitors.json` with 5 tech-industry competitor profiles.
3. Write `data/sample_reviews.json` with 30 mixed-sentiment reviews.
4. Write `data/sample_trends.json` with 12-month monthly trend volumes for keywords: AI, Cloud, Fintech, E-commerce, Cybersecurity.
5. Write `data/sample_demand.json` with demand index data by region and quarter.
6. Write the two `.txt` report files in `data/sample_reports/`.
7. Write `utils/data_loader.py` with typed loader functions that return lists of dicts.

**Relevant Context:** `utils/config.py` from Sub-Task 1.

**Status:** [x] done

---

### Sub-Task 3 — LLM Layer (Granite + Mock)

**Intent:** Implement the IBM Granite LLM wrapper and the demo-mode `MockLLM` so all higher-level components can import a single `get_llm()` factory without knowing which backend is active.

**Expected Outcomes:**
- `models/granite_llm.py` — `GraniteLLM` class wrapping `langchain_ibm.WatsonxLLM` with model parameters loaded from `Config`.
- `models/mock_llm.py` — `MockLLM` class implementing the LangChain `LLM` interface; produces **data-driven structured responses** by loading sample JSON at call time, computing real statistics, and composing multi-paragraph analysis — not static keyword-matched strings.
- `models/__init__.py` — `get_llm()` factory that returns `GraniteLLM` in live mode and `MockLLM` in demo mode.
- No credentials appear anywhere except read from `Config`.

**Todo List:**
1. Write `models/granite_llm.py`:
   - Import `WatsonxLLM` from `langchain_ibm`.
   - Build `WatsonxLLM` with `model_id`, `url`, `apikey`, `project_id` from `Config`.
   - Wrap in a `get_granite_llm()` function.
   - Add try/except: if instantiation fails, raise a descriptive `RuntimeError` that mentions demo mode.
2. Write `models/mock_llm.py`:
   - Subclass `langchain_core.language_models.llms.LLM`.
   - Implement `_call(prompt, stop=None)` with **data-driven structured responses**.
   - Implement `_llm_type` property returning `"mock"`.
   - The mock must: (a) parse the incoming prompt to detect the task type (sentiment, competitor, trend, demand, report, news, rag); (b) load the relevant sample JSON data from `utils/data_loader.py`; (c) compute real statistics (e.g. average rating, top competitor by market share, fastest-growing keyword); (d) compose a structured, multi-paragraph response that references the actual sample data values — not static strings.
   - Include response builders: `_build_sentiment_response()`, `_build_competitor_response()`, `_build_trend_response()`, `_build_demand_response()`, `_build_report_response()`, `_build_news_response()`, `_build_rag_response()`, `_build_generic_response()`.
3. Write `models/__init__.py` with `get_llm()` that checks `Config.is_demo()`.

**Relevant Context:** `utils/config.py` from Sub-Task 1; LangChain `WatsonxLLM` API from `langchain-ibm`.

**Status:** [x] done

---

### Sub-Task 4 — RAG Pipeline

**Intent:** Implement the complete Retrieve-Augment-Generate pipeline: document ingestion, chunking, embedding, ChromaDB vector storage, and a retriever function used by the RAG tool.

**Expected Outcomes:**
- `rag/ingestion.py` — Parses PDF (pypdf), DOCX (python-docx), TXT, and PNG/JPG (pytesseract OCR) files into raw text.
- `rag/chunker.py` — Splits text using `RecursiveCharacterTextSplitter` with chunk size 512, overlap 64.
- `rag/embedder.py` — `get_embedder()` factory: tries `WatsonxEmbeddings` with `slate-30m` model; falls back to `HuggingFaceEmbeddings(all-MiniLM-L6-v2)` in demo mode.
- `rag/vector_store.py` — `VectorStoreManager` class: `add_documents(docs, source_name)`, `similarity_search(query, k=5)`, `get_collection_stats()`. Persists ChromaDB to `data/chroma_db/`.
- On startup, the two sample report TXT files are auto-ingested into ChromaDB if the collection is empty.

**Todo List:**
1. Write `rag/ingestion.py`:
   - `parse_pdf(file_bytes) -> str`
   - `parse_docx(file_bytes) -> str`
   - `parse_txt(file_bytes) -> str`
   - `parse_image(file_bytes) -> str` — pytesseract OCR; always prefix the returned text with `"[OCR-extracted text — image analysis is text-only, not vision-based]\n"`; graceful error if tesseract not installed
   - `ingest_file(file_name, file_bytes) -> str` dispatcher
2. Write `rag/chunker.py`:
   - `chunk_text(text, chunk_size=512, overlap=64) -> list[str]`
   - Uses LangChain `RecursiveCharacterTextSplitter`.
3. Write `rag/embedder.py`:
   - `get_embedder()` factory with watsonx / sentence-transformers fallback logic.
4. Write `rag/vector_store.py`:
   - Initialize `chromadb.PersistentClient` pointing at `data/chroma_db/`.
   - Use `langchain_community.vectorstores.Chroma`.
   - `VectorStoreManager` with `add_documents`, `similarity_search`, `get_collection_stats`.
5. Write `rag/__init__.py` exporting `VectorStoreManager` and `get_embedder`.
6. In `app.py` startup sequence, call `VectorStoreManager.bootstrap_sample_reports()` if collection is empty.

**Relevant Context:** `models/__init__.py` from Sub-Task 3; `rag/ingestion.py` dependencies: pypdf, python-docx, pytesseract.

**Status:** [x] done

---

### Sub-Task 5 — Agent Tools

**Intent:** Implement each of the seven LangChain tools the agent can invoke. Each tool has a clear description string (used by the LLM for tool selection), input schema, and output format. Every tool must handle demo mode gracefully using sample data.

**Expected Outcomes:**
- `tools/rag_tool.py` — `RAGTool`: retrieves top-5 chunks from ChromaDB, builds a context-augmented prompt, calls LLM, returns answer tagged with source chunks.
- `tools/news_tool.py` — `NewsTool`: calls NewsAPI if key present; otherwise returns from `sample_news.json`. Returns top 5 relevant articles summarised by LLM.
- `tools/sentiment_tool.py` — `SentimentTool`: runs TextBlob on `sample_reviews.json` for quantitative scores, then asks LLM for a qualitative summary. Returns sentiment distribution + narrative.
- `tools/competitor_tool.py` — `CompetitorTool`: loads `sample_competitors.json`, builds a comparison table, asks LLM for strategic insights.
- `tools/trend_tool.py` — `TrendTool`: loads `sample_trends.json`, computes growth rates, asks LLM for trend narrative.
- `tools/demand_tool.py` — `DemandTool`: loads `sample_demand.json`, asks LLM to interpret demand indicators.
- `tools/report_tool.py` — `ReportGenTool`: composes a full market research report by orchestrating other tools and asking LLM to synthesise a structured report. Returns both **Markdown** and generates a **downloadable PDF** via `reportlab` (stored in `data/exports/` and surfaced as a `st.download_button` in the chat tab).
- `tools/__init__.py` — exports `get_all_tools(llm, vector_store)` list.

**Todo List:**
1. Write `tools/rag_tool.py`:
   - Wrap as a LangChain `Tool` with name `"document_retrieval"`.
   - Description: retrieves information from uploaded market reports.
   - Input: plain question string.
   - Output: answer string + list of source citations (file name + chunk excerpt).
2. Write `tools/news_tool.py`:
   - Wrap as `Tool` with name `"news_research"`.
   - Try `NewsApiClient(api_key=Config.NEWSAPI_KEY).get_everything(q=query)`.
   - If key absent or call fails, load `data/sample_news.json` and filter by keyword match.
   - Ask LLM to summarise the top 5 results.
3. Write `tools/sentiment_tool.py`:
   - Wrap as `Tool` with name `"sentiment_analysis"`.
   - Load `sample_reviews.json`, compute per-review polarity with TextBlob.
   - Compute aggregate: positive/neutral/negative counts and mean score.
   - Pass raw stats + sample reviews to LLM for narrative summary.
4. Write `tools/competitor_tool.py`:
   - Wrap as `Tool` with name `"competitor_comparison"`.
   - Load `sample_competitors.json`.
   - Format as markdown table and pass to LLM with query for strategic insights.
5. Write `tools/trend_tool.py`:
   - Wrap as `Tool` with name `"trend_analysis"`.
   - Load `sample_trends.json`, compute MoM growth for each keyword.
   - Pass formatted data to LLM for narrative.
6. Write `tools/demand_tool.py`:
   - Wrap as `Tool` with name `"demand_analysis"`.
   - Load `sample_demand.json`, pass to LLM with query.
7. Write `tools/report_tool.py`:
   - Wrap as `Tool` with name `"report_generation"`.
   - Build a multi-section prompt: executive summary, market overview, competitors, trends, sentiment, demand, recommendations.
   - Return structured Markdown report as the tool output string.
   - After LLM returns the Markdown, call a `_markdown_to_pdf(markdown_text, output_path) -> Path` helper that uses `reportlab` to render the report as a PDF saved to `data/exports/<timestamp>_report.pdf`.
   - The `AgentResponse` for a report request carries an additional `pdf_path` field.
   - In `dashboard/chat.py`, when `pdf_path` is set, render a `st.download_button` below the chat message so the user can download the PDF instantly.
8. Write `tools/__init__.py` with `get_all_tools(llm, vector_store) -> list[Tool]`.

**Relevant Context:** `models/__init__.py` (Sub-Task 3); `rag/vector_store.py` (Sub-Task 4); `utils/data_loader.py` (Sub-Task 2). Add `reportlab>=4.0` to `requirements.txt`.

**Status:** [x] done

---

### Sub-Task 6 — Research Agent

**Intent:** Wire all tools into a LangChain `AgentExecutor` using the ReAct (Reasoning + Acting) pattern so the agent autonomously decides which tools to call to answer a user question.

**Expected Outcomes:**
- `agents/research_agent.py` — `ResearchAgent` class.
- Uses `create_react_agent` or `initialize_agent` with `ZERO_SHOT_REACT_DESCRIPTION`.
- System prompt instructs the agent to be a market research expert, always cite sources, and label outputs as *[Retrieved Evidence]*, *[AI Analysis]*, or *[Forecast / Interpretation]*.
- `ResearchAgent.run(query: str) -> AgentResponse` where `AgentResponse` includes: `answer`, `tool_calls_made`, `sources`.
- Agent gracefully handles LLM errors: returns a formatted error message with demo-mode hint.

**Todo List:**
1. Write `agents/research_agent.py`:
   - Import all tools from `tools/__init__.py` and LLM from `models/__init__.py`.
   - Write a custom system prompt emphasising: market research expertise, source labelling, and disclaimer for forecasts.
   - Instantiate `AgentExecutor` with `max_iterations=5`, `verbose=True`, `handle_parsing_errors=True`.
   - Expose `ResearchAgent.run(query)` that returns an `AgentResponse` dataclass.
   - Catch `OutputParserException` and timeout; return graceful fallback.
   - `AgentResponse` dataclass gains an optional `pdf_path: Optional[Path] = None` field populated when `report_generation` tool is used.
2. Write `agents/__init__.py` exporting `ResearchAgent`.

**Relevant Context:** `tools/__init__.py` (Sub-Task 5); `models/__init__.py` (Sub-Task 3); LangChain `initialize_agent` or `create_react_agent`.

**Status:** [x] done

---

### Sub-Task 7 — Explainability Utility

**Intent:** Implement a consistent tagging and formatting layer so every response shown to the user clearly distinguishes retrieved evidence, AI-generated analysis, and forecasts/interpretations. This satisfies requirement 8.

**Expected Outcomes:**
- `utils/explainability.py` — functions that wrap text or dicts with provenance labels.
- `format_response(answer, sources, tool_calls)` returns a structured dict ready for Streamlit rendering.
- Labels: 📄 **Retrieved Evidence**, 🤖 **AI Analysis**, 🔮 **Forecast / Interpretation**.
- A `render_explainability_card(st, response)` helper renders the labelled card in Streamlit with colour coding.

**Todo List:**
1. Write `utils/explainability.py`:
   - Define `ExplainabilityTag` enum: `RETRIEVED`, `AI_ANALYSIS`, `FORECAST`.
   - `tag_text(text, tag) -> dict` returns `{"text": text, "tag": tag}`.
   - `format_response(answer, sources, tool_calls) -> ExplainableResponse` dataclass.
   - `render_explainability_card(st_module, response)` function renders the card using `st.info`, `st.success`, `st.warning` colour coding.
2. Apply `format_response` in `agents/research_agent.py` before returning results.

**Relevant Context:** `agents/research_agent.py` (Sub-Task 6).

**Status:** [x] done

---

### Sub-Task 8 — Dashboard Tabs

**Intent:** Implement all seven Streamlit dashboard tabs as individual modules in `dashboard/`. Each tab consumes the data layer and tools directly for static visualisations; the chat tab routes through the agent.

**Expected Outcomes:**
- `dashboard/overview.py` — `render_overview(st)`: KPI cards (total market size, top competitor, sentiment score, trending keyword), summary table from sample data.
- `dashboard/trends.py` — `render_trends(st)`: Plotly line chart of keyword volumes over 12 months; LLM-generated trend summary.
- `dashboard/competitors.py` — `render_competitors(st)`: Plotly bar chart of market share; interactive comparison table; LLM strategic insights.
- `dashboard/sentiment.py` — `render_sentiment(st)`: Plotly pie chart of sentiment distribution; word cloud (optional); sample reviews with colour-coded sentiment.
- `dashboard/demand.py` — `render_demand(st)`: Plotly grouped bar chart of demand index by region/quarter; LLM demand narrative.
- `dashboard/alerts.py` — `render_alerts(st)`: Rules-based alerts from sample data (e.g. competitor gained >5% market share, keyword growth >20%). Displayed as colour-coded cards.
- `dashboard/chat.py` — `render_chat(st, agent)`: Full chat UI with message history, upload widget, voice input component, and explainability card rendering.

**Todo List:**
1. Write `dashboard/overview.py` using `st.metric`, `st.dataframe`, and summary stats from `data_loader`.
2. Write `dashboard/trends.py` using Plotly `go.Scatter`; pivot `sample_trends.json` into a wide DataFrame.
3. Write `dashboard/competitors.py` using Plotly `go.Bar` for market share; `st.dataframe` for comparison table.
4. Write `dashboard/sentiment.py` using Plotly `go.Pie`; display top 5 positive and negative reviews.
5. Write `dashboard/demand.py` using Plotly `go.Bar` grouped by region; show demand index table.
6. Write `dashboard/alerts.py` with rules: `if growth_pct > 20: emit trend alert`; `if competitor market_share change: emit competitor alert`.
7. Write `dashboard/chat.py`:
   - `st.chat_message` / `st.chat_input` loop with session-state message history.
   - File uploader triggering `VectorStoreManager.add_documents` on upload.
   - After ingesting an image file, display `st.info("🔍 Image processed via OCR — text-only extraction, not vision understanding.")` prominently.
   - `utils/voice_input.py` component for Web Speech API via `st.components.v1.html`.
   - After agent runs, call `render_explainability_card` on the response.
   - If `response.pdf_path` is set, render `st.download_button(label="📥 Download PDF Report", data=..., file_name=..., mime="application/pdf")` directly below the response card.
8. Write `dashboard/__init__.py`.

**Relevant Context:** `utils/data_loader.py` (Sub-Task 2); `utils/explainability.py` (Sub-Task 7); `agents/research_agent.py` (Sub-Task 6); `rag/vector_store.py` (Sub-Task 4).

**Status:** [x] done

---

### Sub-Task 9 — Voice Input Utility

**Intent:** Implement the browser-based voice input component using the Web Speech API injected as a custom Streamlit HTML component. Works in Chrome/Edge without any Python dependencies.

**Expected Outcomes:**
- `utils/voice_input.py` — `render_voice_input(st)` function.
- Renders a "🎤 Start Recording" button via `st.components.v1.html`.
- JavaScript uses `window.SpeechRecognition` and posts the transcript back via `Streamlit.setComponentValue`.
- Returns transcript string or `None`.
- Graceful degradation: if browser does not support Web Speech API, shows a message instead.

**Todo List:**
1. Write `utils/voice_input.py` with inline HTML+JS string.
2. Use `st.components.v1.html(html_code, height=80)` to embed.
3. Capture returned transcript and insert it into the chat input.

**Relevant Context:** `dashboard/chat.py` (Sub-Task 8).

**Status:** [x] done

---

### Sub-Task 10 — Main App Entry Point

**Intent:** Wire everything together in `app.py` — initialise configuration, bootstrap the vector store, instantiate the agent, apply custom CSS theming, and render the tab-based dashboard.

**Expected Outcomes:**
- `app.py` is the single `streamlit run` entry point.
- Page config: wide layout, custom icon, title "Intelligent Market Research Assistant".
- Sidebar: app logo, demo-mode badge (if active), model info, document upload shortcut, session stats.
- Main area: `st.tabs` with 7 tabs calling their respective `dashboard/*.py` render functions.
- On first run: bootstraps sample reports into ChromaDB (if empty).
- Loading spinner shown during agent initialisation.
- Professional colour theme via custom CSS injected with `st.markdown("<style>...</style>")`.

**Todo List:**
1. Write `app.py`:
   - Call `Config` initialisation; show demo-mode warning banner if active.
   - Call `VectorStoreManager.bootstrap_sample_reports()` with spinner.
   - Instantiate `ResearchAgent` once and store in `st.session_state`.
   - Inject custom CSS: dark navy/teal sidebar, gradient header, card styling, coloured metric tiles.
   - Build sidebar with: title, demo badge, model name badge, upload widget.
   - Build 7-tab main area.
2. Ensure `st.session_state` is used for: agent instance, chat history, uploaded document names, collection stats.

**Relevant Context:** All previous sub-tasks.

**Status:** [x] done

---

## Data Flow Summary

```
User action (chat query / file upload / voice)
       │
       ▼
  app.py  ──► dashboard/chat.py
                    │
          ┌─────────┼──────────────┐
          │                        │
   file upload              text/voice query
          │                        │
   rag/ingestion.py         agents/research_agent.py
   rag/chunker.py                  │
   rag/embedder.py          tool selection (ReAct)
   rag/vector_store.py             │
          │              ┌─────────┼──────────────────────┐
          │         RAGTool    NewsTool   SentimentTool  ...
          │              │         │           │
          │         ChromaDB  NewsAPI     TextBlob+LLM
          │              └─────────┼──────────────────────┘
          │                        │
          └────────────────►  Granite LLM (or MockLLM)
                                   │
                         utils/explainability.py
                                   │
                     Labelled response rendered in chat
```

---

## IBM Granite Integration Details

| Aspect | Detail |
|---|---|
| Model ID | `ibm/granite-3-8b-instruct` (overridable via `GRANITE_MODEL_ID`) |
| LangChain class | `langchain_ibm.WatsonxLLM` |
| Auth | `WATSONX_API_KEY` + `WATSONX_PROJECT_ID` from `.env` |
| Endpoint | `WATSONX_URL` (default: `https://us-south.ml.cloud.ibm.com`) |
| Params | `max_new_tokens=1024`, `temperature=0.3`, `top_p=0.9` |
| Embeddings | `WatsonxEmbeddings` with `ibm/slate-30m-english-rtrvr` |
| Fallback | `MockLLM` + `HuggingFaceEmbeddings` when `WATSONX_API_KEY` absent |

---

## Required Dependencies (requirements.txt)

```
streamlit>=1.32.0
langchain>=0.2.0
langchain-community>=0.2.0
langchain-ibm>=0.1.0
ibm-watsonx-ai>=0.2.6
chromadb>=0.5.0
sentence-transformers>=2.7.0
pypdf>=4.0.0
python-docx>=1.1.0
Pillow>=10.0.0
pytesseract>=0.3.10
textblob>=0.18.0
plotly>=5.20.0
pandas>=2.0.0
python-dotenv>=1.0.0
requests>=2.31.0
newsapi-python>=0.2.7
reportlab>=4.0.0
```

---

## Demo Mode Behaviour

When `WATSONX_API_KEY` is absent (or `DEMO_MODE=true`):
- `get_llm()` returns `MockLLM` — loads sample JSON at call time, computes real statistics, and builds structured multi-paragraph analysis responses.
- `get_embedder()` uses `sentence-transformers/all-MiniLM-L6-v2`.
- All tools use bundled JSON data files.
- A yellow "⚠️ Demo Mode" banner appears in the sidebar.
- All dashboard charts render normally from sample data.
- The chat still works; responses come from the data-driven MockLLM.
- Images are always processed via pytesseract OCR; the UI clearly labels results as OCR-based text extraction, not vision understanding.
- Report generation produces both a Markdown response and a downloadable PDF via `reportlab`; a `st.download_button` appears in the chat interface.

---

## Confirmed Modifications

| # | Modification | Applied in Sub-Tasks |
|---|---|---|
| M1 | MockLLM is data-driven: loads sample JSON, computes real stats, builds structured multi-paragraph responses | 3 |
| M2 | Image/OCR inputs always labelled as OCR-based text extraction, not vision understanding | 4, 8 |
| M3 | Report tool generates downloadable PDF via reportlab alongside Markdown | 5 |
| M4 | AgentResponse carries optional pdf_path; chat tab renders st.download_button | 6, 8 |
| M5 | reportlab>=4.0.0 added to requirements | 1, 5 |

---

## Implementation Order

The sub-tasks are designed to be implemented sequentially; each builds on the previous:

1. Scaffold & Config → 2. Sample Data → 3. LLM Layer → 4. RAG Pipeline → 5. Tools → 6. Agent → 7. Explainability → 8. Dashboard → 9. Voice Input → 10. App Entry Point
