# Intelligent Market Research Assistant

> **Problem Statement No. 6** — Agentic AI application built with Python, LangChain, RAG, IBM Granite (watsonx.ai), Streamlit, and ChromaDB.

---

## Overview

This application is an **agentic market research assistant** that:

- Fuses news articles, uploaded industry reports, competitor data, customer reviews, and trend data
- Runs a LangChain ReAct agent that autonomously selects from 7 specialised tools
- Uses IBM Granite (`ibm/granite-3-8b-instruct`) as the primary LLM via watsonx.ai
- Stores and retrieves documents through a ChromaDB vector database (RAG)
- Presents results in a professional 7-tab Streamlit dashboard
- Clearly labels every insight as *Retrieved Evidence*, *AI Analysis*, or *Forecast / Interpretation*
- Works fully offline in **Demo Mode** using bundled sample data and a data-driven MockLLM

---

## Project Structure

```
intelligent-market-research-assistant/
├── app.py                    # Streamlit entry point  (streamlit run app.py)
├── .env.example              # Environment variable template
├── requirements.txt
├── README.md
├── agents/                   # LangChain ReAct AgentExecutor
├── tools/                    # 7 agent tools (RAG, news, sentiment, …)
├── rag/                      # Ingestion → chunking → embedding → ChromaDB
├── models/                   # Granite LLM wrapper + MockLLM
├── dashboard/                # One module per Streamlit tab
├── utils/                    # Config, data loader, explainability, voice input
└── data/                     # Sample JSON data, reports, ChromaDB persistence
```

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | ≥ 3.10 |
| pip | ≥ 23 |
| Tesseract OCR *(optional, for image uploads)* | ≥ 5.0 — [install instructions](https://github.com/tesseract-ocr/tesseract) |

---

## Quick Start

### 1 — Clone and install

```bash
git clone <repo-url>
cd intelligent-market-research-assistant
pip install -r requirements.txt
```

Download NLTK data (one-time, needed by TextBlob):

```bash
python -m textblob.download_corpora
```

### 2 — Configure environment

```bash
cp .env.example .env
# Open .env and fill in your values (all are optional for demo mode)
```

### 3 — Run

```bash
streamlit run app.py
```

Open your browser at **http://localhost:8501**.

---

## Environment Variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `WATSONX_API_KEY` | Live mode only | *(empty)* | IBM Cloud API key |
| `WATSONX_PROJECT_ID` | Live mode only | *(empty)* | watsonx.ai project ID |
| `WATSONX_URL` | No | `https://us-south.ml.cloud.ibm.com` | Regional endpoint |
| `GRANITE_MODEL_ID` | No | `ibm/granite-3-8b-instruct` | Granite model to use |
| `NEWSAPI_KEY` | No | *(empty)* | NewsAPI.org free key |
| `DEMO_MODE` | No | `false` | Force demo mode |

Obtain credentials from [IBM watsonx.ai](https://www.ibm.com/products/watsonx-ai) → your project → Manage → API keys.

---

## Demo Mode

The application automatically activates **Demo Mode** when `WATSONX_API_KEY` is absent (or `DEMO_MODE=true`).

In demo mode:

- The **data-driven MockLLM** replaces Granite — it loads real sample data, computes statistics, and writes structured research responses. Responses look and feel like genuine AI analysis.
- `sentence-transformers/all-MiniLM-L6-v2` is used for embeddings (no external API call).
- All 7 agent tools fall back to bundled JSON data files.
- All dashboard charts, tables, and visualisations are fully functional.
- A yellow ⚠️ **Demo Mode** banner is shown in the sidebar.

---

## Dashboard Tabs

| Tab | Description |
|---|---|
| 📊 Market Overview | KPIs, top competitor, sentiment score, trending keyword |
| 📈 Trend Analysis | Plotly line chart of 5 keywords over 12 months + LLM narrative |
| 🏢 Competitor Comparison | Market share chart, comparison table, strategic insights |
| 😊 Sentiment Analysis | Pie chart of sentiment distribution + top reviews |
| 📉 Demand Forecast | Demand index by region/quarter + LLM interpretation |
| 🔔 Alerts | Rules-based alerts for notable market movements |
| 💬 AI Research Assistant | Full chat with file upload, voice input, explainability cards |

---

## Features

### Multimodal Input
- **Text queries** via the chat interface
- **Document upload** (PDF, DOCX, TXT) — ingested into ChromaDB for RAG
- **Image / chart upload** — text extracted via Tesseract OCR (*text-only extraction, not vision understanding*)
- **Voice input** — browser Web Speech API (Chrome/Edge); transcript inserted into chat

### Explainability
Every AI response is tagged:
- 📄 **Retrieved Evidence** — text pulled directly from indexed documents
- 🤖 **AI Analysis** — LLM-generated interpretation of the evidence
- 🔮 **Forecast / Interpretation** — model-generated projections (not guaranteed facts)

### Report Generation
Ask the assistant *"Generate a full market research report"* to produce:
- A structured Markdown report in the chat
- A **downloadable PDF** (click the 📥 button that appears below the response)

---

## Technology Stack

| Layer | Technology |
|---|---|
| LLM | IBM Granite 3-8b-instruct via `langchain-ibm` WatsonxLLM |
| Agent framework | LangChain ReAct AgentExecutor |
| RAG | ChromaDB (local persistent) + LangChain retrieval chain |
| Embeddings | watsonx `ibm/slate-30m-english-rtrvr` → sentence-transformers fallback |
| UI | Streamlit with Plotly charts |
| Sentiment | TextBlob (quantitative) + Granite (qualitative narrative) |
| PDF export | reportlab |
| OCR | pytesseract / Tesseract |
| News | NewsAPI.org → sample JSON fallback |

---

## Notes for Graders

- All credentials come **exclusively** from environment variables — nothing is hard-coded.
- The application runs **fully offline** in demo mode; no account or API key is needed to evaluate the UI and AI responses.
- Sample data covers: AI, Cloud, Fintech, E-commerce, and Cybersecurity industries.
- The ChromaDB vector store is automatically seeded with two sample market reports on first run.
