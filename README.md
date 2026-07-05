<div align="center">

#  Financial Intelligence Agent

**TCS Financial Intelligence Agent is an enterprise-style Retrieval-Augmented Generation (RAG) platform built using FastAPI, ChromaDB, Sentence Transformers, Groq LLM, and Streamlit. The system transforms TCS annual reports into a structured Open Knowledge Format (OKF)-based financial knowledge repository, enabling semantic search, context-aware financial analysis, and source-grounded question answering. Deployed on Render and Streamlit Cloud, the platform provides interactive financial insights, executive summaries, and natural language access to corporate disclosures and annual report intelligence.**

[![Live App](https://img.shields.io/badge/Live-Demo-brightgreen)](https://financial-intelligence-agent-g.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/Backend-FastAPI-009688)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Frontend-Streamlit-FF4B4B)](https://streamlit.io/)
[![License](https://img.shields.io/badge/License-Educational-lightgrey)](#license)

[Live Demo](https://financial-intelligence-agent-g.streamlit.app/) · [Features](#-features) · [Architecture](#-architecture) · [Setup](#-local-setup) · [API](#-api-reference)

</div>

---

## 📖 Overview

Annual reports typically span hundreds of pages covering financial statements, management commentary, risk disclosures, and strategic initiatives — making it time-consuming to extract meaningful insights.

**TCS Financial Intelligence Agent** solves this by transforming TCS annual reports into a searchable, conversational knowledge base. Investors, analysts, students, and researchers can ask natural language questions and receive **context-aware, source-grounded answers** drawn directly from official disclosures.



Traditional annual reports contain hundreds of pages of financial statements, management discussions, risk disclosures, AI strategy updates, and operational insights.

This project converts those reports into a searchable knowledge base using:

-Open Knowledge Format (OKF)-style document storage
-Vector embeddings
-ChromaDB vector search
-Retrieval-Augmented Generation (RAG)
-Large Language Models via Groq
-Interactive Streamlit dashboard

Instead of manually reading reports, users can ask:
```text
What was TCS revenue in FY2024?
What are the key risks facing TCS?
What did management say about AI?
Who is the CEO of TCS?
Summarize FY2025 performance.
```

The system combines document parsing, vector embeddings, semantic retrieval, and large language model inference into a single end-to-end RAG pipeline.

---



## 🏗️ Architecture

```text
                    ┌──────────────────────┐
                    │   Annual Reports      │
                    │      (PDF Files)      │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Document Parsing    │
                    │   PDF → Markdown       │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │   Chunk Creation &     │
                    │  Metadata Generation   │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │    BGE Embeddings      │
                    │  (SentenceTransformer) │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │       ChromaDB         │
                    │    Vector Storage      │
                    └───────────┬───────────┘
                                │
   User Question ───────────────▶  FastAPI Backend
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Semantic Retrieval    │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │       Groq LLM         │
                    └───────────┬───────────┘
                                │
                                ▼
                    ┌──────────────────────┐
                    │  Contextual Answer     │
                    │    + Source Citation   │
                    └──────────────────────┘
```

### System Workflow

```text
User Question
   → Generate Query Embedding
   → Search ChromaDB
   → Retrieve Relevant Chunks
   → Build Context Prompt
   → Groq LLM Inference
   → Answer + Sources
   → Streamlit Dashboard
```

---

## ✨ Features

### 💬 Financial Question Answering
Ask natural language questions about revenue, profitability, growth, risks, investments, AI initiatives, business strategy, cash flow, leadership changes, and annual report disclosures.

### 🔍 Retrieval-Augmented Generation
Answers are generated from retrieved company documents rather than relying solely on model memory, resulting in:
- Reduced hallucinations
- Source-grounded responses
- Explainable outputs
- Financially relevant context

### 📎 Source Attribution
Every answer includes the supporting report references:

```text
Sources:
annual-report-2022-2023.md
annual-report-2024-2025.md
annual-report-2025-2026.md
```

### 📈 Interactive Financial Dashboard
- Revenue analysis
- Net income trends
- Cash flow metrics
- Balance sheet overview
- Executive insights

### 🧠 AI Executive Insights
Automatically generated summaries covering growth drivers, strategic initiatives, risk factors, and AI transformation efforts.

---

## 🛠️ Technology Stack

| Layer            | Technology              |
|-------------------|--------------------------|
| Frontend          | Streamlit                |
| Backend           | FastAPI                  |
| LLM               | Groq                     |
| Vector Database   | ChromaDB                 |
| Embeddings        | BAAI BGE Small            |
| Data Processing   | Pandas                   |
| Visualization     | Plotly                   |
| API Hosting       | Render                   |
| Frontend Hosting  | Streamlit Cloud           |
| Language          | Python                   |
| Knowledge Format  | Markdown / OKF Style      |

---

## 📁 Project Structure

```text
financial-intelligence-agent/
│
├── backend/
│   ├── api/
│   │   └── main.py
│   │
│   ├── rag/
│   │   ├── rag_pipeline.py
│   │   ├── retriever.py
│   │   ├── vector_store.py
│   │   └── llm.py
│   │
│   ├── chroma_db/
│   │
│   └── requirements.txt
│
├── frontend/
│   ├── app.py
│   ├── utils.py
│   └── requirements.txt
│
├── knowledge/
│   └── annual_reports/
│
├── ingestion/
│
├── README.md
│
└── .gitignore
```

---

## 📚 Knowledge Base

The system uses annual reports converted into structured markdown files, forming an **Open Knowledge Format (OKF)-style** repository:

```text
knowledge/
└── annual_reports/
    ├── annual-report-2020-2021.md
    ├── annual-report-2021-2022.md
    ├── annual-report-2022-2023.md
    ├── annual-report-2023-2024.md
    ├── annual-report-2024-2025.md
    └── annual-report-2025-2026.md
```

This design keeps company knowledge:

- ✅ Human readable
- ✅ Version controllable
- ✅ Rebuildable
- ✅ Auditable

---

## 🔌 API Reference

### Request

```http
GET /ask?question=Who is the CEO of TCS?
```

### Response

```json
{
  "question": "Who is the CEO of TCS?",
  "answer": "K Krithivasan is the CEO and Managing Director of TCS.",
  "sources": [
    "annual-report-2025-2026.md"
  ],
  "num_sources": 1
}
```

---

## 📊 Financial Data Visualizations

Supported dashboard components:

- Revenue Trend Analysis
- Net Income Growth
- Free Cash Flow Analysis
- Capital Expenditure Tracking
- Cash Position Monitoring
- Balance Sheet Snapshot

---

## 🚀 Deployment Architecture

```text
Streamlit Cloud → Frontend Dashboard
        ↓
Render FastAPI API
        ↓
SentenceTransformer
        ↓
     ChromaDB
        ↓
     Groq LLM
```

---

## ⚙️ Local Setup

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/financial-intelligence-agent.git
cd financial-intelligence-agent
```

### 2. Create a Virtual Environment

```bash
python -m venv venv
```

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### 3. Install Backend Dependencies

```bash
pip install -r backend/requirements.txt
```

### 4. Run the FastAPI Backend

```bash
uvicorn backend.api.main:app --reload
```

- Backend: `http://localhost:8000`
- Swagger Docs: `http://localhost:8000/docs`

### 5. Install Frontend Dependencies & Run Streamlit

```bash
pip install -r frontend/requirements.txt
streamlit run frontend/app.py
```

- Frontend: `http://localhost:8501`

---

## 💡 Example Questions

```text
What was TCS revenue in FY2024?
Summarize FY2025 performance.
What are the major risks facing TCS?
What did TCS say about AI?
How has free cash flow changed?
Who is the CEO of TCS?
What are the key growth drivers?
What is the company's digital transformation strategy?
```

---

## 📈 Performance Characteristics

| Metric            | Value                    |
|--------------------|--------------------------|
| Embedding Model     | BAAI/bge-small-en-v1.5    |
| Vector Database     | ChromaDB                 |
| Retrieval Type      | Semantic Search           |
| Context Source      | Annual Reports            |
| LLM Provider        | Groq                      |
| Response Type       | Grounded RAG              |

---

## 🔮 Future Enhancements

**Financial Analysis**
- Ratio analysis
- CAGR calculations
- Margin analysis
- Segment-wise performance

**Advanced RAG**
- Hybrid search
- Cross-encoder reranking
- Query expansion
- Multi-hop retrieval

**User Experience**
- Conversation memory
- Report comparison
- Downloadable research reports
- Citation highlighting

**Data Sources**
- Quarterly reports
- Earnings calls
- Investor presentations
- SEC / regulatory filings
- News integration

---

## 📝 Resume Project Description

> Built an enterprise-style Financial Intelligence Agent using FastAPI, ChromaDB, Sentence Transformers, Groq LLM, and Streamlit. Developed an Open Knowledge Format (OKF)-based financial knowledge repository from annual reports, enabling semantic search and retrieval-augmented question answering with source-grounded responses. Deployed a production-ready RAG pipeline using Render and Streamlit Cloud for interactive financial analysis and executive insights.

---

## 👤 Author

**Harshith Devaraja**

- GitHub: [Harshithpatali](https://github.com/Harshithpatali)
- LinkedIn: [harshith-devaraja](https://www.linkedin.com/in/harshith-devaraja-087ba0228/)
- Portfolio: [harshithpatali.github.io](https://harshithpatali.github.io/harshith-portfolio/)

---

## 📄 License

This project is intended for educational, research, and portfolio purposes. Financial information is derived from publicly available TCS annual reports and should not be considered investment advice.
