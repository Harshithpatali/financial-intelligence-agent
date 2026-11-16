# TCS Financial Intelligence Agent

<p align="center">
  <img src="assets/dashboard.png" alt="TCS financial intelligence dashboard" width="100%">
</p>

**A financial research assistant for TCS disclosures**, combining document ingestion, dense retrieval, BM25, a provenance-backed entity graph and grounded LLM responses.

[![Live Demo](https://img.shields.io/badge/Live-Streamlit%20Demo-brightgreen)](https://financial-intelligence-agent-g.streamlit.app/)
[![API](https://img.shields.io/badge/API-FastAPI-009688)](https://financial-intelligence-agent-pmgy.onrender.com/)
[![Python](https://img.shields.io/badge/Python-3.11%2B-blue)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/Tests-pytest-informational)](.github/workflows/tests.yml)

## What it does

Users can ask questions about TCS revenue, profitability, cash flow, business segments, AI strategy, risks and management commentary. The system searches annual reports, quarterly disclosures, financial statements, earnings-call transcripts and company news, then generates an answer with supporting source references.

## Architecture

1. **Ingestion:** PDF, CSV and news inputs are transformed into a version-controlled Markdown knowledge repository.
2. **Page-aware chunking:** documents are split into overlapping chunks while retaining source and page metadata when available.
3. **Offline document indexing:** GitHub Actions runs BAAI BGE Small once when knowledge sources or indexing code change, then publishes the ChromaDB vectors and NetworkX graph as a versioned GitHub Release asset.
4. **Query embedding:** Streamlit loads the same BAAI BGE Small model on demand and embeds each user question; the 384-dimensional query vector is sent to FastAPI.
5. **Lexical retrieval:** BM25 retrieves exact financial terms and figures.
6. **Knowledge graph:** NetworkX stores financial entities, reporting periods and evidence-chunk nodes. Typed edges distinguish `MENTIONS`, `HAS_EVIDENCE_IN_PERIOD`, `PRECEDES`, qualitative trend evidence and `CO_MENTIONED_WITHIN_CHUNK`. Co-mentions are not causal relationships; trend nodes are unverified qualitative observations.
7. **Hybrid ranking:** reciprocal-rank fusion combines dense, BM25 and graph results.
8. **Generation:** Groq-hosted GPT OSS answers from retrieved context and is instructed to cite source IDs.
9. **Serving:** FastAPI exposes question-answering endpoints; Streamlit provides the user interface and financial dashboards.

> Graph edges describe evidence co-mentions, not causality or directional business relationships. This project uses a local NetworkX graph; it does not require a Neo4j server.

## Data coverage

The repository contains TCS annual reports, quarterly financial reports and earnings-call transcripts, financial statement/ratio data, company profile information and a collection of company news documents. Source files are retained in the `knowledge/` directory for auditability.

## Local setup

```bash
pip install -r backend/requirements.txt
```

Configure `GROQ_API_KEY` in the Render environment. The frontend does not need the Groq key; it only sends the question and its query embedding to the API.

Document vectors are built by the GitHub Actions workflow **Build and publish RAG index**. It runs automatically when files under `knowledge/` or indexing code change, and can also be started manually from the Actions tab. The workflow publishes `tcs-rag-index.zip` as a versioned GitHub Release asset. Render downloads the latest release during startup; the index is not committed into the source tree.

For a local rebuild, install the indexing dependency and run:

```bash
pip install "sentence-transformers>=3.0,<6.0"
python -m backend.rag.build_vector_db
```

The Streamlit frontend also installs `sentence-transformers` and lazily loads `BAAI/bge-small-en-v1.5` to embed each query. The same model and normalized 384-dimensional vectors are used on both sides. Render does not install or load PyTorch/SentenceTransformers.

To build only the graph from an existing Chroma collection:

```bash
python -m backend.rag.build_graph_from_collection
```

Run the API:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 10000
```

## API

- `GET /` service metadata
- `GET /health` liveness check\n- `GET /ready` readiness check for Chroma, graph index and Groq configuration
- `GET /ask?question=...` question answering
- `POST /ask` JSON request: `{"question":"What was TCS revenue in FY2024?"}`

Responses include the answer, source filename, page where available, chunk ID, score and retrieval methods. Older Markdown files without page markers need to be regenerated from their original PDFs to support page-level citations.

## Public market data and news

The Markets & News dashboard uses Yahoo Finance through `yfinance` for quotes/history and Google News RSS for headlines. These sources do not require a paid API key. They are public/unofficial access paths: provider availability, coverage and rate limits can change. Quotes may be delayed and are not exchange-certified real-time prices. Responses are cached briefly in the API process to reduce repeated requests. Always verify prices with the relevant exchange or broker before making decisions.

## Evaluation

A lightweight offline evaluation module provides Recall@K, MRR@K and citation-coverage calculations. Add manually reviewed question-to-chunk relevance labels before reporting retrieval scores. Also manually inspect citation correctness, faithfulness, answer relevance and abstention on questions with no supporting evidence. Compare dense-only, dense+BM25 and dense+BM25+graph on the same frozen benchmark.

GitHub Actions runs the automated test suite on pushes and pull requests. API safeguards include normalized finite 384-dimensional embedding validation, a process-local per-IP sliding-window request limit, structured JSON logs, readiness checks and explicit Groq rate-limit responses with `Retry-After`.

## Limitations and next steps

- The initial entity extractor is deterministic and focused on common TCS/financial concepts; it is not a general-purpose financial NER model.
- Graph edges are evidence co-mentions, not causal or semantic predicates.
- Existing source documents without page markers cannot provide page-accurate citations until re-ingested.
- Benchmark scores should only be reported after relevance labels and answer audits are reviewed.
- Outputs are research assistance, not investment advice.


### Reproducible retrieval benchmark

A benchmark runner is available at `evaluation/benchmark.py`. It calculates mean Recall@K and mean MRR@K plus per-question results from a JSONL file containing manually reviewed relevant chunk IDs and ordered retrieved chunk IDs. Example usage and the required input schema are documented in [evaluation/README.md](evaluation/README.md). No benchmark score is claimed until relevance labels are reviewed against the source documents; citation coverage is reported separately from citation correctness and answer faithfulness.

---

## Daily Market Sentiment Intelligence

The project also publishes a daily, precomputed financial-news sentiment report for seven Indian companies and market benchmarks. The report is generated by GitHub Actions using Google News RSS headlines and Groq-hosted `openai/gpt-oss-20b`, then consumed by the Streamlit Market Sentiment page.

### Coverage

The daily report covers:

| Key | Company / benchmark | RSS search topic |
|---|---|---|
| `tcs` | TCS · Tata Consultancy Services | Tata Consultancy Services TCS |
| `infosys` | Infosys | Infosys India |
| `reliance` | Reliance Industries | Reliance Industries |
| `hdfc_bank` | HDFC Bank | HDFC Bank |
| `icici_bank` | ICICI Bank | ICICI Bank |
| `nifty50` | NIFTY 50 | NIFTY 50 Indian stock market |
| `sensex` | SENSEX | BSE Sensex Indian stock market |

**Collection target:** up to 20 unique headlines per topic (up to 140 across all seven topics). Google News RSS may return fewer than 20 relevant unique items for a query, so 140 is a maximum target, not a guaranteed daily count. The collector removes duplicate titles within each topic. A headline may still appear in more than one topic if it is relevant to multiple searches.

### Sentiment pipeline

```mermaid
flowchart TD
    A[GitHub Actions schedule<br/>06:00 Asia/Kolkata] --> B[Run Python report generator]
    B --> C[Fetch Google News RSS<br/>7 topics · up to 20 each]
    C --> D[Remove duplicate titles<br/>within each topic]
    D --> E[Batch headlines by topic]
    E --> F[Groq API<br/>openai/gpt-oss-20b]
    F --> G[Structured JSON response]
    G --> H[Validate indices, labels<br/>confidence and score ranges]
    H --> I[Aggregate counts,<br/>mean score and date trend]
    I --> J{All 7 topics complete?}
    J -- No --> K[Fail run<br/>retain previous published report]
    J -- Yes --> L[Write report JSON]
    L --> M[Commit latest.json<br/>and dated history file]
    M --> N[Streamlit Market Sentiment UI]
```

### Classification and metrics

Each headline receives:
- **Label:** positive, negative or neutral.
- **Confidence:** model-reported value from 0 to 1. It is not a calibrated probability.
- **Sentiment score:** model-reported value from -1 to +1. Negative values indicate adverse tone; positive values indicate favourable tone; values near zero indicate neutral or mixed tone.
- **Evidence metadata:** headline, publisher, source URL and publication date.

For a topic with (N) classified headlines, the aggregate sentiment score is the arithmetic mean:

`mean sentiment = (sum of headline sentiment scores) / N`

The aggregate label uses the current generator thresholds:
- Positive: mean score greater than +0.08.
- Negative: mean score less than -0.08.
- Neutral: otherwise.

Positive, negative and neutral article counts are calculated directly from the headline labels. The daily trend groups scored headlines by publication date and reports the mean score and article count for each date.

> These values describe the tone of the retrieved news text. They are not stock-return forecasts, calibrated sentiment probabilities, trading signals, or investment advice.

### Current published baseline

The following is the actual published report from **16 November 2026**, before the increase from 8 to 20 headlines per topic. It is included as a reproducible baseline, not as a current 20-headline result.

| Company / benchmark | Articles | Positive | Neutral | Negative | Mean score | Label |
|---|---:|---:|---:|---:|---:|---|
| TCS | 8 | 8 | 0 | 0 | +0.8500 | Positive |
| Infosys | 8 | 7 | 1 | 0 | +0.7000 | Positive |
| Reliance | 8 | 7 | 1 | 0 | +0.6500 | Positive |
| HDFC Bank | 8 | 0 | 6 | 2 | -0.1625 | Negative |
| ICICI Bank | 8 | 0 | 7 | 1 | -0.0625 | Neutral |
| NIFTY 50 | 8 | 5 | 3 | 0 | +0.4125 | Positive |
| SENSEX | 8 | 6 | 0 | 2 | +0.3500 | Positive |
| **Total** | **56** | **33** | **18** | **5** | — | — |

#### Baseline article-volume chart

```mermaid
xychart-beta
    title "Published baseline: headlines per topic (16 Nov 2026)"
    x-axis ["TCS", "Infosys", "Reliance", "HDFC", "ICICI", "NIFTY", "SENSEX"]
    y-axis "Headlines" 0 --> 20
    bar [8, 8, 8, 8, 8, 8, 8]
```

#### Baseline sentiment-score chart

```mermaid
xychart-beta
    title "Published baseline: mean headline sentiment (16 Nov 2026)"
    x-axis ["TCS", "Infosys", "Reliance", "HDFC", "ICICI", "NIFTY", "SENSEX"]
    y-axis "Mean score" -1 --> 1
    bar [0.85, 0.70, 0.65, -0.1625, -0.0625, 0.4125, 0.35]
```

### Report storage and access

- Latest report: [`sentiment-data/latest.json`](https://github.com/Harshithpatali/financial-intelligence-agent/blob/sentiment-data/latest.json)
- Daily history: [`sentiment-data/history/`](https://github.com/Harshithpatali/financial-intelligence-agent/tree/sentiment-data/history)
- Workflow: [Daily Groq sentiment report](https://github.com/Harshithpatali/financial-intelligence-agent/actions/workflows/daily_sentiment.yml)
- Generator: [`scripts/generate_sentiment_report.py`](scripts/generate_sentiment_report.py)

The workflow runs daily at 00:30 UTC (06:00 IST) and can also be started manually from GitHub Actions. It requires the repository Actions secret `GROQ_API_KEY`; the key is not written to the report branch. The workflow publishes only when all seven topics are successfully collected and classified. An incomplete run fails rather than replacing the previous good report.

### Reproduce locally

Python 3.11 or newer is recommended.

```bash
python -m pip install groq
export GROQ_API_KEY="your-groq-api-key"
python scripts/generate_sentiment_report.py --output sentiment-report.json
```

On Windows PowerShell:

```powershell
python -m pip install groq
$env:GROQ_API_KEY="your-groq-api-key"
python scripts/generate_sentiment_report.py --output sentiment-report.json
```

### Validation and quality controls

Automated tests cover the configured topic universe, model identifier, collection limit, classification order and output score ranges. The report generator additionally checks that every headline index is present exactly once, every label belongs to the supported set, confidence is within [0, 1], and sentiment score is within [-1, 1].

Recommended next evaluation steps:
1. Build a manually labelled benchmark of representative Indian financial headlines.
2. Measure label accuracy, macro-F1 and per-class precision/recall against human labels.
3. Review false positives, false negatives, neutral/mixed cases and duplicate or low-relevance stories.
4. Track daily coverage, publisher concentration, missing publication dates and API failures.
5. Compare model scores with a second reviewer or model before using them in research decisions.

No accuracy, F1, calibration or predictive-return metric is claimed yet because a human-labelled evaluation benchmark has not been completed.
