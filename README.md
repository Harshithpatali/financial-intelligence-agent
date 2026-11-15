# TCS Financial Intelligence Agent

A source-grounded financial research assistant for TCS annual reports, quarterly disclosures, financial statements, earnings-call transcripts and company news.

## Architecture

- **Ingestion:** PDF-to-Markdown conversion with page-boundary markers; CSV/news knowledge documents.
- **Chunking:** deterministic overlapping chunks with source, document type and page metadata.
- **Dense retrieval:** BAAI BGE Small embeddings stored in persistent ChromaDB.
- **Lexical retrieval:** BM25 keyword ranking.
- **Knowledge graph:** local NetworkX entity graph; nodes represent financial concepts/reporting periods and edges represent co-mentions within evidence chunks.
- **Fusion:** reciprocal-rank fusion combines dense, BM25 and graph retrieval.
- **Generation:** Groq-hosted Llama model answers only from retrieved evidence and is instructed to cite source IDs.
- **Serving:** FastAPI backend and Streamlit financial dashboard.

> The graph index is a local NetworkX graph, not a managed Neo4j deployment. Its co-mention edges are evidence links, not causal relationships.

## Run locally

Install backend dependencies:

```bash
pip install -r backend/requirements.txt
```

Set environment variables (at minimum `GROQ_API_KEY`; optionally `HF_TOKEN` and `GROQ_MODEL`).

Build/rebuild the vector and graph indexes after changing source documents:

```bash
python -m backend.rag.build_vector_db
```

To rebuild only the graph from an existing Chroma collection:

```bash
python -m backend.rag.build_graph_from_collection
```

Start the API:

```bash
uvicorn backend.api.main:app --host 0.0.0.0 --port 10000
```

Start the Streamlit UI from the repository root using your configured Streamlit entry point.

## API

- `GET /` service status
- `GET /health` health check
- `GET /ask?question=...` question answering
- `POST /ask` JSON body: `{"question":"..." }`

Answers include source filename, page where available, chunk ID, and retrieval score. Existing Markdown documents created before page markers are indexed with page unavailable; rerun PDF ingestion to enable page-level citations.

## Evaluation and quality

Automated tests cover page metadata, graph construction/entity extraction, safe no-evidence behavior, source metadata and API validation. GitHub Actions runs the test suite on pushes and pull requests.

Before describing retrieval quality as improved, evaluate a fixed, reviewed question set and report Recall@k, MRR, citation correctness, faithfulness and answer relevance. The repository does not claim benchmark scores until that evaluation has been run.

## Limitations

- The initial entity extractor is rule-based and focused on common TCS/financial concepts; it is not a general-purpose financial NER model.
- Graph relationships indicate co-mention in a source chunk, not causal or directional business relationships.
- Exact page citations require PDF documents to be re-ingested with page markers.
- Financial outputs are research assistance, not investment advice.
