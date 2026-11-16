# RAG evaluation

The benchmark separates retrieval quality from answer quality. The 50-question TCS set stores expected values, units, source paths and evidence-text anchors. At runtime, every anchor must resolve to one or more stable chunk IDs generated from the exact indexed Markdown chunking process; the run fails closed if an anchor is not found.

## Production-aligned retrieval comparison

The active system uses SQLite FTS5's BM25 ranking and a NetworkX evidence graph. Groq is used only for answer generation. There is no dense-vector baseline because the Groq embeddings endpoint returned `model_not_found`; no Hugging Face model or alternate embedding provider is used.

Run locally after building or downloading the index:

```bash
python -m backend.rag.build_vector_db
python -m evaluation.run_tcs_benchmark
```

The benchmark compares:
- `fts5_only`: SQLite FTS5 BM25 lexical retrieval.
- `graph_only`: entity/period/evidence graph retrieval.
- `fts5_graph_rrf`: reciprocal-rank fusion of lexical and graph rankings.

The same frozen 50 questions and resolved anchor labels are used for all methods. Results include Recall@5, MRR@5 and per-question metrics. The FTS5 method is the same lexical index used by the live API, rather than a separate full-corpus in-memory approximation.

The workflow `.github/workflows/tcs-retrieval-benchmark.yml` runs manually, on benchmark-code changes, or after the index-publishing workflow succeeds. It publishes `retrieval_comparison.json` and `answer_audit_packet.json` as an Actions artifact.

## Answer-level audit

When `GROQ_API_KEY` is configured, the runner generates an answer for each of the 50 questions with rate limiting. The audit packet records cited source excerpts, citation coverage, whether a cited excerpt contains the expected numeric value, and whether cited evidence overlaps a gold anchor. It also includes five human-review fields:
1. Answer relevance: entity, metric, period and comparison match the question.
2. Faithfulness: every material factual claim is supported by its cited excerpt.
3. Numerical accuracy: value, currency, unit, sign, period and accounting basis match.
4. Citation correctness: each citation supports the adjacent claim.
5. Page traceability: source file and page resolve to the original disclosure.

Automated numeric-token checks and citation presence are screening signals only. They do not establish faithfulness, correct accounting basis or citation entailment. Human review remains pending until a reviewer inspects each answer against the original source documents. If Groq is not configured, the retrieval comparison still runs and the audit packet clearly records that answer generation was skipped.

## Interpretation and reporting rules

- Do not report retrieval metrics without naming the benchmark size, K, method and run date.
- Do not compare FTS5/graph metrics directly with historical BGE or dense metrics as if they were the same experiment.
- Source anchors are carefully selected evidence locations, not a substitute for independent double annotation.
- Overlapping chunks containing the same anchor are labelled relevant; alternate valid evidence is not counted unless included in the reviewed labels.
- Report retrieval metrics and answer-level audit results separately.
- Do not claim answer faithfulness or numerical accuracy rates until human review is complete.

A transparent benchmark with explicit limitations is more useful than an impressive but unverified score.
