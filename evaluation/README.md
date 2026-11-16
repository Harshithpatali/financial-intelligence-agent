# RAG evaluation

The evaluation harness computes retrieval metrics over a **manually reviewed** JSONL benchmark. It deliberately does not ship fabricated relevance labels or publish scores before the benchmark has been reviewed.

## Benchmark format

Each record contains a question, relevant chunk IDs confirmed by a reviewer against the source documents, and the ordered chunk IDs returned by the retrieval method under test:

```json
{"question":"What was TCS revenue in FY2024?","relevant_chunk_ids":["reviewed-chunk-id-1","reviewed-chunk-id-2"],"retrieved_chunk_ids":["reviewed-chunk-id-1","candidate-id-2","reviewed-chunk-id-2"]}
```

Use the same frozen questions and relevance labels when comparing dense-only, dense+BM25, and dense+BM25+graph retrieval. Store a separate benchmark result for each retrieval configuration. Do not use the example IDs above as real labels.

## Run

```bash
python -m evaluation.benchmark path/to/retrieval-results.jsonl --k 5 --output evaluation/results/hybrid.json
```

The JSON output includes mean Recall@K, mean MRR@K and per-question metrics. The input must include ordered retrieved IDs as well as reviewed relevant IDs. The harness rejects empty or duplicate relevance labels.

## Answer-level audit (separate from retrieval)

For each evaluated answer, manually inspect citation correctness, faithfulness, answer relevance, numerical accuracy (especially units and reporting periods), and abstention on no-evidence questions.

Citation coverage in `evaluation/metrics.py` measures citation presence only. It does not measure whether a citation supports a claim. Report retrieval metrics and answer-level audit results separately.

## Reporting rule

Do not claim benchmark performance until questions cover financial statements, reporting periods, segment performance, risks, management commentary and no-evidence cases; a reviewer has checked every relevant chunk ID against the original source; the benchmark is frozen before comparing retrieval configurations; and results include benchmark size, K value, retrieval configuration and date.

A small transparent benchmark is more useful than an impressive but unverified score.


## TCS benchmark implementation

`evaluation/tcs_benchmark.json` contains 50 question records sourced from the checked-in TCS FY2026 and FY2025 nine-month disclosures. Each record stores an expected answer, unit, source path and text anchor. The runner resolves each anchor to the stable SHA-1 chunk IDs used by the Chroma index and fails if an anchor cannot be found in a single chunk.

Run the comparison with:

```bash
python -m backend.rag.download_index
python -m evaluation.run_tcs_benchmark
```

The runner compares dense-only BAAI/bge-small-en-v1.5 retrieval, dense plus bounded lexical retrieval using reciprocal-rank fusion, and dense plus lexical plus NetworkX graph candidates. The workflow `.github/workflows/tcs-retrieval-benchmark.yml` runs on benchmark changes or manual dispatch. It publishes `retrieval_comparison.json` and `answer_audit_packet.json` as an Actions artifact. If `GROQ_API_KEY` is configured, it generates answers for a fixed 13-question numerical audit subset.

**Interpretation warning:** source anchors are carefully selected evidence locations, not a substitute for independent double annotation. Numeric-token checks are screening checks, not proof that an answer uses the right period, unit, sign or accounting basis. The answer packet marks human faithfulness and numerical review as pending. Do not describe those as completed until the packet has been inspected against the source documents.
