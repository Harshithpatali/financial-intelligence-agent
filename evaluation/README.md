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
