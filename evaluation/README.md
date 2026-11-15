# RAG evaluation

Do not report retrieval or answer-quality scores until the relevance labels have been reviewed against the source documents.

Create a JSONL benchmark with one record per question:

```json
{"question":"What was TCS revenue in FY2024?","relevant_chunk_ids":["reviewed-chunk-id-1","reviewed-chunk-id-2"]}
```

For each question, capture the ordered retrieved chunk IDs and calculate:
- Recall@5: fraction of manually labelled relevant chunks found in the top five.
- MRR@5: reciprocal rank of the first relevant chunk.
- Citation coverage: proportion of answer sentences carrying a valid source citation.

Also manually audit citation correctness and answer faithfulness; citation presence alone does not prove that a citation supports a claim. Include a no-evidence set and measure abstention accuracy. Keep a frozen benchmark split when comparing dense-only, dense+BM25, and dense+BM25+graph retrieval.
