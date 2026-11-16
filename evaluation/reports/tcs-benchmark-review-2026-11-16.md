# TCS retrieval and answer audit — 16 November 2026

## Run metadata

- Benchmark: 50 source-anchored TCS financial and operating-disclosure questions
- Retrieval cutoff: top 5 chunks
- Embedding model: `BAAI/bge-small-en-v1.5`
- Indexed corpus: 10,106 chunks
- BM25 corpus: 10,106 chunks
- Index used for this run: published `rag-index-6`
- Workflow: [TCS retrieval benchmark run 43854613821](https://github.com/Harshithpatali/financial-intelligence-agent/actions/runs/43854613821)
- Artifact: `tcs-retrieval-benchmark` (retrieval comparison and answer-audit JSON)

## Retrieval comparison

| Method | Recall@5 | MRR@5 |
|---|---:|---:|
| Dense only | 0.0460 | 0.0440 |
| BM25 only | **0.2782** | 0.1830 |
| Dense + BM25 (RRF) | 0.1354 | 0.1490 |
| Dense + BM25 + graph (RRF) | 0.2474 | **0.2440** |

### Interpretation

- BM25-only achieved the highest Recall@5 in this run.
- Dense retrieval alone performed poorly on this financial fact-retrieval set.
- Equal-weight dense/BM25 fusion underperformed BM25 alone.
- Adding graph candidates improved both metrics over dense+BM25 fusion, but did not exceed BM25-only Recall@5.
- The graph-assisted method had the best MRR@5, suggesting it more often placed at least one relevant chunk near the top when it found one.

These are retrieval metrics against the 50 source-anchor labels, not answer accuracy. The current BM25 baseline is an offline full-corpus evaluation; the live API still uses its memory-bounded lexical candidate path. This comparison must not be described as the deployed API's BM25 performance.

## Answer-level audit

Thirteen representative questions were generated with top-5 retrieved context. The answer packet includes source metadata and cited evidence excerpts. Manual comparison against the checked-in source disclosures found:

- **3/13 answers correct**: q01 (FY2026 revenue), q23 (Q4FY26 revenue; ₹706,980 million = ₹70,698 crore), q35 (final dividend).
- **2/13 materially incorrect**:
  - q07 returned operating profit rather than profit for the year.
  - q20 returned approximately ₹1,037 crore, which is the standalone-company sensitivity in the cited annual-report evidence. The benchmark asks for the Group sensitivity, which is approximately ₹713 crore in the consolidated statement.
- **8/13 false abstentions**: q05, q09, q12, q16, q28, q40, q47 and q49. The requested values are present in the designated source disclosures, but the retrieved context did not provide the required evidence to the answer model.

Automated citation coverage across the 13 generated answers was 48.7%. Citation coverage measures whether a valid source marker appears; it does not establish that the cited source supports the claim. The machine flag for exact gold-chunk overlap is also strict: a different official source can support the same fact without matching the single designated source anchor.

## Label review and corrections

- All 50 questions have a unique ID, a checked-in source file and a text anchor.
- CI validates that the expected value appears near the anchor, with explicit handling for comma-formatted values, accounting negatives, abbreviated headcount and spelled-out counts.
- q39 was corrected from 2.8% to 2.4% CBG QoQ constant-currency growth after checking the Q4FY26 press release.
- q20 now explicitly asks for the **Group** foreign-currency sensitivity to distinguish the consolidated ₹713 crore figure from the standalone ₹1,037 crore figure.
- The answer audit remains a 13-question sample, not a complete answer-quality evaluation over all 50 questions.

## Engineering follow-up

1. Improve financial-query retrieval for PBT, EPS, cash flow, assets, FX sensitivity and period-specific operating metrics.
2. Keep BM25 as the current offline retrieval baseline; test weighted fusion on a held-out split before changing production ranking.
3. Rebuild the graph index with the expanded financial-statement entity aliases. The rebuild workflow was triggered separately from this benchmark run; this report uses the earlier published index version.
4. Add alternative valid source chunks to relevance labels where multiple official documents report the same fact.
5. Re-run the 50-question benchmark on the rebuilt index and compare results on a frozen holdout set.
6. Do not claim high answer accuracy until false abstentions, wrong-period answers and wrong accounting-basis answers are materially reduced.

## Overall conclusion

The benchmark demonstrates that the current system has a retrieval weakness on exact financial questions. BM25 is the strongest retrieval baseline so far; graph-assisted fusion improves ranking quality over the current equal-weight hybrid, but it is not yet the best recall method. The answer audit confirms that better retrieval and stricter reporting-period/accounting-basis grounding are still required. No 9+/10 project-quality claim is made from this run.
