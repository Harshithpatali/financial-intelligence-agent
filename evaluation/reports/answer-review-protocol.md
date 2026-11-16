# TCS answer-review protocol

The benchmark contains 50 source-anchored financial and operating questions. The retrieval run compares dense-only, BM25-only, dense+BM25 reciprocal-rank fusion, and dense+BM25+graph fusion at top-5.

## Required reviewer checks

Review every answer in `evaluation/results/answer_audit_packet.json`. For each question, set the five `human_*_review` fields to `pass`, `partial`, or `fail`, and add reviewer notes.

1. **Answer relevance:** Does it answer the exact metric, entity, reporting period and requested comparison?
2. **Faithfulness:** Is every material claim supported by the cited excerpt, without invented causal explanations?
3. **Numerical accuracy:** Verify amount, currency, unit, sign, period and accounting basis. Distinguish consolidated from standalone figures.
4. **Citation correctness:** Does each citation support the claim immediately before it? Citation presence alone is not correctness.
5. **Page traceability:** Can the cited excerpt be located in the original source file and page? A missing page is a failed traceability check, not an inferred page.

## Rules

- Do not convert machine token matches into a human accuracy score.
- Do not mark a result as reviewed until the original disclosure has been opened and checked.
- Keep retrieval metrics separate from answer-quality metrics.
- Report the number reviewed and the denominator for each quality rate.
- Preserve abstentions as outcomes; assess whether the indexed evidence genuinely lacked support.
- Record conflicts, including different values caused by period, unit, entity or accounting basis.

The runner attempts to resolve explicit `Page N of M` and `<!-- PAGE: N -->` markers. It reports page traceability as unknown when no page marker can be resolved.
