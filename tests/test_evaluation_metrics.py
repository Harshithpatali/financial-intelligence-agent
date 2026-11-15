from evaluation.metrics import recall_at_k,reciprocal_rank_at_k,citation_coverage

def test_recall_at_k():
 assert recall_at_k(["a","b","c"],{"a","c","d"},2)==1/3

def test_mrr_at_k():
 assert reciprocal_rank_at_k(["x","b","a"],{"a","b"},2)==0.5
 assert reciprocal_rank_at_k(["x"],{"a"},5)==0

def test_citation_coverage_ignores_invalid_source_ids():
 assert citation_coverage("Revenue rose [S1]. Margin fell [S3].",1)==0.5
