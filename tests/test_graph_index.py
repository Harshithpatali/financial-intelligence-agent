from backend.rag.graph_index import extract_entities, build_graph, graph_retrieve


def test_entity_extraction_uses_word_boundaries():
    entities = extract_entities("TCS total revenue increased in FY2025")
    assert {"Revenue", "Tata Consultancy Services", "FY2025"}.issubset(set(entities))
    assert "Net Income" not in extract_entities("internet income unrelated")


def test_graph_builds_temporal_and_provenance_edges(tmp_path):
    path = tmp_path / "graph.json"
    graph = build_graph([
        {"id": "a", "source": "annual.pdf", "page": 2, "text": "TCS revenue increased in FY2024"},
        {"id": "b", "source": "annual.pdf", "page": 3, "text": "TCS revenue decreased in FY2025"},
    ], path)
    assert "MENTIONS" in graph["Revenue"]["CHUNK:a"]["relations"]
    assert "HAS_EVIDENCE_IN_PERIOD" in graph["Revenue"]["FY2024"]["relations"]
    assert "PRECEDES" in graph["FY2024"]["FY2025"]["relations"]
    assert graph.nodes["TREND:INCREASED:a"]["verified"] is False
    assert graph_retrieve("TCS revenue FY2024", path=path)


def test_cooccurrence_is_not_causality(tmp_path):
    graph = build_graph([
        {"id": "c1", "source": "x.md", "text": "Revenue and operating income in FY2025"}
    ], tmp_path / "graph.json")
    relations = graph["Revenue"]["Operating Income"]["relations"]
    assert "CO_MENTIONED_WITHIN_CHUNK" in relations
    assert "CAUSES" not in relations


def test_financial_statement_aliases_support_targeted_graph_retrieval():
    entities = set(extract_entities(
        "TCS profit before tax, basic and diluted EPS, total assets, "
        "net cash flows generated from operating activities in FY2026"
    ))
    assert {
        "Profit Before Tax",
        "Earnings Per Share",
        "Total Assets",
        "Operating Cash Flow",
        "FY2026",
    }.issubset(entities)


def test_graph_retrieves_profit_before_tax_evidence(tmp_path):
    path = tmp_path / "financial-graph.json"
    build_graph([
        {"id": "pbt", "source": "results.md", "text": "TCS profit before tax was 65,487 crore in FY2026"}
    ], path)
    results = graph_retrieve("TCS profit before tax FY2026", path=path)
    assert results
    assert results[0]["id"] == "pbt"
