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
