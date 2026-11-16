import pytest
from backend.rag.document_store import DocumentIndex

def test_sqlite_fts5_roundtrip_and_search(tmp_path):
    index=DocumentIndex(tmp_path/"index.sqlite3")
    chunks=[{"id":"a","text":"TCS revenue increased to 250000 crore in FY2026","source":"annual.md","page":7,"chunk_id":1},
            {"id":"b","text":"Operating cash flow and total assets were reported","source":"cash.md","page":3,"chunk_id":2}]
    assert index.replace(chunks)==2
    assert index.count()==2
    hits=index.search("TCS revenue FY2026",5)
    assert hits and hits[0]["id"]=="a"
    record=index.get(["b"])
    assert record["documents"][0].startswith("Operating cash flow")
    assert record["metadatas"][0]["source"]=="cash.md"

def test_search_empty_and_missing_index(tmp_path):
    index=DocumentIndex(tmp_path/"missing.sqlite3")
    assert index.search("",5)==[]
    with pytest.raises(FileNotFoundError): index.count()
