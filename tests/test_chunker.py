from backend.rag.chunker import chunk_documents
def test_page_metadata():
 docs=[{"source":"report.md","content":"---\ntype: annual_report\n---\n<!-- PAGE: 4 -->\nRevenue increased.","metadata":{"type":"annual_report"}}]
 chunks=chunk_documents(docs,chunk_size=100,chunk_overlap=10)
 assert chunks and chunks[0]["page"]==4 and chunks[0]["source"]=="report.md"
def test_empty_docs():assert chunk_documents([{"source":"empty.md","content":""}])==[]
