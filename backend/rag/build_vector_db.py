"""Build the SQLite FTS5 lexical index and NetworkX evidence graph."""
import hashlib
from pathlib import Path
from backend.rag.chunker import load_documents,chunk_documents
from backend.rag.document_store import reset_index
from backend.rag.graph_index import build_graph
ROOT=Path(__file__).resolve().parents[2]
def stable_chunk_id(item):
    return hashlib.sha1(f'{item["source"]}|{item.get("page")}|{item["chunk_id"]}|{item["text"]}'.encode("utf-8")).hexdigest()
def main():
    chunks=chunk_documents(load_documents())
    if not chunks: raise RuntimeError("No Markdown chunks found under ./knowledge")
    graph_chunks=[{"id":stable_chunk_id(x),**x} for x in chunks]
    count=reset_index().replace(graph_chunks); graph=build_graph(graph_chunks)
    print(f"Indexed {count} chunks with SQLite FTS5; graph contains {graph.number_of_nodes()} entities and {graph.number_of_edges()} relationships.")
if __name__=="__main__": main()
