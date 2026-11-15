"""Rebuild Chroma vectors and the NetworkX knowledge graph."""
import hashlib
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from backend.rag.chunker import load_documents,chunk_documents
from backend.rag.vector_store import reset_collection
from backend.rag.graph_index import build_graph
def main():
 chunks=chunk_documents(load_documents())
 if not chunks:raise RuntimeError("No Markdown chunks found under ./knowledge")
 model=SentenceTransformer("BAAI/bge-small-en-v1.5"); collection=reset_collection(); graph_chunks=[]
 for start in tqdm(range(0,len(chunks),64),desc="Indexing"):
  batch=chunks[start:start+64]; embeddings=model.encode([x["text"] for x in batch],show_progress_bar=False).tolist(); ids=[]; metas=[]
  for item in batch:
   stable=hashlib.sha1(f'{item["source"]}|{item.get("page")}|{item["chunk_id"]}|{item["text"]}'.encode()).hexdigest(); ids.append(stable)
   metas.append({"source":item["source"],"chunk_id":str(item["chunk_id"]),"page":str(item["page"]) if item.get("page") is not None else "","doc_type":item.get("doc_type","document")})
   graph_chunks.append({"id":stable,**item})
  collection.upsert(ids=ids,embeddings=embeddings,documents=[x["text"] for x in batch],metadatas=metas)
 graph=build_graph(graph_chunks)
 print(f"Indexed {len(chunks)} chunks; graph contains {graph.number_of_nodes()} entities and {graph.number_of_edges()} co-mention relationships.")
if __name__=="__main__":main()
