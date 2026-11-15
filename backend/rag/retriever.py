"""Hybrid retrieval using dense embeddings, BM25 and graph expansion."""
import os
try:
 from dotenv import load_dotenv
 load_dotenv()
except ImportError:
 pass
_model=None
def get_model():
 global _model
 if _model is None:
  from sentence_transformers import SentenceTransformer
  _model=SentenceTransformer(os.getenv("EMBEDDING_MODEL","BAAI/bge-small-en-v1.5"),token=os.getenv("HF_TOKEN") or None)
 return _model
def _all_records(collection):
 r=collection.get(include=["documents","metadatas"]);ids=r.get("ids") or [];docs=r.get("documents") or [];metas=r.get("metadatas") or []
 return [{"id":ids[i],"text":docs[i],"metadata":metas[i] or {}} for i in range(min(len(ids),len(docs)))]
def _bm25_scores(query,records):
 if not records:return {}
 from rank_bm25 import BM25Okapi
 scores=BM25Okapi([r["text"].lower().split() for r in records]).get_scores(query.lower().split())
 return {records[i]["id"]:float(scores[i]) for i in range(len(records)) if scores[i]>0}
def retrieve(query,top_k=5):
 if not query or not query.strip():raise ValueError("Query must not be empty")
 if top_k<1 or top_k>50:raise ValueError("top_k must be between 1 and 50")
 from backend.rag.vector_store import get_collection
 from backend.rag.graph_index import graph_retrieve
 collection=get_collection();count=collection.count()
 if not count:return {"documents":[],"sources":[],"distances":[],"retrieval":[]}
 records=_all_records(collection);model=get_model()
 vector=collection.query(query_embeddings=[model.encode(query).tolist()],n_results=min(top_k*3,count),include=["documents","metadatas","distances"])
 dense={doc_id:1/(1+max(float(vector["distances"][0][i]),0)) for i,doc_id in enumerate(vector["ids"][0])}
 lexical=_bm25_scores(query,records);graph_hits=graph_retrieve(query,top_k=top_k*3);graph_scores={x["id"]:x["score"] for x in graph_hits}
 rankings=[[k for k,_ in sorted(s.items(),key=lambda x:x[1],reverse=True)] for s in (dense,lexical,graph_scores)];fused={}
 for ranking in rankings:
  for rank,key in enumerate(ranking,1):fused[key]=fused.get(key,0)+1/(60+rank)
 by_id={r["id"]:r for r in records};ordered=sorted(fused,key=fused.get,reverse=True)[:top_k];selected=[by_id[k] for k in ordered if k in by_id]
 return {"documents":[r["text"] for r in selected],"sources":[{"source":r["metadata"].get("source","Unknown"),"page":r["metadata"].get("page") or None,"chunk_id":r["id"],"score":round(fused[r["id"]],6)} for r in selected],"distances":[fused[r["id"]] for r in selected],"retrieval":["dense","bm25","knowledge_graph"]}
