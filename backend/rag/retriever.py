"""Hybrid retrieval using cached dense, BM25 and graph indexes."""
import os
from threading import Lock
try:
 from dotenv import load_dotenv
 load_dotenv()
except ImportError:
 pass

_model=None
_records_cache=[]
_bm25_cache=None
_cache_count=-1
_cache_lock=Lock()

def get_model():
 global _model
 if _model is None:
  from sentence_transformers import SentenceTransformer
  _model=SentenceTransformer(os.getenv("EMBEDDING_MODEL","BAAI/bge-small-en-v1.5"),token=os.getenv("HF_TOKEN") or None)
 return _model

def _all_records(collection):
 result=collection.get(include=["documents","metadatas"])
 ids=result.get("ids") or [];docs=result.get("documents") or [];metas=result.get("metadatas") or []
 return [{"id":ids[i],"text":docs[i],"metadata":metas[i] or {}} for i in range(min(len(ids),len(docs)))]

def _get_lexical_index(collection):
 """Load the corpus and BM25 index once per collection size per process."""
 global _records_cache,_bm25_cache,_cache_count
 count=collection.count()
 if count!=_cache_count:
  with _cache_lock:
   if count!=_cache_count:
    records=_all_records(collection)
    if records:
     from rank_bm25 import BM25Okapi
     model=BM25Okapi([item["text"].lower().split() for item in records])
    else:model=None
    _records_cache,_bm25_cache,_cache_count=records,model,count
 return _records_cache,_bm25_cache

def retrieve(query,top_k=5):
 if not query or not query.strip():raise ValueError("Query must not be empty")
 if top_k<1 or top_k>50:raise ValueError("top_k must be between 1 and 50")
 from backend.rag.vector_store import get_collection
 from backend.rag.graph_index import graph_retrieve
 collection=get_collection();count=collection.count()
 if not count:return {"documents":[],"sources":[],"distances":[],"retrieval":[]}
 records,bm25=_get_lexical_index(collection)
 model=get_model()
 vector=collection.query(query_embeddings=[model.encode(query).tolist()],n_results=min(top_k*3,count),include=["documents","metadatas","distances"])
 dense={doc_id:1/(1+max(float(vector["distances"][0][i]),0)) for i,doc_id in enumerate(vector["ids"][0])}
 lexical={}
 if bm25 is not None:
  scores=bm25.get_scores(query.lower().split())
  lexical={records[i]["id"]:float(scores[i]) for i in range(len(records)) if scores[i]>0}
 graph_hits=graph_retrieve(query,top_k=top_k*3);graph_scores={item["id"]:item["score"] for item in graph_hits}
 rankings=[[key for key,_ in sorted(scores.items(),key=lambda item:item[1],reverse=True)] for scores in (dense,lexical,graph_scores)]
 fused={}
 for ranking in rankings:
  for rank,key in enumerate(ranking,1):fused[key]=fused.get(key,0)+1/(60+rank)
 by_id={item["id"]:item for item in records};ordered=sorted(fused,key=fused.get,reverse=True)[:top_k];selected=[by_id[key] for key in ordered if key in by_id]
 return {"documents":[item["text"] for item in selected],
 "sources":[{"source":item["metadata"].get("source","Unknown"),"page":item["metadata"].get("page") or None,"chunk_id":item["id"],"score":round(fused[item["id"]],6)} for item in selected],
 "distances":[fused[item["id"]] for item in selected],"retrieval":["dense","bm25","knowledge_graph"]}
