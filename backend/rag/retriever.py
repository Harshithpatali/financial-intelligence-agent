"""Bounded lexical + graph retrieval using the production SQLite FTS5 index."""
from backend.rag.document_store import get_index
from backend.rag.graph_index import graph_retrieve
def _rrf(*rankings,limit):
    scores={}
    for ranking in rankings:
        for rank,key in enumerate(ranking,1): scores[key]=scores.get(key,0.0)+1.0/(60+rank)
    return [k for k,_ in sorted(scores.items(),key=lambda x:(-x[1],x[0]))[:limit]],scores
def retrieve(query,top_k=3,query_embedding=None):
    if not query or not query.strip(): raise ValueError("Query must not be empty")
    if not isinstance(top_k,int) or not 1<=top_k<=10: raise ValueError("top_k must be between 1 and 10")
    if query_embedding is not None: raise ValueError("Dense embeddings are not configured; retrieval uses SQLite FTS5 and the knowledge graph.")
    index=get_index()
    if index.count()==0: return {"documents":[],"sources":[],"distances":[],"retrieval":[]}
    lexical=index.search(query,limit=top_k*8); graph=graph_retrieve(query,top_k=top_k*8)
    lexical_rank=[x["id"] for x in lexical]; graph_rank=[x["id"] for x in graph]
    ordered,scores=_rrf(lexical_rank,graph_rank,limit=top_k); found=index.get(ordered)
    records={k:{"text":d,"metadata":m} for k,d,m in zip(found["ids"],found["documents"],found["metadatas"])}
    keys=[k for k in ordered if k in records]
    return {"documents":[records[k]["text"][:5000] for k in keys],
      "sources":[{"source":records[k]["metadata"].get("source","Unknown"),"page":records[k]["metadata"].get("page") or None,"chunk_id":k,"score":round(scores[k],6)} for k in keys],
      "distances":[scores[k] for k in keys],
      "retrieval":(["bm25"] if lexical_rank else [])+(["knowledge_graph"] if graph_rank else [])}
