"""SQLite FTS5 document index; no vector or external embedding dependency."""
from __future__ import annotations
import os, re, sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INDEX_PATH = ROOT / "backend" / "indexes" / "tcs_knowledge.sqlite3"
TOKEN_RE = re.compile(r"[a-z0-9]{2,}", re.I)
STOP_WORDS = {"the","and","for","with","from","what","how","was","were","are","that","this","has","have","had","does","did","its","into","about"}

class DocumentIndex:
    def __init__(self, path=None):
        self.path = Path(path or os.getenv("TCS_SQLITE_INDEX_PATH", DEFAULT_INDEX_PATH)).resolve()
    def connect(self):
        if not self.path.exists():
            raise FileNotFoundError(f"RAG document index not found: {self.path}. Build/download the index first.")
        db=sqlite3.connect(str(self.path),timeout=20); db.row_factory=sqlite3.Row; return db
    def count(self):
        with self.connect() as db: return int(db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0])
    def search(self, query, limit=40):
        if not query or not query.strip(): return []
        terms=[]
        for token in TOKEN_RE.findall(query.casefold()):
            if token not in STOP_WORDS and token not in terms: terms.append(token)
            if len(terms)>=12: break
        if not terms: return []
        match=" OR ".join('"'+term.replace('"','""')+'"' for term in terms)
        with self.connect() as db:
            rows=db.execute("""SELECT c.id,c.text,c.source,c.page,c.chunk_id,c.doc_type,bm25(chunk_fts) AS rank
                FROM chunk_fts JOIN chunks c ON c.id=chunk_fts.id
                WHERE chunk_fts MATCH ? ORDER BY rank ASC,c.id ASC LIMIT ?""",
                (match,max(1,min(int(limit),500)))).fetchall()
        return [{"id":r["id"],"text":r["text"],"metadata":{"source":r["source"],"page":r["page"] or "",
                "chunk_id":r["chunk_id"],"doc_type":r["doc_type"]},"score":float(-r["rank"])} for r in rows]
    def get(self, ids=None):
        with self.connect() as db:
            if ids is None: rows=db.execute("SELECT * FROM chunks ORDER BY rowid").fetchall()
            elif not ids: rows=[]
            else:
                q=",".join("?" for _ in ids)
                found=db.execute(f"SELECT * FROM chunks WHERE id IN ({q})",ids).fetchall()
                by_id={r["id"]:r for r in found}; rows=[by_id[k] for k in ids if k in by_id]
        return {"ids":[r["id"] for r in rows],"documents":[r["text"] for r in rows],
                "metadatas":[{"source":r["source"],"page":r["page"] or "","chunk_id":r["chunk_id"],"doc_type":r["doc_type"]} for r in rows]}
    def replace(self,chunks):
        self.path.parent.mkdir(parents=True,exist_ok=True)
        temp=self.path.with_suffix(self.path.suffix+".new")
        if temp.exists(): temp.unlink()
        db=sqlite3.connect(str(temp))
        try:
            db.execute("PRAGMA journal_mode=DELETE")
            db.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY,text TEXT NOT NULL,source TEXT NOT NULL,page TEXT,chunk_id TEXT,doc_type TEXT NOT NULL)")
            db.execute("CREATE VIRTUAL TABLE chunk_fts USING fts5(id UNINDEXED,text,tokenize='unicode61')")
            rows=[(x["id"],x["text"],x.get("source","Unknown"),str(x["page"]) if x.get("page") is not None else None,str(x.get("chunk_id","")),x.get("doc_type","document")) for x in chunks]
            db.executemany("INSERT INTO chunks VALUES(?,?,?,?,?,?)",rows)
            db.executemany("INSERT INTO chunk_fts(id,text) VALUES(?,?)",[(x["id"],x["text"]) for x in chunks])
            db.execute("CREATE INDEX idx_chunks_source ON chunks(source)"); db.commit()
            n=db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]; f=db.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
            if n!=len(chunks) or f!=n: raise RuntimeError(f"Index count mismatch: chunks={n}, fts={f}")
            db.execute("SELECT id FROM chunk_fts WHERE chunk_fts MATCH 'TCS' LIMIT 1").fetchall()
        finally: db.close()
        os.replace(temp,self.path); return len(chunks)

_INDEX=None
def get_index():
    global _INDEX
    if _INDEX is None: _INDEX=DocumentIndex()
    return _INDEX
def reset_index():
    global _INDEX
    _INDEX=DocumentIndex(); return _INDEX
