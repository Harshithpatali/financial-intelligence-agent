"""Download and install the published SQLite FTS5 + graph index."""
import json,os,shutil,sqlite3,tempfile,time,urllib.error,urllib.request,zipfile
from pathlib import Path
REPOSITORY=os.getenv("RAG_INDEX_REPOSITORY","Harshithpatali/financial-intelligence-agent")
ASSET_NAME="rag-index.zip"; ROOT=Path(__file__).resolve().parents[2]
DOWNLOAD_URL=f"https://github.com/{REPOSITORY}/releases/latest/download/{ASSET_NAME}"
DB_MEMBER="backend/indexes/tcs_knowledge.sqlite3"; GRAPH_MEMBER="backend/indexes/tcs_knowledge_graph.json"
def _download(url,destination,attempts=4):
    request=urllib.request.Request(url,headers={"Accept":"application/octet-stream","User-Agent":"financial-intelligence-agent-index-installer"}); last=None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request,timeout=120) as response,destination.open("wb") as output: shutil.copyfileobj(response,output)
            if destination.stat().st_size==0: raise RuntimeError("GitHub returned an empty RAG index archive.")
            return
        except (urllib.error.URLError,TimeoutError,OSError) as exc:
            last=exc
            if attempt==attempts-1: break
            time.sleep(2**attempt)
    raise RuntimeError(f"Unable to download RAG index: {last}") from last
def _validate_archive(archive,staging):
    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None: raise RuntimeError("RAG index ZIP failed integrity check.")
        members=set(bundle.namelist())
        if not {DB_MEMBER,GRAPH_MEMBER}.issubset(members): raise RuntimeError("Archive is missing SQLite FTS5 database or graph.")
        for member in members:
            if not (staging/member).resolve().is_relative_to(staging.resolve()): raise RuntimeError(f"Unsafe archive path: {member}")
        bundle.extractall(staging)
    with sqlite3.connect(str(staging/DB_MEMBER)) as db:
        chunks=db.execute("SELECT COUNT(*) FROM chunks").fetchone()[0]; fts=db.execute("SELECT COUNT(*) FROM chunk_fts").fetchone()[0]
        db.execute("SELECT id FROM chunk_fts WHERE chunk_fts MATCH 'TCS' LIMIT 1").fetchall()
    graph=json.loads((staging/GRAPH_MEMBER).read_text(encoding="utf-8")); nodes=graph.get("nodes",[])
    if chunks<=0 or fts!=chunks: raise RuntimeError(f"Invalid index counts: chunks={chunks}, fts={fts}")
    if not nodes: raise RuntimeError("Downloaded knowledge graph contains zero nodes.")
    return len(nodes),chunks
def _install(staging):
    pairs=[(staging/DB_MEMBER,ROOT/DB_MEMBER),(staging/GRAPH_MEMBER,ROOT/GRAPH_MEMBER)]; backups=[]; installed=[]
    try:
        for source,target in pairs:
            target.parent.mkdir(parents=True,exist_ok=True); backup=target.with_name(target.name+".backup")
            if target.exists(): shutil.copy2(target,backup); backups.append((backup,target))
            temp=target.with_name(target.name+".new"); shutil.copy2(source,temp); os.replace(temp,target); installed.append(target)
    except Exception:
        for target in reversed(installed):
            backup=next((b for b,t in backups if t==target),None)
            if backup and backup.exists(): os.replace(backup,target)
            elif target.exists(): target.unlink()
        raise
    finally:
        for backup,_ in backups:
            if backup.exists(): backup.unlink()
def main():
    ROOT.mkdir(parents=True,exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rag-index-",dir=ROOT) as tmp:
        staging=Path(tmp); archive=staging/ASSET_NAME; _download(DOWNLOAD_URL,archive)
        nodes,chunks=_validate_archive(archive,staging); _install(staging)
    print(f"Installed SQLite FTS5 RAG index: {chunks} chunks, {nodes} graph nodes.")
if __name__=="__main__": main()
