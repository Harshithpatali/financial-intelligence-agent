"""Install the latest precomputed RAG index without using the GitHub Releases API."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = os.getenv("RAG_INDEX_REPOSITORY", "Harshithpatali/financial-intelligence-agent")
ASSET_NAME = "rag-index.zip"
ROOT = Path(__file__).resolve().parents[2]
DOWNLOAD_URL = f"https://github.com/{REPOSITORY}/releases/latest/download/{ASSET_NAME}"
GRAPH_MEMBER = "backend/indexes/tcs_knowledge_graph.json"
CHROMA_MEMBER = "backend/chroma_db/chroma.sqlite3"


def _download(url: str, destination: Path, attempts: int = 4) -> None:
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/octet-stream",
            "User-Agent": "financial-intelligence-agent-index-installer",
        },
    )
    last_error = None
    for attempt in range(attempts):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                with destination.open("wb") as output:
                    shutil.copyfileobj(response, output)
            if destination.stat().st_size == 0:
                raise RuntimeError("GitHub returned an empty RAG index archive.")
            return
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            if attempt == attempts - 1:
                break
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Unable to download the RAG index from {url}: {last_error}") from last_error


def _validate_archive(archive: Path, staging: Path) -> tuple[int, int]:
    import chromadb

    with zipfile.ZipFile(archive) as bundle:
        if bundle.testzip() is not None:
            raise RuntimeError("RAG index ZIP failed its integrity check.")
        members = set(bundle.namelist())
        if GRAPH_MEMBER not in members or CHROMA_MEMBER not in members:
            raise RuntimeError("RAG index archive is missing the Chroma database or knowledge graph.")
        for member in members:
            target = (staging / member).resolve()
            if not target.is_relative_to(staging.resolve()):
                raise RuntimeError(f"Unsafe path in RAG index archive: {member}")
        bundle.extractall(staging)

    graph_path = staging / GRAPH_MEMBER
    graph_data = json.loads(graph_path.read_text(encoding="utf-8"))
    nodes = graph_data.get("nodes", [])
    if not nodes:
        raise RuntimeError("Downloaded knowledge graph contains zero nodes.")

    chroma_path = staging / "backend" / "chroma_db"
    client = chromadb.PersistentClient(path=str(chroma_path))
    try:
        chunks = client.get_collection("tcs_knowledge_base").count()
    finally:
        del client
    if chunks <= 0:
        raise RuntimeError("Downloaded Chroma collection contains zero document chunks.")
    return len(nodes), chunks


def _install_one(source: Path, target: Path) -> tuple[Path, Path | None]:
    """Copy to the target filesystem first, then rename locally to avoid EXDEV."""
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-new-", dir=target.parent))
    staged_target = temporary / target.name
    backup_target = temporary / f"{target.name}.backup"
    try:
        if source.is_dir():
            shutil.copytree(source, staged_target)
        else:
            shutil.copy2(source, staged_target)

        had_old = target.exists()
        if had_old:
            os.replace(target, backup_target)
        try:
            os.replace(staged_target, target)
        except Exception:
            if had_old and backup_target.exists():
                os.replace(backup_target, target)
            raise
        return target, backup_target if had_old else None
    finally:
        # Keep backup until both index components have installed successfully.
        if not (temporary / f"{target.name}.backup").exists():
            shutil.rmtree(temporary, ignore_errors=True)


def _install(staging: Path) -> None:
    source_chroma = staging / "backend" / "chroma_db"
    source_graph = staging / GRAPH_MEMBER
    target_chroma = ROOT / "backend" / "chroma_db"
    target_graph = ROOT / GRAPH_MEMBER
    installed: list[tuple[Path, Path | None]] = []
    try:
        installed.append(_install_one(source_chroma, target_chroma))
        installed.append(_install_one(source_graph, target_graph))
    except Exception:
        for target, backup in reversed(installed):
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise
    for _, backup in installed:
        if backup is not None:
            shutil.rmtree(backup.parent, ignore_errors=True)


def main() -> None:
    ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".rag-index-", dir=ROOT) as temporary:
        staging = Path(temporary)
        archive = staging / ASSET_NAME
        _download(DOWNLOAD_URL, archive)
        nodes, chunks = _validate_archive(archive, staging)
        _install(staging)
    print(f"Installed latest RAG index: {chunks} document chunks, {nodes} knowledge-graph nodes.")


if __name__ == "__main__":
    main()
