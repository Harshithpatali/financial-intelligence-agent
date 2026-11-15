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


def _copy_directory_contents(source: Path, destination: Path) -> None:
    """Copy into the mounted Chroma directory; never rename its mount point."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)


def _restore_directory(backup: Path, destination: Path) -> None:
    if destination.exists():
        for item in destination.iterdir():
            if item.is_dir():
                shutil.rmtree(item)
            else:
                item.unlink()
    _copy_directory_contents(backup, destination)


def _install(staging: Path) -> None:
    source_chroma = staging / "backend" / "chroma_db"
    source_graph = staging / GRAPH_MEMBER
    target_chroma = ROOT / "backend" / "chroma_db"
    target_graph = ROOT / GRAPH_MEMBER

    backup_chroma = staging / "backup_chroma"
    had_chroma = target_chroma.exists()
    if had_chroma:
        shutil.copytree(target_chroma, backup_chroma)

    target_graph.parent.mkdir(parents=True, exist_ok=True)
    backup_graph = staging / "backup_graph.json"
    had_graph = target_graph.exists()
    if had_graph:
        shutil.copy2(target_graph, backup_graph)

    try:
        # Chroma may be a Render persistent-disk mount. Copy into it rather
        # than renaming/replacing the mount directory across filesystems.
        _copy_directory_contents(source_chroma, target_chroma)

        # The graph file is replaced atomically within its own filesystem.
        graph_temp = target_graph.with_name(target_graph.name + ".new")
        shutil.copy2(source_graph, graph_temp)
        os.replace(graph_temp, target_graph)
    except Exception:
        if had_chroma:
            _restore_directory(backup_chroma, target_chroma)
        if had_graph:
            shutil.copy2(backup_graph, target_graph)
        elif target_graph.exists():
            target_graph.unlink()
        raise


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
