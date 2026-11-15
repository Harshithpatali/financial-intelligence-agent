"""Download the latest precomputed RAG index from this repository's GitHub Release."""
from __future__ import annotations

import json
import os
import shutil
import tempfile
import urllib.request
import zipfile
from pathlib import Path

REPOSITORY = os.getenv("RAG_INDEX_REPOSITORY", "Harshithpatali/financial-intelligence-agent")
ASSET_NAME = "tcs-rag-index.zip"
ROOT = Path(__file__).resolve().parents[2]
API_URL = f"https://api.github.com/repos/{REPOSITORY}/releases/latest"


def _request(url: str):
    request = urllib.request.Request(
        url,
        headers={
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "User-Agent": "tcs-financial-intelligence-agent",
        },
    )
    return urllib.request.urlopen(request, timeout=60)


def main() -> None:
    with _request(API_URL) as response:
        release = json.loads(response.read().decode("utf-8"))

    asset = next((item for item in release.get("assets", []) if item.get("name") == ASSET_NAME), None)
    if asset is None:
        raise RuntimeError(
            f"Latest GitHub Release has no {ASSET_NAME}. Run the 'Build and publish RAG index' workflow first."
        )

    with tempfile.TemporaryDirectory(prefix="tcs-rag-index-") as temporary:
        archive = Path(temporary) / ASSET_NAME
        with _request(asset["browser_download_url"]) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)

        with zipfile.ZipFile(archive) as bundle:
            members = bundle.namelist()
            required = {
                "backend/chroma_db/chroma.sqlite3",
                "backend/indexes/tcs_knowledge_graph.json",
            }
            if not required.issubset(set(members)):
                raise RuntimeError("RAG index archive is missing Chroma or graph index files.")

            for member in members:
                target = (ROOT / member).resolve()
                if not target.is_relative_to(ROOT.resolve()):
                    raise RuntimeError(f"Unsafe path in RAG index archive: {member}")

            chroma_path = ROOT / "backend" / "chroma_db"
            graph_path = ROOT / "backend" / "indexes" / "tcs_knowledge_graph.json"
            if chroma_path.exists():
                shutil.rmtree(chroma_path)
            graph_path.parent.mkdir(parents=True, exist_ok=True)
            if graph_path.exists():
                graph_path.unlink()
            bundle.extractall(ROOT)

    print(f"Installed RAG index from release {release.get('tag_name', 'unknown')}.")


if __name__ == "__main__":
    main()
