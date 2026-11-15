from pathlib import Path
import re

PAGE_MARKER = re.compile(r"<!--\s*PAGE:\s*(\d+)\s*-->", re.I)

def _frontmatter(text):
    metadata = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) == 3:
            for line in parts[1].splitlines():
                if ":" in line:
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip().strip('"\'')
    return metadata

def load_documents(root="knowledge"):
    docs = []
    for path in sorted(Path(root).rglob("*.md")):
        content = path.read_text(encoding="utf-8")
        if content.strip():
            docs.append({"source": path.as_posix(), "content": content, "metadata": _frontmatter(content)})
    return docs

def _split_text(text, chunk_size, overlap):
    text = re.sub(r"\s+", " ", text).strip()
    chunks, start = [], 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        if end < len(text):
            boundary = max(text.rfind(". ", start, end), text.rfind("; ", start, end), text.rfind(" ", start, end))
            if boundary > start + chunk_size // 2:
                end = boundary + 1
        piece = text[start:end].strip()
        if piece: chunks.append(piece)
        if end >= len(text): break
        start = max(end - overlap, start + 1)
    return chunks

def chunk_documents(documents, chunk_size=1000, chunk_overlap=150):
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("Require chunk_size > chunk_overlap >= 0")
    chunks = []
    for doc in documents:
        content, metadata = doc["content"], doc.get("metadata", {})
        matches = list(PAGE_MARKER.finditer(content))
        sections = []
        if matches:
            for i, match in enumerate(matches):
                end = matches[i+1].start() if i+1 < len(matches) else len(content)
                sections.append((int(match.group(1)), content[match.end():end]))
        else:
            sections = [(None, content)]
        for page, section in sections:
            section = re.sub(r"^---.*?---\s*", "", section, count=1, flags=re.S)
            for piece in _split_text(section, chunk_size, chunk_overlap):
                chunks.append({"text": piece, "source": doc["source"], "page": page,
                               "chunk_id": len(chunks), "doc_type": metadata.get("type", "document"),
                               "year": metadata.get("year")})
    return chunks
