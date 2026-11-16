"""Page-level source traceability for extracted TCS disclosure text."""
from __future__ import annotations

import re

PAGE_RE = re.compile(
    r"<!--\s*PAGE:\s*(\d+)\s*-->|\bPage\s+(\d+)\s+of\s+\d+\b",
    re.IGNORECASE,
)


def _normalized_with_positions(text: str) -> tuple[str, list[int]]:
    """Collapse whitespace while retaining a map to original character offsets."""
    chars = []
    offsets = []
    in_space = False
    for index, char in enumerate(str(text)):
        if char.isspace():
            if chars and not in_space:
                chars.append(" ")
                offsets.append(index)
            in_space = True
        else:
            chars.append(char.casefold())
            offsets.append(index)
            in_space = False
    return "".join(chars).strip(), offsets


def page_for_anchor(source_text: str, anchor: str) -> int | None:
    """Return the nearest preceding explicit page marker for an evidence anchor."""
    normalized, offsets = _normalized_with_positions(source_text)
    target, _ = _normalized_with_positions(anchor)
    if not target:
        return None
    position = normalized.find(target)
    if position < 0:
        return None
    original_position = offsets[position]
    page = None
    for match in PAGE_RE.finditer(source_text[:original_position]):
        page = int(match.group(1) or match.group(2))
    return page


def page_for_excerpt(source_text: str, excerpt: str) -> int | None:
    """Resolve a retrieved excerpt to the page marker preceding it in its source."""
    normalized_source, offsets = _normalized_with_positions(source_text)
    normalized_excerpt, _ = _normalized_with_positions(excerpt)
    if len(normalized_excerpt) < 40:
        return None
    position = normalized_source.find(normalized_excerpt[: min(160, len(normalized_excerpt))])
    if position < 0:
        return None
    original_position = offsets[position]
    page = None
    for match in PAGE_RE.finditer(source_text[:original_position]):
        page = int(match.group(1) or match.group(2))
    return page
