"""
Heading-aware markdown splitter.

Two strategies live here:
1. `split_by_headings` cuts a markdown document at heading lines (# .. ######) and
   never treats a `#` inside a fenced code block as a heading. Each passage records
   the full heading trail above it plus the explicit `{ #slug }` anchor, if one exists.
2. `split_fixed_window` is the deliberately naive sliding-window splitter, kept only
   as the comparison point for the ablation grid.
"""

import re

HEADING_RE = re.compile(r"^(#{1,6})\s+(.*)$")
FENCE_RE = re.compile(r"^```")
# The docs put explicit slugs on headings, e.g. "# First Steps { #first-steps }"
ANCHOR_RE = re.compile(r"\{\s*#([\w-]+)\s*\}")


def _peel_anchor(raw_heading: str) -> tuple[str, str | None]:
    """Separate a heading's visible text from its trailing `{ #slug }` marker.

    '## Data <dfn ...>conversion</dfn> { #data-conversion }' becomes
    ('Data <dfn ...>conversion</dfn>', 'data-conversion'). Headings without a
    marker return None for the slug.
    """
    match = ANCHOR_RE.search(raw_heading)
    slug = match.group(1) if match else None
    visible = ANCHOR_RE.sub("", raw_heading).strip()
    return visible, slug


def _emit(buffer: list[str], trail: list[tuple[int, str, "str | None"]],
          source_path: str, out: list[dict]) -> None:
    body = "\n".join(buffer).strip()
    if body:
        innermost_slug = trail[-1][2] if trail else None
        out.append({
            "source_path": source_path,
            "header_path": " > ".join(entry[1] for entry in trail),
            "anchor": innermost_slug,
            "text": body,
        })


def split_by_headings(doc_text: str, source_path: str) -> list[dict]:
    """Cut a document at every heading that sits outside a code fence.

    Returns dicts shaped {"source_path", "header_path", "anchor", "text"}.
    """
    passages: list[dict] = []
    trail: list[tuple[int, str, "str | None"]] = []
    buffer: list[str] = []
    inside_fence = False

    for line in doc_text.splitlines():
        if FENCE_RE.match(line):
            inside_fence = not inside_fence
            buffer.append(line)
            continue

        heading = None if inside_fence else HEADING_RE.match(line)
        if heading:
            _emit(buffer, trail, source_path, passages)
            depth = len(heading.group(1))
            visible, slug = _peel_anchor(heading.group(2))
            # drop siblings and deeper entries so the trail stays a proper ancestry
            while trail and trail[-1][0] >= depth:
                trail.pop()
            trail.append((depth, visible, slug))
            buffer = [line]
        else:
            buffer.append(line)

    _emit(buffer, trail, source_path, passages)
    return passages


def split_fixed_window(doc_text: str, source_path: str,
                       size: int = 500, overlap: int = 50) -> list[dict]:
    """Structure-blind baseline: overlapping windows counted in words.

    Words stand in for tokens because this is only a comparison baseline. It will
    happily cut tables and code blocks in half, which is the point.
    """
    words = doc_text.split()
    passages = []
    stride = max(size - overlap, 1)
    for start in range(0, len(words), stride):
        window = words[start:start + size]
        if not window:
            continue
        passages.append({
            "source_path": source_path,
            "header_path": "",
            "anchor": None,
            "text": " ".join(window),
        })
        if start + size >= len(words):
            break
    return passages
