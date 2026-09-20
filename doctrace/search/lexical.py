"""
Keyword search over passages using BM25 (Okapi).

Passage ids are list positions, matching the point ids used in the Qdrant
collection, so the lexical and semantic result lists can be merged by id.
"""

import re

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"\w+")

_index: BM25Okapi | None = None
_passages: list[dict] | None = None


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def build_lexical_index(passages: list[dict]) -> None:
    """Index the given passages. Pass the same list, in the same order, that was
    uploaded to Qdrant, or ids will stop lining up."""
    global _index, _passages
    _passages = passages
    _index = BM25Okapi([_tokenize(p["text"]) for p in passages])


def _ensure_ready() -> None:
    if _index is None or _passages is None:
        import json
        from pathlib import Path
        source = Path("data/processed/passages.json")
        if not source.exists():
            raise RuntimeError(
                "Lexical index not built and data/processed/passages.json is missing "
                "-- call build_lexical_index() or run the indexing script first"
            )
        build_lexical_index(json.loads(source.read_text(encoding="utf-8")))


def lexical_search(query: str, top_k: int = 5) -> list[dict]:
    """Best top_k passages by BM25 score, as
    {"id", "score", "source_path", "header_path", "text"}."""
    _ensure_ready()

    scores = _index.get_scores(_tokenize(query))
    best = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

    return [
        {
            "id": i,
            "score": float(scores[i]),
            "source_path": _passages[i]["source_path"],
            "header_path": _passages[i]["header_path"],
            "text": _passages[i]["text"],
        }
        for i in best
    ]
