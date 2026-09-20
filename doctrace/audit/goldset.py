"""
Gold question set: loading, validation, and ground-truth resolution.

Reads gold/gold_pairs.json and converts each pair's `source_chunks` references
(such as "tutorial/path-params.md#path-parameters-with-types") into real passage
indices in data/processed/passages.json. Resolution goes through the anchors captured
during splitting (doctrace/corpus/splitter.py); guessing a slug from the heading
text proved unreliable for headings containing inline HTML or markdown.
"""

import json
from pathlib import Path

GOLD_PATH = Path("gold/gold_pairs.json")
PASSAGES_PATH = Path("data/processed/passages.json")


def load_gold_pairs() -> list[dict]:
    """Read the hand-built question/answer pairs from gold/gold_pairs.json."""
    return json.loads(GOLD_PATH.read_text(encoding="utf-8"))["pairs"]


def _load_passages() -> list[dict]:
    return json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))


def build_anchor_index(passages: list[dict]) -> dict[tuple[str, str], int]:
    """Map (source_path, anchor) to the passage's list index."""
    index = {}
    for i, p in enumerate(passages):
        if p.get("anchor"):
            index[(p["source_path"], p["anchor"])] = i
    return index


def attach_gold_ids(pairs: list[dict], passages: list[dict] | None = None) -> list[dict]:
    """Add a `ground_truth_ids` list[int] to each pair by resolving its
    source_chunks references to passage anchors.

    no_answer pairs get an empty list since they have no references by design.
    Any unresolvable single_hop/multi_hop reference raises: swallowing it would
    quietly skew the retrieval metrics.
    """
    if passages is None:
        passages = _load_passages()
    index = build_anchor_index(passages)

    out = []
    for pair in pairs:
        ids = []
        for ref in pair["source_chunks"]:
            path, anchor = ref.split("#", 1) if "#" in ref else (ref, None)
            if (path, anchor) not in index:
                raise ValueError(
                    f"QA pair {pair['id']!r} references unresolved passage {ref!r} "
                    f"-- check anchor capture in the splitter or the pair's reference"
                )
            ids.append(index[(path, anchor)])
        out.append({**pair, "ground_truth_ids": ids})
    return out


def check_gold_pairs(pairs: list[dict]) -> None:
    """Basic integrity checks: non-blank question and answer, known category,
    unique ids, unique question text, and source_chunks that suit the category
    (empty for no_answer, populated otherwise)."""
    categories = {"single_hop", "multi_hop", "no_answer"}
    ids_seen = set()
    questions_seen = set()

    for pair in pairs:
        assert pair["question"].strip(), f"{pair['id']}: empty question"
        assert pair["answer"].strip(), f"{pair['id']}: empty answer"
        assert pair["category"] in categories, f"{pair['id']}: invalid category {pair['category']!r}"

        assert pair["id"] not in ids_seen, f"duplicate id: {pair['id']}"
        ids_seen.add(pair["id"])

        normalized = pair["question"].strip().lower()
        assert normalized not in questions_seen, f"{pair['id']}: duplicate question text"
        questions_seen.add(normalized)

        if pair["category"] == "no_answer":
            assert not pair["source_chunks"], f"{pair['id']}: no_answer pair should have empty source_chunks"
        else:
            assert pair["source_chunks"], f"{pair['id']}: {pair['category']} pair has empty source_chunks"
