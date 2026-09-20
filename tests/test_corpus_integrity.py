"""
Checks on the corpus that ships with the repo.

The passage file is tracked because the service builds its keyword index from it at
startup, and because every saved benchmark was measured against it. These tests run
offline and fail if that file is edited, truncated or regenerated from different docs
without the manifest being updated alongside it.

Run with:
    python -m pytest tests/test_corpus_integrity.py -v
"""

import json
from pathlib import Path

import pytest

from doctrace.audit.goldset import attach_gold_ids, check_gold_pairs, load_gold_pairs
from doctrace.corpus.fetcher import DOCS_COMMIT, SNIPPET_RE
from doctrace.corpus.manifest import fingerprint, read_manifest

PASSAGES_PATH = Path("data/processed/passages.json")
EXPECTED_PASSAGES = 756


@pytest.fixture(scope="module")
def passages():
    return json.loads(PASSAGES_PATH.read_text(encoding="utf-8"))


def test_manifest_matches_the_tracked_corpus(passages):
    manifest = read_manifest()
    assert manifest is not None, "data/processed/corpus_manifest.json is missing"
    assert manifest["passage_count"] == len(passages)
    assert manifest["fingerprint"] == fingerprint(passages)


def test_manifest_records_the_pinned_docs_commit():
    """The pin and the corpus have to move together, or the numbers stop matching."""
    assert read_manifest()["docs_commit"] == DOCS_COMMIT


def test_corpus_size_is_what_the_write_up_claims(passages):
    assert len(passages) == EXPECTED_PASSAGES


def test_every_passage_has_the_fields_search_depends_on(passages):
    for p in passages:
        assert p["source_path"] and p["text"].strip()
        assert isinstance(p["header_path"], str)
        assert "anchor" in p


def test_no_passage_has_an_unclosed_code_fence(passages):
    broken = [p["source_path"] for p in passages if p["text"].count("```") % 2]
    assert not broken, f"unclosed code fence in: {broken[:5]}"


def test_no_unresolved_snippet_markers(passages):
    """Match the real `{* path *}` marker, not a bare '{*': Python's `{**kwargs}`
    unpacking appears in the code samples and is not a marker."""
    assert not [p for p in passages if SNIPPET_RE.search(p["text"])]
    assert not [p for p in passages if "[MISSING SNIPPET" in p["text"]]


def test_anchors_are_unique_so_gold_references_are_unambiguous(passages):
    keys = [(p["source_path"], p["anchor"]) for p in passages if p.get("anchor")]
    assert len(keys) == len(set(keys))


def test_gold_set_is_valid_and_resolves_against_the_tracked_corpus(passages):
    pairs = load_gold_pairs()
    check_gold_pairs(pairs)
    assert len(pairs) == 80
    resolved = attach_gold_ids(pairs, passages)
    for pair in resolved:
        if pair["category"] == "no_answer":
            assert pair["ground_truth_ids"] == []
        else:
            assert pair["ground_truth_ids"], f"{pair['id']} resolved to nothing"
            assert all(0 <= i < len(passages) for i in pair["ground_truth_ids"])


def test_multi_hop_questions_really_need_more_than_one_passage():
    pairs = load_gold_pairs()
    for pair in pairs:
        if pair["category"] == "multi_hop":
            assert len(pair["source_passages"]) >= 2, f"{pair['id']} is not multi-hop"
