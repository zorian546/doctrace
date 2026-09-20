"""
Tests for the corpus fingerprint and the index-alignment check.

Run with:
    python -m pytest tests/test_manifest.py -v
"""


import pytest

from doctrace.corpus.manifest import (
    IndexMismatch,
    check_alignment,
    fingerprint,
    read_manifest,
    record_upload,
    write_manifest,
)

A = {"source_path": "tutorial/a.md", "anchor": "a", "text": "first"}
B = {"source_path": "tutorial/b.md", "anchor": "b", "text": "second"}


def test_fingerprint_is_stable_and_order_sensitive():
    assert fingerprint([A, B]) == fingerprint([dict(A), dict(B)])
    assert fingerprint([A, B]) != fingerprint([B, A])


def test_fingerprint_changes_when_text_or_identity_changes():
    assert fingerprint([A]) != fingerprint([{**A, "text": "edited"}])
    assert fingerprint([A]) != fingerprint([{**A, "anchor": "other"}])
    assert fingerprint([A]) != fingerprint([{**A, "source_path": "tutorial/z.md"}])


def test_inserting_a_passage_changes_the_fingerprint():
    """The upstream docs gaining a section is exactly the case this has to catch."""
    assert fingerprint([A, B]) != fingerprint([A, {"source_path": "n.md", "anchor": "n", "text": "new"}, B])


def test_missing_anchor_is_allowed():
    assert fingerprint([{**A, "anchor": None}])


def test_write_then_read_round_trip(tmp_path):
    path = tmp_path / "corpus_manifest.json"
    written = write_manifest([A, B], "abc123", path)
    assert written["passage_count"] == 2
    assert read_manifest(path)["fingerprint"] == fingerprint([A, B])


def test_write_manifest_keeps_an_existing_upload_record(tmp_path):
    path = tmp_path / "m.json"
    write_manifest([A, B], "abc123", path)
    record_upload([A, B], "coll", path)
    write_manifest([A, B], "def456", path)          # re-index, same corpus
    assert read_manifest(path)["vector_index"]["collection"] == "coll"


def test_alignment_passes_when_the_corpus_matches(tmp_path):
    path = tmp_path / "m.json"
    write_manifest([A, B], "abc123", path)
    record_upload([A, B], "coll", path)
    assert check_alignment([A, B], "coll", path) is None


def test_alignment_raises_when_a_passage_was_inserted(tmp_path):
    path = tmp_path / "m.json"
    record_upload([A, B], "coll", path)
    grown = [A, {"source_path": "n.md", "anchor": "n", "text": "new"}, B]
    with pytest.raises(IndexMismatch) as excinfo:
        check_alignment(grown, "coll", path)
    assert "build_index" in str(excinfo.value)


def test_alignment_warns_rather_than_raises_when_nothing_was_recorded(tmp_path):
    path = tmp_path / "missing.json"
    assert "Re-run" in check_alignment([A], "coll", path)


def test_alignment_warns_when_a_different_collection_is_queried(tmp_path):
    path = tmp_path / "m.json"
    record_upload([A], "one", path)
    assert "one" in check_alignment([A], "two", path)


def test_corrupt_manifest_is_treated_as_absent(tmp_path):
    path = tmp_path / "m.json"
    path.write_text("{not json", encoding="utf-8")
    assert read_manifest(path) is None
    assert check_alignment([A], "coll", path) is not None
