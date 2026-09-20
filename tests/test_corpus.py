"""
Sanity checks for the corpus fetcher and splitter. Run before committing.

Run with:  python -m pytest tests/test_corpus.py -v
"""

import subprocess

import pytest

from doctrace.corpus.fetcher import pull_docs, read_docs
from doctrace.corpus.splitter import split_by_headings, split_fixed_window


def _get_docs():
    # these tests need the real docs; skip (do not fail) when the clone is impossible
    try:
        pull_docs()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        pytest.skip("could not fetch the FastAPI docs (offline, or git missing)")
    return read_docs()


def test_docs_load_nonempty():
    docs = _get_docs()
    assert len(docs) == 85, f"expected 85 docs (51 tutorial + 34 advanced), got {len(docs)}"
    empty = [d for d in docs if not d["text"].strip()]
    assert not empty, f"{len(empty)} docs loaded with empty text: {[d['source_path'] for d in empty]}"


def test_snippet_includes_resolved():
    docs = _get_docs()
    missing = [d["source_path"] for d in docs if "[MISSING SNIPPET" in d["text"]]
    assert not missing, f"snippet includes failed to resolve in: {missing}"
    first_steps = next(d for d in docs if d["source_path"] == "tutorial/first-steps.md")
    assert "```py" in first_steps["text"], "expected resolved python code block, snippet not inlined"
    metadata = next(d for d in docs if d["source_path"] == "tutorial/metadata.md")
    assert "{*" not in metadata["text"], "hl[] modifier snippet marker left unresolved"


def test_passages_carry_metadata():
    docs = _get_docs()
    sample = next(d for d in docs if d["source_path"] == "tutorial/metadata.md")
    passages = split_by_headings(sample["text"], sample["source_path"])
    assert len(passages) > 1
    for c in passages:
        assert c["source_path"] == "tutorial/metadata.md"
        assert isinstance(c["header_path"], str)
        assert c["text"].strip()


def test_no_passage_splits_a_code_block():
    docs = _get_docs()
    passages = []
    for d in docs:
        passages.extend(split_by_headings(d["text"], d["source_path"]))
    broken = [c for c in passages if c["text"].count("`" * 3) % 2 != 0]
    assert not broken, f"{len(broken)} passages have an unclosed code fence"


def test_fixed_window_baseline_yields_passages():
    docs = _get_docs()
    sample = next(d for d in docs if d["source_path"] == "tutorial/metadata.md")
    passages = split_fixed_window(sample["text"], sample["source_path"], size=100, overlap=10)
    assert len(passages) > 0
    for c in passages:
        assert c["header_path"] == ""