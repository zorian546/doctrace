"""
Offline tests for the heading-aware splitter (no network needed).

Run with:
    python -m pytest tests/test_splitter.py -v
"""

from doctrace.corpus.splitter import split_by_headings, split_fixed_window

DOC = """# Intro { #intro }
hello

## Setup { #setup }
```python
# this hash is a comment, not a heading
x = 1
```
after code

## Usage
plain heading without an anchor
"""


def test_splits_at_headings_and_keeps_trail_and_slug():
    parts = split_by_headings(DOC, "a.md")
    assert [p["header_path"] for p in parts] == ["Intro", "Intro > Setup", "Intro > Usage"]
    assert [p["anchor"] for p in parts] == ["intro", "setup", None]


def test_hash_inside_code_fence_is_not_a_heading():
    parts = split_by_headings(DOC, "a.md")
    setup = parts[1]["text"]
    assert "# this hash is a comment" in setup and "x = 1" in setup


def test_sibling_heading_pops_the_trail():
    parts = split_by_headings("# A\n1\n## B\n2\n## C\n3\n# D\n4", "a.md")
    assert [p["header_path"] for p in parts] == ["A", "A > B", "A > C", "D"]


def test_blank_document_yields_nothing():
    assert split_by_headings("", "a.md") == []


def test_fixed_window_overlaps_and_covers_everything():
    text = " ".join(str(i) for i in range(25))
    parts = split_fixed_window(text, "a.md", size=10, overlap=2)
    assert parts[0]["text"].split()[:2] == ["0", "1"]
    assert parts[-1]["text"].split()[-1] == "24"
    assert all(p["anchor"] is None and p["header_path"] == "" for p in parts)
