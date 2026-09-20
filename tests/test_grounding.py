"""
Tests for the pure-Python parts of the grounding score (no model needed).

Run with:
    python -m pytest tests/test_grounding.py -v
"""

from doctrace.audit.grounding import _clean_claim_lines, is_abstention


def test_strips_bullets_and_numbering():
    raw = "- First claim.\n2. Second claim.\n* Third claim."
    assert _clean_claim_lines(raw) == ["First claim.", "Second claim.", "Third claim."]


def test_merges_a_sentence_wrapped_across_lines():
    raw = "FastAPI validates the\nrequest body.\nIt also documents it."
    assert _clean_claim_lines(raw) == ["FastAPI validates the request body.", "It also documents it."]


def test_empty_output_gives_no_claims():
    assert _clean_claim_lines("   \n ") == []


def test_abstention_survives_curly_apostrophe():
    assert is_abstention("I don’t have enough information in the provided context to answer this.")


def test_normal_answer_is_not_abstention():
    assert not is_abstention("Use a type annotation such as item_id: int.")
