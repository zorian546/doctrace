"""
Unit tests for ranking metrics, abstention detection and gold-set validation.

Run with:
    python -m pytest tests/test_audit.py -v
"""

import pytest
from doctrace.audit.ranking import _score_query
from doctrace.audit.grounding import is_abstention
from doctrace.audit.goldset import check_gold_pairs


def test_score_query_partial_hit():
    gt = [1, 2, 3]
    retrieved = [5, 2, 7, 8, 9]  # id 2 is at rank 2

    m = _score_query(ground_truth_ids=gt, retrieved_ids=retrieved, k=5)
    assert m["hit"] == 1.0
    assert m["recall"] == pytest.approx(1.0 / 3.0)
    assert m["precision"] == pytest.approx(1.0 / 5.0)
    assert m["mrr"] == pytest.approx(1.0 / 2.0)  # first match at rank 2


def test_score_query_miss():
    gt = [100, 101]
    retrieved = [1, 2, 3, 4, 5]

    m = _score_query(ground_truth_ids=gt, retrieved_ids=retrieved, k=5)
    assert m["hit"] == 0.0
    assert m["recall"] == 0.0
    assert m["precision"] == 0.0
    assert m["mrr"] == 0.0


def test_is_abstention():
    assert is_abstention("I don't have enough information in the provided context to answer this.")
    assert is_abstention("I don't have enough information in the provided context to answer this question.")
    assert not is_abstention("FastAPI uses Pydantic for data validation.")


def test_check_gold_pairs_accepts_valid():
    valid = [
        {"id": "sh_001", "category": "single_hop", "question": "Q1", "answer": "A1", "source_chunks": ["file.md#anchor"]},
        {"id": "na_001", "category": "no_answer", "question": "Q2", "answer": "A2", "source_chunks": []},
    ]
    check_gold_pairs(valid)  # should not raise


def test_check_gold_pairs_rejects_blank():
    invalid = [{"id": "sh_001", "category": "single_hop", "question": "", "answer": "A1", "source_chunks": ["f.md"]}]
    with pytest.raises(AssertionError):
        check_gold_pairs(invalid)
