"""
Unit tests for keyword search, RRF merging and score normalisation.

Run with:
    python -m pytest tests/test_search.py -v
"""

import pytest
from doctrace.search.fusion import rrf_merge, _minmax, rescore
from doctrace.search.lexical import build_lexical_index, lexical_search, _tokenize


def test_tokenize():
    text = "FastAPI with Pydantic v2.0!"
    tokens = _tokenize(text)
    assert "fastapi" in tokens
    assert "pydantic" in tokens
    assert "v2" in tokens
    assert "0" in tokens


def test_bm25_search_scoring():
    sample_passages = [
        {"id": 0, "source_path": "tutorial/first-steps.md", "header_path": "First Steps", "text": "FastAPI is a modern web framework."},
        {"id": 1, "source_path": "tutorial/path-params.md", "header_path": "Path Parameters", "text": "Declare path parameters with type hints."},
        {"id": 2, "source_path": "tutorial/query-params.md", "header_path": "Query Parameters", "text": "Query parameters are also supported."},
    ]
    build_lexical_index(sample_passages)
    results = lexical_search("path parameters", top_k=2)
    assert len(results) == 2
    assert results[0]["id"] == 1
    assert results[0]["source_path"] == "tutorial/path-params.md"


def test_reciprocal_rank_fusion():
    list1 = [{"id": 10, "text": "doc10"}, {"id": 20, "text": "doc20"}]
    list2 = [{"id": 20, "text": "doc20"}, {"id": 30, "text": "doc30"}]

    fused = rrf_merge([list1, list2], k=60)
    assert len(fused) == 3
    # id 20 appears in both list1 (rank 2) and list2 (rank 1), should score highest
    assert fused[0]["id"] == 20
    assert fused[0]["fusion_score"] == pytest.approx((1.0 / (60 + 2)) + (1.0 / (60 + 1)), rel=1e-4)


def test_min_max_normalize():
    assert _minmax([]) == []
    assert _minmax([5.0]) == [0.5]
    assert _minmax([10.0, 20.0, 30.0]) == [0.0, 0.5, 1.0]
