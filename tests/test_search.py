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


class _FakeCrossEncoder:
    """Scores a candidate by the number in its text, so the ordering is predictable."""

    def predict(self, pairs):
        return [float(text) for _, text in pairs]


def _candidates():
    # fusion scores favour "c" and "b"; the cross-encoder favours "a"
    return [
        {"id": 1, "text": "9", "fusion_score": 0.010},
        {"id": 2, "text": "5", "fusion_score": 0.020},
        {"id": 3, "text": "1", "fusion_score": 0.030},
    ]


def test_rescore_weight_one_follows_the_cross_encoder(monkeypatch):
    monkeypatch.setattr("doctrace.search.fusion._load_cross_encoder", lambda: _FakeCrossEncoder())
    out = rescore("q", _candidates(), top_k=3, rerank_weight=1.0)
    assert [c["id"] for c in out] == [1, 2, 3]


def test_rescore_weight_zero_follows_the_fusion_score(monkeypatch):
    monkeypatch.setattr("doctrace.search.fusion._load_cross_encoder", lambda: _FakeCrossEncoder())
    out = rescore("q", _candidates(), top_k=3, rerank_weight=0.0)
    assert [c["id"] for c in out] == [3, 2, 1]


def test_rescore_blend_is_between_the_extremes_and_respects_top_k(monkeypatch):
    monkeypatch.setattr("doctrace.search.fusion._load_cross_encoder", lambda: _FakeCrossEncoder())
    out = rescore("q", _candidates(), top_k=2, rerank_weight=0.5)
    assert len(out) == 2
    assert all(0.0 <= c["blended_score"] <= 1.0 for c in out)
    assert out[0]["blended_score"] >= out[1]["blended_score"]


def test_rescore_empty_pool():
    assert rescore("q", []) == []
