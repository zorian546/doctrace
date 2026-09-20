"""
Tests for pipeline wiring, with search and the answer model replaced by fakes.

Run with:
    python -m pytest tests/test_pipeline.py -v
"""

import pytest

from doctrace import pipeline

HIT = {"id": 3, "score": 0.5, "source_path": "tutorial/a.md", "header_path": "A", "text": "body"}
REFUSAL = "I don't have enough information in the provided context to answer this."


@pytest.fixture
def fakes(monkeypatch):
    monkeypatch.setattr(pipeline, "semantic_search", lambda q, top_k: [HIT])
    monkeypatch.setattr(pipeline, "lexical_search", lambda q, top_k: [HIT])
    monkeypatch.setattr(pipeline, "fused_search",
                        lambda q, **kw: {"results": [HIT], "latency_ms": {"merge_ms": 1.0}})
    monkeypatch.setattr(pipeline, "answer_question",
                        lambda q, hits: {"answer": "Use int.", "evidence": hits})


@pytest.mark.parametrize("mode", pipeline.MODES)
def test_every_mode_returns_answer_evidence_and_timings(fakes, mode):
    out = pipeline.run_query("how?", mode=mode)
    assert out["answer"] == "Use int." and out["abstained"] is False
    assert out["evidence"][0]["source_path"] == "tutorial/a.md"
    assert {"total_retrieval_ms", "generation_ms", "total_pipeline_ms"} <= set(out["latency_ms"])


def test_refusal_is_flagged(fakes, monkeypatch):
    monkeypatch.setattr(pipeline, "answer_question",
                        lambda q, hits: {"answer": REFUSAL, "evidence": hits})
    assert pipeline.run_query("kafka?")["abstained"] is True


def test_unknown_mode_is_rejected(fakes):
    with pytest.raises(ValueError):
        pipeline.run_query("how?", mode="dense")
