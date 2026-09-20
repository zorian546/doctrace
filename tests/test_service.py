"""
Integration tests for the HTTP service (the pipeline itself is faked).

Run with:
    python -m pytest tests/test_service.py -v
"""

import pytest
from fastapi.testclient import TestClient

from doctrace.service import server

client = TestClient(server.app)

FAKE = {
    "query": "how?", "answer": "Use int.", "abstained": False, "mode": "fused", "top_k": 5,
    "evidence": [{"id": 1, "source_path": "tutorial/a.md", "header_path": "A", "text": "t"}],
    "latency_ms": {"total_pipeline_ms": 1.0},
}


@pytest.fixture(autouse=True)
def fake_pipeline(monkeypatch):
    monkeypatch.setattr(server, "run_query", lambda *a, **k: FAKE)
    monkeypatch.delenv("DOCTRACE_API_KEY", raising=False)


def test_status_route():
    data = client.get("/v1/status").json()
    assert data["passages_file_present"] is True
    assert data["passages_indexed"] > 0


def test_scorecard_route():
    assert "retrieval_benchmark" in client.get("/v1/scorecard").json()


def test_ask_returns_the_pipeline_result():
    r = client.post("/v1/ask", json={"query": "how?"})
    assert r.status_code == 200 and r.json()["answer"] == "Use int."


@pytest.mark.parametrize("body", [
    {"query": ""},
    {"query": "x" * 501},
    {"query": "ok", "mode": "dense"},
    {"query": "ok", "top_k": 99},
    {"query": "ok", "rerank_weight": 1.5},
])
def test_ask_rejects_bad_input(body):
    assert client.post("/v1/ask", json=body).status_code == 422


def test_internal_errors_are_not_leaked(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError("secret qdrant url http://internal:6333")
    monkeypatch.setattr(server, "run_query", boom)
    r = client.post("/v1/ask", json={"query": "how?"})
    assert r.status_code == 500
    assert "internal" not in r.text and "qdrant" not in r.text


def test_api_key_is_enforced_when_configured(monkeypatch):
    monkeypatch.setenv("DOCTRACE_API_KEY", "s3cret")
    body = {"query": "how?"}
    assert client.post("/v1/ask", json=body).status_code == 401
    assert client.post("/v1/ask", json=body, headers={"X-API-Key": "nope"}).status_code == 401
    assert client.post("/v1/ask", json=body, headers={"X-API-Key": "s3cret"}).status_code == 200


def test_cors_allows_the_dashboard_and_not_strangers():
    ok = client.options("/v1/ask", headers={"Origin": "http://localhost:8501",
                                            "Access-Control-Request-Method": "POST"})
    bad = client.options("/v1/ask", headers={"Origin": "https://evil.example",
                                             "Access-Control-Request-Method": "POST"})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8501"
    assert "access-control-allow-origin" not in bad.headers
