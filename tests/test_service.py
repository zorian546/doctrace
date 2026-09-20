"""
Integration tests for the HTTP service.

Run with:
    python -m pytest tests/test_service.py -v
"""

from fastapi.testclient import TestClient
from doctrace.service.server import app

client = TestClient(app)


def test_status_route():
    response = client.get("/v1/status")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "passages_indexed" in data
    assert data["passages_file_present"] is True


def test_scorecard_route():
    response = client.get("/v1/scorecard")
    assert response.status_code == 200
    data = response.json()
    assert "retrieval_benchmark" in data


def test_ask_rejects_blank_question():
    # Empty query should fail validation with 422
    response = client.post("/v1/ask", json={"query": ""})
    assert response.status_code == 422
