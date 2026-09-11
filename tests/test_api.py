from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)

SAMPLE_TEXT = (
    "Employees may work remotely up to five days a week with manager approval. "
    "Health insurance eligibility begins after 90 days of employment."
)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "healthy"}


def test_ingest_query_history_analytics_flow():
    ingest_resp = client.post("/documents/ingest", json={"text": SAMPLE_TEXT})
    assert ingest_resp.status_code == 200
    body = ingest_resp.json()
    assert body["num_chunks"] >= 1
    document_id = body["document_id"]

    query_resp = client.post("/query", json={
        "document_id": document_id,
        "question": "How many days a week can employees work remotely?",
    })
    assert query_resp.status_code == 200
    qbody = query_resp.json()
    assert qbody["answer"]
    assert qbody["latency_ms"] > 0

    history_resp = client.get("/history")
    assert history_resp.status_code == 200
    assert any(q["document_id"] == document_id for q in history_resp.json()["queries"])

    analytics_resp = client.get("/analytics")
    assert analytics_resp.status_code == 200
    assert analytics_resp.json()["summary"]["total_queries"] >= 1


def test_query_unknown_document_returns_404():
    resp = client.post("/query", json={"document_id": "does-not-exist", "question": "Anything?"})
    assert resp.status_code == 404


def test_ingest_requires_source():
    resp = client.post("/documents/ingest", json={})
    assert resp.status_code == 422
