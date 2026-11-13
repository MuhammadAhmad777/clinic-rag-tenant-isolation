from tests.conftest import ALDER, DownLLM, make_client


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["records_indexed"] == 12
    assert body["index_version"] == "v2-tenant-strict"


def test_health_degraded_when_llm_down():
    r = make_client(DownLLM()).get("/health")
    assert r.status_code == 200 and r.json()["status"] == "degraded"


def test_missing_api_key_401(client):
    r = client.post("/ask", json={"question": "password reset?"})
    assert r.status_code == 401
    assert r.json() == {"error": "unauthorized", "details": "Missing X-API-Key header"}

def test_invalid_api_key_401(client):
    r = client.post("/ask", json={"question": "password reset?"}, headers={"X-API-Key": "nope"})
    assert r.status_code == 401


def test_question_too_short_422(client):
    r = client.post("/ask", json={"question": "hi"}, headers=ALDER)
    assert r.status_code == 422 and r.json()["error"] == "validation_error"


def test_top_k_out_of_range_422(client):
    r = client.post("/ask", json={"question": "password reset?", "top_k": 50}, headers=ALDER)
    assert r.status_code == 422


def test_body_cannot_choose_tenant_422(client):
    r = client.post("/ask", json={"question": "password reset?", "tenant_id": "clinic-birch"}, headers=ALDER)
    assert r.status_code == 422


def test_ticket_ingest_201(client):
    r = client.post("/tickets", headers=ALDER, json={
        "subject": "Kiosk frozen", "body": "The check-in kiosk freezes after scanning a card.",
        "priority": "high", "topic": "kiosk"})
    assert r.status_code == 201
    body = r.json()
    assert body["tenant_id"] == "clinic-alder" and body["ticket_id"].startswith("TCK-ALD-")
    assert body["quarantined"] is False


def test_ticket_bad_priority_422(client):
    r = client.post("/tickets", headers=ALDER, json={
        "subject": "Kiosk", "body": "The check-in kiosk freezes.", "priority": "urgent!!"})
    assert r.status_code == 422


def test_ticket_tenant_in_body_rejected(client):
    r = client.post("/tickets", headers=ALDER, json={
        "subject": "Kiosk", "body": "The check-in kiosk freezes.", "tenant_id": "clinic-birch"})
    assert r.status_code == 422


def test_llm_down_returns_503():
    r = make_client(DownLLM()).post("/ask", json={"question": "How long is the password reset link valid?"},
                                    headers=ALDER)
    assert r.status_code == 503 and r.json()["error"] == "llm_unavailable"




def test_embedding_failure_returns_503():
    from fastapi.testclient import TestClient
    from app.embeddings import EmbeddingError, HashingEmbedder
    from app.main import create_app
    from tests.conftest import FakeLLM

    class BrokenEmbedder(HashingEmbedder):
        name = "broken"
        def embed_query(self, text):
            raise EmbeddingError("embeddings down (simulated)")

    client = TestClient(create_app(llm=FakeLLM(), embedder=BrokenEmbedder()))
    r = client.post("/ask", json={"question": "How long is the password reset link valid?"}, headers=ALDER)
    assert r.status_code == 503 and r.json()["error"] == "embedding_unavailable"


def test_ollama_embedder_normalizes_and_uses_prefixes(monkeypatch):
    import app.embeddings as emb
    sent = []

    class Resp:
        status_code = 200
        def json(self):
            return {"embeddings": [[3.0, 4.0]]}

    monkeypatch.setattr(emb.httpx, "post", lambda url, json, timeout: sent.append(json) or Resp())
    e = emb.OllamaEmbedder(model="nomic-embed-text")
    vec = e.embed_query("reset link")
    assert abs(vec[0] - 0.6) < 1e-6 and abs(vec[1] - 0.8) < 1e-6
    assert sent[0]["input"] == ["search_query: reset link"]