"""End-to-end tests against the real Ollama models. Opt-in: pytest -m integration"""
import os

import httpx
import pytest
from fastapi.testclient import TestClient

from app.embeddings import OllamaEmbedder
from app.llm import OllamaLLM
from app.main import create_app
from tests.conftest import ALDER, BIRCH

pytestmark = pytest.mark.integration

LLM_MODEL = os.getenv("INTEGRATION_LLM", "clinic-qwen-lora")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")


def ollama_models():
    try:
        r = httpx.get(os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/tags", timeout=2.0)
        return {m["name"].split(":latest")[0] for m in r.json().get("models", [])}
    except (httpx.HTTPError, ValueError):
        return None


@pytest.fixture(scope="module")
def live_client():
    models = ollama_models()
    if models is None:
        pytest.skip("Ollama is not running")
    missing = [m for m in (LLM_MODEL, EMBED_MODEL) if m not in models]
    if missing:
        pytest.skip(f"Ollama models not pulled: {missing}")
    return TestClient(create_app(llm=OllamaLLM(model=LLM_MODEL), embedder=OllamaEmbedder(model=EMBED_MODEL)))


def test_live_health(live_client):
    body = live_client.get("/health").json()
    assert body["status"] == "ok" and body["embedder"] == f"ollama:{EMBED_MODEL}"


def test_live_paraphrase_is_answered_with_alder_citations(live_client):
    r = live_client.post("/ask", headers=ALDER, json={"question": "When does the credential recovery URL stop working?"})
    assert r.status_code == 200
    ids = [c["source_id"] for c in r.json()["citations"]]
    assert {"KB-ALD-002", "KB-ALD-003"} <= set(ids)
    assert all("-BIR-" not in i for i in ids)


def test_live_birch_never_sees_alder(live_client):
    r = live_client.post("/ask", headers=BIRCH, json={"question": "How long is the password reset link valid?"})
    ids = [c["source_id"] for c in r.json()["citations"]]
    assert ids and all("-BIR-" in i for i in ids)
    assert "30 minutes" not in r.json()["answer"] and "60 minutes" not in r.json()["answer"]


def test_live_malicious_record_quarantined(live_client):
    body = live_client.post("/ask", headers=ALDER, json={"question": "How do I set up the front desk printer?"}).json()
    assert "quarantined:KB-ALD-005" in body["security_flags"]
    assert "billing-desk@birch.example" not in body["answer"]


def test_live_out_of_scope_rejected_without_llm(live_client):
    body = live_client.post("/ask", headers=ALDER, json={"question": "Can I bring my dog to the office?"}).json()
    assert body["evidence_status"] == "missing" and body["model"] == "none (not called)"