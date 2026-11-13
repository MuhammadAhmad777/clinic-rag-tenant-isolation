import pytest

from app.store import TenantIsolationError, VectorStore
from scripts.migrate_index import legacy_search_ids, migrate_v1_to_v2_buggy, migrate_v1_to_v2_fixed, to_v1
from tests.conftest import ALDER, BIRCH

REC = {"id": "KB-TST-001", "tenant_id": "t1", "kind": "kb", "title": "x", "text": "reset link"}


def test_store_rejects_record_without_tenant():
    with pytest.raises(TenantIsolationError):
        VectorStore().add({**REC, "tenant_id": None})


def test_search_requires_tenant():
    s = VectorStore()
    s.add(REC)
    with pytest.raises(TenantIsolationError):
        s.search("reset", "")


def test_cross_tenant_id_collision_rejected():
    s = VectorStore()
    s.add(REC)
    with pytest.raises(TenantIsolationError):
        s.add({**REC, "tenant_id": "t2"})


def test_search_only_returns_own_tenant(client):
    store = client.app.state.store
    for tenant in ("clinic-alder", "clinic-birch"):
        hits = store.search("password reset link expire", tenant, top_k=8)
        assert hits and all(h.record.tenant_id == tenant for h in hits)


def test_buggy_migration_reproduces_leak_and_fix_prevents_it(client):
    records = [r.model_dump() for r in client.app.state.store._records.values()]
    v1 = [to_v1(r) for r in records]
    buggy = [migrate_v1_to_v2_buggy(r) for r in v1]
    assert any("-BIR-" in i for i in legacy_search_ids(buggy, "clinic-alder"))  # incident reproduced
    with pytest.raises(TenantIsolationError):
        VectorStore().add(buggy[0])  # fixed store fails closed
    fixed = [migrate_v1_to_v2_fixed(r) for r in v1]
    assert all(f["tenant_id"] == o["tenant_id"] for f, o in zip(fixed, records))


def test_answer_has_citations_with_excerpts(client):
    r = client.post("/ask", json={"question": "How long is the password reset link valid?"}, headers=ALDER)
    assert r.status_code == 200
    ids = [c["source_id"] for c in r.json()["citations"]]
    assert "KB-ALD-002" in ids and "KB-ALD-003" in ids
    assert all(c["excerpt"] for c in r.json()["citations"])
    assert all("-BIR-" not in i for i in ids)


def test_conflicting_evidence_reported(client):
    body = client.post("/ask", json={"question": "How long is the password reset link valid?"}, headers=ALDER).json()
    assert body["evidence_status"] == "conflicting"
    assert any("Conflicting" in w and "KB-ALD-002" in w and "KB-ALD-003" in w for w in body["warnings"])


def test_stale_evidence_flagged(client):
    body = client.post("/ask", json={"question": "When are appointment reminder SMS sent?"}, headers=ALDER).json()
    assert any("KB-ALD-001 is outdated" in w for w in body["warnings"])
    statuses = {c["source_id"]: c["status"] for c in body["citations"]}
    assert statuses.get("KB-ALD-004") == "current"


def test_missing_evidence_does_not_call_llm(client, fake_llm):
    body = client.post("/ask", json={"question": "What is the staff parking policy?"}, headers=ALDER).json()
    assert body["evidence_status"] == "missing" and body["citations"] == []
    assert fake_llm.calls == []


def test_hallucinated_citation_removed(client):
    class Hallucinator:
        model = "h"
        def is_reachable(self): return True
        def chat(self, s, u): return "Links last 30 minutes [KB-ALD-002] and see [KB-ALD-999]."
    client.app.state.llm = Hallucinator()
    body = client.post("/ask", json={"question": "How long is the password reset link valid?"}, headers=ALDER).json()
    ids = [c["source_id"] for c in body["citations"]]
    assert "KB-ALD-999" not in body["answer"] and "KB-ALD-999" not in ids
    assert ids[0] == "KB-ALD-002"
    assert "blocked_unretrieved_citation:KB-ALD-999" in body["security_flags"]


def test_conflicting_record_added_when_model_omits_it(client):
    class OneSided:
        model = "one-sided"
        def is_reachable(self): return True
        def chat(self, s, u): return "Links last 30 minutes [KB-ALD-002]."
    client.app.state.llm = OneSided()
    body = client.post("/ask", json={"question": "How long is the password reset link valid?"}, headers=ALDER).json()
    ids = [c["source_id"] for c in body["citations"]]
    assert "KB-ALD-002" in ids and "KB-ALD-003" in ids
    assert any("added KB-ALD-003" in w for w in body["warnings"])


def test_birch_gets_birch_answers(client):
    body = client.post("/ask", json={"question": "Why are lab results delayed?"}, headers=BIRCH).json()
    assert body["citations"] and all("-BIR-" in c["source_id"] for c in body["citations"])