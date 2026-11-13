import threading
from dataclasses import dataclass

import numpy as np

from app.embeddings import HashingEmbedder
from app.schemas import Record

INDEX_VERSION = "v2-tenant-strict"


class TenantIsolationError(ValueError):
    pass


@dataclass
class Hit:
    record: Record
    score: float


class VectorStore:
    def __init__(self, embedder=None):
        self.embedder = embedder or HashingEmbedder()
        self._records = {}
        self._vectors = {}
        self._lock = threading.Lock()

    def __len__(self):
        return len(self._records)

    def add(self, record):
        if isinstance(record, dict):
            if not record.get("tenant_id"):
                raise TenantIsolationError(f"record {record.get('id')!r} has no tenant_id; refusing to index")
            record = Record(**record)
        if not record.tenant_id:
            raise TenantIsolationError(f"record {record.id!r} has no tenant_id")
        vec = self.embedder.embed_document(f"{record.title}. {record.text}")
        with self._lock:
            existing = self._records.get(record.id)
            if existing and existing.tenant_id != record.tenant_id:
                raise TenantIsolationError(f"id {record.id!r} already owned by another tenant")
            self._records[record.id] = record
            self._vectors[record.id] = vec
        return record

    def get(self, record_id, tenant_id):
        rec = self._records.get(record_id)
        return rec if rec and rec.tenant_id == tenant_id else None

    def search(self, query, tenant_id, top_k=4):
        if not tenant_id:
            raise TenantIsolationError("search requires a tenant_id")
        # filter FIRST: other tenants' records are never even scored
        candidates = [r for r in self._records.values() if r.tenant_id == tenant_id]
        if not candidates:
            return []
        q = self.embedder.embed_query(query)
        hits = [Hit(r, float(np.dot(q, self._vectors[r.id]))) for r in candidates]
        hits.sort(key=lambda h: h.score, reverse=True)
        assert all(h.record.tenant_id == tenant_id for h in hits)
        return hits[:top_k]