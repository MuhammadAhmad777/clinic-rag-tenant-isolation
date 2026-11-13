import json
from pathlib import Path

from app.store import TenantIsolationError, VectorStore

V2_FIELDS = ["id", "tenant_id", "kind", "title", "text", "topic", "fact",
             "status", "superseded_by", "updated_at"]


def to_v1(record):
    # v1 stored the tenant under the old key "tenant"
    r = dict(record)
    r["tenant"] = r.pop("tenant_id")
    return r


def migrate_v1_to_v2_buggy(v1):
    return {f: v1.get(f) for f in V2_FIELDS}  # bug: "tenant" is silently dropped


def migrate_v1_to_v2_fixed(v1):
    out = {f: v1.get(f) for f in V2_FIELDS}
    out["tenant_id"] = v1.get("tenant_id") or v1.get("tenant")
    if not out["tenant_id"]:
        raise TenantIsolationError(f"v1 record {v1.get('id')!r} has no tenant; aborting migration")
    return out


def legacy_search_ids(records, tenant_id):
    # old unsafe filter: tenant matches OR tenant missing ("shared")
    return [r["id"] for r in records if r.get("tenant_id") in (tenant_id, None)]


def main():
    path = Path(__file__).resolve().parent.parent / "data" / "records.json"
    data = json.loads(path.read_text(encoding="utf-8"))["records"]
    v1 = [to_v1(r) for r in data]

    buggy = [migrate_v1_to_v2_buggy(r) for r in v1]
    leaked = [i for i in legacy_search_ids(buggy, "clinic-alder") if "-BIR-" in i]
    print(f"[buggy migration] records missing tenant_id: {sum(r['tenant_id'] is None for r in buggy)}")
    print(f"[buggy migration + legacy filter] Birch records visible to Alder: {leaked}")

    try:
        VectorStore().add(buggy[0])
    except TenantIsolationError as e:
        print(f"[fixed store] refuses buggy record: {e}")

    fixed = [migrate_v1_to_v2_fixed(r) for r in v1]
    store = VectorStore()
    for r in fixed:
        store.add(r)
    hits = store.search("password reset link expiry", "clinic-alder", top_k=8)
    print(f"[fixed migration] Alder search returns: {[h.record.id for h in hits]}")
    assert all(h.record.tenant_id == "clinic-alder" for h in hits)
    print("[fixed migration] OK: no cross-tenant results")


if __name__ == "__main__":
    main()