import json
import sys
from pathlib import Path

from app.embeddings import make_embedder
from app.store import VectorStore

A, B = "clinic-alder", "clinic-birch"

# (tenant, question, ids that count as a correct hit); "para" = reworded, few shared words
IN_SCOPE = [
    ("lexical", A, "How long is the password reset link valid?", {"KB-ALD-002", "KB-ALD-003"}),
    ("lexical", A, "When are appointment reminder SMS messages sent?", {"KB-ALD-004", "KB-ALD-001"}),
    ("lexical", A, "How do I set up the front desk label printer?", {"KB-ALD-005"}),
    ("lexical", A, "What happened during the vector index migration?", {"LOG-ALD-9001"}),
    ("lexical", B, "Why are lab results delayed in the portal?", {"TCK-BIR-2001", "LOG-BIR-9002"}),
    ("lexical", B, "Who handles billing escalations?", {"KB-BIR-001"}),
    ("para", A, "When does the credential recovery URL stop working?", {"KB-ALD-002", "KB-ALD-003"}),
    ("para", A, "How far ahead of a visit do patients get a text?", {"KB-ALD-004", "KB-ALD-001"}),
    ("para", A, "Which driver does the sticker printer at reception need?", {"KB-ALD-005"}),
    ("para", A, "Why did the assistant show us another practice's article?", {"TCK-ALD-1002", "LOG-ALD-9001"}),
    ("para", A, "Patients say the login recovery email is already expired", {"TCK-ALD-1001", "KB-ALD-002", "KB-ALD-003"}),
    ("para", B, "Test outcomes are showing up two days late for patients", {"TCK-BIR-2001", "LOG-BIR-9002"}),
    ("para", B, "Who do I contact about a disputed invoice?", {"KB-BIR-001"}),
    ("para", B, "How quickly does the account recovery link expire?", {"KB-BIR-002"}),
]
OUT_OF_SCOPE = [
    (A, "What is the staff parking policy?"),
    (A, "What are the clinic's holiday opening hours?"),
    (A, "Can I bring my dog to the office?"),
    (B, "How do I request annual leave?"),
]

# validation set: written after the threshold was chosen, never used to tune it
VAL_IN_SCOPE = [
    ("para", A, "Our patients' password links die after half an hour, is that expected?", {"KB-ALD-002", "KB-ALD-003", "TCK-ALD-1001"}),
    ("para", A, "Do reminder texts go out the day before appointments?", {"KB-ALD-004", "KB-ALD-001"}),
    ("para", A, "The reception label printer needs which software driver?", {"KB-ALD-005"}),
    ("para", A, "Logs about tenant ids missing after the index upgrade", {"LOG-ALD-9001"}),
    ("para", B, "Portal sync for lab results is degraded, any logs?", {"LOG-BIR-9002", "TCK-BIR-2001"}),
    ("para", B, "Where should invoice disagreements be escalated?", {"KB-BIR-001"}),
    ("para", B, "What is the expiry time for a Birch password reset?", {"KB-BIR-002"}),
]
VAL_OUT_OF_SCOPE = [
    (A, "How do I book a meeting room?"),
    (A, "What is the staff wifi password?"),
    (A, "What vaccines does the clinic offer?"),
    (B, "Who is the clinic's medical director?"),
    (B, "How do I reset the office coffee machine?"),
]


def build_store(kind):
    store = VectorStore(make_embedder(kind))
    path = Path(__file__).resolve().parent.parent / "data" / "records.json"
    for r in json.loads(path.read_text(encoding="utf-8"))["records"]:
        store.add(r)
    return store


def evaluate(kind, in_scope=IN_SCOPE, out_of_scope=OUT_OF_SCOPE, label="tuning set"):
    store = build_store(kind)
    t = store.embedder.min_score
    rows, misses = [], []
    for group, tenant, q, gold in in_scope:
        # only hits above the threshold count: that is what the app actually uses
        ids = [h.record.id for h in store.search(q, tenant, top_k=12) if h.score >= t]
        rank = next((i + 1 for i, rid in enumerate(ids) if rid in gold), None)
        rows.append((group, rank))
        if rank != 1:
            misses.append(f"  rank={rank}  {q}")
    oos = [(q, store.search(q, tenant, top_k=1)[0]) for tenant, q in out_of_scope]
    rejected = sum(1 for _, h in oos if h.score < t)

    print(f"\n=== {label} | embedder: {store.embedder.name}  (threshold {t}) ===")
    for group in ("lexical", "para", "all"):
        ranks = [r for g, r in rows if group == "all" or g == group]
        if not ranks:
            continue
        hit1 = sum(1 for r in ranks if r == 1) / len(ranks)
        hit3 = sum(1 for r in ranks if r and r <= 3) / len(ranks)
        mrr = sum(1 / r for r in ranks if r) / len(ranks)
        print(f"{group:8} n={len(ranks):2}  hit@1={hit1:.2f}  hit@3={hit3:.2f}  MRR={mrr:.2f}")
    print(f"out-of-scope correctly rejected: {rejected}/{len(oos)}")
    for q, h in oos:
        print(f"  top score {h.score:.3f}  {q}")
    print("in-scope questions not answered at rank 1:")
    print("\n".join(misses) or "  none")


if __name__ == "__main__":
    for kind in sys.argv[1:] or ["hashing", "ollama"]:
        evaluate(kind)
        evaluate(kind, VAL_IN_SCOPE, VAL_OUT_OF_SCOPE, label="validation set")