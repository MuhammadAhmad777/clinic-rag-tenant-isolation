import itertools
import json
import logging
import os
import uuid
from pathlib import Path

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from app.auth import TenantRegistry, current_tenant
from app.embeddings import EmbeddingError, make_embedder
from app.llm import LLMUnavailable, OllamaLLM
from app.rag import answer_question
from app.schemas import AskIn, AskOut, HealthOut, Record, TicketIn, TicketOut
from app.security import flag_injection
from app.store import INDEX_VERSION, TenantIsolationError, VectorStore

DATA_DIR = Path(os.getenv("DATA_DIR", Path(__file__).resolve().parent.parent / "data"))
log = logging.getLogger("clinic_rag")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def load_records(store, path):
    records = json.loads(path.read_text(encoding="utf-8"))["records"]
    for r in records:
        store.add(r)
    return len(records)


def create_app(llm=None, store=None, tenants=None, embedder=None):
    app = FastAPI(title="Multi-tenant Clinic RAG Assistant (synthetic)", version="1.1.0")
    app.state.llm = llm or OllamaLLM()
    app.state.tenants = tenants or TenantRegistry.from_file(DATA_DIR / "tenants.json")
    if store is None:
        store = VectorStore(embedder or make_embedder())
        n = load_records(store, DATA_DIR / "records.json")
        log.info("indexed %d synthetic records (index %s, embedder %s)", n, INDEX_VERSION, store.embedder.name)
    app.state.store = store
    app.state.ticket_seq = itertools.count(5001)

    @app.exception_handler(RequestValidationError)
    async def on_validation(request, exc):
        errors = [{"field": ".".join(str(p) for p in e["loc"][1:]), "msg": e["msg"]} for e in exc.errors()]
        return JSONResponse(status_code=422, content={"error": "validation_error", "details": errors})

    @app.exception_handler(StarletteHTTPException)
    async def on_http_error(request, exc):
        code = {401: "unauthorized", 404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return JSONResponse(status_code=exc.status_code, content={"error": code, "details": exc.detail})

    @app.exception_handler(LLMUnavailable)
    async def on_llm_down(request, exc):
        log.warning("LLM unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"error": "llm_unavailable", "details": str(exc)})

    @app.exception_handler(EmbeddingError)
    async def on_embed_down(request, exc):
        log.warning("embeddings unavailable: %s", exc)
        return JSONResponse(status_code=503, content={"error": "embedding_unavailable", "details": str(exc)})

    @app.exception_handler(TenantIsolationError)
    async def on_isolation(request, exc):
        log.error("tenant isolation violation: %s", exc)
        return JSONResponse(status_code=409, content={"error": "tenant_isolation", "details": str(exc)})

    @app.exception_handler(Exception)
    async def on_unexpected(request, exc):
        ref = uuid.uuid4().hex[:8]
        log.exception("unhandled error ref=%s", ref)
        return JSONResponse(status_code=500, content={"error": "internal_error", "ref": ref})

    @app.get("/health", response_model=HealthOut)
    def health(request: Request):
        st = request.app.state
        ok = st.llm.is_reachable()
        return HealthOut(status="ok" if ok else "degraded", llm_model=getattr(st.llm, "model", "unknown"),
                         llm_reachable=ok, embedder=st.store.embedder.name,
                         index_version=INDEX_VERSION, records_indexed=len(st.store))

    @app.post("/tickets", response_model=TicketOut, status_code=201)
    def ingest_ticket(ticket: TicketIn, request: Request, tenant_id: str = Depends(current_tenant)):
        code = tenant_id.split("-")[-1][:3].upper()
        ticket_id = f"TCK-{code}-{next(request.app.state.ticket_seq)}"
        flags = flag_injection(ticket.subject) + flag_injection(ticket.body)
        record = Record(id=ticket_id, tenant_id=tenant_id, kind="ticket", title=ticket.subject,
                        text=ticket.body, topic=ticket.topic, status="current")
        request.app.state.store.add(record)
        return TicketOut(ticket_id=ticket_id, tenant_id=tenant_id, indexed=True,
                         quarantined=bool(flags), security_flags=flags)

    @app.post("/ask", response_model=AskOut)
    def ask(body: AskIn, request: Request, tenant_id: str = Depends(current_tenant)):
        st = request.app.state
        return answer_question(body.question, tenant_id, st.store, st.llm,
                               st.tenants.foreign_terms(tenant_id), body.top_k)

    return app


app = create_app()