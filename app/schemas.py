from enum import Enum
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Record(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(min_length=3, max_length=64)
    tenant_id: str = Field(min_length=1)
    kind: Literal["kb", "ticket", "log"]
    title: str
    text: str
    topic: Optional[str] = None
    fact: Optional[str] = None
    status: Literal["current", "superseded"] = "current"
    superseded_by: Optional[str] = None
    updated_at: Optional[str] = None


class Priority(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"


class TicketIn(BaseModel):
    # extra="forbid": a client can't sneak tenant_id into the body
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    subject: str = Field(min_length=3, max_length=200)
    body: str = Field(min_length=10, max_length=5000)
    priority: Priority = Priority.medium
    topic: Optional[str] = Field(default=None, max_length=64, pattern=r"^[a-z0-9_]+$")


class TicketOut(BaseModel):
    ticket_id: str
    tenant_id: str
    indexed: bool
    quarantined: bool
    security_flags: list[str]


class AskIn(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    question: str = Field(min_length=3, max_length=1000)
    top_k: int = Field(default=4, ge=1, le=8)

    @field_validator("question")
    @classmethod
    def not_blank(cls, v):
        if not v.strip():
            raise ValueError("question must not be blank")
        return v


class Citation(BaseModel):
    source_id: str
    kind: str
    title: str
    excerpt: str
    score: float
    status: str


class AskOut(BaseModel):
    answer: str
    evidence_status: Literal["sufficient", "missing", "conflicting"]
    citations: list[Citation]
    warnings: list[str]
    security_flags: list[str]
    model: str


class HealthOut(BaseModel):
    status: Literal["ok", "degraded"]
    llm_model: str
    llm_reachable: bool
    embedder: str
    index_version: str
    records_indexed: int
