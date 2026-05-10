"""
GOV-AI 2.0 — Modèles SQLAlchemy 2.0 (ORM async).
Tables PostgreSQL : documents, chunks, sessions, query_logs, audit_events.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


def _uuid() -> str:
    return str(uuid.uuid4())


# ── Documents ─────────────────────────────────────────────────────────────────

class Document(Base):
    """
    Table documents — chaque document ingéré dans le corpus.
    Correspond à l'entité TexteNormatif / Document de l'ontologie.
    """
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    filename: Mapped[str] = mapped_column(String(512), nullable=False)
    file_path: Mapped[Optional[str]] = mapped_column(String(1024), nullable=True)
    source: Mapped[str] = mapped_column(String(512), nullable=False)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="unknown")
    doc_type: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    institution: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    jurisdiction: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    version: Mapped[str] = mapped_column(String(64), nullable=False, default="1.0")
    date_document: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    ocr_used: Mapped[bool] = mapped_column(Boolean, default=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending")
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    extra_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relations
    chunks: Mapped[list["Chunk"]] = relationship(
        "Chunk", back_populates="document", cascade="all, delete-orphan"
    )

    __table_args__ = (
        Index("ix_documents_language", "language"),
        Index("ix_documents_doc_type", "doc_type"),
        Index("ix_documents_institution", "institution"),
        Index("ix_documents_status", "status"),
    )

    def __repr__(self) -> str:
        return f"<Document id={self.id!r} source={self.source!r}>"


# ── Chunks ────────────────────────────────────────────────────────────────────

class Chunk(Base):
    """
    Table chunks — segments de texte indexés dans Milvus + Elasticsearch.
    Provenance complète : source → page → chunk_index.
    """
    __tablename__ = "chunks"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    doc_id: Mapped[str] = mapped_column(
        String(36), ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    content: Mapped[str] = mapped_column(Text, nullable=False)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    page: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    token_count: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    language: Mapped[str] = mapped_column(String(10), nullable=False, default="unknown")
    chunk_strategy: Mapped[str] = mapped_column(String(32), default="fixed_size")
    milvus_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    es_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    ingested_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    extra_metadata: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relations
    document: Mapped["Document"] = relationship("Document", back_populates="chunks")

    __table_args__ = (
        Index("ix_chunks_doc_id", "doc_id"),
        Index("ix_chunks_language", "language"),
        Index("ix_chunks_milvus_id", "milvus_id"),
    )

    def __repr__(self) -> str:
        return f"<Chunk id={self.id!r} doc_id={self.doc_id!r} idx={self.chunk_index}>"


# ── Sessions ──────────────────────────────────────────────────────────────────

class Session(Base):
    """
    Table sessions — suivi des sessions utilisateur.
    Associe un user_id à un historique de requêtes.
    """
    __tablename__ = "sessions"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    language: Mapped[str] = mapped_column(String(10), default="fr")
    profile_type: Mapped[str] = mapped_column(String(32), default="citizen")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )
    last_active: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), onupdate=func.now()
    )
    extra_data: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)

    # Relations
    query_logs: Mapped[list["QueryLog"]] = relationship(
        "QueryLog", back_populates="session", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_sessions_user_id", "user_id"),)


# ── Query Logs ────────────────────────────────────────────────────────────────

class QueryLog(Base):
    """
    Table query_logs — journal complet de chaque requête.
    Traçabilité C10 : requête, chunks récupérés, réponse, scores, latence.
    """
    __tablename__ = "query_logs"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    session_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True
    )
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    language: Mapped[str] = mapped_column(String(10), default="unknown")
    profile: Mapped[str] = mapped_column(String(32), default="citizen")
    intent: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    plan_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    retrieved_chunks_json: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)
    response_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    citations_json: Mapped[Optional[list[Any]]] = mapped_column(JSON, nullable=True)
    score_conf: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    latency_ms: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    model_used: Mapped[Optional[str]] = mapped_column(String(128), nullable=True)
    safety_flags: Mapped[Optional[list[str]]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    # Relations
    session: Mapped[Optional["Session"]] = relationship(
        "Session", back_populates="query_logs"
    )

    __table_args__ = (
        Index("ix_query_logs_session_id", "session_id"),
        Index("ix_query_logs_user_id", "user_id"),
        Index("ix_query_logs_created_at", "created_at"),
        Index("ix_query_logs_language", "language"),
    )


# ── Audit Events ──────────────────────────────────────────────────────────────

class AuditEvent(Base):
    """
    Table audit_events — événements d'audit horodatés et non modifiables.
    Conformité C10 : journalisation sécurisée de toutes les interactions.
    """
    __tablename__ = "audit_events"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=_uuid)
    event_type: Mapped[str] = mapped_column(String(64), nullable=False)
    session_id: Mapped[Optional[str]] = mapped_column(String(36), nullable=True)
    user_id: Mapped[Optional[str]] = mapped_column(String(256), nullable=True)
    payload_json: Mapped[Optional[dict[str, Any]]] = mapped_column(JSON, nullable=True)
    is_flagged: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    __table_args__ = (
        Index("ix_audit_events_event_type", "event_type"),
        Index("ix_audit_events_session_id", "session_id"),
        Index("ix_audit_events_is_flagged", "is_flagged"),
        Index("ix_audit_events_created_at", "created_at"),
    )
