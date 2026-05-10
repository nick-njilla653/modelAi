"""Initial schema GOV-AI 2.0

Revision ID: 001_initial_schema
Revises:
Create Date: 2025-01-01 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "001_initial_schema"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Table documents ──────────────────────────────────────────────────────
    op.create_table(
        "documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("file_path", sa.String(1024), nullable=True),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("language", sa.String(10), nullable=False, server_default="unknown"),
        sa.Column("doc_type", sa.String(64), nullable=True),
        sa.Column("institution", sa.String(256), nullable=True),
        sa.Column("jurisdiction", sa.String(256), nullable=True),
        sa.Column("version", sa.String(64), nullable=False, server_default="1.0"),
        sa.Column("date_document", sa.String(32), nullable=True),
        sa.Column("ocr_used", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("status", sa.String(32), nullable=False, server_default="pending"),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("extra_metadata", sa.JSON(), nullable=True),
    )
    op.create_index("ix_documents_language", "documents", ["language"])
    op.create_index("ix_documents_doc_type", "documents", ["doc_type"])
    op.create_index("ix_documents_institution", "documents", ["institution"])
    op.create_index("ix_documents_status", "documents", ["status"])

    # ── Table chunks ──────────────────────────────────────────────────────────
    op.create_table(
        "chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("doc_id", sa.String(36),
                  sa.ForeignKey("documents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("language", sa.String(10), nullable=False, server_default="unknown"),
        sa.Column("chunk_strategy", sa.String(32), server_default="fixed_size"),
        sa.Column("milvus_id", sa.Integer(), nullable=True),
        sa.Column("es_id", sa.String(36), nullable=True),
        sa.Column("ingested_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("extra_metadata", sa.JSON(), nullable=True),
    )
    op.create_index("ix_chunks_doc_id", "chunks", ["doc_id"])
    op.create_index("ix_chunks_language", "chunks", ["language"])
    op.create_index("ix_chunks_milvus_id", "chunks", ["milvus_id"])

    # ── Table sessions ────────────────────────────────────────────────────────
    op.create_table(
        "sessions",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("user_id", sa.String(256), nullable=True),
        sa.Column("language", sa.String(10), server_default="fr"),
        sa.Column("profile_type", sa.String(32), server_default="citizen"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("last_active", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
        sa.Column("extra_data", sa.JSON(), nullable=True),
    )
    op.create_index("ix_sessions_user_id", "sessions", ["user_id"])

    # ── Table query_logs ──────────────────────────────────────────────────────
    op.create_table(
        "query_logs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("session_id", sa.String(36),
                  sa.ForeignKey("sessions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("user_id", sa.String(256), nullable=True),
        sa.Column("query", sa.Text(), nullable=False),
        sa.Column("language", sa.String(10), server_default="unknown"),
        sa.Column("profile", sa.String(32), server_default="citizen"),
        sa.Column("intent", sa.String(64), nullable=True),
        sa.Column("plan_json", sa.JSON(), nullable=True),
        sa.Column("retrieved_chunks_json", sa.JSON(), nullable=True),
        sa.Column("response_json", sa.JSON(), nullable=True),
        sa.Column("citations_json", sa.JSON(), nullable=True),
        sa.Column("score_conf", sa.Float(), nullable=True),
        sa.Column("latency_ms", sa.Float(), nullable=True),
        sa.Column("model_used", sa.String(128), nullable=True),
        sa.Column("safety_flags", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_query_logs_session_id", "query_logs", ["session_id"])
    op.create_index("ix_query_logs_user_id", "query_logs", ["user_id"])
    op.create_index("ix_query_logs_created_at", "query_logs", ["created_at"])
    op.create_index("ix_query_logs_language", "query_logs", ["language"])

    # ── Table audit_events ────────────────────────────────────────────────────
    op.create_table(
        "audit_events",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("event_type", sa.String(64), nullable=False),
        sa.Column("session_id", sa.String(36), nullable=True),
        sa.Column("user_id", sa.String(256), nullable=True),
        sa.Column("payload_json", sa.JSON(), nullable=True),
        sa.Column("is_flagged", sa.Boolean(), nullable=False, server_default="false"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False,
                  server_default=sa.text("CURRENT_TIMESTAMP")),
    )
    op.create_index("ix_audit_events_event_type", "audit_events", ["event_type"])
    op.create_index("ix_audit_events_session_id", "audit_events", ["session_id"])
    op.create_index("ix_audit_events_is_flagged", "audit_events", ["is_flagged"])
    op.create_index("ix_audit_events_created_at", "audit_events", ["created_at"])


def downgrade() -> None:
    op.drop_table("audit_events")
    op.drop_table("query_logs")
    op.drop_table("sessions")
    op.drop_table("chunks")
    op.drop_table("documents")
