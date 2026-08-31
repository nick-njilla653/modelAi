"""Documents attachés à une conversation

Distincts du corpus global : ils ne sont écrits ni dans Milvus ni dans
Elasticsearch, et disparaissent avec la conversation. Leur vecteur est stocké
ici même, ce qui rend l'isolement structurel plutôt que dépendant d'un filtre.

Revision ID: 003_session_documents
Revises: 002_chunk_article_ref
Create Date: 2026-08-31 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "003_session_documents"
down_revision: Union[str, None] = "002_chunk_article_ref"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "session_documents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "session_id",
            sa.String(36),
            sa.ForeignKey("sessions.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("filename", sa.String(512), nullable=False),
        sa.Column("source", sa.String(512), nullable=False),
        sa.Column("language", sa.String(10), server_default="unknown"),
        sa.Column("doc_type", sa.String(64), nullable=True),
        sa.Column("page_count", sa.Integer(), nullable=True),
        sa.Column("ocr_used", sa.Boolean(), server_default="false"),
        sa.Column("chunk_count", sa.Integer(), server_default="0"),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
        ),
    )
    op.create_index(
        "ix_session_documents_session", "session_documents", ["session_id"]
    )

    op.create_table(
        "session_document_chunks",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "document_id",
            sa.String(36),
            sa.ForeignKey("session_documents.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("session_id", sa.String(36), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("chunk_index", sa.Integer(), server_default="0"),
        sa.Column("page", sa.Integer(), nullable=True),
        sa.Column("article_ref", sa.String(64), nullable=True),
        sa.Column("token_count", sa.Integer(), nullable=True),
        sa.Column("embedding", sa.JSON(), nullable=True),
    )
    op.create_index(
        "ix_session_document_chunks_session",
        "session_document_chunks",
        ["session_id"],
    )


def downgrade() -> None:
    op.drop_index(
        "ix_session_document_chunks_session", table_name="session_document_chunks"
    )
    op.drop_table("session_document_chunks")
    op.drop_index("ix_session_documents_session", table_name="session_documents")
    op.drop_table("session_documents")
