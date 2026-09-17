"""Référence d'article citable sur les chunks

Extraite à l'ingestion (« Article 103 »), persistée ici puis réinjectée dans le
prompt de génération. Sans elle, le modèle reconstruit les numéros d'articles à
partir du texte brut et les attribue au mauvais article.

Revision ID: 002_chunk_article_ref
Revises: 001_initial_schema
Create Date: 2026-08-31 00:00:00.000000
"""
from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

revision: str = "002_chunk_article_ref"
down_revision: Union[str, None] = "001_initial_schema"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "chunks",
        sa.Column("article_ref", sa.String(64), nullable=True),
    )
    # Les requêtes de vérification de citation filtrent sur (doc, article).
    op.create_index("ix_chunks_article_ref", "chunks", ["article_ref"])


def downgrade() -> None:
    op.drop_index("ix_chunks_article_ref", table_name="chunks")
    op.drop_column("chunks", "article_ref")
