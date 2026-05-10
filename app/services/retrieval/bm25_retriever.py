"""
GOV-AI 2.0 — Sparse retrieval BM25 via Elasticsearch.
Support bilingue FR/EN avec analyzeurs linguistiques.
"""
from __future__ import annotations

from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.elasticsearch_client import search_bm25

logger = get_logger(__name__)


class BM25Retriever:
    """Wrapper retrieval lexical Elasticsearch."""

    def __init__(self) -> None:
        self.settings = get_settings()

    async def search(
        self,
        query: str,
        top_k: int = 20,
        language: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Recherche BM25 avec analyzeurs bilingues FR/EN.

        Args:
            query: Texte de la requête
            top_k: Nombre de résultats
            language: Langue pour choisir l'analyseur (fr ou en)
            filters: Filtres métadonnées

        Returns:
            Liste de chunks avec sparse_score
        """
        try:
            results = await search_bm25(
                query=query,
                top_k=top_k,
                language=language,
                filters=filters,
            )
            logger.debug("bm25_search_done", results=len(results), top_k=top_k)
            return results
        except Exception as exc:
            logger.error("bm25_search_failed", error=str(exc))
            return []
