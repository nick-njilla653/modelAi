"""
GOV-AI 2.0 — Dense retrieval via Milvus (HNSW + COSINE).
"""
from __future__ import annotations

from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.storage.milvus_client import search_dense

logger = get_logger(__name__)


class MilvusRetriever:
    """Wrapper retrieval vectoriel Milvus."""

    def __init__(self) -> None:
        self.settings = get_settings()

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 20,
        filters: Optional[dict[str, Any]] = None,
    ) -> list[dict[str, Any]]:
        """
        Recherche par similarité vectorielle.

        Args:
            query_embedding: Vecteur d'embedding de la requête
            top_k: Nombre de résultats
            filters: Filtres métadonnées (ex: {"language": "fr"})

        Returns:
            Liste de chunks avec dense_score
        """
        # Construire l'expression de filtre Milvus
        filter_expr = _build_milvus_filter(filters)

        try:
            results = search_dense(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filter_expr,
                ef=self.settings.milvus_ef,
            )
            logger.debug("milvus_search_done", results=len(results), top_k=top_k)
            return results
        except Exception as exc:
            logger.error("milvus_search_failed", error=str(exc))
            return []


def _build_milvus_filter(filters: Optional[dict[str, Any]]) -> Optional[str]:
    """Convertit un dict de filtres en expression Milvus."""
    if not filters:
        return None

    clauses = []
    for field, value in filters.items():
        if isinstance(value, str):
            clauses.append(f'{field} == "{value}"')
        elif isinstance(value, list):
            values_str = ", ".join(f'"{v}"' for v in value)
            clauses.append(f"{field} in [{values_str}]")
    return " && ".join(clauses) if clauses else None
