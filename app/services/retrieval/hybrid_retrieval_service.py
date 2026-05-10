"""
GOV-AI 2.0 — Retrieval hybride avec fusion et reranking (Algorithme 2 du mémoire).

Pipeline complet :
  1. [RÉÉCRITURE] expansion requête (optionnel)
  2. [RECHERCHE SPARSE] BM25 (Elasticsearch)
  3. [RECHERCHE DENSE] dense (Milvus)
  4. [FUSION RRF] Reciprocal Rank Fusion
  5. [RERANKING] Cross-encoder + pondération métadonnées
  6. Retour des top_k_final chunks ordonnés
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import RetrievedChunk
from app.models.domain import Language
from app.services.retrieval.bm25_retriever import BM25Retriever
from app.services.retrieval.milvus_retriever import MilvusRetriever
from app.services.retrieval.rrf_fusion import merge_results_with_rrf

logger = get_logger(__name__)


class HybridRetrievalService:
    """
    Retrieval hybride : BM25 + dense + RRF + reranking.
    Implémente l'Algorithme 2 du mémoire (§4.3.2).
    """

    def __init__(
        self,
        embedding_service: Any = None,
        reranking_service: Any = None,
    ) -> None:
        self.settings = get_settings()
        self._embedding_service = embedding_service
        self._reranking_service = reranking_service
        self._bm25 = BM25Retriever()
        self._milvus = MilvusRetriever()

    def set_embedding_service(self, svc: Any) -> None:
        self._embedding_service = svc

    def set_reranking_service(self, svc: Any) -> None:
        self._reranking_service = svc

    async def retrieve(
        self,
        query: str,
        top_k: Optional[int] = None,
        top_k_sparse: int = 20,
        top_k_dense: int = 20,
        top_k_rerank: int = 20,
        top_k_final: int = 5,
        language: Optional[str] = None,
        filters: Optional[dict[str, Any]] = None,
        expand_query: bool = True,
    ) -> list[RetrievedChunk]:
        """
        Pipeline de retrieval hybride complet.

        Args:
            query: Requête de l'utilisateur
            top_k_sparse: Nombre de résultats BM25
            top_k_dense: Nombre de résultats dense
            top_k_rerank: Nombre de résultats à reranker
            top_k_final: Nombre final de résultats
            language: Langue détectée (pour l'analyseur BM25)
            filters: Filtres métadonnées
            expand_query: Activer la réécriture de requête

        Returns:
            Liste de RetrievedChunk ordonnés par score final
        """
        # top_k est un raccourci pour top_k_final (compatibilité orchestrateur)
        if top_k is not None:
            top_k_final = top_k

        # Étape 1 : Réécriture de requête (heuristique légère)
        query_expanded = query
        if expand_query:
            query_expanded = self._expand_query_heuristic(query)

        # Étapes 2 & 3 : Recherche parallèle (sparse + dense)
        sparse_results, dense_results = await asyncio.gather(
            self._bm25.search(
                query=query_expanded,
                top_k=top_k_sparse,
                language=language,
                filters=filters,
            ),
            self._dense_search(query_expanded, top_k_dense, filters),
        )

        logger.debug(
            "retrieval_raw_results",
            sparse=len(sparse_results),
            dense=len(dense_results),
        )

        if not sparse_results and not dense_results:
            logger.warning("retrieval_no_results", query=query[:100])
            return []

        # Étape 4 : Fusion RRF
        rrf_results = merge_results_with_rrf(
            dense_results=dense_results,
            sparse_results=sparse_results,
            top_k=top_k_rerank,
            k=self.settings.rrf_k,
        )

        # Étape 5 : Reranking (si disponible)
        if self._reranking_service is not None and rrf_results:
            ranked_results = await self._reranking_service.rerank(
                query=query,
                chunks=rrf_results,
                top_k=top_k_final,
            )
        else:
            # Sans reranker : utiliser le score RRF directement
            ranked_results = rrf_results[:top_k_final]
            for doc in ranked_results:
                doc["final_score"] = doc.get("rrf_score", 0.0)

        # Étape 6 : Conversion en RetrievedChunk
        retrieved_chunks = [
            self._to_retrieved_chunk(doc) for doc in ranked_results[:top_k_final]
        ]

        logger.info(
            "retrieval_completed",
            query=query[:50],
            chunks_returned=len(retrieved_chunks),
        )
        return retrieved_chunks

    async def _dense_search(
        self,
        query: str,
        top_k: int,
        filters: Optional[dict[str, Any]],
    ) -> list[dict]:
        """Embedding + recherche dense Milvus."""
        if self._embedding_service is None:
            logger.warning("no_embedding_service_for_dense_retrieval")
            return []

        try:
            query_embedding = await self._embedding_service.embed_text(query)
            return self._milvus.search(
                query_embedding=query_embedding,
                top_k=top_k,
                filters=filters,
            )
        except Exception as exc:
            logger.error("dense_search_failed", error=str(exc))
            return []

    def _expand_query_heuristic(self, query: str) -> str:
        """
        Expansion légère de requête sans LLM.
        Ajoute des synonymes courants du corpus juridique camerounais.
        Pour une expansion complète (LLM), voir cognitive_orchestrator.
        """
        # Expansions simples bilingues
        expansions = {
            "décret": "décret présidentiel",
            "arrêté": "arrêté ministériel",
            "demande": "demande officielle formulaire",
            "permis": "permis autorisation licence",
        }
        query_lower = query.lower()
        for term, expansion in expansions.items():
            if term in query_lower and expansion not in query_lower:
                return f"{query} {expansion}"
        return query

    def _to_retrieved_chunk(self, doc: dict) -> RetrievedChunk:
        """Convertit un dict brut en RetrievedChunk Pydantic."""
        return RetrievedChunk(
            chunk_id=doc.get("chunk_id", ""),
            doc_id=doc.get("doc_id", ""),
            content=doc.get("content", ""),
            source=doc.get("source", ""),
            language=Language(doc.get("language", "fr")),
            page=doc.get("page"),
            chunk_index=doc.get("chunk_index", 0),
            dense_score=doc.get("dense_score"),
            sparse_score=doc.get("sparse_score"),
            rrf_score=doc.get("rrf_score"),
            rerank_score=doc.get("rerank_score"),
            final_score=doc.get("final_score", doc.get("rrf_score", 0.0)),
            metadata={
                "doc_type": doc.get("doc_type", ""),
                "institution": doc.get("institution", ""),
                "jurisdiction": doc.get("jurisdiction", ""),
            },
        )
