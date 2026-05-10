"""
GOV-AI 2.0 — Service de reranking cross-encoder.
Implémente l'étape [RERANKING] de l'Algorithme 2.

Score final (Équation 4.2 du mémoire) :
  s_final(q, c_i) = s_rerank(q, c_i) + β · w_meta(c_i)

Modèle : Qwen3-Reranker ou BAAI/bge-reranker-v2-m3 (déployé localement).
"""
from __future__ import annotations

import asyncio
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RerankingService:
    """
    Cross-encoder multilingue FR/EN pour le reranking.
    Pondération additionnelle par les métadonnées (pertinence documentaire).
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None

    async def _ensure_model_loaded(self) -> None:
        """Chargement lazy du cross-encoder."""
        if self._model is not None:
            return

        try:
            from sentence_transformers import CrossEncoder

            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: CrossEncoder(
                    self.settings.reranker_model,
                    device=self.settings.reranker_device,
                    max_length=512,
                ),
            )
            logger.info("reranker_model_loaded", model=self.settings.reranker_model)
        except ImportError:
            logger.warning(
                "reranker_model_unavailable",
                model=self.settings.reranker_model,
                fallback="rrf_score",
            )
        except Exception as exc:
            logger.error("reranker_load_failed", error=str(exc))

    async def rerank(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Reranke les chunks candidats via cross-encoder.

        Équation 4.2 :
          s_final = α · s_rerank + β · w_meta
          α = reranker_weight_score (0.85)
          β = reranker_weight_meta  (0.15)

        Args:
            query: Requête de l'utilisateur
            chunks: Chunks candidats (sortie RRF)
            top_k: Nombre final de résultats

        Returns:
            Chunks reranked avec rerank_score et final_score
        """
        if not chunks:
            return []

        await self._ensure_model_loaded()

        if self._model is None:
            # Fallback : utiliser le score RRF
            for chunk in chunks:
                chunk["rerank_score"] = chunk.get("rrf_score", 0.0)
                chunk["final_score"] = chunk.get("rrf_score", 0.0)
            return sorted(chunks, key=lambda c: c.get("final_score", 0.0), reverse=True)[:top_k]

        try:
            return await self._rerank_with_cross_encoder(query, chunks, top_k)
        except Exception as exc:
            logger.error("reranking_failed", error=str(exc))
            # Fallback sur RRF
            for chunk in chunks:
                chunk["final_score"] = chunk.get("rrf_score", 0.0)
            return sorted(chunks, key=lambda c: c["final_score"], reverse=True)[:top_k]

    async def _rerank_with_cross_encoder(
        self,
        query: str,
        chunks: list[dict[str, Any]],
        top_k: int,
    ) -> list[dict[str, Any]]:
        """Reranking effectif avec cross-encoder."""
        loop = asyncio.get_event_loop()

        # Préparer les paires (query, passage)
        pairs = [(query, chunk.get("content", "")) for chunk in chunks]

        # Prédiction en executor (bloquant)
        scores = await loop.run_in_executor(
            None,
            lambda: self._model.predict(pairs, show_progress_bar=False).tolist(),
        )

        α = self.settings.reranker_weight_score
        β = self.settings.reranker_weight_meta

        for chunk, score in zip(chunks, scores):
            chunk["rerank_score"] = float(score)
            w_meta = self._compute_metadata_weight(chunk)
            chunk["final_score"] = α * float(score) + β * w_meta

        # Trier par score final décroissant
        reranked = sorted(chunks, key=lambda c: c.get("final_score", 0.0), reverse=True)
        return reranked[:top_k]

    def _compute_metadata_weight(self, chunk: dict[str, Any]) -> float:
        """
        Calcule le poids des métadonnées (w_meta).
        Favorise les textes officiels (Constitution, lois) et les documents récents.
        Normalisation sur [0, 1].
        """
        weight = 0.5  # Base

        doc_type = chunk.get("doc_type", "").lower()
        institution = chunk.get("institution", "").lower()

        # Boost par type documentaire (hiérarchie normative camerounaise)
        type_weights = {
            "constitution": 1.0,
            "loi_organique": 0.9,
            "loi": 0.8,
            "ordonnance": 0.75,
            "decret": 0.7,
            "arrete": 0.65,
            "circulaire": 0.6,
            "acte_ohada": 0.75,
        }
        weight = max(weight, type_weights.get(doc_type, 0.5))

        # Boost institutions officielles
        if any(inst in institution for inst in ["presidence", "primature", "assemblee"]):
            weight = min(weight + 0.1, 1.0)

        return weight

    async def health_check(self) -> bool:
        """Vérifie que le reranker est opérationnel."""
        try:
            await self._ensure_model_loaded()
            return self._model is not None
        except Exception:
            return False
