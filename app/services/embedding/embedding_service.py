"""
GOV-AI 2.0 — Service d'embedding multilingue.
Supporte : local (sentence-transformers) ou Ollama.
Modèle par défaut : Qwen3-Embedding / BGE-M3 (déployé localement, souveraineté C9).
"""
from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any, Optional

from app.core.config import get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """
    Service d'embedding multilingue FR/EN.
    Supporte deux providers : local (sentence-transformers) ou Ollama.
    """

    def __init__(self) -> None:
        self.settings = get_settings()
        self._model: Any = None
        self._is_local = self.settings.embedding_provider == "local"

    async def _ensure_model_loaded(self) -> None:
        """Charge le modèle si pas encore initialisé (lazy loading)."""
        if self._model is not None:
            return

        if self._is_local:
            await self._load_local_model()
        else:
            logger.info(
                "embedding_provider_ollama",
                model=self.settings.embedding_model,
                url=self.settings.llm_base_url,
            )

    async def _load_local_model(self) -> None:
        """Charge le modèle sentence-transformers en tâche async."""
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("embedding_model_loading", model=self.settings.embedding_model)
            # Chargement dans un thread pour ne pas bloquer l'event loop
            loop = asyncio.get_event_loop()
            self._model = await loop.run_in_executor(
                None,
                lambda: SentenceTransformer(
                    self.settings.embedding_model,
                    device=self.settings.embedding_device,
                    cache_folder=self.settings.model_cache_path,
                ),
            )
            logger.info(
                "embedding_model_loaded",
                model=self.settings.embedding_model,
                dim=self.settings.embedding_dim,
            )
        except ImportError:
            raise EmbeddingError(
                "sentence-transformers non disponible. "
                "Installer via : pip install sentence-transformers"
            )
        except Exception as exc:
            raise EmbeddingError(f"Erreur chargement modèle embedding : {exc}") from exc

    async def embed_text(self, text: str) -> list[float]:
        """Calcule l'embedding d'un texte unique."""
        embeddings = await self.embed_texts([text])
        return embeddings[0]

    async def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Calcule les embeddings pour une liste de textes.
        Traitement par batch pour optimiser la mémoire.
        """
        if not texts:
            return []

        await self._ensure_model_loaded()

        try:
            if self._is_local:
                return await self._embed_local(texts)
            else:
                return await self._embed_ollama(texts)
        except Exception as exc:
            logger.error("embedding_failed", count=len(texts), error=str(exc))
            raise EmbeddingError(f"Erreur embedding : {exc}") from exc

    async def _embed_local(self, texts: list[str]) -> list[list[float]]:
        """Embedding avec sentence-transformers (local)."""
        loop = asyncio.get_event_loop()
        batch_size = self.settings.embedding_batch_size

        all_embeddings: list[list[float]] = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            embeddings = await loop.run_in_executor(
                None,
                lambda b=batch: self._model.encode(
                    b,
                    normalize_embeddings=True,
                    show_progress_bar=False,
                ).tolist(),
            )
            all_embeddings.extend(embeddings)

        return all_embeddings

    async def _embed_ollama(self, texts: list[str]) -> list[list[float]]:
        """Embedding via Ollama API (compatible avec mxbai-embed-large, etc.)."""
        import httpx

        all_embeddings: list[list[float]] = []
        base_url = self.settings.llm_base_url.rstrip("/")

        async with httpx.AsyncClient(timeout=60.0) as client:
            for text in texts:
                response = await client.post(
                    f"{base_url}/api/embeddings",
                    json={"model": self.settings.embedding_model, "prompt": text},
                )
                response.raise_for_status()
                data = response.json()
                embedding = data.get("embedding", [])
                if not embedding:
                    raise EmbeddingError(f"Embedding vide retourné par Ollama pour : {text[:50]}")
                all_embeddings.append(embedding)

        return all_embeddings

    async def health_check(self) -> bool:
        """Vérifie que le service d'embedding est opérationnel."""
        try:
            test_embedding = await self.embed_text("test de santé GOV-AI")
            return len(test_embedding) == self.settings.embedding_dim
        except Exception as exc:
            logger.error("embedding_health_check_failed", error=str(exc))
            return False


# ── Instance globale (singleton) ─────────────────────────────────────────────

_embedding_service: Optional[EmbeddingService] = None


def get_embedding_service() -> EmbeddingService:
    """Retourne le singleton EmbeddingService."""
    global _embedding_service
    if _embedding_service is None:
        _embedding_service = EmbeddingService()
    return _embedding_service
