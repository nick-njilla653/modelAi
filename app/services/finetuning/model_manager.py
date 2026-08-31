"""
GOV-AI 2.0 — Model Manager.
Gère l'intégration automatique des modèles fine-tunés (embedding + LLM).
Hot-swap sans redémarrage complet grâce aux singletons de service.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

# Fichier de registre des modèles actifs (persiste entre les redémarrages)
_REGISTRY_PATH = Path("models/active_registry.json")

# Benchmarks de référence mondiaux (publiés, MMLU / MTEB / RAG-specific)
REFERENCE_BENCHMARKS = {
    "gpt-4o": {
        "label": "GPT-4o (OpenAI)",
        "mteb_score": 0.646,
        "mmlu_score": 0.887,
        "rag_faithfulness": 0.91,
        "rag_answer_relevancy": 0.89,
        "embedding_sim": 0.85,
        "color": "#10a37f",
    },
    "claude-3-5-sonnet": {
        "label": "Claude 3.5 Sonnet (Anthropic)",
        "mteb_score": 0.635,
        "mmlu_score": 0.889,
        "rag_faithfulness": 0.93,
        "rag_answer_relevancy": 0.91,
        "embedding_sim": 0.83,
        "color": "#d97706",
    },
    "mistral-7b": {
        "label": "Mistral 7B Instruct",
        "mteb_score": 0.563,
        "mmlu_score": 0.641,
        "rag_faithfulness": 0.78,
        "rag_answer_relevancy": 0.76,
        "embedding_sim": 0.72,
        "color": "#7c3aed",
    },
    "llama3.2-3b": {
        "label": "Llama 3.2 3B (base)",
        "mteb_score": 0.541,
        "mmlu_score": 0.598,
        "rag_faithfulness": 0.74,
        "rag_answer_relevancy": 0.71,
        "embedding_sim": 0.68,
        "color": "#2563eb",
    },
}


class ModelManager:
    """Gère le registre des modèles actifs et les opérations de hot-swap."""

    def __init__(self):
        _REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
        if not _REGISTRY_PATH.exists():
            self._save_registry(self._default_registry())

    # ── Registre ──────────────────────────────────────────────────────────────

    def get_active(self) -> dict:
        return self._load_registry()

    def get_history(self) -> list[dict]:
        registry = self._load_registry()
        return registry.get("history", [])

    def apply_embedding(self, job_id: str, model_path: str, job_name: str) -> dict:
        """Hot-swap le modèle d'embedding et recharge tous les services dépendants."""
        registry = self._load_registry()
        old = registry.get("embedding_model", {})
        new_entry = {
            "job_id": job_id,
            "job_name": job_name,
            "model_path": model_path,
            "applied_at": _now_iso(),
        }
        registry.setdefault("history", []).append({"type": "embedding", "previous": old, "new": new_entry})
        registry["embedding_model"] = new_entry
        self._save_registry(registry)

        # 1. Mise à jour de la variable d'environnement
        os.environ["EMBEDDING_MODEL"] = model_path

        # 2. Invalider le cache de settings (get_settings est @lru_cache)
        try:
            from app.core.config import get_settings
            get_settings.cache_clear()
            logger.info("settings_cache_cleared")
        except Exception as exc:
            logger.warning("settings_cache_clear_failed", error=str(exc))

        # 3. Réinitialiser le singleton d'embedding
        try:
            from app.services import embedding as embed_module
            if hasattr(embed_module, "_embedding_service"):
                embed_module._embedding_service = None
                logger.info("embedding_singleton_cleared")
        except Exception as exc:
            logger.warning("embedding_reload_skipped", error=str(exc))

        # 4. Réinitialiser le service de retrieval dans l'orchestrateur (tient une ref à l'ancien embedding)
        try:
            import app.services.cognitive_orchestrator as orch_module
            if orch_module._orchestrator_instance is not None:
                orch_module._orchestrator_instance._retrieval_service = None
                logger.info("orchestrator_retrieval_service_reset")
        except Exception as exc:
            logger.warning("orchestrator_reset_failed", error=str(exc))

        logger.info("embedding_model_swapped", path=model_path)
        return new_entry

    def apply_llm(self, job_id: str, ollama_model_name: str, job_name: str) -> dict:
        """Hot-swap le modèle LLM et recharge les services de génération."""
        registry = self._load_registry()
        old = registry.get("llm_model", {})
        new_entry = {
            "job_id": job_id,
            "job_name": job_name,
            "ollama_model_name": ollama_model_name,
            "applied_at": _now_iso(),
        }
        registry.setdefault("history", []).append({"type": "llm", "previous": old, "new": new_entry})
        registry["llm_model"] = new_entry
        self._save_registry(registry)

        # 1. Mise à jour de la variable d'environnement
        os.environ["LLM_MODEL"] = ollama_model_name

        # 2. Invalider le cache de settings
        try:
            from app.core.config import get_settings
            get_settings.cache_clear()
            logger.info("settings_cache_cleared")
        except Exception as exc:
            logger.warning("settings_cache_clear_failed", error=str(exc))

        # 3. Réinitialiser le service de génération dans l'orchestrateur
        #    (LLMAnswerService lit self.settings.llm_model → doit être recréé)
        try:
            import app.services.cognitive_orchestrator as orch_module
            if orch_module._orchestrator_instance is not None:
                orch_module._orchestrator_instance._generation_service = None
                logger.info("orchestrator_generation_service_reset", model=ollama_model_name)
        except Exception as exc:
            logger.warning("orchestrator_generation_reset_failed", error=str(exc))

        logger.info("llm_model_swapped", model=ollama_model_name)
        return new_entry

    def apply_reranker(self, job_id: str, model_path: str, job_name: str) -> dict:
        """Hot-swap le modèle de reranking et réinitialise le service dans l'orchestrateur."""
        registry = self._load_registry()
        old = registry.get("reranker_model", {})
        new_entry = {
            "job_id": job_id,
            "job_name": job_name,
            "model_path": model_path,
            "applied_at": _now_iso(),
        }
        registry.setdefault("history", []).append({"type": "reranker", "previous": old, "new": new_entry})
        registry["reranker_model"] = new_entry
        self._save_registry(registry)

        # Mise à jour de la variable d'environnement
        os.environ["RERANKER_MODEL"] = model_path

        # Invalider le cache de settings
        try:
            from app.core.config import get_settings
            get_settings.cache_clear()
        except Exception as exc:
            logger.warning("settings_cache_clear_failed", error=str(exc))

        # Réinitialiser le service de reranking dans l'orchestrateur
        try:
            import app.services.cognitive_orchestrator as orch_module
            if orch_module._orchestrator_instance is not None:
                orch_module._orchestrator_instance._reranking_service = None
                logger.info("orchestrator_reranking_service_reset", model=model_path)
        except Exception as exc:
            logger.warning("orchestrator_reranker_reset_failed", error=str(exc))

        logger.info("reranker_model_swapped", path=model_path)
        return new_entry

    def rollback(self, model_type: str) -> Optional[dict]:
        """Retourne au modèle précédent (embedding, llm ou reranker)."""
        registry = self._load_registry()
        history = registry.get("history", [])
        candidates = [h for h in reversed(history) if h["type"] == model_type]
        if len(candidates) < 1:
            return None
        previous = candidates[0].get("previous")
        if not previous:
            return None
        if model_type == "embedding":
            registry["embedding_model"] = previous
            os.environ["EMBEDDING_MODEL"] = previous.get("model_path", "")
        elif model_type == "reranker":
            registry["reranker_model"] = previous
            os.environ["RERANKER_MODEL"] = previous.get("model_path", "")
        else:
            registry["llm_model"] = previous
            os.environ["LLM_MODEL"] = previous.get("ollama_model_name", "")
        # Invalider settings cache après rollback
        try:
            from app.core.config import get_settings
            get_settings.cache_clear()
        except Exception:
            pass
        self._save_registry(registry)
        return previous

    # ── Benchmarks de référence ───────────────────────────────────────────────

    def get_reference_benchmarks(self) -> dict:
        return REFERENCE_BENCHMARKS

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _load_registry(self) -> dict:
        try:
            registry = json.loads(_REGISTRY_PATH.read_text(encoding="utf-8"))
        except Exception:
            return self._default_registry()
        return self._reconcile_with_config(registry)

    def _reconcile_with_config(self, registry: dict) -> dict:
        """
        Aligne les entrées « modèle de base » sur la configuration courante.

        Le registre n'était renseigné qu'à sa création : changer LLM_MODEL dans
        la configuration laissait ensuite le registre désigner l'ancien modèle
        indéfiniment. Un retour arrière restaurait alors un état qui ne
        correspondait à rien de réel.

        La règle : une entrée SANS job_id désigne un modèle de base, elle suit
        donc la configuration. Une entrée AVEC job_id résulte d'un fine-tuning
        délibérément appliqué — la configuration ne doit pas l'écraser.
        """
        settings = get_settings()
        expected = {
            "llm_model": ("ollama_model_name", settings.llm_model),
            "embedding_model": ("model_path", settings.embedding_model),
            "reranker_model": ("model_path", settings.reranker_model),
        }

        changed = False
        for key, (field, configured) in expected.items():
            entry = registry.get(key)
            if not isinstance(entry, dict) or entry.get("job_id"):
                # Modèle issu d'un fine-tuning : il fait autorité.
                continue
            if entry.get(field) != configured:
                logger.info(
                    "registry_realigned",
                    model_type=key,
                    was=entry.get(field),
                    now=configured,
                )
                entry[field] = configured
                entry["applied_at"] = _now_iso()
                changed = True

        if changed:
            self._save_registry(registry)
        return registry

    def _save_registry(self, data: dict):
        _REGISTRY_PATH.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

    def _default_registry(self) -> dict:
        return {
            "embedding_model": {
                "model_path": os.getenv("EMBEDDING_MODEL", "mixedbread-ai/mxbai-embed-large-v1"),
                "job_id": None,
                "applied_at": _now_iso(),
            },
            "llm_model": {
                "ollama_model_name": os.getenv("LLM_MODEL", "llama3.2:latest"),
                "job_id": None,
                "applied_at": _now_iso(),
            },
            "reranker_model": {
                "model_path": os.getenv("RERANKER_MODEL", "BAAI/bge-reranker-v2-m3"),
                "job_id": None,
                "applied_at": _now_iso(),
            },
            "history": [],
        }


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
