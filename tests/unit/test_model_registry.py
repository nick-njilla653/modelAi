"""
Tests unitaires — Registre des modèles actifs (model_manager).

Le registre n'était renseigné qu'à sa création : changer LLM_MODEL dans la
configuration le laissait ensuite désigner l'ancien modèle indéfiniment. Un
retour arrière restaurait alors un état sans rapport avec le système en marche.

Règle vérifiée ici :
  - entrée SANS job_id  → modèle de base, elle suit la configuration ;
  - entrée AVEC job_id  → issue d'un fine-tuning appliqué, elle fait autorité
    et la configuration ne l'écrase pas.
"""
import json

import pytest

from app.services.finetuning import model_manager as mm


@pytest.fixture
def registry_file(tmp_path, monkeypatch):
    """Isole le registre dans un fichier temporaire."""
    path = tmp_path / "active_registry.json"
    monkeypatch.setattr(mm, "_REGISTRY_PATH", path)
    return path


def _write(path, llm_entry: dict) -> None:
    path.write_text(
        json.dumps({
            "embedding_model": {"model_path": "mxbai-embed-large", "job_id": None},
            "llm_model": llm_entry,
            "reranker_model": {"model_path": "BAAI/bge-reranker-v2-m3", "job_id": None},
            "history": [],
        }),
        encoding="utf-8",
    )


class TestRegistryReconciliation:

    def test_base_model_follows_configuration(self, registry_file, settings):
        """Régression : le registre désignait llama3.2 alors qu'un 7B tournait."""
        _write(registry_file, {"ollama_model_name": "llama3.2:latest", "job_id": None})

        active = mm.ModelManager().get_active()

        assert active["llm_model"]["ollama_model_name"] == settings.llm_model

    def test_realignment_is_persisted(self, registry_file, settings):
        """La correction doit survivre au redémarrage, pas vivre en mémoire."""
        _write(registry_file, {"ollama_model_name": "llama3.2:latest", "job_id": None})

        mm.ModelManager().get_active()
        on_disk = json.loads(registry_file.read_text(encoding="utf-8"))

        assert on_disk["llm_model"]["ollama_model_name"] == settings.llm_model

    def test_finetuned_model_is_not_overwritten(self, registry_file):
        """Un modèle fine-tuné appliqué délibérément fait autorité."""
        _write(registry_file, {
            "ollama_model_name": "govai-cpp-v1",
            "job_id": "job-1234",
            "job_name": "Corpus CPP",
        })

        active = mm.ModelManager().get_active()

        assert active["llm_model"]["ollama_model_name"] == "govai-cpp-v1"
        assert active["llm_model"]["job_id"] == "job-1234"

    def test_already_aligned_registry_is_left_alone(self, registry_file, settings):
        """Sans divergence, aucune réécriture — donc pas d'horodatage qui bouge."""
        _write(registry_file, {
            "ollama_model_name": settings.llm_model,
            "job_id": None,
            "applied_at": "2020-01-01T00:00:00+00:00",
        })

        active = mm.ModelManager().get_active()

        assert active["llm_model"]["applied_at"] == "2020-01-01T00:00:00+00:00"

    def test_embedding_and_reranker_follow_the_same_rule(self, registry_file, settings):
        path = registry_file
        path.write_text(
            json.dumps({
                "embedding_model": {"model_path": "ancien-embedding", "job_id": None},
                "llm_model": {"ollama_model_name": settings.llm_model, "job_id": None},
                "reranker_model": {"model_path": "ancien-reranker", "job_id": None},
                "history": [],
            }),
            encoding="utf-8",
        )

        active = mm.ModelManager().get_active()

        assert active["embedding_model"]["model_path"] == settings.embedding_model
        assert active["reranker_model"]["model_path"] == settings.reranker_model

    def test_unreadable_registry_falls_back_to_defaults(self, registry_file, settings):
        registry_file.write_text("{ ceci n'est pas du JSON", encoding="utf-8")

        active = mm.ModelManager().get_active()

        assert active["llm_model"]["ollama_model_name"] == settings.llm_model
