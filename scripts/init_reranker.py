"""
GOV-AI 2.0 — Pré-téléchargement du cross-encoder de reranking.

Le service de reranking charge le modèle paresseusement, à la première requête,
sous un timeout de 45 s (app/services/reranking/reranking_service.py). Or le
premier chargement doit télécharger ~2,2 Go depuis HuggingFace : le timeout tombe
toujours avant la fin, le pipeline retombe silencieusement sur le score RRF, et
l'étape de reranking décrite dans l'architecture ne s'exécute jamais.

Ce script effectue le téléchargement une bonne fois, sans timeout. Une fois le
modèle en cache (volume govai_storage, HF_HOME=/app/storage/model_cache), le
chargement se fait depuis le disque en quelques secondes et le timeout de 45 s
n'est plus un obstacle.

À lancer une fois après `docker compose up`, comme init-models.ps1 pour Ollama :
    docker exec govai2-api python scripts/init_reranker.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


def main() -> int:
    from app.core.config import get_settings

    settings = get_settings()
    model_name = settings.reranker_model
    device = settings.reranker_device

    print(f"Modèle    : {model_name}")
    print(f"Device    : {device}")
    print("Téléchargement puis chargement (plusieurs minutes au premier appel)…")

    start = time.perf_counter()
    try:
        from sentence_transformers import CrossEncoder

        model = CrossEncoder(model_name, device=device, max_length=512)
    except ImportError:
        print("sentence-transformers n'est pas installé dans cet environnement.")
        return 1
    except Exception as exc:
        print(f"Échec du chargement : {exc}")
        return 1

    elapsed = time.perf_counter() - start
    print(f"Modèle chargé en {elapsed:.1f}s.")

    # Vérification fonctionnelle : le cross-encoder doit classer le passage
    # pertinent avant le passage hors sujet.
    pairs = [
        ("Qu'est-ce que le flagrant délit ?",
         "Est qualifié crime ou délit flagrant, le crime ou le délit qui se commet actuellement."),
        ("Qu'est-ce que le flagrant délit ?",
         "Le Tribunal peut ordonner tout transport sur les lieux à la demande des parties."),
    ]
    scores = model.predict(pairs)
    print(f"Score passage pertinent   : {scores[0]:.4f}")
    print(f"Score passage hors sujet  : {scores[1]:.4f}")

    if scores[0] <= scores[1]:
        print("ATTENTION : le modèle ne discrimine pas les deux passages.")
        return 1

    print("Reranker opérationnel — le pipeline ne retombera plus sur le score RRF.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
