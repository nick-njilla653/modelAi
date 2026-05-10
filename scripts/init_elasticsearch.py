"""
GOV-AI 2.0 — Script d'initialisation Elasticsearch (index BM25 FR/EN).
Usage : python scripts/init_elasticsearch.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    from app.core.config import get_settings
    from app.storage.elasticsearch_client import get_elasticsearch_client, ensure_index_exists

    settings = get_settings()
    print(f"Connexion Elasticsearch : {settings.elasticsearch_url}")

    try:
        es = get_elasticsearch_client()
        ok = await es.ping()
        if not ok:
            print("Elasticsearch inaccessible.")
            sys.exit(1)
        print("Connexion Elasticsearch : OK")

        await ensure_index_exists()
        print(f"Index '{settings.elasticsearch_index_chunks}' prêt.")
        print("  Analyseurs : fr (french_analyzer) + en (english_analyzer)")
        print("Initialisation Elasticsearch terminée.")

    except Exception as exc:
        print(f"Erreur : {exc}")
        sys.exit(1)
    finally:
        try:
            from app.storage.elasticsearch_client import close_elasticsearch
            await close_elasticsearch()
        except Exception:
            pass


if __name__ == "__main__":
    asyncio.run(main())
