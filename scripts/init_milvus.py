"""
GOV-AI 2.0 — Script d'initialisation Milvus (collection + index HNSW).
Usage : python scripts/init_milvus.py
"""
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def main() -> None:
    from app.core.config import get_settings
    from app.storage.milvus_client import (
        connect_milvus,
        ensure_collection_exists,
        check_milvus_connection,
    )

    settings = get_settings()
    print(f"Connexion Milvus : {settings.milvus_host}:{settings.milvus_port}")

    try:
        connect_milvus()
        ok = await check_milvus_connection()
        if not ok:
            print("Milvus inaccessible.")
            sys.exit(1)

        await ensure_collection_exists()
        print(f"Collection '{settings.milvus_collection_chunks}' prête.")
        print(f"  dim={settings.milvus_dim}, index={settings.milvus_index_type}, metric={settings.milvus_metric_type}")
        print("Initialisation Milvus terminée.")

    except Exception as exc:
        print(f"Erreur : {exc}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
