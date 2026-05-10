"""
GOV-AI 2.0 — Script d'initialisation PostgreSQL.
Lance les migrations Alembic et insère les données de référence.
Usage : python scripts/init_db.py
"""
import asyncio
import subprocess
import sys
from pathlib import Path

# Ajouter la racine au PYTHONPATH
sys.path.insert(0, str(Path(__file__).parent.parent))


async def run_migrations() -> None:
    print("Lancement des migrations Alembic...")
    result = subprocess.run(
        ["alembic", "upgrade", "head"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"Erreur migrations : {result.stderr}")
        sys.exit(1)
    print(result.stdout)
    print("Migrations terminées.")


async def check_connection() -> bool:
    from app.storage.postgres_client import check_postgres_connection
    ok = await check_postgres_connection()
    if ok:
        print("Connexion PostgreSQL : OK")
    else:
        print("Connexion PostgreSQL : ECHEC")
    return ok


async def main() -> None:
    from app.core.config import get_settings
    settings = get_settings()
    print(f"Base de données : {settings.database_url}")

    if not await check_connection():
        sys.exit(1)

    await run_migrations()
    print("Initialisation PostgreSQL terminée.")


if __name__ == "__main__":
    asyncio.run(main())
