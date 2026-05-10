#!/usr/bin/env python3
"""
Script de démarrage pour l'API du système RAG juridique camerounais.
"""

import os
import sys
import logging
import subprocess
import argparse
from pathlib import Path

from app.core.config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Démarrage de l'API RAG Juridique")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Adresse d'hôte")
    parser.add_argument("--port", type=int, default=8000, help="Port")
    parser.add_argument("--reload", action="store_true", help="Rechargement automatique")
    parser.add_argument("--debug", action="store_true", help="Mode debug")
    return parser.parse_args()


def check_environment():
    settings = get_settings()
    data_dirs = [
        settings.DATA_PATH,
        settings.METADATA_PATH,
        str(Path(settings.METADATA_PATH) / "chunks"),
        str(Path(settings.METADATA_PATH) / "chat_history"),
    ]
    for directory in data_dirs:
        if not os.path.exists(directory):
            logger.info(f"Création du répertoire {directory}")
            os.makedirs(directory, exist_ok=True)


def main():
    args = parse_args()
    settings = get_settings()

    logger.info("🚀 Démarrage du système RAG juridique camerounais")

    os.environ.setdefault("MILVUS_HOST", settings.MILVUS_HOST)
    os.environ.setdefault("MILVUS_PORT", str(settings.MILVUS_PORT))
    logger.info(
        f"Configuration Milvus : host={settings.MILVUS_HOST}, port={settings.MILVUS_PORT}"
    )
    logger.info(f"Embedding : {settings.EMBEDDING_SERVICE_URL} (modèle: {settings.EMBEDDING_MODEL})")
    logger.info(f"LLM : {settings.LLM_SERVICE_URL} (modèle: {settings.LLM_MODEL})")

    check_environment()

    project_root = Path(__file__).resolve().parent
    cmd = [
        sys.executable,
        "-m",
        "uvicorn",
        "app.main:app",
        "--host",
        args.host,
        "--port",
        str(args.port),
    ]

    if args.reload:
        cmd.append("--reload")

    cmd.append("--log-level=debug" if args.debug else "--log-level=info")

    logger.info(f"🌐 Démarrage de l'API sur {args.host}:{args.port}")
    try:
        subprocess.run(cmd, cwd=project_root)
    except KeyboardInterrupt:
        logger.info("👋 Arrêt de l'API")
    except Exception as e:
        logger.error(f"❌ Erreur lors du démarrage : {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
