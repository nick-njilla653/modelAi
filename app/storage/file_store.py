"""
GOV-AI 2.0 — Stockage fichiers documents (on-premise).
"""
from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Optional

import aiofiles

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_doc_path(doc_id: str, filename: str) -> Path:
    settings = get_settings()
    doc_dir = Path(settings.documents_path) / doc_id
    doc_dir.mkdir(parents=True, exist_ok=True)
    return doc_dir / filename


async def save_document(doc_id: str, filename: str, content: bytes) -> str:
    """Sauvegarde un fichier document. Retourne le chemin absolu."""
    path = _get_doc_path(doc_id, filename)
    async with aiofiles.open(path, "wb") as f:
        await f.write(content)
    logger.info("file_saved", doc_id=doc_id, path=str(path), size_bytes=len(content))
    return str(path)


def get_document_path(doc_id: str, filename: str) -> Optional[str]:
    """Retourne le chemin si le fichier existe."""
    path = _get_doc_path(doc_id, filename)
    return str(path) if path.exists() else None


def delete_document_files(doc_id: str) -> None:
    """Supprime tous les fichiers d'un document."""
    settings = get_settings()
    doc_dir = Path(settings.documents_path) / doc_id
    if doc_dir.exists():
        shutil.rmtree(doc_dir)
        logger.info("file_deleted", doc_id=doc_id)


def compute_file_hash(content: bytes) -> str:
    """Calcule le hash SHA-256 d'un fichier (déduplication)."""
    return hashlib.sha256(content).hexdigest()


def get_storage_stats() -> dict[str, int]:
    """Statistiques du stockage fichiers."""
    settings = get_settings()
    docs_path = Path(settings.documents_path)
    if not docs_path.exists():
        return {"documents": 0, "total_size_bytes": 0}

    files = list(docs_path.rglob("*"))
    total_size = sum(f.stat().st_size for f in files if f.is_file())
    return {
        "documents": len([f for f in docs_path.iterdir() if f.is_dir()]),
        "total_size_bytes": total_size,
    }
