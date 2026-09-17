"""
GOV-AI 2.0 — Suivi des ingestions en cours.

L'ingestion d'un PDF scanné dure plusieurs minutes : maintenir une requête HTTP
ouverte pendant tout ce temps expose aux coupures de proxy et laisse l'interface
sans le moindre retour. Le téléversement rend donc la main immédiatement, et
l'avancement se consulte par sondage.

Le registre est en mémoire du processus : il suffit à un déploiement à une seule
instance d'API, qui est le cadre visé. Une exécution multi-instances demanderait
un stockage partagé (Redis ou PostgreSQL).
"""
from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

# Nombre de travaux conservés après achèvement, pour que l'interface puisse
# encore afficher le résultat d'un téléversement terminé.
_MAX_RETAINED = 40


@dataclass
class IngestJob:
    """État d'une ingestion, du dépôt du fichier à l'indexation."""
    job_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    filename: str = ""
    source: str = ""
    status: str = "pending"          # pending | running | completed | failed
    stage: str = "file"              # file | extraction | ocr | chunking | indexation | …
    detail: str = "En attente de traitement"
    progress: int = 0
    chunks_created: int = 0
    document_id: Optional[str] = None
    language: Optional[str] = None
    ocr_used: bool = False
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None
    started_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "job_id": self.job_id,
            "filename": self.filename,
            "source": self.source,
            "status": self.status,
            "stage": self.stage,
            "detail": self.detail,
            "progress": self.progress,
            "chunks_created": self.chunks_created,
            "document_id": self.document_id,
            "language": self.language,
            "ocr_used": self.ocr_used,
            "warnings": self.warnings,
            "error": self.error,
            "elapsed_s": round((self.finished_at or time.time()) - self.started_at, 1),
        }


class IngestJobRegistry:
    """Registre des ingestions, ordonné du plus récent au plus ancien."""

    def __init__(self) -> None:
        self._jobs: dict[str, IngestJob] = {}
        self._lock = asyncio.Lock()

    async def create(self, filename: str, source: str) -> IngestJob:
        job = IngestJob(filename=filename, source=source)
        async with self._lock:
            self._jobs[job.job_id] = job
            self._prune()
        return job

    def get(self, job_id: str) -> Optional[IngestJob]:
        return self._jobs.get(job_id)

    def recent(self, limit: int = 20) -> list[IngestJob]:
        return sorted(self._jobs.values(), key=lambda j: j.started_at, reverse=True)[:limit]

    def _prune(self) -> None:
        """Écarte les travaux terminés les plus anciens au-delà du plafond."""
        finished = [j for j in self._jobs.values() if j.finished_at is not None]
        if len(finished) <= _MAX_RETAINED:
            return
        for job in sorted(finished, key=lambda j: j.finished_at or 0)[: len(finished) - _MAX_RETAINED]:
            self._jobs.pop(job.job_id, None)


_registry = IngestJobRegistry()


def get_ingest_registry() -> IngestJobRegistry:
    return _registry


async def run_ingestion(job: IngestJob, tmp_path: str, request) -> None:
    """
    Exécute l'ingestion en tâche de fond et tient le travail à jour.

    Toute exception est capturée : une tâche de fond qui meurt en silence
    laisserait le travail bloqué en « running » indéfiniment.
    """
    from pathlib import Path

    from app.services.embedding import get_embedding_service
    from app.services.ingestion.ingestion_service import IngestionService

    job.status = "running"
    job.stage = "extraction"
    job.detail = "Lecture du document"

    async def progress(stage: str, percent: int, detail: str) -> None:
        job.stage = stage
        job.progress = percent
        job.detail = detail

    try:
        service = IngestionService()
        service.set_embedding_service(get_embedding_service())

        result = await service.ingest_document(
            file_path=tmp_path,
            request=request,
            progress_cb=progress,
        )

        job.status = "completed"
        job.stage = "termine"
        job.progress = 100
        job.chunks_created = len(result.chunks)
        job.document_id = result.document_id
        job.language = result.metadata.language.value if result.metadata else None
        job.ocr_used = result.ocr_used
        job.warnings = list(result.warnings)
        job.detail = f"{len(result.chunks)} passages indexés"

        logger.info(
            "ingest_job_completed",
            job_id=job.job_id,
            chunks=job.chunks_created,
            source=job.source,
        )
    except Exception as exc:
        job.status = "failed"
        job.error = str(exc) or type(exc).__name__
        job.detail = "L'ingestion a échoué"
        logger.error("ingest_job_failed", job_id=job.job_id, error=job.error, exc_info=True)
    finally:
        job.finished_at = time.time()
        Path(tmp_path).unlink(missing_ok=True)
