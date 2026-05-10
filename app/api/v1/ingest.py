"""
GOV-AI 2.0 — Route d'ingestion de documents (/api/v1/ingest).
"""
from __future__ import annotations

import tempfile
import time
import uuid
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.core.dependencies import get_audit_dep
from app.core.logging import get_logger
from app.models.schemas import IngestResponse
from app.services.audit.security_filters import validate_filename

router = APIRouter(tags=["ingestion"])
logger = get_logger(__name__)

_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB


@router.post(
    "/ingest",
    response_model=IngestResponse,
    summary="Ingestion d'un document juridique",
    description=(
        "Upload un document PDF/TXT/MD. "
        "Extraction → chunking → embedding → indexation Milvus + Elasticsearch."
    ),
)
async def ingest_document(
    file: UploadFile = File(..., description="Document à ingérer (PDF, TXT, MD)"),
    doc_type: str = Form("autre", description="Type documentaire (loi, decret, arrete…)"),
    institution: str = Form("", description="Institution émettrice"),
    jurisdiction: str = Form("national", description="Juridiction (national, NW, SW…)"),
    force_ocr: bool = Form(False, description="Forcer l'OCR même si le texte est extractible"),
    audit=Depends(get_audit_dep),
) -> IngestResponse:
    """
    Pipeline d'ingestion :
    1. Validation sécurité (nom de fichier, taille, extension)
    2. Extraction texte (PyMuPDF + OCR si nécessaire)
    3. Chunking structurel / taille fixe
    4. Embedding (mxbai-embed-large via Ollama)
    5. Indexation Milvus (dense) + Elasticsearch (BM25)
    6. Persistance métadonnées PostgreSQL
    """
    trace_id = str(uuid.uuid4())
    start = time.perf_counter()

    filename = file.filename or "document"

    # Validation nom de fichier
    is_valid, err_msg = validate_filename(filename)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    # Lecture et vérification taille
    content = await file.read()
    if len(content) > _MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"Fichier trop volumineux (max {_MAX_FILE_SIZE // 1024 // 1024} MB)",
        )

    suffix = Path(filename).suffix.lower() or ".bin"

    try:
        from app.services.ingestion.ingestion_service import IngestionService
        from app.models.schemas import IngestRequest
        from app.services.embedding import get_embedding_service

        ingest_request = IngestRequest(
            source=filename,           # dérivé du nom de fichier
            doc_type=doc_type or None,
            institution=institution or None,
            jurisdiction=jurisdiction or None,
            force_ocr=force_ocr,
        )

        ingestion_service = IngestionService()
        ingestion_service.set_embedding_service(get_embedding_service())

        # Écriture dans un fichier temporaire (le service attend un file_path)
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(content)
            tmp_path = tmp.name

        result = await ingestion_service.ingest_document(
            file_path=tmp_path,
            request=ingest_request,
        )

        # Nettoyage du fichier temporaire
        Path(tmp_path).unlink(missing_ok=True)

        response = ingestion_service.to_response(result)
        # to_response() utilise path.name (fichier temp) — on restaure le nom original
        response = response.model_copy(update={"filename": filename})

    except HTTPException:
        raise
    except Exception as exc:
        logger.error("ingest_endpoint_error", trace_id=trace_id, error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur d'ingestion : {exc}")

    latency_ms = (time.perf_counter() - start) * 1000

    await audit.log_ingestion(
        trace_id=trace_id,
        doc_id=response.document_id,
        filename=filename,
        doc_type=doc_type,
        language=response.language_detected.value if response.language_detected else "unknown",
        chunks_count=response.chunks_created,
        ocr_used=response.ocr_used,
        latency_ms=latency_ms,
        status=response.status,
    )

    return response


@router.delete(
    "/ingest/{doc_id}",
    summary="Suppression d'un document du corpus",
)
async def delete_document(doc_id: str, audit=Depends(get_audit_dep)) -> dict:
    """Supprime un document de Milvus, Elasticsearch et PostgreSQL."""
    try:
        from app.storage.milvus_client import delete_by_doc_id
        from app.storage.elasticsearch_client import get_es_client
        from app.storage.postgres_client import get_db_session
        from app.models.db_models import Chunk, Document
        from app.core.config import get_settings
        import sqlalchemy as sa

        # Milvus — appel synchrone
        delete_by_doc_id(doc_id)

        # Elasticsearch — appel asynchrone
        settings = get_settings()
        es = get_es_client()
        await es.delete_by_query(
            index=settings.elasticsearch_index_chunks,
            body={"query": {"term": {"doc_id.keyword": doc_id}}},
        )

        # PostgreSQL — suppression en cascade (chunks + document)
        async with get_db_session() as db_session:
            await db_session.execute(sa.delete(Chunk).where(Chunk.doc_id == doc_id))
            await db_session.execute(sa.delete(Document).where(Document.id == doc_id))

        return {"status": "deleted", "doc_id": doc_id}

    except Exception as exc:
        logger.error("delete_document_error", doc_id=doc_id, error=str(exc))
        raise HTTPException(status_code=500, detail=f"Erreur de suppression : {exc}")
