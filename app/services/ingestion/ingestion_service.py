"""
GOV-AI 2.0 — Service d'ingestion documentaire (coordinateur).
Pipeline complet : extraction → nettoyage → chunking → embedding → indexation.

Équation 4.1 du mémoire :
  d_brut → f_extract → f_clean → f_chunk → {c_1,...,c_m} → f_embed → {(v_1,m_1),...}
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.models.domain import IngestionStatus, Language
from app.models.schemas import IngestRequest, IngestResponse
from app.services.ingestion.chunker import ChunkResult, chunk_document
from app.services.ingestion.metadata_extractor import DocumentMetadata, extract_metadata
from app.services.ingestion.ocr_processor import ocr_image_bytes
from app.services.ingestion.pdf_extractor import ExtractedDocument, extract_pdf_text, extract_pdf_as_images
from app.services.ingestion.text_cleaner import clean_extracted_text, is_content_sufficient

logger = get_logger(__name__)


@dataclass
class IngestionResult:
    document_id: str
    filename: str
    status: IngestionStatus
    chunks: list[ChunkResult] = field(default_factory=list)
    metadata: Optional[DocumentMetadata] = None
    ocr_used: bool = False
    latency_ms: float = 0.0
    warnings: list[str] = field(default_factory=list)
    error: Optional[str] = None


class IngestionService:
    """
    Coordinateur du pipeline d'ingestion documentaire.
    Orchestration : PDF → OCR si nécessaire → nettoyage → chunking → stockage.
    """

    def __init__(
        self,
        embedding_service: Any = None,  # Injecté après init (évite import circulaire)
    ) -> None:
        self.settings = get_settings()
        self._embedding_service = embedding_service

    def set_embedding_service(self, svc: Any) -> None:
        self._embedding_service = svc

    async def ingest_document(
        self,
        file_path: str | Path,
        request: IngestRequest,
        document_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingère un document complet.

        Args:
            file_path: Chemin vers le fichier (PDF ou texte)
            request: Métadonnées fournies par l'utilisateur
            document_id: ID forcé (sinon auto-généré)

        Returns:
            IngestionResult avec les chunks créés et les métadonnées
        """
        start_time = time.perf_counter()
        doc_id = document_id or str(uuid.uuid4())
        path = Path(file_path)
        warnings: list[str] = []

        logger.info(
            "ingestion_started",
            doc_id=doc_id,
            filename=path.name,
            source=request.source,
        )

        try:
            # ── Étape 1 : Extraction du texte ────────────────────────────────
            raw_text, pages_content, ocr_used = await self._extract_text(
                path, request.force_ocr, warnings
            )

            if not raw_text.strip():
                raise IngestionError(f"Impossible d'extraire du texte de {path.name}")

            # ── Étape 2 : Métadonnées ────────────────────────────────────────
            metadata = extract_metadata(
                text=raw_text[:5000],  # Utiliser le début pour la détection
                filename=path.name,
                source=request.source,
                provided_metadata=request.model_dump(exclude_none=True),
            )
            metadata.page_count = len(pages_content) if pages_content else 1

            # ── Étape 3 : Nettoyage page par page + chunking ─────────────────
            all_chunks: list[ChunkResult] = []
            global_chunk_index = 0

            if pages_content:
                # Traitement page par page
                for page_num, page_text in pages_content:
                    cleaned = clean_extracted_text(page_text, metadata.language.value)
                    if not is_content_sufficient(cleaned):
                        continue
                    page_chunks = chunk_document(
                        text=cleaned,
                        strategy=request.chunking_strategy,
                        max_tokens=self.settings.chunk_size,
                        overlap_tokens=self.settings.chunk_overlap,
                        page=page_num,
                    )
                    for chunk in page_chunks:
                        chunk.chunk_index = global_chunk_index
                        global_chunk_index += 1
                    all_chunks.extend(page_chunks)
            else:
                # Document texte sans pagination
                cleaned = clean_extracted_text(raw_text, metadata.language.value)
                all_chunks = chunk_document(
                    text=cleaned,
                    strategy=request.chunking_strategy,
                    max_tokens=self.settings.chunk_size,
                    overlap_tokens=self.settings.chunk_overlap,
                )

            if not all_chunks:
                raise IngestionError(f"Aucun chunk créé pour {path.name}")

            logger.info(
                "ingestion_chunks_created",
                doc_id=doc_id,
                chunks=len(all_chunks),
                language=metadata.language,
            )

            # ── Étape 4 : Embedding + Indexation ─────────────────────────────
            await self._index_chunks(doc_id, all_chunks, metadata, request)

            # ── Étape 5 : Persistance PostgreSQL ─────────────────────────────
            await self._persist_to_db(doc_id, path, metadata, all_chunks, ocr_used, request)

            latency_ms = (time.perf_counter() - start_time) * 1000
            logger.info(
                "ingestion_completed",
                doc_id=doc_id,
                chunks=len(all_chunks),
                latency_ms=round(latency_ms, 2),
                ocr_used=ocr_used,
            )

            return IngestionResult(
                document_id=doc_id,
                filename=path.name,
                status=IngestionStatus.COMPLETED,
                chunks=all_chunks,
                metadata=metadata,
                ocr_used=ocr_used,
                latency_ms=latency_ms,
                warnings=warnings,
            )

        except IngestionError:
            raise
        except Exception as exc:
            logger.error("ingestion_failed", doc_id=doc_id, error=str(exc), exc_info=True)
            raise IngestionError(f"Erreur d'ingestion : {exc}") from exc

    async def _extract_text(
        self,
        path: Path,
        force_ocr: bool,
        warnings: list[str],
    ) -> tuple[str, list[tuple[int, str]], bool]:
        """
        Extrait le texte brut. Retourne (full_text, pages_list, ocr_used).
        """
        suffix = path.suffix.lower()

        if suffix in (".txt", ".md"):
            text = path.read_text(encoding="utf-8", errors="replace")
            return text, [], False

        if suffix == ".pdf":
            extracted: ExtractedDocument = extract_pdf_text(path, force_ocr)

            pages_content: list[tuple[int, str]] = []
            ocr_used = False

            if extracted.needs_ocr:
                # OCR des pages scannées
                ocr_used = True
                warnings.append(
                    "Certaines pages ont nécessité l'OCR (document scanné détecté)"
                )
                page_images = extract_pdf_as_images(path)
                for page_num, img_bytes in page_images:
                    ocr_text = ocr_image_bytes(
                        img_bytes, language=self.settings.tesseract_lang
                    )
                    pages_content.append((page_num, ocr_text))
            else:
                for page in extracted.pages:
                    pages_content.append((page.page_number, page.text))

            full_text = "\n\n".join(text for _, text in pages_content if text.strip())
            return full_text, pages_content, ocr_used

        # Format non supporté
        raise IngestionError(f"Format de fichier non supporté : {suffix}")

    async def _index_chunks(
        self,
        doc_id: str,
        chunks: list[ChunkResult],
        metadata: DocumentMetadata,
        request: IngestRequest,
    ) -> None:
        """Calcule les embeddings et indexe dans Milvus + Elasticsearch."""
        if self._embedding_service is None:
            logger.warning("no_embedding_service_configured", doc_id=doc_id)
            return

        from app.storage.elasticsearch_client import bulk_index_chunks
        from app.storage.milvus_client import insert_chunks

        contents = [c.content for c in chunks]

        # Embedding par batch
        embeddings = await self._embedding_service.embed_texts(contents)

        # Préparer les données Milvus
        chunk_ids = [c.chunk_id for c in chunks]
        doc_ids = [doc_id] * len(chunks)
        sources = [metadata.source] * len(chunks)
        languages = [metadata.language.value] * len(chunks)
        pages = [c.page or 0 for c in chunks]
        indexes = [c.chunk_index for c in chunks]
        doc_types = [metadata.doc_type.value if metadata.doc_type else ""] * len(chunks)
        institutions = [metadata.institution or ""] * len(chunks)
        jurisdictions = [metadata.jurisdiction or ""] * len(chunks)

        milvus_ids = insert_chunks(
            chunk_ids=chunk_ids,
            doc_ids=doc_ids,
            contents=contents,
            sources=sources,
            languages=languages,
            pages=pages,
            chunk_indexes=indexes,
            doc_types=doc_types,
            institutions=institutions,
            jurisdictions=jurisdictions,
            embeddings=embeddings,
        )

        # Mettre à jour les milvus_id dans les chunks
        for chunk, mid in zip(chunks, milvus_ids):
            chunk.chunk_id = chunk.chunk_id  # Garde l'UUID PostgreSQL

        # Indexation Elasticsearch (BM25)
        es_docs = [
            {
                "chunk_id": chunks[i].chunk_id,
                "doc_id": doc_id,
                "content": contents[i],
                "source": metadata.source,
                "language": metadata.language.value,
                "page": pages[i],
                "chunk_index": indexes[i],
                "doc_type": doc_types[i],
                "institution": institutions[i],
                "jurisdiction": jurisdictions[i],
            }
            for i in range(len(chunks))
        ]
        await bulk_index_chunks(es_docs)

    async def _persist_to_db(
        self,
        doc_id: str,
        path: Path,
        metadata: DocumentMetadata,
        chunks: list[ChunkResult],
        ocr_used: bool,
        request: IngestRequest,
    ) -> None:
        """Persiste le document et ses chunks dans PostgreSQL."""
        from app.models.db_models import Chunk, Document
        from app.storage.postgres_client import get_db_session

        async with get_db_session() as session:
            # Document
            doc = Document(
                id=doc_id,
                filename=request.source,  # nom original, pas le fichier temp
                file_path=None,           # fichier temp supprimé après ingestion
                source=metadata.source,
                language=metadata.language.value,
                doc_type=metadata.doc_type.value if metadata.doc_type else None,
                institution=metadata.institution,
                jurisdiction=metadata.jurisdiction,
                version=metadata.version,
                date_document=metadata.date_document,
                ocr_used=ocr_used,
                status=IngestionStatus.COMPLETED.value,
            )
            session.add(doc)

            # Chunks
            for chunk in chunks:
                db_chunk = Chunk(
                    id=chunk.chunk_id,
                    doc_id=doc_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    page=chunk.page,
                    token_count=chunk.token_count,
                    language=metadata.language.value,
                    chunk_strategy=chunk.strategy,
                )
                session.add(db_chunk)

    def to_response(self, result: IngestionResult) -> IngestResponse:
        """Convertit IngestionResult en IngestResponse API."""
        return IngestResponse(
            document_id=result.document_id,
            filename=result.filename,
            chunks_created=len(result.chunks),
            language_detected=result.metadata.language if result.metadata else Language.UNKNOWN,
            doc_type=result.metadata.doc_type if result.metadata else None,
            ocr_used=result.ocr_used,
            ingestion_latency_ms=result.latency_ms,
            status=result.status.value,
            warnings=result.warnings,
        )
