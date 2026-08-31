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
from typing import Any, Awaitable, Callable, Optional

from app.core.config import get_settings
from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.models.domain import IngestionStatus, Language
from app.models.schemas import IngestRequest, IngestResponse
from app.services.ingestion.chunker import (
    ChunkResult,
    chunk_document,
    consolidate_article_chunks,
)
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
        progress_cb: Optional[Callable[[str, int, str], Awaitable[None]]] = None,
    ) -> IngestionResult:
        """
        Ingère un document complet.

        Args:
            file_path: Chemin vers le fichier (PDF ou texte)
            request: Métadonnées fournies par l'utilisateur
            document_id: ID forcé (sinon auto-généré)
            progress_cb: Rappel (étape, pourcentage, détail) pour suivre
                l'avancement. L'OCR d'un document scanné dure plusieurs minutes :
                sans retour d'étape, l'appelant ne peut pas distinguer un
                traitement long d'un blocage.

        Returns:
            IngestionResult avec les chunks créés et les métadonnées
        """
        start_time = time.perf_counter()
        doc_id = document_id or str(uuid.uuid4())
        path = Path(file_path)
        warnings: list[str] = []

        async def report(stage: str, percent: int, detail: str = "") -> None:
            if progress_cb is not None:
                try:
                    await progress_cb(stage, percent, detail)
                except Exception as exc:  # le suivi ne doit jamais casser l'ingestion
                    logger.warning("progress_callback_failed", error=str(exc))

        logger.info(
            "ingestion_started",
            doc_id=doc_id,
            filename=path.name,
            source=request.source,
        )

        try:
            # ── Étape 1 : Extraction du texte ────────────────────────────────
            await report("extraction", 5, "Lecture du document")
            raw_text, pages_content, ocr_used = await self._extract_text(
                path, request.force_ocr, warnings, report
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
            await report("chunking", 55, "Découpage en articles")
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

            # Recolle les articles coupés par une frontière de page et propage la
            # référence citable aux chunks de continuation (chunking page par page).
            all_chunks = consolidate_article_chunks(
                all_chunks, language=metadata.language.value
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
            await report(
                "indexation", 70,
                f"Calcul des vecteurs et indexation de {len(all_chunks)} passages",
            )
            await self._index_chunks(doc_id, all_chunks, metadata, request)

            # ── Étape 5 : Persistance PostgreSQL ─────────────────────────────
            await report("persistance", 90, "Enregistrement des métadonnées")
            await self._persist_to_db(doc_id, path, metadata, all_chunks, ocr_used, request)

            # ── Étape 6 : Graphe de connaissances ────────────────────────────
            # Réservé au corpus global. Les pièces jointes de conversation
            # passent par `extract_and_chunk` et n'atteignent jamais ce point :
            # le graphe est partagé par toutes les conversations.
            await report("graphe", 96, "Construction du graphe de connaissances")
            await self._build_graph(doc_id, metadata, all_chunks, warnings)

            await report("termine", 100, f"{len(all_chunks)} passages indexés")

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

    async def _build_graph(
        self,
        doc_id: str,
        metadata: DocumentMetadata,
        chunks: list[ChunkResult],
        warnings: list[str],
    ) -> None:
        """
        Alimente le graphe de connaissances à partir des articles du document.

        Un échec ici ne doit pas faire échouer l'ingestion : le document reste
        interrogeable par le corpus vectoriel et lexical même sans graphe. Mais
        il est signalé, faute de quoi le graphe resterait vide en silence — ce
        qui s'est produit jusqu'ici.
        """
        try:
            import sqlalchemy as sa

            from app.models.db_models import Document
            from app.services.knowledge_graph.graph_builder import build_document_graph
            from app.storage.postgres_client import get_db_session

            # Les autres documents du corpus, pour résoudre un renvoi sortant
            # vers leur véritable disposition plutôt que vers un nœud vide.
            async with get_db_session() as db:
                rows = (await db.execute(sa.select(Document.source, Document.id))).all()
            corpus_titles = {source: identifier for source, identifier in rows}

            stats = await build_document_graph(
                doc_id=doc_id,
                title=metadata.source,
                doc_type=metadata.doc_type.value if metadata.doc_type else None,
                jurisdiction=metadata.jurisdiction,
                chunks=chunks,
                corpus_titles=corpus_titles,
                institution=metadata.institution,
            )
            logger.info("ingestion_graph_built", doc_id=doc_id, **{
                k: v for k, v in stats.to_dict().items() if k != "external_texts"
            })
        except Exception as exc:
            warnings.append(
                f"Le graphe de connaissances n'a pas pu être alimenté : {exc}"
            )
            logger.warning("ingestion_graph_failed", doc_id=doc_id, error=str(exc))

    async def extract_and_chunk(
        self,
        file_path: str | Path,
        request: IngestRequest,
    ) -> tuple[list[ChunkResult], DocumentMetadata, bool, list[str]]:
        """
        Extrait et découpe un document, sans rien persister ni indexer.

        Sépare le traitement de sa destination. `ingest_document` s'en sert pour
        alimenter le corpus global ; les pièces jointes d'une conversation s'en
        servent pour rester chez elles. Sans cette séparation, attacher un
        document à une conversation le versait au corpus de tous — la
        persistance PostgreSQL s'exécutant quoi qu'il arrive.

        Returns:
            (passages, métadonnées, OCR utilisé, avertissements)
        """
        path = Path(file_path)
        warnings: list[str] = []

        raw_text, pages_content, ocr_used = await self._extract_text(
            path, request.force_ocr, warnings
        )
        if not raw_text.strip():
            raise IngestionError(f"Impossible d'extraire du texte de {path.name}")

        metadata = extract_metadata(
            text=raw_text[:5000],
            filename=path.name,
            source=request.source,
            provided_metadata=request.model_dump(exclude_none=True),
        )
        metadata.page_count = len(pages_content) if pages_content else 1

        all_chunks: list[ChunkResult] = []
        global_index = 0

        if pages_content:
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
                    chunk.chunk_index = global_index
                    global_index += 1
                all_chunks.extend(page_chunks)
        else:
            cleaned = clean_extracted_text(raw_text, metadata.language.value)
            all_chunks = chunk_document(
                text=cleaned,
                strategy=request.chunking_strategy,
                max_tokens=self.settings.chunk_size,
                overlap_tokens=self.settings.chunk_overlap,
            )

        all_chunks = consolidate_article_chunks(
            all_chunks, language=metadata.language.value
        )
        if not all_chunks:
            raise IngestionError(f"Aucun chunk créé pour {path.name}")

        return all_chunks, metadata, ocr_used, warnings

    async def _extract_text(
        self,
        path: Path,
        force_ocr: bool,
        warnings: list[str],
        report: Optional[Callable[..., Awaitable[None]]] = None,
    ) -> tuple[str, list[tuple[int, str]], bool]:
        """
        Extrait le texte brut. Retourne (full_text, pages_list, ocr_used).

        L'OCR est la phase la plus longue du pipeline — plusieurs secondes par
        page. Elle rend donc l'avancement page par page.
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
                total_pages = len(page_images)
                for index, (page_num, img_bytes) in enumerate(page_images, start=1):
                    ocr_text = ocr_image_bytes(
                        img_bytes, language=self.settings.tesseract_lang
                    )
                    pages_content.append((page_num, ocr_text))
                    if report is not None and total_pages:
                        # L'OCR occupe la tranche 5–50 % de la progression globale.
                        percent = 5 + int(45 * index / total_pages)
                        await report(
                            "ocr", percent, f"Reconnaissance de texte : page {index}/{total_pages}"
                        )
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
        article_refs = [(c.article_ref or "")[:64] for c in chunks]

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
            article_refs=article_refs,
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
                "article_ref": article_refs[i],
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
                    article_ref=chunk.article_ref,
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
