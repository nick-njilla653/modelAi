"""
GOV-AI 2.0 — Documents attachés à une conversation.

Deux ingestions coexistent, et leur séparation est délibérée :

- **Le corpus global** (page « Corpus ») nourrit tout le système : PostgreSQL,
  Milvus, Elasticsearch. Ce qui y entre est visible de toutes les conversations
  et y reste jusqu'à suppression explicite.

- **La pièce jointe d'une conversation** (bouton « + » du composeur) ne quitte
  jamais celle-ci. Elle n'est écrite dans aucun index partagé : ses passages et
  leurs vecteurs vivent dans PostgreSQL, et la comparaison se fait en mémoire.

Ce choix rend l'isolement structurel. L'alternative — écrire dans Milvus avec un
champ `session_id` filtré à la lecture — ferait dépendre la confidentialité d'un
filtre à ne jamais oublier ; un seul chemin de récupération négligent suffirait
à faire apparaître le contrat d'un utilisateur dans la réponse faite à un autre.
Ici, la fuite est impossible parce que la donnée n'est pas là.

Le volume le permet : une pièce jointe fait quelques dizaines de passages, et un
produit scalaire sur autant de vecteurs est instantané.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from app.core.logging import get_logger
from app.models.domain import Language
from app.models.schemas import RetrievedChunk

logger = get_logger(__name__)


@dataclass
class SessionDocumentResult:
    """Ce qu'a produit l'attachement d'un document à une conversation."""
    document_id: str
    filename: str
    source: str
    language: str
    doc_type: Optional[str]
    chunk_count: int
    ocr_used: bool
    warnings: list[str]
    latency_ms: float


class SessionDocumentService:
    """Ingère, retrouve et supprime les documents propres à une conversation."""

    def __init__(self, embedding_service: Any = None) -> None:
        self._embedding_service = embedding_service

    def set_embedding_service(self, svc: Any) -> None:
        self._embedding_service = svc

    # ── Ingestion ─────────────────────────────────────────────────────────────

    async def attach(
        self,
        session_id: str,
        file_path: str | Path,
        filename: str,
        source: Optional[str] = None,
        force_ocr: bool = False,
    ) -> SessionDocumentResult:
        """
        Attache un document à une conversation.

        Le pipeline d'extraction et de découpage est celui du corpus — même
        gestion du PDF, de l'OCR et des articles coupés par une frontière de
        page. Seule la destination change : rien n'est indexé globalement.
        """
        import sqlalchemy as sa

        from app.models.db_models import (
            Session as DBSession,
            SessionDocument,
            SessionDocumentChunk,
        )
        from app.models.schemas import IngestRequest
        from app.services.ingestion.ingestion_service import IngestionService
        from app.storage.postgres_client import get_db_session

        started = time.perf_counter()
        label = (source or filename).strip() or filename

        # `extract_and_chunk` traite le document sans rien persister ni indexer.
        # Passer par `ingest_document` versait la pièce jointe au corpus global :
        # sa persistance PostgreSQL s'exécute quoi qu'il arrive, et priver le
        # service d'embedding ne bloquait que Milvus et Elasticsearch.
        extractor = IngestionService()
        request = IngestRequest(source=label, force_ocr=force_ocr)

        chunks, metadata, ocr_used, extraction_warnings = await extractor.extract_and_chunk(
            file_path=str(file_path),
            request=request,
        )
        contents = [c.content for c in chunks]

        embeddings: list[Optional[list[float]]] = [None] * len(chunks)
        if self._embedding_service is not None and contents:
            try:
                embeddings = await self._embedding_service.embed_texts(contents)
            except Exception as exc:
                # Sans vecteurs, la recherche retombe sur les mots-clés : le
                # document reste exploitable, moins finement.
                logger.warning("session_document_embedding_failed", error=str(exc))
                embeddings = [None] * len(chunks)

        language = metadata.language.value if metadata else "unknown"
        doc_type = metadata.doc_type.value if (metadata and metadata.doc_type) else None

        async with get_db_session() as db:
            # La conversation doit exister : la clé étrangère en dépend, et son
            # absence signifierait qu'on attache à une conversation fantôme.
            existing = (await db.execute(
                sa.select(DBSession).where(DBSession.id == session_id)
            )).scalar_one_or_none()
            if existing is None:
                db.add(DBSession(id=session_id, language=language))

            document = SessionDocument(
                session_id=session_id,
                filename=filename,
                source=label,
                language=language,
                doc_type=doc_type,
                page_count=metadata.page_count if metadata else None,
                ocr_used=ocr_used,
                chunk_count=len(chunks),
            )
            db.add(document)
            await db.flush()

            for chunk, vector in zip(chunks, embeddings):
                db.add(SessionDocumentChunk(
                    document_id=document.id,
                    session_id=session_id,
                    content=chunk.content,
                    chunk_index=chunk.chunk_index,
                    page=chunk.page,
                    article_ref=chunk.article_ref,
                    token_count=chunk.token_count,
                    embedding=vector,
                ))

            document_id = document.id

        latency_ms = (time.perf_counter() - started) * 1000
        logger.info(
            "session_document_attached",
            session_id=session_id,
            document_id=document_id,
            chunks=len(chunks),
            ocr_used=ocr_used,
        )

        return SessionDocumentResult(
            document_id=document_id,
            filename=filename,
            source=label,
            language=language,
            doc_type=doc_type,
            chunk_count=len(chunks),
            ocr_used=ocr_used,
            warnings=list(extraction_warnings),
            latency_ms=round(latency_ms, 2),
        )

    # ── Recherche ─────────────────────────────────────────────────────────────

    async def search(
        self,
        session_id: str,
        query: str,
        top_k: int = 4,
    ) -> list[RetrievedChunk]:
        """
        Cherche dans les documents de la conversation, et nulle part ailleurs.

        Comparaison vectorielle si les vecteurs existent, sinon repli sur un
        recouvrement de mots : une pièce jointe doit rester exploitable même
        lorsque le service d'embedding est indisponible.
        """
        import sqlalchemy as sa

        from app.models.db_models import SessionDocument, SessionDocumentChunk
        from app.storage.postgres_client import get_db_session

        async with get_db_session() as db:
            rows = (await db.execute(
                sa.select(SessionDocumentChunk, SessionDocument)
                .join(SessionDocument, SessionDocument.id == SessionDocumentChunk.document_id)
                .where(SessionDocumentChunk.session_id == session_id)
            )).all()

        if not rows:
            return []

        query_vector: Optional[list[float]] = None
        if self._embedding_service is not None:
            try:
                query_vector = (await self._embedding_service.embed_texts([query]))[0]
            except Exception as exc:
                logger.warning("session_document_query_embedding_failed", error=str(exc))

        scored: list[tuple[float, Any, Any]] = []
        query_words = {w for w in _normalize_words(query) if len(w) > 2}

        for chunk, document in rows:
            if query_vector is not None and chunk.embedding:
                score = _cosine(query_vector, chunk.embedding)
            else:
                content_words = set(_normalize_words(chunk.content))
                overlap = len(query_words & content_words)
                score = overlap / max(len(query_words), 1)
            scored.append((score, chunk, document))

        scored.sort(key=lambda item: item[0], reverse=True)

        results: list[RetrievedChunk] = []
        for score, chunk, document in scored[:top_k]:
            try:
                language = Language(document.language)
            except ValueError:
                language = Language.UNKNOWN
            results.append(RetrievedChunk(
                chunk_id=chunk.id,
                doc_id=document.id,
                content=chunk.content,
                source=document.source,
                language=language,
                page=chunk.page,
                chunk_index=chunk.chunk_index,
                dense_score=round(score, 4),
                final_score=round(score, 4),
                metadata={
                    "doc_type": document.doc_type or "",
                    "article_ref": chunk.article_ref or "",
                    # Marque l'origine : la réponse doit pouvoir distinguer une
                    # pièce jointe d'un texte du corpus officiel.
                    "scope": "conversation",
                },
            ))
        return results

    # ── Inventaire et suppression ────────────────────────────────────────────

    async def list_documents(self, session_id: str) -> list[dict]:
        import sqlalchemy as sa

        from app.models.db_models import SessionDocument
        from app.storage.postgres_client import get_db_session

        async with get_db_session() as db:
            documents = (await db.execute(
                sa.select(SessionDocument)
                .where(SessionDocument.session_id == session_id)
                .order_by(SessionDocument.created_at)
            )).scalars().all()

        return [
            {
                "document_id": d.id,
                "filename": d.filename,
                "source": d.source,
                "language": d.language,
                "doc_type": d.doc_type,
                "chunk_count": d.chunk_count,
                "ocr_used": d.ocr_used,
                "created_at": d.created_at,
            }
            for d in documents
        ]

    async def delete_document(self, session_id: str, document_id: str) -> bool:
        """Supprime une pièce jointe. Les passages suivent par cascade."""
        import sqlalchemy as sa

        from app.models.db_models import SessionDocument
        from app.storage.postgres_client import get_db_session

        async with get_db_session() as db:
            document = (await db.execute(
                sa.select(SessionDocument).where(
                    SessionDocument.id == document_id,
                    SessionDocument.session_id == session_id,
                )
            )).scalar_one_or_none()
            if document is None:
                return False
            await db.delete(document)

        logger.info("session_document_deleted", session_id=session_id, document_id=document_id)
        return True


def _normalize_words(text: str) -> list[str]:
    import re
    import unicodedata

    decomposed = unicodedata.normalize("NFD", text.lower())
    stripped = "".join(c for c in decomposed if unicodedata.category(c) != "Mn")
    return re.findall(r"[a-z0-9]+", stripped)


def _cosine(a: list[float], b: list[float]) -> float:
    """Similarité cosinus, bornée à [0, 1] comme les scores du corpus."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return max(0.0, min(1.0, dot / (norm_a * norm_b)))


_service: Optional[SessionDocumentService] = None


def get_session_document_service() -> SessionDocumentService:
    global _service
    if _service is None:
        from app.services.embedding import get_embedding_service

        _service = SessionDocumentService(embedding_service=get_embedding_service())
    return _service
