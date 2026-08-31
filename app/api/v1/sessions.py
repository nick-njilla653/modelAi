"""
GOV-AI 2.0 — Routes de conversation (/api/v1/sessions, /api/v1/documents).

L'historique de session était persisté en PostgreSQL depuis le Sprint 3 sans
qu'aucune route ne l'expose : impossible de lister les conversations ni d'en
rouvrir une. Ces routes alimentent la barre latérale de l'interface et le
panneau des sources.
"""
from __future__ import annotations

from typing import Any, Optional

import sqlalchemy as sa
from fastapi import APIRouter, File, Form, HTTPException, Query, Response, UploadFile, status

from app.core.logging import get_logger
from app.models.db_models import QueryLog, Session as DBSession
from app.models.schemas import (
    CorpusDocument,
    SessionDetail,
    SessionSummary,
    SessionTurn,
)
from app.storage.postgres_client import get_db_session

router = APIRouter(tags=["sessions"])
logger = get_logger(__name__)

# Longueur du titre dérivé de la première question, à la façon d'un fil de
# discussion : assez long pour être reconnaissable dans une liste, assez court
# pour tenir sur une ligne de barre latérale.
_TITLE_MAX_CHARS = 60


def _derive_title(first_query: Optional[str]) -> str:
    """Titre de conversation dérivé de la première question posée."""
    if not first_query:
        return "Conversation vide"
    cleaned = " ".join(first_query.split())
    if len(cleaned) <= _TITLE_MAX_CHARS:
        return cleaned
    return cleaned[: _TITLE_MAX_CHARS - 1].rstrip() + "…"


@router.get(
    "/sessions",
    response_model=list[SessionSummary],
    summary="Liste des conversations",
)
async def list_sessions(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user_id: Optional[str] = Query(default=None),
) -> list[SessionSummary]:
    """
    Conversations les plus récentes en premier.

    Le titre est dérivé de la première question de chaque conversation : aucune
    colonne dédiée n'est nécessaire, et le titre reste cohérent avec le contenu.
    """
    async with get_db_session() as db:
        first_query = (
            sa.select(
                QueryLog.session_id,
                sa.func.min(QueryLog.created_at).label("started_at"),
                sa.func.count(QueryLog.id).label("turns"),
            )
            .group_by(QueryLog.session_id)
            .subquery()
        )

        stmt = (
            sa.select(DBSession, first_query.c.turns, first_query.c.started_at)
            .join(first_query, first_query.c.session_id == DBSession.id, isouter=True)
            .order_by(DBSession.last_active.desc())
            .limit(limit)
            .offset(offset)
        )
        if user_id:
            stmt = stmt.where(DBSession.user_id == user_id)

        rows = (await db.execute(stmt)).all()

        summaries: list[SessionSummary] = []
        for session, turns, started_at in rows:
            title_row = (await db.execute(
                sa.select(QueryLog.query)
                .where(QueryLog.session_id == session.id)
                .order_by(QueryLog.created_at)
                .limit(1)
            )).scalar_one_or_none()

            # Sans question posée, la conversation se nomme par sa pièce jointe :
            # une liste de « Conversation sans question » n'aide personne à
            # retrouver la sienne.
            if not title_row:
                from app.models.db_models import SessionDocument

                title_row = (await db.execute(
                    sa.select(SessionDocument.source)
                    .where(SessionDocument.session_id == session.id)
                    .order_by(SessionDocument.created_at)
                    .limit(1)
                )).scalar_one_or_none()

            summaries.append(SessionSummary(
                session_id=session.id,
                title=_derive_title(title_row),
                language=session.language,
                profile=session.profile_type,
                turns=turns or 0,
                created_at=session.created_at,
                last_active=session.last_active,
            ))

    return summaries


@router.get(
    "/sessions/{session_id}",
    response_model=SessionDetail,
    summary="Transcription complète d'une conversation",
)
async def get_session(session_id: str) -> SessionDetail:
    """
    Restitue les tours d'une conversation : question, réponse complète,
    citations et drapeaux de sûreté, dans l'ordre chronologique.
    """
    async with get_db_session() as db:
        session = (await db.execute(
            sa.select(DBSession).where(DBSession.id == session_id)
        )).scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {session_id} introuvable.",
            )

        logs = (await db.execute(
            sa.select(QueryLog)
            .where(QueryLog.session_id == session_id)
            .order_by(QueryLog.created_at)
        )).scalars().all()

    turns: list[SessionTurn] = []
    for log in logs:
        response: dict[str, Any] = log.response_json or {}
        turns.append(SessionTurn(
            turn_id=log.id,
            query=log.query,
            # Les journaux antérieurs à cette route ne conservaient qu'un aperçu
            # de 200 caractères ; on le sert faute de mieux plutôt que du vide.
            answer=response.get("answer") or response.get("answer_preview", ""),
            answer_truncated="answer" not in response,
            citations=log.citations_json or [],
            safety_flags=log.safety_flags or [],
            intent=log.intent,
            confidence=log.score_conf,
            latency_ms=log.latency_ms,
            model_used=log.model_used,
            created_at=log.created_at,
        ))

    return SessionDetail(
        session_id=session.id,
        title=_derive_title(turns[0].query if turns else None),
        language=session.language,
        profile=session.profile_type,
        created_at=session.created_at,
        last_active=session.last_active,
        turns=turns,
    )


@router.delete(
    "/sessions/{session_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    # Un 204 ne peut pas porter de corps : la classe de réponse JSON par
    # défaut de FastAPI en produirait un et refuserait d'enregistrer la route.
    response_class=Response,
    summary="Supprime une conversation et ses tours",
)
async def delete_session(session_id: str) -> Response:
    """Supprime la conversation. Les QueryLog liés suivent par cascade."""
    async with get_db_session() as db:
        session = (await db.execute(
            sa.select(DBSession).where(DBSession.id == session_id)
        )).scalar_one_or_none()

        if session is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Conversation {session_id} introuvable.",
            )

        await db.delete(session)

    logger.info("session_deleted", session_id=session_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get(
    "/documents",
    response_model=list[CorpusDocument],
    summary="Documents du corpus",
)
async def list_corpus_documents() -> list[CorpusDocument]:
    """
    Documents indexés et leur volumétrie.

    Alimente le panneau des sources : un utilisateur qui ignore l'étendue du
    corpus pose des questions auxquelles il est impossible de répondre.
    """
    from app.services.retrieval.corpus_outline import list_documents

    documents = await list_documents()
    return [
        CorpusDocument(
            doc_id=doc["doc_id"],
            source=doc["source"],
            language=doc["language"],
            doc_type=doc["doc_type"] or None,
            chunks=doc["chunks"],
        )
        for doc in documents
    ]


# ── Documents attachés à une conversation ─────────────────────────────────────
# Distincts de /ingest, qui nourrit le corpus global. Ceux-ci ne quittent pas
# leur conversation : aucun index partagé, suppression en cascade avec elle.


@router.post(
    "/sessions/{session_id}/documents",
    status_code=status.HTTP_201_CREATED,
    summary="Attache un document à une conversation",
)
async def attach_session_document(
    session_id: str,
    file: UploadFile = File(..., description="Document (PDF, TXT, MD)"),
    source: str = Form("", description="Libellé cité ; par défaut le nom du fichier"),
    force_ocr: bool = Form(False),
) -> dict:
    """
    Le document est lu, découpé et vectorisé pour cette conversation seule.

    Contrairement à l'ingestion du corpus, rien n'est écrit dans Milvus ni dans
    Elasticsearch : la pièce jointe n'existe que pour cette conversation.
    """
    import tempfile
    from pathlib import Path as _Path

    from app.services.audit.security_filters import validate_filename
    from app.services.ingestion.session_documents import get_session_document_service

    filename = file.filename or "document"
    is_valid, err_msg = validate_filename(filename)
    if not is_valid:
        raise HTTPException(status_code=400, detail=err_msg)

    content = await file.read()
    max_bytes = 25 * 1024 * 1024
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                "Pièce jointe trop volumineuse (25 Mo maximum). Pour un document "
                "de référence destiné à toutes les conversations, utilisez "
                "l'ingestion du corpus."
            ),
        )

    suffix = _Path(filename).suffix.lower() or ".bin"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = await get_session_document_service().attach(
            session_id=session_id,
            file_path=tmp_path,
            filename=filename,
            source=source or None,
            force_ocr=force_ocr,
        )
    except Exception as exc:
        logger.error("session_document_attach_failed", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Lecture du document impossible : {exc}")
    finally:
        _Path(tmp_path).unlink(missing_ok=True)

    return {
        "document_id": result.document_id,
        "filename": result.filename,
        "source": result.source,
        "language": result.language,
        "doc_type": result.doc_type,
        "chunk_count": result.chunk_count,
        "ocr_used": result.ocr_used,
        "warnings": result.warnings,
        "latency_ms": result.latency_ms,
    }


@router.get(
    "/sessions/{session_id}/documents",
    summary="Documents attachés à une conversation",
)
async def list_session_documents(session_id: str) -> list[dict]:
    from app.services.ingestion.session_documents import get_session_document_service

    return await get_session_document_service().list_documents(session_id)


@router.delete(
    "/sessions/{session_id}/documents/{document_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    response_class=Response,
    summary="Détache un document d'une conversation",
)
async def delete_session_document(session_id: str, document_id: str) -> Response:
    from app.services.ingestion.session_documents import get_session_document_service

    deleted = await get_session_document_service().delete_document(session_id, document_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Document introuvable dans cette conversation.",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
