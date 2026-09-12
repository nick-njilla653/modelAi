"""
GOV-AI 2.0 — Route de requête principale (/api/v1/query).
Implémente le pipeline complet : orchestrateur cognitif → réponse avec citations.
"""
from __future__ import annotations

import time
import uuid
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from app.core.dependencies import get_orchestrator_dep, get_audit_dep
from app.core.logging import get_logger
from app.models.schemas import QueryRequest, QueryResponse

router = APIRouter(tags=["query"])
logger = get_logger(__name__)


@router.post(
    "/query",
    response_model=QueryResponse,
    summary="Requête juridique / administrative",
    description=(
        "Soumet une question à GOV-AI 2.0. "
        "Retourne une réponse ancrée sur le corpus camerounais avec citations obligatoires."
    ),
)
async def query(
    request: QueryRequest,
    http_request: Request,
    orchestrator=Depends(get_orchestrator_dep),
    audit=Depends(get_audit_dep),
) -> QueryResponse:
    """
    Pipeline complet :
    PERCEVOIR → COMPRENDRE → DÉLIBÉRER → AGIR (RAG) → GÉNÉRER → VÉRIFIER → ADAPTER
    """
    trace_id = http_request.headers.get("X-Trace-ID", str(uuid.uuid4()))
    start = time.perf_counter()

    try:
        response = await orchestrator.process(request)
    except Exception as exc:
        logger.error("query_endpoint_error", trace_id=trace_id, error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur interne : {exc}")

    latency_ms = (time.perf_counter() - start) * 1000

    # Audit
    await audit.log_query(
        trace_id=trace_id,
        session_id=str(request.session_id),
        query=request.query,
        language=response.language_detected.value if response.language_detected else "unknown",
        profile=request.profile.value if request.profile else "citizen",
        intent="unknown",  # enrichi par orchestrateur en Sprint 2
        retrieved_count=len(response.retrieved_chunks),
        confidence_score=response.uncertainty_score,
        latency_ms=latency_ms,
        citations_count=len(response.citations),
        safety_flags=[f.value for f in (response.safety_flags or [])],
        model_used=response.model_used or "",
        escalation=any(
            f.value == "ESCALATION_RECOMMENDED"
            for f in (response.safety_flags or [])
        ),
    )

    return response


@router.post(
    "/query/stream",
    summary="Requête en streaming (SSE)",
    description=(
        "Diffuse la réponse au fil de l'eau via le cycle cognitif complet. "
        "Émet des événements JSON typés : meta (intention, plan), token "
        "(fragments de texte), done (citations résolues, drapeaux, latence)."
    ),
)
async def query_stream(
    request: QueryRequest,
    orchestrator=Depends(get_orchestrator_dep),
) -> StreamingResponse:
    """
    Streaming SSE passant par l'orchestrateur.

    À 2 tokens par seconde, une réponse demande plusieurs minutes : le streaming
    n'est pas un confort mais la condition d'utilisabilité de l'interface.
    Les citations n'arrivent qu'à la fin — elles se déduisent des références
    réellement écrites, donc du texte complet.
    """
    import json as _json

    # Séparateur d'événement SSE : deux sauts de ligne closent un message.
    sep = "\n\n"

    async def _event_generator() -> AsyncGenerator[str, None]:
        try:
            async for event in orchestrator.process_stream(request):
                payload = _json.dumps(event, ensure_ascii=False, default=str)
                yield f"data: {payload}{sep}"
        except Exception as exc:
            logger.error("stream_error", error=str(exc), exc_info=True)
            payload = _json.dumps({"type": "error", "message": str(exc)})
            yield f"data: {payload}{sep}"
        finally:
            yield f"data: [DONE]{sep}"

    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Empêche la mise en tampon par un proxy : sans cela le flux
            # n'arrive qu'une fois la génération terminée.
            "X-Accel-Buffering": "no",
        },
    )
