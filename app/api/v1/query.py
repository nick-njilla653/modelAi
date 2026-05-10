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
    summary="Requête en streaming (token par token)",
    description="Génère la réponse en streaming SSE. Pas de citations structurées.",
)
async def query_stream(
    request: QueryRequest,
    orchestrator=Depends(get_orchestrator_dep),
) -> StreamingResponse:
    """
    Streaming token par token via Ollama.
    Format : text/event-stream (SSE).
    """
    from app.services.generation.llm_answer_service import LLMAnswerService
    from app.services.retrieval.hybrid_retrieval_service import HybridRetrievalService
    from app.models.domain import Language, UserProfile

    async def _token_generator() -> AsyncGenerator[str, None]:
        import json as _json
        try:
            from app.services.embedding import get_embedding_service
            retrieval_svc = HybridRetrievalService()
            retrieval_svc.set_embedding_service(get_embedding_service())
            language = request.language or Language.FR
            chunks = await retrieval_svc.retrieve(
                query=request.query,
                language=language,
                top_k=request.top_k,
            )

            generation_svc = LLMAnswerService()
            async for token in generation_svc.generate_stream(
                query=request.query,
                retrieved_chunks=chunks,
                language=language,
                profile=request.profile or UserProfile.CITIZEN,
                session_context=request.session_context or "",
            ):
                # JSON-encode le token pour préserver \n et tout caractère spécial
                yield f"data: {_json.dumps(token)}\n\n"

            yield "data: [DONE]\n\n"
        except Exception as exc:
            logger.error("stream_error", error=str(exc))
            yield f"data: [ERROR] {exc}\n\n"

    return StreamingResponse(
        _token_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
