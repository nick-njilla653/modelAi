"""
GOV-AI 2.0 — Middleware d'audit : injection du trace_id dans chaque requête.
"""
from __future__ import annotations

import time
import uuid

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

from app.core.logging import get_logger, set_trace_id

logger = get_logger(__name__)


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware qui :
    1. Génère ou propage un trace_id (X-Trace-ID header)
    2. Logue chaque requête HTTP entrante et sortante
    3. Injecte le trace_id dans le contexte de logging structlog
    """

    async def dispatch(self, request: Request, call_next) -> Response:
        trace_id = request.headers.get("X-Trace-ID") or str(uuid.uuid4())
        set_trace_id(trace_id)

        start = time.perf_counter()

        logger.info(
            "http_request",
            trace_id=trace_id,
            method=request.method,
            path=request.url.path,
            client=request.client.host if request.client else "unknown",
        )

        try:
            response = await call_next(request)
        except Exception as exc:
            logger.error(
                "http_request_error",
                trace_id=trace_id,
                method=request.method,
                path=request.url.path,
                error=str(exc),
            )
            raise

        latency_ms = (time.perf_counter() - start) * 1000
        response.headers["X-Trace-ID"] = trace_id
        response.headers["X-Latency-Ms"] = str(round(latency_ms, 2))

        logger.info(
            "http_response",
            trace_id=trace_id,
            status_code=response.status_code,
            latency_ms=round(latency_ms, 2),
        )

        return response
