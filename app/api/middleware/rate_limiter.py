"""
GOV-AI 2.0 — Middleware de rate limiting (Contrainte C8).
Limite : 60 req/min et 1000 req/h par IP (configurable).
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class RateLimiterMiddleware(BaseHTTPMiddleware):
    """
    Rate limiter par IP — fenêtre glissante.

    Limites (depuis config) :
      - rate_limit_per_minute (défaut: 60)
      - rate_limit_per_hour (défaut: 1000)
    """

    def __init__(self, app: Any) -> None:
        super().__init__(app)
        settings = get_settings()
        self.limit_per_minute = settings.rate_limit_per_minute
        self.limit_per_hour = settings.rate_limit_per_hour
        self._minute_windows: defaultdict[str, deque] = defaultdict(deque)
        self._hour_windows: defaultdict[str, deque] = defaultdict(deque)

    async def dispatch(self, request: Request, call_next) -> Response:
        # Exclure les endpoints de santé
        if request.url.path in ("/health", "/api/v1/health", "/docs", "/openapi.json"):
            return await call_next(request)

        client_ip = request.client.host if request.client else "0.0.0.0"
        now = time.monotonic()

        # Vérification fenêtre 1 minute
        minute_window = self._minute_windows[client_ip]
        while minute_window and minute_window[0] < now - 60:
            minute_window.popleft()

        if len(minute_window) >= self.limit_per_minute:
            logger.warning("rate_limit_exceeded", ip=client_ip, window="minute")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Limite dépassée : {self.limit_per_minute} requêtes/minute.",
                    "retry_after": 60,
                },
                headers={"Retry-After": "60"},
            )

        # Vérification fenêtre 1 heure
        hour_window = self._hour_windows[client_ip]
        while hour_window and hour_window[0] < now - 3600:
            hour_window.popleft()

        if len(hour_window) >= self.limit_per_hour:
            logger.warning("rate_limit_exceeded", ip=client_ip, window="hour")
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": f"Limite dépassée : {self.limit_per_hour} requêtes/heure.",
                    "retry_after": 3600,
                },
                headers={"Retry-After": "3600"},
            )

        minute_window.append(now)
        hour_window.append(now)

        return await call_next(request)
