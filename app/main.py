"""
GOV-AI 2.0 — Application FastAPI principale.
Factory pattern avec lifespan pour l'initialisation et le nettoyage des services.
"""
from __future__ import annotations

import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.responses import JSONResponse

from app.core.config import get_settings
from app.core.exceptions import register_exception_handlers
from app.core.logging import configure_logging, get_logger

settings = get_settings()
configure_logging(log_level=settings.log_level, log_format=settings.log_format)
logger = get_logger(__name__)


# ── Lifespan (startup + shutdown) ─────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Gère le cycle de vie de l'application.
    Remplace les @app.on_event("startup") / @app.on_event("shutdown") dépréciés.
    """
    logger.info("govai2_starting", version="2.0.0-sprint1", env=settings.app_env)

    # ── Startup ──────────────────────────────────────────────────────────────
    # ── Migration PostgreSQL (create_all si tables absentes) ─────────────────
    try:
        from app.storage.postgres_client import get_engine
        from app.models.db_models import Base
        _engine = get_engine()
        async with _engine.begin() as _conn:
            await _conn.run_sync(Base.metadata.create_all)
        logger.info("postgres_schema_ready")
    except Exception as exc:
        logger.warning("postgres_schema_init_failed", error=str(exc))

    try:
        # Connexion Milvus
        from app.storage.milvus_client import connect_milvus, ensure_collection
        connect_milvus()
        ensure_collection()
        logger.info("milvus_connected")
    except Exception as exc:
        logger.warning("milvus_init_failed", error=str(exc))

    try:
        # Pré-chargement de l'orchestrateur cognitif
        from app.services.cognitive_orchestrator import get_orchestrator
        orchestrator = get_orchestrator()
        logger.info("cognitive_orchestrator_ready")
    except Exception as exc:
        logger.error("cognitive_orchestrator_init_failed", error=str(exc))

    # Pré-chargement service embedding (warm-up)
    try:
        from app.services.embedding import get_embedding_service
        embed_svc = get_embedding_service()
        logger.info("embedding_service_ready", provider=settings.embedding_provider)
    except Exception as exc:
        logger.warning("embedding_service_init_failed", error=str(exc))

    # Rétrocompatibilité v1 : init LangChain si activé
    try:
        _init_v1_langchain()
    except Exception as exc:
        logger.warning("v1_langchain_init_skipped", error=str(exc))

    logger.info("govai2_ready", host=settings.app_host, port=settings.app_port)

    yield  # Application en service

    # ── Shutdown ─────────────────────────────────────────────────────────────
    logger.info("govai2_shutting_down")

    try:
        from app.storage.postgres_client import close_postgres
        await close_postgres()
        logger.info("postgres_connection_closed")
    except Exception as exc:
        logger.warning("postgres_close_error", error=str(exc))

    try:
        from app.storage.elasticsearch_client import close_es_client
        await close_es_client()
        logger.info("elasticsearch_connection_closed")
    except Exception as exc:
        logger.warning("elasticsearch_close_error", error=str(exc))

    try:
        from app.services.langchain_init import reset_orchestrator
        reset_orchestrator()
    except Exception:
        pass

    logger.info("govai2_stopped")


def _init_v1_langchain() -> None:
    """Initialise le service LangChain v1 si INIT_LANGCHAIN_ON_STARTUP est True."""
    if not getattr(settings, "INIT_LANGCHAIN_ON_STARTUP", False):
        return
    from app.services.langchain_init import init_langchain
    init_langchain(
        data_path=getattr(settings, "DATA_PATH", "./data"),
        metadata_path=getattr(settings, "METADATA_PATH", "./data/metadata"),
        embedding_service_url=getattr(settings, "EMBEDDING_SERVICE_URL", settings.llm_base_url),
        embedding_model=getattr(settings, "EMBEDDING_MODEL", settings.embedding_model),
        milvus_host=settings.milvus_host,
        milvus_port=settings.milvus_port,
        milvus_collection=settings.milvus_collection_chunks,
        llm_service_url=settings.llm_base_url,
        llm_model=settings.llm_model,
        embedding_dim=settings.embedding_dim,
        milvus_connection_timeout=getattr(settings, "MILVUS_CONNECTION_TIMEOUT", 10),
        milvus_max_retries=getattr(settings, "MILVUS_MAX_RETRIES", 3),
        embedding_request_timeout=getattr(settings, "EMBEDDING_REQUEST_TIMEOUT", 30),
        embedding_health_timeout=getattr(settings, "EMBEDDING_HEALTH_CHECK_TIMEOUT", 5),
    )
    logger.info("v1_langchain_initialized")


# ── Factory de l'application ──────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory FastAPI — peut être importée pour les tests."""
    app = FastAPI(
        title="GOV-AI 2.0",
        description=(
            "Assistant gouvernemental intelligent pour l'administration publique camerounaise. "
            "Bilingue FR/EN, bijuridique (droit civil / common law), souverain (100% on-premise)."
        ),
        version="2.0.0-sprint1",
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # ── CORS ─────────────────────────────────────────────────────────────────
    origins = ["*"] if not settings.is_production else [
        "https://govai.cenadi.cm",
        "https://admin.cenadi.cm",
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "DELETE", "OPTIONS"],
        allow_headers=["*"],
        expose_headers=["X-Trace-ID", "X-Latency-Ms", "Content-Disposition"],
    )

    # ── Middleware audit + rate limiting ──────────────────────────────────────
    from app.api.middleware.audit_middleware import AuditMiddleware
    from app.api.middleware.rate_limiter import RateLimiterMiddleware
    app.add_middleware(AuditMiddleware)
    app.add_middleware(RateLimiterMiddleware)

    # ── Gestionnaires d'exceptions ────────────────────────────────────────────
    register_exception_handlers(app)

    # ── Routes GOV-AI 2.0 (v2) ───────────────────────────────────────────────
    from app.api.v1 import api_router as api_router_v2
    app.include_router(api_router_v2, prefix=settings.api_v1_prefix)

    # ── Routes v1 rétrocompatibilité ─────────────────────────────────────────
    try:
        from app.api.api import api_router as api_router_v1
        app.include_router(api_router_v1, prefix=settings.api_prefix)
        logger.info("v1_routes_registered", prefix=settings.api_prefix)
    except ImportError:
        logger.warning("v1_routes_not_found")

    # ── Fichiers statiques (interface web v1) ─────────────────────────────────
    _front_dir = Path(__file__).resolve().parent / "front"
    if _front_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(_front_dir), html=True), name="front")

    # ── Routes de base ────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": "GOV-AI 2.0",
            "description": "Assistant gouvernemental intelligent — Administration camerounaise",
            "version": "2.0.0-sprint1",
            "documentation": "/docs",
            "health": "/api/v1/health",
            "status": "online",
        }

    # Routes /health au niveau racine (rétrocompatibilité + Docker healthcheck)
    @app.get("/health", include_in_schema=False)
    async def root_health() -> dict:
        return {"status": "ok", "timestamp": time.time(), "version": "2.0.0-sprint1"}

    @app.get("/health/live", include_in_schema=False)
    async def root_health_live() -> dict:
        return {"status": "alive"}

    return app


# ── Instance singleton ────────────────────────────────────────────────────────
app = create_app()

# ── Point d'entrée Uvicorn ────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.app_host,
        port=settings.app_port,
        reload=settings.app_debug,
        log_level=settings.log_level.lower(),
    )
