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
    logger.info("govai2_starting", version="2.0.0-sprint3", env=settings.app_env)

    # ── Startup ──────────────────────────────────────────────────────────────
    # ── Schéma PostgreSQL ────────────────────────────────────────────────────
    # `create_all` ne s'exécute que sur une base vierge. Sur une base déjà
    # versionnée par Alembic, il créerait les tables des migrations en attente
    # avant qu'elles ne soient appliquées : la migration échouerait ensuite sur
    # « relation already exists », et la version resterait figée en arrière.
    # Les deux mécanismes se disputaient le schéma ; celui-ci tranche.
    try:
        import sqlalchemy as sa

        from app.models.db_models import Base
        from app.storage.postgres_client import get_engine

        _engine = get_engine()
        async with _engine.begin() as _conn:
            versioned = await _conn.run_sync(
                lambda sync_conn: sa.inspect(sync_conn).has_table("alembic_version")
            )
            if versioned:
                logger.info(
                    "postgres_schema_managed_by_alembic",
                    hint="alembic upgrade head pour appliquer les migrations en attente",
                )
            else:
                await _conn.run_sync(Base.metadata.create_all)
                logger.info("postgres_schema_created")
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

    # Connexion Neo4j + création du schéma (Sprint 2)
    try:
        from app.storage.neo4j_client import get_neo4j_driver, ensure_schema
        get_neo4j_driver()
        await ensure_schema()
        logger.info("neo4j_connected")
    except Exception as exc:
        logger.warning("neo4j_init_failed", error=str(exc))

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
        from app.storage.neo4j_client import close_neo4j
        await close_neo4j()
        logger.info("neo4j_connection_closed")
    except Exception as exc:
        logger.warning("neo4j_close_error", error=str(exc))

    logger.info("govai2_stopped")


# ── Factory de l'application ──────────────────────────────────────────────────

def create_app() -> FastAPI:
    """Factory FastAPI — peut être importée pour les tests."""
    app = FastAPI(
        title="GOV-AI 2.0",
        description=(
            "Assistant gouvernemental intelligent pour l'administration publique camerounaise. "
            "Bilingue FR/EN, bijuridique (droit civil / common law), souverain (100% on-premise)."
        ),
        version="2.0.0-sprint3",
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

    # ── Fichiers statiques (interfaces web) ──────────────────────────────────
    # L'ancienne interface reste servie sous /ui/legacy : elle porte encore les
    # onglets Ingestion, Évaluation et Fine-Tuning, non repris par la nouvelle.
    _repo_root = Path(__file__).resolve().parent.parent
    _legacy_dir = Path(__file__).resolve().parent / "front"
    if _legacy_dir.exists():
        app.mount(
            "/ui/legacy",
            StaticFiles(directory=str(_legacy_dir), html=True),
            name="front-legacy",
        )

    # Interface React (frontend/dist). Produite par `npm run build` côté hôte :
    # l'image applicative n'embarque pas Node.
    _webui_dir = _repo_root / "frontend" / "dist"
    if _webui_dir.exists():
        app.mount("/ui", StaticFiles(directory=str(_webui_dir), html=True), name="webui")
        logger.info("webui_mounted", path=str(_webui_dir))
    elif _legacy_dir.exists():
        # Sans build disponible, /ui sert l'ancienne interface plutôt qu'un 404.
        app.mount("/ui", StaticFiles(directory=str(_legacy_dir), html=True), name="front")
        logger.warning(
            "webui_build_missing",
            expected=str(_webui_dir),
            hint="Lancez `npm run build` dans frontend/ pour servir la nouvelle interface",
        )

    # ── Routes de base ────────────────────────────────────────────────────────
    @app.get("/", include_in_schema=False)
    async def root() -> dict:
        return {
            "name": "GOV-AI 2.0",
            "description": "Assistant gouvernemental intelligent — Administration camerounaise",
            "version": "2.0.0-sprint3",
            "documentation": "/docs",
            "health": "/api/v1/health",
            "status": "online",
        }

    # Routes /health au niveau racine (rétrocompatibilité + Docker healthcheck)
    @app.get("/health", include_in_schema=False)
    async def root_health() -> dict:
        return {"status": "ok", "timestamp": time.time(), "version": "2.0.0-sprint3"}

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
