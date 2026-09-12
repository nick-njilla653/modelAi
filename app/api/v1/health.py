"""
GOV-AI 2.0 — Routes de santé (/api/v1/health).
"""
from __future__ import annotations

from typing import Optional

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.schemas import GraphVolumetry, HealthResponse

router = APIRouter(tags=["health"])
logger = get_logger(__name__)


@router.get("/health", response_model=HealthResponse, summary="Health check global")
async def health_check() -> HealthResponse:
    """
    Vérifie la disponibilité de GOV-AI 2.0 et de ses services dépendants.
    """
    settings = get_settings()
    services: dict[str, str] = {}

    # Milvus (connexion synchrone)
    try:
        from app.storage.milvus_client import check_milvus_connection
        ok = check_milvus_connection()
        services["milvus"] = "ok" if ok else "degraded"
    except Exception as exc:
        services["milvus"] = f"error: {exc}"

    # Elasticsearch
    try:
        from app.storage.elasticsearch_client import get_es_client, check_es_connection
        ok = await check_es_connection()
        services["elasticsearch"] = "ok" if ok else "degraded"
    except Exception as exc:
        services["elasticsearch"] = f"error: {exc}"

    # PostgreSQL
    try:
        from app.storage.postgres_client import check_postgres_connection
        ok = await check_postgres_connection()
        services["postgres"] = "ok" if ok else "degraded"
    except Exception as exc:
        services["postgres"] = f"error: {exc}"

    # Embedding
    try:
        from app.services.embedding.embedding_service import get_embedding_service
        svc = get_embedding_service()
        ok = await svc.health_check()
        services["embedding"] = "ok" if ok else "degraded"
    except Exception as exc:
        services["embedding"] = f"error: {exc}"

    # Neo4j (Sprint 2) — et ce qu'il contient. Une connexion établie sur un
    # graphe vide s'annonçait « ok » : c'est l'état dans lequel il est resté
    # tout un sprint sans que rien ne le dise.
    graph: Optional[GraphVolumetry] = None
    try:
        from app.storage.neo4j_client import check_neo4j_connection
        ok = await check_neo4j_connection()
        services["neo4j"] = "ok" if ok else "degraded"
        if ok:
            from app.services.knowledge_graph.graph_builder import graph_stats
            counts = await graph_stats()
            graph = GraphVolumetry(
                texts=counts.get("texts", 0),
                articles=counts.get("articles", 0),
                references=counts.get("refs", 0),
                institutions=counts.get("institutions", 0),
                mentions=counts.get("mentions", 0),
                external_texts=counts.get("external_texts", 0),
            )
    except Exception as exc:
        services["neo4j"] = f"error: {exc}"

    # Reranker (Sprint 2) — vérifie la disponibilité du package sans télécharger le modèle
    try:
        from sentence_transformers import CrossEncoder  # noqa: F401
        services["reranker"] = "ok"
    except ImportError:
        services["reranker"] = "unavailable"
    except Exception as exc:
        services["reranker"] = f"error: {exc}"

    overall = "ok" if all(v == "ok" for v in services.values()) else "degraded"

    return HealthResponse(
        status=overall,
        version="2.0.0-sprint3",
        services=services,
        model=settings.llm_model,
        graph=graph,
    )


@router.get("/health/live", summary="Liveness probe (Kubernetes)")
async def liveness() -> dict:
    """Simple liveness probe — répond OK si le processus tourne."""
    return {"status": "alive"}


@router.get("/health/ready", summary="Readiness probe (Kubernetes)")
async def readiness() -> JSONResponse:
    """Readiness probe — vérifie que les services critiques sont disponibles."""
    try:
        from app.storage.milvus_client import check_milvus_connection
        milvus_ok = check_milvus_connection()
    except Exception:
        milvus_ok = False

    if not milvus_ok:
        return JSONResponse(
            status_code=503,
            content={"status": "not_ready", "reason": "milvus_unavailable"},
        )

    return JSONResponse(content={"status": "ready"})
