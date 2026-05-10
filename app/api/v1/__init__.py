from fastapi import APIRouter
from app.api.v1 import health, query, ingest, evaluation

api_router = APIRouter()
api_router.include_router(health.router)
api_router.include_router(query.router)
api_router.include_router(ingest.router)
api_router.include_router(evaluation.router)

__all__ = ["api_router"]
