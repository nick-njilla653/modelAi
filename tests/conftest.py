"""
GOV-AI 2.0 — Configuration pytest (fixtures partagées).
"""
from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest
import pytest_asyncio
from fastapi.testclient import TestClient
from httpx import AsyncClient, ASGITransport

# ── Variables d'environnement de test ─────────────────────────────────────────
os.environ.setdefault("APP_ENV", "test")
os.environ.setdefault("LLM_PROVIDER", "ollama")
os.environ.setdefault("EMBEDDING_PROVIDER", "ollama")
os.environ.setdefault("LOG_FORMAT", "text")
os.environ.setdefault("LOG_LEVEL", "WARNING")
os.environ.setdefault("DATABASE_URL", "postgresql+asyncpg://govai:govai_secret@localhost:5432/govai2_test")


# ── Fixtures de base ──────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def event_loop():
    """Event loop partagé pour les tests async."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def settings():
    from app.core.config import get_settings
    return get_settings()


@pytest.fixture
def app():
    from app.main import create_app
    return create_app()


@pytest.fixture
def client(app):
    with TestClient(app, raise_server_exceptions=False) as c:
        yield c


@pytest_asyncio.fixture
async def async_client(app) -> AsyncGenerator[AsyncClient, None]:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as client:
        yield client


# ── Fixtures de données ────────────────────────────────────────────────────────

@pytest.fixture
def sample_chunk_fr():
    from app.models.schemas import RetrievedChunk
    return RetrievedChunk(
        chunk_id="chunk-001",
        doc_id="doc-001",
        content=(
            "Article 5 de la Constitution camerounaise : La République du Cameroun "
            "est un État unitaire décentralisé. Elle est une et indivisible, laïque, "
            "démocratique et sociale."
        ),
        source="Constitution_Cameroun_1996.pdf",
        language="fr",
        page=3,
        chunk_index=0,
        dense_score=0.87,
        sparse_score=0.72,
        rrf_score=0.019,
        rerank_score=0.91,
        final_score=0.89,
        metadata={"doc_type": "constitution", "institution": "presidence", "jurisdiction": "national"},
    )


@pytest.fixture
def sample_chunk_en():
    from app.models.schemas import RetrievedChunk
    return RetrievedChunk(
        chunk_id="chunk-002",
        doc_id="doc-002",
        content=(
            "Section 3 of the Business Companies Act: A private company limited by shares "
            "must have at least one shareholder and one director. "
            "Registration is mandatory with the RCCM."
        ),
        source="Business_Companies_Act_NW_SW.pdf",
        language="en",
        page=7,
        chunk_index=2,
        dense_score=0.81,
        sparse_score=0.68,
        rrf_score=0.016,
        rerank_score=0.85,
        final_score=0.83,
        metadata={"doc_type": "loi", "institution": "assemblee", "jurisdiction": "NW/SW"},
    )


@pytest.fixture
def sample_query_fr():
    from app.models.schemas import QueryRequest
    from app.models.domain import Language, UserProfile
    return QueryRequest(
        query="Quelles sont les obligations d'une entreprise pour son immatriculation au Cameroun ?",
        language=Language.FR,
        profile=UserProfile.ENTERPRISE,
    )


@pytest.fixture
def sample_query_en():
    from app.models.schemas import QueryRequest
    from app.models.domain import Language, UserProfile
    return QueryRequest(
        query="What are the steps to register a company in Cameroon?",
        language=Language.EN,
        profile=UserProfile.CITIZEN,
    )
