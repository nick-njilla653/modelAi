"""
GOV-AI 2.0 — Schémas Pydantic v2 pour les API request/response.
Format de sortie standardisé : answer, citations, retrieved_chunks,
graph_evidence, uncertainty_score, safety_flags.
"""
from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

from app.models.domain import (
    ConfidenceLevel,
    DocumentType,
    IntentType,
    JuridicalSystem,
    Language,
    SafetyFlag,
    UserProfile,
)


# ── Modèles de base ───────────────────────────────────────────────────────────

class Citation(BaseModel):
    """Une citation vérifiable pointant vers une source documentaire."""
    source_id: str = Field(..., description="ID unique du document source")
    doc_title: str = Field(..., description="Titre du document")
    doc_type: Optional[DocumentType] = None
    institution: Optional[str] = None
    jurisdiction: Optional[str] = None
    language: Language = Language.FR
    page: Optional[int] = None
    article: Optional[str] = None
    chunk_id: str = Field(..., description="ID du chunk source")
    excerpt: str = Field(..., description="Extrait du passage cité (max 300 chars)")
    relevance_score: float = Field(..., ge=0.0, le=1.0)
    date_document: Optional[str] = None

    @field_validator("excerpt")
    @classmethod
    def truncate_excerpt(cls, v: str) -> str:
        return v[:300] if len(v) > 300 else v


class RetrievedChunk(BaseModel):
    """Un chunk récupéré avec ses métadonnées et score."""
    chunk_id: str
    doc_id: str
    content: str
    source: str
    language: Language = Language.FR
    page: Optional[int] = None
    chunk_index: int = 0
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    rrf_score: Optional[float] = None
    rerank_score: Optional[float] = None
    final_score: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ChunkMetadata(BaseModel):
    """Métadonnées associées à un chunk lors de l'ingestion."""
    source: str
    language: Language = Language.UNKNOWN
    doc_type: Optional[DocumentType] = None
    institution: Optional[str] = None
    jurisdiction: Optional[str] = None
    date_document: Optional[str] = None
    version: Optional[str] = None
    page: Optional[int] = None
    chunk_index: int = 0
    total_chunks: int = 1
    chunk_strategy: str = "fixed_size"


# ── Requêtes ──────────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    """Requête principale GOV-AI 2.0."""
    query: str = Field(
        ..., min_length=3, max_length=2000, description="Question de l'utilisateur"
    )
    user_id: Optional[str] = Field(default=None)
    session_id: Optional[str] = Field(default=None)
    language: Optional[Language] = Field(default=None)
    profile: UserProfile = Field(default=UserProfile.CITIZEN)
    juridical_system: Optional[JuridicalSystem] = Field(default=None)
    top_k: int = Field(default=5, ge=1, le=20)
    filters: dict[str, Any] = Field(default_factory=dict)
    stream: bool = Field(default=False)
    include_chunks: bool = Field(default=True)
    session_context: Optional[str] = Field(
        default=None,
        max_length=4000,
        description=(
            "Résumé des échanges précédents pour la cohérence multi-tour. "
            "Format : 'Q: <question>\nR: <réponse>' pour chaque tour, séparés par \\n\\n."
        ),
    )

    @field_validator("query")
    @classmethod
    def sanitize_query(cls, v: str) -> str:
        return v.strip()

    @model_validator(mode="after")
    def set_session_id(self) -> "QueryRequest":
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        return self


class IngestRequest(BaseModel):
    """Métadonnées accompagnant un fichier à ingérer."""
    source: str = Field(..., description="Nom/identifiant de la source")
    doc_type: Optional[DocumentType] = None
    institution: Optional[str] = None
    jurisdiction: Optional[str] = None
    language: Optional[Language] = None
    date_document: Optional[str] = None
    version: str = Field(default="1.0")
    force_ocr: bool = Field(default=False)
    chunking_strategy: str = Field(default="hybrid")


class EvaluationRequest(BaseModel):
    """Requête d'évaluation sur un jeu de données annoté."""
    dataset_path: Optional[str] = None
    baselines: list[str] = Field(default=["b0", "b1", "b2", "b3", "b4"])
    top_k_values: list[int] = Field(default=[1, 3, 5, 10])
    output_format: str = Field(default="json")


# ── Réponses ──────────────────────────────────────────────────────────────────

class QueryResponse(BaseModel):
    """Réponse principale GOV-AI 2.0 — format standardisé."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    graph_evidence: list[dict[str, Any]] = Field(default_factory=list)
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    safety_flags: list[SafetyFlag] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    language_detected: Language = Language.FR
    juridical_system_detected: Optional[JuridicalSystem] = None
    intent_detected: Optional[IntentType] = None
    latency_ms: Optional[float] = None
    model_used: Optional[str] = None
    session_id: Optional[str] = None

    @model_validator(mode="after")
    def set_confidence_level(self) -> "QueryResponse":
        self.confidence_level = ConfidenceLevel.from_score(self.uncertainty_score)
        return self


class IngestResponse(BaseModel):
    """Réponse après ingestion d'un document."""
    document_id: str
    filename: str
    chunks_created: int
    language_detected: Language
    doc_type: Optional[DocumentType] = None
    ocr_used: bool = False
    ingestion_latency_ms: float
    status: str = "completed"
    warnings: list[str] = Field(default_factory=list)


class DocumentInfo(BaseModel):
    """Informations sur un document indexé."""
    document_id: str
    filename: str
    source: str
    language: Language
    doc_type: Optional[DocumentType] = None
    institution: Optional[str] = None
    jurisdiction: Optional[str] = None
    chunks_count: int
    ingested_at: datetime
    version: str = "1.0"
    status: str


class MetricsResponse(BaseModel):
    """Métriques d'évaluation Sprint 1."""
    precision_at_k: dict[int, float] = Field(default_factory=dict)
    recall_at_k: dict[int, float] = Field(default_factory=dict)
    mrr: Optional[float] = None
    ndcg_at_k: dict[int, float] = Field(default_factory=dict)
    hit_rate_at_k: dict[int, float] = Field(default_factory=dict)
    reranker_gain: Optional[float] = None
    faithfulness_score: Optional[float] = None
    citation_precision: Optional[float] = None
    citation_recall: Optional[float] = None
    hallucination_rate: Optional[float] = None
    latency_p50_ms: Optional[float] = None
    latency_p95_ms: Optional[float] = None
    latency_p99_ms: Optional[float] = None
    isb: Optional[float] = None
    evaluated_at: datetime = Field(default_factory=datetime.utcnow)
    baseline: Optional[str] = None
    dataset_size: int = 0


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    version: str = "2.0.0"
    services: dict[str, str] = Field(default_factory=dict)
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Rétrocompatibilité v1 ─────────────────────────────────────────────────────

class TextsRequest(BaseModel):
    """Compatibilité v1 : embeddings directs."""
    texts: list[str] = Field(..., description="Textes à transformer en embeddings")


class EmbeddingsResponse(BaseModel):
    """Compatibilité v1."""
    embeddings: list[list[float]] = Field(..., description="Embeddings générés")
