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
    # Recherche web : None laisse le réglage global décider (déclenchement
    # automatique si la confiance est basse), True/False force par requête —
    # ce que l'interface expose sous forme de bouton.
    web_search: Optional[bool] = Field(
        default=None,
        description="Force ou interdit la recherche web pour cette requête",
    )
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

class WebSource(BaseModel):
    """
    Une page web officielle citée en complément du corpus.

    Distincte de `Citation` : une citation renvoie au texte normatif indexé, une
    source web à sa présentation par une institution. Les confondre laisserait
    croire qu'une page ministérielle a la même valeur qu'un article de loi.
    """
    title: str
    url: str
    snippet: str = ""
    domain: str
    institution: str
    acronym: str
    #: 0 = institution suprême … 5 = presse publique.
    priority: int = 1
    #: Faux si le site ne répondait pas au dernier contrôle de disponibilité.
    reachable: bool = True


class GraphArticleRef(BaseModel):
    """Une disposition désignée par le graphe, avec le texte dont elle relève."""
    number: str
    doc_title: str


class GraphEvidence(BaseModel):
    """
    Ce que le graphe de connaissances a apporté à une réponse.

    Distincte d'une `Citation` : une citation atteste un passage du corpus,
    une pièce de graphe atteste une *relation* — entre deux dispositions, ou
    entre une disposition et l'autorité qu'elle nomme. Les confondre
    présenterait une déduction structurelle comme un extrait de texte, ce qui
    est précisément l'erreur que l'ancrage cherche à empêcher.

    Le champ `graph_evidence` de `QueryResponse` était déclaré depuis le
    Sprint 2 mais n'a jamais été rempli : le graphe atteignait le modèle sans
    que rien ne l'expose.
    """
    #: « institution » = autorité nommée ou déduite ; « article » = disposition
    #: retrouvée par une entité de la question.
    kind: str
    label: str
    #: Catégorie de l'institution, ou texte d'appartenance de l'article.
    detail: Optional[str] = None
    #: « passages » : déduit des dispositions retrouvées, sans décompte global.
    #: « corpus » : l'entité est nommée dans la question, le décompte porte sur
    #: tout le corpus indexé. La distinction évite d'annoncer « 66 articles »
    #: à propos de cinq passages.
    scope: Optional[str] = None
    articles: list[GraphArticleRef] = Field(default_factory=list)
    #: Nombre total dans la portée ; peut dépasser `len(articles)`, tronqué.
    article_count: int = 0
    #: Textes dont cette institution est l'émettrice, le cas échéant.
    issued_texts: list[str] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """Réponse principale GOV-AI 2.0 — format standardisé."""
    query_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    answer: str
    citations: list[Citation] = Field(default_factory=list)
    retrieved_chunks: list[RetrievedChunk] = Field(default_factory=list)
    web_sources: list[WebSource] = Field(
        default_factory=list,
        description="Pages officielles consultées sur le web, hors corpus indexé",
    )
    graph_evidence: list[GraphEvidence] = Field(
        default_factory=list,
        description="Relations apportées par le graphe, hors extraits cités",
    )
    uncertainty_score: float = Field(..., ge=0.0, le=1.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    safety_flags: list[SafetyFlag] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    language_detected: Language = Language.FR
    juridical_system_detected: Optional[JuridicalSystem] = None
    intent_detected: Optional[IntentType] = None
    action_plan: list[str] = Field(default_factory=list)
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


class GraphVolumetry(BaseModel):
    """
    Ce que contient effectivement le graphe de connaissances.

    « neo4j : ok » atteste une connexion, pas des données. Le graphe est resté
    vide et connecté pendant toute la durée du Sprint 2, sans que rien ne le
    signale. Ces compteurs distinguent les deux états.
    """
    texts: int = 0
    articles: int = 0
    references: int = 0
    institutions: int = 0
    mentions: int = 0
    #: Textes cités par le corpus mais absents de celui-ci — le graphe sait
    #: nommer ce qui lui manque.
    external_texts: int = 0


class HealthResponse(BaseModel):
    """Réponse du health check."""
    status: str
    version: str = "2.0.0-sprint3"
    services: dict[str, str] = Field(default_factory=dict)
    model: Optional[str] = None
    #: Absente si Neo4j est injoignable — un compteur à zéro et une absence de
    #: réponse ne se disent pas de la même façon.
    graph: Optional[GraphVolumetry] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# ── Conversations et corpus (interface) ───────────────────────────────────────

class SessionSummary(BaseModel):
    """Une conversation telle qu'elle apparaît dans la liste latérale."""
    session_id: str
    title: str = Field(..., description="Dérivé de la première question posée")
    language: str = "fr"
    profile: str = "citizen"
    turns: int = 0
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None


class SessionTurn(BaseModel):
    """Un tour de conversation : la question et la réponse qui lui a été faite."""
    turn_id: str
    query: str
    answer: str
    answer_truncated: bool = Field(
        default=False,
        description="Vrai pour les tours antérieurs à la persistance intégrale "
                    "des réponses : seul un aperçu est disponible",
    )
    citations: list[dict[str, Any]] = Field(default_factory=list)
    safety_flags: list[str] = Field(default_factory=list)
    intent: Optional[str] = None
    confidence: Optional[float] = None
    latency_ms: Optional[float] = None
    model_used: Optional[str] = None
    created_at: Optional[datetime] = None


class SessionDetail(BaseModel):
    """Transcription complète d'une conversation."""
    session_id: str
    title: str
    language: str = "fr"
    profile: str = "citizen"
    created_at: Optional[datetime] = None
    last_active: Optional[datetime] = None
    turns: list[SessionTurn] = Field(default_factory=list)


class CorpusDocument(BaseModel):
    """Un document indexé, tel que présenté dans le panneau des sources."""
    doc_id: str
    source: str
    language: str = "fr"
    doc_type: Optional[str] = None
    chunks: int = 0


# ── Rétrocompatibilité v1 ─────────────────────────────────────────────────────

class TextsRequest(BaseModel):
    """Compatibilité v1 : embeddings directs."""
    texts: list[str] = Field(..., description="Textes à transformer en embeddings")


class EmbeddingsResponse(BaseModel):
    """Compatibilité v1."""
    embeddings: list[list[float]] = Field(..., description="Embeddings générés")
