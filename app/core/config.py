"""
GOV-AI 2.0 — Configuration centralisée via pydantic-settings.
Toutes les valeurs sont lues depuis les variables d'environnement ou .env.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_name: str = "gov-ai-2"
    app_env: Literal["development", "production", "test"] = "development"
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_debug: bool = False
    secret_key: str = "changeme-super-secret-key-at-least-32-chars"
    api_v1_prefix: str = "/api/v1"

    # ── PostgreSQL ────────────────────────────────────────────────────────────
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "govai2"
    postgres_user: str = "govai"
    postgres_password: str = "govai_secret"
    database_url: str = (
        "postgresql+asyncpg://govai:govai_secret@localhost:5432/govai2"
    )

    # ── Milvus ────────────────────────────────────────────────────────────────
    milvus_host: str = "localhost"
    milvus_port: int = 19530
    milvus_collection_chunks: str = "govai_chunks"
    milvus_dim: int = 1024
    milvus_index_type: str = "HNSW"
    milvus_metric_type: str = "COSINE"
    milvus_nlist: int = 128
    milvus_ef: int = 64

    # ── Elasticsearch ─────────────────────────────────────────────────────────
    elasticsearch_url: str = "http://localhost:9200"
    elasticsearch_index_chunks: str = "govai_chunks_bm25"
    elasticsearch_user: str = ""
    elasticsearch_password: str = ""

    # ── Neo4j (Sprint 2) ──────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "neo4j_secret"
    neo4j_database: str = "govai"

    # ── LLM ──────────────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai-compatible"] = "ollama"
    llm_base_url: str = "http://localhost:11434"
    llm_model: str = "llama3.2:latest"
    llm_temperature: float = 0.1
    llm_max_tokens: int = 512
    llm_timeout: int = 300
    llm_stream: bool = True

    # ── Embedding ─────────────────────────────────────────────────────────────
    embedding_provider: Literal["local", "ollama"] = "ollama"
    embedding_model: str = "mxbai-embed-large"
    embedding_dim: int = 1024
    embedding_batch_size: int = 32
    embedding_device: str = "cpu"

    # ── Reranker ─────────────────────────────────────────────────────────────
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    reranker_device: str = "cpu"
    reranker_top_k: int = 20

    # ── Retrieval ─────────────────────────────────────────────────────────────
    retrieval_dense_top_k: int = 20
    retrieval_sparse_top_k: int = 20
    retrieval_final_top_k: int = 5
    rrf_k: int = 60
    reranker_weight_score: float = 0.85
    reranker_weight_meta: float = 0.15

    # ── Generation ───────────────────────────────────────────────────────────
    citation_min_score: float = 0.5
    min_chunks_for_answer: int = 1
    confidence_threshold: float = 0.6   # τ_conf
    escalation_threshold: float = 0.3   # τ_esc

    # ── Ingestion ─────────────────────────────────────────────────────────────
    chunk_size: int = 512
    chunk_overlap: int = 64
    max_file_size_mb: int = 50
    supported_languages: str = "fr,en"
    documents_path: str = "./storage/documents"
    ocr_cache_path: str = "./storage/ocr_cache"
    model_cache_path: str = "./storage/model_cache"

    # ── OCR ──────────────────────────────────────────────────────────────────
    ocr_engine: Literal["tesseract", "paddleocr"] = "tesseract"
    tesseract_path: str = "/usr/bin/tesseract"
    tesseract_lang: str = "fra+eng"
    force_ocr: bool = False

    # ── Sécurité ──────────────────────────────────────────────────────────────
    jwt_secret_key: str = "changeme-jwt-secret"
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 1440
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000

    # ── Évaluation ───────────────────────────────────────────────────────────
    eval_dataset_path: str = "./eval/datasets/qa_bilingual_annotated.json"
    eval_reports_path: str = "./eval/reports"
    eval_top_k_values: str = "1,3,5,10"

    # ── Logging ───────────────────────────────────────────────────────────────
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    log_format: Literal["json", "text"] = "json"
    log_file: str = "./logs/govai.log"

    # ── Rétrocompatibilité v1 (aliases) ──────────────────────────────────────
    api_prefix: str = "/api"

    # ── Propriétés calculées ──────────────────────────────────────────────────
    @property
    def supported_languages_list(self) -> list[str]:
        return [lang.strip() for lang in self.supported_languages.split(",")]

    @property
    def eval_top_k_list(self) -> list[int]:
        return [int(k.strip()) for k in self.eval_top_k_values.split(",")]

    @property
    def documents_path_obj(self) -> Path:
        return Path(self.documents_path)

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @field_validator("confidence_threshold", "escalation_threshold")
    @classmethod
    def validate_thresholds(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError("Les seuils doivent être entre 0 et 1")
        return v

    @model_validator(mode="after")
    def ensure_storage_dirs(self) -> "Settings":
        for path_str in [
            self.documents_path,
            self.ocr_cache_path,
            self.model_cache_path,
        ]:
            Path(path_str).mkdir(parents=True, exist_ok=True)
        Path("./logs").mkdir(parents=True, exist_ok=True)
        Path(self.eval_reports_path).mkdir(parents=True, exist_ok=True)
        return self


@lru_cache
def get_settings() -> Settings:
    """Singleton settings — utiliser cette fonction partout."""
    return Settings()


# Instance globale pour compatibilité avec le code existant
settings = get_settings()
