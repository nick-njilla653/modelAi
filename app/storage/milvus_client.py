"""
GOV-AI 2.0 — Client Milvus (base vectorielle HNSW).
Gère la collection govai_chunks avec index HNSW + COSINE.
"""
from __future__ import annotations

from typing import Any, Optional

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    MilvusException,
    connections,
    utility,
)

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_collection: Collection | None = None


def connect_milvus() -> None:
    """Établit la connexion Milvus."""
    settings = get_settings()
    try:
        connections.connect(
            alias="default",
            host=settings.milvus_host,
            port=settings.milvus_port,
        )
        logger.info("milvus_connected", host=settings.milvus_host, port=settings.milvus_port)
    except MilvusException as exc:
        logger.error("milvus_connection_failed", error=str(exc))
        raise


def get_collection_schema(dim: int) -> CollectionSchema:
    """Définit le schéma de la collection chunks."""
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="chunk_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=36),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=4096),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="language", dtype=DataType.VARCHAR, max_length=10),
        FieldSchema(name="page", dtype=DataType.INT64),
        FieldSchema(name="chunk_index", dtype=DataType.INT64),
        FieldSchema(name="doc_type", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="institution", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="jurisdiction", dtype=DataType.VARCHAR, max_length=256),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim),
    ]
    return CollectionSchema(
        fields=fields,
        description="GOV-AI 2.0 — Chunks documentaires avec embeddings multilingues",
    )


def ensure_collection() -> Collection:
    """Crée ou récupère la collection Milvus."""
    global _collection
    settings = get_settings()
    collection_name = settings.milvus_collection_chunks

    if _collection is not None:
        return _collection

    if not utility.has_collection(collection_name):
        schema = get_collection_schema(settings.milvus_dim)
        _collection = Collection(name=collection_name, schema=schema)
        logger.info("milvus_collection_created", name=collection_name)

        # Index HNSW
        index_params = {
            "index_type": settings.milvus_index_type,
            "metric_type": settings.milvus_metric_type,
            "params": {"M": 16, "efConstruction": 200},
        }
        _collection.create_index(field_name="embedding", index_params=index_params)
        logger.info("milvus_index_created", index_type=settings.milvus_index_type)
    else:
        _collection = Collection(name=collection_name)

    _collection.load()
    return _collection


def insert_chunks(
    chunk_ids: list[str],
    doc_ids: list[str],
    contents: list[str],
    sources: list[str],
    languages: list[str],
    pages: list[int],
    chunk_indexes: list[int],
    doc_types: list[str],
    institutions: list[str],
    jurisdictions: list[str],
    embeddings: list[list[float]],
) -> list[int]:
    """Insère des chunks dans Milvus. Retourne les IDs auto-générés."""
    collection = ensure_collection()
    data = [
        chunk_ids, doc_ids, contents, sources, languages, pages,
        chunk_indexes, doc_types, institutions, jurisdictions, embeddings,
    ]
    result = collection.insert(data)
    collection.flush()
    inserted_ids: list[int] = result.primary_keys
    logger.info("milvus_chunks_inserted", count=len(inserted_ids))
    return inserted_ids


def search_dense(
    query_embedding: list[float],
    top_k: int = 20,
    filters: Optional[str] = None,
    ef: int = 64,
) -> list[dict[str, Any]]:
    """Recherche par similarité vectorielle (dense retrieval)."""
    collection = ensure_collection()
    search_params = {"metric_type": "COSINE", "params": {"ef": ef}}

    results = collection.search(
        data=[query_embedding],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        expr=filters,
        output_fields=[
            "chunk_id", "doc_id", "content", "source",
            "language", "page", "chunk_index", "doc_type",
            "institution", "jurisdiction",
        ],
    )

    hits = []
    for hit in results[0]:
        hits.append({
            "milvus_id": hit.id,
            "chunk_id": hit.entity.get("chunk_id", ""),
            "doc_id": hit.entity.get("doc_id", ""),
            "content": hit.entity.get("content", ""),
            "source": hit.entity.get("source", ""),
            "language": hit.entity.get("language", "unknown"),
            "page": hit.entity.get("page", 0),
            "chunk_index": hit.entity.get("chunk_index", 0),
            "doc_type": hit.entity.get("doc_type", ""),
            "institution": hit.entity.get("institution", ""),
            "jurisdiction": hit.entity.get("jurisdiction", ""),
            "dense_score": float(hit.score),
        })
    return hits


def delete_by_doc_id(doc_id: str) -> None:
    """Supprime tous les chunks d'un document."""
    collection = ensure_collection()
    collection.delete(expr=f'doc_id == "{doc_id}"')
    logger.info("milvus_chunks_deleted", doc_id=doc_id)


def check_milvus_connection() -> bool:
    """Vérifie la connexion Milvus (auto-connect si besoin)."""
    try:
        utility.list_collections()
        return True
    except Exception:
        # Tentative de connexion automatique
        try:
            connect_milvus()
            utility.list_collections()
            return True
        except Exception as exc:
            logger.error("milvus_health_check_failed", error=str(exc))
            return False


def disconnect_milvus() -> None:
    """Ferme la connexion Milvus."""
    global _collection
    _collection = None
    connections.disconnect("default")
    logger.info("milvus_disconnected")
