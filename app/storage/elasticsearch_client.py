"""
GOV-AI 2.0 — Client Elasticsearch async (BM25 lexical search).
Index bilingue FR/EN avec analyzeurs linguistiques adaptés.
"""
from __future__ import annotations

from typing import Any, Optional

from elasticsearch import AsyncElasticsearch, NotFoundError

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

_client: AsyncElasticsearch | None = None


def get_es_client() -> AsyncElasticsearch:
    """Singleton client Elasticsearch."""
    global _client
    if _client is None:
        settings = get_settings()
        kwargs: dict[str, Any] = {"hosts": [settings.elasticsearch_url]}
        if settings.elasticsearch_user:
            kwargs["basic_auth"] = (
                settings.elasticsearch_user,
                settings.elasticsearch_password,
            )
        _client = AsyncElasticsearch(**kwargs)
    return _client


def _get_index_mapping() -> dict[str, Any]:
    """Mapping Elasticsearch avec analyzeurs bilingues FR/EN."""
    return {
        "settings": {
            "number_of_shards": 1,
            "number_of_replicas": 0,
            "analysis": {
                "analyzer": {
                    "french_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "french_elision",
                            "french_stop",
                            "french_stemmer",
                        ],
                    },
                    "english_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": [
                            "lowercase",
                            "english_stop",
                            "english_stemmer",
                            "english_possessive_stemmer",
                        ],
                    },
                    "multilingual_analyzer": {
                        "type": "custom",
                        "tokenizer": "standard",
                        "filter": ["lowercase", "asciifolding"],
                    },
                },
                "filter": {
                    "french_elision": {
                        "type": "elision",
                        "articles_case": True,
                        "articles": [
                            "l", "m", "t", "qu", "n", "s", "j", "d", "c",
                            "jusqu", "quoiqu", "lorsqu", "puisqu",
                        ],
                    },
                    "french_stop": {"type": "stop", "stopwords": "_french_"},
                    "french_stemmer": {"type": "stemmer", "language": "light_french"},
                    "english_stop": {"type": "stop", "stopwords": "_english_"},
                    "english_stemmer": {"type": "stemmer", "language": "english"},
                    "english_possessive_stemmer": {
                        "type": "stemmer",
                        "language": "possessive_english",
                    },
                },
            },
        },
        "mappings": {
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "content": {
                    "type": "text",
                    "analyzer": "multilingual_analyzer",
                    "fields": {
                        "fr": {"type": "text", "analyzer": "french_analyzer"},
                        "en": {"type": "text", "analyzer": "english_analyzer"},
                    },
                },
                "source": {"type": "keyword"},
                "language": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "institution": {"type": "keyword"},
                "jurisdiction": {"type": "keyword"},
                "page": {"type": "integer"},
                "chunk_index": {"type": "integer"},
                "ingested_at": {"type": "date"},
            }
        },
    }


async def ensure_index() -> None:
    """Crée l'index BM25 s'il n'existe pas."""
    client = get_es_client()
    settings = get_settings()
    index_name = settings.elasticsearch_index_chunks
    try:
        exists = await client.indices.exists(index=index_name)
        if not exists:
            await client.indices.create(index=index_name, body=_get_index_mapping())
            logger.info("elasticsearch_index_created", index=index_name)
        else:
            logger.debug("elasticsearch_index_exists", index=index_name)
    except Exception as exc:
        logger.error("elasticsearch_index_creation_failed", error=str(exc))
        raise


async def index_chunk(
    chunk_id: str,
    doc_id: str,
    content: str,
    source: str,
    language: str,
    page: int,
    chunk_index: int,
    doc_type: str = "",
    institution: str = "",
    jurisdiction: str = "",
) -> str:
    """Indexe un chunk dans Elasticsearch."""
    client = get_es_client()
    settings = get_settings()
    from datetime import datetime

    body = {
        "chunk_id": chunk_id,
        "doc_id": doc_id,
        "content": content,
        "source": source,
        "language": language,
        "doc_type": doc_type,
        "institution": institution,
        "jurisdiction": jurisdiction,
        "page": page,
        "chunk_index": chunk_index,
        "ingested_at": datetime.utcnow().isoformat(),
    }
    response = await client.index(
        index=settings.elasticsearch_index_chunks, id=chunk_id, body=body
    )
    return response["_id"]


async def bulk_index_chunks(chunks: list[dict[str, Any]]) -> int:
    """Indexe plusieurs chunks en bulk."""
    from elasticsearch.helpers import async_bulk

    client = get_es_client()
    settings = get_settings()
    from datetime import datetime

    actions = [
        {
            "_index": settings.elasticsearch_index_chunks,
            "_id": chunk["chunk_id"],
            "_source": {
                **chunk,
                "ingested_at": datetime.utcnow().isoformat(),
            },
        }
        for chunk in chunks
    ]
    success, _ = await async_bulk(client, actions)
    logger.info("elasticsearch_bulk_indexed", count=success)
    return success


async def search_bm25(
    query: str,
    top_k: int = 20,
    language: Optional[str] = None,
    filters: Optional[dict[str, Any]] = None,
) -> list[dict[str, Any]]:
    """Recherche BM25 avec filtres optionnels."""
    client = get_es_client()
    settings = get_settings()

    # Choisir l'analyseur selon la langue
    content_field = "content"
    if language == "fr":
        content_field = "content.fr"
    elif language == "en":
        content_field = "content.en"

    must_clauses: list[dict[str, Any]] = [
        {"multi_match": {
            "query": query,
            "fields": [content_field, "content^0.5"],
            "type": "best_fields",
        }}
    ]

    filter_clauses: list[dict[str, Any]] = []
    if filters:
        for field, value in filters.items():
            filter_clauses.append({"term": {field: value}})

    es_query: dict[str, Any] = {
        "bool": {
            "must": must_clauses,
            "filter": filter_clauses,
        }
    }

    response = await client.search(
        index=settings.elasticsearch_index_chunks,
        body={"query": es_query, "size": top_k},
    )

    hits = []
    for hit in response["hits"]["hits"]:
        src = hit["_source"]
        hits.append({
            "chunk_id": src.get("chunk_id", hit["_id"]),
            "doc_id": src.get("doc_id", ""),
            "content": src.get("content", ""),
            "source": src.get("source", ""),
            "language": src.get("language", "unknown"),
            "page": src.get("page", 0),
            "chunk_index": src.get("chunk_index", 0),
            "doc_type": src.get("doc_type", ""),
            "institution": src.get("institution", ""),
            "jurisdiction": src.get("jurisdiction", ""),
            "sparse_score": hit["_score"],
        })
    return hits


async def delete_by_doc_id(doc_id: str) -> int:
    """Supprime tous les chunks d'un document."""
    client = get_es_client()
    settings = get_settings()
    response = await client.delete_by_query(
        index=settings.elasticsearch_index_chunks,
        body={"query": {"term": {"doc_id": doc_id}}},
    )
    deleted: int = response.get("deleted", 0)
    logger.info("elasticsearch_chunks_deleted", doc_id=doc_id, count=deleted)
    return deleted


async def check_es_connection() -> bool:
    """Vérifie la connexion Elasticsearch."""
    try:
        await get_es_client().ping()
        return True
    except Exception as exc:
        logger.error("elasticsearch_health_check_failed", error=str(exc))
        return False


async def close_es_client() -> None:
    """Ferme le client Elasticsearch."""
    global _client
    if _client is not None:
        await _client.close()
        _client = None
        logger.info("elasticsearch_client_closed")
