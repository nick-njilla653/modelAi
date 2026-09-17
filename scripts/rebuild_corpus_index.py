"""
GOV-AI 2.0 — Reconstruction du corpus indexé à partir de PostgreSQL.

Motif : les chunks déjà ingérés ont été découpés page par page, ce qui a séparé
certains en-têtes d'article (« Article 102 : » en fin de page) de leur corps
(page suivante). Résultat : des chunks vides de sens à l'indexation, et des corps
d'article sans référence citable — le modèle reconstruisait alors les numéros
d'articles au jugé.

Ce script rejoue la consolidation sur les chunks existants, puis réindexe
Milvus et Elasticsearch avec le champ `article_ref`.

PostgreSQL fait ici office de source de vérité : les PDF d'origine ne sont pas
conservés après ingestion. Une ré-ingestion depuis le PDF source reste
préférable lorsqu'il est disponible.

Usage :
    python scripts/rebuild_corpus_index.py --dry-run   # rapport, aucune écriture
    python scripts/rebuild_corpus_index.py             # applique
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

BATCH_SIZE = 32


async def load_documents() -> list[dict]:
    """Charge les documents et leurs chunks, ordonnés par position d'origine."""
    import sqlalchemy as sa
    from app.models.db_models import Chunk, Document
    from app.storage.postgres_client import get_db_session

    async with get_db_session() as db:
        docs = (await db.execute(sa.select(Document))).scalars().all()
        result = []
        for doc in docs:
            rows = (await db.execute(
                sa.select(Chunk)
                .where(Chunk.doc_id == doc.id)
                .order_by(Chunk.chunk_index)
            )).scalars().all()
            result.append({
                "id": doc.id,
                "source": doc.source,
                "language": doc.language,
                "doc_type": doc.doc_type or "",
                "institution": doc.institution or "",
                "jurisdiction": doc.jurisdiction or "",
                "chunks": [
                    {
                        "id": c.id,
                        "content": c.content,
                        "page": c.page,
                        "chunk_index": c.chunk_index,
                        "strategy": c.chunk_strategy,
                        "token_count": c.token_count or 0,
                    }
                    for c in rows
                ],
            })
        return result


def consolidate(doc: dict) -> list:
    """Rejoue la consolidation des articles sur les chunks d'un document."""
    from app.services.ingestion.chunker import ChunkResult, consolidate_article_chunks

    chunks = [
        ChunkResult(
            chunk_id=c["id"],
            content=c["content"],
            chunk_index=c["chunk_index"],
            page=c["page"],
            strategy=c["strategy"],
            token_count=c["token_count"],
        )
        for c in doc["chunks"]
    ]
    return consolidate_article_chunks(chunks, language=doc["language"])


async def rewrite_postgres(doc: dict, chunks: list) -> None:
    """Remplace les chunks du document par leur version consolidée."""
    import sqlalchemy as sa
    from app.models.db_models import Chunk
    from app.storage.postgres_client import get_db_session

    async with get_db_session() as db:
        await db.execute(sa.delete(Chunk).where(Chunk.doc_id == doc["id"]))
        for chunk in chunks:
            db.add(Chunk(
                id=chunk.chunk_id,
                doc_id=doc["id"],
                content=chunk.content,
                chunk_index=chunk.chunk_index,
                page=chunk.page,
                token_count=chunk.token_count,
                language=doc["language"],
                chunk_strategy=chunk.strategy,
                article_ref=chunk.article_ref,
            ))


async def reindex(doc: dict, chunks: list) -> int:
    """Recalcule les embeddings et réindexe Milvus + Elasticsearch."""
    from app.services.embedding.embedding_service import get_embedding_service
    from app.storage.elasticsearch_client import bulk_index_chunks
    from app.storage.milvus_client import insert_chunks

    embedding_service = get_embedding_service()
    indexed = 0

    for start in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[start : start + BATCH_SIZE]
        contents = [c.content for c in batch]
        embeddings = await embedding_service.embed_texts(contents)
        size = len(batch)

        insert_chunks(
            chunk_ids=[c.chunk_id for c in batch],
            doc_ids=[doc["id"]] * size,
            contents=contents,
            sources=[doc["source"]] * size,
            languages=[doc["language"]] * size,
            pages=[c.page or 0 for c in batch],
            chunk_indexes=[c.chunk_index for c in batch],
            doc_types=[doc["doc_type"]] * size,
            institutions=[doc["institution"]] * size,
            jurisdictions=[doc["jurisdiction"]] * size,
            article_refs=[(c.article_ref or "")[:64] for c in batch],
            embeddings=embeddings,
        )

        await bulk_index_chunks([
            {
                "chunk_id": c.chunk_id,
                "doc_id": doc["id"],
                "content": c.content,
                "source": doc["source"],
                "language": doc["language"],
                "page": c.page or 0,
                "chunk_index": c.chunk_index,
                "doc_type": doc["doc_type"],
                "institution": doc["institution"],
                "jurisdiction": doc["jurisdiction"],
                "article_ref": c.article_ref or "",
            }
            for c in batch
        ])

        indexed += size
        print(f"    {indexed}/{len(chunks)} chunks indexés", end="\r", flush=True)

    print(f"    {indexed}/{len(chunks)} chunks indexés")
    return indexed


async def main(dry_run: bool) -> None:
    from app.services.ingestion.chunker import _is_header_only
    from app.storage.elasticsearch_client import recreate_index
    from app.storage.milvus_client import connect_milvus, drop_collection, ensure_collection

    documents = await load_documents()
    if not documents:
        print("Aucun document dans le corpus.")
        return

    print(f"{len(documents)} document(s) dans le corpus.\n")

    plans = []
    for doc in documents:
        before = doc["chunks"]
        after = consolidate(doc)
        orphans_before = sum(1 for c in before if _is_header_only(c["content"]))
        with_ref = sum(1 for c in after if c.article_ref)
        literal_ref = sum(
            1 for c in after if c.article_ref and not c.article_ref.endswith((" (suite)", " (cont.)"))
        )

        print(f"  {doc['source']}  [{doc['language']}]")
        print(f"    chunks            : {len(before)} -> {len(after)}")
        print(f"    en-têtes orphelins: {orphans_before} -> "
              f"{sum(1 for c in after if _is_header_only(c.content))}")
        print(f"    référence d'article: {with_ref}/{len(after)} "
              f"(dont {literal_ref} littérales, {with_ref - literal_ref} héritées d'une continuation)")
        plans.append((doc, after))

    if dry_run:
        print("\n--dry-run : aucune écriture effectuée.")
        return

    print("\nRecréation des index (schéma modifié : ajout de article_ref)…")
    connect_milvus()
    drop_collection()
    ensure_collection()
    await recreate_index()

    total = 0
    for doc, chunks in plans:
        print(f"\n  {doc['source']}")
        await rewrite_postgres(doc, chunks)
        total += await reindex(doc, chunks)

    print(f"\nReconstruction terminée : {total} chunks réindexés.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Affiche l'effet de la consolidation sans rien écrire",
    )
    args = parser.parse_args()
    asyncio.run(main(args.dry_run))
