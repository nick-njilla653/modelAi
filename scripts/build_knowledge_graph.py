"""
GOV-AI 2.0 — Construction du graphe de connaissances depuis le corpus indexé.

Le graphe n'a jamais été alimenté : le schéma existait, le lecteur aussi, mais
aucune requête d'écriture. Les documents déjà ingérés doivent donc être
rattrapés — sans les ré-ingérer, puisque PostgreSQL conserve leurs passages et
leurs références d'article.

Usage :
    python scripts/build_knowledge_graph.py --dry-run   # rapport, sans écriture
    python scripts/build_knowledge_graph.py             # construit
    python scripts/build_knowledge_graph.py --reset     # efface puis reconstruit
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


async def load_documents() -> list[dict]:
    """Documents du corpus et leurs passages, dans l'ordre du texte."""
    import sqlalchemy as sa

    from app.models.db_models import Chunk, Document
    from app.storage.postgres_client import get_db_session

    async with get_db_session() as db:
        documents = (await db.execute(sa.select(Document))).scalars().all()
        result = []
        for doc in documents:
            chunks = (await db.execute(
                sa.select(Chunk)
                .where(Chunk.doc_id == doc.id)
                .order_by(Chunk.chunk_index)
            )).scalars().all()
            result.append({"doc": doc, "chunks": chunks})
        return result


async def main(dry_run: bool, reset: bool) -> None:
    from app.services.knowledge_graph.graph_builder import (
        build_document_graph,
        extract_references,
        graph_stats,
    )
    from app.storage.neo4j_client import ensure_schema, run_query

    documents = await load_documents()
    if not documents:
        print("Aucun document dans le corpus.")
        return

    print(f"{len(documents)} document(s) au corpus.\n")

    for entry in documents:
        doc, chunks = entry["doc"], entry["chunks"]
        with_ref = [c for c in chunks if c.article_ref and "(suite)" not in c.article_ref]
        numbers = {c.article_ref for c in with_ref}
        internal = external = 0
        for chunk in chunks:
            for _, cited in extract_references(chunk.content or ""):
                if cited:
                    external += 1
                else:
                    internal += 1

        print(f"  {doc.source}")
        print(f"    passages          : {len(chunks)}")
        print(f"    articles distincts: {len(numbers)}")
        print(f"    renvois internes  : {internal}")
        print(f"    renvois sortants  : {external}")

    if dry_run:
        print("\n--dry-run : aucune écriture dans le graphe.")
        return

    await ensure_schema()

    if reset:
        print("\nEffacement du graphe existant…")
        await run_query("MATCH (n) DETACH DELETE n")

    # Les renvois d'un code vers un autre ne se résolvent que si le second est
    # connu : on passe donc la table complète des documents du corpus.
    corpus_titles = {e["doc"].source: e["doc"].id for e in documents}

    # Deux passes, et c'est nécessaire. Un renvoi d'un code vers un autre n'est
    # créé que si l'article visé existe déjà — la cible est cherchée par MATCH et
    # non par MERGE, pour ne pas fabriquer de disposition vide. Le premier
    # document traité ne peut donc pas encore pointer vers le second. La seconde
    # passe, idempotente, établit ces liens une fois tous les articles présents.
    for pass_number in (1, 2):
        print(f"\nConstruction — passe {pass_number}…")
        for entry in documents:
            doc, chunks = entry["doc"], entry["chunks"]
            stats = await build_document_graph(
                doc_id=doc.id,
                title=doc.source,
                doc_type=doc.doc_type,
                jurisdiction=doc.jurisdiction,
                chunks=chunks,
                corpus_titles=corpus_titles,
                institution=doc.institution,
            )
            print(
                f"  {doc.source[:44]:46} "
                f"{stats.articles:4d} articles, {stats.references:4d} renvois, "
                f"{stats.institutions:3d} institutions ({stats.mentions} mentions)"
            )
            if stats.emitter:
                print(f"      émetteur : {stats.emitter}")
            elif doc.institution:
                print(f"      émetteur hors vocabulaire : {doc.institution}")
            if stats.external_texts:
                print(f"      textes cités hors corpus : {', '.join(sorted(stats.external_texts))}")

    await report_institution_selectivity()
    print("\nGraphe :", await graph_stats())


async def report_institution_selectivity() -> None:
    """
    Part des articles citant chaque institution.

    Une institution présente dans la quasi-totalité des articles n'indexe rien :
    elle ne distingue aucun article des autres, et son arête ne fait que
    grossir le graphe. Ce rapport rend ce jugement vérifiable au lieu de le
    laisser à l'intuition de qui a rédigé la liste.
    """
    from app.storage.neo4j_client import run_query

    rows = await run_query(
        """
        MATCH (a:Article) WITH count(a) AS total
        MATCH (i:Institution)<-[:MENTIONNE]-(a:Article)
        WITH i, total, count(DISTINCT a) AS cites
        RETURN i.label AS label, i.category AS category, cites,
               toFloat(cites) / total AS part
        ORDER BY cites DESC
        """
    )
    if not rows:
        print("\nAucune institution citée.")
        return

    print("\nSélectivité (part des articles citant l'institution) :")
    for row in rows:
        part = row["part"]
        alerte = "  <-- trop répandue pour indexer" if part > 0.5 else ""
        print(f"  {row['label'][:44]:46} {row['cites']:4d}  {part:5.1%}{alerte}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Rapport sans écriture")
    parser.add_argument("--reset", action="store_true", help="Efface le graphe avant de construire")
    args = parser.parse_args()
    asyncio.run(main(args.dry_run, args.reset))
