"""
GOV-AI 2.0 — Corpus Re-Indexer.

Deux responsabilités :
  1. Re-indexation Milvus après swap du modèle d'embedding :
     Les vecteurs existants ont été calculés avec l'ancien modèle (espace vectoriel A).
     Le nouveau modèle fine-tuné opère dans un espace B ≠ A → la recherche cosinus
     devient invalide. Tous les chunks doivent être re-embeddés.

  2. Ingestion corpus : route les documents uploadés (ZIP) vers le pipeline d'ingestion
     existant afin d'enrichir Neo4j, Milvus et Elasticsearch simultanément.
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Callable, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)

INGEST_SUPPORTED = {".pdf", ".txt", ".md", ".docx"}


class CorpusReindexer:
    """Utilitaires pour maintenir la cohérence du corpus après fine-tuning."""

    # ── Re-indexation Milvus ──────────────────────────────────────────────────

    async def reindex_milvus(
        self,
        new_embedding_model_path: str,
        batch_size: int = 64,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """
        Re-calcule tous les embeddings avec le nouveau modèle et recharge Milvus.
        À appeler OBLIGATOIREMENT après apply_embedding().
        """
        if progress_cb:
            await progress_cb(0, "Chargement des chunks depuis PostgreSQL…")

        chunks = await self._load_all_chunks()
        total = len(chunks)

        if not total:
            return {"reindexed": 0, "message": "Aucun chunk à re-indexer"}

        logger.info("milvus_reindex_start", total_chunks=total, model=new_embedding_model_path)

        if progress_cb:
            await progress_cb(5, f"{total} chunks à re-embedder avec le nouveau modèle…")

        # Charger le nouveau modèle
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(new_embedding_model_path)

        texts = [c["content"] for c in chunks]
        ids   = [c["id"] for c in chunks]
        doc_ids  = [c["doc_id"] for c in chunks]
        sources  = [c.get("source", "") for c in chunks]
        languages = [c.get("language", "fr") for c in chunks]
        pages    = [c.get("page", 0) or 0 for c in chunks]
        indexes  = [c.get("chunk_index", 0) for c in chunks]
        doc_types = [c.get("doc_type", "") for c in chunks]
        institutions = [c.get("institution", "") for c in chunks]
        jurisdictions = [c.get("jurisdiction", "") for c in chunks]
        article_refs = [c.get("article_ref", "") for c in chunks]

        # Suppression de l'ancienne collection Milvus
        if progress_cb:
            await progress_cb(8, "Suppression des anciens vecteurs Milvus…")

        from app.storage.milvus_client import (
            connect_milvus, ensure_collection, drop_collection, insert_chunks
        )
        from app.core.config import get_settings
        settings = get_settings()

        try:
            drop_collection()
        except Exception as exc:
            logger.warning("milvus_drop_failed", error=str(exc))

        connect_milvus()
        ensure_collection()

        # Re-embedding par batch avec mise à jour de la progression
        reindexed = 0
        for start in range(0, total, batch_size):
            batch_texts = texts[start : start + batch_size]
            batch_embeddings = model.encode(
                batch_texts,
                normalize_embeddings=True,
                show_progress_bar=False,
            ).tolist()

            insert_chunks(
                chunk_ids=ids[start : start + batch_size],
                doc_ids=doc_ids[start : start + batch_size],
                contents=batch_texts,
                embeddings=batch_embeddings,
                sources=sources[start : start + batch_size],
                languages=languages[start : start + batch_size],
                pages=pages[start : start + batch_size],
                chunk_indexes=indexes[start : start + batch_size],
                doc_types=doc_types[start : start + batch_size],
                institutions=institutions[start : start + batch_size],
                jurisdictions=jurisdictions[start : start + batch_size],
                article_refs=article_refs[start : start + batch_size],
            )

            reindexed += len(batch_texts)

            if progress_cb:
                pct = int((reindexed / total) * 85) + 10
                await progress_cb(pct, f"Re-indexation : {reindexed}/{total} chunks…")

        if progress_cb:
            await progress_cb(98, f"Re-indexation terminée : {reindexed} chunks mis à jour.")

        logger.info("milvus_reindex_done", reindexed=reindexed)
        return {
            "reindexed": reindexed,
            "model_used": new_embedding_model_path,
            "message": f"{reindexed} vecteurs recalculés avec le nouveau modèle d'embedding",
        }

    # ── Ingestion corpus depuis ZIP ───────────────────────────────────────────

    async def ingest_corpus_from_zip(
        self,
        zip_path: str | Path,
        doc_type: str = "autre",
        institution: str = "",
        jurisdiction: str = "national",
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """
        Extrait le ZIP et ingère chaque document via le pipeline standard :
        extraction → OCR si nécessaire → chunking → embedding → Milvus + ES + Neo4j.
        """
        import tempfile, shutil

        zip_path = Path(zip_path)
        tmp = zip_path.parent / f"_ingest_{zip_path.stem}"
        tmp.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)

            files = [
                f for f in tmp.rglob("*")
                if f.is_file() and f.suffix.lower() in INGEST_SUPPORTED
            ]

            if not files:
                return {"ingested": 0, "failed": 0, "message": "Aucun fichier supporté trouvé"}

            ingested, failed = 0, 0

            for idx, fp in enumerate(files):
                if progress_cb:
                    pct = int((idx / len(files)) * 90)
                    await progress_cb(pct, f"Ingestion : {fp.name}…")

                try:
                    await self._ingest_single_file(fp, doc_type, institution, jurisdiction)
                    ingested += 1
                    logger.info("corpus_file_ingested", file=fp.name)
                except Exception as exc:
                    failed += 1
                    logger.warning("corpus_file_ingest_failed", file=fp.name, error=str(exc))

            if progress_cb:
                await progress_cb(100, f"Corpus mis à jour : {ingested} documents ingérés, {failed} échec(s).")

            return {
                "ingested": ingested,
                "failed": failed,
                "total": len(files),
                "message": f"{ingested}/{len(files)} documents ingérés (PostgreSQL + Milvus + Elasticsearch + graphe)",
            }

        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def _ingest_single_file(
        self,
        file_path: Path,
        doc_type: str,
        institution: str,
        jurisdiction: str,
    ):
        from app.services.ingestion.ingestion_service import IngestionService
        from app.models.schemas import IngestRequest
        from app.services.embedding import get_embedding_service

        request = IngestRequest(
            source=file_path.name,
            doc_type=doc_type or None,
            institution=institution or None,
            jurisdiction=jurisdiction or None,
            force_ocr=False,
        )
        svc = IngestionService()
        svc.set_embedding_service(get_embedding_service())
        await svc.ingest_document(file_path=str(file_path), request=request)

    # ── Helpers PostgreSQL ────────────────────────────────────────────────────

    async def _load_all_chunks(self) -> list[dict]:
        """Charge tous les chunks de PostgreSQL pour la re-indexation."""
        try:
            import sqlalchemy as sa
            from app.storage.postgres_client import get_db_session
            from app.models.db_models import Chunk, Document

            async with get_db_session() as db:
                rows = (await db.execute(
                    sa.select(
                        Chunk.id,
                        Chunk.doc_id,
                        Chunk.content,
                        Chunk.language,
                        Chunk.page,
                        Chunk.chunk_index,
                        Chunk.article_ref,
                        Document.source,
                        Document.doc_type,
                        Document.institution,
                        Document.jurisdiction,
                    ).join(Document, Chunk.doc_id == Document.id)
                )).all()

                return [
                    {
                        "id": str(r.id),
                        "doc_id": str(r.doc_id),
                        "content": r.content,
                        "language": r.language,
                        "page": r.page or 0,
                        "chunk_index": r.chunk_index,
                        "source": r.source or "",
                        "doc_type": r.doc_type or "",
                        "institution": r.institution or "",
                        "jurisdiction": r.jurisdiction or "",
                        "article_ref": r.article_ref or "",
                    }
                    for r in rows
                ]
        except Exception as exc:
            logger.error("load_chunks_failed", error=str(exc))
            return []
