"""
Tests unitaires — Chunker (app/services/ingestion/chunker.py).
Vérifie le chunking structurel et à taille fixe pour les documents juridiques FR/EN.
"""
import pytest

from app.services.ingestion.chunker import (
    chunk_document,
    chunk_fixed_size,
    chunk_structural,
    chunk_hybrid,
)


# ── Données de test ────────────────────────────────────────────────────────────

CONSTITUTION_TEXT = """
TITRE I — DE L'ÉTAT

Article 1 : La République du Cameroun est un État unitaire décentralisé.
Elle est une et indivisible, laïque, démocratique et sociale.

Article 2 : La souveraineté nationale appartient au peuple camerounais qui l'exerce,
soit par l'intermédiaire du Président de la République et des membres du Parlement,
soit par voie de référendum.

Article 3 : Le peuple camerounais, conscient de la nécessité d'une parfaite entente
et d'une solidarité entre tous les membres de la communauté nationale,
affirme sa volonté de construire une patrie camerounaise indivisible.

TITRE II — DES LIBERTÉS ET DES DROITS

Article 4 : La liberté de la personne humaine est inviolable. Nul ne peut être arrêté
ou détenu que dans les cas et selon les formes déterminés par la loi.

Article 5 : Le domicile est inviolable. Il ne peut y être opéré de perquisitions
que dans les cas et selon les formes déterminés par la loi.
"""

SHORT_TEXT = "Ceci est un texte très court sans structure légale."

LONG_ARTICLE_TEXT = """
Article 1 : """ + ("x" * 3000) + """

Article 2 : Disposition secondaire.
"""


# ── Tests chunking structurel ─────────────────────────────────────────────────

class TestStructuralChunker:

    def test_detects_articles(self):
        chunks = chunk_structural(CONSTITUTION_TEXT, chunk_size=512, overlap=64)
        assert len(chunks) >= 3, "Doit détecter au moins 3 articles"

    def test_each_chunk_has_content(self):
        chunks = chunk_structural(CONSTITUTION_TEXT, chunk_size=512, overlap=64)
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_chunk_index_sequential(self):
        chunks = chunk_structural(CONSTITUTION_TEXT, chunk_size=512, overlap=64)
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_long_article_subdivided(self):
        """Un article dépassant chunk_size doit être subdivisé."""
        chunks = chunk_structural(LONG_ARTICLE_TEXT, chunk_size=256, overlap=32)
        # L'article 1 de 3000 chars doit être splitté en plusieurs sous-chunks
        assert len(chunks) > 2

    def test_empty_text_returns_empty(self):
        chunks = chunk_structural("", chunk_size=512, overlap=64)
        assert chunks == []


# ── Tests chunking à taille fixe ─────────────────────────────────────────────

class TestFixedSizeChunker:

    def test_short_text_single_chunk(self):
        chunks = chunk_fixed_size(SHORT_TEXT, chunk_size=512, overlap=64)
        assert len(chunks) == 1

    def test_long_text_multiple_chunks(self):
        long_text = "Mot " * 1000  # ~4000 tokens approx
        chunks = chunk_fixed_size(long_text, chunk_size=128, overlap=16)
        assert len(chunks) > 1

    def test_no_empty_chunks(self):
        text = "Texte juridique court. " * 50
        chunks = chunk_fixed_size(text, chunk_size=64, overlap=8)
        for chunk in chunks:
            assert len(chunk.content.strip()) > 0

    def test_overlap_content_shared(self):
        """Les chunks doivent partager du contenu via l'overlap."""
        text = " ".join(f"mot{i}" for i in range(200))
        chunks = chunk_fixed_size(text, chunk_size=50, overlap=10)
        if len(chunks) >= 2:
            # Le dernier mot du chunk[0] doit apparaître dans chunk[1]
            last_words_c0 = chunks[0].content.split()[-5:]
            first_words_c1 = chunks[1].content.split()[:15]
            overlap_found = any(w in first_words_c1 for w in last_words_c0)
            assert overlap_found, "Overlap non détecté entre chunks consécutifs"


# ── Tests chunking hybride ────────────────────────────────────────────────────

class TestHybridChunker:

    def test_structured_text_uses_structural(self):
        chunks = chunk_hybrid(CONSTITUTION_TEXT, chunk_size=512, overlap=64)
        assert len(chunks) >= 2

    def test_unstructured_text_uses_fixed_size(self):
        text = "Texte sans structure légale. " * 100
        chunks = chunk_hybrid(text, chunk_size=128, overlap=16)
        assert len(chunks) >= 1


# ── Tests point d'entrée principal ───────────────────────────────────────────

class TestChunkDocument:

    def test_chunk_document_structural_strategy(self):
        from app.models.domain import ChunkStrategy
        chunks = chunk_document(
            text=CONSTITUTION_TEXT,
            strategy=ChunkStrategy.STRUCTURAL,
            chunk_size=512,
            overlap=64,
        )
        assert len(chunks) >= 2

    def test_chunk_document_fixed_strategy(self):
        from app.models.domain import ChunkStrategy
        chunks = chunk_document(
            text=CONSTITUTION_TEXT,
            strategy=ChunkStrategy.FIXED_SIZE,
            chunk_size=256,
            overlap=32,
        )
        assert len(chunks) >= 1

    def test_chunk_document_hybrid_strategy(self):
        from app.models.domain import ChunkStrategy
        chunks = chunk_document(
            text=CONSTITUTION_TEXT,
            strategy=ChunkStrategy.HYBRID,
            chunk_size=512,
            overlap=64,
        )
        assert len(chunks) >= 2

    def test_metadata_preserved(self):
        from app.models.domain import ChunkStrategy
        metadata = {"doc_type": "constitution", "institution": "presidence"}
        chunks = chunk_document(
            text=CONSTITUTION_TEXT,
            strategy=ChunkStrategy.HYBRID,
            chunk_size=512,
            overlap=64,
            metadata=metadata,
        )
        for chunk in chunks:
            assert chunk.metadata.get("doc_type") == "constitution"
