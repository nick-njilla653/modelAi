"""
Tests unitaires — Chunker (app/services/ingestion/chunker.py).
Vérifie le chunking structurel et à taille fixe pour les documents juridiques FR/EN.
"""
import pytest

from app.services.ingestion.chunker import (
    ChunkResult,
    chunk_document,
    chunk_fixed_size,
    chunk_structural,
    chunk_hybrid,
    consolidate_article_chunks,
    _extract_article_ref,
    _is_header_only,
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


# ── Consolidation des articles coupés par une frontière de page ───────────────
# Le chunking se fait page par page : un en-tête « Article 102 : » qui termine une
# page est séparé de son corps, resté sur la page suivante.


class TestExtractArticleRef:

    @pytest.mark.parametrize("raw,expected", [
        ("Article 103 : Le crime...", "Article 103"),
        ("ARTICLE 103.- Le crime...", "Article 103"),
        ("Art. 103 Le crime...", "Article 103"),
        ("art 103 : Le crime...", "Article 103"),
        ("Articles 12 et suivants", "Article 12"),
        ("Section 12 : Definition", "Section 12"),
        ("Article 5 bis : disposition ajoutee", "Article 5 bis"),
        # Dispositions insérées : « Section 25-1 » et « Section 25-2 » sont
        # distinctes de « Section 25 » — les confondre attribuerait un texte
        # à la mauvaise disposition.
        ("SECTION 1-1: No exemption", "Section 1-1"),
        ("SECTION 26 -1: Reparatory sentence", "Section 26-1"),
        ("SECTION 25 - 2: Dissolution", "Section 25-2"),
    ])
    def test_normalises_label(self, raw, expected):
        assert _extract_article_ref(raw) == expected

    def test_returns_none_without_header(self):
        assert _extract_article_ref("(1) La procedure est secrete.") is None
        assert _extract_article_ref("") is None


class TestIsHeaderOnly:

    @pytest.mark.parametrize("raw", [
        "Article 102 :",
        "Article 104 :\n(1)",
        "ARTICLE 106.-",
    ])
    def test_detects_orphan_header(self, raw):
        assert _is_header_only(raw)

    @pytest.mark.parametrize("raw", [
        "Article 105 : Les objets qui ne sont pas utiles a la manifestation de la verite sont restitues.",
        "(1) La procedure durant l'enquete est secrete.",
        "",
    ])
    def test_rejects_real_content(self, raw):
        assert not _is_header_only(raw)


class TestConsolidateArticleChunks:

    @pytest.fixture
    def split_across_pages(self):
        """Séquence réelle observée dans le corpus (CPP, pages 42-43)."""
        return [
            ChunkResult(content="Article 101 :\n(1) L'officier de police judiciaire peut charger...",
                        chunk_index=0, page=42, strategy="structural"),
            ChunkResult(content="Article 102 :", chunk_index=1, page=42, strategy="structural"),
            ChunkResult(content="(1) La procedure durant l'enquete de police judiciaire est secrete.",
                        chunk_index=2, page=43, strategy="structural"),
        ]

    def test_merges_orphan_header_with_body(self, split_across_pages):
        out = consolidate_article_chunks(split_across_pages)

        assert len(out) == 2
        assert out[1].content.startswith("Article 102 :")
        assert "La procedure durant l'enquete" in out[1].content
        assert out[1].article_ref == "Article 102"

    def test_merged_chunk_keeps_header_page(self, split_across_pages):
        """L'article commence à la page de son en-tête, pas à celle de son corps."""
        out = consolidate_article_chunks(split_across_pages)
        assert out[1].page == 42

    def test_no_orphan_header_remains(self, split_across_pages):
        out = consolidate_article_chunks(split_across_pages)
        assert not any(_is_header_only(c.content) for c in out)

    def test_chunk_index_is_recomputed(self, split_across_pages):
        out = consolidate_article_chunks(split_across_pages)
        assert [c.chunk_index for c in out] == [0, 1]

    def test_continuation_inherits_marked_reference(self):
        """Un corps qui déborde hérite de la référence, explicitement marquée."""
        chunks = [
            ChunkResult(content="Article 104 : Dispositions relatives au crime flagrant...",
                        chunk_index=0, page=43, strategy="structural"),
            ChunkResult(content="a) En cas de crime flagrant, l'officier avise informe le Procureur.",
                        chunk_index=1, page=44, strategy="structural"),
        ]
        out = consolidate_article_chunks(chunks, language="fr")

        assert out[0].article_ref == "Article 104"
        assert out[1].article_ref == "Article 104 (suite)"

    def test_continuation_suffix_follows_language(self):
        chunks = [
            ChunkResult(content="Section 104: Provisions on flagrant offences.",
                        chunk_index=0, page=43, strategy="structural"),
            ChunkResult(content="a) The judicial police officer shall inform the State Counsel.",
                        chunk_index=1, page=44, strategy="structural"),
        ]
        out = consolidate_article_chunks(chunks, language="en")
        assert out[1].article_ref == "Section 104 (cont.)"

    def test_new_article_stops_propagation(self):
        chunks = [
            ChunkResult(content="Article 104 : Premier article avec du contenu.", chunk_index=0, page=43),
            ChunkResult(content="Suite du premier article sans en-tete.", chunk_index=1, page=44),
            ChunkResult(content="Article 105 : Second article avec du contenu.", chunk_index=2, page=44),
        ]
        out = consolidate_article_chunks(chunks)
        assert [c.article_ref for c in out] == [
            "Article 104", "Article 104 (suite)", "Article 105",
        ]

    def test_is_idempotent(self, split_across_pages):
        """La reconstruction du corpus peut être rejouée sans dégrader les chunks."""
        once = consolidate_article_chunks(split_across_pages)
        twice = consolidate_article_chunks(list(once))

        assert [c.content for c in twice] == [c.content for c in once]
        assert [c.article_ref for c in twice] == [c.article_ref for c in once]

    def test_empty_input(self):
        assert consolidate_article_chunks([]) == []

    def test_trailing_header_is_kept(self):
        """En-tête en toute fin de document : aucun corps à lui rattacher."""
        chunks = [
            ChunkResult(content="Article 700 : Disposition finale du texte.", chunk_index=0, page=180),
            ChunkResult(content="Article 701 :", chunk_index=1, page=181),
        ]
        out = consolidate_article_chunks(chunks)
        assert len(out) == 2
        assert out[1].article_ref == "Article 701"
