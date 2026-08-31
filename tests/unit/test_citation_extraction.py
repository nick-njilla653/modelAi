"""
Tests unitaires — Résolution des citations (llm_answer_service).

Contrat d'ancrage vérifié ici :
  1. Seul un chunk EFFECTIVEMENT cité devient une citation.
  2. Une référence qui ne correspond à aucun extrait fourni est signalée
     (hallucination de référence), jamais silencieusement acceptée.
"""
import pytest

from app.models.domain import Language
from app.models.schemas import RetrievedChunk
from app.services.generation.llm_answer_service import (
    LLMAnswerService,
    _normalize_source,
    _sources_match,
)


@pytest.fixture
def service():
    return LLMAnswerService()


@pytest.fixture
def chunks():
    return [
        RetrievedChunk(
            chunk_id="c1", doc_id="d1",
            content="Le crime ou le délit est qualifié de flagrant lorsqu'il se commet actuellement.",
            source="code penal.pdf", language=Language.FR, page=43,
            chunk_index=0, dense_score=0.71, final_score=0.71,
            metadata={"doc_type": "loi", "article_ref": "103"},
        ),
        RetrievedChunk(
            chunk_id="c2", doc_id="d1",
            content="Les déchéances sont prévues pour les personnes condamnées.",
            source="code penal.pdf", language=Language.FR, page=63,
            chunk_index=1, dense_score=0.68, final_score=0.68,
            metadata={"doc_type": "loi", "article_ref": "30"},
        ),
        RetrievedChunk(
            chunk_id="c3", doc_id="d2",
            content="La République du Cameroun est un État unitaire décentralisé.",
            source="Constitution_1996.pdf", language=Language.FR, page=1,
            chunk_index=0, dense_score=0.66, final_score=0.66,
            metadata={"doc_type": "constitution"},
        ),
    ]


class TestNormalisation:

    def test_strips_extension_and_punctuation(self):
        assert _normalize_source("Code Penal.pdf") == "code penal"
        assert _normalize_source("code_penal.PDF") == "code penal"

    def test_matches_naming_variants(self):
        assert _sources_match("code penal", "code penal")
        assert _sources_match("code penal", "code penal 2016")

    def test_rejects_degenerate_short_match(self):
        """Un fragment trop court ne doit pas matcher n'importe quel document."""
        assert not _sources_match("cod", "code penal")
        assert not _sources_match("", "code penal")


class TestExtractCitations:

    def test_only_cited_chunks_become_citations(self, service, chunks):
        answer = "Le flagrant délit est défini [Source: code penal.pdf, p. 43]."
        citations, unresolved = service._extract_citations(answer, chunks)

        assert unresolved == []
        assert [c.chunk_id for c in citations] == ["c1"]

    def test_high_score_uncited_chunk_is_not_a_citation(self, service, chunks):
        """Régression : un chunk bien classé mais non cité n'appuie aucune affirmation."""
        answer = "La République du Cameroun est unitaire [Source: Constitution_1996.pdf, p. 1]."
        citations, _ = service._extract_citations(answer, chunks)

        cited_ids = {c.chunk_id for c in citations}
        assert cited_ids == {"c3"}
        assert "c1" not in cited_ids  # score 0.71 > citation_min_score, mais non cité

    def test_answer_without_citation_yields_none(self, service, chunks):
        citations, unresolved = service._extract_citations(
            "Le Code pénal réprime de nombreuses infractions.", chunks
        )
        assert citations == []
        assert unresolved == []

    def test_unknown_document_is_flagged(self, service, chunks):
        answer = "Le délai est de 30 jours [Source: code de procedure penale.pdf, art. 103]."
        citations, unresolved = service._extract_citations(answer, chunks)

        assert citations == []
        assert len(unresolved) == 1
        assert "procedure" in unresolved[0]

    def test_wrong_page_on_known_document_is_flagged(self, service, chunks):
        """Le document existe, mais la page citée n'est dans aucun extrait fourni."""
        answer = "Disposition inventée [Source: code penal.pdf, p. 999]."
        citations, unresolved = service._extract_citations(answer, chunks)

        assert citations == []
        assert unresolved == ["code penal.pdf, p. 999"]

    def test_multiple_citations_deduplicated_in_order(self, service, chunks):
        answer = (
            "Premier point [Source: code penal.pdf, p. 63]. "
            "Deuxième point [Source: code penal.pdf, p. 43]. "
            "Rappel du premier [Source: code penal.pdf, p. 63]."
        )
        citations, unresolved = service._extract_citations(answer, chunks)

        assert unresolved == []
        assert [c.chunk_id for c in citations] == ["c2", "c1"]

    def test_citation_without_page_resolves_document_chunks(self, service, chunks):
        answer = "Le texte le prévoit [Source: code penal.pdf]."
        citations, unresolved = service._extract_citations(answer, chunks)

        assert unresolved == []
        assert {c.chunk_id for c in citations} == {"c1", "c2"}

    def test_citation_carries_page_and_metadata(self, service, chunks):
        answer = "Définition [Source: code penal.pdf, art. 103, p. 43]."
        citations, _ = service._extract_citations(answer, chunks)

        citation = citations[0]
        assert citation.page == 43
        assert citation.doc_type == "loi"
        assert citation.excerpt.startswith("Le crime ou le délit")


class TestIndexAttestedPages:
    """
    Requête agrégative : le modèle cite le plan du document — « [Source: CPP,
    p. 3] » pour un titre situé page 3. La référence est exacte, mais aucun
    extrait ne l'accompagne. Sans distinction, chaque vue d'ensemble serait
    signalée comme hallucinée.
    """

    @pytest.fixture
    def index_pages(self):
        return {"code penal.pdf": {3, 7, 43, 63}}

    def test_outline_page_is_neither_citation_nor_alert(self, service, chunks, index_pages):
        answer = "Le titre III commence [Source: code penal.pdf, p. 7]."
        citations, unresolved = service._extract_citations(answer, chunks, index_pages)

        assert citations == []
        assert unresolved == []

    def test_same_reference_alerts_without_outline(self, service, chunks):
        """Hors requête agrégative, la référence reste invérifiable."""
        answer = "Le titre III commence [Source: code penal.pdf, p. 7]."
        _, unresolved = service._extract_citations(answer, chunks, None)

        assert len(unresolved) == 1

    def test_excerpt_still_resolves(self, service, chunks, index_pages):
        answer = "Le flagrant délit [Source: code penal.pdf, p. 43]."
        citations, unresolved = service._extract_citations(answer, chunks, index_pages)

        assert [c.chunk_id for c in citations] == ["c1"]
        assert unresolved == []

    def test_page_outside_index_and_excerpts_still_alerts(self, service, chunks, index_pages):
        answer = "Disposition inventée [Source: code penal.pdf, p. 999]."
        citations, unresolved = service._extract_citations(answer, chunks, index_pages)

        assert citations == []
        assert unresolved == ["code penal.pdf, p. 999"]


class TestMetadataCoercion:
    """
    Régression : une métadonnée absente arrivait sous forme de chaîne vide.

    Les couches de récupération uniformisent leurs dictionnaires en remplissant
    les champs manquants avec "". Or `doc_type` est une énumération : "" n'en
    est pas une valeur, Pydantic rejetait la citation, et la requête entière
    échouait au lieu d'afficher la réponse.
    """

    def test_empty_doc_type_becomes_none(self, service):
        chunk = RetrievedChunk(
            chunk_id="s1", doc_id="d9", content="Le loyer mensuel est de 150 000 francs.",
            source="Mon contrat de bail", language=Language.FR, page=1, chunk_index=0,
            dense_score=0.6, final_score=0.6,
            metadata={"doc_type": "", "institution": "", "jurisdiction": "", "article_ref": ""},
        )
        citation = service._chunk_to_citation(chunk)

        assert citation.doc_type is None
        assert citation.institution is None
        assert citation.jurisdiction is None
        assert citation.article is None

    def test_whitespace_only_is_treated_as_absent(self, service):
        chunk = RetrievedChunk(
            chunk_id="s2", doc_id="d9", content="Texte.", source="Pièce jointe",
            language=Language.FR, chunk_index=0, final_score=0.4,
            metadata={"doc_type": "   ", "article_ref": " "},
        )
        citation = service._chunk_to_citation(chunk)

        assert citation.doc_type is None
        assert citation.article is None

    def test_real_values_are_preserved(self, service):
        chunk = RetrievedChunk(
            chunk_id="s3", doc_id="d1", content="Article 103 : …", source="CPP",
            language=Language.FR, page=43, chunk_index=0, final_score=0.7,
            metadata={"doc_type": "loi", "article_ref": "Article 103"},
        )
        citation = service._chunk_to_citation(chunk)

        assert citation.doc_type == "loi"
        assert citation.article == "Article 103"

    def test_attached_document_yields_a_valid_citation(self, service):
        """Le cas exact qui faisait échouer la requête depuis l'interface."""
        chunk = RetrievedChunk(
            chunk_id="s4", doc_id="d9",
            content="Article 2 : Le loyer mensuel est fixé à 150 000 francs CFA.",
            source="Mon contrat de bail", language=Language.FR, page=1,
            chunk_index=1, dense_score=0.596, final_score=0.596,
            metadata={"doc_type": "", "scope": "conversation", "article_ref": ""},
        )
        answer = "Le loyer est de 150 000 francs [Source: Mon contrat de bail, p. 1]."
        citations, unresolved = service._extract_citations(answer, [chunk])

        assert unresolved == []
        assert len(citations) == 1
        assert citations[0].doc_title == "Mon contrat de bail"
