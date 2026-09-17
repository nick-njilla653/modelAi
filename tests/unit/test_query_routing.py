"""
Tests unitaires — Routage des requêtes (P2).

Trois familles de requêtes appellent trois traitements différents :
  - bavardage    → réponse déterministe, aucune recherche documentaire ;
  - agrégative   → parcours structurel du corpus (le top-k n'y répond pas) ;
  - ponctuelle   → pipeline RAG hybride inchangé.

Le risque principal est le faux positif : classer « Bonjour, quelle est la durée
de la garde à vue ? » en bavardage escamoterait une vraie question.
"""
import pytest

from app.models.domain import IntentType, Language
from app.services.cognitive_orchestrator import CognitiveOrchestrator
from app.services.retrieval.corpus_outline import (
    DocumentOutline,
    _clean_heading,
    resolve_target_documents,
)


@pytest.fixture
def orchestrator():
    return CognitiveOrchestrator()


class TestChitchatDetection:

    @pytest.mark.parametrize("query", [
        "bonjour", "Bonjour !", "salut", "merci", "Merci beaucoup"[:5],
        "qui es-tu ?", "hello", "Hi", "thanks",
    ])
    def test_detects_chitchat(self, orchestrator, query):
        assert orchestrator._classify_intent(query, Language.FR) == IntentType.CHITCHAT

    @pytest.mark.parametrize("query", [
        "Bonjour, quelle est la duree de la garde a vue ?",
        "Merci de me preciser les conditions du flagrant delit",
        "Salut, comment creer une SARL selon l'OHADA ?",
    ])
    def test_greeting_prefix_is_not_chitchat(self, orchestrator, query):
        """Une salutation suivie d'une vraie question reste une vraie question."""
        assert orchestrator._classify_intent(query, Language.FR) != IntentType.CHITCHAT


class TestAggregativeDetection:

    @pytest.mark.parametrize("query", [
        "Fais moi un resume tres detaille du code penal",
        "Donne-moi une vue d'ensemble du Code de procedure penale",
        "Que contient le code penal ?",
        "Quelle est la structure du CPP ?",
        "Quel est le plan du Code penal ?",
        "De quoi parle ce code ?",
    ])
    def test_detects_aggregative_fr(self, orchestrator, query):
        assert orchestrator._classify_intent(query, Language.FR) == IntentType.AGGREGATIVE

    @pytest.mark.parametrize("query", [
        "summarize the penal code",
        "give me an overview of the penal code",
        "what does the penal code contain",
    ])
    def test_detects_aggregative_en(self, orchestrator, query):
        assert orchestrator._classify_intent(query, Language.EN) == IntentType.AGGREGATIVE

    @pytest.mark.parametrize("query", [
        "Quelles sont les conditions du flagrant delit ?",
        "Quelle est la peine encourue pour vol ?",
        "Comment creer une SARL selon l'OHADA ?",
    ])
    def test_pointed_question_is_not_aggregative(self, orchestrator, query):
        """Une question précise doit rester sur le pipeline RAG."""
        assert orchestrator._classify_intent(query, Language.FR) != IntentType.AGGREGATIVE


class TestResolveTargetDocuments:

    @pytest.fixture
    def documents(self):
        return [
            {"doc_id": "d1", "source": "Code de Procédure Pénale (Loi n° 2005/007)",
             "language": "fr", "doc_type": "loi", "chunks": 678},
            {"doc_id": "d2", "source": "Code Pénal — version anglaise (Law No. 2016/007)",
             "language": "en", "doc_type": "loi", "chunks": 548},
        ]

    def test_targets_procedure_code(self, documents):
        targets = resolve_target_documents("vue d'ensemble du code de procedure penale", documents)
        assert [d["doc_id"] for d in targets] == ["d1"]

    def test_unmatched_query_covers_whole_corpus(self, documents):
        """Sans recouvrement de noms, on ne devine pas : tout le corpus est couvert."""
        targets = resolve_target_documents("que contient le corpus", documents)
        assert len(targets) == 2

    def test_empty_corpus(self):
        assert resolve_target_documents("resume", []) == []


class TestCleanHeading:
    """Le corpus scanné porte des tampons officiels qui débordent dans le texte."""

    def test_keeps_clean_heading(self):
        assert _clean_heading("CHAPTER V ATTEMPT AND CONSPIRACY") == "CHAPTER V ATTEMPT AND CONSPIRACY"

    def test_repairs_ocr_roman_numerals(self):
        assert _clean_heading("CHAPTER lil:") == "CHAPTER III"
        assert _clean_heading("PART Ill CRIMINAL RESPONSIBILITY OF NATURAL PERSONS") == (
            "PART III CRIMINAL RESPONSIBILITY OF NATURAL PERSONS"
        )

    def test_drops_official_stamp_text(self):
        """« PRÉSIDENCE DE LA RÉPUBLIQUE » vient du tampon, pas du plan du code."""
        assert _clean_heading("CHAPTER Vil SAZLOENCE DE LA REPUBLIQUE ya") == "CHAPTER VII"
        assert _clean_heading("CHAPTER VIII CA DELA REPUBLIQUE") == "CHAPTER VIII"

    def test_truncates_at_ocr_fragment(self):
        cleaned = _clean_heading("CHAPTER I! OF FENCES AGAINST THE CONSTITULDS cea ye uaue")
        assert cleaned == "CHAPTER II OF FENCES AGAINST THE CONSTITULDS"

    def test_returns_none_without_heading(self):
        assert _clean_heading("(1) La procedure durant l'enquete est secrete.") is None


class TestOutlineNamesWhatItCannotRead:
    """
    Régression : une étiquette nue (« CHAPTER VII ») laissée sans mention pousse
    le modèle à lui inventer un titre vraisemblable, présenté ensuite comme le
    plan officiel du code. L'absence de titre doit être écrite, pas sous-entendue.
    """

    def _outline(self, headings):
        return DocumentOutline(
            doc_id="d1", source="Code X", language="fr", doc_type="loi",
            total_chunks=10, article_count=8, page_min=1, page_max=20,
            headings=headings,
        )

    def test_bare_label_is_marked_illegible(self):
        rendered = self._outline([(3, "CHAPITRE VII")]).render("fr")
        assert "[intitulé illisible dans le scan]" in rendered

    def test_titled_heading_is_left_alone(self):
        rendered = self._outline([(8, "TITRE VI DES AMENDES FORFAITAIRES")]).render("fr")
        assert "TITRE VI DES AMENDES FORFAITAIRES" in rendered
        assert "illisible" not in rendered

    def test_headings_are_numbered_and_declared_closed(self):
        """La liste est présentée comme close pour décourager les ajouts."""
        rendered = self._outline([(1, "LIVRE I DES SOURCES"), (5, "TITRE II DU JUGE")]).render("fr")
        assert "[H1]" in rendered and "[H2]" in rendered
        assert "CLOSE de 2 entrées" in rendered


class TestDocumentOutlineRendering:

    @pytest.fixture
    def outline(self):
        return DocumentOutline(
            doc_id="d1", source="Code Pénal (Law No. 2016/007)", language="en",
            doc_type="loi", total_chunks=548, article_count=515,
            first_article="Section 1", last_article="Section 372",
            page_min=1, page_max=139,
            headings=[(38, "BOOK II"), (36, "CHAPTER V ATTEMPT AND CONSPIRACY")],
        )

    def test_states_volume(self, outline):
        rendered = outline.render("en")
        assert "515 distinct provisions" in rendered
        assert "548 indexed passages" in rendered
        assert "pages 1-139" in rendered

    def test_lists_headings(self, outline):
        rendered = outline.render("en")
        assert "BOOK II" in rendered
        assert "CHAPTER V ATTEMPT AND CONSPIRACY" in rendered

    def test_french_rendering(self, outline):
        rendered = outline.render("fr")
        assert "515 dispositions distinctes" in rendered
        assert "pages 1 à 139" in rendered
