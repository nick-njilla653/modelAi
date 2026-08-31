"""
Tests unitaires — Peuplement du graphe de connaissances.

Le graphe était à moitié écrit : schéma créé, lecteur implémenté, messages
d'ingestion annonçant « KG + Milvus + ES » — mais aucune requête d'écriture.
Ces tests portent sur la partie qui manquait, et surtout sur l'extraction des
renvois, seule à comporter du jugement.

L'enjeu de justesse est réel : un renvoi mal lu crée un lien entre deux
dispositions qui ne se citent pas, et le graphe se met à répondre faux avec
l'assurance d'une donnée structurée.
"""
import pytest

from app.services.knowledge_graph.graph_builder import (
    _article_number,
    _resolve_cited_text,
    _text_key,
    extract_references,
)


class TestArticleNumber:

    @pytest.mark.parametrize("ref,expected", [
        ("Article 103", "103"),
        ("Section 25-1", "25-1"),
        ("Article 5 bis", "5 bis"),
        ("Section 12", "12"),
    ])
    def test_extracts_the_number(self, ref, expected):
        assert _article_number(ref) == expected

    @pytest.mark.parametrize("ref", ["Article 104 (suite)", "Section 12 (cont.)"])
    def test_continuations_are_skipped(self, ref):
        """
        Une continuation désigne l'article précédent : en faire un nœud
        dupliquerait la disposition dans le graphe.
        """
        assert _article_number(ref) is None

    def test_empty_reference(self):
        assert _article_number("") is None


class TestReferenceExtraction:

    def test_self_header_is_not_a_reference(self):
        """« Article 100 : … » commence par sa propre référence."""
        refs = extract_references("Article 100 : Cette disposition est autonome.")
        assert refs == []

    def test_internal_reference(self):
        refs = extract_references(
            "Article 158 : conformément à l'article 157 (1), la partie consigne."
        )
        assert ("157", None) in refs

    def test_range_is_expanded(self):
        refs = extract_references(
            "Article 100 : L'inobservation des formalités prescrites aux articles 93 à 99."
        )
        numbers = [n for n, _ in refs]
        assert numbers == ["93", "94", "95", "96", "97", "98", "99"]

    def test_outgoing_reference_names_its_text(self):
        """
        C'est la distinction qui manquait à l'origine : le modèle prenait les
        renvois sortants du CPP pour des articles du CPP.
        """
        refs = extract_references(
            "Article 156 : dans les conditions prévues à l'article 152 du Code Pénal, "
            "passible des peines de l'article 169 du Code Pénal."
        )
        assert ("152", "Code Pénal") in refs
        assert ("169", "Code Pénal") in refs

    def test_english_sections_are_recognised(self):
        """Le Code pénal du corpus est anglais et dit « Section », non « article »."""
        refs = extract_references(
            "SECTION 156: matters covered by section 152 of the Penal Code."
        )
        assert ("152", "Penal Code") in refs

    def test_unknown_cited_text_is_not_invented(self):
        """
        Une capture ouverte transformait la suite de la phrase en titre de texte
        (« Code Pénal est passible des peines prévues à l'articl »). Un texte
        hors liste n'est plus rattaché plutôt que mal rattaché.
        """
        refs = extract_references("conformément à l'article 12 du règlement intérieur")
        assert refs == [("12", None)]

    def test_absurd_range_is_not_expanded(self):
        """400 renvois fictifs pollueraient le graphe."""
        refs = extract_references("les articles 10 à 900 du présent code")
        assert [n for n, _ in refs] == ["10"]

    def test_text_without_reference(self):
        assert extract_references(
            "Est qualifié flagrant le délit qui se commet actuellement."
        ) == []


class TestCitedTextResolution:
    """
    Un texte cité doit pointer vers le document réel quand il est au corpus.
    Sinon « Code Pénal » et « Penal Code » créent deux nœuds fantômes à côté du
    document véritable, et le Code pénal — qui se cite lui-même — se dédouble.
    """

    @pytest.fixture
    def corpus(self):
        return {
            "code de procedure penale loi n 2005 007": "cpp-id",
            "code penal version anglaise law no 2016 007": "penal-id",
        }

    def test_resolves_french_name(self, corpus):
        assert _resolve_cited_text("Code Pénal", corpus) == "penal-id"

    def test_resolves_english_name_to_the_same_document(self, corpus):
        """Le corpus porte le titre français ; la citation est anglaise."""
        assert _resolve_cited_text("Penal Code", corpus) == "penal-id"

    def test_unknown_text_stays_unresolved(self, corpus):
        assert _resolve_cited_text("Code Minier", corpus) is None

    def test_empty_corpus(self):
        assert _resolve_cited_text("Code Pénal", {}) is None


class TestTextKey:

    def test_key_is_stable_and_slugged(self):
        assert _text_key("Code Pénal") == _text_key("code penal")
        assert " " not in _text_key("Code de Procédure Pénale")
