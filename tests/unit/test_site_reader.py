"""
Tests unitaires — Lecture directe des sites institutionnels.

Le lecteur remplace un moteur tiers qui ne fournissait qu'un extrait de 500
caractères. Trois propriétés le rendent utilisable :

  1. Il ne sort jamais du domaine de l'institution.
  2. Il retient le passage où la question trouve un écho, non le début de page.
  3. Il fait reculer les sommaires — une page de catégorie cite « impôts »,
     « TVA », « déclaration » des dizaines de fois sans rien expliquer.
"""
import pytest

from app.services.websearch.official_sources import SOURCES
from app.services.websearch.site_reader import (
    _keywords,
    extract_links,
    extract_text,
    select_passage,
)


@pytest.fixture
def dgi():
    return next(s for s in SOURCES if s.acronym == "DGI")


CONTENT_PAGE = """
<html><head><title>Télédéclaration | DGI</title></head>
<body>
  <nav><a href="/fr/accueil">Accueil</a><a href="/fr/contact">Contact</a></nav>
  <main>
    <p>La télédéclaration permet au contribuable de déposer sa déclaration en
    ligne. Le contribuable doit disposer d'un numéro identifiant unique et se
    connecter au portail avec ses identifiants personnels.</p>
  </main>
  <footer><a href="/fr/mentions">Mentions légales</a></footer>
</body></html>
"""

LISTING_PAGE = """
<html><head><title>Entreprises | MINFI</title></head>
<body>
  <a href="/a1">Impôts : nouvelles mesures fiscales</a>
  <a href="/a2">TVA et taxe sur les transferts</a>
  <a href="/a3">Déclaration statistique et fiscale</a>
  <a href="/a4">Impôts et taxes du secteur minier</a>
  <a href="/a5">Déclaration des contribuables</a>
</body></html>
"""


class TestExtraction:

    def test_extracts_title_and_body(self):
        title, text, _ = extract_text(CONTENT_PAGE)
        assert "Télédéclaration" in title
        assert "numéro identifiant unique" in text

    def test_strips_navigation_and_footer(self):
        """Menus et pieds de page se répètent partout et noieraient le contenu."""
        _, text, _ = extract_text(CONTENT_PAGE)
        assert "Mentions légales" not in text
        assert "Accueil" not in text

    def test_link_density_separates_content_from_listing(self):
        _, _, content_density = extract_text(CONTENT_PAGE)
        _, _, listing_density = extract_text(LISTING_PAGE)
        assert content_density < 0.45
        assert listing_density > 0.45

    def test_malformed_html_does_not_raise(self):
        title, text, density = extract_text("<html><body><p>coupé")
        assert isinstance(title, str) and isinstance(text, str)
        assert 0.0 <= density <= 1.0


class TestLinks:

    def test_keeps_only_internal_links(self, dgi):
        html = """
        <a href="/fr/teledeclaration">Télédéclarer</a>
        <a href="https://impots.cm/fr/aide">Aide</a>
        <a href="https://www.facebook.com/dgi">Facebook</a>
        <a href="https://evil.example/impots.cm">Piège</a>
        """
        urls = [u for u, _ in extract_links(html, "https://impots.cm/", dgi)]
        assert any("teledeclaration" in u for u in urls)
        assert not any("facebook" in u for u in urls)
        assert not any("evil.example" in u for u in urls)

    def test_ignores_anchors_and_schemes(self, dgi):
        html = '<a href="#haut">Haut</a><a href="mailto:x@y.cm">Écrire</a>'
        assert extract_links(html, "https://impots.cm/", dgi) == []

    def test_follows_alias_domains(self):
        """« ebulletin.minfi.cm » appartient bien au MINFI."""
        minfi = next(s for s in SOURCES if s.acronym == "MINFI")
        html = '<a href="https://ebulletin.minfi.cm/solde">Bulletin</a>'
        urls = [u for u, _ in extract_links(html, "https://minfi.gov.cm/", minfi)]
        assert urls and "ebulletin.minfi.cm" in urls[0]


class TestPassageSelection:

    def test_selects_the_window_where_the_question_echoes(self):
        """
        Une page ministérielle pèse des centaines de milliers de caractères :
        en livrer le début serait livrer le menu d'accueil.
        """
        filler = "texte sans rapport. " * 300
        text = filler + "Le taux de la TVA est fixé par la loi de finances. " + filler
        passage, score = select_passage(text, {"tva": 3, "taux": 3}, width=400)

        assert "TVA" in passage
        assert score > 0

    def test_question_words_outweigh_institution_vocabulary(self):
        """
        Sans cet écart, l'organigramme du ministère devance la réponse : ses
        pages sont saturées du vocabulaire de l'institution.
        """
        weights = _keywords("Quel est le taux de TVA ?", next(s for s in SOURCES if s.acronym == "MINFI"))
        assert weights["tva"] > weights.get("budget", 0)
        assert weights["taux"] > weights.get("douane", 0)

    def test_short_text_returned_whole(self):
        passage, _ = select_passage("Texte court sur la TVA.", {"tva": 3}, width=400)
        assert passage == "Texte court sur la TVA."

    def test_empty_text_is_safe(self):
        assert select_passage("", {"tva": 3}) == ("", 0.0)
