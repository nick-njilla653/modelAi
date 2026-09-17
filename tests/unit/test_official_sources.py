"""
Tests unitaires — Registre des sources officielles et routage des questions.

Deux propriétés fondent le dispositif :

  1. Le filtre de confiance décide de ce qui peut être présenté comme officiel.
     Un faux positif ferait passer un site quelconque pour une source d'État.
  2. Le routage nomme l'institution compétente dans la requête — mesuré comme
     la condition pour que le moteur remonte des domaines camerounais.
"""
import pytest

from app.services.websearch.official_sources import (
    SOURCES,
    Priority,
    find_source,
    is_official,
    registry_stats,
    route_question,
)


class TestWhitelist:

    @pytest.mark.parametrize("url", [
        "https://www.impots.cm/fr/teledeclaration",
        "http://minfi.gov.cm/actualites",
        "https://prc.cm/fr/actualites/discours",
        "https://www.ohada.org/index.php/fr/actes-uniformes",
    ])
    def test_recognises_official_urls(self, url):
        assert is_official(url)

    def test_recognises_subdomains(self):
        """« ebulletin.minfi.cm » relève bien du MINFI."""
        source = find_source("https://ebulletin.minfi.cm/solde")
        assert source is not None
        assert source.acronym == "MINFI"

    @pytest.mark.parametrize("url", [
        "https://www.service-public.gouv.fr/particuliers/vosdroits/N360",
        "https://fr.wikipedia.org/wiki/Cameroun",
        "https://www.jeuneafrique.com/actualite",
        "https://minfi.gov.cm.attaquant.example/phishing",
    ])
    def test_rejects_non_official_urls(self, url):
        """Y compris un domaine forgé qui contient un domaine officiel."""
        assert not is_official(url)

    def test_availability_does_not_gate_trust(self):
        """
        Un site injoignable reste une source officielle : le moteur peut en
        servir une page indexée, et son origine ne change pas.
        """
        assert is_official("https://mintss.gov.cm/textes")
        unreachable = [s for s in SOURCES if not s.reachable]
        assert unreachable, "le relevé de disponibilité doit être renseigné"


class TestRouting:

    @pytest.mark.parametrize("question,expected", [
        ("Quel est le taux de TVA au Cameroun ?", {"MINFI", "DGI"}),
        ("Un employeur peut-il me licencier pendant mon congé ?", {"MINTSS"}),
        ("Comment obtenir un titre foncier à Douala ?", {"MINDCAF"}),
        ("Quels concours de la fonction publique sont ouverts ?", {"MINFOPRA"}),
        ("Comment créer une SARL selon l'OHADA ?", {"OHADA"}),
        ("Où puis-je me faire vacciner ?", {"MINSANTE"}),
    ])
    def test_routes_to_competent_authority(self, question, expected):
        acronyms = {s.acronym for s in route_question(question).sources}
        assert expected & acronyms, f"attendu au moins un de {expected}, obtenu {acronyms}"

    def test_accent_insensitive(self):
        """« congé » et « conge » désignent le même sujet."""
        with_accent = {s.acronym for s in route_question("licenciement pendant le congé").sources}
        without = {s.acronym for s in route_question("licenciement pendant le conge").sources}
        assert with_accent == without

    def test_unmatched_question_falls_back_to_generalists(self):
        routing = route_question("azerty qwerty question sans rapport")
        assert not routing.is_specific
        assert all(s.priority == Priority.INSTITUTION_SUPREME for s in routing.sources)

    @pytest.mark.parametrize("question,expected", [
        # Les exemples de référence du cahier des charges.
        ("Comment créer une entreprise au Cameroun ?",
         {"ServicePublic", "MINPMEESA", "eRegulations", "MINCOMMERCE"}),
        ("Combien coûte le passeport camerounais et quelles pièces fournir ?",
         {"ServicePublic", "MINREX", "DGSN"}),
        ("Comment obtenir un titre foncier à Douala ?",
         {"ServicePublic", "MINDCAF"}),
    ])
    def test_procedural_questions_include_the_service_portal(self, question, expected):
        """
        Le portail des services publics décrit pièces, coûts et guichets : il
        complète l'autorité sectorielle sur toute question de démarche.
        """
        acronyms = {s.acronym for s in route_question(question).sources}
        assert expected & acronyms == expected or "ServicePublic" in acronyms

    def test_non_procedural_question_skips_the_portal(self):
        """Une question de pur droit n'a pas besoin du portail des démarches."""
        acronyms = {s.acronym for s in route_question("Quel est le taux de TVA ?").sources}
        assert acronyms == {"MINFI", "DGI"}

    def test_action_verb_is_matched_like_the_noun(self):
        """« créer une entreprise » doit router comme « création d'entreprise »."""
        verb = {s.acronym for s in route_question("créer une entreprise").sources}
        noun = {s.acronym for s in route_question("création d'entreprise").sources}
        assert verb & noun

    def test_routing_is_bounded(self):
        """Trop d'institutions dans une requête et le moteur ne renvoie rien."""
        routing = route_question("impot travail sante foncier education transport", max_sources=3)
        assert len(routing.sources) <= 3


class TestRegistryIntegrity:

    def test_domains_are_unique(self):
        domains = [s.domain for s in SOURCES]
        assert len(domains) == len(set(domains))

    def test_every_source_declares_topics_and_search_terms(self):
        """Sans sujets une source n'est jamais routée ; sans termes, jamais trouvée."""
        for source in SOURCES:
            assert source.topics, f"{source.domain} n'a aucun sujet"
            assert source.search_terms, f"{source.domain} n'a aucun terme de recherche"

    def test_supreme_institutions_are_present(self):
        acronyms = {s.acronym for s in SOURCES if s.priority == Priority.INSTITUTION_SUPREME}
        assert {"PRC", "SPM", "ServicePublic"} <= acronyms

    def test_stats_are_consistent(self):
        stats = registry_stats()
        assert stats["total"] == len(SOURCES)
        assert stats["reachable"] <= stats["total"]
