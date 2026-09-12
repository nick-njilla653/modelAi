"""
Tests unitaires — Vocabulaire des institutions.

Le risque porte entièrement sur les faux positifs. Une institution rattachée à
tort à un article crée une compétence qui n'existe pas, et le graphe répond
« le Procureur est compétent » avec l'assurance d'une donnée structurée. Les
cas ci-dessous sont donc pour moitié des pièges : des mots qui *contiennent*
un alias sans le désigner.
"""
import pytest

from app.services.knowledge_graph.institutions import (
    INSTITUTIONS,
    find_institutions,
    get_institution,
    resolve_emitter,
)


class TestReconnaissance:

    @pytest.mark.parametrize("texte,attendu", [
        ("Le Procureur de la République requiert la force publique.", "procureur-republique"),
        ("Le juge d'instruction décerne un mandat de dépôt.", "juge-instruction"),
        ("L'affaire est portée devant la Cour d'Appel.", "cour-appel"),
        ("Le prévenu comparaît devant le Tribunal de Première Instance.", "tribunal-premiere-instance"),
    ])
    def test_denomination_francaise(self, texte, attendu):
        assert attendu in find_institutions(texte)

    @pytest.mark.parametrize("texte,attendu", [
        ("The State Counsel shall be informed.", "procureur-republique"),
        ("A Judicial Police Officer may proceed.", "officier-police-judiciaire"),
        ("The case goes to the Court of Appeal.", "cour-appel"),
        ("The Legal Department opposes the release.", "ministere-public"),
    ])
    def test_denomination_anglaise(self, texte, attendu):
        """
        Le Code pénal du corpus est la version anglaise. Sans les alias, la
        moitié du corpus ne produirait aucune institution.
        """
        assert attendu in find_institutions(texte)

    def test_sigles(self):
        assert "officier-police-judiciaire" in find_institutions("L'OPJ dresse procès-verbal.")
        assert "tribunal-criminel-special" in find_institutions("Compétence du TCS.")


class TestFauxPositifs:
    """La moitié qui protège le graphe."""

    @pytest.mark.parametrize("texte", [
        "La procuration est établie par écrit.",       # contient « procur… »
        "Tout jugement est rendu publiquement.",        # contient « juge »
        "Le recours est introduit dans les délais.",
        "copje opjx tcsx",                              # sigles noyés dans un mot
    ])
    def test_un_alias_dans_un_mot_ne_compte_pas(self, texte):
        assert find_institutions(texte) == []

    def test_le_plus_long_alias_gagne(self):
        """
        « Procureur Général » ne doit pas être imputé au Procureur de la
        République : deux organes, deux compétences.
        """
        trouve = find_institutions("Le Procureur Général près la Cour d'Appel.")
        assert "procureur-general" in trouve
        assert "procureur-republique" not in trouve


class TestOrdreEtUnicite:

    def test_ordre_de_premiere_apparition(self):
        texte = "Le greffier transmet au Procureur de la République."
        assert find_institutions(texte) == ["greffe", "procureur-republique"]

    def test_pas_de_doublon(self):
        texte = "Le procureur décide ; le Procureur de la République notifie."
        assert find_institutions(texte).count("procureur-republique") == 1


class TestEmetteur:
    """Le champ « institution » du formulaire d'ingestion est libre."""

    @pytest.mark.parametrize("saisie", [
        "Ministère de la justice",
        "MINJUSTICE",
        "Ministry of Justice",
        "  ministere de la Justice  ",
    ])
    def test_les_variantes_donnent_un_seul_noeud(self, saisie):
        emetteur = resolve_emitter(saisie)
        assert emetteur is not None and emetteur.id == "ministere-justice"

    @pytest.mark.parametrize("saisie", ["", None, "Organisation inconnue"])
    def test_saisie_non_reconnue(self, saisie):
        """Mieux vaut aucun nœud qu'un nœud faux."""
        assert resolve_emitter(saisie) is None


class TestIntegriteDuVocabulaire:

    def test_identifiants_uniques(self):
        ids = [inst.id for inst in INSTITUTIONS]
        assert len(ids) == len(set(ids))

    def test_chaque_institution_est_retrouvable(self):
        for inst in INSTITUTIONS:
            assert get_institution(inst.id) is inst

    def test_chaque_libelle_se_reconnait_lui_meme(self):
        """
        Une institution que son propre libellé ne déclenche pas est une entrée
        morte : elle ne sera jamais rattachée à un article.
        """
        for inst in INSTITUTIONS:
            assert inst.id in find_institutions(inst.label), inst.id


class TestSerialisationDuContexte:
    """
    Les deux portées ne disent pas la même chose. « cité dans 66 articles du
    corpus » et « nommée par les dispositions retrouvées » répondent à deux
    questions différentes ; les confondre ferait annoncer au modèle un
    décompte qu'il n'a pas établi.
    """

    @staticmethod
    def _contexte(**kw):
        from app.services.knowledge_graph.kg_service import KGContext
        base = {
            "institution_id": "juge-instruction",
            "label": "Juge d'instruction",
            "category": "judiciaire",
            "articles": [{"number": "218", "doc": "CPP"}],
            "article_count": 1,
            "textes_emis": [],
            "scope": "corpus",
        }
        base.update(kw)
        return KGContext(institutions=[base])

    def _rendu(self, ctx):
        from app.services.knowledge_graph.kg_service import KnowledgeGraphService
        return KnowledgeGraphService().kg_context_to_prompt_str(ctx)

    def test_portee_corpus_annonce_le_decompte(self):
        rendu = self._rendu(self._contexte(article_count=66))
        assert "66 article(s) du corpus" in rendu

    def test_portee_passages_n_annonce_aucun_decompte(self):
        rendu = self._rendu(self._contexte(scope="passages", article_count=1))
        assert "dispositions retrouvées" in rendu
        assert "article(s) du corpus" not in rendu

    def test_liste_tronquee_est_signalee(self):
        """Huit articles montrés sur soixante-six : le dire, ou induire en erreur."""
        rendu = self._rendu(self._contexte(article_count=66))
        assert "liste partielle" in rendu

    def test_contexte_vide_ne_produit_rien(self):
        from app.services.knowledge_graph.kg_service import KGContext
        assert self._rendu(KGContext()) == ""

    def test_les_institutions_seules_suffisent_a_produire_un_bloc(self):
        """
        Le rendu s'arrêtait à « pas d'articles, pas de concepts » : une
        institution trouvée seule n'aurait jamais atteint le prompt.
        """
        assert self._rendu(self._contexte()) != ""


class TestArticulationAvecLExtracteur:
    """
    Le détecteur d'institution de l'ingestion (`metadata_extractor`) tient sa
    propre liste et produit des slugs. Si les deux vocabulaires divergent,
    l'émetteur est perdu en silence : `build_document_graph` journalise
    « emitter_unknown » et n'écrit aucune arête. Ce test rend la divergence
    bruyante.
    """

    #: Volontairement non résolu : « ministère » sans plus de précision ne
    #: désigne aucun organe. Un nœud générique confondrait la Justice, la
    #: Défense et les Finances — pire qu'une absence.
    GENERIQUES = {"ministere"}

    def test_tout_slug_detecte_se_resout(self):
        from app.services.ingestion.metadata_extractor import _INSTITUTION_PATTERNS

        non_resolus = [
            slug for _, slug in _INSTITUTION_PATTERNS
            if slug not in self.GENERIQUES and resolve_emitter(slug) is None
        ]
        assert not non_resolus, f"slugs sans institution correspondante : {non_resolus}"

    def test_le_slug_generique_reste_non_resolu(self):
        """Mieux vaut « émetteur inconnu » qu'un ministère indifférencié."""
        assert resolve_emitter("ministere") is None


class TestQuestionDeCompetence:
    """
    Le routage. Sans ce détecteur, « quelle autorité peut décerner un mandat de
    détention provisoire » est classée `factual_query` et n'atteint jamais le
    graphe — la traversée inverse existait sans que rien ne l'appelle.
    """

    @pytest.mark.parametrize("question", [
        "Quelle autorité peut décerner un mandat de détention provisoire ?",
        "Qui peut ordonner une perquisition ?",
        "Quel organe est compétent en matière de flagrant délit ?",
        "Devant qui le prévenu comparaît-il ?",
        "Qui décerne le mandat d'arrêt ?",
        "Which authority may order a search?",
        "Who may issue a warrant of arrest?",
        "Before whom is the accused brought?",
    ])
    def test_question_de_competence_reconnue(self, question):
        from app.services.knowledge_graph.institutions import asks_about_authority
        assert asks_about_authority(question)

    @pytest.mark.parametrize("question", [
        "Quelles sont les conditions du flagrant délit ?",
        "Quelle est la durée de la détention provisoire ?",
        "Que dit l'article 103 ?",
        "What is the penalty for theft?",
    ])
    def test_question_ordinaire_non_reconnue(self, question):
        """
        Un déclenchement systématique ferait une requête Neo4j inutile à chaque
        question, et injecterait un bloc d'institutions sans rapport.
        """
        from app.services.knowledge_graph.institutions import asks_about_authority
        assert not asks_about_authority(question)

    def test_le_routage_couvre_les_deux_chemins(self):
        """
        Institution nommée et institution cherchée mènent au graphe par des
        voies opposées ; le routage doit reconnaître les deux.
        """
        from app.services.cognitive_orchestrator import CognitiveOrchestrator
        concerne = CognitiveOrchestrator._concerns_an_institution
        assert concerne("Quel est le rôle du Procureur de la République ?")   # nommée
        assert concerne("Qui peut ordonner une perquisition ?")               # cherchée
        assert not concerne("Quelle est la durée de la détention provisoire ?")
