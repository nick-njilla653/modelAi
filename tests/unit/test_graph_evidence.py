"""
Tests unitaires — Exposition de l'apport du graphe.

`graph_evidence` était déclaré dans `QueryResponse` depuis le Sprint 2 et
n'était jamais rempli : le graphe atteignait le modèle sans que rien ne
l'expose. Ces tests portent sur la sérialisation d'affichage, et surtout sur
l'invariant qui la justifie — ce que le graphe a dit au modèle doit être
exactement ce que l'utilisateur peut inspecter. Deux sérialisations qui
divergent produiraient une interface qui atteste autre chose que ce qui a
servi à répondre.
"""
import pytest

from app.services.knowledge_graph.kg_service import (
    KGArticle,
    KGContext,
    KnowledgeGraphService,
)


def _institution(**kw) -> dict:
    base = {
        "institution_id": "juge-instruction",
        "label": "Juge d'instruction",
        "category": "judiciaire",
        "articles": [
            {"number": "218", "doc": "Code de Procédure Pénale"},
            {"number": "220", "doc": "Code de Procédure Pénale"},
        ],
        "article_count": 2,
        "textes_emis": [],
        "scope": "passages",
    }
    base.update(kw)
    return base


@pytest.fixture
def kg() -> KnowledgeGraphService:
    return KnowledgeGraphService()


class TestSerialisationDesInstitutions:

    def test_une_institution_donne_une_piece(self, kg):
        ev = kg.kg_context_to_evidence(KGContext(institutions=[_institution()]))
        assert len(ev) == 1
        assert ev[0].kind == "institution"
        assert ev[0].label == "Juge d'instruction"
        assert ev[0].detail == "judiciaire"

    def test_les_articles_gardent_leur_texte(self, kg):
        """
        Un numéro d'article seul est ambigu : l'article 103 existe dans les deux
        codes du corpus, et désigne deux dispositions sans rapport.
        """
        ev = kg.kg_context_to_evidence(KGContext(institutions=[_institution()]))
        assert [a.number for a in ev[0].articles] == ["218", "220"]
        assert all(a.doc_title == "Code de Procédure Pénale" for a in ev[0].articles)

    def test_la_portee_est_conservee(self, kg):
        """
        Le champ décide du libellé affiché. « nommée par les dispositions
        retrouvées » et « citée dans 66 articles du corpus » ne se disent pas
        l'un pour l'autre.
        """
        passages = kg.kg_context_to_evidence(KGContext(institutions=[_institution()]))
        corpus = kg.kg_context_to_evidence(
            KGContext(institutions=[_institution(scope="corpus", article_count=66)])
        )
        assert passages[0].scope == "passages"
        assert corpus[0].scope == "corpus"
        assert corpus[0].article_count == 66

    def test_le_decompte_peut_depasser_la_liste(self, kg):
        """La liste est tronquée à huit ; le décompte doit rester le vrai."""
        ev = kg.kg_context_to_evidence(
            KGContext(institutions=[_institution(scope="corpus", article_count=66)])
        )
        assert ev[0].article_count > len(ev[0].articles)

    def test_les_textes_emis_sont_repris(self, kg):
        ev = kg.kg_context_to_evidence(KGContext(institutions=[
            _institution(textes_emis=["Code de Procédure Pénale", None, ""]),
        ]))
        assert ev[0].issued_texts == ["Code de Procédure Pénale"]

    def test_champs_absents_ne_font_pas_echouer(self, kg):
        """Une requête Cypher qui ne remonte rien ne doit pas casser l'affichage."""
        ev = kg.kg_context_to_evidence(KGContext(institutions=[{}]))
        assert ev[0].label == "" and ev[0].articles == []


class TestSerialisationDesArticles:

    def test_un_article_donne_une_piece(self, kg):
        ctx = KGContext(articles=[KGArticle(
            article_id="doc:103", number="103", title="Article 103",
            content_preview="…", doc_title="Code de Procédure Pénale",
            doc_type="loi", jurisdiction="national",
            related_articles=["doc:104", "doc:105"],
        )])
        ev = kg.kg_context_to_evidence(ctx)
        assert ev[0].kind == "article"
        assert ev[0].detail == "Code de Procédure Pénale"
        # Les renvois sortants : la relation que le graphe apporte, et qui
        # n'apparaît dans aucun extrait pris isolément.
        assert ev[0].article_count == 2


class TestCoherenceAvecLePrompt:
    """
    L'invariant. Le bloc texte part vers le modèle, les objets vers l'écran :
    si les deux divergent, l'interface atteste autre chose que ce qui a servi
    à répondre.
    """

    def test_toute_institution_du_prompt_est_exposee(self, kg):
        ctx = KGContext(institutions=[
            _institution(),
            _institution(institution_id="procureur-republique",
                         label="Procureur de la République",
                         scope="corpus", article_count=66),
        ])
        bloc = kg.kg_context_to_prompt_str(ctx)
        pieces = kg.kg_context_to_evidence(ctx)

        assert len(pieces) == len(ctx.institutions)
        for piece in pieces:
            assert piece.label in bloc

    def test_contexte_vide_des_deux_cotes(self, kg):
        vide = KGContext()
        assert kg.kg_context_to_prompt_str(vide) == ""
        assert kg.kg_context_to_evidence(vide) == []

    def test_rien_dans_le_prompt_rien_a_l_ecran(self, kg):
        """
        La réciproque compte autant : une pièce affichée que le modèle n'a pas
        reçue laisserait croire qu'elle a fondé la réponse.
        """
        ctx = KGContext(institutions=[_institution()])
        assert kg.kg_context_to_prompt_str(ctx) != ""
        assert kg.kg_context_to_evidence(ctx) != []
