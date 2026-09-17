"""
Tests unitaires — Service d'évaluation et mesure d'exactitude.

Ces tests portent sur ce qui décide des chiffres présentés : une mesure qui
récompense un refus, ou un rapport qui affiche « contrainte non respectée » pour
une génération jamais mesurée, produirait un justificatif faux.
"""
import pytest

from app.services.evaluation.evaluation_service import (
    EvaluationReport,
    EvaluationService,
    _first_relevant_rank,
)
from app.services.evaluation.generation_metrics import (
    GenerationMetrics,
    aggregate_generation_metrics,
    keyword_recall_score,
)


class TestRappelDesMotsCles:

    def test_ignore_casse_et_accents(self):
        """Le modèle écrit parfois « prorogee » pour « prorogée »."""
        answer = "La DETENTION peut etre PROROGEE par ordonnance motivee."
        assert keyword_recall_score(answer, ["prorogée", "ordonnance motivée"]) == 1.0

    def test_rappel_partiel(self):
        assert keyword_recall_score("Le délai est de six (6) mois.", ["six (6) mois", "prorogée"]) == 0.5

    def test_sans_mots_cles(self):
        assert keyword_recall_score("Une réponse.", []) == 0.0

    def test_apostrophe_typographique(self):
        assert keyword_recall_score("Le juge d’instruction décide.", ["juge d'instruction"]) == 1.0


class TestAgregation:

    def test_un_refus_compte_comme_un_echec(self):
        """
        Refuser de répondre à une question dont la réponse est au corpus est un
        échec : l'exclure du rappel gonflerait le score d'un système qui se tait.
        """
        responses = [
            {"answer": "Le délai est de dix (10) jours.", "context_chunks": ["délai de dix (10) jours"],
             "retrieved_source_names": [], "language": "fr", "is_refusal": False,
             "gold_keywords": ["dix (10) jours"]},
            {"answer": "", "context_chunks": [], "retrieved_source_names": [],
             "language": "fr", "is_refusal": True, "gold_keywords": ["une année"]},
        ]
        metrics = aggregate_generation_metrics(responses)
        assert metrics.keyword_recall == pytest.approx(0.5)
        assert metrics.refusal_rate == pytest.approx(0.5)

    def test_rappel_expose_dans_le_rapport(self):
        assert "keyword_recall" in GenerationMetrics(keyword_recall=0.8).to_dict()


class TestRapport:

    def test_recherche_seule_sans_generation_ni_contraintes(self):
        """
        B0 à B3 ne génèrent rien : afficher des métriques de génération à zéro
        ferait lire « contrainte non respectée » là où rien n'a été mesuré.
        """
        data = EvaluationReport(baseline_id="B2", baseline_description="…", mode="retrieval").to_dict()
        assert data["generation"] is None
        assert data["constraints_met"] == {}
        assert "retrieval" in data["system"] and "end_to_end" not in data["system"]

    def test_bout_en_bout_expose_les_contraintes(self):
        report = EvaluationReport(
            baseline_id="B4", baseline_description="…", mode="end_to_end",
            generation=GenerationMetrics(citation_precision=1.0, faithfulness=1.0, isb=0.9),
        )
        data = report.to_dict()
        assert "end_to_end" in data["system"]
        assert "citation_precision_ge_95pct" in data["constraints_met"]


class TestRangDuPremierBonPassage:

    def test_rang(self):
        assert _first_relevant_rank(["a", "b", "c"], {"c"}) == 3

    def test_absent(self):
        assert _first_relevant_rank(["a", "b"], {"z"}) is None


class TestBaselines:

    def test_baseline_inconnue_refusee(self):
        import asyncio
        with pytest.raises(ValueError):
            asyncio.run(EvaluationService().run_evaluation(baseline_id="B9"))

    def test_chaque_baseline_de_recherche_a_son_etape(self):
        """B0 à B3 doivent mesurer une étape distincte, pas cinq fois le pipeline."""
        from app.services.evaluation.evaluation_service import _STAGES
        assert _STAGES == {"B0": "bm25", "B1": "dense", "B2": "rrf", "B3": "rerank"}
