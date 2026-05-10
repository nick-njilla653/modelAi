"""
Tests unitaires — RRF Fusion (app/services/retrieval/rrf_fusion.py).
Vérifie l'implémentation de l'Équation 3.2 du mémoire.
"""
import pytest

from app.services.retrieval.rrf_fusion import reciprocal_rank_fusion, merge_results_with_rrf


# ── Données de test ────────────────────────────────────────────────────────────

DENSE_RESULTS = [
    {"chunk_id": "c1", "content": "Article 1", "source": "doc1.pdf", "dense_score": 0.95},
    {"chunk_id": "c2", "content": "Article 2", "source": "doc1.pdf", "dense_score": 0.88},
    {"chunk_id": "c3", "content": "Article 3", "source": "doc2.pdf", "dense_score": 0.75},
    {"chunk_id": "c4", "content": "Article 4", "source": "doc2.pdf", "dense_score": 0.60},
]

SPARSE_RESULTS = [
    {"chunk_id": "c2", "content": "Article 2", "source": "doc1.pdf", "sparse_score": 0.90},
    {"chunk_id": "c5", "content": "Article 5", "source": "doc3.pdf", "sparse_score": 0.82},
    {"chunk_id": "c1", "content": "Article 1", "source": "doc1.pdf", "sparse_score": 0.77},
    {"chunk_id": "c3", "content": "Article 3", "source": "doc2.pdf", "sparse_score": 0.55},
]


class TestReciprocalRankFusion:

    def test_basic_rrf_formula(self):
        """RRF(d) = Σ 1/(k + rank_i), k=60 par défaut."""
        results = reciprocal_rank_fusion(
            ranked_lists=[DENSE_RESULTS, SPARSE_RESULTS],
            id_key="chunk_id",
            k=60,
        )
        assert len(results) > 0
        # Tous les résultats doivent avoir un rrf_score
        for r in results:
            assert "rrf_score" in r
            assert r["rrf_score"] > 0

    def test_rrf_ordered_by_score_descending(self):
        """Les résultats doivent être triés par rrf_score décroissant."""
        results = reciprocal_rank_fusion(
            ranked_lists=[DENSE_RESULTS, SPARSE_RESULTS],
            id_key="chunk_id",
            k=60,
        )
        scores = [r["rrf_score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_consensus_chunk_ranked_higher(self):
        """c1 et c2 apparaissent dans les deux listes → score plus élevé."""
        results = reciprocal_rank_fusion(
            ranked_lists=[DENSE_RESULTS, SPARSE_RESULTS],
            id_key="chunk_id",
            k=60,
        )
        ids_in_order = [r["chunk_id"] for r in results]
        # c1 et c2 sont dans les 2 premiers
        assert "c1" in ids_in_order[:3]
        assert "c2" in ids_in_order[:3]

    def test_empty_lists(self):
        results = reciprocal_rank_fusion(ranked_lists=[], id_key="chunk_id")
        assert results == []

    def test_single_list(self):
        results = reciprocal_rank_fusion(
            ranked_lists=[DENSE_RESULTS],
            id_key="chunk_id",
            k=60,
        )
        assert len(results) == len(DENSE_RESULTS)

    def test_k_parameter_effect(self):
        """k plus grand = scores plus lisses (moins de différenciation)."""
        results_k1 = reciprocal_rank_fusion([DENSE_RESULTS, SPARSE_RESULTS], "chunk_id", k=1)
        results_k60 = reciprocal_rank_fusion([DENSE_RESULTS, SPARSE_RESULTS], "chunk_id", k=60)
        # Avec k=1, la variance des scores doit être plus élevée
        scores_k1 = [r["rrf_score"] for r in results_k1]
        scores_k60 = [r["rrf_score"] for r in results_k60]
        var_k1 = sum((s - sum(scores_k1)/len(scores_k1))**2 for s in scores_k1)
        var_k60 = sum((s - sum(scores_k60)/len(scores_k60))**2 for s in scores_k60)
        assert var_k1 >= var_k60

    def test_rrf_exact_formula(self):
        """Vérification exacte de la formule pour un cas simple."""
        list1 = [{"chunk_id": "A"}, {"chunk_id": "B"}]
        list2 = [{"chunk_id": "B"}, {"chunk_id": "A"}]
        k = 60
        results = reciprocal_rank_fusion([list1, list2], "chunk_id", k=k)
        result_map = {r["chunk_id"]: r["rrf_score"] for r in results}

        # RRF(A) = 1/(60+1) + 1/(60+2) = 1/61 + 1/62
        expected_a = 1/61 + 1/62
        # RRF(B) = 1/(60+2) + 1/(60+1) = même valeur
        expected_b = 1/62 + 1/61
        assert abs(result_map["A"] - expected_a) < 1e-8
        assert abs(result_map["B"] - expected_b) < 1e-8


class TestMergeResultsWithRRF:

    def test_merge_preserves_all_fields(self):
        results = merge_results_with_rrf(
            dense_results=DENSE_RESULTS,
            sparse_results=SPARSE_RESULTS,
            k=60,
        )
        # Les champs originaux doivent être préservés
        for r in results:
            assert "chunk_id" in r
            assert "content" in r
            assert "source" in r
            assert "rrf_score" in r

    def test_merge_union_of_both_lists(self):
        results = merge_results_with_rrf(
            dense_results=DENSE_RESULTS,
            sparse_results=SPARSE_RESULTS,
            k=60,
        )
        result_ids = {r["chunk_id"] for r in results}
        dense_ids = {r["chunk_id"] for r in DENSE_RESULTS}
        sparse_ids = {r["chunk_id"] for r in SPARSE_RESULTS}
        assert result_ids == dense_ids | sparse_ids

    def test_individual_scores_preserved(self):
        results = merge_results_with_rrf(
            dense_results=DENSE_RESULTS,
            sparse_results=SPARSE_RESULTS,
            k=60,
        )
        # c1 est dans dense (dense_score=0.95) et sparse (sparse_score=0.77)
        c1 = next(r for r in results if r["chunk_id"] == "c1")
        assert "dense_score" in c1
        assert "sparse_score" in c1
