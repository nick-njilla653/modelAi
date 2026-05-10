"""
Tests unitaires — Métriques de retrieval (app/services/evaluation/retrieval_metrics.py).
Vérifie P@k, R@k, MRR, nDCG@k, Hit Rate.
"""
import math
import pytest

from app.services.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    mean_reciprocal_rank,
    dcg_at_k,
    ndcg_at_k,
    hit_rate_at_k,
    aggregate_retrieval_metrics,
)


class TestPrecisionAtK:

    def test_all_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"a", "b", "c"}, k=3) == 1.0

    def test_none_relevant(self):
        assert precision_at_k(["a", "b", "c"], {"d", "e"}, k=3) == 0.0

    def test_partial_relevant(self):
        # 2 sur 3 premiers sont pertinents
        assert precision_at_k(["a", "b", "c"], {"a", "b"}, k=3) == pytest.approx(2/3)

    def test_k_less_than_retrieved(self):
        # k=2 : seulement les 2 premiers sont considérés
        assert precision_at_k(["a", "b", "c", "d"], {"c", "d"}, k=2) == 0.0

    def test_k_zero(self):
        assert precision_at_k(["a"], {"a"}, k=0) == 0.0

    def test_empty_retrieved(self):
        assert precision_at_k([], {"a"}, k=5) == 0.0


class TestRecallAtK:

    def test_all_relevant_retrieved(self):
        assert recall_at_k(["a", "b"], {"a", "b"}, k=2) == 1.0

    def test_none_retrieved(self):
        assert recall_at_k(["c", "d"], {"a", "b"}, k=2) == 0.0

    def test_partial_recall(self):
        assert recall_at_k(["a", "c"], {"a", "b"}, k=2) == 0.5

    def test_empty_relevant(self):
        assert recall_at_k(["a"], set(), k=1) == 0.0


class TestMeanReciprocalRank:

    def test_first_position(self):
        # Pertinent en première position → RR = 1/1 = 1.0
        mrr = mean_reciprocal_rank([["a", "b", "c"]], [{"a"}])
        assert mrr == pytest.approx(1.0)

    def test_second_position(self):
        # Pertinent en deuxième position → RR = 1/2
        mrr = mean_reciprocal_rank([["b", "a", "c"]], [{"a"}])
        assert mrr == pytest.approx(0.5)

    def test_not_found(self):
        mrr = mean_reciprocal_rank([["b", "c"]], [{"a"}])
        assert mrr == pytest.approx(0.0)

    def test_multiple_queries(self):
        # Q1 : pertinent en pos 1 → RR=1, Q2 : pertinent en pos 2 → RR=0.5
        # MRR = (1 + 0.5) / 2 = 0.75
        mrr = mean_reciprocal_rank(
            [["a", "b"], ["b", "a"]],
            [{"a"}, {"a"}],
        )
        assert mrr == pytest.approx(0.75)

    def test_empty_input(self):
        assert mean_reciprocal_rank([], []) == 0.0


class TestDCGandNDCG:

    def test_dcg_first_position(self):
        # DCG = rel_1 / log2(2) = 1.0 / 1.0 = 1.0
        rel = {"a": 1.0, "b": 0.5}
        result = dcg_at_k(["a", "b"], rel, k=2)
        expected = 1.0 / math.log2(2) + 0.5 / math.log2(3)
        assert result == pytest.approx(expected)

    def test_ndcg_perfect_order(self):
        # Ordre parfait : pertinent d'abord → nDCG = 1.0
        rel = {"a": 1.0, "b": 0.5}
        result = ndcg_at_k(["a", "b"], rel, k=2)
        assert result == pytest.approx(1.0)

    def test_ndcg_reversed_order(self):
        # Moins bon ordre → nDCG < 1.0
        rel = {"a": 1.0, "b": 0.5}
        result = ndcg_at_k(["b", "a"], rel, k=2)
        assert 0.0 < result < 1.0

    def test_ndcg_no_relevant(self):
        assert ndcg_at_k(["c", "d"], {"a": 1.0}, k=2) == 0.0

    def test_ndcg_empty(self):
        assert ndcg_at_k([], {}, k=5) == 0.0


class TestHitRate:

    def test_hit_in_top_k(self):
        assert hit_rate_at_k(["a", "b", "c"], {"b"}, k=3) == 1.0

    def test_hit_outside_top_k(self):
        assert hit_rate_at_k(["a", "b", "c"], {"c"}, k=2) == 0.0

    def test_no_hit(self):
        assert hit_rate_at_k(["a", "b"], {"z"}, k=2) == 0.0


class TestAggregateRetrievalMetrics:

    def test_aggregate_basic(self):
        queries = [
            {
                "retrieved_ids": ["a", "b", "c"],
                "relevant_ids": {"a", "b"},
                "relevance_scores": {"a": 1.0, "b": 0.5},
            },
            {
                "retrieved_ids": ["b", "a", "d"],
                "relevant_ids": {"a"},
                "relevance_scores": {"a": 1.0},
            },
        ]
        metrics = aggregate_retrieval_metrics(queries, k_values=[1, 3])
        assert 0.0 <= metrics.mrr <= 1.0
        assert 0.0 <= metrics.precision_at_k.get(3, 0) <= 1.0
        assert 0.0 <= metrics.ndcg_at_k.get(3, 0) <= 1.0
        assert metrics.num_queries == 2

    def test_aggregate_empty(self):
        metrics = aggregate_retrieval_metrics([])
        assert metrics.mrr == 0.0
        assert metrics.num_queries == 0
