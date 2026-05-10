"""
GOV-AI 2.0 — Métriques de retrieval (§5.3 du mémoire).
Implémente : P@k, R@k, MRR, nDCG@k, Hit Rate, Reranker Gain.

Baselines B0→B4 pour l'étude ablative.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class RetrievalMetrics:
    """Métriques de retrieval agrégées sur un dataset."""
    precision_at_k: dict[int, float] = field(default_factory=dict)   # P@k
    recall_at_k: dict[int, float] = field(default_factory=dict)      # R@k
    mrr: float = 0.0                                                  # MRR
    ndcg_at_k: dict[int, float] = field(default_factory=dict)        # nDCG@k
    hit_rate_at_k: dict[int, float] = field(default_factory=dict)    # HR@k
    reranker_gain: float = 0.0                                        # Gain reranking
    num_queries: int = 0
    avg_retrieved: float = 0.0

    def to_dict(self) -> dict:
        return {
            "precision_at_k": self.precision_at_k,
            "recall_at_k": self.recall_at_k,
            "mrr": round(self.mrr, 4),
            "ndcg_at_k": self.ndcg_at_k,
            "hit_rate_at_k": self.hit_rate_at_k,
            "reranker_gain": round(self.reranker_gain, 4),
            "num_queries": self.num_queries,
            "avg_retrieved": round(self.avg_retrieved, 2),
        }


def precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    P@k = |Relevant ∩ Retrieved[:k]| / k

    Args:
        retrieved_ids: IDs des documents récupérés, dans l'ordre de ranking
        relevant_ids: IDs des documents pertinents (ground truth)
        k: seuil

    Returns:
        float ∈ [0, 1]
    """
    if k <= 0 or not retrieved_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / k


def recall_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    R@k = |Relevant ∩ Retrieved[:k]| / |Relevant|

    Returns:
        float ∈ [0, 1]
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    hits = sum(1 for doc_id in top_k if doc_id in relevant_ids)
    return hits / len(relevant_ids)


def mean_reciprocal_rank(
    retrieved_ids_list: list[list[str]],
    relevant_ids_list: list[set[str]],
) -> float:
    """
    MRR = (1/|Q|) Σ 1/rank_first_relevant

    Args:
        retrieved_ids_list: Liste des listes d'IDs récupérés (une par requête)
        relevant_ids_list: Liste des sets de pertinents (une par requête)

    Returns:
        float ∈ [0, 1]
    """
    if not retrieved_ids_list:
        return 0.0

    rr_sum = 0.0
    for retrieved, relevant in zip(retrieved_ids_list, relevant_ids_list):
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                rr_sum += 1.0 / rank
                break

    return rr_sum / len(retrieved_ids_list)


def dcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """
    DCG@k = Σ_{i=1}^{k} rel_i / log2(i+1)

    Args:
        retrieved_ids: IDs récupérés dans l'ordre de ranking
        relevance_scores: Pertinence de chaque document (0 si non pertinent)
        k: seuil

    Returns:
        float >= 0
    """
    dcg = 0.0
    for i, doc_id in enumerate(retrieved_ids[:k], start=1):
        rel = relevance_scores.get(doc_id, 0.0)
        dcg += rel / math.log2(i + 1)
    return dcg


def ndcg_at_k(
    retrieved_ids: list[str],
    relevance_scores: dict[str, float],
    k: int,
) -> float:
    """
    nDCG@k = DCG@k / IDCG@k

    IDCG = DCG du classement idéal (pertinents en tête).

    Returns:
        float ∈ [0, 1]
    """
    if not retrieved_ids or not relevance_scores:
        return 0.0

    actual_dcg = dcg_at_k(retrieved_ids, relevance_scores, k)

    # Classement idéal : pertinents triés par score décroissant
    ideal_order = sorted(relevance_scores.keys(), key=lambda d: relevance_scores[d], reverse=True)
    ideal_dcg = dcg_at_k(ideal_order, relevance_scores, k)

    return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


def hit_rate_at_k(
    retrieved_ids: list[str],
    relevant_ids: set[str],
    k: int,
) -> float:
    """
    HR@k = 1 si au moins un document pertinent dans les k premiers, 0 sinon.
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if top_k & relevant_ids else 0.0


def compute_reranker_gain(
    pre_rerank_scores: list[float],
    post_rerank_scores: list[float],
    relevant_ids: set[str],
    retrieved_ids_pre: list[str],
    retrieved_ids_post: list[str],
    k: int = 5,
) -> float:
    """
    Reranker Gain = nDCG@k(post) - nDCG@k(pre).

    Mesure l'amélioration apportée par le cross-encoder reranker.
    """
    relevance = {doc_id: 1.0 for doc_id in relevant_ids}
    pre_ndcg = ndcg_at_k(retrieved_ids_pre, relevance, k)
    post_ndcg = ndcg_at_k(retrieved_ids_post, relevance, k)
    return post_ndcg - pre_ndcg


def aggregate_retrieval_metrics(
    queries_results: list[dict],
    k_values: Optional[list[int]] = None,
) -> RetrievalMetrics:
    """
    Agrège les métriques sur l'ensemble du dataset.

    Args:
        queries_results: Liste de dicts avec keys :
            - retrieved_ids: list[str]
            - relevant_ids: set[str]
            - relevance_scores: dict[str, float]  (optionnel)
        k_values: Valeurs de k à évaluer (défaut: [1, 3, 5, 10])

    Returns:
        RetrievalMetrics agrégées
    """
    if k_values is None:
        k_values = [1, 3, 5, 10]

    n = len(queries_results)
    if n == 0:
        return RetrievalMetrics()

    p_sums = {k: 0.0 for k in k_values}
    r_sums = {k: 0.0 for k in k_values}
    ndcg_sums = {k: 0.0 for k in k_values}
    hr_sums = {k: 0.0 for k in k_values}
    mrr_sum = 0.0
    total_retrieved = 0

    retrieved_all: list[list[str]] = []
    relevant_all: list[set[str]] = []

    for result in queries_results:
        retrieved = result["retrieved_ids"]
        relevant = set(result["relevant_ids"])
        rel_scores = result.get("relevance_scores", {doc: 1.0 for doc in relevant})

        retrieved_all.append(retrieved)
        relevant_all.append(relevant)
        total_retrieved += len(retrieved)

        for k in k_values:
            p_sums[k] += precision_at_k(retrieved, relevant, k)
            r_sums[k] += recall_at_k(retrieved, relevant, k)
            ndcg_sums[k] += ndcg_at_k(retrieved, rel_scores, k)
            hr_sums[k] += hit_rate_at_k(retrieved, relevant, k)

        # MRR (contribution individuelle)
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                mrr_sum += 1.0 / rank
                break

    metrics = RetrievalMetrics(
        precision_at_k={k: round(p_sums[k] / n, 4) for k in k_values},
        recall_at_k={k: round(r_sums[k] / n, 4) for k in k_values},
        mrr=mrr_sum / n,
        ndcg_at_k={k: round(ndcg_sums[k] / n, 4) for k in k_values},
        hit_rate_at_k={k: round(hr_sums[k] / n, 4) for k in k_values},
        num_queries=n,
        avg_retrieved=total_retrieved / n,
    )
    return metrics
