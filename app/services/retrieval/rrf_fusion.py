"""
GOV-AI 2.0 — Reciprocal Rank Fusion (RRF).
Fusionne les résultats de la recherche dense et sparse.

RRF(d) = Σ_{r ∈ R} 1 / (k_RRF + rank_r(d))   [Équation 3.2 du mémoire]
k_RRF = 60 (valeur standard)
"""
from __future__ import annotations

from app.core.config import get_settings


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    id_field: str = "chunk_id",
    k: int = 60,
) -> dict[str, float]:
    """
    Calcule le score RRF pour chaque document à partir de plusieurs listes de résultats.

    Args:
        result_lists: Liste de listes de résultats (chaque liste = un retriever)
        id_field: Champ servant d'identifiant unique
        k: Paramètre RRF (défaut 60)

    Returns:
        Dict {chunk_id: rrf_score}
    """
    rrf_scores: dict[str, float] = {}

    for result_list in result_lists:
        for rank, doc in enumerate(result_list, start=1):
            doc_id = doc.get(id_field, "")
            if not doc_id:
                continue
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k + rank)

    return rrf_scores


def merge_results_with_rrf(
    dense_results: list[dict],
    sparse_results: list[dict],
    top_k: int = 20,
    k: int = 60,
) -> list[dict]:
    """
    Fusionne dense + sparse via RRF, retourne les top_k résultats avec score RRF.

    Args:
        dense_results: Résultats du retrieval vectoriel (triés par score desc)
        sparse_results: Résultats BM25 (triés par score desc)
        top_k: Nombre de résultats à retourner après fusion
        k: Paramètre RRF

    Returns:
        Liste de docs fusionnés, triés par score RRF desc, avec scores individuels
    """
    # Index par chunk_id pour enrichir les docs
    all_docs: dict[str, dict] = {}

    for doc in dense_results:
        cid = doc.get("chunk_id", "")
        if cid:
            all_docs[cid] = {**doc}

    for doc in sparse_results:
        cid = doc.get("chunk_id", "")
        if cid:
            if cid in all_docs:
                all_docs[cid]["sparse_score"] = doc.get("sparse_score", 0.0)
            else:
                all_docs[cid] = {**doc}

    # Calcul RRF
    rrf_scores = reciprocal_rank_fusion(
        [dense_results, sparse_results],
        id_field="chunk_id",
        k=k,
    )

    # Enrichir chaque doc avec son score RRF
    for cid, score in rrf_scores.items():
        if cid in all_docs:
            all_docs[cid]["rrf_score"] = score

    # Trier par score RRF décroissant
    sorted_docs = sorted(
        all_docs.values(),
        key=lambda d: d.get("rrf_score", 0.0),
        reverse=True,
    )

    return sorted_docs[:top_k]
