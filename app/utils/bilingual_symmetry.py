"""
GOV-AI 2.0 — Calcul de l'Indice de Symétrie Bilingue (ISB).
ISB = min(S_FR, S_EN) / max(S_FR, S_EN)  [Équation 2.1 du mémoire]
Seuil d'acceptabilité : ISB >= 0.85 (Contrainte C1)
"""
from __future__ import annotations

from app.core.constants import ISB_MIN_ACCEPTABLE


def compute_isb(score_fr: float, score_en: float) -> float:
    """
    Calcule l'Indice de Symétrie Bilingue.

    Args:
        score_fr: Score moyen sur les questions en français
        score_en: Score moyen sur les questions en anglais

    Returns:
        ISB in [0, 1]. ISB=1.0 = parité parfaite.
    """
    if score_fr <= 0.0 and score_en <= 0.0:
        return 1.0  # Pas de données
    if score_fr <= 0.0 or score_en <= 0.0:
        return 0.0
    return min(score_fr, score_en) / max(score_fr, score_en)


def is_bilingual_parity_acceptable(
    score_fr: float,
    score_en: float,
    threshold: float = ISB_MIN_ACCEPTABLE,
) -> bool:
    """Vérifie que la parité bilingue est acceptable (ISB >= seuil)."""
    return compute_isb(score_fr, score_en) >= threshold


def compute_bilingual_parity_gap(
    metric_scores: dict[str, dict[str, float]]
) -> dict[str, float]:
    """
    Calcule le gap de parité bilingue pour un ensemble de métriques.

    Args:
        metric_scores: {metric_name: {"fr": score, "en": score}}

    Returns:
        {metric_name: isb_value}
    """
    result = {}
    for metric, scores in metric_scores.items():
        fr_score = scores.get("fr", 0.0)
        en_score = scores.get("en", 0.0)
        result[metric] = compute_isb(fr_score, en_score)
    return result
