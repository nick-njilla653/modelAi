"""
GOV-AI 2.0 — Métriques de génération (§5.4 du mémoire).
Implémente : Faithfulness, Citation Precision, Hallucination Rate, ISB.

Métriques clés (contraintes du mémoire) :
  - Citation Precision ≥ 95%
  - Hallucination Rate ≤ 5%
  - ISB ≥ 0.85
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.utils.bilingual_symmetry import compute_isb


@dataclass
class GenerationMetrics:
    """Métriques de qualité de génération agrégées."""
    faithfulness: float = 0.0          # Fraction assertions supportées par corpus
    citation_precision: float = 0.0   # Fraction citations correctement formatées et vérifiables
    hallucination_rate: float = 0.0   # Fraction assertions non ancrées dans le contexte
    isb: float = 0.0                  # Indice de Symétrie Bilingue
    avg_citations_per_response: float = 0.0
    avg_response_length: float = 0.0
    refusal_rate: float = 0.0
    num_responses: int = 0

    def to_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "citation_precision": round(self.citation_precision, 4),
            "hallucination_rate": round(self.hallucination_rate, 4),
            "isb": round(self.isb, 4),
            "avg_citations_per_response": round(self.avg_citations_per_response, 2),
            "avg_response_length": round(self.avg_response_length, 1),
            "refusal_rate": round(self.refusal_rate, 4),
            "num_responses": self.num_responses,
            "meets_constraints": self.meets_constraints(),
        }

    def meets_constraints(self) -> dict[str, bool]:
        """Vérifie les contraintes du mémoire (C4-C6)."""
        return {
            "citation_precision_ge_95pct": self.citation_precision >= 0.95,
            "hallucination_rate_le_5pct": self.hallucination_rate <= 0.05,
            "isb_ge_85pct": self.isb >= 0.85,
        }


# ── Métriques individuelles ───────────────────────────────────────────────────

_CITATION_PATTERN = re.compile(
    r"\[(?:Source|Src|Ref)\s*:\s*[^\]]+\]",
    re.IGNORECASE,
)


def count_citations_in_response(answer: str) -> int:
    """Compte le nombre de citations [Source: ...] dans la réponse."""
    return len(_CITATION_PATTERN.findall(answer))


def citation_precision_score(
    answer: str,
    retrieved_source_names: list[str],
) -> float:
    """
    Citation Precision = citations_vérifiables / total_citations

    Une citation est "vérifiable" si le nom de source est dans les chunks récupérés.

    Args:
        answer: Réponse générée par le LLM
        retrieved_source_names: Noms des sources des chunks récupérés

    Returns:
        float ∈ [0, 1]
    """
    cited_sources = _CITATION_PATTERN.findall(answer)
    if not cited_sources:
        return 0.0  # Pas de citations = précision 0 (problème)

    sources_lower = {s.lower() for s in retrieved_source_names}
    verified = 0

    for citation in cited_sources:
        # Extraire le nom de source de [Source: nom, p. X]
        inner = citation.strip("[]")
        parts = inner.split(":", 1)
        if len(parts) < 2:
            continue
        source_name = parts[1].split(",")[0].strip().lower()
        if any(source_name in s or s in source_name for s in sources_lower):
            verified += 1

    return verified / len(cited_sources)


def faithfulness_score(
    answer: str,
    context_chunks: list[str],
    min_overlap_ratio: float = 0.3,
) -> float:
    """
    Faithfulness = fraction de phrases de la réponse ancrées dans le contexte.

    Méthode légère (sans LLM juge) : overlap de n-grammes entre chaque phrase
    et l'ensemble du contexte.

    Args:
        answer: Réponse générée
        context_chunks: Textes des chunks récupérés
        min_overlap_ratio: Seuil d'overlap minimal pour considérer une phrase ancrée

    Returns:
        float ∈ [0, 1]
    """
    # Splitter la réponse en phrases (simpliste pour MVP)
    sentences = [s.strip() for s in re.split(r"[.!?]\s+", answer) if len(s.strip()) > 20]
    if not sentences:
        return 1.0  # Réponse vide = pas de hallucination

    context_text = " ".join(context_chunks).lower()
    context_bigrams = _get_ngrams(context_text, n=2)

    anchored = 0
    for sentence in sentences:
        sentence_bigrams = _get_ngrams(sentence.lower(), n=2)
        if not sentence_bigrams:
            continue
        overlap = len(sentence_bigrams & context_bigrams) / len(sentence_bigrams)
        if overlap >= min_overlap_ratio:
            anchored += 1

    return anchored / len(sentences)


def hallucination_rate(faithfulness: float) -> float:
    """Hallucination Rate = 1 - Faithfulness."""
    return max(0.0, 1.0 - faithfulness)


def compute_isb_score(
    score_fr: float,
    score_en: float,
) -> float:
    """
    ISB = min(S_FR, S_EN) / max(S_FR, S_EN)
    Équation 2.1 du mémoire. Plage [0, 1].
    """
    return compute_isb(score_fr, score_en)


def _get_ngrams(text: str, n: int = 2) -> set[tuple[str, ...]]:
    """Génère les n-grammes d'un texte tokenisé."""
    tokens = text.split()
    return set(tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1))


# ── Agrégation ────────────────────────────────────────────────────────────────

def aggregate_generation_metrics(
    responses: list[dict],
) -> GenerationMetrics:
    """
    Agrège les métriques de génération sur un dataset.

    Args:
        responses: Liste de dicts avec keys :
            - answer: str
            - context_chunks: list[str] (textes des chunks récupérés)
            - retrieved_source_names: list[str]
            - language: "fr" | "en"
            - is_refusal: bool (optionnel)

    Returns:
        GenerationMetrics agrégées
    """
    n = len(responses)
    if n == 0:
        return GenerationMetrics()

    total_faithfulness = 0.0
    total_citation_precision = 0.0
    total_citations = 0
    total_length = 0
    total_refusals = 0

    fr_scores: list[float] = []
    en_scores: list[float] = []

    for resp in responses:
        answer = resp.get("answer", "")
        chunks = resp.get("context_chunks", [])
        sources = resp.get("retrieved_source_names", [])
        lang = resp.get("language", "fr")
        is_refusal = resp.get("is_refusal", False)

        if is_refusal:
            total_refusals += 1
            continue

        faith = faithfulness_score(answer, chunks)
        cit_prec = citation_precision_score(answer, sources)

        total_faithfulness += faith
        total_citation_precision += cit_prec
        total_citations += count_citations_in_response(answer)
        total_length += len(answer)

        # ISB : collecter score par langue
        if lang == "fr":
            fr_scores.append(faith)
        elif lang == "en":
            en_scores.append(faith)

    non_refusal = n - total_refusals
    if non_refusal == 0:
        return GenerationMetrics(refusal_rate=1.0, num_responses=n)

    avg_faith = total_faithfulness / non_refusal
    avg_cit = total_citation_precision / non_refusal

    # ISB sur faithfulness FR vs EN
    avg_fr = sum(fr_scores) / len(fr_scores) if fr_scores else 0.0
    avg_en = sum(en_scores) / len(en_scores) if en_scores else 0.0
    isb = compute_isb(avg_fr, avg_en) if (avg_fr > 0 and avg_en > 0) else 0.0

    return GenerationMetrics(
        faithfulness=avg_faith,
        citation_precision=avg_cit,
        hallucination_rate=hallucination_rate(avg_faith),
        isb=isb,
        avg_citations_per_response=total_citations / non_refusal,
        avg_response_length=total_length / non_refusal,
        refusal_rate=total_refusals / n,
        num_responses=n,
    )
