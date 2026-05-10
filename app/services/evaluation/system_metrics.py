"""
GOV-AI 2.0 — Métriques système : latence, disponibilité, débit.
Contraintes du mémoire : p50 ≤ 5s, p95 ≤ 15s (requêtes end-to-end).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LatencyStats:
    """Statistiques de latence en millisecondes."""
    p50: float = 0.0
    p75: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    mean: float = 0.0
    min: float = 0.0
    max: float = 0.0
    count: int = 0

    # Contraintes mémoire
    P50_THRESHOLD_MS: float = 5_000.0   # 5s
    P95_THRESHOLD_MS: float = 15_000.0  # 15s

    def meets_constraints(self) -> dict[str, bool]:
        return {
            "p50_le_5s": self.p50 <= self.P50_THRESHOLD_MS,
            "p95_le_15s": self.p95 <= self.P95_THRESHOLD_MS,
        }

    def to_dict(self) -> dict:
        return {
            "p50_ms": round(self.p50, 1),
            "p75_ms": round(self.p75, 1),
            "p95_ms": round(self.p95, 1),
            "p99_ms": round(self.p99, 1),
            "mean_ms": round(self.mean, 1),
            "min_ms": round(self.min, 1),
            "max_ms": round(self.max, 1),
            "count": self.count,
            "meets_constraints": self.meets_constraints(),
        }


def compute_percentile(sorted_values: list[float], percentile: float) -> float:
    """
    Calcule le percentile d'une liste de valeurs triées.

    Args:
        sorted_values: Liste triée croissante
        percentile: Percentile souhaité (ex: 95.0 pour p95)

    Returns:
        Valeur au percentile demandé
    """
    if not sorted_values:
        return 0.0
    n = len(sorted_values)
    if n == 1:
        return sorted_values[0]
    index = (percentile / 100.0) * (n - 1)
    lower = int(math.floor(index))
    upper = int(math.ceil(index))
    if lower == upper:
        return sorted_values[lower]
    return sorted_values[lower] + (sorted_values[upper] - sorted_values[lower]) * (index - lower)


def compute_latency_stats(latencies_ms: list[float]) -> LatencyStats:
    """
    Calcule les statistiques de latence (p50, p75, p95, p99, mean, min, max).

    Args:
        latencies_ms: Liste de latences en millisecondes

    Returns:
        LatencyStats
    """
    if not latencies_ms:
        return LatencyStats()

    sorted_lat = sorted(latencies_ms)
    n = len(sorted_lat)

    return LatencyStats(
        p50=compute_percentile(sorted_lat, 50),
        p75=compute_percentile(sorted_lat, 75),
        p95=compute_percentile(sorted_lat, 95),
        p99=compute_percentile(sorted_lat, 99),
        mean=sum(sorted_lat) / n,
        min=sorted_lat[0],
        max=sorted_lat[-1],
        count=n,
    )


@dataclass
class SystemMetrics:
    """Métriques système globales."""
    end_to_end_latency: LatencyStats = field(default_factory=LatencyStats)
    retrieval_latency: LatencyStats = field(default_factory=LatencyStats)
    generation_latency: LatencyStats = field(default_factory=LatencyStats)
    reranking_latency: LatencyStats = field(default_factory=LatencyStats)
    throughput_rps: float = 0.0    # Requêtes par seconde
    error_rate: float = 0.0        # Fraction de requêtes en erreur
    availability: float = 1.0     # Uptime fraction

    def to_dict(self) -> dict:
        return {
            "end_to_end": self.end_to_end_latency.to_dict(),
            "retrieval": self.retrieval_latency.to_dict(),
            "generation": self.generation_latency.to_dict(),
            "reranking": self.reranking_latency.to_dict(),
            "throughput_rps": round(self.throughput_rps, 3),
            "error_rate": round(self.error_rate, 4),
            "availability": round(self.availability, 4),
        }

    def meets_constraints(self) -> dict[str, bool]:
        constraints = self.end_to_end_latency.meets_constraints()
        constraints["error_rate_le_1pct"] = self.error_rate <= 0.01
        return constraints
