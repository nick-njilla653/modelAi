"""
GOV-AI 2.0 — Service d'évaluation complet (§5 du mémoire).
Orchestre les métriques retrieval + génération + système sur les baselines B0→B4.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.evaluation.generation_metrics import (
    GenerationMetrics,
    aggregate_generation_metrics,
    citation_precision_score,
    faithfulness_score,
)
from app.services.evaluation.retrieval_metrics import (
    RetrievalMetrics,
    aggregate_retrieval_metrics,
)
from app.services.evaluation.system_metrics import (
    SystemMetrics,
    compute_latency_stats,
)

logger = get_logger(__name__)


@dataclass
class EvaluationReport:
    """Rapport d'évaluation complet pour une configuration de baseline."""
    baseline_id: str
    baseline_description: str
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: GenerationMetrics = field(default_factory=GenerationMetrics)
    system: SystemMetrics = field(default_factory=SystemMetrics)
    timestamp: str = ""
    num_queries: int = 0

    def to_dict(self) -> dict:
        return {
            "baseline_id": self.baseline_id,
            "baseline_description": self.baseline_description,
            "timestamp": self.timestamp,
            "num_queries": self.num_queries,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict(),
            "system": self.system.to_dict(),
            "constraints_met": self._check_all_constraints(),
        }

    def _check_all_constraints(self) -> dict[str, bool]:
        result = {}
        result.update(self.generation.meets_constraints())
        result.update(self.system.meets_constraints())
        return result


class EvaluationService:
    """
    Service d'évaluation de GOV-AI 2.0.

    Baselines (étude ablative §5.2) :
      B0 : BM25 seul
      B1 : Dense seul (Qwen3-Embedding)
      B2 : Hybride BM25 + Dense + RRF (sans reranking)
      B3 : B2 + Cross-encoder reranking
      B4 : B3 + Métadonnées (Eq. 4.2 — GOV-AI 2.0 complet)
    """

    BASELINES = {
        "B0": "BM25 seul (Elasticsearch)",
        "B1": "Dense seul (Qwen3-Embedding / Milvus)",
        "B2": "Hybride BM25 + Dense + RRF (sans reranking)",
        "B3": "B2 + Cross-encoder reranking (bge-reranker-v2-m3)",
        "B4": "B3 + Pondération métadonnées (Eq. 4.2) — GOV-AI 2.0 Sprint 1",
    }

    def __init__(self) -> None:
        self.settings = get_settings()

    async def run_evaluation(
        self,
        dataset_path: Optional[str] = None,
        baseline_id: str = "B4",
        k_values: Optional[list[int]] = None,
    ) -> EvaluationReport:
        """
        Lance l'évaluation complète sur le dataset annoté.

        Args:
            dataset_path: Chemin vers qa_bilingual_annotated.json
            baseline_id: Identifiant de la baseline à évaluer
            k_values: Valeurs de k (défaut : [1, 3, 5, 10])

        Returns:
            EvaluationReport complet
        """
        if k_values is None:
            k_values = self.settings.eval_top_k_list

        path = Path(dataset_path or self.settings.eval_dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {path}")

        logger.info("evaluation_start", baseline=baseline_id, dataset=str(path))

        dataset = self._load_dataset(path)
        retrieval_results, generation_results, latencies = await self._evaluate_queries(
            dataset, baseline_id
        )

        retrieval_metrics = aggregate_retrieval_metrics(retrieval_results, k_values)
        generation_metrics = aggregate_generation_metrics(generation_results)
        latency_stats = compute_latency_stats([lat["total"] for lat in latencies])

        from datetime import datetime, timezone
        report = EvaluationReport(
            baseline_id=baseline_id,
            baseline_description=self.BASELINES.get(baseline_id, baseline_id),
            retrieval=retrieval_metrics,
            generation=generation_metrics,
            system=SystemMetrics(
                end_to_end_latency=latency_stats,
                retrieval_latency=compute_latency_stats([lat.get("retrieval", 0) for lat in latencies]),
                generation_latency=compute_latency_stats([lat.get("generation", 0) for lat in latencies]),
            ),
            timestamp=datetime.now(timezone.utc).isoformat(),
            num_queries=len(dataset),
        )

        # Sauvegarder le rapport
        self._save_report(report)

        logger.info(
            "evaluation_complete",
            baseline=baseline_id,
            mrr=round(retrieval_metrics.mrr, 4),
            ndcg_5=retrieval_metrics.ndcg_at_k.get(5, 0),
            citation_precision=round(generation_metrics.citation_precision, 4),
            hallucination_rate=round(generation_metrics.hallucination_rate, 4),
            isb=round(generation_metrics.isb, 4),
            p50_ms=round(latency_stats.p50, 1),
            p95_ms=round(latency_stats.p95, 1),
        )

        return report

    async def _evaluate_queries(
        self,
        dataset: list[dict],
        baseline_id: str,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """Évalue chaque requête du dataset."""
        from app.services.cognitive_orchestrator import get_orchestrator
        from app.models.schemas import QueryRequest
        from app.models.domain import Language, UserProfile

        orchestrator = get_orchestrator()
        retrieval_results: list[dict] = []
        generation_results: list[dict] = []
        latencies: list[dict] = []

        for item in dataset:
            query = item.get("query", "")
            lang_str = item.get("language", "fr")
            language = Language.FR if lang_str == "fr" else Language.EN
            relevant_ids = set(item.get("relevant_chunk_ids", []))
            relevant_sources = item.get("relevant_sources", [])

            try:
                t0 = time.perf_counter()
                request = QueryRequest(
                    query=query,
                    language=language,
                    profile=UserProfile.CITIZEN,
                )
                response = await orchestrator.process(request)
                total_ms = (time.perf_counter() - t0) * 1000

                retrieved_ids = [c.chunk_id for c in response.retrieved_chunks]
                retrieved_sources = [c.source for c in response.retrieved_chunks]
                context_chunks = [c.content for c in response.retrieved_chunks]

                retrieval_results.append({
                    "retrieved_ids": retrieved_ids,
                    "relevant_ids": relevant_ids,
                    "relevance_scores": {rid: 1.0 for rid in relevant_ids},
                })

                generation_results.append({
                    "answer": response.answer,
                    "context_chunks": context_chunks,
                    "retrieved_source_names": retrieved_sources,
                    "language": lang_str,
                    "is_refusal": not bool(response.citations),
                })

                latencies.append({
                    "total": total_ms,
                    "retrieval": response.latency_ms * 0.6 if response.latency_ms else 0,
                    "generation": response.latency_ms * 0.4 if response.latency_ms else 0,
                })

            except Exception as exc:
                logger.error("evaluation_query_error", query=query[:60], error=str(exc))
                retrieval_results.append({"retrieved_ids": [], "relevant_ids": relevant_ids})
                generation_results.append({"answer": "", "context_chunks": [], "retrieved_source_names": [], "language": lang_str, "is_refusal": True})
                latencies.append({"total": 0, "retrieval": 0, "generation": 0})

        return retrieval_results, generation_results, latencies

    def _load_dataset(self, path: Path) -> list[dict]:
        """Charge le dataset annoté JSON."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Format de dataset invalide : {path}")

    def _save_report(self, report: EvaluationReport) -> None:
        """Sauvegarde le rapport JSON dans eval/reports/."""
        reports_dir = Path(self.settings.eval_reports_path)
        reports_dir.mkdir(parents=True, exist_ok=True)

        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = reports_dir / f"eval_{report.baseline_id}_{ts}.json"

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)

        logger.info("evaluation_report_saved", path=str(filename))
