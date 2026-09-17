"""
GOV-AI 2.0 — Service d'évaluation (§5 du mémoire).

Étude ablative :
  B0 — BM25 seul (Elasticsearch)
  B1 — Dense seul (mxbai-embed-large / Milvus)
  B2 — Hybride BM25 + dense, fusion RRF, sans reranking
  B3 — B2 + reranking cross-encoder (bge-reranker-v2-m3)
  B4 — Système complet : recherche B3, routage, graphe, génération

B0 à B3 comparent la recherche : elles sont mesurées sans génération. B4 mesure
le système tel qu'un usager le rencontre — récupération, qualité de la réponse,
latence de bout en bout.

Trois défauts de la version Sprint 1 sont corrigés ici, parce qu'ils faisaient
publier des chiffres faux :
  - `baseline_id` n'était qu'une étiquette : B0 à B3 exécutaient le pipeline
    complet, et l'« étude ablative » produisait cinq copies de B4.
  - Les latences de recherche et de génération étaient fixées à 60 % et 40 % du
    total, sans aucune mesure.
  - B4 annonçait une pondération par métadonnées (Éq. 4.2) absente du code.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from app.core.config import get_settings
from app.core.logging import get_logger
from app.services.evaluation.generation_metrics import (
    GenerationMetrics,
    aggregate_generation_metrics,
    citation_precision_score,
    faithfulness_score,
    keyword_recall_score,
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

#: Profondeur de classement mesurée pour B0 à B3. B4 est borné aux cinq
#: passages que l'orchestrateur transmet réellement au modèle : ses métriques
#: au-delà de k = 5 ne sont donc pas comparables à celles des autres baselines.
_ABLATION_DEPTH = 10

_STAGES = {"B0": "bm25", "B1": "dense", "B2": "rrf", "B3": "rerank"}


def _first_relevant_rank(ranked: list[str], relevant: set[str]) -> Optional[int]:
    for rank, chunk_id in enumerate(ranked, start=1):
        if chunk_id in relevant:
            return rank
    return None


@dataclass
class EvaluationReport:
    """Rapport d'évaluation d'une baseline."""
    baseline_id: str
    baseline_description: str
    #: « retrieval » (B0 à B3) ou « end_to_end » (B4). Un rapport de recherche
    #: seule n'a pas de métriques de génération : les afficher à zéro ferait
    #: lire « contrainte non respectée » là où rien n'a été mesuré.
    mode: str = "retrieval"
    retrieval: RetrievalMetrics = field(default_factory=RetrievalMetrics)
    generation: Optional[GenerationMetrics] = None
    system: SystemMetrics = field(default_factory=SystemMetrics)
    timestamp: str = ""
    dataset: str = ""
    num_queries: int = 0
    #: Détail question par question : ce qui permet de vérifier un agrégat au
    #: lieu de le croire.
    per_query: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        system: dict[str, Any] = {"error_rate": round(self.system.error_rate, 4)}
        if self.mode == "end_to_end":
            system["end_to_end"] = self.system.end_to_end_latency.to_dict()
        else:
            system["retrieval"] = self.system.retrieval_latency.to_dict()

        return {
            "baseline_id": self.baseline_id,
            "baseline_description": self.baseline_description,
            "mode": self.mode,
            "timestamp": self.timestamp,
            "dataset": self.dataset,
            "num_queries": self.num_queries,
            "retrieval": self.retrieval.to_dict(),
            "generation": self.generation.to_dict() if self.generation else None,
            "system": system,
            "constraints_met": self._check_all_constraints(),
            "per_query": self.per_query,
        }

    def _check_all_constraints(self) -> dict[str, bool]:
        # Les contraintes du mémoire portent sur la réponse et sa latence de
        # bout en bout : elles ne s'appliquent qu'au système complet.
        if self.mode != "end_to_end" or self.generation is None:
            return {}
        result: dict[str, bool] = {}
        result.update(self.generation.meets_constraints())
        result.update(self.system.meets_constraints())
        return result


class EvaluationService:
    """Évaluation de GOV-AI 2.0 sur un jeu de questions annoté."""

    BASELINES = {
        "B0": "BM25 seul (Elasticsearch)",
        "B1": "Dense seul (mxbai-embed-large / Milvus)",
        "B2": "Hybride BM25 + dense, fusion RRF, sans reranking",
        "B3": "B2 + reranking cross-encoder (bge-reranker-v2-m3)",
        "B4": "Système complet : recherche B3, routage, graphe de connaissances, génération",
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
        Évalue une baseline sur le jeu annoté et sauvegarde le rapport.

        Raises:
            FileNotFoundError: jeu de questions introuvable.
            ValueError: baseline inconnue.
        """
        if baseline_id not in self.BASELINES:
            raise ValueError(f"Baseline inconnue : {baseline_id}")
        if k_values is None:
            k_values = self.settings.eval_top_k_list

        path = Path(dataset_path or self.settings.eval_dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset introuvable : {path}")

        dataset = self._load_dataset(path)
        logger.info("evaluation_start", baseline=baseline_id, dataset=str(path), queries=len(dataset))

        if baseline_id != "B0":
            await self._ensure_dense_available()

        if baseline_id in _STAGES:
            report = await self._evaluate_retrieval_stage(dataset, baseline_id, k_values)
        else:
            report = await self._evaluate_end_to_end(dataset, k_values)

        report.timestamp = datetime.now(timezone.utc).isoformat()
        report.dataset = str(path)
        report.num_queries = len(dataset)
        self._save_report(report)

        logger.info(
            "evaluation_complete",
            baseline=baseline_id,
            mrr=round(report.retrieval.mrr, 4),
            hit_at_5=report.retrieval.hit_rate_at_k.get(5),
            keyword_recall=round(report.generation.keyword_recall, 4) if report.generation else None,
        )
        return report

    async def _ensure_dense_available(self) -> None:
        """
        Ouvre la connexion Milvus et vérifie que la recherche dense répond.

        Lancé par `docker exec`, ce service tourne hors du cycle de démarrage de
        FastAPI : Milvus n'y est pas connecté. `_dense_search` avale l'erreur et
        renvoie une liste vide — B1 obtenait alors 0 partout et B2 reproduisait
        B0 au chiffre près, sans une seule erreur signalée. Un score nul doit
        venir du système mesuré, jamais d'une connexion absente.
        """
        from app.services.cognitive_orchestrator import get_orchestrator
        from app.storage.milvus_client import connect_milvus, ensure_collection

        connect_milvus()
        ensure_collection()
        retrieval = get_orchestrator()._get_retrieval_service()
        probe = await retrieval.retrieve_stage("détention provisoire", "dense", depth=1)
        if not probe:
            raise RuntimeError(
                "Recherche dense indisponible : Milvus ne renvoie aucun résultat. "
                "Évaluation interrompue plutôt que faussée."
            )

    # ── B0 à B3 : la recherche seule ──────────────────────────────────────────

    async def _evaluate_retrieval_stage(
        self,
        dataset: list[dict],
        baseline_id: str,
        k_values: list[int],
    ) -> EvaluationReport:
        from app.services.cognitive_orchestrator import get_orchestrator

        # Le service de l'orchestrateur, non une instance neuve : il porte le
        # reranker déjà chargé, et c'est précisément lui que B4 utilisera.
        retrieval = get_orchestrator()._get_retrieval_service()
        stage = _STAGES[baseline_id]

        results: list[dict] = []
        latencies: list[float] = []
        per_query: list[dict] = []
        errors = 0

        for item in dataset:
            relevant = set(item.get("relevant_chunk_ids", []))
            record: dict[str, Any] = {
                "id": item.get("id"),
                "language": item.get("language"),
                "query": item.get("query"),
                "relevant_articles": item.get("relevant_articles", []),
            }
            t0 = time.perf_counter()
            try:
                docs = await retrieval.retrieve_stage(
                    item.get("query", ""), stage, depth=20, language=item.get("language"),
                )
                elapsed = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed)
                docs = docs[:_ABLATION_DEPTH]
                ranked = [d.get("chunk_id", "") for d in docs]
                record.update({
                    "first_relevant_rank": _first_relevant_rank(ranked, relevant),
                    "retrieved_articles": [d.get("article_ref") or None for d in docs[:5]],
                    "latency_ms": round(elapsed, 1),
                })
            except Exception as exc:
                errors += 1
                ranked = []
                record["error"] = str(exc)
                logger.error("evaluation_query_error", baseline=baseline_id, query=item.get("query", "")[:60], error=str(exc))

            results.append({"retrieved_ids": ranked, "relevant_ids": relevant})
            per_query.append(record)

        return EvaluationReport(
            baseline_id=baseline_id,
            baseline_description=self.BASELINES[baseline_id],
            mode="retrieval",
            retrieval=aggregate_retrieval_metrics(results, k_values),
            system=SystemMetrics(
                retrieval_latency=compute_latency_stats(latencies),
                error_rate=errors / len(dataset) if dataset else 0.0,
            ),
            per_query=per_query,
        )

    # ── B4 : le système complet ───────────────────────────────────────────────

    async def _evaluate_end_to_end(
        self,
        dataset: list[dict],
        k_values: list[int],
    ) -> EvaluationReport:
        from app.models.domain import Language, UserProfile
        from app.models.schemas import QueryRequest
        from app.services.cognitive_orchestrator import get_orchestrator

        orchestrator = get_orchestrator()
        retrieval_results: list[dict] = []
        generation_results: list[dict] = []
        latencies: list[float] = []
        per_query: list[dict] = []
        errors = 0

        for item in dataset:
            query = item.get("query", "")
            lang = item.get("language", "fr")
            relevant = set(item.get("relevant_chunk_ids", []))
            gold = item.get("gold_answer_keywords", [])
            record: dict[str, Any] = {
                "id": item.get("id"),
                "language": lang,
                "query": query,
                "relevant_articles": item.get("relevant_articles", []),
                "gold_answer_keywords": gold,
            }
            t0 = time.perf_counter()
            try:
                response = await orchestrator.process(QueryRequest(
                    query=query,
                    language=Language.FR if lang == "fr" else Language.EN,
                    profile=UserProfile.CITIZEN,
                    # La recherche web est coupée : l'évaluation mesure le
                    # système sur son corpus, et la disponibilité intermittente
                    # des sites publics rendrait deux passes incomparables.
                    web_search=False,
                ))
                elapsed = (time.perf_counter() - t0) * 1000
                latencies.append(elapsed)

                ranked = [c.chunk_id for c in response.retrieved_chunks]
                contexts = [c.content for c in response.retrieved_chunks]
                sources = [c.source for c in response.retrieved_chunks]
                refusal = not response.citations

                retrieval_results.append({"retrieved_ids": ranked, "relevant_ids": relevant})
                generation_results.append({
                    "answer": response.answer,
                    "context_chunks": contexts,
                    "retrieved_source_names": sources,
                    "language": lang,
                    "is_refusal": refusal,
                    "gold_keywords": gold,
                })
                record.update({
                    "first_relevant_rank": _first_relevant_rank(ranked, relevant),
                    "retrieved_articles": [c.metadata.get("article_ref") or None for c in response.retrieved_chunks],
                    "answer": response.answer,
                    "cited_articles": [c.article for c in response.citations],
                    "keyword_recall": round(keyword_recall_score(response.answer, gold), 4),
                    "faithfulness": round(faithfulness_score(response.answer, contexts), 4),
                    "citation_precision": round(citation_precision_score(response.answer, sources), 4),
                    "refusal": refusal,
                    "action_plan": response.action_plan,
                    "graph_evidence": len(response.graph_evidence),
                    "safety_flags": [f.value for f in response.safety_flags],
                    "latency_ms": round(elapsed, 1),
                })
            except Exception as exc:
                errors += 1
                # Aucune latence enregistrée pour un échec : un zéro ferait
                # baisser les percentiles d'un système qui n'a pas répondu.
                retrieval_results.append({"retrieved_ids": [], "relevant_ids": relevant})
                generation_results.append({
                    "answer": "", "context_chunks": [], "retrieved_source_names": [],
                    "language": lang, "is_refusal": True, "gold_keywords": gold,
                })
                record["error"] = str(exc)
                logger.error("evaluation_query_error", baseline="B4", query=query[:60], error=str(exc))

            per_query.append(record)
            logger.info("evaluation_query_done", id=item.get("id"), done=len(per_query), total=len(dataset))

        return EvaluationReport(
            baseline_id="B4",
            baseline_description=self.BASELINES["B4"],
            mode="end_to_end",
            retrieval=aggregate_retrieval_metrics(retrieval_results, k_values),
            generation=aggregate_generation_metrics(generation_results),
            system=SystemMetrics(
                end_to_end_latency=compute_latency_stats(latencies),
                error_rate=errors / len(dataset) if dataset else 0.0,
            ),
            per_query=per_query,
        )

    # ── Entrées / sorties ─────────────────────────────────────────────────────

    def _load_dataset(self, path: Path) -> list[dict]:
        """Charge le jeu de questions annoté."""
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "queries" in data:
            return data["queries"]
        if isinstance(data, list):
            return data
        raise ValueError(f"Format de dataset invalide : {path}")

    def _save_report(self, report: EvaluationReport) -> Path:
        """Sauvegarde le rapport JSON dans eval/reports/."""
        reports_dir = Path(self.settings.eval_reports_path)
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = reports_dir / f"eval_{report.baseline_id}_{ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, ensure_ascii=False, indent=2)
        logger.info("evaluation_report_saved", path=str(filename))
        return filename

    def save_ablation_summary(self, reports: list[EvaluationReport]) -> Path:
        """
        Tableau comparatif des baselines, sans le détail par question.

        C'est la pièce qui répond à « que vaut chaque étape du pipeline » :
        les rapports individuels restent la source, ce résumé ne fait que les
        mettre côte à côte.
        """
        rows = []
        for r in reports:
            ret = r.retrieval
            row: dict[str, Any] = {
                "baseline_id": r.baseline_id,
                "description": r.baseline_description,
                "mode": r.mode,
                "mrr": round(ret.mrr, 4),
                "hit_rate_at_k": ret.hit_rate_at_k,
                "ndcg_at_k": ret.ndcg_at_k,
                "recall_at_k": ret.recall_at_k,
                "error_rate": round(r.system.error_rate, 4),
            }
            if r.mode == "end_to_end":
                row["end_to_end_latency"] = r.system.end_to_end_latency.to_dict()
                row["generation"] = r.generation.to_dict() if r.generation else None
            else:
                row["retrieval_latency"] = r.system.retrieval_latency.to_dict()
            rows.append(row)

        reports_dir = Path(self.settings.eval_reports_path)
        reports_dir.mkdir(parents=True, exist_ok=True)
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filename = reports_dir / f"ablation_{ts}.json"
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "num_queries": reports[0].num_queries if reports else 0,
                "dataset": reports[0].dataset if reports else "",
                "baselines": rows,
            }, f, ensure_ascii=False, indent=2)
        logger.info("evaluation_ablation_saved", path=str(filename))
        return filename
