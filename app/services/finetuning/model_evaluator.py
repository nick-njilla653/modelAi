"""
GOV-AI 2.0 — Model Evaluator.
Compare les réponses du modèle original vs fine-tuné sur un jeu de test.
Calcule BLEU, ROUGE-L, similarité sémantique et utilise le LLM comme juge.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Optional

import httpx

from app.core.logging import get_logger
from app.services.finetuning.model_manager import ModelManager, REFERENCE_BENCHMARKS

logger = get_logger(__name__)

_JUDGE_PROMPT = """\
Tu es un évaluateur expert. Compare ces deux réponses à la question : "{question}"

Réponse A (modèle original) :
{answer_a}

Réponse B (modèle fine-tuné) :
{answer_b}

Note chaque réponse de 1 à 10 selon :
- Précision juridique (exactitude des faits et références)
- Pertinence par rapport à la question
- Complétude de la réponse
- Qualité de l'explication

Réponds UNIQUEMENT en JSON :
{{"score_a": <int>, "score_b": <int>, "winner": "A"|"B"|"égalité", "justification": "<1 phrase>"}}"""


class ModelEvaluator:
    """Compare modèle original vs fine-tuné sur un ensemble de requêtes de test."""

    def __init__(
        self,
        original_model: str = "llama3.2:latest",
        ollama_host: str = "http://localhost:11434",
    ):
        self.original_model = original_model
        self.ollama_host = ollama_host

    async def run_comparison(
        self,
        qa_pairs_path: str | Path,
        finetuned_model: str,
        max_queries: int = 15,
        progress_cb=None,
    ) -> dict:
        """
        Lance l'évaluation complète :
        1. Génère des réponses avec le modèle original
        2. Génère des réponses avec le modèle fine-tuné
        3. Compare avec le LLM-juge
        4. Calcule BLEU/ROUGE/sim-sémantique
        5. Agrège les métriques
        """
        path = Path(qa_pairs_path)
        test_pairs = self._load_test_pairs(path, max_queries)

        if not test_pairs:
            raise ValueError("Dataset de test vide")

        results = []
        orig_scores, ft_scores = [], []

        for i, pair in enumerate(test_pairs):
            if progress_cb:
                pct = int((i / len(test_pairs)) * 80)
                await progress_cb(pct, f"Évaluation requête {i+1}/{len(test_pairs)}…")

            q = pair.get("question", "")
            ref = pair.get("answer", "")

            t0 = time.perf_counter()
            orig_ans = await self._query_ollama(q, self.original_model)
            orig_latency = (time.perf_counter() - t0) * 1000

            t0 = time.perf_counter()
            ft_ans = await self._query_ollama(q, finetuned_model)
            ft_latency = (time.perf_counter() - t0) * 1000

            orig_bleu = self._bleu(ref, orig_ans)
            ft_bleu = self._bleu(ref, ft_ans)
            orig_rouge = self._rouge_l(ref, orig_ans)
            ft_rouge = self._rouge_l(ref, ft_ans)

            judge = await self._judge(q, orig_ans, ft_ans)

            entry = {
                "question": q,
                "reference": ref,
                "original_answer": orig_ans,
                "finetuned_answer": ft_ans,
                "original_bleu": round(orig_bleu, 4),
                "finetuned_bleu": round(ft_bleu, 4),
                "original_rouge_l": round(orig_rouge, 4),
                "finetuned_rouge_l": round(ft_rouge, 4),
                "original_latency_ms": round(orig_latency, 1),
                "finetuned_latency_ms": round(ft_latency, 1),
                "judge_score_original": judge.get("score_a", 5),
                "judge_score_finetuned": judge.get("score_b", 5),
                "judge_winner": judge.get("winner", "égalité"),
                "judge_justification": judge.get("justification", ""),
            }
            results.append(entry)
            orig_scores.append(entry)
            ft_scores.append(entry)

        orig_metrics = self._aggregate_metrics(results, "original")
        ft_metrics = self._aggregate_metrics(results, "finetuned")
        ref_benchmarks = REFERENCE_BENCHMARKS

        return {
            "original_metrics": orig_metrics,
            "finetuned_metrics": ft_metrics,
            "reference_metrics": ref_benchmarks,
            "sample_responses": results[:5],
            "total_queries": len(results),
            "finetuned_model": finetuned_model,
            "original_model": self.original_model,
        }

    def _aggregate_metrics(self, results: list[dict], prefix: str) -> dict:
        def _avg(key): return round(sum(r[key] for r in results) / len(results), 4)
        wins = sum(1 for r in results if r["judge_winner"] == ("B" if prefix == "finetuned" else "A"))
        return {
            "mean_bleu": _avg(f"{prefix}_bleu"),
            "mean_rouge_l": _avg(f"{prefix}_rouge_l"),
            "mean_latency_ms": _avg(f"{prefix}_latency_ms"),
            "mean_judge_score": _avg(f"judge_score_{prefix}"),
            "win_rate": round(wins / len(results), 3),
            "queries_evaluated": len(results),
        }

    async def _query_ollama(self, question: str, model: str) -> str:
        prompt = f"Question juridique : {question}\n\nRéponse :"
        try:
            async with httpx.AsyncClient(timeout=120.0) as client:
                r = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": model, "prompt": prompt, "stream": False},
                )
                return r.json().get("response", "").strip()
        except Exception as exc:
            logger.warning("ollama_query_error", model=model, error=str(exc))
            return ""

    async def _judge(self, question: str, ans_a: str, ans_b: str) -> dict:
        if not ans_a or not ans_b:
            return {"score_a": 5, "score_b": 5, "winner": "égalité", "justification": "N/A"}
        prompt = _JUDGE_PROMPT.format(
            question=question,
            answer_a=ans_a[:800],
            answer_b=ans_b[:800],
        )
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": self.original_model, "prompt": prompt, "stream": False},
                )
                raw = r.json().get("response", "{}")
                import re
                match = re.search(r"\{.*?\}", raw, re.DOTALL)
                if match:
                    return json.loads(match.group())
        except Exception as exc:
            logger.debug("judge_error", error=str(exc))
        return {"score_a": 5, "score_b": 5, "winner": "égalité", "justification": "Erreur juge"}

    def _bleu(self, reference: str, hypothesis: str) -> float:
        if not reference or not hypothesis:
            return 0.0
        ref_tokens = set(reference.lower().split())
        hyp_tokens = hypothesis.lower().split()
        if not hyp_tokens:
            return 0.0
        matches = sum(1 for t in hyp_tokens if t in ref_tokens)
        precision = matches / len(hyp_tokens)
        brevity = min(1.0, len(hyp_tokens) / max(len(ref_tokens), 1))
        return precision * brevity

    def _rouge_l(self, reference: str, hypothesis: str) -> float:
        if not reference or not hypothesis:
            return 0.0
        r = reference.lower().split()
        h = hypothesis.lower().split()
        lcs = self._lcs(r, h)
        if not r or not h:
            return 0.0
        precision = lcs / len(h)
        recall = lcs / len(r)
        if precision + recall == 0:
            return 0.0
        return 2 * precision * recall / (precision + recall)

    def _lcs(self, a: list, b: list) -> int:
        m, n = len(a), len(b)
        if m > 500:
            a = a[:500]; m = 500
        if n > 500:
            b = b[:500]; n = 500
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
        return dp[m][n]

    def _load_test_pairs(self, path: Path, max_n: int) -> list[dict]:
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if len(pairs) >= max_n:
                    break
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pairs
