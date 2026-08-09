"""
GOV-AI 2.0 — Leakage Detector.

Détecte et élimine le data leakage entre le dataset d'entraînement (paires QA générées
depuis les documents uploadés) et le dataset d'évaluation annoté (qa_bilingual_annotated.json).

3 niveaux de détection (appliqués en cascade) :

  Niveau 1 — Source exacte (instant)
    Si le nom de fichier source d'une paire d'entraînement apparaît dans relevant_sources
    d'une question d'évaluation → leakage direct.

  Niveau 2 — Chevauchement de mots-clés (rapide, < 1 s)
    Si des gold_answer_keywords d'une question d'évaluation sont présents dans le contexte
    de la paire d'entraînement → leakage partiel.

  Niveau 3 — Similarité sémantique (précis, quelques secondes)
    Embedding de la question d'entraînement et des questions d'évaluation ;
    si cosine similarity > seuil → leakage sémantique.

Résultat : rapport structuré + paires nettoyées (leaked_pairs exclues de l'entraînement).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.core.logging import get_logger

logger = get_logger(__name__)

# Seuils par défaut
_DEFAULT_KEYWORD_THRESHOLD  = 0.40   # ≥ 40 % des mots-clés gold présents → leakage
_DEFAULT_SEMANTIC_THRESHOLD = 0.82   # similarité cosinus ≥ 0.82 → leakage sémantique
_DEFAULT_EVAL_DATASET       = Path("eval/datasets/qa_bilingual_annotated.json")


class LeakageDetector:
    """Détecte et filtre le data leakage dans les paires d'entraînement."""

    def __init__(
        self,
        eval_dataset_path: str | Path = _DEFAULT_EVAL_DATASET,
        keyword_threshold: float = _DEFAULT_KEYWORD_THRESHOLD,
        semantic_threshold: float = _DEFAULT_SEMANTIC_THRESHOLD,
    ):
        self.eval_path = Path(eval_dataset_path)
        self.keyword_threshold = keyword_threshold
        self.semantic_threshold = semantic_threshold
        self._eval_queries: list[dict[str, Any]] | None = None

    # ── API publique ──────────────────────────────────────────────────────────

    def run(
        self,
        train_pairs: list[dict],
        use_semantic: bool = True,
    ) -> dict:
        """
        Analyse toutes les paires d'entraînement et retourne :
          - clean_pairs : paires sans leakage (à utiliser pour l'entraînement)
          - leaked_pairs : paires filtrées (exclues de l'entraînement)
          - report : détail des détections par question d'évaluation
        """
        if not self.eval_path.exists():
            logger.warning("eval_dataset_not_found", path=str(self.eval_path))
            return {
                "clean_pairs": train_pairs,
                "leaked_pairs": [],
                "report": [],
                "summary": {
                    "total_train": len(train_pairs),
                    "leaked": 0,
                    "clean": len(train_pairs),
                    "leakage_rate": 0.0,
                    "warning": "Dataset d'évaluation introuvable — vérification ignorée",
                },
            }

        eval_queries = self._load_eval_queries()
        eval_sources = self._build_source_index(eval_queries)
        eval_keywords = self._build_keyword_index(eval_queries)

        leaked_indices: set[int] = set()
        report: list[dict] = []

        for eval_q in eval_queries:
            q_id      = eval_q.get("id", "?")
            q_text    = eval_q.get("query", "")
            sources   = set(s.lower() for s in eval_q.get("relevant_sources", []))
            keywords  = [k.lower() for k in eval_q.get("gold_answer_keywords", [])]

            matched_by_source: list[int]   = []
            matched_by_keyword: list[int]  = []
            matched_by_semantic: list[int] = []

            for idx, pair in enumerate(train_pairs):
                # ── Niveau 1 : source exacte ─────────────────────────────────
                src = (pair.get("source") or "").lower()
                if src and any(src in s or s in src for s in sources):
                    leaked_indices.add(idx)
                    matched_by_source.append(idx)
                    continue

                # ── Niveau 2 : mots-clés ──────────────────────────────────────
                if keywords:
                    ctx = (pair.get("context", "") + " " + pair.get("answer", "")).lower()
                    hits = sum(1 for kw in keywords if kw in ctx)
                    ratio = hits / len(keywords)
                    if ratio >= self.keyword_threshold:
                        leaked_indices.add(idx)
                        matched_by_keyword.append(idx)

            if matched_by_source or matched_by_keyword:
                report.append({
                    "eval_query_id": q_id,
                    "eval_query": q_text,
                    "level_1_source_matches": len(matched_by_source),
                    "level_2_keyword_matches": len(matched_by_keyword),
                    "level_3_semantic_matches": 0,
                })

        # ── Niveau 3 : similarité sémantique ─────────────────────────────────
        if use_semantic and len(train_pairs) > 0:
            semantic_leaks = self._semantic_check(
                train_pairs=train_pairs,
                eval_queries=eval_queries,
                already_leaked=leaked_indices,
            )
            for idx, q_id, q_text, sim in semantic_leaks:
                leaked_indices.add(idx)
                # Ajouter au rapport
                existing = next((r for r in report if r["eval_query_id"] == q_id), None)
                if existing:
                    existing["level_3_semantic_matches"] += 1
                else:
                    report.append({
                        "eval_query_id": q_id,
                        "eval_query": q_text,
                        "level_1_source_matches": 0,
                        "level_2_keyword_matches": 0,
                        "level_3_semantic_matches": 1,
                        "max_similarity": round(sim, 4),
                    })

        clean   = [p for i, p in enumerate(train_pairs) if i not in leaked_indices]
        leaked  = [p for i, p in enumerate(train_pairs) if i in leaked_indices]
        rate    = len(leaked) / max(len(train_pairs), 1)

        logger.info(
            "leakage_detection_done",
            total=len(train_pairs),
            leaked=len(leaked),
            clean=len(clean),
            leakage_rate=round(rate, 3),
        )

        return {
            "clean_pairs": clean,
            "leaked_pairs": leaked,
            "report": sorted(report, key=lambda r: -(r["level_1_source_matches"] + r["level_2_keyword_matches"])),
            "summary": {
                "total_train": len(train_pairs),
                "leaked": len(leaked),
                "clean": len(clean),
                "leakage_rate": round(rate, 3),
                "leakage_pct": round(rate * 100, 1),
                "eval_questions_affected": len(report),
                "levels_triggered": self._levels_triggered(report),
            },
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _semantic_check(
        self,
        train_pairs: list[dict],
        eval_queries: list[dict],
        already_leaked: set[int],
    ) -> list[tuple[int, str, str, float]]:
        """Comparaison sémantique par embedding (level 3)."""
        try:
            from sentence_transformers import SentenceTransformer
            import numpy as np
        except ImportError:
            logger.debug("sentence_transformers_not_available_for_leakage_check")
            return []

        # Ne traiter que les paires pas déjà filtrées
        candidates = [(i, p) for i, p in enumerate(train_pairs) if i not in already_leaked]
        if not candidates:
            return []

        # Utilise un petit modèle rapide pour la détection (pas le modèle principal)
        try:
            model = SentenceTransformer("all-MiniLM-L6-v2")
        except Exception:
            try:
                from app.core.config import get_settings
                model = SentenceTransformer(get_settings().embedding_model)
            except Exception as exc:
                logger.debug("semantic_leakage_model_load_failed", error=str(exc))
                return []

        train_questions = [p.get("question", "") for _, p in candidates]
        eval_texts      = [q.get("query", "") for q in eval_queries]

        train_emb = model.encode(train_questions, normalize_embeddings=True, show_progress_bar=False)
        eval_emb  = model.encode(eval_texts,      normalize_embeddings=True, show_progress_bar=False)

        # Matrice de similarité (n_train × n_eval)
        sim_matrix = train_emb @ eval_emb.T

        results = []
        for local_idx, (global_idx, _) in enumerate(candidates):
            max_sim_eval_idx = int(np.argmax(sim_matrix[local_idx]))
            max_sim = float(sim_matrix[local_idx, max_sim_eval_idx])
            if max_sim >= self.semantic_threshold:
                eq = eval_queries[max_sim_eval_idx]
                results.append((global_idx, eq.get("id", "?"), eq.get("query", ""), max_sim))

        return results

    def _load_eval_queries(self) -> list[dict]:
        if self._eval_queries is None:
            data = json.loads(self.eval_path.read_text(encoding="utf-8"))
            self._eval_queries = data.get("queries", [])
        return self._eval_queries

    def _build_source_index(self, queries: list[dict]) -> set[str]:
        sources = set()
        for q in queries:
            for s in q.get("relevant_sources", []):
                sources.add(s.lower())
        return sources

    def _build_keyword_index(self, queries: list[dict]) -> dict[str, list[str]]:
        return {
            q.get("id", str(i)): [k.lower() for k in q.get("gold_answer_keywords", [])]
            for i, q in enumerate(queries)
        }

    def _levels_triggered(self, report: list[dict]) -> list[str]:
        levels = []
        if any(r["level_1_source_matches"] > 0 for r in report):
            levels.append("L1-source")
        if any(r["level_2_keyword_matches"] > 0 for r in report):
            levels.append("L2-keyword")
        if any(r.get("level_3_semantic_matches", 0) > 0 for r in report):
            levels.append("L3-semantic")
        return levels


def run_leakage_check(
    qa_pairs_path: str | Path,
    eval_dataset_path: str | Path = _DEFAULT_EVAL_DATASET,
    use_semantic: bool = True,
    keyword_threshold: float = _DEFAULT_KEYWORD_THRESHOLD,
    semantic_threshold: float = _DEFAULT_SEMANTIC_THRESHOLD,
) -> dict:
    """
    Fonction utilitaire : charge les paires depuis un fichier JSONL,
    exécute la détection et réécrit le fichier épuré.
    Retourne le rapport complet.
    """
    path = Path(qa_pairs_path)
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            try:
                pairs.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    detector = LeakageDetector(
        eval_dataset_path=eval_dataset_path,
        keyword_threshold=keyword_threshold,
        semantic_threshold=semantic_threshold,
    )
    result = detector.run(pairs, use_semantic=use_semantic)

    # Réécrire le fichier avec les paires propres uniquement
    clean = result["clean_pairs"]
    with open(path, "w", encoding="utf-8") as f:
        for pair in clean:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")

    # Sauvegarder les paires exclues pour audit
    if result["leaked_pairs"]:
        leaked_path = path.parent / "leaked_pairs_excluded.jsonl"
        with open(leaked_path, "w", encoding="utf-8") as f:
            for pair in result["leaked_pairs"]:
                f.write(json.dumps(pair, ensure_ascii=False) + "\n")
        result["leaked_pairs_saved_to"] = str(leaked_path)

    return result
