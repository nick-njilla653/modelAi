"""
GOV-AI 2.0 — Embedding Trainer.
Fine-tune le modèle d'embedding (mxbai-embed-large) via sentence-transformers
sur des paires (anchor, positive) extraites du corpus documentaire.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingTrainer:
    """Fine-tune un SentenceTransformer avec MultipleNegativesRankingLoss."""

    def __init__(self, base_model: str = "mixedbread-ai/mxbai-embed-large-v1"):
        self.base_model = base_model

    async def train(
        self,
        embedding_pairs_path: str | Path,
        output_dir: str | Path,
        epochs: int = 3,
        batch_size: int = 16,
        learning_rate: float = 2e-5,
        warmup_ratio: float = 0.1,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Lance l'entraînement et retourne les métriques."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool,
                self._train_sync,
                embedding_pairs_path,
                output_dir,
                epochs,
                batch_size,
                learning_rate,
                warmup_ratio,
                progress_cb,
            )
        return result

    def _train_sync(
        self,
        embedding_pairs_path,
        output_dir,
        epochs,
        batch_size,
        learning_rate,
        warmup_ratio,
        progress_cb,
    ) -> dict:
        from sentence_transformers import SentenceTransformer, InputExample, losses
        from torch.utils.data import DataLoader

        path = Path(embedding_pairs_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        examples = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                pair = json.loads(line)
                examples.append(InputExample(texts=[pair["anchor"], pair["positive"]]))

        if not examples:
            raise ValueError("Dataset d'embedding vide")

        logger.info("embedding_training_start", examples=len(examples), epochs=epochs, base=self.base_model)

        model = SentenceTransformer(self.base_model)
        loader = DataLoader(examples, shuffle=True, batch_size=batch_size, drop_last=False)
        loss_fn = losses.MultipleNegativesRankingLoss(model)
        warmup_steps = int(len(loader) * epochs * warmup_ratio)

        loss_history: list[dict] = []

        class _CB:
            def __call__(self, score, epoch, steps):
                loss_history.append({"epoch": epoch, "step": steps, "loss": round(float(score), 6)})
                if progress_cb:
                    import asyncio
                    pct = 55 + int(((epoch * len(loader) + steps) / (epochs * len(loader))) * 35)
                    try:
                        loop = asyncio.get_event_loop()
                        if loop.is_running():
                            loop.call_soon_threadsafe(
                                lambda: asyncio.ensure_future(
                                    progress_cb(min(pct, 90), f"Epoch {epoch+1}/{epochs} — loss {score:.4f}")
                                )
                            )
                    except Exception:
                        pass

        model.fit(
            train_objectives=[(loader, loss_fn)],
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=str(out),
            show_progress_bar=False,
            callback=_CB(),
        )

        logger.info("embedding_training_done", output=str(out))
        return {
            "output_path": str(out),
            "examples_trained": len(examples),
            "epochs": epochs,
            "loss_history": loss_history,
        }

    def compare_models(
        self,
        original_model_path: str,
        finetuned_model_path: str,
        test_pairs: list[dict],
    ) -> dict:
        """Compare cosine similarity des deux modèles sur les paires de test."""
        from sentence_transformers import SentenceTransformer
        import numpy as np

        orig = SentenceTransformer(original_model_path)
        ft = SentenceTransformer(finetuned_model_path)

        queries = [p["anchor"] for p in test_pairs]
        passages = [p["positive"] for p in test_pairs]

        def _mean_sim(m):
            q_emb = m.encode(queries, normalize_embeddings=True)
            p_emb = m.encode(passages, normalize_embeddings=True)
            return float(np.mean((q_emb * p_emb).sum(axis=1)))

        orig_sim = _mean_sim(orig)
        ft_sim = _mean_sim(ft)
        gain = ft_sim - orig_sim

        return {
            "original_cosine_sim": round(orig_sim, 4),
            "finetuned_cosine_sim": round(ft_sim, 4),
            "absolute_gain": round(gain, 4),
            "relative_gain_pct": round((gain / max(abs(orig_sim), 1e-6)) * 100, 2),
        }
