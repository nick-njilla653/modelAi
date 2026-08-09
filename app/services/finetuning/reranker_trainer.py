"""
GOV-AI 2.0 — Reranker Trainer.
Fine-tune le CrossEncoder BAAI/bge-reranker-v2-m3 sur des triplets
(question, passage_positif, passage_négatif) synthétisés depuis les paires QA.
"""
from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Callable, Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


class RerankerTrainer:
    """Fine-tune un CrossEncoder sentence-transformers pour le reranking."""

    def __init__(self, base_model: str = "BAAI/bge-reranker-v2-m3", device: str = "cpu"):
        self.base_model = base_model
        self.device = device

    async def train(
        self,
        qa_pairs_path: str | Path,
        output_dir: str | Path,
        epochs: int = 3,
        batch_size: int = 8,
        learning_rate: float = 2e-5,
        negatives_per_positive: int = 3,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Lance le fine-tuning du reranker dans un thread séparé."""
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool,
                self._train_sync,
                qa_pairs_path,
                output_dir,
                epochs,
                batch_size,
                learning_rate,
                negatives_per_positive,
                progress_cb,
            )
        return result

    def _train_sync(
        self,
        qa_pairs_path,
        output_dir,
        epochs,
        batch_size,
        learning_rate,
        negatives_per_positive,
        progress_cb,
    ) -> dict:
        from sentence_transformers import CrossEncoder
        from sentence_transformers.cross_encoder.evaluation import CERerankingEvaluator
        from torch.utils.data import DataLoader

        qa_path = Path(qa_pairs_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        pairs = self._load_qa_pairs(qa_path)
        if len(pairs) < 4:
            raise ValueError(f"Trop peu de paires QA pour fine-tuner le reranker ({len(pairs)} < 4)")

        # Construction des exemples d'entraînement avec labels binaires
        all_passages = [p.get("context", p.get("answer", "")) for p in pairs if p.get("context") or p.get("answer")]
        train_samples = []

        for pair in pairs:
            question = pair.get("question", "")
            positive = pair.get("context", pair.get("answer", ""))
            if not question or not positive:
                continue

            # Passage positif → label 1
            train_samples.append((question, positive, 1))

            # Passages négatifs → label 0 (passages aléatoires d'autres questions)
            negatives = random.sample(
                [p for p in all_passages if p != positive],
                min(negatives_per_positive, len(all_passages) - 1)
            )
            for neg in negatives:
                train_samples.append((question, neg, 0))

        if not train_samples:
            raise ValueError("Aucun exemple d'entraînement généré pour le reranker")

        logger.info(
            "reranker_training_start",
            samples=len(train_samples),
            positives=sum(1 for s in train_samples if s[2] == 1),
            negatives=sum(1 for s in train_samples if s[2] == 0),
            epochs=epochs,
        )

        model = CrossEncoder(
            self.base_model,
            num_labels=1,
            device=self.device,
            max_length=512,
        )

        # Formater pour sentence-transformers CrossEncoder
        from sentence_transformers import InputExample
        train_examples = [
            InputExample(texts=[q, p], label=float(label))
            for q, p, label in train_samples
        ]

        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=batch_size)
        warmup_steps = int(len(train_dataloader) * epochs * 0.1)

        model.fit(
            train_dataloader=train_dataloader,
            epochs=epochs,
            warmup_steps=warmup_steps,
            optimizer_params={"lr": learning_rate},
            output_path=str(out),
            show_progress_bar=False,
        )

        logger.info("reranker_training_done", output=str(out))
        return {
            "output_path": str(out),
            "train_samples": len(train_samples),
            "positives": sum(1 for s in train_samples if s[2] == 1),
            "negatives": sum(1 for s in train_samples if s[2] == 0),
            "epochs": epochs,
        }

    def _load_qa_pairs(self, path: Path) -> list[dict]:
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pairs
