"""
GOV-AI 2.0 — LLM Trainer.
Deux modes :
  1. Modelfile (défaut, sans GPU) : crée un modèle Ollama enrichi avec un system prompt
     extrait du corpus et des exemples QA annotés.
  2. LoRA (si torch + CUDA disponibles) : fine-tuning LoRA via HuggingFace PEFT/TRL,
     export GGUF puis enregistrement Ollama.
"""
from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Callable, Optional

import httpx

from app.core.logging import get_logger

logger = get_logger(__name__)

_SYSTEM_TEMPLATE = """\
Tu es GOV-AI, un assistant juridique et administratif expert spécialisé dans le droit \
camerounais (droit civil francophone et common law anglophone). \
Tu réponds en français ou en anglais selon la langue de l'utilisateur. \
Tu bases toujours tes réponses sur des textes juridiques officiels camerounais. \
Tu cites les articles et textes sources avec précision. \
Tu n'inventes jamais de faits juridiques.

=== CONNAISSANCES DOMAINE (extraites du corpus) ===
{domain_knowledge}

=== EXEMPLES DE QUESTIONS / RÉPONSES TYPES ===
{qa_examples}
"""


class LLMTrainer:
    """Adapte le modèle LLM (Ollama) au domaine via Modelfile ou LoRA."""

    def __init__(
        self,
        base_model: str = "llama3.2:latest",
        ollama_host: str = "http://localhost:11434",
    ):
        self.base_model = base_model
        self.ollama_host = ollama_host

    # ── Mode 1 : Modelfile (CPU-friendly) ────────────────────────────────────

    async def create_modelfile_model(
        self,
        qa_pairs_path: str | Path,
        output_dir: str | Path,
        model_name: str,
        temperature: float = 0.05,
        num_ctx: int = 8192,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Génère un Modelfile enrichi et l'enregistre dans Ollama."""
        qa_path = Path(qa_pairs_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if progress_cb:
            await progress_cb(60, "Lecture des paires QA pour le Modelfile…")

        pairs = self._load_qa_pairs(qa_path, max_pairs=40)
        domain_snippets = self._extract_domain_knowledge(pairs, max_chars=3000)
        qa_examples = self._format_qa_examples(pairs[:15])

        system_prompt = _SYSTEM_TEMPLATE.format(
            domain_knowledge=domain_snippets,
            qa_examples=qa_examples,
        )

        modelfile_content = (
            f"FROM {self.base_model}\n\n"
            f"SYSTEM \"\"\"\n{system_prompt}\n\"\"\"\n\n"
            f"PARAMETER temperature {temperature}\n"
            f"PARAMETER num_ctx {num_ctx}\n"
            f"PARAMETER num_predict 2048\n"
        )

        modelfile_path = out / "Modelfile"
        modelfile_path.write_text(modelfile_content, encoding="utf-8")

        if progress_cb:
            await progress_cb(75, f"Création du modèle Ollama '{model_name}'…")

        await self._register_ollama_model(model_name, modelfile_path)

        if progress_cb:
            await progress_cb(92, f"Modèle '{model_name}' enregistré dans Ollama.")

        logger.info("ollama_model_created", name=model_name, base=self.base_model)
        return {
            "mode": "modelfile",
            "model_name": model_name,
            "modelfile_path": str(modelfile_path),
            "qa_examples_used": len(pairs[:15]),
            "temperature": temperature,
            "num_ctx": num_ctx,
        }

    # ── Mode 2 : LoRA (nécessite torch + GPU) ────────────────────────────────

    async def train_lora(
        self,
        qa_pairs_path: str | Path,
        output_dir: str | Path,
        model_name: str,
        epochs: int = 3,
        learning_rate: float = 2e-4,
        lora_r: int = 16,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        """Fine-tuning LoRA. Nécessite torch + transformers + peft + trl."""
        self._check_lora_deps()
        import asyncio
        from concurrent.futures import ThreadPoolExecutor

        loop = asyncio.get_event_loop()
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = await loop.run_in_executor(
                pool,
                self._train_lora_sync,
                qa_pairs_path,
                output_dir,
                model_name,
                epochs,
                learning_rate,
                lora_r,
                progress_cb,
            )
        return result

    def _train_lora_sync(self, qa_pairs_path, output_dir, model_name, epochs, lr, lora_r, progress_cb):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model, TaskType
        from trl import SFTTrainer, SFTConfig
        from datasets import Dataset

        qa_path = Path(qa_pairs_path)
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        hf_model_id = self._ollama_to_hf(self.base_model)
        pairs = self._load_qa_pairs(qa_path)

        dataset = Dataset.from_list([
            {"text": f"Question: {p['question']}\nAnswer: {p['answer']}"}
            for p in pairs
        ])

        bnb_cfg = None
        if torch.cuda.is_available():
            bnb_cfg = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)

        tokenizer = AutoTokenizer.from_pretrained(hf_model_id)
        tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            hf_model_id,
            quantization_config=bnb_cfg,
            device_map="auto" if torch.cuda.is_available() else "cpu",
        )

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_r * 2,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
        )
        model = get_peft_model(model, lora_cfg)

        training_args = SFTConfig(
            output_dir=str(out / "checkpoints"),
            num_train_epochs=epochs,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            learning_rate=lr,
            logging_dir=str(out / "logs"),
            save_strategy="epoch",
            report_to="none",
        )

        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            args=training_args,
            train_dataset=dataset,
        )
        trainer.train()

        adapter_path = out / "lora_adapter"
        trainer.model.save_pretrained(str(adapter_path))

        return {
            "mode": "lora",
            "adapter_path": str(adapter_path),
            "base_model": hf_model_id,
            "examples_trained": len(pairs),
            "epochs": epochs,
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    async def _register_ollama_model(self, model_name: str, modelfile_path: Path):
        try:
            proc = subprocess.run(
                ["ollama", "create", model_name, "-f", str(modelfile_path)],
                capture_output=True, text=True, timeout=300
            )
            if proc.returncode != 0:
                raise RuntimeError(f"ollama create failed: {proc.stderr}")
        except FileNotFoundError:
            raise RuntimeError("Ollama CLI non trouvé. Vérifiez que Ollama est installé.")

    async def list_ollama_models(self) -> list[str]:
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                r = await client.get(f"{self.ollama_host}/api/tags")
                return [m["name"] for m in r.json().get("models", [])]
        except Exception:
            return []

    async def delete_ollama_model(self, model_name: str):
        subprocess.run(["ollama", "rm", model_name], capture_output=True, timeout=60)

    def _load_qa_pairs(self, path: Path, max_pairs: int = 500) -> list[dict]:
        pairs = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                if len(pairs) >= max_pairs:
                    break
                try:
                    pairs.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return pairs

    def _extract_domain_knowledge(self, pairs: list[dict], max_chars: int = 3000) -> str:
        seen, parts = set(), []
        total = 0
        for p in pairs:
            ctx = p.get("context", "")
            key = ctx[:100]
            if key not in seen and ctx:
                seen.add(key)
                snippet = ctx[:400]
                parts.append(snippet)
                total += len(snippet)
                if total >= max_chars:
                    break
        return "\n\n---\n\n".join(parts)

    def _format_qa_examples(self, pairs: list[dict]) -> str:
        lines = []
        for p in pairs:
            lines.append(f"Q: {p.get('question', '')}\nR: {p.get('answer', '')}")
        return "\n\n".join(lines)

    def _check_lora_deps(self):
        missing = []
        for pkg in ("torch", "transformers", "peft", "trl", "datasets"):
            try:
                __import__(pkg)
            except ImportError:
                missing.append(pkg)
        if missing:
            raise ImportError(
                f"Dépendances manquantes pour LoRA : {', '.join(missing)}. "
                "Installez-les avec : pip install torch transformers peft trl datasets"
            )

    def _ollama_to_hf(self, ollama_model: str) -> str:
        mapping = {
            "llama3.2:latest": "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct",
            "llama3.2:1b": "meta-llama/Llama-3.2-1B-Instruct",
            "llama3:latest": "meta-llama/Meta-Llama-3-8B-Instruct",
            "mistral:latest": "mistralai/Mistral-7B-Instruct-v0.3",
            "qwen2.5:latest": "Qwen/Qwen2.5-7B-Instruct",
        }
        return mapping.get(ollama_model, ollama_model)
