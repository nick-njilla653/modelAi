"""
GOV-AI 2.0 — Dataset Builder pour le fine-tuning.
Extrait le texte d'un dossier de documents et génère des paires QA via le LLM courant.
"""
from __future__ import annotations

import json
import re
import shutil
import zipfile
from pathlib import Path
from typing import Callable, Optional

import fitz  # PyMuPDF

from app.core.logging import get_logger

logger = get_logger(__name__)

SUPPORTED_EXT = {".pdf", ".txt", ".md", ".docx", ".doc"}

_QA_PROMPT = """\
Voici un extrait de document juridique ou administratif camerounais :

---
{text}
---

Génère exactement {n} paires question-réponse en français, ancrées dans ce texte.
Réponds UNIQUEMENT avec un JSON valide (liste), sans aucun commentaire :
[
  {{"question": "...", "answer": "..."}},
  ...
]"""


class DatasetBuilder:
    """Construit un dataset JSONL d'entraînement à partir de documents."""

    def __init__(
        self,
        llm_model: str = "llama3.2:latest",
        ollama_host: str = "http://localhost:11434",
    ):
        self.llm_model = llm_model
        self.ollama_host = ollama_host

    # ── Public API ────────────────────────────────────────────────────────────

    async def build_from_zip(
        self,
        zip_path: str | Path,
        output_dir: str | Path,
        pairs_per_chunk: int = 3,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        zip_path = Path(zip_path)
        tmp = zip_path.parent / f"_ft_extract_{zip_path.stem}"
        tmp.mkdir(exist_ok=True)
        try:
            with zipfile.ZipFile(zip_path, "r") as zf:
                zf.extractall(tmp)
            return await self.build_from_folder(tmp, output_dir, pairs_per_chunk, progress_cb)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    async def build_from_folder(
        self,
        folder: str | Path,
        output_dir: str | Path,
        pairs_per_chunk: int = 3,
        progress_cb: Optional[Callable] = None,
    ) -> dict:
        folder = Path(folder)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in SUPPORTED_EXT]
        if not files:
            raise ValueError(f"Aucun fichier PDF/TXT/MD dans {folder}")

        qa_pairs: list[dict] = []
        embed_pairs: list[dict] = []

        for idx, fp in enumerate(files):
            if progress_cb:
                pct = int((idx / len(files)) * 55)
                await progress_cb(pct, f"Lecture : {fp.name}")
            try:
                text = self._read_text(fp)
                if len(text.strip()) < 50:
                    continue
                chunks = self._chunk(text)
                for chunk in chunks[:8]:
                    pairs = await self._gen_qa(chunk, pairs_per_chunk)
                    for p in pairs:
                        p["source"] = fp.name
                        p["context"] = chunk
                        qa_pairs.append(p)
                        embed_pairs.append({"anchor": p["question"], "positive": chunk})
            except Exception as exc:
                logger.warning("dataset_builder_skip", file=fp.name, error=str(exc))

        if not qa_pairs:
            raise ValueError("Aucune paire générée — les documents sont peut-être vides ou illisibles.")

        qa_path = output_dir / "qa_pairs.jsonl"
        embed_path = output_dir / "embedding_pairs.jsonl"

        qa_path.write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in qa_pairs), encoding="utf-8"
        )
        embed_path.write_text(
            "\n".join(json.dumps(p, ensure_ascii=False) for p in embed_pairs), encoding="utf-8"
        )

        logger.info(
            "dataset_built",
            files=len(files),
            qa_pairs=len(qa_pairs),
            embed_pairs=len(embed_pairs),
        )
        return {
            "files_processed": len(files),
            "total_pairs": len(qa_pairs),
            "embedding_pairs": len(embed_pairs),
            "qa_path": str(qa_path),
            "embedding_path": str(embed_path),
        }

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _read_text(self, path: Path) -> str:
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            return self._read_pdf(path)
        if suffix == ".docx":
            return self._read_docx(path)
        if suffix == ".doc":
            return self._read_doc(path)
        return path.read_text(encoding="utf-8", errors="ignore")

    def _read_pdf(self, path: Path) -> str:
        """Extraction PDF avec fallback OCR automatique pour les pages scannées."""
        try:
            from app.services.ingestion.pdf_extractor import (
                extract_pdf_text,
                extract_pdf_as_images,
            )
            from app.services.ingestion.ocr_processor import ocr_image_bytes

            extracted = extract_pdf_text(path, force_ocr=False)
            if not extracted.needs_ocr:
                return extracted.full_text

            # PDF scanné → OCR page par page (pipeline existant, 300 DPI)
            logger.info("pdf_ocr_fallback", file=path.name)
            page_images = extract_pdf_as_images(path)
            texts = []
            for _page_num, img_bytes in page_images:
                ocr_text = ocr_image_bytes(img_bytes, language="fra+eng")
                if ocr_text.strip():
                    texts.append(ocr_text)
            return "\n\n".join(texts)

        except Exception:
            # Fallback minimal si le pipeline d'ingestion n'est pas disponible
            import fitz
            doc = fitz.open(str(path))
            return "\n".join(p.get_text() for p in doc)

    def _read_docx(self, path: Path) -> str:
        """Extraction Word (.docx) via python-docx."""
        try:
            import docx as python_docx
            doc = python_docx.Document(str(path))
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
            # Tableaux
            for table in doc.tables:
                for row in table.rows:
                    cells = [c.text.strip() for c in row.cells if c.text.strip()]
                    if cells:
                        paragraphs.append(" | ".join(cells))
            return "\n".join(paragraphs)
        except ImportError:
            logger.warning("python_docx_not_installed", file=path.name)
            return ""
        except Exception as exc:
            logger.warning("docx_read_error", file=path.name, error=str(exc))
            return ""

    def _read_doc(self, path: Path) -> str:
        """Extraction Word ancien format (.doc) via python-docx ou antiword."""
        # python-docx ne supporte pas .doc, on tente antiword ou on ignore
        try:
            import subprocess
            result = subprocess.run(
                ["antiword", str(path)], capture_output=True, text=True, timeout=30
            )
            if result.returncode == 0 and result.stdout:
                return result.stdout
        except Exception:
            pass
        logger.warning("doc_format_skipped", file=path.name,
                       hint="Convertissez en .docx ou installez antiword")
        return ""

    def _chunk(self, text: str, size: int = 700, overlap: int = 80) -> list[str]:
        words = text.split()
        chunks, i = [], 0
        while i < len(words):
            chunk = " ".join(words[i : i + size])
            if len(chunk.strip()) > 60:
                chunks.append(chunk)
            i += size - overlap
        return chunks

    async def _gen_qa(self, text: str, n: int) -> list[dict]:
        import httpx

        prompt = _QA_PROMPT.format(text=text[:1600], n=n)
        try:
            async with httpx.AsyncClient(timeout=90.0) as client:
                r = await client.post(
                    f"{self.ollama_host}/api/generate",
                    json={"model": self.llm_model, "prompt": prompt, "stream": False},
                )
                raw = r.json().get("response", "")
                match = re.search(r"\[.*?\]", raw, re.DOTALL)
                if match:
                    parsed = json.loads(match.group())
                    return [p for p in parsed if isinstance(p, dict) and "question" in p and "answer" in p]
        except Exception as exc:
            logger.debug("qa_gen_failed", error=str(exc))
        return []
