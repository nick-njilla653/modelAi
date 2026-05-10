"""
GOV-AI 2.0 — Extraction de texte depuis des PDF avec PyMuPDF.
Détecte automatiquement si le PDF est natif ou scanné (→ OCR).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.core.logging import get_logger

logger = get_logger(__name__)


@dataclass
class PageContent:
    page_number: int
    text: str
    needs_ocr: bool = False
    width: float = 0.0
    height: float = 0.0


@dataclass
class ExtractedDocument:
    filename: str
    pages: list[PageContent]
    total_pages: int
    needs_ocr: bool = False
    extraction_method: str = "native"

    @property
    def full_text(self) -> str:
        return "\n\n".join(p.text for p in self.pages if p.text.strip())


_MIN_CHARS_PER_PAGE = 50  # En dessous → page probablement scannée


def extract_pdf_text(
    file_path: str | Path,
    force_ocr: bool = False,
) -> ExtractedDocument:
    """
    Extrait le texte d'un PDF page par page.
    Si une page contient moins de MIN_CHARS, elle est marquée pour OCR.

    Args:
        file_path: Chemin vers le fichier PDF
        force_ocr: Forcer l'OCR même si le texte natif est disponible

    Returns:
        ExtractedDocument avec le contenu de chaque page
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        raise ImportError("PyMuPDF requis : pip install pymupdf")

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Fichier PDF introuvable : {path}")

    pages: list[PageContent] = []
    needs_global_ocr = False

    try:
        doc = fitz.open(str(path))
        total_pages = len(doc)

        for page_num in range(total_pages):
            page = doc[page_num]
            rect = page.rect

            if force_ocr:
                pages.append(PageContent(
                    page_number=page_num + 1,
                    text="",
                    needs_ocr=True,
                    width=rect.width,
                    height=rect.height,
                ))
                needs_global_ocr = True
                continue

            # Extraction texte natif
            text = page.get_text("text")

            if len(text.strip()) < _MIN_CHARS_PER_PAGE:
                # Probable page scannée
                pages.append(PageContent(
                    page_number=page_num + 1,
                    text="",
                    needs_ocr=True,
                    width=rect.width,
                    height=rect.height,
                ))
                needs_global_ocr = True
            else:
                pages.append(PageContent(
                    page_number=page_num + 1,
                    text=text,
                    needs_ocr=False,
                    width=rect.width,
                    height=rect.height,
                ))

        doc.close()
        logger.info(
            "pdf_extracted",
            filename=path.name,
            total_pages=total_pages,
            needs_ocr=needs_global_ocr,
        )

    except Exception as exc:
        logger.error("pdf_extraction_failed", filename=str(path), error=str(exc))
        raise

    return ExtractedDocument(
        filename=path.name,
        pages=pages,
        total_pages=total_pages,
        needs_ocr=needs_global_ocr,
        extraction_method="ocr" if needs_global_ocr else "native",
    )


def extract_pdf_as_images(file_path: str | Path) -> list[tuple[int, bytes]]:
    """
    Convertit les pages PDF en images PNG pour l'OCR.
    Retourne [(page_number, png_bytes), ...]
    """
    try:
        import fitz
    except ImportError:
        raise ImportError("PyMuPDF requis")

    path = Path(file_path)
    images = []

    doc = fitz.open(str(path))
    for page_num in range(len(doc)):
        page = doc[page_num]
        # Résolution 300 DPI pour une bonne qualité OCR
        mat = fitz.Matrix(300 / 72, 300 / 72)
        pix = page.get_pixmap(matrix=mat)
        images.append((page_num + 1, pix.tobytes("png")))
    doc.close()

    return images
