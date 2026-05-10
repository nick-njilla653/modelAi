"""
GOV-AI 2.0 — OCR Tesseract 5 pour les documents scannés.
Support bilingue fra+eng, pré-traitement image, cache résultats.
"""
from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from typing import Optional

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


def _get_cache_path(image_hash: str) -> Path:
    """Chemin du cache OCR pour une image donnée."""
    settings = get_settings()
    return Path(settings.ocr_cache_path) / f"{image_hash}.json"


def _load_ocr_cache(image_hash: str) -> Optional[str]:
    """Charge le résultat OCR depuis le cache."""
    cache_path = _get_cache_path(image_hash)
    if cache_path.exists():
        data = json.loads(cache_path.read_text(encoding="utf-8"))
        return data.get("text", "")
    return None


def _save_ocr_cache(image_hash: str, text: str) -> None:
    """Sauvegarde le résultat OCR dans le cache."""
    cache_path = _get_cache_path(image_hash)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps({"text": text}), encoding="utf-8")


def ocr_image_bytes(
    image_bytes: bytes,
    language: str = "fra+eng",
    use_cache: bool = True,
) -> str:
    """
    Applique l'OCR Tesseract 5 sur des bytes d'image.

    Args:
        image_bytes: Bytes PNG/JPEG de l'image
        language: Codes langue Tesseract (fra+eng pour bilingue)
        use_cache: Utiliser le cache si disponible

    Returns:
        Texte extrait par OCR
    """
    try:
        import pytesseract
        from PIL import Image
    except ImportError:
        raise ImportError("pytesseract et Pillow requis")

    settings = get_settings()

    # Configurer le chemin Tesseract si spécifié
    if settings.tesseract_path and Path(settings.tesseract_path).exists():
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_path

    # Cache check
    image_hash = hashlib.sha256(image_bytes).hexdigest()[:16]
    if use_cache:
        cached = _load_ocr_cache(image_hash)
        if cached is not None:
            logger.debug("ocr_cache_hit", hash=image_hash)
            return cached

    # Pré-traitement image
    image = Image.open(BytesIO(image_bytes))
    image = _preprocess_image(image)

    # Configuration Tesseract : PSM 3 = auto page segmentation (défaut)
    custom_config = "--oem 3 --psm 3"

    try:
        text = pytesseract.image_to_string(
            image,
            lang=language,
            config=custom_config,
        )
    except Exception as exc:
        logger.error("ocr_failed", hash=image_hash, error=str(exc))
        return ""

    if use_cache:
        _save_ocr_cache(image_hash, text)

    logger.debug("ocr_completed", hash=image_hash, chars=len(text))
    return text


def _preprocess_image(image: "Image.Image") -> "Image.Image":  # type: ignore
    """
    Pré-traitement pour améliorer la qualité OCR :
    - Conversion en niveaux de gris
    - Seuillage (binarisation)
    """
    try:
        import numpy as np
        from PIL import Image, ImageFilter

        # Convertir en niveaux de gris
        gray = image.convert("L")

        # Améliorer la netteté
        sharpened = gray.filter(ImageFilter.SHARPEN)

        return sharpened
    except Exception:
        return image


def compute_ocr_cer(reference: str, hypothesis: str) -> float:
    """
    Calcule le Character Error Rate (CER) pour évaluer la qualité OCR.
    CER = edit_distance(ref, hyp) / len(ref)
    """
    if not reference:
        return 0.0 if not hypothesis else 1.0

    # Algorithme Levenshtein simple
    ref_chars = list(reference)
    hyp_chars = list(hypothesis)
    m, n = len(ref_chars), len(hyp_chars)

    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j

    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if ref_chars[i - 1] == hyp_chars[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

    return dp[m][n] / m
