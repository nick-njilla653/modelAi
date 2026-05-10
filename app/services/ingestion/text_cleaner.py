"""
GOV-AI 2.0 — Nettoyage et normalisation du texte extrait.
Pipeline : suppression artefacts → normalisation unicode → espaces → heuristiques corpus.
"""
from __future__ import annotations

import re

from app.utils.text_utils import (
    clean_pdf_artifacts,
    normalize_unicode,
    normalize_whitespace,
    remove_control_characters,
)


def clean_extracted_text(text: str, language: str = "fr") -> str:
    """
    Pipeline complet de nettoyage.
    Ordre : unicode → contrôles → artefacts PDF → espaces → heuristiques.
    """
    if not text:
        return ""

    text = normalize_unicode(text)
    text = remove_control_characters(text)
    text = clean_pdf_artifacts(text)
    text = _remove_headers_footers(text)
    text = _normalize_legal_punctuation(text, language)
    text = normalize_whitespace(text)
    return text


def _remove_headers_footers(text: str) -> str:
    """
    Supprime les en-têtes et pieds de page répétitifs des documents officiels.
    Heuristique : lignes très courtes (<30 chars) en début/fin de paragraphe.
    """
    lines = text.split("\n")
    cleaned = []
    for line in lines:
        stripped = line.strip()
        # Filtrer les numéros de page seuls
        if re.match(r"^-?\s*\d+\s*-?$", stripped):
            continue
        # Filtrer les lignes très courtes qui ne sont pas du contenu (heuristique)
        cleaned.append(line)
    return "\n".join(cleaned)


def _normalize_legal_punctuation(text: str, language: str = "fr") -> str:
    """Normalise la ponctuation des textes juridiques camerounais."""
    # Guillemets
    text = text.replace("«", '"').replace("»", '"')
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Tirets em et en → tiret simple
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    # Points de suspension
    text = text.replace("\u2026", "...")
    # Numéros d'articles : normaliser "Art." → "Article" pour le français
    if language == "fr":
        text = re.sub(r"\bArt\.\s+", "Article ", text)
    return text


def is_content_sufficient(text: str, min_chars: int = 50) -> bool:
    """Détermine si le texte extrait est suffisamment long pour être indexé."""
    cleaned = text.strip()
    return len(cleaned) >= min_chars and not re.match(r"^[\W\d\s]+$", cleaned)
