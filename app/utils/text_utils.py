"""
GOV-AI 2.0 — Utilitaires texte communs.
"""
from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Normalise les espaces blancs (multiples espaces, tabs, newlines)."""
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def remove_control_characters(text: str) -> str:
    """Supprime les caractères de contrôle sauf newline/tab."""
    return "".join(
        ch for ch in text
        if unicodedata.category(ch)[0] != "C" or ch in "\n\t"
    )


def normalize_unicode(text: str) -> str:
    """Normalisation Unicode NFC (pour les accents français)."""
    return unicodedata.normalize("NFC", text)


def clean_pdf_artifacts(text: str) -> str:
    """Nettoie les artefacts courants des PDF (ligatures, coupures de mots)."""
    # Ligatures
    text = text.replace("\ufb01", "fi").replace("\ufb02", "fl")
    text = text.replace("\ufb00", "ff").replace("\ufb03", "ffi").replace("\ufb04", "ffl")
    # Coupures de mots en fin de ligne (tiret + saut de ligne)
    text = re.sub(r"-\n(\w)", r"\1", text)
    # Numéros de page isolés
    text = re.sub(r"^\s*\d+\s*$", "", text, flags=re.MULTILINE)
    return text


def truncate_text(text: str, max_chars: int = 300, suffix: str = "...") -> str:
    """Tronque le texte à max_chars caractères."""
    if len(text) <= max_chars:
        return text
    return text[:max_chars - len(suffix)].rstrip() + suffix


def count_tokens_approx(text: str, chars_per_token: float = 4.0) -> int:
    """Estimation rapide du nombre de tokens (heuristique 4 chars/token)."""
    return max(1, int(len(text) / chars_per_token))


def extract_article_number(text: str) -> str | None:
    """Extrait le numéro d'article d'un texte législatif (FR/EN)."""
    patterns = [
        r"^Article\s+(\d+(?:\s*bis)?(?:\s*ter)?)",
        r"^Art\.\s+(\d+)",
        r"^Section\s+(\d+)",
        r"^Alinéa\s+(\d+)",
    ]
    for pattern in patterns:
        match = re.match(pattern, text.strip(), re.IGNORECASE)
        if match:
            return match.group(1)
    return None


def is_structural_separator(text: str) -> bool:
    """Détermine si un texte est un séparateur structurel (article, section…)."""
    from app.core.constants import STRUCTURAL_CHUNK_PATTERNS
    for pattern in STRUCTURAL_CHUNK_PATTERNS:
        if re.match(pattern, text.strip(), re.IGNORECASE):
            return True
    return False
