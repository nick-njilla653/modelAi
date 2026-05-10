"""
GOV-AI 2.0 — Détection de langue avec langdetect.
Support bilingue FR/EN, code-switching camerounais.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.models.domain import Language

logger = get_logger(__name__)

# Lazy import pour éviter l'overhead au démarrage
_langdetect_available = False
try:
    from langdetect import detect, detect_langs
    from langdetect.lang_detect_exception import LangDetectException
    _langdetect_available = True
except ImportError:
    logger.warning("langdetect_not_available", fallback="fr")


def detect_language(text: str, min_text_length: int = 20) -> Language:
    """
    Détecte la langue d'un texte. Retourne FR ou EN.
    Pour le code-switching camerounais (FR/EN mixte), retourne la langue dominante.
    """
    if not text or len(text.strip()) < min_text_length:
        return Language.FR  # Défaut : français (langue dominante Cameroun)

    if not _langdetect_available:
        return Language.FR

    try:
        detected = detect(text[:1000])  # Limiter pour performance
        if detected == "fr":
            return Language.FR
        elif detected == "en":
            return Language.EN
        else:
            # Langue non supportée → fallback français
            logger.debug("unsupported_language_detected", detected=detected, fallback="fr")
            return Language.FR
    except LangDetectException:
        return Language.FR
    except Exception as exc:
        logger.warning("language_detection_error", error=str(exc))
        return Language.FR


def detect_language_with_confidence(
    text: str,
) -> tuple[Language, float]:
    """Retourne (langue, confiance) — utile pour le logging."""
    if not _langdetect_available or not text:
        return Language.FR, 1.0

    try:
        from langdetect import detect_langs
        from langdetect.lang_detect_exception import LangDetectException

        langs = detect_langs(text[:1000])
        if not langs:
            return Language.FR, 0.5

        top = langs[0]
        lang_str = str(top.lang)
        prob = float(top.prob)

        if lang_str == "fr":
            return Language.FR, prob
        elif lang_str == "en":
            return Language.EN, prob
        else:
            return Language.FR, prob
    except Exception:
        return Language.FR, 0.5


def is_bilingual_text(text: str, threshold: float = 0.3) -> bool:
    """
    Détecte si un texte est bilingue (code-switching FR/EN).
    Utilisé pour le corpus camerounais où les deux langues coexistent.
    """
    if not _langdetect_available or not text:
        return False

    try:
        from langdetect import detect_langs
        langs = detect_langs(text[:2000])
        if len(langs) >= 2:
            probs = {str(l.lang): float(l.prob) for l in langs}
            fr_prob = probs.get("fr", 0.0)
            en_prob = probs.get("en", 0.0)
            return fr_prob >= threshold and en_prob >= threshold
    except Exception:
        pass
    return False
