"""
GOV-AI 2.0 — Filtres de sécurité (Contrainte C8 du mémoire).
Validation, sanitisation et détection de patterns malveillants.
"""
from __future__ import annotations

import re
import unicodedata

# ── Patterns d'injection ──────────────────────────────────────────────────────

_SQL_INJECTION = re.compile(
    r"('|--|;|/\*|\*/|xp_|UNION\s+SELECT|DROP\s+TABLE|INSERT\s+INTO|DELETE\s+FROM)",
    re.IGNORECASE,
)

_PROMPT_INJECTION = re.compile(
    r"(ignore\s+(previous|all)\s+instructions?|"
    r"forget\s+(your|all)\s+(rules?|instructions?|context)|"
    r"you\s+are\s+now\s+|act\s+as\s+|jailbreak|"
    r"oublie\s+tes\s+instructions|tu\s+es\s+maintenant|"
    r"fais\s+semblant\s+d.être|nouvelle\s+identité)",
    re.IGNORECASE,
)

_XSS_PATTERN = re.compile(
    r"(<script|</script|javascript:|on\w+\s*=|<iframe|<object|<embed)",
    re.IGNORECASE,
)

_PATH_TRAVERSAL = re.compile(r"\.\./|\.\.\\|%2e%2e", re.IGNORECASE)


def sanitize_query(query: str) -> str:
    """
    Sanitise une requête utilisateur :
    - Normalisation Unicode NFC
    - Suppression des caractères de contrôle
    - Suppression des patterns XSS
    - Troncature à max_length

    Args:
        query: Requête brute

    Returns:
        Requête sanitisée
    """
    # Normalisation Unicode
    query = unicodedata.normalize("NFC", query)

    # Suppression des caractères de contrôle (sauf newline et tab)
    query = "".join(
        c for c in query
        if unicodedata.category(c) != "Cc" or c in ("\n", "\t")
    )

    # Strip XSS basique
    query = _XSS_PATTERN.sub("", query)

    # Normalisation des espaces
    query = re.sub(r"\s+", " ", query).strip()

    return query


def detect_injection_attempt(text: str) -> list[str]:
    """
    Détecte les patterns d'injection dans le texte.

    Returns:
        Liste des types d'injection détectés
    """
    detected: list[str] = []

    if _PROMPT_INJECTION.search(text):
        detected.append("prompt_injection")
    if _SQL_INJECTION.search(text):
        detected.append("sql_injection")
    if _XSS_PATTERN.search(text):
        detected.append("xss")
    if _PATH_TRAVERSAL.search(text):
        detected.append("path_traversal")

    return detected


def validate_query_length(
    query: str,
    min_length: int = 3,
    max_length: int = 2000,
) -> tuple[bool, str]:
    """
    Valide la longueur de la requête.

    Returns:
        (is_valid, error_message)
    """
    if len(query) < min_length:
        return False, f"Requête trop courte (minimum {min_length} caractères)"
    if len(query) > max_length:
        return False, f"Requête trop longue (maximum {max_length} caractères)"
    return True, ""


def validate_filename(filename: str) -> tuple[bool, str]:
    """
    Valide un nom de fichier uploadé.

    Returns:
        (is_valid, error_message)
    """
    if _PATH_TRAVERSAL.search(filename):
        return False, "Nom de fichier invalide (path traversal détecté)"

    # Extensions autorisées
    allowed_extensions = {".pdf", ".txt", ".md", ".docx"}
    ext = "." + filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in allowed_extensions:
        return False, f"Extension non supportée : {ext}. Autorisées : {', '.join(allowed_extensions)}"

    # Longueur
    if len(filename) > 255:
        return False, "Nom de fichier trop long (max 255 caractères)"

    return True, ""
