"""
GOV-AI 2.0 — Extraction de métadonnées documentaires.
Métadonnées : source, langue, date, institution, version, juridiction, page.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from app.models.domain import DocumentType, JuridicalSystem, Language
from app.utils.language_detection import detect_language


@dataclass
class DocumentMetadata:
    """Métadonnées complètes d'un document ingéré."""
    source: str
    filename: str
    language: Language = Language.UNKNOWN
    doc_type: Optional[DocumentType] = None
    institution: Optional[str] = None
    jurisdiction: Optional[str] = None
    juridical_system: Optional[JuridicalSystem] = None
    date_document: Optional[str] = None
    version: str = "1.0"
    page_count: int = 0
    char_count: int = 0
    extra: dict = field(default_factory=dict)


# Patterns de détection du type documentaire (textes camerounais)
_DOC_TYPE_PATTERNS: list[tuple[re.Pattern, DocumentType]] = [
    (re.compile(r"\bconstitution\b", re.I), DocumentType.CONSTITUTION),
    (re.compile(r"\bloi\s+organique\b", re.I), DocumentType.LOI_ORGANIQUE),
    (re.compile(r"\bordonnance\b", re.I), DocumentType.ORDONNANCE),
    (re.compile(r"\bdécret\b|\bdecret\b", re.I), DocumentType.DECRET),
    (re.compile(r"\barrêté\b|\barrête\b|\barreted\b", re.I), DocumentType.ARRETE),
    (re.compile(r"\bcirculaire\b", re.I), DocumentType.CIRCULAIRE),
    (re.compile(r"\bnote\s+de\s+service\b", re.I), DocumentType.NOTE_DE_SERVICE),
    (re.compile(r"\bformulaire\b", re.I), DocumentType.FORMULAIRE),
    (re.compile(r"\bguide\b.*\bprocédure\b|\bguide\b.*\bprocedure\b", re.I), DocumentType.GUIDE_PROCEDURE),
    (re.compile(r"\bohada\b", re.I), DocumentType.ACTE_OHADA),
    (re.compile(r"\bjurisprudence\b|\barrêt\b|\bjugement\b", re.I), DocumentType.JURISPRUDENCE),
    (re.compile(r"\bloi\b", re.I), DocumentType.LOI),
]

# Patterns de détection d'institution
_INSTITUTION_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r"\bprésidence\b|\bpresidence\b|\bpalais\s+d.étoile\b", re.I), "presidence"),
    (re.compile(r"\bprimature\b|\bpremier\s+ministre\b", re.I), "primature"),
    (re.compile(r"\bassemblée\s+nationale\b|\bnational\s+assembly\b", re.I), "assemblee_nationale"),
    (re.compile(r"\bsénat\b|\bsenate\b", re.I), "senat"),
    (re.compile(r"\bcour\s+suprême\b|\bsupreme\s+court\b", re.I), "cour_supreme"),
    (re.compile(r"\bministère\b|\bministry\b|\bministre\b|\bminister\b", re.I), "ministere"),
    (re.compile(r"\bmairie\b|\bcommune\b|\bmunicipal\b", re.I), "mairie"),
    (re.compile(r"\bpréfecture\b|\bprefecture\b|\bsous-préfecture\b", re.I), "prefecture"),
    (re.compile(r"\bohada\b", re.I), "ohada"),
    (re.compile(r"\bcenadi\b", re.I), "cenadi"),
]

# Détection du système juridique
_COMMON_LAW_REGIONS = ["north west", "south west", "nord-ouest", "sud-ouest", "nw", "sw"]
_CIVIL_LAW_REGIONS = ["centre", "littoral", "nord", "adamaoua", "est", "ouest", "sud", "extrême-nord"]


def extract_metadata(
    text: str,
    filename: str,
    source: str,
    provided_metadata: Optional[dict] = None,
) -> DocumentMetadata:
    """
    Extrait les métadonnées d'un document à partir de son texte.
    Les métadonnées fournies manuellement ont priorité sur les détections auto.
    """
    meta = DocumentMetadata(source=source, filename=filename)
    provided = provided_metadata or {}

    # Langue
    meta.language = Language(provided.get("language") or detect_language(text[:2000]))

    # Type de document
    if provided.get("doc_type"):
        try:
            meta.doc_type = DocumentType(provided["doc_type"])
        except ValueError:
            pass
    else:
        meta.doc_type = _detect_doc_type(text[:500])

    # Institution
    meta.institution = provided.get("institution") or _detect_institution(text[:1000])

    # Juridiction
    meta.jurisdiction = provided.get("jurisdiction") or _detect_juridical_system(text[:2000])

    # Date
    meta.date_document = provided.get("date_document") or _extract_date(text[:500])

    # Version
    meta.version = provided.get("version", "1.0")

    # Stats
    meta.char_count = len(text)

    return meta


def _detect_doc_type(text: str) -> Optional[DocumentType]:
    """Détecte le type de document depuis les premiers paragraphes."""
    for pattern, doc_type in _DOC_TYPE_PATTERNS:
        if pattern.search(text):
            return doc_type
    return None


def _detect_institution(text: str) -> Optional[str]:
    """Détecte l'institution émettrice depuis le texte."""
    for pattern, institution in _INSTITUTION_PATTERNS:
        if pattern.search(text):
            return institution
    return None


def _detect_juridical_system(text: str) -> Optional[str]:
    """
    Détecte le système juridique applicable (civil law / common law / OHADA).
    Règle R1 du mémoire : détecter selon les indices régionaux.
    """
    text_lower = text.lower()

    if any(region in text_lower for region in _COMMON_LAW_REGIONS):
        return JuridicalSystem.COMMON_LAW
    if "ohada" in text_lower:
        return JuridicalSystem.OHADA
    if any(region in text_lower for region in _CIVIL_LAW_REGIONS):
        return JuridicalSystem.CIVIL_LAW
    return None


def _extract_date(text: str) -> Optional[str]:
    """Extrait une date de document (format DD/MM/YYYY ou DD mois YYYY)."""
    # Format numérique
    num_match = re.search(r"\b(\d{1,2})[/.-](\d{1,2})[/.-](\d{4})\b", text)
    if num_match:
        return f"{num_match.group(3)}-{num_match.group(2):0>2}-{num_match.group(1):0>2}"

    # Format textuel français
    months = {
        "janvier": "01", "février": "02", "mars": "03", "avril": "04",
        "mai": "05", "juin": "06", "juillet": "07", "août": "08",
        "septembre": "09", "octobre": "10", "novembre": "11", "décembre": "12",
    }
    for month_fr, month_num in months.items():
        pattern = re.compile(
            rf"\b(\d{{1,2}})\s+{month_fr}\s+(\d{{4}})\b", re.IGNORECASE
        )
        match = pattern.search(text)
        if match:
            return f"{match.group(2)}-{month_num}-{int(match.group(1)):02d}"

    return None
