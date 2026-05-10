"""
GOV-AI 2.0 — Constantes domaine.
Centralise toutes les constantes métier de l'administration camerounaise.
"""

# ── Profils utilisateurs (Tableau 2.1 du mémoire) ────────────────────────────
USER_PROFILES = {
    "citizen": {
        "label_fr": "Citoyen",
        "label_en": "Citizen",
        "register": "vulgarisé, pédagogique",
        "detail_level": "synthétique",
        "jargon_tolerance": "faible",
        "citations_mode": "optional",
    },
    "agent": {
        "label_fr": "Agent public",
        "label_en": "Public servant",
        "register": "administratif, structuré",
        "detail_level": "moyen",
        "jargon_tolerance": "moyenne",
        "citations_mode": "systematic",
    },
    "enterprise": {
        "label_fr": "Entreprise",
        "label_en": "Enterprise",
        "register": "professionnel, pratique",
        "detail_level": "moyen",
        "jargon_tolerance": "moyenne",
        "citations_mode": "systematic",
    },
    "jurist": {
        "label_fr": "Juriste",
        "label_en": "Jurist",
        "register": "technique, juridique",
        "detail_level": "approfondi",
        "jargon_tolerance": "élevée",
        "citations_mode": "exhaustive",
    },
}

# ── Systèmes juridiques (bijuridisme camerounais) ─────────────────────────────
JURIDICAL_SYSTEMS = {
    "civil_law": {
        "label_fr": "Droit civil",
        "label_en": "Civil law",
        "regions": ["Centre", "Littoral", "Nord", "Adamaoua", "Est", "Ouest", "Sud", "Extrême-Nord"],
    },
    "common_law": {
        "label_fr": "Common law",
        "label_en": "Common law",
        "regions": ["Nord-Ouest", "Sud-Ouest"],
    },
    "ohada": {
        "label_fr": "Droit OHADA",
        "label_en": "OHADA law",
        "regions": ["all"],
    },
    "constitutional": {
        "label_fr": "Droit constitutionnel",
        "label_en": "Constitutional law",
        "regions": ["all"],
    },
}

# ── Types d'intentions ────────────────────────────────────────────────────────
INTENT_TYPES = [
    "factual_query",       # Question factuelle simple
    "procedural_query",    # Question procédurale (comment faire)
    "normative_query",     # Question sur texte de loi
    "comparative_query",   # Comparaison bijuridique
    "document_request",    # Demande de document
    "out_of_scope",        # Hors domaine
]

# ── Types de documents ────────────────────────────────────────────────────────
DOCUMENT_TYPES = [
    "constitution",
    "loi_organique",
    "loi",
    "ordonnance",
    "decret",
    "arrete",
    "circulaire",
    "note_de_service",
    "guide_procedure",
    "formulaire",
    "acte_ohada",
    "convention_internationale",
    "jurisprudence",
    "doctrine",
    "autre",
]

# ── Institutions ──────────────────────────────────────────────────────────────
INSTITUTIONS = [
    "presidence",
    "primature",
    "ministere",
    "assemblee_nationale",
    "senat",
    "conseil_constitutionnel",
    "cour_supreme",
    "tribunal",
    "mairie",
    "prefecture",
    "cenadi",
    "ohada",
    "cemac",
    "autre",
]

# ── Seuils de confiance (alignés sur les algorithmes du mémoire) ──────────────
CONFIDENCE_THRESHOLDS = {
    "HIGH": 0.8,      # Réponse directe avec citations
    "MEDIUM": 0.6,    # Réponse avec avertissement (τ_conf)
    "LOW": 0.3,       # Escalade recommandée (τ_esc)
    "REFUSE": 0.0,    # Refus de répondre
}

# ── Indice de Symétrie Bilingue ───────────────────────────────────────────────
ISB_MIN_ACCEPTABLE = 0.85  # Contrainte C1 du mémoire

# ── Chunking ─────────────────────────────────────────────────────────────────
DEFAULT_CHUNK_SIZE = 512
DEFAULT_CHUNK_OVERLAP = 64
STRUCTURAL_CHUNK_PATTERNS = [
    r"^(Article|Alinéa|Section|Chapitre|Titre|Partie)\s+\d+",
    r"^(Article|Section|Chapter|Part|Title)\s+\d+",
]

# ── Langues supportées ────────────────────────────────────────────────────────
SUPPORTED_LANGUAGES = ["fr", "en"]
DEFAULT_LANGUAGE = "fr"

# ── Pagination API ────────────────────────────────────────────────────────────
DEFAULT_PAGE_SIZE = 20
MAX_PAGE_SIZE = 100
