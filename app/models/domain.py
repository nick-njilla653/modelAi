"""
GOV-AI 2.0 — Énumérations et types domaine.
"""
from __future__ import annotations

from enum import Enum


class Language(str, Enum):
    FR = "fr"
    EN = "en"
    UNKNOWN = "unknown"


class UserProfile(str, Enum):
    CITIZEN = "citizen"      # Citoyen
    AGENT = "agent"          # Agent public
    ENTERPRISE = "enterprise"  # Entreprise
    JURIST = "jurist"        # Juriste / magistrat
    UNKNOWN = "unknown"


class JuridicalSystem(str, Enum):
    CIVIL_LAW = "civil_law"
    COMMON_LAW = "common_law"
    OHADA = "ohada"
    CONSTITUTIONAL = "constitutional"
    UNKNOWN = "unknown"


class IntentType(str, Enum):
    FACTUAL = "factual_query"
    PROCEDURAL = "procedural_query"
    NORMATIVE = "normative_query"
    COMPARATIVE = "comparative_query"
    DOCUMENT_REQUEST = "document_request"
    OUT_OF_SCOPE = "out_of_scope"
    UNKNOWN = "unknown"


class DocumentType(str, Enum):
    CONSTITUTION = "constitution"
    LOI_ORGANIQUE = "loi_organique"
    LOI = "loi"
    ORDONNANCE = "ordonnance"
    DECRET = "decret"
    ARRETE = "arrete"
    CIRCULAIRE = "circulaire"
    NOTE_DE_SERVICE = "note_de_service"
    GUIDE_PROCEDURE = "guide_procedure"
    FORMULAIRE = "formulaire"
    ACTE_OHADA = "acte_ohada"
    CONVENTION = "convention_internationale"
    JURISPRUDENCE = "jurisprudence"
    DOCTRINE = "doctrine"
    AUTRE = "autre"


class IngestionStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    PARTIAL = "partial"


class ConfidenceLevel(str, Enum):
    HIGH = "high"      # >= 0.8
    MEDIUM = "medium"  # >= 0.6
    LOW = "low"        # >= 0.3
    INSUFFICIENT = "insufficient"  # < 0.3

    @classmethod
    def from_score(cls, score: float) -> "ConfidenceLevel":
        if score >= 0.8:
            return cls.HIGH
        elif score >= 0.6:
            return cls.MEDIUM
        elif score >= 0.3:
            return cls.LOW
        else:
            return cls.INSUFFICIENT


class SafetyFlag(str, Enum):
    LOW_CONFIDENCE = "LOW_CONFIDENCE"
    ESCALATION_RECOMMENDED = "ESCALATION_RECOMMENDED"
    CONTRADICTION_DETECTED = "CONTRADICTION_DETECTED"
    OUT_OF_CORPUS = "OUT_OF_CORPUS"
    PROMPT_INJECTION_ATTEMPT = "PROMPT_INJECTION_ATTEMPT"
    UNSUPPORTED_CLAIMS = "UNSUPPORTED_CLAIMS"
    JURIDICAL_DIVERGENCE = "JURIDICAL_DIVERGENCE"


class ChunkStrategy(str, Enum):
    STRUCTURAL = "structural"   # Par article/section (documents structurés)
    FIXED_SIZE = "fixed_size"   # Taille fixe avec overlap
    HYBRID = "hybrid"           # Structural puis fixed si trop grand


class AuditEventType(str, Enum):
    QUERY = "query"
    INGEST = "ingest"
    RETRIEVAL = "retrieval"
    GENERATION = "generation"
    VERIFICATION = "verification"
    SECURITY_FLAG = "security_flag"
    MEMORY_WRITE = "memory_write"
    MEMORY_READ = "memory_read"
    WEB_SEARCH = "web_search"
    ERROR = "error"
