"""
GOV-AI 2.0 — Exceptions domaine et handlers FastAPI.
"""
from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.responses import JSONResponse


# ── Exceptions domaine ────────────────────────────────────────────────────────

class GovAIException(Exception):
    """Exception de base GOV-AI 2.0."""

    def __init__(self, message: str, code: str = "GOV_AI_ERROR") -> None:
        self.message = message
        self.code = code
        super().__init__(message)


class DocumentNotFoundError(GovAIException):
    def __init__(self, doc_id: str) -> None:
        super().__init__(f"Document {doc_id!r} introuvable", "DOCUMENT_NOT_FOUND")
        self.doc_id = doc_id


class IngestionError(GovAIException):
    def __init__(self, message: str) -> None:
        super().__init__(message, "INGESTION_ERROR")


class EmbeddingError(GovAIException):
    def __init__(self, message: str) -> None:
        super().__init__(message, "EMBEDDING_ERROR")


class RetrievalError(GovAIException):
    def __init__(self, message: str) -> None:
        super().__init__(message, "RETRIEVAL_ERROR")


class GenerationError(GovAIException):
    def __init__(self, message: str) -> None:
        super().__init__(message, "GENERATION_ERROR")


class InsufficientEvidenceError(GovAIException):
    """Levée quand le score de confiance est trop bas pour répondre."""

    def __init__(self, score: float, threshold: float) -> None:
        super().__init__(
            f"Preuves insuffisantes (score={score:.2f} < seuil={threshold:.2f}). "
            "Veuillez reformuler ou consulter un professionnel.",
            "INSUFFICIENT_EVIDENCE",
        )
        self.score = score
        self.threshold = threshold


class StorageError(GovAIException):
    def __init__(self, message: str) -> None:
        super().__init__(message, "STORAGE_ERROR")


class SecurityViolationError(GovAIException):
    def __init__(self, message: str) -> None:
        super().__init__(message, "SECURITY_VIOLATION")


class UnsupportedLanguageError(GovAIException):
    def __init__(self, language: str) -> None:
        super().__init__(
            f"Langue {language!r} non supportée. Langues acceptées : fr, en",
            "UNSUPPORTED_LANGUAGE",
        )


# ── Handlers FastAPI ──────────────────────────────────────────────────────────

def register_exception_handlers(app: FastAPI) -> None:
    """Enregistre tous les handlers d'exception sur l'application FastAPI."""

    @app.exception_handler(DocumentNotFoundError)
    async def document_not_found_handler(
        request: Request, exc: DocumentNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"error": exc.code, "message": exc.message},
        )

    @app.exception_handler(InsufficientEvidenceError)
    async def insufficient_evidence_handler(
        request: Request, exc: InsufficientEvidenceError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_200_OK,  # Ce n'est pas une erreur technique
            content={
                "error": exc.code,
                "message": exc.message,
                "answer": exc.message,
                "citations": [],
                "uncertainty_score": exc.score,
                "safety_flags": ["LOW_CONFIDENCE"],
            },
        )

    @app.exception_handler(SecurityViolationError)
    async def security_violation_handler(
        request: Request, exc: SecurityViolationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"error": exc.code, "message": "Requête non autorisée"},
        )

    @app.exception_handler(GovAIException)
    async def govai_exception_handler(
        request: Request, exc: GovAIException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"error": exc.code, "message": exc.message},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "INTERNAL_ERROR",
                "message": "Une erreur interne est survenue. Veuillez réessayer.",
            },
        )
