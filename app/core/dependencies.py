"""
GOV-AI 2.0 — Dépendances FastAPI (v2).
Fournit les singletons des services via injection de dépendances.
"""
from __future__ import annotations

_orchestrator_v2 = None
_audit_service_v2 = None
_evaluation_service_v2 = None


def get_orchestrator_dep():
    """Retourne l'orchestrateur cognitif GOV-AI 2.0 (Algo 1)."""
    global _orchestrator_v2
    if _orchestrator_v2 is None:
        from app.services.cognitive_orchestrator import CognitiveOrchestrator
        _orchestrator_v2 = CognitiveOrchestrator()
    return _orchestrator_v2


def get_audit_dep():
    """Retourne le service d'audit GOV-AI 2.0."""
    global _audit_service_v2
    if _audit_service_v2 is None:
        from app.services.audit.audit_service import AuditService
        _audit_service_v2 = AuditService()
    return _audit_service_v2


def get_evaluation_dep():
    """Retourne le service d'évaluation GOV-AI 2.0."""
    global _evaluation_service_v2
    if _evaluation_service_v2 is None:
        from app.services.evaluation.evaluation_service import EvaluationService
        _evaluation_service_v2 = EvaluationService()
    return _evaluation_service_v2
