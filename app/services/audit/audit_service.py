"""
GOV-AI 2.0 — Service d'audit et de traçabilité (Contrainte C10 du mémoire).
Journalise chaque action avec : trace_id, session_id, timestamp, entité, flags.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any, Optional

from app.core.logging import get_logger
from app.models.domain import AuditEventType, SafetyFlag

logger = get_logger(__name__)


class AuditService:
    """
    Service d'audit structuré.

    Implémente la contrainte C10 (traçabilité) :
    - Log JSON de chaque requête avec tous les champs de traçabilité
    - Persistance PostgreSQL optionnelle (Sprint 1 : log structuré uniquement)
    - Alertes sur flags de sécurité
    """

    def __init__(self) -> None:
        from app.core.config import get_settings
        self.settings = get_settings()

    async def log_query(
        self,
        trace_id: str,
        session_id: str,
        query: str,
        language: str,
        profile: str,
        intent: str,
        retrieved_count: int,
        confidence_score: float,
        latency_ms: float,
        citations_count: int,
        safety_flags: list[str],
        model_used: str,
        escalation: bool = False,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Journalise une requête complète."""
        event = {
            "event_type": AuditEventType.QUERY.value,
            "trace_id": trace_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "query_hash": _hash_query(query),
            "query_length": len(query),
            "language": language,
            "profile": profile,
            "intent": intent,
            "retrieved_chunks": retrieved_count,
            "confidence_score": round(confidence_score, 4),
            "latency_ms": round(latency_ms, 2),
            "citations": citations_count,
            "safety_flags": safety_flags,
            "model_used": model_used,
            "escalation_triggered": escalation,
            **(metadata or {}),
        }

        if safety_flags:
            logger.warning("audit_query_with_flags", **event)
        else:
            logger.info("audit_query", **event)

        # Sprint 2 : persistance PostgreSQL
        # await self._persist_to_db(event)

    async def log_ingestion(
        self,
        trace_id: str,
        doc_id: str,
        filename: str,
        doc_type: str,
        language: str,
        chunks_count: int,
        ocr_used: bool,
        latency_ms: float,
        status: str,
        error: Optional[str] = None,
    ) -> None:
        """Journalise une ingestion de document."""
        event = {
            "event_type": AuditEventType.INGEST.value,
            "trace_id": trace_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "doc_id": doc_id,
            "filename": filename,
            "doc_type": doc_type,
            "language": language,
            "chunks_created": chunks_count,
            "ocr_used": ocr_used,
            "latency_ms": round(latency_ms, 2),
            "status": status,
        }
        if error:
            event["error"] = error
            logger.error("audit_ingest_error", **event)
        else:
            logger.info("audit_ingest", **event)

    async def log_security_event(
        self,
        trace_id: str,
        session_id: str,
        flag: SafetyFlag,
        description: str,
        query_preview: str,
        metadata: Optional[dict[str, Any]] = None,
    ) -> None:
        """Journalise un événement de sécurité (injection, hors-périmètre, etc.)."""
        event = {
            "event_type": AuditEventType.SECURITY_FLAG.value,
            "trace_id": trace_id,
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "flag": flag.value,
            "description": description,
            "query_preview": query_preview[:100],
            **(metadata or {}),
        }
        logger.warning("audit_security_event", **event)

    async def log_evaluation(
        self,
        baseline_id: str,
        num_queries: int,
        mrr: float,
        ndcg_5: float,
        citation_precision: float,
        hallucination_rate: float,
        isb: float,
        p50_ms: float,
        p95_ms: float,
        report_path: str,
    ) -> None:
        """Journalise un run d'évaluation."""
        logger.info(
            "audit_evaluation_run",
            event_type=AuditEventType.QUERY.value,
            timestamp=datetime.now(timezone.utc).isoformat(),
            baseline_id=baseline_id,
            num_queries=num_queries,
            mrr=round(mrr, 4),
            ndcg_at_5=round(ndcg_5, 4),
            citation_precision=round(citation_precision, 4),
            hallucination_rate=round(hallucination_rate, 4),
            isb=round(isb, 4),
            p50_ms=round(p50_ms, 1),
            p95_ms=round(p95_ms, 1),
            constraints_met={
                "citation_precision_ok": citation_precision >= 0.95,
                "hallucination_ok": hallucination_rate <= 0.05,
                "isb_ok": isb >= 0.85,
                "p50_ok": p50_ms <= 5000,
                "p95_ok": p95_ms <= 15000,
            },
            report_path=report_path,
        )


def _hash_query(query: str) -> str:
    """SHA-256 tronqué de la requête pour la traçabilité sans stocker le texte brut."""
    import hashlib
    return hashlib.sha256(query.encode()).hexdigest()[:16]


# ── Singleton ─────────────────────────────────────────────────────────────────

_audit_service: Optional[AuditService] = None


def get_audit_service() -> AuditService:
    global _audit_service
    if _audit_service is None:
        _audit_service = AuditService()
    return _audit_service
