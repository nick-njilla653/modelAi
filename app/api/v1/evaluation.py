"""
GOV-AI 2.0 — Route d'évaluation (/api/v1/evaluation).
"""
from __future__ import annotations

from fastapi import APIRouter, HTTPException

from app.core.logging import get_logger
from app.models.schemas import EvaluationRequest

router = APIRouter(tags=["evaluation"])
logger = get_logger(__name__)


@router.post("/evaluation/run", summary="Lance une évaluation sur le dataset annoté")
async def run_evaluation(request: EvaluationRequest) -> dict:
    """
    Lance l'évaluation complète sur le dataset bilingue annoté.
    Retourne les métriques : P@k, R@k, MRR, nDCG@k, citation precision, ISB.
    """
    try:
        from app.services.evaluation.evaluation_service import EvaluationService
        svc = EvaluationService()
        report = await svc.run_evaluation(
            dataset_path=request.dataset_path,
            baseline_id=request.baseline_id,
            k_values=request.k_values,
        )
        return report.to_dict()
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except Exception as exc:
        logger.error("evaluation_endpoint_error", error=str(exc), exc_info=True)
        raise HTTPException(status_code=500, detail=f"Erreur d'évaluation : {exc}")


@router.get("/evaluation/baselines", summary="Liste des baselines disponibles")
async def list_baselines() -> dict:
    """Retourne les baselines d'évaluation B0→B4."""
    from app.services.evaluation.evaluation_service import EvaluationService
    return {"baselines": EvaluationService.BASELINES}
