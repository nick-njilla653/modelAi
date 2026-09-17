"""
GOV-AI 2.0 — Routes Fine-Tuning (/api/v1/finetune).
Gère l'upload de dossiers, le lancement d'entraînements, le suivi en temps réel
via WebSocket et l'intégration automatique du modèle fine-tuné.
"""
from __future__ import annotations

import asyncio
import json
import shutil
import tempfile
import threading
import time
import uuid
import zipfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.models.db_models import EvaluationComparison, FinetuneJob

logger = get_logger(__name__)
settings = get_settings()

router = APIRouter(prefix="/finetune", tags=["fine-tuning"])

# ── In-process job registry (progress + WebSocket broadcast) ─────────────────
_JOBS: dict[str, dict[str, Any]] = {}        # job_id → {status, progress, logs, result}
_WS_CLIENTS: dict[str, list[WebSocket]] = {}  # job_id → [WebSocket]
_CANCEL_FLAGS: dict[str, threading.Event] = {}  # set() → annulation demandée
_PAUSE_FLAGS:  dict[str, threading.Event] = {}  # set() → en cours, clear() → en pause

_FT_DATA_DIR = Path("models/finetune_data")
_FT_DATA_DIR.mkdir(parents=True, exist_ok=True)


async def _broadcast(job_id: str):
    """Diffuse l'état courant du job à tous les WebSocket connectés."""
    for ws in list(_WS_CLIENTS.get(job_id, [])):
        try:
            await ws.send_json(_JOBS[job_id])
        except Exception:
            pass


# ── Schemas Pydantic ──────────────────────────────────────────────────────────

class StartJobRequest(BaseModel):
    name: str
    target: str = "llm"               # "embedding" | "llm" | "reranker"
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-5
    pairs_per_chunk: int = 3
    lora_mode: bool = False            # True → LoRA (GPU requis), False → Modelfile
    dataset_id: str                    # ID retourné par /upload
    ingest_corpus: bool = True         # Ingérer aussi les docs dans KG + Milvus + ES
    doc_type: str = "autre"
    institution: str = ""
    jurisdiction: str = "national"


class ApplyModelRequest(BaseModel):
    job_id: str
    reindex_milvus: bool = True        # Re-indexer Milvus après swap embedding (recommandé)


# ── Routes ────────────────────────────────────────────────────────────────────

@router.post("/upload", summary="Upload d'un dossier (ZIP) pour fine-tuning")
async def upload_folder(
    file: UploadFile = File(..., description="Archive ZIP contenant les documents (PDF/TXT/MD)"),
) -> dict:
    """
    Accepte une archive ZIP d'un dossier complet de documents.
    Retourne un dataset_id à passer à /start.
    """
    if not file.filename or not file.filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="Fichier ZIP requis (.zip)")

    content = await file.read()
    if len(content) > 500 * 1024 * 1024:  # 500 MB max
        raise HTTPException(status_code=413, detail="Archive trop volumineuse (max 500 MB)")

    dataset_id = str(uuid.uuid4())
    dataset_dir = _FT_DATA_DIR / dataset_id
    dataset_dir.mkdir(parents=True)

    zip_path = dataset_dir / "upload.zip"
    zip_path.write_bytes(content)

    # Compter les fichiers dans le ZIP
    try:
        with zipfile.ZipFile(zip_path, "r") as zf:
            members = [m for m in zf.namelist() if not m.endswith("/")]
            supported = [m for m in members if Path(m).suffix.lower() in {".pdf", ".txt", ".md"}]
    except zipfile.BadZipFile:
        shutil.rmtree(dataset_dir, ignore_errors=True)
        raise HTTPException(status_code=400, detail="Archive ZIP invalide ou corrompue")

    return {
        "dataset_id": dataset_id,
        "zip_size_mb": round(len(content) / 1024 / 1024, 2),
        "total_files": len(members),
        "supported_files": len(supported),
        "message": f"{len(supported)} document(s) prêts pour le fine-tuning",
    }


@router.post("/start", summary="Démarre un job de fine-tuning")
async def start_job(request: StartJobRequest, background_tasks: BackgroundTasks) -> dict:
    """Lance l'entraînement en arrière-plan. Suivre la progression via /progress/{job_id}."""
    dataset_dir = _FT_DATA_DIR / request.dataset_id
    if not dataset_dir.exists():
        raise HTTPException(status_code=404, detail="dataset_id inconnu — re-uploadez d'abord.")

    target = request.target.lower()
    if target not in {"embedding", "llm", "reranker"}:
        raise HTTPException(status_code=400, detail="target doit être 'embedding', 'llm' ou 'reranker'")

    job_id = str(uuid.uuid4())
    _JOBS[job_id] = {
        "id": job_id,
        "name": request.name,
        "target": target,
        "status": "pending",
        "progress": 0,
        "logs": [],
        "result": None,
        "error": None,
        "created_at": time.time(),
    }

    background_tasks.add_task(
        _run_job,
        job_id=job_id,
        dataset_dir=dataset_dir,
        request=request,
    )

    return {"job_id": job_id, "status": "pending", "message": "Entraînement démarré en arrière-plan"}


@router.get("/jobs", summary="Liste tous les jobs de fine-tuning")
async def list_jobs() -> dict:
    """Retourne les jobs en mémoire + la base de données."""
    db_jobs = await _load_db_jobs()
    memory_jobs = list(_JOBS.values())
    # Fusionner: les jobs mémoire (plus récents) priment
    mem_ids = {j["id"] for j in memory_jobs}
    merged = memory_jobs + [j for j in db_jobs if j["id"] not in mem_ids]
    return {"jobs": merged, "total": len(merged)}


@router.get("/jobs/{job_id}", summary="Statut d'un job de fine-tuning")
async def get_job(job_id: str) -> dict:
    if job_id in _JOBS:
        return _JOBS[job_id]
    # Cherche en DB
    job = await _find_db_job(job_id)
    if job:
        return job
    raise HTTPException(status_code=404, detail="Job introuvable")


@router.post("/apply/{job_id}", summary="Intègre automatiquement le modèle fine-tuné")
async def apply_model(
    job_id: str,
    background_tasks: BackgroundTasks,
    reindex_milvus: bool = True,
) -> dict:
    """
    Swaps le modèle actif (embedding, LLM ou reranker) vers la version fine-tunée.
    Si target=embedding et reindex_milvus=True, re-calcule tous les vecteurs Milvus
    avec le nouveau modèle (opération obligatoire pour maintenir la cohérence de l'espace vectoriel).
    """
    job = _JOBS.get(job_id) or await _find_db_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Le job doit être 'completed' avant d'appliquer")

    from app.services.finetuning.model_manager import ModelManager
    mgr = ModelManager()
    result = job.get("result", {})
    target = job["target"]

    if target == "embedding":
        model_path = result.get("embedding_output")
        if not model_path:
            raise HTTPException(status_code=400, detail="Chemin du modèle d'embedding introuvable")
        entry = mgr.apply_embedding(job_id, model_path, job["name"])
        if reindex_milvus:
            background_tasks.add_task(_run_milvus_reindex, job_id=job_id, model_path=model_path)
        return {
            "applied": "embedding",
            "model_path": model_path,
            "entry": entry,
            "reindex_milvus": reindex_milvus,
            "message": "Re-indexation Milvus lancée en arrière-plan." if reindex_milvus else
                       "⚠️ Milvus non re-indexé — la recherche dense peut être dégradée.",
        }

    elif target == "reranker":
        model_path = result.get("reranker_output")
        if not model_path:
            raise HTTPException(status_code=400, detail="Chemin du modèle reranker introuvable")
        entry = mgr.apply_reranker(job_id, model_path, job["name"])
        return {"applied": "reranker", "model_path": model_path, "entry": entry}

    else:  # llm
        model_name = result.get("ollama_model_name")
        if not model_name:
            raise HTTPException(status_code=400, detail="Nom du modèle Ollama introuvable")
        entry = mgr.apply_llm(job_id, model_name, job["name"])
        return {"applied": "llm", "ollama_model_name": model_name, "entry": entry}


@router.get("/reindex-status/{job_id}", summary="Statut de la re-indexation Milvus")
async def reindex_status(job_id: str) -> dict:
    job = _JOBS.get(job_id, {})
    return {
        "reindex_status": job.get("reindex_status", "not_started"),
        "reindex_progress": job.get("reindex_progress", 0),
        "reindex_result": job.get("reindex_result"),
    }


@router.post("/rollback/{model_type}", summary="Retour au modèle précédent")
async def rollback(model_type: str) -> dict:
    if model_type not in {"embedding", "llm"}:
        raise HTTPException(status_code=400, detail="model_type doit être 'embedding' ou 'llm'")
    from app.services.finetuning.model_manager import ModelManager
    previous = ModelManager().rollback(model_type)
    if not previous:
        raise HTTPException(status_code=404, detail="Aucun modèle précédent trouvé")
    return {"rolled_back": model_type, "previous": previous}


@router.post("/jobs/{job_id}/cancel", summary="Annule un job en cours")
async def cancel_job(job_id: str) -> dict:
    """Envoie le signal d'annulation. Le job s'arrête à la prochaine étape sûre."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job["status"] in ("completed", "failed", "cancelled"):
        raise HTTPException(status_code=400, detail=f"Job déjà terminé ({job['status']})")

    # Débloquer une éventuelle pause d'abord, puis signaler l'annulation
    if job_id in _PAUSE_FLAGS:
        _PAUSE_FLAGS[job_id].set()
    if job_id in _CANCEL_FLAGS:
        _CANCEL_FLAGS[job_id].set()

    _JOBS[job_id]["status"] = "cancelling"
    _JOBS[job_id]["logs"].append("Annulation demandée — arrêt à la prochaine étape…")
    await _broadcast(job_id)
    return {"cancelled": job_id, "message": "Signal d'annulation envoyé"}


@router.post("/jobs/{job_id}/pause", summary="Met un job en pause")
async def pause_job(job_id: str) -> dict:
    """Suspend l'entraînement. Le job reprend depuis ce point avec /resume."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    active = {"running", "dataset_building", "leakage_check", "ingesting_corpus", "training"}
    if job["status"] not in active:
        raise HTTPException(status_code=400, detail=f"Impossible de mettre en pause un job '{job['status']}'")

    if job_id in _PAUSE_FLAGS:
        _PAUSE_FLAGS[job_id].clear()  # clear = en pause

    _JOBS[job_id]["status"] = "paused"
    _JOBS[job_id]["logs"].append("Job mis en pause — cliquez sur Reprendre pour continuer.")
    await _broadcast(job_id)
    return {"paused": job_id}


@router.post("/jobs/{job_id}/resume", summary="Reprend un job en pause")
async def resume_job(job_id: str) -> dict:
    """Reprend l'entraînement depuis le point de pause."""
    job = _JOBS.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job["status"] != "paused":
        raise HTTPException(status_code=400, detail="Le job n'est pas en pause")

    if job_id in _PAUSE_FLAGS:
        _PAUSE_FLAGS[job_id].set()  # set = reprendre

    _JOBS[job_id]["status"] = "running"
    _JOBS[job_id]["logs"].append("Job repris.")
    await _broadcast(job_id)
    return {"resumed": job_id}


@router.get("/active-models", summary="Modèles actuellement actifs")
async def active_models() -> dict:
    from app.services.finetuning.model_manager import ModelManager
    return ModelManager().get_active()


@router.post("/evaluate/{job_id}", summary="Lance une évaluation comparative")
async def run_evaluation(job_id: str, background_tasks: BackgroundTasks) -> dict:
    """Compare modèle original vs fine-tuné via LLM-juge, BLEU, ROUGE-L."""
    job = _JOBS.get(job_id) or await _find_db_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    if job.get("status") != "completed":
        raise HTTPException(status_code=400, detail="Job pas encore terminé")

    background_tasks.add_task(_run_evaluation, job_id=job_id, job=job)
    return {"status": "evaluation_started", "job_id": job_id}


@router.get("/evaluate/{job_id}/results", summary="Résultats de l'évaluation comparative")
async def get_evaluation_results(job_id: str) -> dict:
    job = _JOBS.get(job_id) or await _find_db_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Job introuvable")
    eval_result = job.get("evaluation")
    if not eval_result:
        raise HTTPException(status_code=404, detail="Évaluation pas encore disponible")
    from app.services.finetuning.model_manager import REFERENCE_BENCHMARKS
    eval_result["reference_benchmarks"] = REFERENCE_BENCHMARKS
    return eval_result


@router.delete("/jobs/{job_id}", summary="Supprime un job et ses données")
async def delete_job(job_id: str) -> dict:
    if job_id in _JOBS:
        del _JOBS[job_id]
    data_dir = _FT_DATA_DIR / job_id
    if data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    return {"deleted": job_id}


# ── WebSocket progress ────────────────────────────────────────────────────────

@router.websocket("/progress/{job_id}")
async def progress_websocket(ws: WebSocket, job_id: str):
    """Stream WebSocket de progression du job (progress 0→100, logs, statut)."""
    await ws.accept()
    _WS_CLIENTS.setdefault(job_id, []).append(ws)
    try:
        # Envoie l'état courant immédiatement
        if job_id in _JOBS:
            await ws.send_json(_JOBS[job_id])
        # Maintient la connexion jusqu'à déconnexion
        while True:
            try:
                await asyncio.wait_for(ws.receive_text(), timeout=30.0)
            except asyncio.TimeoutError:
                await ws.send_json({"ping": True})
    except WebSocketDisconnect:
        pass
    finally:
        if job_id in _WS_CLIENTS:
            _WS_CLIENTS[job_id] = [c for c in _WS_CLIENTS[job_id] if c != ws]


# ── Helpers partagés ──────────────────────────────────────────────────────────

async def _broadcast(job_id: str):
    """Diffuse l'état courant du job à tous les WebSocket connectés."""
    for ws in list(_WS_CLIENTS.get(job_id, [])):
        try:
            await ws.send_json(_JOBS[job_id])
        except Exception:
            pass


# ── Background tasks ──────────────────────────────────────────────────────────

async def _run_job(job_id: str, dataset_dir: Path, request: StartJobRequest):
    """Exécute le pipeline complet : dataset → training → (auto-save DB)."""

    # ── Initialisation des signaux cancel / pause ──────────────────────────
    cancel_ev = _CANCEL_FLAGS[job_id] = threading.Event()
    pause_ev  = _PAUSE_FLAGS[job_id]  = threading.Event()
    pause_ev.set()  # état initial : en cours

    async def _update(progress: int, log: str, status: str = "running"):
        _JOBS[job_id]["progress"] = progress
        _JOBS[job_id]["status"] = status
        _JOBS[job_id]["logs"].append(log)
        logger.debug("finetune_progress", job=job_id, pct=progress, msg=log)
        await _broadcast(job_id)

    async def _check(step: str = ""):
        """Lève CancelledError si annulation demandée, bloque si en pause."""
        if cancel_ev.is_set():
            raise asyncio.CancelledError(step)
        while not pause_ev.is_set():
            if cancel_ev.is_set():
                raise asyncio.CancelledError(step)
            await asyncio.sleep(0.4)

    try:
        _JOBS[job_id]["status"] = "dataset_building"
        await _update(5, "Démarrage du pipeline fine-tuning…")

        from app.core.config import get_settings
        cfg = get_settings()
        job_output_dir = dataset_dir / "output"
        job_output_dir.mkdir(exist_ok=True)

        # ── Étape 1 : Construction du dataset ──────────────────────────────
        from app.services.finetuning.dataset_builder import DatasetBuilder

        builder = DatasetBuilder(
            llm_model=cfg.llm_model,
            ollama_host=cfg.llm_base_url,
        )

        dataset_info = await builder.build_from_zip(
            zip_path=dataset_dir / "upload.zip",
            output_dir=job_output_dir / "dataset",
            pairs_per_chunk=request.pairs_per_chunk,
            progress_cb=_update,
            cancel_event=cancel_ev,
            pause_event=pause_ev,
        )
        _JOBS[job_id]["dataset_info"] = dataset_info
        await _check("après dataset building")

        # ── Étape 1b : Vérification du data leakage ──────────────────────────
        _JOBS[job_id]["status"] = "leakage_check"
        await _update(56, "Vérification du data leakage (L1 source · L2 mots-clés · L3 sémantique)…")
        try:
            from app.services.finetuning.leakage_detector import run_leakage_check
            loop = asyncio.get_event_loop()

            # Paires QA (check complet avec sémantique)
            leakage_result = await loop.run_in_executor(
                None,
                lambda: run_leakage_check(
                    qa_pairs_path=dataset_info["qa_path"],
                    use_semantic=True,
                ),
            )
            # Paires embedding (check rapide sans sémantique — fichier différent)
            await loop.run_in_executor(
                None,
                lambda: run_leakage_check(
                    qa_pairs_path=dataset_info["embedding_path"],
                    use_semantic=False,
                ),
            )

            lk_summary = leakage_result.get("summary", {})
            lk_details = leakage_result.get("report", [])
            _JOBS[job_id]["leakage_report"] = lk_summary
            _JOBS[job_id]["leakage_details"] = lk_details

            leaked_n = lk_summary.get("leaked", 0)
            clean_n  = lk_summary.get("clean", 0)
            pct      = lk_summary.get("leakage_pct", 0.0)
            levels   = ", ".join(lk_summary.get("levels_triggered", [])) or "aucun"

            if clean_n == 0:
                raise ValueError(
                    f"Toutes les paires générées ({leaked_n}) chevauchent le dataset d'évaluation "
                    f"(niveaux : {levels}). Utilisez des documents différents."
                )

            await _update(
                58,
                f"Leakage : {leaked_n} paire(s) exclue(s) ({pct}%) · {clean_n} propres conservées · Niveaux : {levels}",
            )

        except ValueError:
            raise
        except Exception as exc:
            logger.warning("leakage_check_failed", job=job_id, error=str(exc))
            await _update(58, f"[Attention] Vérification leakage ignorée : {exc}")

        await _check("après leakage check")

        # ── Étape 1c (optionnelle) : Ingestion corpus (KG + Milvus + ES) ────────
        if request.ingest_corpus:
            _JOBS[job_id]["status"] = "ingesting_corpus"
            await _update(56, "Ingestion des documents dans le corpus (KG + Milvus + ES)…")
            try:
                from app.services.finetuning.corpus_reindexer import CorpusReindexer
                reindexer = CorpusReindexer()
                ingest_result = await reindexer.ingest_corpus_from_zip(
                    zip_path=dataset_dir / "upload.zip",
                    doc_type=request.doc_type,
                    institution=request.institution,
                    jurisdiction=request.jurisdiction,
                    progress_cb=lambda pct, msg: _update(
                        59 + int(pct * 0.04), f"[Corpus] {msg}"
                    ),
                )
                _JOBS[job_id]["corpus_ingestion"] = ingest_result
                await _update(63, f"Corpus enrichi : {ingest_result.get('ingested', 0)} documents ajoutés (KG + Milvus + ES).")
            except Exception as exc:
                logger.warning("corpus_ingestion_partial_fail", job=job_id, error=str(exc))
                _JOBS[job_id]["corpus_ingestion"] = {"error": str(exc)}
                await _update(63, f"[Attention] Ingestion corpus partielle : {exc}")

        await _check("après ingestion corpus")

        # ── Étape 2 : Entraînement ─────────────────────────────────────────
        _JOBS[job_id]["status"] = "training"
        await _update(64, f"Dataset prêt ({dataset_info['total_pairs']} paires). Lancement de l'entraînement…")
        await _check("avant entraînement")

        result: dict = {}

        if request.target == "embedding":
            from app.services.finetuning.embedding_trainer import EmbeddingTrainer
            trainer = EmbeddingTrainer(base_model=cfg.embedding_model)
            embed_out = job_output_dir / "embedding_model"
            train_result = await trainer.train(
                embedding_pairs_path=dataset_info["embedding_path"],
                output_dir=embed_out,
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                progress_cb=_update,
                cancel_event=cancel_ev,
                pause_event=pause_ev,
            )
            result = {**train_result, "embedding_output": str(embed_out)}

        elif request.target == "reranker":
            from app.services.finetuning.reranker_trainer import RerankerTrainer
            trainer_r = RerankerTrainer(
                base_model=cfg.reranker_model,
                device=cfg.reranker_device,
            )
            reranker_out = job_output_dir / "reranker_model"
            train_result = await trainer_r.train(
                qa_pairs_path=dataset_info["qa_path"],
                output_dir=reranker_out,
                epochs=request.epochs,
                batch_size=min(request.batch_size, 8),
                learning_rate=request.learning_rate,
                progress_cb=_update,
                cancel_event=cancel_ev,
                pause_event=pause_ev,
            )
            result = {**train_result, "reranker_output": str(reranker_out)}

        else:  # llm
            from app.services.finetuning.llm_trainer import LLMTrainer
            trainer = LLMTrainer(
                base_model=cfg.llm_model,
                ollama_host=cfg.llm_base_url,
            )
            model_name = f"govai-ft-{job_id[:8]}"

            if request.lora_mode:
                result = await trainer.train_lora(
                    qa_pairs_path=dataset_info["qa_path"],
                    output_dir=job_output_dir / "lora",
                    model_name=model_name,
                    epochs=request.epochs,
                    learning_rate=request.learning_rate,
                    progress_cb=_update,
                    cancel_event=cancel_ev,
                )
            else:
                result = await trainer.create_modelfile_model(
                    qa_pairs_path=dataset_info["qa_path"],
                    output_dir=job_output_dir / "modelfile",
                    model_name=model_name,
                    progress_cb=_update,
                    cancel_event=cancel_ev,
                )

            result["ollama_model_name"] = model_name

        await _update(95, "Entraînement terminé. Sauvegarde…", status="completed")

        _JOBS[job_id]["status"] = "completed"
        _JOBS[job_id]["progress"] = 100
        _JOBS[job_id]["result"] = result
        _JOBS[job_id]["logs"].append("Fine-tuning complété avec succès.")

        # Broadcast final
        for ws in list(_WS_CLIENTS.get(job_id, [])):
            try:
                await ws.send_json(_JOBS[job_id])
            except Exception:
                pass

        await _save_job_to_db(job_id, request, dataset_info, result)

    except asyncio.CancelledError as exc:
        reason = str(exc) if str(exc) else "utilisateur"
        _JOBS[job_id]["status"] = "cancelled"
        _JOBS[job_id]["logs"].append(f"Fine-tuning annulé ({reason}).")
        logger.info("finetune_job_cancelled", job=job_id, reason=reason)
        await _broadcast(job_id)
    except Exception as exc:
        if cancel_ev.is_set():
            # Exception levée depuis un thread d'entraînement après cancel
            _JOBS[job_id]["status"] = "cancelled"
            _JOBS[job_id]["logs"].append("Fine-tuning annulé pendant l'entraînement.")
            logger.info("finetune_job_cancelled_in_training", job=job_id)
        else:
            logger.error("finetune_job_error", job=job_id, error=str(exc), exc_info=True)
            _JOBS[job_id]["status"] = "failed"
            _JOBS[job_id]["error"] = str(exc)
            _JOBS[job_id]["logs"].append(f"ERREUR : {exc}")
        await _broadcast(job_id)


async def _run_milvus_reindex(job_id: str, model_path: str):
    """
    Re-calcule tous les vecteurs Milvus avec le nouveau modèle d'embedding.
    Obligatoire après apply_embedding() pour maintenir la cohérence vectorielle.
    """
    _JOBS.setdefault(job_id, {})["reindex_status"] = "running"

    async def _cb(pct: int, msg: str):
        _JOBS[job_id]["reindex_progress"] = pct
        _JOBS[job_id]["reindex_log"] = msg
        for ws in list(_WS_CLIENTS.get(job_id, [])):
            try:
                await ws.send_json({**_JOBS[job_id], "_reindex_update": True})
            except Exception:
                pass

    try:
        from app.services.finetuning.corpus_reindexer import CorpusReindexer
        result = await CorpusReindexer().reindex_milvus(
            new_embedding_model_path=model_path,
            progress_cb=_cb,
        )
        _JOBS[job_id]["reindex_status"] = "completed"
        _JOBS[job_id]["reindex_progress"] = 100
        _JOBS[job_id]["reindex_result"] = result
        logger.info("milvus_reindex_complete", job=job_id, reindexed=result.get("reindexed"))
    except Exception as exc:
        logger.error("milvus_reindex_error", job=job_id, error=str(exc), exc_info=True)
        _JOBS[job_id]["reindex_status"] = "failed"
        _JOBS[job_id]["reindex_result"] = {"error": str(exc)}


async def _run_evaluation(job_id: str, job: dict):
    """Lance l'évaluation comparative en arrière-plan."""
    try:
        from app.core.config import get_settings
        from app.services.finetuning.model_evaluator import ModelEvaluator

        cfg = get_settings()
        result = job.get("result", {})
        dataset_info = job.get("dataset_info", {})
        qa_path = dataset_info.get("qa_path")
        finetuned_model = result.get("ollama_model_name") or result.get("embedding_output")

        if not qa_path or not finetuned_model:
            _JOBS[job_id]["evaluation"] = {"error": "Données insuffisantes pour l'évaluation"}
            return

        evaluator = ModelEvaluator(
            original_model=cfg.llm_model,
            ollama_host=cfg.llm_base_url,
        )

        eval_result = await evaluator.run_comparison(
            qa_pairs_path=qa_path,
            finetuned_model=finetuned_model,
        )

        _JOBS[job_id]["evaluation"] = eval_result

        # Persist en DB
        await _save_evaluation_to_db(job_id, eval_result)

    except Exception as exc:
        logger.error("evaluation_error", job=job_id, error=str(exc), exc_info=True)
        _JOBS[job_id]["evaluation"] = {"error": str(exc)}


# ── DB helpers ────────────────────────────────────────────────────────────────

async def _save_job_to_db(job_id: str, request: StartJobRequest, dataset_info: dict, result: dict):
    try:
        from app.storage.postgres_client import get_db_session
        async with get_db_session() as db:
            job_rec = FinetuneJob(
                id=job_id,
                name=request.name,
                target=request.target,
                status="completed",
                base_model=settings.llm_model if request.target == "llm" else settings.embedding_model,
                output_model_path=result.get("embedding_output") or result.get("adapter_path"),
                ollama_model_name=result.get("ollama_model_name"),
                documents_count=dataset_info.get("files_processed", 0),
                dataset_size=dataset_info.get("total_pairs", 0),
                epochs=request.epochs,
                batch_size=request.batch_size,
                learning_rate=request.learning_rate,
                progress=100.0,
                loss_history=result.get("loss_history"),
                logs=_JOBS.get(job_id, {}).get("logs"),
            )
            db.add(job_rec)
    except Exception as exc:
        logger.warning("finetune_db_save_failed", error=str(exc))


async def _save_evaluation_to_db(job_id: str, eval_result: dict):
    try:
        from app.storage.postgres_client import get_db_session
        async with get_db_session() as db:
            rec = EvaluationComparison(
                job_id=job_id,
                original_metrics=eval_result.get("original_metrics"),
                finetuned_metrics=eval_result.get("finetuned_metrics"),
                reference_metrics=eval_result.get("reference_metrics"),
                sample_responses=eval_result.get("sample_responses"),
            )
            db.add(rec)
    except Exception as exc:
        logger.warning("eval_db_save_failed", error=str(exc))


async def _load_db_jobs() -> list[dict]:
    try:
        from app.storage.postgres_client import get_db_session
        import sqlalchemy as sa
        async with get_db_session() as db:
            rows = (await db.execute(
                sa.select(FinetuneJob).order_by(FinetuneJob.created_at.desc()).limit(50)
            )).scalars().all()
            return [
                {
                    "id": r.id, "name": r.name, "target": r.target, "status": r.status,
                    "progress": r.progress, "epochs": r.epochs, "base_model": r.base_model,
                    "documents_count": r.documents_count, "dataset_size": r.dataset_size,
                    "loss_history": r.loss_history or [], "logs": r.logs or [],
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "result": {
                        "ollama_model_name": r.ollama_model_name,
                        "embedding_output": r.output_model_path,
                    },
                }
                for r in rows
            ]
    except Exception as exc:
        logger.debug("load_db_jobs_failed", error=str(exc))
        return []


async def _find_db_job(job_id: str) -> Optional[dict]:
    jobs = await _load_db_jobs()
    return next((j for j in jobs if j["id"] == job_id), None)
