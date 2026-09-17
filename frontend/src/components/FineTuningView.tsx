import { useCallback, useEffect, useRef, useState } from "react";

import * as api from "../api";
import { IconUpload } from "../icons";
import { MetricComparison } from "./MetricComparison";
import type {
  ActiveModels,
  DatasetUploadResult,
  FinetuneJob,
  HealthResponse,
  ModelComparison,
} from "../types";

const TARGETS = [
  { value: "llm", label: "Modèle de génération", hint: "Répond aux questions" },
  { value: "embedding", label: "Modèle d'embedding", hint: "Vectorise les passages — impose une réindexation" },
  { value: "reranker", label: "Reranker", hint: "Reclasse les extraits récupérés" },
];

const STATUS_LABELS: Record<string, string> = {
  pending: "En attente",
  running: "En cours",
  paused: "En pause",
  completed: "Terminé",
  failed: "Échec",
  cancelled: "Annulé",
};

function formatDate(epoch: number): string {
  if (!epoch) return "—";
  return new Date(epoch * 1000).toLocaleString("fr-FR");
}

export function FineTuningView() {
  const [dataset, setDataset] = useState<DatasetUploadResult | null>(null);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);

  const [name, setName] = useState("");
  const [target, setTarget] = useState("llm");
  const [loraMode, setLoraMode] = useState(false);
  const [epochs, setEpochs] = useState(3);
  const [batchSize, setBatchSize] = useState(16);
  const [learningRate, setLearningRate] = useState("2e-5");
  const [pairsPerChunk, setPairsPerChunk] = useState(3);
  const [ingestCorpus, setIngestCorpus] = useState(true);
  const [reindexMilvus, setReindexMilvus] = useState(true);

  const [jobs, setJobs] = useState<FinetuneJob[]>([]);
  const [activeJob, setActiveJob] = useState<FinetuneJob | null>(null);
  const [models, setModels] = useState<ActiveModels | null>(null);
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [comparison, setComparison] = useState<ModelComparison | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  const socketRef = useRef<WebSocket | null>(null);
  const logRef = useRef<HTMLPreElement>(null);

  const refresh = useCallback(async () => {
    try {
      const [jobList, activeModels] = await Promise.all([
        api.listFinetuneJobs(),
        api.getActiveModels(),
      ]);
      setJobs(jobList.jobs);
      setModels(activeModels);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Impossible de charger les jobs.");
    }
  }, []);

  useEffect(() => {
    void refresh();
    void api.getHealth().then(setHealth).catch(() => setHealth(null));
  }, [refresh]);

  // Le journal suit sa propre fin, sans arracher le défilement à l'utilisateur
  // s'il est remonté pour lire.
  useEffect(() => {
    const el = logRef.current;
    if (!el) return;
    if (el.scrollHeight - el.scrollTop - el.clientHeight < 80) el.scrollTop = el.scrollHeight;
  }, [activeJob?.logs]);

  const watchJob = useCallback((jobId: string) => {
    socketRef.current?.close();
    socketRef.current = api.openFinetuneProgress(jobId, (job) => {
      setActiveJob(job);
      if (["completed", "failed", "cancelled"].includes(job.status)) {
        void refresh();
      }
    });
  }, [refresh]);

  useEffect(() => () => socketRef.current?.close(), []);

  const upload = async (file: File) => {
    setUploading(true);
    setUploadError(null);
    try {
      setDataset(await api.uploadDataset(file));
      if (!name) setName(file.name.replace(/\.zip$/i, ""));
    } catch (err) {
      setUploadError(err instanceof Error ? err.message : "Le dépôt a échoué.");
    } finally {
      setUploading(false);
    }
  };

  const start = async () => {
    if (!dataset) return;
    setError(null);
    setNotice(null);
    try {
      const started = await api.startFinetune({
        name: name.trim() || "Job sans nom",
        target,
        epochs,
        batch_size: batchSize,
        learning_rate: Number.parseFloat(learningRate) || 2e-5,
        pairs_per_chunk: pairsPerChunk,
        lora_mode: loraMode,
        dataset_id: dataset.dataset_id,
        ingest_corpus: ingestCorpus,
      });
      setComparison(null);
      watchJob(started.job_id);
      setActiveJob({
        id: started.job_id,
        name: name.trim() || "Job sans nom",
        target,
        status: "pending",
        progress: 0,
        logs: [],
        result: null,
        error: null,
        created_at: Date.now() / 1000,
      });
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Le démarrage a échoué.");
    }
  };

  const act = async (label: string, action: () => Promise<unknown>) => {
    setError(null);
    try {
      await action();
      setNotice(label);
      await refresh();
    } catch (err) {
      setError(err instanceof Error ? err.message : `${label} — échec.`);
    }
  };

  const loadComparison = async (jobId: string) => {
    setError(null);
    try {
      setComparison(await api.getComparison(jobId));
    } catch (err) {
      setError(
        err instanceof Error
          ? `${err.message} — lancez d'abord l'évaluation comparative.`
          : "Résultats indisponibles.",
      );
    }
  };

  // Le registre est figé au premier démarrage : il peut désigner un modèle que
  // le système n'utilise plus. Le taire rendrait un « rollback » trompeur.
  const registryLlm = models?.llm_model?.ollama_model_name ?? null;
  const runningLlm = health?.model ?? null;
  const llmDiverges = Boolean(registryLlm && runningLlm && registryLlm !== runningLlm);

  const leakage = activeJob?.leakage;
  const leakRatio = leakage?.leak_ratio ?? 0;

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Fine-tuning</h1>
          <p className="view-sub">
            Spécialiser un modèle sur un corpus juridique, mesurer ce que
            l'entraînement apporte, puis décider de l'appliquer ou non.
          </p>
        </div>
        <button type="button" className="btn-ghost" onClick={() => void refresh()}>
          Rafraîchir
        </button>
      </div>

      {notice && <p className="job-detail job-ok">{notice}</p>}
      {error && <p className="field-error">{error}</p>}

      {/* ── Modèles actifs ─────────────────────────────────────────────── */}
      <section className="ft-section">
        <h2>Modèles actifs</h2>

        {llmDiverges && (
          <div className="notice notice-warn">
            <div className="notice-body">
              <span className="notice-title">Registre désynchronisé</span>
              Le registre déclare <span className="mono">{registryLlm}</span> alors que le
              système répond avec <span className="mono">{runningLlm}</span>. Un retour
              arrière restaurerait un état qui ne correspond pas à la configuration
              réelle.
            </div>
          </div>
        )}

        <div className="ft-models">
          {([
            ["llm_model", "Génération", "llm"],
            ["embedding_model", "Embedding", "embedding"],
            ["reranker_model", "Reranker", "reranker"],
          ] as const).map(([key, label, type]) => {
            const entry = models?.[key];
            const value = entry?.ollama_model_name ?? entry?.model_path ?? "—";
            return (
              <div key={key} className="ft-model-card">
                <span className="viz-tile-label">{label}</span>
                <span className="mono ft-model-name">{value}</span>
                <span className="field-hint">
                  {entry?.job_id ? `issu du job ${entry.job_id.slice(0, 8)}` : "modèle de base"}
                </span>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => void act(`Retour arrière ${label} effectué.`, () => api.rollbackModel(type))}
                >
                  Revenir au précédent
                </button>
              </div>
            );
          })}
        </div>
      </section>

      {/* ── Corpus d'entraînement ──────────────────────────────────────── */}
      <section className="ft-section">
        <h2>Corpus d'entraînement</h2>
        <p className="view-sub">
          Une archive ZIP de documents PDF, TXT ou MD. Les paires question-réponse
          en seront dérivées automatiquement.
        </p>

        <label className="field">
          <span className="field-label">Archive</span>
          <input
            type="file"
            accept=".zip"
            onChange={(event) => {
              const file = event.target.files?.[0];
              if (file) void upload(file);
            }}
          />
        </label>

        {uploading && <p className="job-detail">Dépôt en cours…</p>}
        {uploadError && <p className="field-error">{uploadError}</p>}

        {dataset && (
          <div className="ft-dataset">
            <span className="mono">{dataset.dataset_id.slice(0, 8)}</span>
            <span>
              {dataset.supported_files} document{dataset.supported_files > 1 ? "s" : ""} exploitable
              {dataset.supported_files > 1 ? "s" : ""} sur {dataset.total_files} · {dataset.zip_size_mb} Mo
            </span>
          </div>
        )}
      </section>

      {/* ── Configuration ──────────────────────────────────────────────── */}
      <section className="ft-section">
        <h2>Configuration</h2>

        <div className="ft-grid">
          <label className="field">
            <span className="field-label">Nom du job</span>
            <input
              type="text"
              value={name}
              onChange={(event) => setName(event.target.value)}
              placeholder="Ex. : GOV-AI Sprint 4 v1"
            />
          </label>

          <label className="field">
            <span className="field-label">Cible d'entraînement</span>
            <select value={target} onChange={(event) => setTarget(event.target.value)}>
              {TARGETS.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
            <span className="field-hint">
              {TARGETS.find((t) => t.value === target)?.hint}
            </span>
          </label>

          <label className="field">
            <span className="field-label">Époques</span>
            <input
              type="number"
              min={1}
              value={epochs}
              onChange={(event) => setEpochs(Number(event.target.value))}
            />
          </label>

          <label className="field">
            <span className="field-label">Taille de lot</span>
            <input
              type="number"
              min={1}
              value={batchSize}
              onChange={(event) => setBatchSize(Number(event.target.value))}
            />
          </label>

          <label className="field">
            <span className="field-label">Taux d'apprentissage</span>
            <input
              type="text"
              value={learningRate}
              onChange={(event) => setLearningRate(event.target.value)}
            />
          </label>

          <label className="field">
            <span className="field-label">Paires QA par passage</span>
            <input
              type="number"
              min={1}
              value={pairsPerChunk}
              onChange={(event) => setPairsPerChunk(Number(event.target.value))}
            />
          </label>
        </div>

        {target === "llm" && (
          <label className="field-inline">
            <input
              type="checkbox"
              checked={loraMode}
              onChange={(event) => setLoraMode(event.target.checked)}
            />
            <span>
              Entraînement LoRA
              <span className="field-hint">
                Exige un GPU disponible. Sans cela, un Modelfile Ollama est produit —
                la carte de cette machine ne dispose que de 4 Go.
              </span>
            </span>
          </label>
        )}

        <label className="field-inline">
          <input
            type="checkbox"
            checked={ingestCorpus}
            onChange={(event) => setIngestCorpus(event.target.checked)}
          />
          <span>
            Ingérer aussi ces documents dans le corpus
            <span className="field-hint">
              Les rend interrogeables, en plus de servir à l'entraînement.
            </span>
          </span>
        </label>

        <button
          type="button"
          className="btn-primary"
          onClick={() => void start()}
          disabled={!dataset}
        >
          <IconUpload size={15} /> Lancer l'entraînement
        </button>
        {!dataset && (
          <span className="field-hint">Déposez d'abord une archive.</span>
        )}
      </section>

      {/* ── Progression ────────────────────────────────────────────────── */}
      {activeJob && (
        <section className="ft-section">
          <h2>
            {activeJob.name}{" "}
            <span className={`ft-status ${activeJob.status}`}>
              {STATUS_LABELS[activeJob.status] ?? activeJob.status}
            </span>
          </h2>

          <div
            className="progress"
            role="progressbar"
            aria-valuenow={activeJob.progress}
            aria-valuemin={0}
            aria-valuemax={100}
          >
            <div className="progress-fill" style={{ width: `${activeJob.progress}%` }} />
          </div>
          <p className="job-detail">{activeJob.progress} %</p>

          {leakage && (
            <div className={`notice ${leakRatio > 0.3 ? "notice-bad" : "notice-warn"}`}>
              <div className="notice-body">
                <span className="notice-title">Fuite de données</span>
                {leakage.leaked_pairs ?? 0} paire{(leakage.leaked_pairs ?? 0) > 1 ? "s" : ""} sur{" "}
                {leakage.total_pairs ?? 0} figurent aussi dans le jeu d'évaluation
                {leakRatio ? ` (${(leakRatio * 100).toFixed(1)} %)` : ""}. Les métriques
                obtenues sur ces paires surestiment le modèle.
                {leakage.warning && <> {leakage.warning}</>}
              </div>
            </div>
          )}

          {activeJob.error && <p className="field-error">{activeJob.error}</p>}

          <pre className="ft-log" ref={logRef}>
            {activeJob.logs.length > 0 ? activeJob.logs.join("\n") : "En attente de journal…"}
          </pre>

          <div className="ft-actions">
            {activeJob.status === "running" && (
              <>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => void act("Job mis en pause.", () => api.pauseFinetuneJob(activeJob.id))}
                >
                  Mettre en pause
                </button>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => void act("Job annulé.", () => api.cancelFinetuneJob(activeJob.id))}
                >
                  Annuler
                </button>
              </>
            )}
            {activeJob.status === "paused" && (
              <button
                type="button"
                className="btn-ghost"
                onClick={() => void act("Job repris.", () => api.resumeFinetuneJob(activeJob.id))}
              >
                Reprendre
              </button>
            )}
            {activeJob.status === "completed" && (
              <>
                <button
                  type="button"
                  className="btn-primary"
                  onClick={() =>
                    void act(
                      "Modèle appliqué.",
                      () => api.applyFinetunedModel(activeJob.id, reindexMilvus),
                    )
                  }
                >
                  Appliquer ce modèle
                </button>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() =>
                    void act("Évaluation comparative lancée.", () => api.startComparison(activeJob.id))
                  }
                >
                  Comparer à l'original
                </button>
                <button
                  type="button"
                  className="btn-ghost"
                  onClick={() => void loadComparison(activeJob.id)}
                >
                  Voir les résultats
                </button>
              </>
            )}
          </div>

          {activeJob.status === "completed" && target === "embedding" && (
            <label className="field-inline">
              <input
                type="checkbox"
                checked={reindexMilvus}
                onChange={(event) => setReindexMilvus(event.target.checked)}
              />
              <span>
                Réindexer Milvus après application
                <span className="field-hint">
                  Obligatoire pour un modèle d'embedding : les vecteurs déjà stockés
                  ont été produits par l'ancien modèle et ne sont plus comparables.
                </span>
              </span>
            </label>
          )}
        </section>
      )}

      {/* ── Comparaison ────────────────────────────────────────────────── */}
      {comparison && (
        <section className="ft-section">
          <MetricComparison comparison={comparison} />
        </section>
      )}

      {/* ── Historique ─────────────────────────────────────────────────── */}
      <section className="ft-section">
        <h2>Jobs</h2>
        {jobs.length === 0 ? (
          <p className="panel-empty">Aucun entraînement lancé pour l'instant.</p>
        ) : (
          <div className="table-scroll">
            <table className="metric-table">
              <thead>
                <tr>
                  <th scope="col">Nom</th>
                  <th scope="col">Cible</th>
                  <th scope="col">État</th>
                  <th scope="col">Créé</th>
                  <th scope="col" />
                </tr>
              </thead>
              <tbody>
                {jobs.map((job) => (
                  <tr key={job.id}>
                    <th scope="row">{job.name}</th>
                    <td>{job.target}</td>
                    <td>
                      <span className={`ft-status ${job.status}`}>
                        {STATUS_LABELS[job.status] ?? job.status}
                      </span>
                    </td>
                    <td className="num">{formatDate(job.created_at)}</td>
                    <td className="ft-row-actions">
                      <button type="button" className="btn-ghost" onClick={() => watchJob(job.id)}>
                        Suivre
                      </button>
                      <button
                        type="button"
                        className="btn-ghost"
                        onClick={() => void loadComparison(job.id)}
                      >
                        Résultats
                      </button>
                      <button
                        type="button"
                        className="btn-ghost"
                        onClick={() => void act("Job supprimé.", () => api.deleteFinetuneJob(job.id))}
                      >
                        Supprimer
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>
    </div>
  );
}
