/**
 * Client de l'API GOV-AI 2.0.
 *
 * Le chemin est relatif : en développement Vite relaie /api vers :8000, en
 * production FastAPI sert l'interface et l'API sur la même origine.
 */
import type {
  ActiveModels,
  Citation,
  CorpusDocument,
  DatasetUploadResult,
  EvaluationReport,
  FinetuneJob,
  GraphEvidence,
  HealthResponse,
  IngestJob,
  ModelComparison,
  QueryRequest,
  QueryResponse,
  SafetyFlag,
  SessionDetail,
  SessionDocument,
  SessionSummary,
  WebSource,
} from "./types";

const API = "/api/v1";

class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API}${path}`, {
    ...init,
    headers: { "Content-Type": "application/json", ...(init?.headers ?? {}) },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* corps non JSON : on garde le statut HTTP */
    }
    throw new ApiError(detail, response.status);
  }

  if (response.status === 204) return undefined as T;
  return (await response.json()) as T;
}

export const listSessions = (limit = 60) =>
  request<SessionSummary[]>(`/sessions?limit=${limit}`);

export const getSession = (sessionId: string) =>
  request<SessionDetail>(`/sessions/${sessionId}`);

export const deleteSession = (sessionId: string) =>
  request<void>(`/sessions/${sessionId}`, { method: "DELETE" });

export const listDocuments = () => request<CorpusDocument[]>("/documents");

/**
 * Dépose un document et rend la main aussitôt : l'ingestion se poursuit en
 * tâche de fond, son avancement se consulte par `getIngestJob`.
 *
 * Le corps est un multipart, donc pas d'en-tête Content-Type explicite — le
 * navigateur doit produire lui-même la frontière de séparation.
 */
export async function ingestDocument(input: {
  file: File;
  source: string;
  docType: string;
  forceOcr: boolean;
}): Promise<IngestJob> {
  const form = new FormData();
  form.append("file", input.file);
  form.append("source", input.source);
  form.append("doc_type", input.docType);
  form.append("force_ocr", String(input.forceOcr));

  const response = await fetch(`${API}/ingest/async`, { method: "POST", body: form });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* corps non JSON */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as IngestJob;
}

export const getIngestJob = (jobId: string) => request<IngestJob>(`/ingest/jobs/${jobId}`);

/**
 * Attache un document à UNE conversation.
 *
 * À ne pas confondre avec `ingestDocument`, qui verse au corpus global : ce
 * document-ci n'est visible que dans sa conversation et disparaît avec elle.
 */
export async function attachSessionDocument(
  sessionId: string,
  input: { file: File; source: string; forceOcr: boolean },
): Promise<SessionDocument> {
  const form = new FormData();
  form.append("file", input.file);
  form.append("source", input.source);
  form.append("force_ocr", String(input.forceOcr));

  const response = await fetch(`${API}/sessions/${sessionId}/documents`, {
    method: "POST",
    body: form,
  });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* corps non JSON */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as SessionDocument;
}

export const listSessionDocuments = (sessionId: string) =>
  request<SessionDocument[]>(`/sessions/${sessionId}/documents`);

export const detachSessionDocument = (sessionId: string, documentId: string) =>
  request<void>(`/sessions/${sessionId}/documents/${documentId}`, { method: "DELETE" });

export const getHealth = () => request<HealthResponse>("/health");

export const listBaselines = () =>
  request<{ baselines: Record<string, string> }>("/evaluation/baselines");

export const runEvaluation = (payload: {
  dataset_path: string | null;
  baselines: string[];
  top_k_values: number[];
}) =>
  request<EvaluationReport>("/evaluation/run", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const askOnce = (payload: QueryRequest) =>
  request<QueryResponse>("/query", {
    method: "POST",
    body: JSON.stringify(payload),
  });

/** Événements émis par /query/stream, dans l'ordre : meta → token* → done. */
export type StreamEvent =
  | {
      /** Jalon de progression : la recherche precede le premier token de
       *  pres de deux minutes, l'interface doit pouvoir le dire. */
      type: "stage";
      stage: "analyzing" | "retrieving" | "surveying" | "generating";
      label: string;
    }
  | {
      type: "meta";
      trace_id: string;
      session_id: string | null;
      intent: string;
      action_plan: string[];
      language: string;
      profile: string;
      chunks_retrieved: number;
    }
  | { type: "token"; v: string }
  | {
      type: "done";
      answer: string;
      citations: Citation[];
      web_sources?: WebSource[];
      graph_evidence?: GraphEvidence[];
      safety_flags: SafetyFlag[];
      warnings: string[];
      confidence_level?: string;
      uncertainty_score?: number;
      intent?: string;
      model_used: string | null;
      latency_ms: number;
    }
  | { type: "error"; message: string };

/**
 * Consomme le flux SSE de /query/stream.
 *
 * `signal` permet d'interrompre la génération depuis le bouton d'arrêt — à
 * deux tokens par seconde, pouvoir couper une réponse partie de travers n'est
 * pas un luxe.
 */
export async function askStream(
  payload: QueryRequest,
  onEvent: (event: StreamEvent) => void,
  signal?: AbortSignal,
): Promise<void> {
  const response = await fetch(`${API}/query/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ ...payload, stream: true }),
    signal,
  });

  if (!response.ok || !response.body) {
    throw new ApiError(
      `Le serveur a refusé la requête (${response.status}).`,
      response.status,
    );
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;

    buffer += decoder.decode(value, { stream: true });

    // Un événement SSE se termine par une ligne vide.
    let boundary = buffer.indexOf("\n\n");
    while (boundary !== -1) {
      const raw = buffer.slice(0, boundary);
      buffer = buffer.slice(boundary + 2);
      boundary = buffer.indexOf("\n\n");

      const line = raw.trim();
      if (!line.startsWith("data:")) continue;

      const payloadText = line.slice(5).trim();
      if (payloadText === "[DONE]") return;

      try {
        onEvent(JSON.parse(payloadText) as StreamEvent);
      } catch {
        // Fragment illisible : on l'ignore plutôt que d'interrompre le flux.
      }
    }
  }
}

export { ApiError };

/** ── Fine-tuning ──────────────────────────────────────────────────────── */

const FT = `${API}/finetune`;

/** Dépose l'archive du corpus d'entraînement et retourne son identifiant. */
export async function uploadDataset(file: File): Promise<DatasetUploadResult> {
  const form = new FormData();
  form.append("file", file);
  const response = await fetch(`${FT}/upload`, { method: "POST", body: form });
  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = (await response.json()) as { detail?: string };
      if (body.detail) detail = body.detail;
    } catch {
      /* corps non JSON */
    }
    throw new ApiError(detail, response.status);
  }
  return (await response.json()) as DatasetUploadResult;
}

export const startFinetune = (payload: {
  name: string;
  target: string;
  epochs: number;
  batch_size: number;
  learning_rate: number;
  pairs_per_chunk: number;
  lora_mode: boolean;
  dataset_id: string;
  ingest_corpus: boolean;
}) =>
  request<{ job_id: string; status: string; message: string }>("/finetune/start", {
    method: "POST",
    body: JSON.stringify(payload),
  });

export const listFinetuneJobs = () =>
  request<{ jobs: FinetuneJob[]; total: number }>("/finetune/jobs");

export const getFinetuneJob = (jobId: string) =>
  request<FinetuneJob>(`/finetune/jobs/${jobId}`);

export const cancelFinetuneJob = (jobId: string) =>
  request<Record<string, unknown>>(`/finetune/jobs/${jobId}/cancel`, { method: "POST" });

export const pauseFinetuneJob = (jobId: string) =>
  request<Record<string, unknown>>(`/finetune/jobs/${jobId}/pause`, { method: "POST" });

export const resumeFinetuneJob = (jobId: string) =>
  request<Record<string, unknown>>(`/finetune/jobs/${jobId}/resume`, { method: "POST" });

export const deleteFinetuneJob = (jobId: string) =>
  request<Record<string, unknown>>(`/finetune/jobs/${jobId}`, { method: "DELETE" });

export const applyFinetunedModel = (jobId: string, reindexMilvus: boolean) =>
  request<Record<string, unknown>>(`/finetune/apply/${jobId}`, {
    method: "POST",
    body: JSON.stringify({ job_id: jobId, reindex_milvus: reindexMilvus }),
  });

export const rollbackModel = (modelType: string) =>
  request<Record<string, unknown>>(`/finetune/rollback/${modelType}`, { method: "POST" });

export const getActiveModels = () => request<ActiveModels>("/finetune/active-models");

export const startComparison = (jobId: string) =>
  request<Record<string, unknown>>(`/finetune/evaluate/${jobId}`, { method: "POST" });

export const getComparison = (jobId: string) =>
  request<ModelComparison>(`/finetune/evaluate/${jobId}/results`);

export const getReindexStatus = (jobId: string) =>
  request<Record<string, unknown>>(`/finetune/reindex-status/${jobId}`);

/**
 * Ouvre la WebSocket de progression d'un job.
 *
 * Le serveur diffuse l'état complet du job à chaque changement — on ne
 * reconstruit donc rien côté client, on remplace.
 */
export function openFinetuneProgress(
  jobId: string,
  onUpdate: (job: FinetuneJob) => void,
  onClose?: () => void,
): WebSocket {
  const scheme = location.protocol === "https:" ? "wss" : "ws";
  const socket = new WebSocket(`${scheme}://${location.host}${API}/finetune/progress/${jobId}`);
  socket.onmessage = (event) => {
    try {
      onUpdate(JSON.parse(event.data) as FinetuneJob);
    } catch {
      /* trame illisible : on attend la suivante */
    }
  };
  socket.onclose = () => onClose?.();
  return socket;
}
