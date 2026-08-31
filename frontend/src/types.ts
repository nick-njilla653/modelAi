/**
 * Types miroir des schémas Pydantic du backend (app/models/schemas.py).
 *
 * Ils sont écrits à la main plutôt que générés : le contrat est petit et stable,
 * et une génération depuis OpenAPI ajouterait une étape de build pour un gain
 * marginal. Toute évolution de `QueryResponse` doit être répercutée ici.
 */

export type Language = "fr" | "en" | "unknown";

export type UserProfile = "citizen" | "agent" | "enterprise" | "jurist";

/** Drapeaux remontés par le pipeline de vérification (app/models/domain.py). */
export type SafetyFlag =
  | "LOW_CONFIDENCE"
  | "ESCALATION_RECOMMENDED"
  | "CONTRADICTION_DETECTED"
  | "OUT_OF_CORPUS"
  | "PROMPT_INJECTION_ATTEMPT"
  | "UNSUPPORTED_CLAIMS"
  | "JURIDICAL_DIVERGENCE";

export type ConfidenceLevel = "high" | "medium" | "low" | "very_low";

/**
 * Une citation vérifiable. `article` porte la référence extraite à l'ingestion
 * (« Article 103 »), jamais un numéro reconstruit par le modèle.
 */
export interface Citation {
  source_id: string;
  doc_title: string;
  doc_type?: string | null;
  institution?: string | null;
  jurisdiction?: string | null;
  language: Language;
  page?: number | null;
  article?: string | null;
  chunk_id: string;
  excerpt: string;
  relevance_score: number;
  date_document?: string | null;
}

export interface RetrievedChunk {
  chunk_id: string;
  doc_id: string;
  content: string;
  source: string;
  language: Language;
  page?: number | null;
  chunk_index: number;
  final_score: number;
  metadata: Record<string, unknown>;
}

export interface QueryRequest {
  query: string;
  session_id?: string | null;
  language?: Language | null;
  profile: UserProfile;
  top_k?: number;
  stream?: boolean;
  include_chunks?: boolean;
  /** null laisse le réglage serveur décider ; true/false force par requête. */
  web_search?: boolean | null;
}

export interface QueryResponse {
  query_id: string;
  answer: string;
  citations: Citation[];
  retrieved_chunks: RetrievedChunk[];
  graph_evidence: GraphEvidence[];
  uncertainty_score: number;
  confidence_level: ConfidenceLevel;
  safety_flags: SafetyFlag[];
  warnings: string[];
  language_detected: Language;
  intent_detected?: string | null;
  action_plan: string[];
  latency_ms?: number | null;
  model_used?: string | null;
  session_id?: string | null;
}

export interface SessionSummary {
  session_id: string;
  title: string;
  language: string;
  profile: string;
  turns: number;
  created_at?: string | null;
  last_active?: string | null;
}

export interface SessionTurn {
  turn_id: string;
  query: string;
  answer: string;
  /** Vrai pour les tours antérieurs à la persistance intégrale des réponses. */
  answer_truncated: boolean;
  citations: Citation[];
  safety_flags: SafetyFlag[];
  intent?: string | null;
  confidence?: number | null;
  latency_ms?: number | null;
  model_used?: string | null;
  created_at?: string | null;
}

export interface SessionDetail {
  session_id: string;
  title: string;
  language: string;
  profile: string;
  created_at?: string | null;
  last_active?: string | null;
  turns: SessionTurn[];
}

export interface CorpusDocument {
  doc_id: string;
  source: string;
  language: string;
  doc_type?: string | null;
  chunks: number;
}

/** État d'un message dans le fil, côté interface. */
export interface ChatMessage {
  id: string;
  role: "user" | "assistant";
  content: string;
  /** Le texte arrive encore par le flux : afficher le curseur d'écriture. */
  streaming?: boolean;
  /** Étape en cours, tant qu'aucun texte n'est arrivé. */
  stage?: string;
  citations?: Citation[];
  webSources?: WebSource[];
  graphEvidence?: GraphEvidence[];
  safetyFlags?: SafetyFlag[];
  warnings?: string[];
  intent?: string | null;
  latencyMs?: number | null;
  modelUsed?: string | null;
  /** Message d'erreur à afficher à la place de la réponse. */
  error?: string;
}

/** Suivi d'une ingestion lancée en tâche de fond. */
export interface IngestJob {
  job_id: string;
  filename: string;
  source: string;
  status: "pending" | "running" | "completed" | "failed";
  stage: string;
  detail: string;
  progress: number;
  chunks_created: number;
  document_id: string | null;
  language: string | null;
  ocr_used: boolean;
  warnings: string[];
  error: string | null;
  elapsed_s: number;
}

/** Réponse de /health : état de chaque dépendance du pipeline. */
/**
 * Volumétrie du graphe de connaissances.
 *
 * « neo4j : ok » atteste une connexion, pas des données : le graphe est resté
 * vide et connecté tout un sprint sans que rien ne le signale.
 */
export interface GraphVolumetry {
  texts: number;
  articles: number;
  references: number;
  institutions: number;
  mentions: number;
  external_texts: number;
}

export interface HealthResponse {
  status: string;
  version: string;
  services: Record<string, string>;
  model?: string | null;
  /** Absente si Neo4j est injoignable — distinct d'un graphe vide. */
  graph?: GraphVolumetry | null;
  timestamp: string;
}

/** Rapport d'évaluation (étude ablative B0 → B4). */
export interface EvaluationReport {
  baseline_id: string;
  baseline_description: string;
  timestamp: string;
  num_queries: number;
  retrieval: Record<string, unknown>;
  generation: Record<string, unknown>;
  system: Record<string, unknown>;
  constraints_met: Record<string, boolean>;
}

/** Section affichée dans la zone principale. */
export type AppView = "chat" | "corpus" | "evaluation" | "health" | "finetuning";

/** ── Fine-tuning ──────────────────────────────────────────────────────── */

export interface DatasetUploadResult {
  dataset_id: string;
  zip_size_mb: number;
  total_files: number;
  supported_files: number;
  message: string;
}

export type FinetuneTarget = "llm" | "embedding" | "reranker";

export interface FinetuneJob {
  id: string;
  name: string;
  target: FinetuneTarget | string;
  status: "pending" | "running" | "paused" | "completed" | "failed" | "cancelled" | string;
  progress: number;
  logs: string[];
  result: Record<string, unknown> | null;
  error: string | null;
  created_at: number;
  /** Rapport de fuite de données, présent une fois le jeu construit. */
  leakage?: {
    leaked_pairs?: number;
    total_pairs?: number;
    leak_ratio?: number;
    warning?: string;
  } | null;
  evaluation?: ModelComparison | null;
}

/** Métriques agrégées d'un modèle sur le jeu d'évaluation. */
export interface ModelMetrics {
  mean_bleu: number;
  mean_rouge_l: number;
  mean_latency_ms: number;
  mean_judge_score: number;
  win_rate: number;
  queries_evaluated: number;
}

export interface ModelComparison {
  original_metrics: ModelMetrics;
  finetuned_metrics: ModelMetrics;
  reference_metrics?: Record<string, Partial<ModelMetrics>> | null;
  sample_responses: Array<Record<string, unknown>>;
  total_queries: number;
  finetuned_model: string;
  original_model: string;
  reference_benchmarks?: Record<string, unknown> | null;
}

export interface ActiveModelEntry {
  model_path?: string | null;
  ollama_model_name?: string | null;
  job_id: string | null;
  applied_at: string | null;
}

export interface ActiveModels {
  embedding_model: ActiveModelEntry;
  llm_model: ActiveModelEntry;
  reranker_model: ActiveModelEntry;
  history: Array<Record<string, unknown>>;
}

/**
 * Une page officielle consultée sur le web.
 *
 * Distincte d'une citation : une citation renvoie au texte normatif indexé,
 * une source web à sa présentation par une institution. L'interface doit les
 * séparer, sans quoi une page ministérielle paraîtrait faire loi.
 */
export interface WebSource {
  title: string;
  url: string;
  snippet: string;
  domain: string;
  institution: string;
  acronym: string;
  /** 0 = institution suprême … 5 = presse publique. */
  priority: number;
  /** Faux si le site ne répondait pas au dernier contrôle de disponibilité. */
  reachable: boolean;
}

/** Une disposition désignée par le graphe, avec le texte dont elle relève. */
export interface GraphArticleRef {
  number: string;
  doc_title: string;
}

/**
 * Ce que le graphe de connaissances a apporté à une réponse.
 *
 * Distincte d'une citation : une citation atteste un passage du corpus, une
 * pièce de graphe atteste une *relation* — entre deux dispositions, ou entre
 * une disposition et l'autorité qu'elle nomme. L'interface doit les présenter
 * séparément, sinon une déduction structurelle passerait pour un extrait.
 */
export interface GraphEvidence {
  kind: "institution" | "article";
  label: string;
  /** Catégorie de l'institution, ou texte d'appartenance de l'article. */
  detail?: string | null;
  /**
   * « passages » : déduit des dispositions retrouvées, sans décompte global.
   * « corpus » : l'entité est nommée dans la question, le décompte porte sur
   * tout le corpus. Ne pas afficher le second libellé pour le premier cas.
   */
  scope?: "passages" | "corpus" | null;
  articles: GraphArticleRef[];
  /** Peut dépasser `articles.length` : la liste affichée est tronquée. */
  article_count: number;
  issued_texts: string[];
}

/**
 * Document attaché à une conversation.
 *
 * Distinct d'un document du corpus : il ne quitte jamais sa conversation, n'est
 * écrit dans aucun index partagé, et disparaît avec elle.
 */
export interface SessionDocument {
  document_id: string;
  filename: string;
  source: string;
  language: string;
  doc_type: string | null;
  chunk_count: number;
  ocr_used: boolean;
  created_at?: string | null;
  warnings?: string[];
  latency_ms?: number;
}
