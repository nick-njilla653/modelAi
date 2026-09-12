import { useEffect, useState } from "react";

import * as api from "../api";
import type { EvaluationReport } from "../types";

/** Métriques attendues par le mémoire, avec leur seuil de conformité. */
const METRIC_LABELS: Record<string, string> = {
  recall_at_k: "Rappel @k",
  precision_at_k: "Précision @k",
  mrr: "MRR",
  ndcg_at_k: "nDCG @k",
  citation_precision: "Précision des citations",
  faithfulness: "Fidélité aux sources",
  answer_relevance: "Pertinence de la réponse",
  hallucination_rate: "Taux d'hallucination",
  latency_p50_ms: "Latence médiane",
  latency_p95_ms: "Latence p95",
  throughput_qps: "Débit (req/s)",
};

function formatValue(key: string, value: unknown): string {
  if (typeof value !== "number") return String(value ?? "—");
  if (key.includes("latency")) return `${Math.round(value)} ms`;
  if (key.includes("qps")) return value.toFixed(2);
  if (value >= 0 && value <= 1) return `${(value * 100).toFixed(1)} %`;
  return value.toFixed(3);
}

function MetricGroup({ title, data }: { title: string; data: Record<string, unknown> }) {
  const entries = Object.entries(data).filter(([, v]) => typeof v !== "object" || v === null);
  if (entries.length === 0) return null;

  return (
    <section className="metric-group">
      <h3>{title}</h3>
      <div className="table-scroll">
        <table className="metric-table">
          <tbody>
            {entries.map(([key, value]) => (
              <tr key={key}>
                <th scope="row">{METRIC_LABELS[key] ?? key}</th>
                <td className="num">{formatValue(key, value)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function EvaluationView() {
  const [baselines, setBaselines] = useState<Record<string, string>>({});
  const [baseline, setBaseline] = useState("B4");
  const [datasetPath, setDatasetPath] = useState("eval/datasets/qa_bilingual_annotated.json");
  const [topK, setTopK] = useState("1,3,5,10");
  const [running, setRunning] = useState(false);
  const [report, setReport] = useState<EvaluationReport | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    void api
      .listBaselines()
      .then((data) => setBaselines(data.baselines))
      .catch(() => setBaselines({}));
  }, []);

  const run = async () => {
    setRunning(true);
    setError(null);
    setReport(null);
    try {
      setReport(
        await api.runEvaluation({
          dataset_path: datasetPath.trim() || null,
          baselines: [baseline.toLowerCase()],
          top_k_values: topK
            .split(",")
            .map((k) => Number.parseInt(k.trim(), 10))
            .filter((k) => Number.isFinite(k)),
        }),
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "L'évaluation a échoué.");
    } finally {
      setRunning(false);
    }
  };

  const constraints = Object.entries(report?.constraints_met ?? {});

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Évaluation</h1>
          <p className="view-sub">
            Étude ablative B0 → B4 : chaque palier ajoute un étage au pipeline de
            récupération, pour mesurer ce que cet étage apporte réellement.
          </p>
        </div>
      </div>

      <div className="eval-form">
        <label className="field">
          <span className="field-label">Configuration évaluée</span>
          <select value={baseline} onChange={(event) => setBaseline(event.target.value)}>
            {Object.entries(baselines).map(([id, description]) => (
              <option key={id} value={id}>
                {id} — {description}
              </option>
            ))}
            {Object.keys(baselines).length === 0 && <option value="B4">B4</option>}
          </select>
        </label>

        <label className="field">
          <span className="field-label">Jeu de données annoté</span>
          <input
            type="text"
            value={datasetPath}
            onChange={(event) => setDatasetPath(event.target.value)}
          />
          <span className="field-hint">Chemin relatif à la racine du projet.</span>
        </label>

        <label className="field">
          <span className="field-label">Valeurs de k</span>
          <input type="text" value={topK} onChange={(event) => setTopK(event.target.value)} />
        </label>

        <button type="button" className="btn-primary" onClick={() => void run()} disabled={running}>
          {running ? "Évaluation en cours…" : "Lancer l'évaluation"}
        </button>

        {running && (
          <p className="job-detail">
            Chaque question du jeu de données déclenche une génération complète.
            Au débit observé, comptez plusieurs minutes par question.
          </p>
        )}

        {error && <p className="field-error">{error}</p>}
      </div>

      {report && (
        <div className="eval-results">
          <div className="view-head">
            <div>
              <h2>
                {report.baseline_id} — {report.baseline_description}
              </h2>
              <p className="view-sub">
                {report.num_queries} question{report.num_queries > 1 ? "s" : ""} évaluée
                {report.num_queries > 1 ? "s" : ""} · {report.timestamp}
              </p>
            </div>
          </div>

          {constraints.length > 0 && (
            <div className="constraints">
              {constraints.map(([name, met]) => (
                <span key={name} className={`constraint ${met ? "met" : "unmet"}`}>
                  {met ? "✓" : "✗"} {name}
                </span>
              ))}
            </div>
          )}

          <MetricGroup title="Récupération" data={report.retrieval ?? {}} />
          <MetricGroup title="Génération" data={report.generation ?? {}} />
          <MetricGroup title="Système" data={report.system ?? {}} />
        </div>
      )}
    </div>
  );
}
