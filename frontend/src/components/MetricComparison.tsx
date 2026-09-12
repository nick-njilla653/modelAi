import { useState } from "react";

import type { ModelComparison, ModelMetrics } from "../types";

/**
 * Comparaison modèle d'origine / modèle fine-tuné.
 *
 * Une barre horizontale par métrique, cadrée sur SON propre maximum : BLEU,
 * ROUGE-L et le taux de victoire vivent entre 0 et 1, le score du juge non.
 * Les superposer sur une échelle commune — ou sur un radar — écraserait les
 * premières et gonflerait le second. La latence n'apparaît pas ici : autre
 * unité, autre sens de lecture, donc autre bloc.
 */

type SeriesKey = "original" | "finetuned" | "reference";

const SERIES: { key: SeriesKey; label: string; slot: number }[] = [
  { key: "original", label: "Modèle d'origine", slot: 1 },
  { key: "finetuned", label: "Modèle fine-tuné", slot: 2 },
  { key: "reference", label: "Référence", slot: 3 },
];

const METRICS: { key: keyof ModelMetrics; label: string; hint: string }[] = [
  { key: "mean_bleu", label: "BLEU", hint: "Recouvrement lexical avec la réponse attendue" },
  { key: "mean_rouge_l", label: "ROUGE-L", hint: "Plus longue sous-séquence commune" },
  { key: "mean_judge_score", label: "Score du juge", hint: "Notation par un modèle arbitre" },
  { key: "win_rate", label: "Taux de victoire", hint: "Part des questions où ce modèle l'emporte" },
];

function formatMetric(key: keyof ModelMetrics, value: number): string {
  if (key === "win_rate") return `${(value * 100).toFixed(1)} %`;
  return value.toFixed(3);
}

interface MetricComparisonProps {
  comparison: ModelComparison;
}

export function MetricComparison({ comparison }: MetricComparisonProps) {
  const [showTable, setShowTable] = useState(false);

  // La référence est facultative : on ne l'affiche que si elle porte des valeurs.
  const referenceEntry = Object.entries(comparison.reference_metrics ?? {})[0];
  const reference = referenceEntry?.[1];
  const referenceLabel = referenceEntry?.[0];

  const valueOf = (series: SeriesKey, metric: keyof ModelMetrics): number | null => {
    if (series === "original") return comparison.original_metrics?.[metric] ?? null;
    if (series === "finetuned") return comparison.finetuned_metrics?.[metric] ?? null;
    const raw = reference?.[metric];
    return typeof raw === "number" ? raw : null;
  };

  const activeSeries = SERIES.filter(
    (s) => s.key !== "reference" || reference !== undefined,
  );

  const latencyOriginal = comparison.original_metrics?.mean_latency_ms ?? 0;
  const latencyFinetuned = comparison.finetuned_metrics?.mean_latency_ms ?? 0;
  const latencyDelta = latencyFinetuned - latencyOriginal;

  return (
    <div className="viz-root">
      <div className="viz-head">
        <div>
          <h3>Comparaison des modèles</h3>
          <p className="view-sub">
            {comparison.total_queries} question{comparison.total_queries > 1 ? "s" : ""} évaluée
            {comparison.total_queries > 1 ? "s" : ""} · {comparison.original_model} contre{" "}
            {comparison.finetuned_model}
          </p>
        </div>
        <button type="button" className="btn-ghost" onClick={() => setShowTable((v) => !v)}>
          {showTable ? "Voir le graphique" : "Voir le tableau"}
        </button>
      </div>

      <div className="viz-legend">
        {activeSeries.map((series) => (
          <span key={series.key} className="viz-legend-item">
            <span className={`viz-swatch series-${series.slot}`} />
            {series.key === "reference" && referenceLabel ? referenceLabel : series.label}
          </span>
        ))}
      </div>

      {showTable ? (
        <div className="table-scroll">
          <table className="metric-table">
            <thead>
              <tr>
                <th scope="col">Métrique</th>
                {activeSeries.map((series) => (
                  <th key={series.key} scope="col" className="num">
                    {series.key === "reference" && referenceLabel ? referenceLabel : series.label}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {METRICS.map((metric) => (
                <tr key={metric.key}>
                  <th scope="row">{metric.label}</th>
                  {activeSeries.map((series) => {
                    const value = valueOf(series.key, metric.key);
                    return (
                      <td key={series.key} className="num">
                        {value == null ? "—" : formatMetric(metric.key, value)}
                      </td>
                    );
                  })}
                </tr>
              ))}
              <tr>
                <th scope="row">Latence moyenne</th>
                <td className="num">{Math.round(latencyOriginal)} ms</td>
                <td className="num">{Math.round(latencyFinetuned)} ms</td>
                {reference !== undefined && <td className="num">—</td>}
              </tr>
            </tbody>
          </table>
        </div>
      ) : (
        <div className="viz-metrics">
          {METRICS.map((metric) => {
            const values = activeSeries.map((series) => valueOf(series.key, metric.key));
            const max = Math.max(...values.map((v) => v ?? 0), 0.0001) * 1.15;

            return (
              <div key={metric.key} className="viz-metric">
                <div className="viz-metric-head">
                  <span className="viz-metric-label">{metric.label}</span>
                  <span className="viz-metric-hint">{metric.hint}</span>
                </div>
                <div className="viz-bars">
                  {activeSeries.map((series, index) => {
                    const value = values[index];
                    if (value == null) return null;
                    const width = Math.max((value / max) * 100, 0.6);
                    return (
                      <div key={series.key} className="viz-bar-row">
                        <div
                          className={`viz-bar series-${series.slot}`}
                          style={{ width: `${width}%` }}
                          title={`${series.label} — ${formatMetric(metric.key, value)}`}
                        />
                        {/* Étiquette directe : elle porte l'identité et la valeur,
                            de sorte que la couleur ne soit jamais seule à informer. */}
                        <span className="viz-bar-value">{formatMetric(metric.key, value)}</span>
                      </div>
                    );
                  })}
                </div>
              </div>
            );
          })}
        </div>
      )}

      <div className="viz-tiles">
        <div className="viz-tile">
          <span className="viz-tile-label">Latence — modèle d'origine</span>
          <span className="viz-tile-value">{Math.round(latencyOriginal)} ms</span>
        </div>
        <div className="viz-tile">
          <span className="viz-tile-label">Latence — modèle fine-tuné</span>
          <span className="viz-tile-value">{Math.round(latencyFinetuned)} ms</span>
          {latencyOriginal > 0 && (
            <span className={`viz-tile-delta ${latencyDelta <= 0 ? "better" : "worse"}`}>
              {latencyDelta <= 0 ? "−" : "+"}
              {Math.abs(Math.round(latencyDelta))} ms
              {latencyDelta <= 0 ? " (plus rapide)" : " (plus lent)"}
            </span>
          )}
        </div>
      </div>
    </div>
  );
}
