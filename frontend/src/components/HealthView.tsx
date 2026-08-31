import { useCallback, useEffect, useState } from "react";

import * as api from "../api";
import type { GraphVolumetry, HealthResponse } from "../types";

/** Noms lisibles des services : « milvus » ne dit rien de sa fonction. */
const SERVICE_LABELS: Record<string, string> = {
  milvus: "Milvus — index vectoriel",
  elasticsearch: "Elasticsearch — recherche lexicale",
  postgres: "PostgreSQL — documents et conversations",
  embedding: "Embeddings — vectorisation des passages",
  neo4j: "Neo4j — graphe de connaissances",
  reranker: "Reranker — reclassement des extraits",
};

export function HealthView() {
  const [health, setHealth] = useState<HealthResponse | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setHealth(await api.getHealth());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Service injoignable.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const services = Object.entries(health?.services ?? {});
  const degraded = services.filter(([, state]) => state !== "ok");

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Santé des services</h1>
          <p className="view-sub">
            {degraded.length === 0 && services.length > 0
              ? "Tous les services répondent."
              : degraded.length > 0
                ? `${degraded.length} service${degraded.length > 1 ? "s" : ""} en défaut — les réponses seront dégradées.`
                : "État inconnu."}
          </p>
        </div>
        <button type="button" className="btn-ghost" onClick={() => void refresh()} disabled={loading}>
          {loading ? "Vérification…" : "Rafraîchir"}
        </button>
      </div>

      {error && <p className="field-error">{error}</p>}

      <div className="health-grid">
        {services.map(([name, state]) => (
          <div key={name} className={`health-card ${state === "ok" ? "up" : "down"}`}>
            <span className={`health-dot ${state === "ok" ? "up" : "down"}`} />
            <div>
              <div className="health-name">{SERVICE_LABELS[name] ?? name}</div>
              <div className="health-state">{state}</div>
            </div>
          </div>
        ))}
      </div>

      {health?.graph && <GraphPanel graph={health.graph} />}

      {health && (
        <dl className="kv">
          <div>
            <dt>Modèle de génération</dt>
            <dd className="mono">{health.model ?? "—"}</dd>
          </div>
          <div>
            <dt>Version</dt>
            <dd className="mono">{health.version}</dd>
          </div>
          <div>
            <dt>Relevé</dt>
            <dd className="mono">{health.timestamp}</dd>
          </div>
        </dl>
      )}
    </div>
  );
}


/** Libellés des compteurs : « refs » ne dit pas ce qui est compté. */
const GRAPH_ROWS: { key: keyof GraphVolumetry; label: string; hint: string }[] = [
  { key: "texts", label: "Textes normatifs", hint: "Codes et lois indexés" },
  { key: "articles", label: "Articles", hint: "Dispositions distinctes" },
  { key: "references", label: "Renvois", hint: "Un article en cite un autre" },
  { key: "institutions", label: "Institutions", hint: "Autorités reconnues" },
  { key: "mentions", label: "Mentions", hint: "Un article nomme une autorité" },
  {
    key: "external_texts",
    label: "Textes cités hors corpus",
    hint: "Cités par le corpus mais absents de celui-ci",
  },
];

/**
 * Volumétrie du graphe.
 *
 * « neo4j : ok » atteste une connexion, pas des données : le graphe est resté
 * vide et connecté tout un sprint sans que rien ne le signale. Un graphe vide
 * doit se voir, et se distinguer d'un Neo4j injoignable — d'où l'avertissement
 * explicite plutôt qu'une grille de zéros silencieuse.
 */
function GraphPanel({ graph }: { graph: GraphVolumetry }) {
  const vide = graph.articles === 0;

  return (
    <section className="graph-panel">
      <h2 className="graph-panel-title">Graphe de connaissances</h2>
      {vide ? (
        <p className="graph-panel-empty">
          Le graphe est connecté mais vide : aucun article n'y figure.
          L'enrichissement par le graphe ne produira rien tant que le corpus n'y
          aura pas été versé (<code>scripts/build_knowledge_graph.py</code>).
        </p>
      ) : (
        <div className="graph-stats">
          {GRAPH_ROWS.map((row) => (
            <div key={row.key} className="graph-stat">
              <span className="graph-stat-value">{graph[row.key].toLocaleString("fr-FR")}</span>
              <span className="graph-stat-label">{row.label}</span>
              <span className="graph-stat-hint">{row.hint}</span>
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
