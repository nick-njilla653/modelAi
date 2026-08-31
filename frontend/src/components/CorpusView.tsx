import { useCallback, useEffect, useState } from "react";

import * as api from "../api";
import { IconUpload } from "../icons";
import { IngestDialog } from "./IngestDialog";
import type { CorpusDocument } from "../types";

/**
 * Le corpus documentaire global.
 *
 * C'est ici qu'on nourrit le système : ce qui entre est indexé dans
 * PostgreSQL, Milvus et Elasticsearch, et devient consultable depuis toutes
 * les conversations. À distinguer des pièces jointes du composeur, qui restent
 * confinées à leur conversation.
 */
export function CorpusView() {
  const [documents, setDocuments] = useState<CorpusDocument[]>([]);
  const [dialogOpen, setDialogOpen] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = useCallback(async () => {
    setLoading(true);
    try {
      setDocuments(await api.listDocuments());
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Corpus illisible.");
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const totalChunks = documents.reduce((sum, doc) => sum + doc.chunks, 0);

  return (
    <div className="view">
      <div className="view-head">
        <div>
          <h1>Corpus documentaire</h1>
          <p className="view-sub">
            Les textes indexés ici alimentent tout le système et sont consultables
            depuis toutes les conversations. Un document propre à un échange
            particulier se joint depuis le composeur, sans entrer dans le corpus.
          </p>
        </div>
        <button type="button" className="btn-primary" onClick={() => setDialogOpen(true)}>
          <IconUpload size={15} /> Ajouter un document
        </button>
      </div>

      {error && <p className="field-error">{error}</p>}

      <section className="ft-section">
        <h2>
          {documents.length} document{documents.length > 1 ? "s" : ""} ·{" "}
          {totalChunks.toLocaleString("fr-FR")} passages indexés
        </h2>

        {loading && documents.length === 0 ? (
          <p className="job-detail">Chargement…</p>
        ) : documents.length === 0 ? (
          <p className="panel-empty">
            Le corpus est vide. Sans document indexé, le système ne peut répondre
            à aucune question de fond.
          </p>
        ) : (
          <div className="table-scroll">
            <table className="metric-table">
              <thead>
                <tr>
                  <th scope="col">Document</th>
                  <th scope="col">Langue</th>
                  <th scope="col">Type</th>
                  <th scope="col" className="num">Passages</th>
                </tr>
              </thead>
              <tbody>
                {documents.map((doc) => (
                  <tr key={doc.doc_id}>
                    <th scope="row">{doc.source}</th>
                    <td><span className="lang-tag">{doc.language}</span></td>
                    <td>{doc.doc_type ?? "—"}</td>
                    <td className="num">{doc.chunks.toLocaleString("fr-FR")}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </section>

      {dialogOpen && (
        <IngestDialog
          scope="corpus"
          onClose={() => {
            setDialogOpen(false);
            void refresh();
          }}
          onIngested={() => void refresh()}
        />
      )}
    </div>
  );
}
