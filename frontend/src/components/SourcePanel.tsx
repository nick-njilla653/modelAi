import { IconClose } from "../icons";
import type { Citation, CorpusDocument } from "../types";

interface SourcePanelProps {
  mode: "citation" | "corpus";
  citations: Citation[];
  selectedCitationId: string | null;
  documents: CorpusDocument[];
  onClose: () => void;
}

/**
 * Panneau latéral droit : soit les sources de la réponse affichée, soit
 * l'inventaire du corpus.
 *
 * C'est la contrepartie visible du principe d'ancrage : une affirmation sans
 * extrait consultable n'est pas vérifiable, quelle que soit son assurance.
 */
export function SourcePanel({
  mode,
  citations,
  selectedCitationId,
  documents,
  onClose,
}: SourcePanelProps) {
  return (
    <aside className="panel">
      <div className="panel-head">
        <span className="panel-title">
          {mode === "corpus" ? "Documents du corpus" : "Sources citées"}
        </span>
        <button type="button" className="icon-button" onClick={onClose} aria-label="Fermer le panneau">
          <IconClose />
        </button>
      </div>

      <div className="panel-body">
        {mode === "corpus" ? (
          documents.length === 0 ? (
            <p className="panel-empty">
              Le corpus est vide. Ingérez des documents avant d'interroger le système.
            </p>
          ) : (
            <div>
              {documents.map((doc) => (
                <div key={doc.doc_id} className="doc-row">
                  <span>
                    {doc.source} <span className="lang-tag">{doc.language}</span>
                  </span>
                  <span className="doc-row-count">{doc.chunks} passages</span>
                </div>
              ))}
            </div>
          )
        ) : citations.length === 0 ? (
          <p className="panel-empty">
            Aucune source pour ce message. Cliquez sur une citation d'une réponse
            pour en afficher l'extrait.
          </p>
        ) : (
          citations.map((citation) => (
            <div
              key={citation.chunk_id}
              id={`source-${citation.chunk_id}`}
              className={`source-card${
                citation.chunk_id === selectedCitationId ? " highlight" : ""
              }`}
            >
              <div className="source-doc">{citation.doc_title}</div>
              <div className="source-meta">
                {citation.article && <span>{citation.article}</span>}
                {citation.page != null && <span>p. {citation.page}</span>}
                {citation.doc_type && <span>{citation.doc_type}</span>}
                <span>{Math.round(citation.relevance_score * 100)} %</span>
              </div>
              <blockquote className="source-excerpt">{citation.excerpt}</blockquote>
            </div>
          ))
        )}
      </div>
    </aside>
  );
}
