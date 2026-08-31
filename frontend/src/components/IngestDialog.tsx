import { useEffect, useRef, useState } from "react";

import * as api from "../api";
import { IconClose, IconUpload } from "../icons";
import type { IngestJob, SessionDocument } from "../types";

const DOC_TYPES = [
  { value: "loi", label: "Loi" },
  { value: "decret", label: "Décret" },
  { value: "arrete", label: "Arrêté" },
  { value: "circulaire", label: "Circulaire" },
  { value: "constitution", label: "Constitution" },
  { value: "autre", label: "Autre" },
];

/**
 * Deux ingestions, un seul dialogue — mais des portées explicitement distinctes.
 *
 * « corpus »       : nourrit tout le système, visible de toutes les conversations,
 *                    permanent jusqu'à suppression.
 * « conversation » : ne quitte pas la conversation, n'entre dans aucun index
 *                    partagé, disparaît avec elle.
 *
 * La confusion entre les deux n'est pas anodine : verser un contrat privé au
 * corpus le rendrait consultable depuis toute autre conversation. Le dialogue
 * énonce donc la portée avant que l'utilisateur ne dépose son fichier.
 */
export type IngestScope = "corpus" | "conversation";

interface IngestDialogProps {
  scope: IngestScope;
  sessionId?: string;
  onClose: () => void;
  onIngested: () => void;
}

export function IngestDialog({ scope, sessionId, onClose, onIngested }: IngestDialogProps) {
  const isCorpus = scope === "corpus";

  const [file, setFile] = useState<File | null>(null);
  const [source, setSource] = useState("");
  const [docType, setDocType] = useState("loi");
  const [forceOcr, setForceOcr] = useState(false);
  const [job, setJob] = useState<IngestJob | null>(null);
  const [attached, setAttached] = useState<SessionDocument | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const notifiedRef = useRef(false);

  // Sondage de l'avancement — l'ingestion du corpus est asynchrone, l'OCR d'un
  // document scanné pouvant durer plusieurs minutes.
  useEffect(() => {
    if (!job || job.status === "completed" || job.status === "failed") return;
    const timer = setInterval(async () => {
      try {
        setJob(await api.getIngestJob(job.job_id));
      } catch {
        /* un sondage manqué n'est pas grave */
      }
    }, 2000);
    return () => clearInterval(timer);
  }, [job]);

  useEffect(() => {
    const finished = job?.status === "completed" || attached !== null;
    if (finished && !notifiedRef.current) {
      notifiedRef.current = true;
      onIngested();
    }
  }, [job, attached, onIngested]);

  const submit = async () => {
    if (!file) return;
    setSubmitting(true);
    setError(null);
    try {
      if (isCorpus) {
        setJob(await api.ingestDocument({
          file,
          source: source.trim() || file.name,
          docType,
          forceOcr,
        }));
      } else {
        if (!sessionId) throw new Error("Aucune conversation active.");
        setAttached(await api.attachSessionDocument(sessionId, {
          file,
          source: source.trim() || file.name,
          forceOcr,
        }));
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Le téléversement a échoué.");
    } finally {
      setSubmitting(false);
    }
  };

  const done = job?.status === "completed" || attached !== null;
  const failed = job?.status === "failed";
  const started = job !== null || attached !== null;

  return (
    <div className="modal-scrim" role="dialog" aria-modal="true">
      <div className="modal">
        <div className="modal-head">
          <span className="panel-title">
            {isCorpus ? "Ajouter au corpus" : "Joindre un document à la conversation"}
          </span>
          <button type="button" className="icon-button" onClick={onClose} aria-label="Fermer">
            <IconClose />
          </button>
        </div>

        <div className="modal-body">
          <p className={`scope-note ${isCorpus ? "scope-global" : "scope-local"}`}>
            {isCorpus
              ? "Ce document nourrit tout le système : il sera indexé et consultable depuis toutes les conversations, jusqu'à sa suppression."
              : "Ce document ne quitte pas cette conversation. Il n'entre dans aucun index partagé et disparaît avec elle. Pour un texte de référence destiné à toutes les conversations, utilisez la page Corpus."}
          </p>

          {!started && (
            <>
              <label className="field">
                <span className="field-label">Fichier</span>
                <input
                  type="file"
                  accept=".pdf,.txt,.md"
                  onChange={(event) => {
                    const picked = event.target.files?.[0] ?? null;
                    setFile(picked);
                    if (picked && !source) setSource(picked.name.replace(/\.[^.]+$/, ""));
                  }}
                />
                <span className="field-hint">
                  PDF, TXT ou MD — {isCorpus ? "50" : "25"} Mo maximum.
                </span>
              </label>

              <label className="field">
                <span className="field-label">Libellé cité dans les réponses</span>
                <input
                  type="text"
                  value={source}
                  onChange={(event) => setSource(event.target.value)}
                  placeholder={isCorpus ? "Ex. : Code Pénal (Loi n° 2016/007)" : "Ex. : Mon contrat de bail"}
                />
                <span className="field-hint">
                  C'est ce texte qui apparaîtra dans chaque citation. Nommez le
                  document par ce qu'il contient, pas par son nom de fichier.
                </span>
              </label>

              {isCorpus && (
                <label className="field">
                  <span className="field-label">Type de document</span>
                  <select value={docType} onChange={(event) => setDocType(event.target.value)}>
                    {DOC_TYPES.map((type) => (
                      <option key={type.value} value={type.value}>
                        {type.label}
                      </option>
                    ))}
                  </select>
                </label>
              )}

              <label className="field-inline">
                <input
                  type="checkbox"
                  checked={forceOcr}
                  onChange={(event) => setForceOcr(event.target.checked)}
                />
                <span>
                  Forcer la reconnaissance de texte
                  <span className="field-hint">
                    Nécessaire pour un PDF scanné sans couche texte. Compter
                    plusieurs secondes par page.
                  </span>
                </span>
              </label>

              {error && <p className="field-error">{error}</p>}
            </>
          )}

          {job && (
            <div className="job-status">
              <div className="job-file">{job.filename}</div>
              <div className="job-source">cité comme « {job.source} »</div>

              {!done && !failed && (
                <>
                  <div
                    className="progress"
                    role="progressbar"
                    aria-valuenow={job.progress}
                    aria-valuemin={0}
                    aria-valuemax={100}
                  >
                    <div className="progress-fill" style={{ width: `${job.progress}%` }} />
                  </div>
                  <p className="job-detail">{job.detail} — {job.progress} %</p>
                </>
              )}

              {done && (
                <p className="job-detail job-ok">
                  {job.chunks_created} passages indexés en {job.elapsed_s} s
                  {job.ocr_used ? " (reconnaissance de texte appliquée)" : ""}.
                </p>
              )}
              {failed && <p className="field-error">{job.error}</p>}
            </div>
          )}

          {attached && (
            <div className="job-status">
              <div className="job-file">{attached.filename}</div>
              <div className="job-source">cité comme « {attached.source} »</div>
              <p className="job-detail job-ok">
                {attached.chunk_count} passages lus pour cette conversation
                {attached.ocr_used ? " (reconnaissance de texte appliquée)" : ""}.
                Posez votre question : ce document sera consulté en priorité.
              </p>
              {(attached.warnings ?? []).map((warning) => (
                <p key={warning} className="job-detail">{warning}</p>
              ))}
            </div>
          )}
        </div>

        <div className="modal-foot">
          {!started ? (
            <>
              <button type="button" className="btn-ghost" onClick={onClose}>
                Annuler
              </button>
              <button
                type="button"
                className="btn-primary"
                onClick={() => void submit()}
                disabled={!file || submitting}
              >
                <IconUpload size={15} />
                {submitting ? "Envoi…" : isCorpus ? "Ajouter au corpus" : "Joindre"}
              </button>
            </>
          ) : (
            <button type="button" className="btn-primary" onClick={onClose}>
              {done || failed ? "Fermer" : "Poursuivre en arrière-plan"}
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
