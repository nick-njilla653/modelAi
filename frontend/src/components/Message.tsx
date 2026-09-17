import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { IconAlert, IconGlobe, IconGraph, IconQuote } from "../icons";
import type { ChatMessage, Citation, GraphEvidence, SafetyFlag } from "../types";

/** Étiquette du niveau de confiance d'une source web. */
const PRIORITY_LABELS: Record<number, string> = {
  0: "Institution suprême",
  1: "Ministère",
  2: "Administration",
  3: "Organisme public",
  4: "Organisation régionale",
  5: "Presse publique",
};

/**
 * Décrit ce que le graphe a établi, en distinguant les deux portées.
 *
 * « nommée par les dispositions retrouvées » et « citée dans 66 articles du
 * corpus » ne répondent pas à la même question : la première est déduite des
 * cinq passages de cette réponse, la seconde compte sur tout le corpus.
 * Afficher le décompte global sous la première ferait croire à une couverture
 * que le graphe n'a pas établie ici.
 */
function describeScope(piece: GraphEvidence): string {
  const refs = piece.articles
    .map((a) => `art. ${a.number}`)
    .join(", ");

  if (piece.kind === "article") {
    const doc = piece.detail ? ` (${piece.detail})` : "";
    return piece.article_count > 0
      ? `${piece.article_count} renvoi${piece.article_count > 1 ? "s" : ""} sortant${
          piece.article_count > 1 ? "s" : ""
        }${doc}`
      : `Disposition retrouvée dans le graphe${doc}`;
  }

  if (piece.scope === "passages") {
    return `Nommée par les dispositions retrouvées${refs ? ` : ${refs}` : ""}`;
  }

  const tronque = piece.article_count > piece.articles.length ? ", liste partielle" : "";
  return `Citée dans ${piece.article_count} article${
    piece.article_count > 1 ? "s" : ""
  } du corpus${refs ? ` : ${refs}` : ""}${tronque}`;
}

/**
 * Libellés des drapeaux de sûreté remontés par le pipeline de vérification.
 * Le drapeau brut (« UNSUPPORTED_CLAIMS ») ne dit rien à un citoyen : on nomme
 * la conséquence, pas le code interne.
 */
const FLAG_LABELS: Record<SafetyFlag, { title: string; body: string; tone: "warn" | "bad" }> = {
  UNSUPPORTED_CLAIMS: {
    title: "Affirmations non appuyées",
    body: "Une partie de la réponse ne renvoie à aucun extrait vérifiable du corpus. Recoupez avec le texte officiel.",
    tone: "bad",
  },
  ESCALATION_RECOMMENDED: {
    title: "Confiance faible",
    body: "Les documents trouvés couvrent mal la question. Consultez l'autorité compétente ou un professionnel.",
    tone: "warn",
  },
  LOW_CONFIDENCE: {
    title: "Confiance faible",
    body: "Le rapprochement entre votre question et le corpus est incertain.",
    tone: "warn",
  },
  OUT_OF_CORPUS: {
    title: "Hors corpus",
    body: "La question sort des documents indexés.",
    tone: "warn",
  },
  CONTRADICTION_DETECTED: {
    title: "Sources divergentes",
    body: "Les extraits retenus se contredisent sur un point.",
    tone: "warn",
  },
  JURIDICAL_DIVERGENCE: {
    title: "Divergence bijuridique",
    body: "Droit civil et common law traitent ce point différemment.",
    tone: "warn",
  },
  PROMPT_INJECTION_ATTEMPT: {
    title: "Requête bloquée",
    body: "La requête a été rejetée pour des raisons de sécurité.",
    tone: "bad",
  },
};

function citationLabel(citation: Citation): string {
  if (citation.article) return citation.article;
  if (citation.page) return `p. ${citation.page}`;
  return "source";
}

interface MessageProps {
  message: ChatMessage;
  selectedCitationId: string | null;
  onSelectCitation: (citation: Citation) => void;
}

export function Message({ message, selectedCitationId, onSelectCitation }: MessageProps) {
  if (message.role === "user") {
    return (
      <div className="msg user">
        <div className="msg-user-bubble">{message.content}</div>
      </div>
    );
  }

  const flags = message.safetyFlags ?? [];
  const warnings = message.warnings ?? [];
  const citations = message.citations ?? [];

  return (
    <div className="msg assistant">
      <div className="msg-role">GOV-AI 2.0</div>

      <div className="msg-body">
        {message.error ? (
          <div className="notice notice-bad">
            <span className="notice-icon"><IconAlert /></span>
            <div className="notice-body">
              <span className="notice-title">La requête n'a pas abouti</span>
              {message.error}
            </div>
          </div>
        ) : (
          <>
            <ReactMarkdown
              remarkPlugins={[remarkGfm]}
              components={{
                // Un code juridique produit de vrais tableaux : ils doivent
                // défiler seuls plutôt que d'élargir toute la conversation.
                table: ({ children }) => (
                  <div className="table-wrap"><table>{children}</table></div>
                ),
              }}
            >
              {message.content}
            </ReactMarkdown>
            {message.streaming && !message.content && message.stage && (
              <p className="stage-line">
                <span className="stage-dot" /> {message.stage}…
              </p>
            )}
            {message.streaming && <span className="caret" />}
          </>
        )}
      </div>

      {flags.map((flag) => {
        const info = FLAG_LABELS[flag];
        if (!info) return null;
        return (
          <div key={flag} className={`notice notice-${info.tone}`}>
            <span className="notice-icon"><IconAlert /></span>
            <div className="notice-body">
              <span className="notice-title">{info.title}</span>
              {info.body}
            </div>
          </div>
        );
      })}

      {warnings
        .filter((w) => w.startsWith("Références citées introuvables"))
        .map((warning) => (
          <div key={warning} className="notice notice-bad">
            <span className="notice-icon"><IconAlert /></span>
            <div className="notice-body">
              <span className="notice-title">Références invérifiables</span>
              {warning.replace("Références citées introuvables dans le contexte fourni : ", "")}
            </div>
          </div>
        ))}

      {citations.length > 0 && (
        <div className="citations">
          <span className="citations-label">
            <IconQuote size={11} /> {citations.length} source
            {citations.length > 1 ? "s" : ""} vérifiable{citations.length > 1 ? "s" : ""}
          </span>
          {citations.map((citation) => (
            <button
              key={citation.chunk_id}
              type="button"
              className={`citation-chip${
                selectedCitationId === citation.chunk_id ? " selected" : ""
              }`}
              onClick={() => onSelectCitation(citation)}
              title={`${citation.doc_title} — ${citationLabel(citation)}`}
            >
              <span className="citation-ref">{citationLabel(citation)}</span>
              <span className="citation-doc">{citation.doc_title}</span>
            </button>
          ))}
        </div>
      )}

      {(message.webSources?.length ?? 0) > 0 && (
        <div className="web-sources">
          <span className="citations-label">
            <IconGlobe size={11} /> {message.webSources!.length} source
            {message.webSources!.length > 1 ? "s" : ""} officielle
            {message.webSources!.length > 1 ? "s" : ""} en ligne
          </span>
          <p className="web-sources-caveat">
            Ces pages émanent d'institutions publiques mais ne sont pas le texte
            normatif lui-même : elles peuvent être datées.
          </p>
          {message.webSources!.map((source) => (
            <a
              key={source.url}
              className="web-source"
              href={source.url}
              target="_blank"
              rel="noreferrer noopener"
            >
              <span className="web-source-head">
                <span className="web-source-acronym">{source.acronym}</span>
                <span className="web-source-kind">
                  {PRIORITY_LABELS[source.priority] ?? "Source officielle"}
                </span>
                {!source.reachable && (
                  <span className="web-source-stale">injoignable au dernier contrôle</span>
                )}
              </span>
              <span className="web-source-title">{source.title || source.domain}</span>
              <span className="web-source-domain">{source.domain}</span>
            </a>
          ))}
        </div>
      )}

      {(message.graphEvidence?.length ?? 0) > 0 && (
        <div className="graph-evidence">
          <span className="citations-label">
            <IconGraph size={11} /> Apport du graphe de connaissances
          </span>
          <p className="graph-evidence-caveat">
            Des relations entre dispositions, et non des extraits : le graphe
            indique quelle autorité un article nomme, pas ce qu'il dit.
          </p>
          {message.graphEvidence!.map((piece) => (
            <div key={`${piece.kind}-${piece.label}`} className="graph-piece">
              <span className="graph-piece-head">
                <span className="graph-piece-label">{piece.label}</span>
                {piece.detail && (
                  <span className="graph-piece-kind">{piece.detail}</span>
                )}
              </span>
              <span className="graph-piece-body">{describeScope(piece)}</span>
              {piece.issued_texts.length > 0 && (
                <span className="graph-piece-body">
                  Émettrice de&nbsp;: {piece.issued_texts.join(", ")}
                </span>
              )}
            </div>
          ))}
        </div>
      )}

      {!message.streaming && !message.error && (message.latencyMs || message.intent) && (
        <div className="msg-meta">
          {message.intent && <span>{message.intent}</span>}
          {message.modelUsed && <span>{message.modelUsed}</span>}
          {message.latencyMs != null && (
            <span>{(message.latencyMs / 1000).toFixed(1)} s</span>
          )}
        </div>
      )}
    </div>
  );
}
