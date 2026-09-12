import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import * as api from "./api";
import { Composer } from "./components/Composer";
import { CorpusView } from "./components/CorpusView";
import { EvaluationView } from "./components/EvaluationView";
import { FineTuningView } from "./components/FineTuningView";
import { HealthView } from "./components/HealthView";
import { IngestDialog } from "./components/IngestDialog";
import { Message } from "./components/Message";
import { Sidebar } from "./components/Sidebar";
import { SourcePanel } from "./components/SourcePanel";
import { IconBook, IconSidebar } from "./icons";
import type {
  AppView,
  ChatMessage,
  SessionDocument,
  Citation,
  CorpusDocument,
  Language,
  SessionSummary,
  UserProfile,
} from "./types";

const SUGGESTIONS = [
  { kind: "Question ponctuelle", text: "Quelles sont les conditions de la garde à vue ?" },
  { kind: "Vue d'ensemble", text: "Donne-moi une vue d'ensemble du Code de procédure pénale" },
  { kind: "Question ponctuelle", text: "Quelles sont les conditions du flagrant délit ?" },
  { kind: "Vue d'ensemble", text: "Que contient le Code pénal ?" },
];

const newId = () =>
  globalThis.crypto?.randomUUID?.() ?? `id-${Date.now()}-${Math.random().toString(16).slice(2)}`;

export function App() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [documents, setDocuments] = useState<CorpusDocument[]>([]);
  const [sessionId, setSessionId] = useState<string>(() => newId());
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null);

  const [profile, setProfile] = useState<UserProfile>("citizen");
  const [language, setLanguage] = useState<Language>("fr");
  const [webSearch, setWebSearch] = useState(false);

  const [busy, setBusy] = useState(false);
  const [panelMode, setPanelMode] = useState<"citation" | "corpus" | null>(null);
  const [selectedCitationId, setSelectedCitationId] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [ingestOpen, setIngestOpen] = useState(false);
  const [sessionDocs, setSessionDocs] = useState<SessionDocument[]>([]);
  const [view, setView] = useState<AppView>("chat");
  const [theme, setTheme] = useState<"dark" | "light">(() => {
    try {
      const stored = localStorage.getItem("govai-theme");
      if (stored === "light" || stored === "dark") return stored;
    } catch {
      /* stockage indisponible (navigation privée) : on garde le défaut */
    }
    return "dark";
  });

  const abortRef = useRef<AbortController | null>(null);
  const threadRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    document.documentElement.setAttribute("data-theme", theme);
    try {
      localStorage.setItem("govai-theme", theme);
    } catch {
      /* sans persistance, le thème reste valable pour la session */
    }
  }, [theme]);

  const refreshSessions = useCallback(async () => {
    try {
      setSessions(await api.listSessions());
    } catch {
      // La liste des conversations est un confort : son échec ne doit pas
      // empêcher de poser une question.
    }
  }, []);

  useEffect(() => {
    void refreshSessions();
    void api.listDocuments().then(setDocuments).catch(() => setDocuments([]));
  }, [refreshSessions]);

  // Suit le bas du fil pendant la génération.
  useEffect(() => {
    const el = threadRef.current;
    if (!el) return;
    const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 220;
    if (nearBottom) el.scrollTop = el.scrollHeight;
  }, [messages]);

  const lastCitations = useMemo(() => {
    for (let i = messages.length - 1; i >= 0; i -= 1) {
      const message = messages[i];
      if (message?.role === "assistant" && message.citations?.length) return message.citations;
    }
    return [] as Citation[];
  }, [messages]);

  const startNewChat = () => {
    setView("chat");
    abortRef.current?.abort();
    setMessages([]);
    setSessionId(newId());
    setActiveSessionId(null);
    setSessionDocs([]);
    setSelectedCitationId(null);
    setPanelMode(null);
    setSidebarOpen(false);
  };

  const openSession = async (id: string) => {
    setView("chat");
    abortRef.current?.abort();
    setSidebarOpen(false);
    try {
      const detail = await api.getSession(id);
      const restored: ChatMessage[] = [];
      for (const turn of detail.turns) {
        restored.push({ id: `${turn.turn_id}-q`, role: "user", content: turn.query });
        restored.push({
          id: turn.turn_id,
          role: "assistant",
          content: turn.answer_truncated
            ? `${turn.answer}\n\n*(Réponse enregistrée avant la conservation intégrale : seul un aperçu est disponible.)*`
            : turn.answer,
          citations: turn.citations,
          safetyFlags: turn.safety_flags,
          intent: turn.intent,
          latencyMs: turn.latency_ms,
          modelUsed: turn.model_used,
        });
      }
      setMessages(restored);
      setSessionId(detail.session_id);
      setActiveSessionId(detail.session_id);
      void api.listSessionDocuments(detail.session_id).then(setSessionDocs).catch(() => setSessionDocs([]));
      if (detail.profile) setProfile(detail.profile as UserProfile);
      if (detail.language === "fr" || detail.language === "en") setLanguage(detail.language);
    } catch (error) {
      setMessages([
        {
          id: newId(),
          role: "assistant",
          content: "",
          error: error instanceof Error ? error.message : "Conversation illisible.",
        },
      ]);
    }
  };

  const removeSession = async (id: string) => {
    try {
      await api.deleteSession(id);
      if (id === activeSessionId) startNewChat();
      await refreshSessions();
    } catch {
      /* la conversation reste affichée : rien de destructeur n'a eu lieu */
    }
  };

  const send = async (text: string) => {
    const assistantId = newId();
    setMessages((prev) => [
      ...prev,
      { id: newId(), role: "user", content: text },
      { id: assistantId, role: "assistant", content: "", streaming: true },
    ]);
    setBusy(true);
    setSelectedCitationId(null);

    const controller = new AbortController();
    abortRef.current = controller;

    const patch = (changes: Partial<ChatMessage>) =>
      setMessages((prev) =>
        prev.map((message) =>
          message.id === assistantId ? { ...message, ...changes } : message,
        ),
      );

    let buffer = "";

    try {
      await api.askStream(
        {
          query: text,
          session_id: sessionId,
          language,
          profile,
          web_search: webSearch ? true : null,
          top_k: 5,
        },
        (event) => {
          if (event.type === "stage") {
            patch({ stage: event.label });
          } else if (event.type === "token") {
            buffer += event.v;
            patch({ content: buffer, stage: undefined });
          } else if (event.type === "done") {
            patch({
              content: event.answer || buffer,
              streaming: false,
              citations: event.citations,
              webSources: event.web_sources ?? [],
              graphEvidence: event.graph_evidence ?? [],
              safetyFlags: event.safety_flags,
              warnings: event.warnings,
              intent: event.intent ?? null,
              latencyMs: event.latency_ms,
              modelUsed: event.model_used,
            });
            if (event.citations.length > 0) setPanelMode("citation");
          } else if (event.type === "error") {
            patch({ streaming: false, error: event.message });
          }
        },
        controller.signal,
      );
    } catch (error) {
      if (controller.signal.aborted) {
        // Arrêt demandé : on garde le texte déjà reçu plutôt que de le perdre.
        patch({
          streaming: false,
          content: buffer ? `${buffer}\n\n*(Génération interrompue.)*` : "",
          error: buffer ? undefined : "Génération interrompue avant toute réponse.",
        });
      } else {
        patch({
          streaming: false,
          error:
            error instanceof Error
              ? error.message
              : "Le serveur n'a pas répondu. Vérifiez que l'API est démarrée.",
        });
      }
    } finally {
      setBusy(false);
      abortRef.current = null;
      setActiveSessionId(sessionId);
      void refreshSessions();
    }
  };

  const stop = () => abortRef.current?.abort();

  const selectCitation = (citation: Citation) => {
    setSelectedCitationId(citation.chunk_id);
    setPanelMode("citation");
    requestAnimationFrame(() => {
      document
        .getElementById(`source-${citation.chunk_id}`)
        ?.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
  };

  const shellClass = [
    "shell",
    panelMode ? "panel-open" : "",
    sidebarOpen ? "sidebar-mobile-open" : "",
  ]
    .filter(Boolean)
    .join(" ");

  return (
    <div className={shellClass}>
      <Sidebar
        sessions={sessions}
        activeSessionId={activeSessionId}
        onNewChat={startNewChat}
        onSelectSession={openSession}
        onDeleteSession={removeSession}
        onOpenCorpus={() => setPanelMode("corpus")}
        theme={theme}
        onToggleTheme={() => setTheme((t) => (t === "dark" ? "light" : "dark"))}
        corpusCount={documents.length}
        view={view}
        onViewChange={(next) => {
          setView(next);
          setSidebarOpen(false);
        }}
      />

      {sidebarOpen && (
        <button
          type="button"
          className="scrim"
          aria-label="Fermer le menu"
          onClick={() => setSidebarOpen(false)}
        />
      )}

      <main className="main">
        <div className="topbar">
          <button
            type="button"
            className="icon-button"
            onClick={() => setSidebarOpen((open) => !open)}
            aria-label="Afficher les conversations"
          >
            <IconSidebar />
          </button>
          <span className="topbar-title">
            {view === "corpus"
              ? "Corpus documentaire"
              : view === "health"
              ? "Santé des services"
              : view === "evaluation"
                ? "Évaluation"
                : view === "finetuning"
                  ? "Fine-tuning"
                  : messages.length === 0
                    ? "Nouvelle conversation"
                    : messages[0]?.content}
          </span>
          <button
            type="button"
            className="icon-button"
            onClick={() => setPanelMode(panelMode === "corpus" ? null : "corpus")}
            aria-label="Documents du corpus"
          >
            <IconBook size={17} />
          </button>
        </div>

        {view === "health" && <HealthView />}
        {view === "corpus" && <CorpusView />}
        {view === "evaluation" && <EvaluationView />}
        {view === "finetuning" && <FineTuningView />}

        {view === "chat" && sessionDocs.length > 0 && (
          <div className="attachments">
            <span className="attachments-label">
              Document{sessionDocs.length > 1 ? "s" : ""} joint
              {sessionDocs.length > 1 ? "s" : ""} à cette conversation
            </span>
            {sessionDocs.map((doc) => (
              <span key={doc.document_id} className="attachment">
                <span className="attachment-name">{doc.source}</span>
                <span className="attachment-meta">{doc.chunk_count} passages</span>
                <button
                  type="button"
                  className="attachment-remove"
                  aria-label={`Détacher « ${doc.source} »`}
                  onClick={() => {
                    void api
                      .detachSessionDocument(sessionId, doc.document_id)
                      .then(() =>
                        setSessionDocs((docs) =>
                          docs.filter((d) => d.document_id !== doc.document_id),
                        ),
                      )
                      .catch(() => undefined);
                  }}
                >
                  ×
                </button>
              </span>
            ))}
          </div>
        )}

        {view === "chat" && (
        <div className="thread" ref={threadRef}>
          {messages.length === 0 ? (
            <div className="welcome">
              <h1>Que pouvons-nous chercher dans les textes ?</h1>
              <p>
                Chaque réponse cite le texte source dont elle est tirée. Le corpus
                contient actuellement{" "}
                {documents.length > 0
                  ? documents.map((d) => d.source).join(" et ")
                  : "aucun document"}
                .
              </p>
              <div className="suggestions">
                {SUGGESTIONS.map((suggestion) => (
                  <button
                    key={suggestion.text}
                    type="button"
                    className="suggestion"
                    onClick={() => void send(suggestion.text)}
                  >
                    <span className="suggestion-kind">{suggestion.kind}</span>
                    {suggestion.text}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="thread-inner">
              {messages.map((message) => (
                <Message
                  key={message.id}
                  message={message}
                  selectedCitationId={selectedCitationId}
                  onSelectCitation={selectCitation}
                />
              ))}
            </div>
          )}
        </div>
        )}

        {view === "chat" && (
        <Composer
          onSend={(text) => void send(text)}
          onStop={stop}
          busy={busy}
          profile={profile}
          onProfileChange={setProfile}
          language={language}
          onLanguageChange={setLanguage}
          webSearch={webSearch}
          onWebSearchChange={setWebSearch}
          onOpenCorpus={() => setPanelMode("corpus")}
          onOpenIngest={() => setIngestOpen(true)}
        />
        )}
      </main>

      {ingestOpen && (
        <IngestDialog
          scope="conversation"
          sessionId={sessionId}
          onClose={() => setIngestOpen(false)}
          onIngested={() => {
            // La pièce jointe n'entre pas au corpus : rien à rafraîchir côté
            // corpus, on recharge la liste des documents de la conversation.
            void api
              .listSessionDocuments(sessionId)
              .then(setSessionDocs)
              .catch(() => undefined);
          }}
        />
      )}

      {panelMode && (
        <SourcePanel
          mode={panelMode}
          citations={lastCitations}
          selectedCitationId={selectedCitationId}
          documents={documents}
          onClose={() => setPanelMode(null)}
        />
      )}
    </div>
  );
}
