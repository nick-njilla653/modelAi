import {
  IconBook,
  IconChart,
  IconChat,
  IconMoon,
  IconPlus,
  IconPulse,
  IconScale,
  IconSliders,
  IconSun,
  IconTrash,
} from "../icons";
import type { AppView, SessionSummary } from "../types";

/** Regroupe les conversations comme une messagerie : aujourd'hui, hier, avant. */
function groupSessions(sessions: SessionSummary[]) {
  const now = new Date();
  const startOfToday = new Date(now.getFullYear(), now.getMonth(), now.getDate()).getTime();
  const startOfYesterday = startOfToday - 86_400_000;
  const startOfWeek = startOfToday - 7 * 86_400_000;

  const groups: { label: string; items: SessionSummary[] }[] = [
    { label: "Aujourd'hui", items: [] },
    { label: "Hier", items: [] },
    { label: "7 derniers jours", items: [] },
    { label: "Plus ancien", items: [] },
  ];

  for (const session of sessions) {
    const stamp = session.last_active ? Date.parse(session.last_active) : 0;
    if (stamp >= startOfToday) groups[0]!.items.push(session);
    else if (stamp >= startOfYesterday) groups[1]!.items.push(session);
    else if (stamp >= startOfWeek) groups[2]!.items.push(session);
    else groups[3]!.items.push(session);
  }

  return groups.filter((group) => group.items.length > 0);
}

interface SidebarProps {
  sessions: SessionSummary[];
  activeSessionId: string | null;
  onNewChat: () => void;
  onSelectSession: (sessionId: string) => void;
  onDeleteSession: (sessionId: string) => void;
  onOpenCorpus: () => void;
  theme: "dark" | "light";
  onToggleTheme: () => void;
  corpusCount: number;
  view: AppView;
  onViewChange: (view: AppView) => void;
}

export function Sidebar({
  sessions,
  activeSessionId,
  onNewChat,
  onSelectSession,
  onDeleteSession,
  onOpenCorpus,
  theme,
  onToggleTheme,
  corpusCount,
  view,
  onViewChange,
}: SidebarProps) {
  const groups = groupSessions(sessions);

  return (
    <aside className="sidebar">
      <div className="sidebar-head">
        <div className="brand">
          <span className="brand-mark"><IconScale /></span>
          <span>
            <span className="brand-name">GOV-AI 2.0</span>
            <span className="brand-sub">Administration camerounaise</span>
          </span>
        </div>

        <button type="button" className="new-chat" onClick={onNewChat}>
          <IconPlus size={16} /> Nouvelle conversation
        </button>

        <nav className="view-nav" aria-label="Sections">
          {([
            { id: "chat", label: "Conversation", icon: <IconChat size={14} /> },
            { id: "corpus", label: "Corpus", icon: <IconBook size={14} /> },
            { id: "evaluation", label: "Évaluation", icon: <IconChart size={14} /> },
            { id: "health", label: "Santé", icon: <IconPulse size={14} /> },
            { id: "finetuning", label: "Fine-tuning", icon: <IconSliders size={14} /> },
          ] as const).map((item) => (
            <button
              key={item.id}
              type="button"
              className={`view-nav-item${view === item.id ? " active" : ""}`}
              onClick={() => onViewChange(item.id)}
              aria-current={view === item.id ? "page" : undefined}
            >
              {item.icon} {item.label}
            </button>
          ))}
        </nav>
      </div>

      {view === "chat" && (
      <div className="session-list">
        {groups.length === 0 && (
          <p className="panel-empty">Aucune conversation pour l'instant.</p>
        )}

        {groups.map((group) => (
          <div key={group.label}>
            <div className="sidebar-section">{group.label}</div>
            {group.items.map((session) => (
              <div
                key={session.session_id}
                className={`session-item${
                  session.session_id === activeSessionId ? " active" : ""
                }`}
                onClick={() => onSelectSession(session.session_id)}
                onKeyDown={(event) => {
                  if (event.key === "Enter" || event.key === " ") {
                    event.preventDefault();
                    onSelectSession(session.session_id);
                  }
                }}
                role="button"
                tabIndex={0}
              >
                <span className="session-title">{session.title}</span>
                <button
                  type="button"
                  className="session-delete"
                  aria-label={`Supprimer « ${session.title} »`}
                  onClick={(event) => {
                    event.stopPropagation();
                    onDeleteSession(session.session_id);
                  }}
                >
                  <IconTrash />
                </button>
              </div>
            ))}
          </div>
        ))}
      </div>
      )}

      {view !== "chat" && <div className="session-list" />}

      <div className="sidebar-foot">
        <button type="button" className="corpus-button" onClick={onOpenCorpus}>
          <IconBook size={14} /> Corpus
          {corpusCount > 0 && <span className="lang-tag">{corpusCount}</span>}
        </button>
        <button
          type="button"
          className="icon-button"
          onClick={onToggleTheme}
          aria-label={theme === "dark" ? "Passer en thème clair" : "Passer en thème sombre"}
        >
          {theme === "dark" ? <IconSun /> : <IconMoon />}
        </button>
      </div>
    </aside>
  );
}
