import { useEffect, useRef, useState } from "react";

import { IconBook, IconGlobe, IconPlus, IconSend, IconStop, IconUpload } from "../icons";
import type { Language, UserProfile } from "../types";

const PROFILES: { value: UserProfile; label: string; hint: string }[] = [
  { value: "citizen", label: "Citoyen", hint: "Langage simple, orienté démarches" },
  { value: "agent", label: "Agent public", hint: "Structuré, fondement légal de chaque étape" },
  { value: "enterprise", label: "Entreprise", hint: "Obligations, seuils, délais, sanctions" },
  { value: "jurist", label: "Juriste", hint: "Vocabulaire technique, références exhaustives" },
];

interface ComposerProps {
  onSend: (text: string) => void;
  onStop: () => void;
  busy: boolean;
  profile: UserProfile;
  onProfileChange: (profile: UserProfile) => void;
  language: Language;
  onLanguageChange: (language: Language) => void;
  webSearch: boolean;
  onWebSearchChange: (enabled: boolean) => void;
  onOpenCorpus: () => void;
  onOpenIngest: () => void;
}

export function Composer({
  onSend,
  onStop,
  busy,
  profile,
  onProfileChange,
  language,
  onLanguageChange,
  webSearch,
  onWebSearchChange,
  onOpenCorpus,
  onOpenIngest,
}: ComposerProps) {
  const [text, setText] = useState("");
  const [menuOpen, setMenuOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);

  // Hauteur suivant le contenu, plafonnée par la CSS.
  useEffect(() => {
    const el = textareaRef.current;
    if (!el) return;
    el.style.height = "auto";
    el.style.height = `${el.scrollHeight}px`;
  }, [text]);

  useEffect(() => {
    if (!menuOpen) return;
    const onPointerDown = (event: MouseEvent) => {
      if (!menuRef.current?.contains(event.target as Node)) setMenuOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") setMenuOpen(false);
    };
    document.addEventListener("mousedown", onPointerDown);
    document.addEventListener("keydown", onKeyDown);
    return () => {
      document.removeEventListener("mousedown", onPointerDown);
      document.removeEventListener("keydown", onKeyDown);
    };
  }, [menuOpen]);

  const submit = () => {
    const trimmed = text.trim();
    if (!trimmed || busy) return;
    onSend(trimmed);
    setText("");
  };

  const onKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    // Entrée envoie, Maj+Entrée passe à la ligne — convention des messageries.
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submit();
    }
  };

  const activeProfile = PROFILES.find((p) => p.value === profile);

  return (
    <div className="composer-wrap">
      <div className="composer">
        <textarea
          ref={textareaRef}
          rows={1}
          value={text}
          onChange={(event) => setText(event.target.value)}
          onKeyDown={onKeyDown}
          placeholder="Posez une question de droit ou d'administration…"
          aria-label="Votre question"
        />

        <div className="composer-bar">
          <div className="menu-anchor" ref={menuRef}>
            <button
              type="button"
              className="icon-button"
              onClick={() => setMenuOpen((open) => !open)}
              aria-haspopup="menu"
              aria-expanded={menuOpen}
              aria-label="Outils et documents"
            >
              <IconPlus />
            </button>

            {menuOpen && (
              <div className="menu" role="menu">
                <button
                  type="button"
                  className="menu-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenCorpus();
                    setMenuOpen(false);
                  }}
                >
                  <IconBook /> Documents du corpus
                </button>
                <button
                  type="button"
                  className="menu-item"
                  role="menuitem"
                  onClick={() => {
                    onWebSearchChange(!webSearch);
                    setMenuOpen(false);
                  }}
                >
                  <IconGlobe /> Recherche web
                  <span className="menu-item-note">{webSearch ? "activée" : "désactivée"}</span>
                </button>
                <div className="menu-sep" />
                <button
                  type="button"
                  className="menu-item"
                  role="menuitem"
                  onClick={() => {
                    onOpenIngest();
                    setMenuOpen(false);
                  }}
                >
                  <IconUpload /> Joindre un document à cette conversation
                </button>
              </div>
            )}
          </div>

          <button
            type="button"
            className={`tool-toggle${webSearch ? " on" : ""}`}
            onClick={() => onWebSearchChange(!webSearch)}
            aria-pressed={webSearch}
            title={
              webSearch
                ? "La recherche web complètera le corpus pour cette question"
                : "Répondre uniquement à partir du corpus indexé"
            }
          >
            <IconGlobe /> Recherche web
          </button>

          <div className="composer-spacer" />

          <select
            className="select-pill"
            value={profile}
            onChange={(event) => onProfileChange(event.target.value as UserProfile)}
            title={activeProfile?.hint}
            aria-label="Profil de réponse"
          >
            {PROFILES.map((item) => (
              <option key={item.value} value={item.value}>
                {item.label}
              </option>
            ))}
          </select>

          <select
            className="select-pill"
            value={language}
            onChange={(event) => onLanguageChange(event.target.value as Language)}
            aria-label="Langue de réponse"
          >
            <option value="fr">FR</option>
            <option value="en">EN</option>
          </select>

          {busy ? (
            <button
              type="button"
              className="send-button stop"
              onClick={onStop}
              aria-label="Arrêter la génération"
              title="Arrêter la génération"
            >
              <IconStop />
            </button>
          ) : (
            <button
              type="button"
              className="send-button"
              onClick={submit}
              disabled={!text.trim()}
              aria-label="Envoyer"
            >
              <IconSend />
            </button>
          )}
        </div>
      </div>

      <p className="composer-hint">
        Chaque affirmation renvoie à son texte source. Vérifiez les citations avant
        tout usage officiel.
      </p>
    </div>
  );
}
