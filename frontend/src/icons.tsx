/**
 * Icônes SVG inline.
 *
 * Aucune police d'icônes ni bibliothèque : le déploiement visé est souverain et
 * hors ligne, une dépendance à un CDN d'icônes serait un point de rupture.
 */
type IconProps = { size?: number; className?: string };

const base = (size: number) => ({
  width: size,
  height: size,
  viewBox: "0 0 24 24",
  fill: "none",
  stroke: "currentColor",
  strokeWidth: 1.7,
  strokeLinecap: "round" as const,
  strokeLinejoin: "round" as const,
  "aria-hidden": true,
});

export const IconPlus = ({ size = 18 }: IconProps) => (
  <svg {...base(size)}><path d="M12 5v14M5 12h14" /></svg>
);

export const IconSend = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}><path d="M12 19V5M5 12l7-7 7 7" /></svg>
);

export const IconStop = ({ size = 15 }: IconProps) => (
  <svg {...base(size)}><rect x="7" y="7" width="10" height="10" rx="1.5" fill="currentColor" /></svg>
);

export const IconSidebar = ({ size = 18 }: IconProps) => (
  <svg {...base(size)}>
    <rect x="3" y="4" width="18" height="16" rx="2" />
    <path d="M9 4v16" />
  </svg>
);

export const IconGlobe = ({ size = 15 }: IconProps) => (
  <svg {...base(size)}>
    <circle cx="12" cy="12" r="9" />
    <path d="M3 12h18M12 3a15 15 0 0 1 0 18a15 15 0 0 1 0-18" />
  </svg>
);

export const IconBook = ({ size = 15 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M4 5.5A2.5 2.5 0 0 1 6.5 3H19v15H6.5A2.5 2.5 0 0 0 4 20.5z" />
    <path d="M4 20.5A2.5 2.5 0 0 1 6.5 18H19v3H6.5A2.5 2.5 0 0 1 4 20.5z" />
  </svg>
);

export const IconTrash = ({ size = 14 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M4 7h16M9 7V5h6v2M6 7l1 13h10l1-13" />
  </svg>
);

export const IconClose = ({ size = 17 }: IconProps) => (
  <svg {...base(size)}><path d="M6 6l12 12M18 6L6 18" /></svg>
);

export const IconQuote = ({ size = 15 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M7 15a3 3 0 1 1 0-6c0-2 1-3.5 3-4M16 15a3 3 0 1 1 0-6c0-2 1-3.5 3-4" />
  </svg>
);

export const IconAlert = ({ size = 15 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M12 4l9 16H3z" />
    <path d="M12 10v4M12 17h.01" />
  </svg>
);

export const IconScale = ({ size = 17 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M12 4v16M7 20h10M5 8h14M5 8l-2 6h4zM19 8l2 6h-4z" />
  </svg>
);

export const IconSun = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}>
    <circle cx="12" cy="12" r="4" />
    <path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4" />
  </svg>
);

export const IconMoon = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}><path d="M20 14.5A8.5 8.5 0 0 1 9.5 4a8.5 8.5 0 1 0 10.5 10.5z" /></svg>
);

export const IconSliders = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M4 6h10M18 6h2M4 12h4M12 12h8M4 18h12M20 18h0" />
    <circle cx="16" cy="6" r="2" />
    <circle cx="10" cy="12" r="2" />
    <circle cx="18" cy="18" r="2" />
  </svg>
);

export const IconChat = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M21 12a8 8 0 0 1-8 8H7l-4 3v-4.5A8 8 0 0 1 11 4h2a8 8 0 0 1 8 8z" />
  </svg>
);

export const IconChart = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M4 20V10M10 20V4M16 20v-7M22 20H2" />
  </svg>
);

export const IconPulse = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}><path d="M3 12h4l2.5-7 4 14L16 12h5" /></svg>
);

export const IconUpload = ({ size = 16 }: IconProps) => (
  <svg {...base(size)}>
    <path d="M12 16V5M8 9l4-4 4 4" />
    <path d="M4 17v2a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-2" />
  </svg>
);

/** Trois nœuds reliés : la relation, non le document. */
export const IconGraph = ({ size = 15 }: IconProps) => (
  <svg {...base(size)}>
    <circle cx="6" cy="6" r="2.4" />
    <circle cx="18" cy="10" r="2.4" />
    <circle cx="9" cy="19" r="2.4" />
    <path d="M8.1 7.3 15.9 9M7.2 8.2 8.4 16.6M16.6 12.1 10.8 17.4" />
  </svg>
);
