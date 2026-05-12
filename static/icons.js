/* ── SPADE SVG Icon Library ─────────────────────────────────────────────── */

/* ── Category SVG icons (64×64, stroke="currentColor") ───────────────────── */
const CAT_SVGS = {
  bottle: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <path d="M25 10h14v4l6 9v26a4 4 0 0 1-4 4H23a4 4 0 0 1-4-4V23l6-9V10z"/>
    <line x1="19" y1="30" x2="45" y2="30" stroke-opacity="0.35"/>
  </svg>`,

  cable: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <path d="M6 22 C14 22 14 14 22 14 C30 14 30 30 38 30 C46 30 46 18 54 18"/>
    <path d="M6 44 C14 44 14 36 22 36 C30 36 30 52 38 52 C46 52 46 40 54 40"/>
    <rect x="2" y="18" width="6" height="8" rx="1.5" fill="currentColor" fill-opacity="0.12"/>
    <rect x="56" y="36" width="6" height="8" rx="1.5" fill="currentColor" fill-opacity="0.12"/>
  </svg>`,

  capsule: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
    <rect x="10" y="22" width="44" height="20" rx="10"/>
    <line x1="32" y1="22" x2="32" y2="42"/>
    <rect x="10" y="22" width="22" height="20" rx="10" fill="currentColor" fill-opacity="0.08"/>
  </svg>`,

  carpet: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.6">
    <rect x="8" y="8" width="48" height="48" rx="4"/>
    <line x1="8" y1="20" x2="56" y2="20" stroke-opacity="0.6"/>
    <line x1="8" y1="32" x2="56" y2="32" stroke-opacity="0.6"/>
    <line x1="8" y1="44" x2="56" y2="44" stroke-opacity="0.6"/>
    <line x1="20" y1="8" x2="20" y2="56" stroke-opacity="0.6"/>
    <line x1="32" y1="8" x2="32" y2="56" stroke-opacity="0.6"/>
    <line x1="44" y1="8" x2="44" y2="56" stroke-opacity="0.6"/>
  </svg>`,

  grid: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2.5" stroke-linecap="round">
    <line x1="22" y1="8"  x2="20" y2="56"/>
    <line x1="44" y1="8"  x2="42" y2="56"/>
    <line x1="8"  y1="22" x2="56" y2="18"/>
    <line x1="10" y1="44" x2="58" y2="40"/>
  </svg>`,

  hazelnut: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
    <ellipse cx="32" cy="38" rx="20" ry="20"/>
    <path d="M32 18 Q28 10 32 6 Q36 10 32 18" fill="currentColor" fill-opacity="0.1"/>
    <line x1="32" y1="22" x2="32" y2="54" stroke-opacity="0.3"/>
    <line x1="14" y1="32" x2="50" y2="32" stroke-opacity="0.3"/>
    <line x1="15" y1="44" x2="49" y2="44" stroke-opacity="0.3"/>
  </svg>`,

  leather: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.5">
    <rect x="8" y="8" width="48" height="48" rx="4"/>
    <path d="M8 20L20 8M8 32L32 8M8 44L44 8M8 56L56 8M20 56L56 20M32 56L56 32M44 56L56 44" stroke-opacity="0.4"/>
  </svg>`,

  metal_nut: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8">
    <polygon points="32,6 54,19 54,45 32,58 10,45 10,19"/>
    <circle cx="32" cy="32" r="10"/>
  </svg>`,

  pill: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round">
    <rect x="8" y="22" width="48" height="20" rx="10"/>
    <line x1="32" y1="22" x2="32" y2="42" stroke-opacity="0.4"/>
    <rect x="8" y="22" width="24" height="20" rx="10" fill="currentColor" fill-opacity="0.07"/>
  </svg>`,

  screw: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
    <circle cx="32" cy="18" r="13"/>
    <line x1="20" y1="18" x2="44" y2="18"/>
    <line x1="32" y1="6"  x2="32" y2="30"/>
    <rect x="27" y="31" width="10" height="24" rx="2"/>
    <line x1="27" y1="38" x2="37" y2="38" stroke-opacity="0.45"/>
    <line x1="27" y1="45" x2="37" y2="45" stroke-opacity="0.45"/>
    <line x1="27" y1="52" x2="37" y2="52" stroke-opacity="0.45"/>
  </svg>`,

  tile: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <rect x="6"  y="6"  width="24" height="24" rx="3"/>
    <rect x="34" y="6"  width="24" height="24" rx="3"/>
    <rect x="6"  y="34" width="24" height="24" rx="3"/>
    <rect x="34" y="34" width="24" height="24" rx="3"/>
  </svg>`,

  toothbrush: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <rect x="22" y="6" width="20" height="36" rx="5"/>
    <rect x="18" y="38" width="28" height="12" rx="4"/>
    <line x1="24" y1="12" x2="24" y2="38" stroke-opacity="0.35" stroke-width="1.4"/>
    <line x1="32" y1="12" x2="32" y2="38" stroke-opacity="0.35" stroke-width="1.4"/>
    <line x1="40" y1="12" x2="40" y2="38" stroke-opacity="0.35" stroke-width="1.4"/>
    <line x1="32" y1="50" x2="32" y2="58" stroke-width="2"/>
  </svg>`,

  transistor: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <rect x="12" y="14" width="40" height="28" rx="4"/>
    <circle cx="32" cy="28" r="7"/>
    <line x1="20" y1="42" x2="20" y2="58"/>
    <line x1="32" y1="42" x2="32" y2="58"/>
    <line x1="44" y1="42" x2="44" y2="58"/>
  </svg>`,

  wood: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round">
    <rect x="6" y="12" width="52" height="40" rx="4"/>
    <path d="M6 23 Q18 21 32 23 Q46 25 58 23" stroke-opacity="0.45"/>
    <path d="M6 33 Q18 31 32 33 Q46 35 58 33" stroke-opacity="0.45"/>
    <path d="M6 43 Q18 41 32 43 Q46 45 58 43" stroke-opacity="0.45"/>
  </svg>`,

  zipper: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
    <line x1="32" y1="4" x2="32" y2="60" stroke-width="2"/>
    <rect x="17" y="10" width="13" height="8" rx="2"/>
    <rect x="34" y="19" width="13" height="8" rx="2"/>
    <rect x="17" y="28" width="13" height="8" rx="2"/>
    <rect x="34" y="37" width="13" height="8" rx="2"/>
    <rect x="17" y="46" width="13" height="8" rx="2"/>
  </svg>`,

  default: `<svg width="64" height="64" viewBox="0 0 64 64" fill="none" stroke="currentColor" stroke-width="1.8">
    <rect x="10" y="10" width="44" height="44" rx="8"/>
    <circle cx="32" cy="32" r="12"/>
  </svg>`,
};

const ICONS = {

  lock: `<svg width="12" height="14" viewBox="0 0 12 14" fill="none" xmlns="http://www.w3.org/2000/svg">
    <rect x="1" y="6" width="10" height="7" rx="2" stroke="currentColor" stroke-width="1.3"/>
    <path d="M3.5 6V4a2.5 2.5 0 0 1 5 0v2" stroke="currentColor" stroke-width="1.3" stroke-linecap="round"/>
    <circle cx="6" cy="9.5" r="1" fill="currentColor"/>
  </svg>`,

  radar: `<svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">
    <circle cx="10" cy="10" r="8.5" stroke="rgba(255,255,255,0.5)" stroke-width="1.2"/>
    <circle cx="10" cy="10" r="5"   stroke="rgba(255,255,255,0.3)" stroke-width="1"/>
    <circle cx="10" cy="10" r="1.5" fill="rgba(255,255,255,0.8)"/>
    <line x1="10" y1="10" x2="10" y2="1.5" stroke="rgba(255,255,255,0.9)" stroke-width="1.5" stroke-linecap="round"
      style="transform-origin:10px 10px; animation:radarSweep 2s linear infinite"/>
    <circle cx="10" cy="4" r="1.5" fill="rgba(6,182,212,0.8)"
      style="animation:radarPing 2s linear infinite"/>
    <style>
      @keyframes radarSweep { from{transform:rotate(0deg)} to{transform:rotate(360deg)} }
      @keyframes radarPing  { 0%,100%{opacity:0} 15%,30%{opacity:1} }
    </style>
  </svg>`,

  warning: `<svg width="18" height="18" viewBox="0 0 18 18" fill="none">
    <path d="M9 2L16.5 15H1.5L9 2Z" stroke="currentColor" stroke-width="1.5" stroke-linejoin="round"/>
    <line x1="9" y1="7" x2="9" y2="11" stroke="currentColor" stroke-width="1.5" stroke-linecap="round"/>
    <circle cx="9" cy="13.5" r="0.8" fill="currentColor"/>
  </svg>`,

  check: `<svg width="18" height="18" viewBox="0 0 18 18" fill="none">
    <circle cx="9" cy="9" r="7.5" stroke="currentColor" stroke-width="1.5"/>
    <path d="M5.5 9l2.5 2.5 4.5-5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`,

  backArrow: `<svg width="16" height="16" viewBox="0 0 16 16" fill="none">
    <path d="M10 3L5 8l5 5" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"/>
  </svg>`,

  hexReticle(size = 120) {
    const cx = size / 2, cy = size / 2;
    const r  = size * 0.46;
    const pts = Array.from({length: 6}, (_, i) => {
      const a = (i * 60 - 90) * Math.PI / 180;
      return `${cx + r * Math.cos(a)},${cy + r * Math.sin(a)}`;
    }).join(' ');
    const perim = 6 * r; // approximate for regular hex
    return `<svg width="${size}" height="${size}" viewBox="0 0 ${size} ${size}" fill="none">
      <polygon points="${pts}" stroke="#06B6D4" stroke-width="1.5" fill="none"
        stroke-dasharray="${perim}" stroke-dashoffset="${perim}"
        style="animation:hexDraw 0.5s ease forwards">
      </polygon>
      <style>
        @keyframes hexDraw { to { stroke-dashoffset: 0; } }
      </style>
    </svg>`;
  },
};
