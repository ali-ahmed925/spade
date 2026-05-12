/* ═══════════════════════════════════════════════════════════════════════════
   SPADE Demo — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */

'use strict';

/* ── 1. State ─────────────────────────────────────────────────────────────── */
const S = {
  category:   null,
  imagePath:  null,
  imageB64:   null,
  defectType: null,
  running:    false,
};

/* ── 2. API helpers ───────────────────────────────────────────────────────── */
const api = {
  get:  url        => fetch(url).then(r => { if (!r.ok) throw new Error(r.status); return r.json(); }),
  post: (url, body) => fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body),
  }).then(async r => { if (!r.ok) throw new Error(await r.text()); return r.json(); }),
};

const sleep = ms => new Promise(r => setTimeout(r, ms));

/* ── 3. Boot sequence ─────────────────────────────────────────────────────── */
async function boot() {
  try {
    const [cats, cfg] = await Promise.all([
      api.get('/api/categories'),
      api.get('/api/config'),
    ]);
    buildSidebar(cfg.hyperparameters);
    buildCategoryGrid(cats);
    document.getElementById('modeChip').textContent = (cfg.mode || 'LOCALHOST').toUpperCase();
  } catch (e) {
    setStatus('error', 'Backend unreachable');
  }

  // Dismiss boot screen — landing screen is already visible beneath
  const bootEl = document.getElementById('boot');
  bootEl.classList.add('hiding');
  setTimeout(() => bootEl.remove(), 700);

  // Sidebar accent + weight bars (for when app becomes visible later)
  setTimeout(animateWeightBars, 500);
}

/* ── 4. Sidebar ───────────────────────────────────────────────────────────── */
function buildSidebar(hp) {
  const container = document.getElementById('cfgRows');
  const rows = [
    { key: 'Backbone',    content: `<span class="tech-chip">${hp.backbone}</span>` },
    { key: 'Queries (Q)', content: `<span class="cfg-val">${hp.n_queries}</span>` },
    { key: 'Patch N-Max', content: `<span class="cfg-val">${hp.n_max}</span>` },
    { key: 'Patch N-Min', content: `<span class="cfg-val">${hp.n_min}</span>` },
    { key: 'HPA',         content: `<span class="hpa-badge">ENABLED</span>` },
    { key: 'Scoring',     content: `<span class="cfg-val">${hp.scoring}</span>` },
    { key: 'Aggregation', content: `<span class="cfg-val">${hp.aggregation}</span>` },
    { key: 'Freq Stream', content: `<span class="cfg-val">${hp.frequency_stream ? 'ON' : 'OFF'}</span>` },
    {
      key: 'Weights α β γ',
      content: `<div class="weights-wrap">
        ${[['α', hp.alpha], ['β', hp.beta], ['γ', hp.gamma]].map(([sym, val]) => `
          <div class="weight-row">
            <span class="weight-sym">${sym}</span>
            <div class="weight-track"><div class="weight-fill" data-target="${val / (hp.alpha + hp.beta + hp.gamma)}" style="width:0%"></div></div>
            <span class="weight-num">${val}</span>
          </div>`).join('')}
      </div>`,
      wide: true,
    },
  ];

  container.innerHTML = rows.map(r => `
    <div class="cfg-row" style="${r.wide ? 'flex-direction:column;align-items:flex-start;gap:6px' : ''}">
      <span class="cfg-key">
        <span class="lock-svg" style="color:var(--muted)">${ICONS.lock}</span>
        <span>${r.key}</span>
        <span class="tooltip">Read-only — model parameters locked for evaluation</span>
      </span>
      ${r.content}
    </div>`).join('');
}

function animateWeightBars() {
  document.querySelectorAll('.weight-fill[data-target]').forEach((el, i) => {
    setTimeout(() => {
      el.style.width = (parseFloat(el.dataset.target) * 80) + '%';
    }, i * 120);
  });
}

/* ── 5. Landing — category grid ───────────────────────────────────────────── */
function buildCategoryGrid(cats) {
  const grid = document.getElementById('catGrid');
  grid.innerHTML = cats.map(c => `
    <div class="cat-card ${!c.has_checkpoint ? 'disabled' : ''}"
         data-cat="${c.name}"
         ${!c.has_checkpoint ? 'title="No checkpoint available"' : ''}>
      <div class="cat-card-icon">
        ${CAT_SVGS[c.name] || CAT_SVGS.default}
      </div>
      <div class="cat-card-name">${c.name.replace(/_/g, '\u00a0')}</div>
    </div>`).join('');

  grid.querySelectorAll('.cat-card:not(.disabled)').forEach(card => {
    card.addEventListener('click', () => onCardClick(card.dataset.cat, card));
  });
}

/* ── 6. Card click → landing exit → app enter ────────────────────────────── */
async function onCardClick(cat, cardEl) {
  if (S.running) return;

  // Start API fetch immediately (parallel with animation)
  const fetchPromise = Promise.all([
    api.get(`/api/slot-images/${cat}`).catch(() => []),
    api.post('/api/select-image', { category: cat }).catch(() => null),
  ]);

  // Card flash + fade other cards
  cardEl.classList.add('card-selected');
  document.getElementById('catGrid').classList.add('has-winner');

  // Wait for animation and data in parallel
  const [dataResults] = await Promise.all([
    fetchPromise,
    sleep(440),
  ]);
  const [slotUrls, selected] = dataResults;

  if (!selected) {
    document.getElementById('catGrid').classList.remove('has-winner');
    cardEl.classList.remove('card-selected');
    setStatus('error', 'Failed to load image');
    return;
  }

  // Page transition: landing slides up, app slides up from below
  const landing = document.getElementById('landingScreen');
  const app     = document.getElementById('app');

  landing.classList.add('landing-exit');

  // App enter from below (start at translateY(40px), no transition)
  app.style.transition = 'none';
  app.style.opacity    = '0';
  app.style.transform  = 'translateY(40px)';
  app.style.pointerEvents = '';

  // Trigger transition on next frame
  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      app.style.transition = 'transform 0.55s cubic-bezier(0.4,0,0.2,1), opacity 0.55s cubic-bezier(0.4,0,0.2,1)';
      app.style.opacity    = '1';
      app.style.transform  = 'translateY(0)';
    });
  });

  await sleep(580);
  landing.style.display = 'none';

  // Set state
  S.category   = cat;
  S.imagePath  = selected.image_path;
  S.imageB64   = selected.image_b64;
  S.defectType = selected.defect_type;

  // Update app UI
  const backBtn = document.getElementById('backBtn');
  backBtn?.classList.add('shown');
  document.getElementById('catLabel').textContent  = cat.toUpperCase();
  document.getElementById('imageMeta').textContent = selected.rel_path;
  setStatus('ready', 'Image selected');
  setTimeout(() => {
    document.getElementById('accentLine')?.classList.add('grown');
    animateWeightBars();
  }, 200);

  // Run slot machine then enable run button
  await runSlotMachine(slotUrls, selected);
  document.getElementById('runBtn').disabled = false;
  resetResults();
}

/* ── 7. Back to landing ───────────────────────────────────────────────────── */
function showLanding() {
  if (S.running) return;

  const landing = document.getElementById('landingScreen');
  const app     = document.getElementById('app');

  // Reset grid to unselected state
  document.getElementById('catGrid')?.classList.remove('has-winner');
  document.querySelectorAll('.cat-card').forEach(c => c.classList.remove('card-selected'));

  // App exits down
  app.style.transition    = 'transform 0.42s cubic-bezier(0.4,0,0.2,1), opacity 0.42s';
  app.style.opacity       = '0';
  app.style.transform     = 'translateY(40px)';
  app.style.pointerEvents = 'none';

  // Landing slides down in
  landing.style.display    = '';
  landing.style.transition = 'none';
  landing.style.transform  = 'translateY(-30px)';
  landing.style.opacity    = '0';
  landing.classList.remove('landing-exit');

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      landing.style.transition = 'transform 0.42s cubic-bezier(0.4,0,0.2,1), opacity 0.42s';
      landing.style.transform  = 'translateY(0)';
      landing.style.opacity    = '1';
    });
  });

  // Reset UI state
  document.getElementById('backBtn')?.classList.remove('shown');
  document.getElementById('accentLine')?.classList.remove('grown');
  document.getElementById('runBtn').disabled = true;
  S.category  = null;
  S.imagePath = null;
  S.imageB64  = null;
  S.running   = false;
  resetResults();
}

/* ── 8. Slot machine animation ────────────────────────────────────────────── */
async function runSlotMachine(slotUrls, selected) {
  const area = document.getElementById('imageArea');
  document.getElementById('idleState')?.remove();

  const N    = 16;
  const imgs = slotUrls.length > 0 ? slotUrls : [`data:image/png;base64,${selected.image_b64}`];

  // Preload images
  await Promise.all(imgs.map(src => new Promise(res => {
    const i = new Image(); i.onload = i.onerror = res; i.src = src;
  })));

  // Build grid
  const grid = document.createElement('div');
  grid.className = 'slot-grid';
  grid.style.transform  = 'perspective(900px) rotateX(6deg)';
  grid.style.transition = 'transform 2s cubic-bezier(0.4,0,0.2,1)';

  const cells = Array.from({length: N}, (_, i) => {
    const div = document.createElement('div');
    div.className = 'slot-cell';
    const img = document.createElement('img');
    img.src = imgs[i % imgs.length];
    img.draggable = false;
    div.appendChild(img);
    return div;
  });
  cells.forEach(c => grid.appendChild(c));
  area.innerHTML = '';
  area.appendChild(grid);

  setTimeout(() => { grid.style.transform = 'perspective(900px) rotateX(0deg)'; }, 200);

  // Spin
  const spinners = cells.map((cell) => {
    const interval = 70 + Math.random() * 60;
    let idx = Math.floor(Math.random() * imgs.length);
    return setInterval(() => {
      idx = (idx + 1) % imgs.length;
      const imgEl = cell.querySelector('img');
      imgEl.src = imgs[idx];
      imgEl.style.filter = `hue-rotate(${Math.random() * 180}deg) brightness(${0.6 + Math.random() * 0.3})`;
    }, interval);
  });

  await sleep(2400);
  spinners.forEach(clearInterval);

  // Outside-in freeze (winner = index 5)
  const WINNER = 5;
  const dist = i => {
    const r = Math.floor(i / 4), c = i % 4;
    const wr = Math.floor(WINNER / 4), wc = WINNER % 4;
    return Math.abs(r - wr) + Math.abs(c - wc);
  };
  const order = Array.from({length: N}, (_, i) => i)
    .filter(i => i !== WINNER)
    .sort((a, b) => dist(b) - dist(a));

  for (const idx of order) {
    const cell = cells[idx];
    cell.querySelector('img').style.filter = '';
    cell.classList.add('frozen');
    const flash = document.createElement('div');
    flash.style.cssText = 'position:absolute;inset:0;background:rgba(6,182,212,0.4);animation:fadeOut 0.3s forwards;pointer-events:none';
    cell.appendChild(flash);
    setTimeout(() => flash.remove(), 350);
    await sleep(90);
  }

  await sleep(350);

  // Winner reveal
  const winner = cells[WINNER];
  winner.querySelector('img').src    = `data:image/png;base64,${selected.image_b64}`;
  winner.querySelector('img').style.filter = '';
  winner.classList.remove('frozen');
  winner.classList.add('winner');

  // Hex reticle
  winner.insertAdjacentHTML('beforeend',
    `<div class="hex-reticle">${ICONS.hexReticle(Math.min(winner.offsetWidth, winner.offsetHeight) * 0.85)}</div>`);

  // Collapse losers
  await sleep(180);
  const winRect  = winner.getBoundingClientRect();
  const gridRect = grid.getBoundingClientRect();
  const wx = winRect.left  - gridRect.left + winRect.width  / 2;
  const wy = winRect.top   - gridRect.top  + winRect.height / 2;

  cells.forEach((cell, i) => {
    if (i === WINNER) return;
    const cr = cell.getBoundingClientRect();
    const cx = cr.left - gridRect.left + cr.width  / 2;
    const cy = cr.top  - gridRect.top  + cr.height / 2;
    const dx = (wx - cx) / (cr.width  / 2 + 4);
    const dy = (wy - cy) / (cr.height / 2 + 4);
    cell.classList.add('loser');
    cell.style.transform = `translate(${dx * 15}px, ${dy * 15}px) scale(0)`;
    cell.style.opacity   = '0';
  });

  await sleep(450);
  cells.forEach((cell, i) => { if (i !== WINNER) cell.style.display = 'none'; });
  winner.style.gridColumn = '1 / 5';
  winner.style.gridRow    = '1 / 5';
  winner.style.transition = 'all 0.5s cubic-bezier(0.34, 1.56, 0.64, 1)';
  winner.querySelector('.hex-reticle')?.remove();

  const lbl = document.createElement('div');
  lbl.className = 'result-img-label';
  lbl.textContent = selected.defect_type.replace(/_/g, ' ').toUpperCase();
  winner.appendChild(lbl);
}

/* ── 9. Run Analysis ──────────────────────────────────────────────────────── */

// Steps shown while SPADE inference runs (fires /api/analyze)
const INFER_STEPS = [
  { msg: 'Extracting ViT-G patch tokens',        pct: 8,  dwell: 1200 },
  { msg: 'Computing FFT frequency features',     pct: 16, dwell: 1100 },
  { msg: 'Initializing Q-Former bridge',         pct: 24, dwell: 1000 },
  { msg: 'Hybrid Patch Annealing — pass 1 / 5',  pct: 32, dwell: 900  },
  { msg: 'Hybrid Patch Annealing — pass 2 / 5',  pct: 40, dwell: 850  },
  { msg: 'Hybrid Patch Annealing — pass 3 / 5',  pct: 48, dwell: 800  },
  { msg: 'Hybrid Patch Annealing — pass 4 / 5',  pct: 56, dwell: 750  },
  { msg: 'Hybrid Patch Annealing — pass 5 / 5',  pct: 64, dwell: 700  },
  { msg: 'Computing dual Mahalanobis scores',    pct: 72, dwell: 950  },
  { msg: 'Fusing spatial and frequency streams', pct: 80, dwell: 850  },
  { msg: 'Generating anomaly heatmap',           pct: 87, dwell: 850  },
];

// Steps shown while LLM explanation runs (fires /api/explain); loops until done
const LLM_STEPS = [
  { msg: 'Querying vision-language model…',        dwell: 5000 },
  { msg: 'Analyzing defect morphology…',           dwell: 6000 },
  { msg: 'Decoding anomalous patch regions…',      dwell: 6500 },
  { msg: 'Synthesizing natural language report…',  dwell: 7000 },
  { msg: 'Extracting semantic fault descriptors…', dwell: 6000 },
  { msg: 'Cross-referencing defect patterns…',     dwell: 7500 },
  { msg: 'Finalizing defect description…',         dwell: 6000 },
];

let _msgAbort = false;

/** Show one status message on the inference overlay, with fade-in + underline draw. */
function showOverlayStep(step) {
  if (step.pct !== undefined) {
    const fill = document.getElementById('progressFill');
    fill.style.transition = 'width 0.4s var(--ease)';
    fill.style.width = step.pct + '%';
  }

  const area = document.getElementById('imageArea');
  let overlay = document.getElementById('inferenceOverlay');
  if (!overlay) {
    overlay = document.createElement('div');
    overlay.id        = 'inferenceOverlay';
    overlay.className = 'inference-overlay';
    area.appendChild(overlay);
  }

  const pctHtml = step.pct !== undefined
    ? `<span class="infer-pct">${step.pct}%</span>`
    : '';

  overlay.innerHTML = `
    <div class="infer-msg-wrap" id="inferMsgWrap">
      <span class="infer-msg">${step.msg}</span>
      <div class="infer-underline" id="inferUnderline"></div>
      ${pctHtml}
    </div>`;

  requestAnimationFrame(() => {
    requestAnimationFrame(() => {
      document.getElementById('inferMsgWrap')?.classList.add('in');
      setTimeout(() => {
        const ul = document.getElementById('inferUnderline');
        if (ul) ul.style.width = '100%';
      }, 160);
    });
  });
}

/**
 * Cycle through `steps` until `donePromise` resolves/rejects or `_msgAbort` is set.
 *
 * loop=false + waitAfterLast=true (default): show each step once, then wait silently
 *   on the last message until done fires.
 * loop=false + waitAfterLast=false: show each step once and return immediately — do
 *   NOT block waiting for done. Use this for Phase 1 so we always transition to
 *   Phase 2 even when inference is slower than all steps combined.
 * loop=true: after exhausting all steps, start over from step 0 indefinitely.
 */
async function cycleUntilDone(steps, donePromise, onStep, { loop = false, waitAfterLast = true } = {}) {
  const done = donePromise.then(() => '__done__').catch(() => '__done__');

  // Show each step once
  for (let i = 0; i < steps.length && !_msgAbort; i++) {
    onStep(steps[i]);
    const winner = await Promise.race([sleep(steps[i].dwell), done]);
    if (winner === '__done__' || _msgAbort) return;
  }

  if (_msgAbort) return;

  if (loop) {
    // Loop back through steps until done
    let i = 0;
    while (!_msgAbort) {
      onStep(steps[i % steps.length]);
      const winner = await Promise.race([sleep(steps[i % steps.length].dwell), done]);
      if (winner === '__done__' || _msgAbort) return;
      i++;
    }
  } else if (waitAfterLast) {
    // All steps shown — hold on last message until done
    await done;
  }
  // waitAfterLast=false: exit immediately, caller handles the wait
}

async function runAnalysis() {
  if (!S.imagePath || !S.category || S.running) return;
  S.running = true;
  _msgAbort = false;

  const btn  = document.getElementById('runBtn');
  const fill = document.getElementById('progressFill');
  btn.disabled = true;
  btn.classList.add('running');
  setStatus('loading', 'Running inference…');
  fill.style.transition = 'width 0.4s var(--ease)';
  fill.style.width = '0%';
  document.getElementById('terminalMsg').textContent = 'Running SPADE pipeline…';
  document.getElementById('termCursor').classList.add('active');
  fill.classList.add('active');

  // Fire both requests in parallel immediately
  const inferPromise  = api.post('/api/analyze', { image_path: S.imagePath, category: S.category });
  const explainPromise = api.post('/api/explain', { image_path: S.imagePath, category: S.category });
  // Prevent unhandled rejection if we abandon explain due to infer failure
  explainPromise.catch(() => {});

  // ── Phase 1: cycle SPADE messages, exit immediately after all steps ─────
  // waitAfterLast:false ensures we always transition to LLM messages even when
  // inference takes longer than the combined step dwell time (~10.4s).
  await cycleUntilDone(INFER_STEPS, inferPromise, showOverlayStep, { loop: false, waitAfterLast: false });

  // ── Phase 2: crawl progress + cycle LLM messages until BOTH settle ──────
  // p90 latency on this model is ~263s — use 420s crawl to match request timeout.
  // Use allSettled so LLM messages continue even if inference is still running.
  const bothSettled = Promise.allSettled([inferPromise, explainPromise]);
  fill.style.transition = 'width 420s linear';
  fill.style.width = '99%';

  await cycleUntilDone(LLM_STEPS, bothSettled, showOverlayStep, { loop: true });

  // ── Both promises have settled — check results ───────────────────────────
  let inferResult;
  try {
    inferResult = await inferPromise;
  } catch (e) {
    _msgAbort = true;
    document.getElementById('inferenceOverlay')?.remove();
    document.getElementById('terminalMsg').textContent = `Error: ${e.message}`;
    document.getElementById('termCursor').classList.remove('active');
    fill.classList.remove('active');
    setStatus('error', 'Inference failed');
    S.running = false;
    btn.disabled = false;
    btn.classList.remove('running');
    return;
  }

  let explainResult;
  try {
    explainResult = await explainPromise;
  } catch (e) {
    explainResult = { explanation: 'Language decoder output unavailable.' };
  }

  // ── Done ─────────────────────────────────────────────────────────────────
  _msgAbort = true;
  document.getElementById('inferenceOverlay')?.remove();
  fill.style.transition = 'width 0.35s var(--ease)';
  fill.style.width = '100%';
  document.getElementById('termCursor').classList.remove('active');
  fill.classList.remove('active');
  document.getElementById('terminalMsg').textContent = 'Analysis complete ✓';
  setStatus('ready', 'Model Ready');

  await sleep(300);
  await showResults({ ...inferResult, llm_explanation: explainResult.explanation });

  S.running = false;
  btn.disabled = false;
  btn.classList.remove('running');
}

/* ── 10. Results display (4-panel 2×2 grid) ───────────────────────────────── */
async function showResults(result) {
  const { anomaly_score, original_b64, heatmap_b64, overlay_b64,
          ground_truth_b64, verdict, llm_explanation } = result;

  const area = document.getElementById('imageArea');
  area.innerHTML = `
    <div class="result-grid">
      <div class="result-panel" id="rp-0">
        <img src="data:image/png;base64,${original_b64}" alt="Original"/>
        <div class="result-panel-label">Original Image</div>
      </div>
      <div class="result-panel" id="rp-1">
        ${ground_truth_b64
          ? `<img src="data:image/png;base64,${ground_truth_b64}" alt="Ground Truth"/>`
          : `<div class="gt-unavailable">
               <svg width="24" height="24" viewBox="0 0 24 24" fill="none" opacity="0.3">
                 <circle cx="12" cy="12" r="10" stroke="#64748B" stroke-width="1.5"/>
                 <path d="M15 9l-6 6M9 9l6 6" stroke="#64748B" stroke-width="1.5" stroke-linecap="round"/>
               </svg>
               <span>Ground Truth Unavailable</span>
             </div>`}
        <div class="result-panel-label">Ground Truth Mask</div>
      </div>
      <div class="result-panel" id="rp-2">
        <img src="data:image/png;base64,${heatmap_b64}" alt="Heatmap"/>
        <div class="result-panel-label">Anomaly Heatmap</div>
      </div>
      <div class="result-panel overlay-panel" id="rp-3">
        <img src="data:image/png;base64,${overlay_b64}" alt="Overlay"/>
        <div class="result-panel-label">Heatmap Overlay</div>
      </div>
    </div>`;

  // Staggered entrance
  for (let i = 0; i < 4; i++) {
    await sleep(60);
    document.getElementById(`rp-${i}`)?.classList.add('visible');
  }

  // Normalize score (log scale for wide dynamic range)
  const norm = Math.min(1, Math.max(0, Math.log1p(anomaly_score) / Math.log1p(1e6)));

  await sleep(80);
  animateGauge(norm, anomaly_score);

  // Verdict
  await sleep(200);
  const isAnom = verdict === 'ANOMALOUS';
  const vslot  = document.getElementById('verdictSlot');
  vslot.innerHTML = `
    <div class="verdict-badge ${isAnom ? 'anomalous' : 'normal'}">
      <span class="${isAnom ? 'warn-flicker' : ''}">${isAnom ? ICONS.warning : ICONS.check}</span>
      ${verdict}
    </div>`;
  await sleep(30);
  vslot.querySelector('.verdict-badge').classList.add('visible');

  // Confidence bar
  await sleep(150);
  document.getElementById('confVal').textContent   = (norm * 100).toFixed(1) + '%';
  document.getElementById('confFill').style.width  = (norm * 100) + '%';
  const glow = document.getElementById('confGlow');
  glow.style.left = `calc(${norm * 100}% - 6px)`;
  glow.classList.add('active');

  // Typewriter LLM explanation
  await sleep(280);
  typewrite(llm_explanation);
}

/* ── 11. Gauge animation ──────────────────────────────────────────────────── */
function animateGauge(norm, rawScore) {
  const tickG = document.getElementById('gaugeTicks');
  if (tickG && !tickG._drawn) {
    tickG._drawn = true;
    const CX = 100, CY = 120, R = 82;
    for (let i = 0; i <= 10; i++) {
      const angle = (135 + i * 27) * Math.PI / 180;
      const inner = i % 5 === 0 ? R - 8 : R - 5;
      const x1 = CX + R * Math.cos(angle), y1 = CY + R * Math.sin(angle);
      const x2 = CX + inner * Math.cos(angle), y2 = CY + inner * Math.sin(angle);
      const line = document.createElementNS('http://www.w3.org/2000/svg', 'line');
      line.setAttribute('x1', x1); line.setAttribute('y1', y1);
      line.setAttribute('x2', x2); line.setAttribute('y2', y2);
      line.setAttribute('stroke', '#1E2D42');
      line.setAttribute('stroke-width', '1.5');
      line.setAttribute('stroke-linecap', 'round');
      tickG.appendChild(line);
    }
  }

  const overlay = document.getElementById('gaugeOverlay');
  const needle  = document.getElementById('gaugeNeedle');
  const valEl   = document.getElementById('gaugeVal');   // HTML element now
  const maxArc  = 339;
  const maskLen = maxArc * (1 - norm);
  overlay.setAttribute('stroke-dasharray', `${maskLen} ${452 - maskLen}`);

  const angle = -135 + norm * 270;
  needle.style.transform = `rotate(${angle}deg)`;

  const color = norm < 0.35 ? '#10B981' : norm < 0.65 ? '#F59E0B' : '#EF4444';
  needle.querySelector('polygon').setAttribute('fill', color);

  // Animate counter — always show full decimal, no K/M abbreviation
  const start = performance.now();
  const dur   = 1200;
  function tick(now) {
    const t = Math.min((now - start) / dur, 1);
    const e = 1 - Math.pow(1 - t, 3);
    const v = rawScore * e;
    valEl.style.color   = color;
    valEl.textContent   = v.toFixed(2);
    if (t < 1) requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
}

/* ── 12. Typewriter ───────────────────────────────────────────────────────── */
async function typewrite(text) {
  const body = document.getElementById('terminalBody');
  body.innerHTML = '';

  const safeText = (text && typeof text === 'string') ? text.trim() : '';
  if (!safeText) {
    body.innerHTML = `<div class="term-placeholder"><span class="term-ph-text">Language decoder output unavailable.</span></div>`;
    return;
  }

  const out    = document.createElement('div');
  out.className = 'term-output';
  const cursor = document.createElement('span');
  cursor.className = 'term-block-cursor';
  out.appendChild(cursor);
  body.appendChild(out);

  for (const ch of safeText) {
    out.insertBefore(document.createTextNode(ch), cursor);
    await sleep(ch === '.' ? 35 : 14);
  }
  await sleep(2500);
  cursor.remove();
}

/* ── 13. Reset ────────────────────────────────────────────────────────────── */
function resetResults() {
  const overlay = document.getElementById('gaugeOverlay');
  const needle  = document.getElementById('gaugeNeedle');
  const valEl   = document.getElementById('gaugeVal');   // HTML element
  if (overlay) overlay.setAttribute('stroke-dasharray', '339 0');
  if (needle)  needle.style.transform = 'rotate(-135deg)';
  if (valEl)   { valEl.textContent = '—'; valEl.style.color = '#64748B'; }

  document.getElementById('verdictSlot').innerHTML = `
    <div class="verdict-placeholder">
      <svg width="20" height="20" viewBox="0 0 20 20" fill="none" opacity="0.3">
        <circle cx="10" cy="10" r="8" stroke="#64748B" stroke-width="1.5"/>
        <path d="M10 6v4M10 13h.01" stroke="#64748B" stroke-width="1.5" stroke-linecap="round"/>
      </svg>
      <span>Run analysis to see verdict</span>
    </div>`;

  document.getElementById('confFill').style.width  = '0%';
  document.getElementById('confVal').textContent   = '—';
  const glow = document.getElementById('confGlow');
  glow.style.left = '0%';
  glow.classList.remove('active');

  document.getElementById('terminalBody').innerHTML = `
    <div class="term-placeholder"><span class="term-ph-text">Awaiting inference output…</span></div>`;

  document.getElementById('progressFill').style.width  = '0%';
  document.getElementById('terminalMsg').textContent   = '';
  document.getElementById('termCursor').classList.remove('active');
}

/* ── 14. Status helpers ───────────────────────────────────────────────────── */
function setStatus(state, label) {
  const dot  = document.getElementById('pingDot');
  const r1   = document.getElementById('pingRing1');
  const r2   = document.getElementById('pingRing2');
  const text = document.getElementById('statusText');
  const sfx  = state === 'loading' ? ' amber' : state === 'error' ? ' red' : '';
  dot.className  = 'ping-dot'              + sfx;
  r1.className   = 'ping-ring ping-ring-1' + sfx;
  r2.className   = 'ping-ring ping-ring-2' + sfx;
  text.textContent = label;
}

/* ── 15. Init ─────────────────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
  const ri = document.getElementById('radarIcon');
  if (ri) ri.innerHTML = ICONS.radar;
});

window.addEventListener('load', boot);
