/**
 * tour.js — Product tour interactivo SSIP v3
 * Fixes: responsive, scroll-aware, viewport-safe, mobile-friendly
 */

const TOUR_KEY = 'ssip_tour_v3';

const STEPS = [
  {
    target:   '.source-tabs',
    title:    '📷 Elegí tu fuente de video',
    content:  '<strong>Webcam:</strong> cámara del equipo o USB externa.<br><strong>IP/WebRTC:</strong> cámara IP del local con go2rtc.<br><strong>Archivo:</strong> analizá un video grabado.',
    position: 'bottom',
  },
  {
    target:   '#btnDrawZone',
    title:    '📐 Dibujá tu zona crítica',
    content:  'Hacé clic aquí y luego <strong>clic a clic</strong> sobre el video para trazar la zona a monitorear. <strong>Doble clic</strong> o <strong>Enter</strong> para cerrar. Hasta 6 zonas con colores distintos.',
    position: 'auto',
  },
  {
    target:   '#btnEditZone',
    title:    '✏️ Editá la zona',
    content:  'Activá este modo para <strong>arrastrar los vértices</strong> de cualquier zona y ajustar su posición. Útil cuando la cámara se mueve un poco.',
    position: 'auto',
  },
  {
    target:   '#btnDetection',
    title:    '🎯 Iniciá el análisis IA',
    content:  'Activa la detección en tiempo real. La IA analiza:<br>· Manos dentro de la zona<br>· Permanencia prolongada<br>· Manos en bolsillos / ocultas<br>· Brazos cruzados (ocultamiento bajo ropa)',
    position: 'auto',
  },
  {
    target:   '#sliderMovement',
    title:    '⚙️ Ajustá la sensibilidad',
    content:  '<strong>Umbral movimiento:</strong> qué tan brusco debe ser el gesto.<br><strong>Tiempo en zona:</strong> segundos antes de alertar.<br><strong>Cooldown:</strong> pausa mínima entre alertas.',
    position: 'auto',
  },
  {
    target:   '#zonesCard',
    title:    '🗺️ Tus zonas activas',
    content:  'Acá ves todas las zonas con su color. Podés eliminar una individual con el botón <strong>✕</strong>. Si agregás muchas, la sección hace scroll automático.',
    position: 'auto',
  },
  {
    target:   '.events-card',
    title:    '📋 Historial de eventos',
    content:  'Cada alerta se guarda con <strong>foto</strong>, tipo y hora exacta. Se sincronizan en tu panel principal y se borran automáticamente después de 48 horas. Exportá con el botón CSV.',
    position: 'auto',
  },
];

const CSS = `
  .t-hl {
    position: fixed;
    z-index: 8900;
    border-radius: 8px;
    pointer-events: none;
    box-shadow: 0 0 0 9999px rgba(0,0,0,0.78);
    transition: top .32s ease, left .32s ease, width .32s ease, height .32s ease;
    outline: 2px solid rgba(0, 212, 255, 0.6);
    outline-offset: 2px;
  }

  .t-bub {
    position: fixed;
    z-index: 9000;
    background: #0a1118;
    border: 1.5px solid rgba(0, 212, 255, 0.5);
    border-radius: 14px;
    padding: 18px 18px 14px;
    width: min(290px, calc(100vw - 24px));
    box-shadow:
      0 0 30px rgba(0, 212, 255, 0.1),
      0 20px 50px rgba(0, 0, 0, 0.9);
    font-family: 'Barlow', sans-serif;
    animation: tBubIn .25s cubic-bezier(.4,0,.2,1);
    box-sizing: border-box;
  }

  /* Arrows */
  .t-bub::before {
    content: '';
    position: absolute;
    width: 10px;
    height: 10px;
    background: #0a1118;
    border: 1.5px solid rgba(0, 212, 255, 0.5);
    transform: rotate(45deg);
  }
  .t-bub.arr-right::before  { left: -6px; top: 22px; border-right: none; border-top: none; }
  .t-bub.arr-left::before   { right: -6px; top: 22px; border-left: none; border-bottom: none; }
  .t-bub.arr-bottom::before { top: -6px; left: 22px; border-bottom: none; border-right: none; }
  .t-bub.arr-top::before    { bottom: -6px; left: 22px; border-top: none; border-left: none; }
  .t-bub.arr-none::before   { display: none; }

  .t-bub .t-title {
    font-size: 13px;
    font-weight: 700;
    color: #00d4ff;
    margin-bottom: 8px;
    line-height: 1.3;
  }
  .t-bub .t-body {
    font-size: 12px;
    color: #7a9aaa;
    line-height: 1.75;
  }
  .t-bub .t-body strong { color: #b8d8e8; }

  .t-foot {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-top: 13px;
    padding-top: 11px;
    border-top: 1px solid #142030;
    gap: 8px;
  }
  .t-dots { display: flex; gap: 4px; flex-shrink: 0; }
  .t-dot {
    width: 5px; height: 5px;
    border-radius: 50%;
    background: #1a2535;
    transition: background .2s;
    flex-shrink: 0;
  }
  .t-dot.on  { background: #00d4ff; box-shadow: 0 0 5px #00d4ff; }
  .t-dot.was { background: rgba(0, 212, 255, 0.35); }

  .t-acts { display: flex; gap: 6px; flex-shrink: 0; }
  .t-btn {
    padding: 5px 12px;
    border-radius: 6px;
    font-size: 11px;
    font-weight: 700;
    cursor: pointer;
    font-family: 'Barlow', sans-serif;
    transition: all .18s;
    white-space: nowrap;
  }
  .t-skip {
    background: transparent;
    border: 1px solid #1a2535;
    color: #3a5468;
  }
  .t-skip:hover { color: #ff3d3d; border-color: #ff3d3d; }
  .t-next {
    background: rgba(0, 212, 255, 0.1);
    border: 1px solid #00d4ff;
    color: #00d4ff;
  }
  .t-next:hover { background: rgba(0, 212, 255, 0.2); }
  .t-done {
    background: rgba(0, 230, 118, 0.1);
    border: 1px solid #00e676;
    color: #00e676;
  }
  .t-done:hover { background: rgba(0, 230, 118, 0.2); }

  .t-prog {
    font-family: 'Share Tech Mono', monospace;
    font-size: 10px;
    color: #2e4558;
    flex-shrink: 0;
  }

  .t-relaunch {
    position: fixed;
    bottom: 20px;
    left: 20px;
    z-index: 8000;
    width: 32px;
    height: 32px;
    border-radius: 50%;
    background: rgba(0, 212, 255, 0.08);
    border: 1px solid rgba(0, 212, 255, 0.3);
    color: #00d4ff;
    font-size: 14px;
    font-weight: 700;
    cursor: pointer;
    display: flex;
    align-items: center;
    justify-content: center;
    opacity: 0.55;
    transition: all .2s;
  }
  .t-relaunch:hover { opacity: 1; background: rgba(0, 212, 255, 0.18); }

  @keyframes tBubIn {
    from { opacity: 0; transform: scale(.93) translateY(-6px); }
    to   { opacity: 1; transform: scale(1)  translateY(0); }
  }

  /* Mobile adjustments */
  @media (max-width: 600px) {
    .t-bub {
      width: calc(100vw - 16px) !important;
      left: 8px !important;
      top: auto !important;
      bottom: 12px !important;
      position: fixed !important;
    }
    .t-bub::before { display: none !important; }
    .t-hl { display: none !important; }
  }
`;

export function initTour(force = false) {
  if (!force && localStorage.getItem(TOUR_KEY)) {
    _addRelaunchButton();
    return;
  }

  // Inject CSS once
  if (!document.getElementById('t-style')) {
    const style = document.createElement('style');
    style.id = 't-style';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  let hl  = null;
  let bub = null;
  let currentIdx = 0;

  function clean() {
    hl?.remove();  hl  = null;
    bub?.remove(); bub = null;
  }

  function done() {
    clean();
    localStorage.setItem(TOUR_KEY, '1');
    _addRelaunchButton();
  }

  /**
   * Scroll element into view if needed, then show the bubble.
   */
  function show(idx) {
    clean();
    if (idx >= STEPS.length) { done(); return; }
    currentIdx = idx;

    const step = STEPS[idx];
    const el   = document.querySelector(step.target);
    if (!el) { show(idx + 1); return; }

    // Scroll into view if not visible
    const rect = el.getBoundingClientRect();
    const inView = rect.top >= 0 && rect.bottom <= window.innerHeight;
    if (!inView) {
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      // Wait for scroll to finish before rendering
      setTimeout(() => _render(idx, el), 420);
    } else {
      _render(idx, el);
    }
  }

  function _render(idx, el) {
    clean();
    const step = STEPS[idx];
    const r    = el.getBoundingClientRect();
    const pad  = 7;

    // Highlight
    hl = document.createElement('div');
    hl.className = 't-hl';
    Object.assign(hl.style, {
      top:    (r.top    - pad) + 'px',
      left:   (r.left   - pad) + 'px',
      width:  (r.width  + pad * 2) + 'px',
      height: (r.height + pad * 2) + 'px',
    });
    document.body.appendChild(hl);

    // Bubble
    bub = document.createElement('div');
    const last = idx === STEPS.length - 1;
    const dots = STEPS.map((_, i) =>
      `<div class="t-dot ${i < idx ? 'was' : ''} ${i === idx ? 'on' : ''}"></div>`
    ).join('');

    bub.innerHTML = `
      <div class="t-title">${step.title}</div>
      <div class="t-body">${step.content}</div>
      <div class="t-foot">
        <div style="display:flex;align-items:center;gap:8px;">
          <div class="t-dots">${dots}</div>
          <span class="t-prog">${idx + 1}/${STEPS.length}</span>
        </div>
        <div class="t-acts">
          ${!last ? `<button class="t-btn t-skip" id="tSkip">Saltar</button>` : ''}
          <button class="t-btn ${last ? 't-done' : 't-next'}" id="tNext">
            ${last ? 'Listo ✓' : 'Siguiente →'}
          </button>
        </div>
      </div>
    `;

    // Position bubble smartly
    const pos = _calcPosition(r, step.position);
    bub.className = `t-bub arr-${pos.arrow}`;
    Object.assign(bub.style, { top: pos.top + 'px', left: pos.left + 'px' });

    document.body.appendChild(bub);

    document.getElementById('tNext')?.addEventListener('click', () => show(idx + 1));
    document.getElementById('tSkip')?.addEventListener('click', done);
  }

  /**
   * Smart positioning: tries preferred side, falls back automatically.
   * Returns { top, left, arrow }
   */
  function _calcPosition(r, preferred) {
    const BW   = 300;  // bubble width (slightly more than CSS for safety)
    const BH   = 220;  // estimated bubble height
    const GAP  = 14;
    const vw   = window.innerWidth;
    const vh   = window.innerHeight;
    const SAFE = 8;

    const spaceRight  = vw - r.right  - GAP;
    const spaceLeft   = r.left        - GAP;
    const spaceBottom = vh - r.bottom - GAP;
    const spaceTop    = r.top         - GAP;

    // Determine actual side
    let side = preferred;
    if (side === 'auto' || side === 'right') {
      if (spaceRight >= BW)        side = 'right';
      else if (spaceLeft >= BW)    side = 'left';
      else if (spaceBottom >= BH)  side = 'bottom';
      else if (spaceTop >= BH)     side = 'top';
      else                          side = 'bottom'; // last resort
    } else if (side === 'bottom' && spaceBottom < BH && spaceTop >= BH) {
      side = 'top';
    } else if (side === 'left' && spaceLeft < BW && spaceRight >= BW) {
      side = 'right';
    }

    let top, left, arrow = side;

    if (side === 'right') {
      top  = r.top + r.height / 2 - 90;
      left = r.right + GAP;
    } else if (side === 'left') {
      top  = r.top + r.height / 2 - 90;
      left = r.left - BW - GAP;
    } else if (side === 'bottom') {
      top  = r.bottom + GAP;
      left = r.left;
    } else { // top
      top  = r.top - BH - GAP;
      left = r.left;
    }

    // Clamp to viewport
    top  = Math.max(SAFE, Math.min(top,  vh - BH - SAFE));
    left = Math.max(SAFE, Math.min(left, vw - BW - SAFE));

    // If clamping pushed bubble far from target, hide arrow
    if (side === 'right'  && left > r.right + GAP + 40) arrow = 'none';
    if (side === 'left'   && left + BW < r.left - 20)   arrow = 'none';
    if (side === 'bottom' && top > r.bottom + GAP + 40)  arrow = 'none';
    if (side === 'top'    && top + BH < r.top - 20)      arrow = 'none';

    return { top, left, arrow };
  }

  // Handle resize: re-render current step
  let resizeTimer;
  const onResize = () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => {
      const step = STEPS[currentIdx];
      const el   = step && document.querySelector(step.target);
      if (el && hl) _render(currentIdx, el); // only re-render if tour is active
    }, 150);
  };
  window.addEventListener('resize', onResize);

  // Start after short delay to let layout settle
  setTimeout(() => show(0), 400);
}

function _addRelaunchButton() {
  if (document.querySelector('.t-relaunch')) return;

  // Inject minimal CSS if not already there
  if (!document.getElementById('t-style')) {
    const style = document.createElement('style');
    style.id = 't-style';
    style.textContent = CSS;
    document.head.appendChild(style);
  }

  const btn = document.createElement('button');
  btn.className   = 't-relaunch';
  btn.title       = 'Ver tutorial de nuevo';
  btn.textContent = '?';
  btn.onclick = () => {
    btn.remove();
    localStorage.removeItem(TOUR_KEY);
    initTour(true);
  };
  document.body.appendChild(btn);
}