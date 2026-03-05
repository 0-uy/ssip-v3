/**
 * alerts.js — SSIP v2.1
 * Fixes sobre v2.0:
 *   [FIX 1] Constructor defensivo — no crashea si los elementos DOM no existen
 *            (permite usar AlertManager en multicam sin los IDs de monitor.html)
 *   [FIX 2] _captureSnapshot acepta videoElement por parámetro o lo busca dinámicamente
 *   [FIX 3] modalClose listener solo se registra si el elemento existe
 */

const SEVERITY_CONFIG = {
  low: {
    label:    'AVISO',
    color:    '#00c8ff',
    cssClass: 'severity-low',
    duration: 2000,
  },
  medium: {
    label:    'ALERTA',
    color:    '#ffaa00',
    cssClass: 'severity-med',
    duration: 3000,
  },
  high: {
    label:    '⚠ PELIGRO',
    color:    '#ff3a3a',
    cssClass: 'severity-high',
    duration: 4000,
  },
};

export class AlertManager {
  /**
   * @param {HTMLCanvasElement} snapshotCanvas - Canvas del que tomar screenshots
   * @param {string|null}       videoId        - ID del elemento <video> para el snapshot
   *                                             Si es null, se intenta 'videoElement' como fallback
   */
  constructor(snapshotCanvas, videoId = null) {
    this.snapshotCanvas = snapshotCanvas;
    this._videoId       = videoId; // [FIX 2] ID parametrizable

    this.events = [];

    // [FIX 1] Todos los getElementById protegidos con || null
    // Si alguno no existe (ej: multicam.html) queda null y los métodos lo chequean antes de usarlo
    this._alertOverlay  = document.getElementById('alertOverlay')  || null;
    this._alertText     = document.getElementById('alertText')      || null;
    this._eventsList    = document.getElementById('eventsList')     || null;
    this._statusBadge   = document.getElementById('systemStatus')   || null;
    this._metricTotal   = document.getElementById('metricTotal')    || null;
    this._metricToday   = document.getElementById('metricToday')    || null;
    this._metricActive  = document.getElementById('metricActive')   || null;

    this._modalBackdrop = document.getElementById('modalBackdrop')  || null;
    this._modalTitle    = document.getElementById('modalTitle')      || null;
    this._modalSnapshot = document.getElementById('modalSnapshot')  || null;
    this._modalMeta     = document.getElementById('modalMeta')      || null;

    // [FIX 3] Solo registrar listener si el botón existe
    const modalCloseBtn = document.getElementById('modalClose');
    if (modalCloseBtn && this._modalBackdrop) {
      modalCloseBtn.addEventListener('click', () => {
        this._modalBackdrop.classList.add('hidden');
      });
    }

    this._alertTimer   = null;
    this._activeAlerts = 0;
    this._today        = new Date().toDateString();
  }

  /* ══════════════════════════════════════════════════════
     Disparar alerta
  ══════════════════════════════════════════════════════ */
  trigger(eventType, severity = 'medium') {
    const cfg  = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.medium;
    const now  = new Date();
    const ts   = now.toLocaleTimeString('es-UY', { hour12: false });
    const date = now.toLocaleDateString('es-UY');

    const snapshot = this._captureSnapshot();

    const event = {
      id:        `evt-${Date.now()}-${Math.random().toString(36).slice(2,6)}`,
      type:      eventType,
      severity,
      timestamp: now.toISOString(),
      timeStr:   ts,
      dateStr:   date,
      snapshot,
    };

    this.events.unshift(event);
    if (this.events.length > 200) this.events = this.events.slice(0, 200);

    this._showVisualAlert(cfg, eventType);
    this._renderEvent(event, cfg);
    this._updateMetrics();
    this._setSystemStatus('alert');

    return event;
  }

  /* ── Overlay de alerta ───────────────────────────────── */
  _showVisualAlert(cfg, eventType) {
    if (!this._alertOverlay || !this._alertText) return; // [FIX 1]

    this._alertText.textContent = `${cfg.label}: ${eventType}`;
    this._alertText.style.background = cfg.color === '#ff3a3a'
      ? 'rgba(255,58,58,0.9)'
      : cfg.color === '#ffaa00'
        ? 'rgba(200,130,0,0.9)'
        : 'rgba(0,100,160,0.9)';

    this._alertOverlay.classList.remove('hidden');
    this._activeAlerts++;

    if (this._alertTimer) clearTimeout(this._alertTimer);
    this._alertTimer = setTimeout(() => {
      this._alertOverlay?.classList.add('hidden');
      this._activeAlerts = Math.max(0, this._activeAlerts - 1);
      this._updateMetrics();
      if (this._activeAlerts === 0) this._setSystemStatus('online');
    }, cfg.duration);
  }

  /* ── Renderizar item en lista ────────────────────────── */
  _renderEvent(event, cfg) {
    if (!this._eventsList) return; // [FIX 1]

    const emptyEl = this._eventsList.querySelector('.events-empty');
    if (emptyEl) emptyEl.remove();

    const item = document.createElement('div');
    item.className = `event-item ${cfg.cssClass}`;
    item.dataset.eventId = event.id;
    item.innerHTML = `
      <img class="event-thumb" src="${event.snapshot || ''}" alt="snap"/>
      <div class="event-info">
        <div class="event-type">${this._escapeHtml(event.type)}</div>
        <div class="event-time">${event.dateStr} ${event.timeStr}</div>
      </div>`;
    item.addEventListener('click', () => this._openModal(event));
    this._eventsList.insertBefore(item, this._eventsList.firstChild);

    const items = this._eventsList.querySelectorAll('.event-item');
    if (items.length > 50) items[items.length - 1].remove();
  }

  /* ── Modal de detalle ────────────────────────────────── */
  _openModal(event) {
    if (!this._modalBackdrop) return; // [FIX 1]
    if (this._modalTitle)    this._modalTitle.textContent = event.type;
    if (this._modalSnapshot) this._modalSnapshot.src = event.snapshot || '';
    if (this._modalMeta) {
      this._modalMeta.innerHTML = `
        ID: ${event.id}<br/>
        Fecha: ${event.dateStr} ${event.timeStr}<br/>
        Severidad: ${event.severity.toUpperCase()}<br/>
        Tipo: ${this._escapeHtml(event.type)}`;
    }
    this._modalBackdrop.classList.remove('hidden');
  }

  /* ── Snapshot: video + overlay combinados ────────────── */
  _captureSnapshot() {
    try {
      const tmp = document.createElement('canvas');
      tmp.width  = this.snapshotCanvas.width;
      tmp.height = this.snapshotCanvas.height;
      const ctx  = tmp.getContext('2d');

      // [FIX 2] Busca por _videoId si está definido, sino cae a 'videoElement'
      const videoId = this._videoId || 'videoElement';
      const video   = document.getElementById(videoId);
      if (video && video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, tmp.width, tmp.height);
      } else {
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, tmp.width, tmp.height);
      }
      ctx.drawImage(this.snapshotCanvas, 0, 0);
      return tmp.toDataURL('image/jpeg', 0.75);
    } catch(e) {
      console.warn('Error capturando snapshot:', e);
      return '';
    }
  }

  /* ── Métricas ────────────────────────────────────────── */
  _updateMetrics() {
    if (this._metricTotal) this._metricTotal.textContent = this.events.length;
    if (this._metricToday) {
      const todayStr   = new Date().toDateString();
      const todayCount = this.events.filter(e => new Date(e.timestamp).toDateString() === todayStr).length;
      this._metricToday.textContent = todayCount;
    }
    if (this._metricActive) this._metricActive.textContent = this._activeAlerts;
  }

  /* ── Estado sistema ──────────────────────────────────── */
  _setSystemStatus(state) {
    if (!this._statusBadge) return; // [FIX 1]
    this._statusBadge.className = `status-badge ${state}`;
    const label = this._statusBadge.querySelector('.status-label');
    if (label) {
      const labels = { offline:'OFFLINE', online:'EN LÍNEA', alert:'ALERTA', starting:'INICIANDO' };
      label.textContent = labels[state] || state.toUpperCase();
    }
  }

  setOnline()  { this._setSystemStatus('online'); }
  setOffline() { this._setSystemStatus('offline'); }

  /* ── Exportar CSV ────────────────────────────────────── */
  exportCSV() {
    if (!this.events.length) { alert('No hay eventos para exportar.'); return; }
    const headers = ['ID','Fecha','Hora','Tipo de Evento','Severidad'];
    const rows    = this.events.map(e => [
      e.id, e.dateStr, e.timeStr,
      `"${e.type.replace(/"/g,'""')}"`, e.severity,
    ]);
    const csv = [
      '# SSIP — Sistema de Supervisión Inteligente Preventiva',
      `# Exportado: ${new Date().toLocaleString('es-UY')}`,
      '', headers.join(','), ...rows.map(r => r.join(',')),
    ].join('\n');
    const a = document.createElement('a');
    a.href     = URL.createObjectURL(new Blob([csv],{type:'text/csv;charset=utf-8;'}));
    a.download = `ssip_eventos_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(a); a.click(); document.body.removeChild(a);
    URL.revokeObjectURL(a.href);
  }

  clearHistory() {
    this.events = [];
    if (this._eventsList)
      this._eventsList.innerHTML = '<div class="events-empty">Sin eventos registrados</div>';
    this._updateMetrics();
  }

  _escapeHtml(str) {
    return str.replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
  }
}