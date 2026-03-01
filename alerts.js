/**
 * alerts.js — Sistema de alertas y registro de eventos
 *
 * Responsabilidades:
 *  - Generar alertas visuales en pantalla
 *  - Registrar eventos con timestamp + snapshot
 *  - Mantener historial en memoria
 *  - Exportar a CSV
 */

// Severidad → configuración visual
const SEVERITY_CONFIG = {
  low: {
    label: 'AVISO',
    color: '#00c8ff',
    cssClass: 'severity-low',
    duration: 2000, // ms que se muestra la alerta
  },
  medium: {
    label: 'ALERTA',
    color: '#ffaa00',
    cssClass: 'severity-med',
    duration: 3000,
  },
  high: {
    label: '⚠ PELIGRO',
    color: '#ff3a3a',
    cssClass: 'severity-high',
    duration: 4000,
  },
};

export class AlertManager {
  /**
   * @param {HTMLCanvasElement} snapshotCanvas - Canvas del que tomar screenshots
   */
  constructor(snapshotCanvas) {
    this.snapshotCanvas = snapshotCanvas;

    // Historial de eventos en memoria
    this.events = [];

    // Referencia al overlay de alerta
    this._alertOverlay  = document.getElementById('alertOverlay');
    this._alertText     = document.getElementById('alertText');
    this._eventsList    = document.getElementById('eventsList');
    this._statusBadge   = document.getElementById('systemStatus');
    this._metricTotal   = document.getElementById('metricTotal');
    this._metricToday   = document.getElementById('metricToday');
    this._metricActive  = document.getElementById('metricActive');

    // Modal de detalle
    this._modalBackdrop = document.getElementById('modalBackdrop');
    this._modalTitle    = document.getElementById('modalTitle');
    this._modalSnapshot = document.getElementById('modalSnapshot');
    this._modalMeta     = document.getElementById('modalMeta');
    document.getElementById('modalClose').addEventListener('click', () => {
      this._modalBackdrop.classList.add('hidden');
    });

    // Timer para ocultar la alerta visual
    this._alertTimer = null;

    // Contador de alertas activas
    this._activeAlerts = 0;

    // Fecha de hoy para métricas
    this._today = new Date().toDateString();
  }

  /* ══════════════════════════════════════════════════════
     Disparar alerta
  ══════════════════════════════════════════════════════ */
  /**
   * @param {string} eventType  - Tipo de evento detectado
   * @param {string} severity   - 'low' | 'medium' | 'high'
   */
  trigger(eventType, severity = 'medium') {
    const cfg = SEVERITY_CONFIG[severity] || SEVERITY_CONFIG.medium;
    const now  = new Date();
    const ts   = now.toLocaleTimeString('es-UY', { hour12: false });
    const date = now.toLocaleDateString('es-UY');

    // Capturar snapshot del canvas actual
    const snapshot = this._captureSnapshot();

    // Crear registro
    const event = {
      id:        `evt-${Date.now()}-${Math.random().toString(36).slice(2,6)}`,
      type:      eventType,
      severity,
      timestamp: now.toISOString(),
      timeStr:   ts,
      dateStr:   date,
      snapshot,
    };

    this.events.unshift(event); // más reciente primero

    // Limitar historial en memoria (últimos 200 eventos)
    if (this.events.length > 200) {
      this.events = this.events.slice(0, 200);
    }

    // Mostrar alerta visual
    this._showVisualAlert(cfg, eventType);

    // Actualizar lista de eventos
    this._renderEvent(event, cfg);

    // Actualizar métricas
    this._updateMetrics();

    // Actualizar estado del sistema header
    this._setSystemStatus('alert');

    return event;
  }

  /* ── Mostrar overlay de alerta ─────────────────────────── */
  _showVisualAlert(cfg, eventType) {
    if (!this._alertOverlay) return;

    this._alertText.textContent = `${cfg.label}: ${eventType}`;
    this._alertText.style.background = cfg.color === '#ff3a3a'
      ? 'rgba(255,58,58,0.9)'
      : cfg.color === '#ffaa00'
        ? 'rgba(200,130,0,0.9)'
        : 'rgba(0,100,160,0.9)';

    this._alertOverlay.classList.remove('hidden');
    this._activeAlerts++;

    // Limpiar timer previo
    if (this._alertTimer) clearTimeout(this._alertTimer);

    this._alertTimer = setTimeout(() => {
      this._alertOverlay.classList.add('hidden');
      this._activeAlerts = Math.max(0, this._activeAlerts - 1);
      this._updateMetrics();

      // Restaurar estado si no hay más alertas
      if (this._activeAlerts === 0) {
        this._setSystemStatus('online');
      }
    }, cfg.duration);
  }

  /* ── Renderizar item en la lista ──────────────────────── */
  _renderEvent(event, cfg) {
    // Eliminar "sin eventos"
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
      </div>
    `;

    // Click → abrir modal con detalle
    item.addEventListener('click', () => this._openModal(event));

    // Insertar al inicio
    this._eventsList.insertBefore(item, this._eventsList.firstChild);

    // Limitar DOM a 50 items visibles
    const items = this._eventsList.querySelectorAll('.event-item');
    if (items.length > 50) {
      items[items.length - 1].remove();
    }
  }

  /* ── Modal de detalle de evento ───────────────────────── */
  _openModal(event) {
    this._modalTitle.textContent = event.type;
    this._modalSnapshot.src = event.snapshot || '';
    this._modalMeta.innerHTML = `
      ID: ${event.id}<br/>
      Fecha: ${event.dateStr} ${event.timeStr}<br/>
      Severidad: ${event.severity.toUpperCase()}<br/>
      Tipo: ${this._escapeHtml(event.type)}
    `;
    this._modalBackdrop.classList.remove('hidden');
  }

  /* ── Capturar snapshot: video real + overlay combinados ── */
  _captureSnapshot() {
    try {
      // Crear canvas temporal del mismo tamaño
      const tmp = document.createElement('canvas');
      tmp.width  = this.snapshotCanvas.width;
      tmp.height = this.snapshotCanvas.height;
      const ctx = tmp.getContext('2d');

      // 1. Dibujar el video real debajo
      const video = document.getElementById('videoElement');
      if (video && video.readyState >= 2) {
        ctx.drawImage(video, 0, 0, tmp.width, tmp.height);
      } else {
        // Fondo negro si no hay video
        ctx.fillStyle = '#000';
        ctx.fillRect(0, 0, tmp.width, tmp.height);
      }

      // 2. Dibujar el overlay de detección encima
      ctx.drawImage(this.snapshotCanvas, 0, 0);

      return tmp.toDataURL('image/jpeg', 0.75);
    } catch (e) {
      console.warn('Error capturando snapshot:', e);
      return '';
    }
  }

  /* ── Actualizar contadores ────────────────────────────── */
  _updateMetrics() {
    if (this._metricTotal) {
      this._metricTotal.textContent = this.events.length;
    }
    if (this._metricToday) {
      const todayStr = new Date().toDateString();
      const todayCount = this.events.filter(e =>
        new Date(e.timestamp).toDateString() === todayStr
      ).length;
      this._metricToday.textContent = todayCount;
    }
    if (this._metricActive) {
      this._metricActive.textContent = this._activeAlerts;
    }
  }

  /* ── Estado del sistema en header ─────────────────────── */
  _setSystemStatus(state) {
    if (!this._statusBadge) return;
    this._statusBadge.className = `status-badge ${state}`;
    const label = this._statusBadge.querySelector('.status-label');
    if (label) {
      const labels = {
        offline:  'OFFLINE',
        online:   'EN LÍNEA',
        alert:    'ALERTA',
        starting: 'INICIANDO',
      };
      label.textContent = labels[state] || state.toUpperCase();
    }
  }

  setOnline()  { this._setSystemStatus('online'); }
  setOffline() { this._setSystemStatus('offline'); }

  /* ══════════════════════════════════════════════════════
     Exportar a CSV
  ══════════════════════════════════════════════════════ */
  exportCSV() {
    if (this.events.length === 0) {
      alert('No hay eventos para exportar.');
      return;
    }

    const headers = ['ID', 'Fecha', 'Hora', 'Tipo de Evento', 'Severidad'];
    const rows = this.events.map(e => [
      e.id,
      e.dateStr,
      e.timeStr,
      `"${e.type.replace(/"/g, '""')}"`,
      e.severity,
    ]);

    const csvContent = [
      '# SSIP — Sistema de Supervisión Inteligente Preventiva',
      `# Exportado: ${new Date().toLocaleString('es-UY')}`,
      '',
      headers.join(','),
      ...rows.map(r => r.join(',')),
    ].join('\n');

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url  = URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href     = url;
    link.download = `ssip_eventos_${new Date().toISOString().slice(0,10)}.csv`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  }

  /* ── Utilidades ────────────────────────────────────────── */
  _escapeHtml(str) {
    return str
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  clearHistory() {
    this.events = [];
    this._eventsList.innerHTML = '<div class="events-empty">Sin eventos registrados</div>';
    this._updateMetrics();
  }
}