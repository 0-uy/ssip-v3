/**
 * multicam-app.js — SSIP v3.0
 * ─────────────────────────────────────────────────────────────────────────
 * Script principal de multicam.html, separado para claridad.
 *
 * Fixes sobre v2.x:
 *   [FIX 1] CameraSource en cada celda — loadedmetadata garantizado
 *   [FIX 2] Canvas redimensionado DESPUÉS de que el video tiene dimensiones
 *   [FIX 3] Streams liberados al navegar a otra página (pagehide)
 *   [FIX 4] DeviceSelector con preferencia persistida por slot
 *   [FIX 5] trigger binding limpio — sin double-wrap frágil
 *   [FIX 6] Soporte: Webcam USB/integrada, IP/WebRTC, Archivo en cada slot
 *   [FIX 7] Panel de config no cierra si el user interactúa con el select
 */

import { onAuth, getEmpresa, getPlanEmpresa, guardarEvento } from './firebase-config.js';
import { ZoneManager }     from './zones.js';
import { DetectionEngine } from './detection.js';
import { AlertManager }    from './alerts.js';
import { CameraSource, DeviceSelector, savePref } from './camera-manager.js';
import { listProfiles } from './store-profiles.js';

// ── Reloj ─────────────────────────────────────────────────────────────────
setInterval(() => {
  const el = document.getElementById('hTime');
  if (el) el.textContent = new Date().toLocaleTimeString('es-UY', { hour12: false });
}, 1000);

// ── Zone modal ────────────────────────────────────────────────────────────
function openZoneModal() {
  return new Promise(resolve => {
    const back   = document.getElementById('zoneModalBack');
    const input  = document.getElementById('zmInput');
    const ok     = document.getElementById('zmOk');
    const cancel = document.getElementById('zmCancel');

    input.value = '';
    back.classList.add('open');
    setTimeout(() => input.focus(), 80);

    document.querySelectorAll('.zm-sug').forEach(b => {
      b.onclick = () => { input.value = b.dataset.val; };
    });

    const finish = (val) => {
      back.classList.remove('open');
      ok.removeEventListener('click', onOk);
      cancel.removeEventListener('click', onCancel);
      input.removeEventListener('keydown', onKey);
      resolve(val);
    };
    const onOk     = ()  => finish(input.value.trim() || null);
    const onCancel = ()  => finish(null);
    const onKey    = (e) => {
      if (e.key === 'Enter')  finish(input.value.trim() || null);
      if (e.key === 'Escape') finish(null);
    };

    ok.addEventListener('click',     onOk);
    cancel.addEventListener('click', onCancel);
    input.addEventListener('keydown', onKey);
    back.addEventListener('click', e => { if (e.target === back) finish(null); }, { once: true });
  });
}

// ── Toast ─────────────────────────────────────────────────────────────────
function toast(msg, type = 'info', ms = 3500) {
  const cols = { info: '#00c8ff', warn: '#ffaa00', ok: '#00e676', err: '#ff3a3a' };
  const n = document.createElement('div');
  n.style.cssText = `position:fixed;top:66px;right:18px;z-index:9990;
    background:#0a1320;border:1px solid ${cols[type] || cols.info};
    border-radius:10px;padding:12px 18px;max-width:300px;
    font-family:'Barlow',sans-serif;font-size:13px;color:#c8dde8;
    box-shadow:0 8px 32px rgba(0,0,0,.6);animation:_tIn .22s ease;line-height:1.5;`;
  n.textContent = msg;
  document.body.appendChild(n);
  setTimeout(() => { n.style.opacity = '0'; n.style.transition = 'opacity .4s'; }, ms);
  setTimeout(() => n.remove(), ms + 450);
}

// ── Status bar del sidebar (Estado del Sistema) ───────────────────────────
let _sysStatusTimer = null;
function setSysStatus(msg, type = 'loading', autoDismissMs = 0) {
  const bar  = document.getElementById('sysStatus');
  const text = document.getElementById('sysStatusMsg');
  if (!bar || !text) return;
  clearTimeout(_sysStatusTimer);
  bar.className  = `sys-status ${type}`;
  text.textContent = msg;
  if (autoDismissMs > 0) {
    _sysStatusTimer = setTimeout(() => {
      bar.style.opacity = '0';
      setTimeout(() => { bar.classList.add('hidden'); bar.style.opacity = ''; }, 400);
    }, autoDismissMs);
  }
}
function hideSysStatus() {
  const bar = document.getElementById('sysStatus');
  if (!bar) return;
  clearTimeout(_sysStatusTimer);
  bar.style.opacity = '0';
  setTimeout(() => { bar.classList.add('hidden'); bar.style.opacity = ''; }, 400);
}

// ── Auth ──────────────────────────────────────────────────────────────────
onAuth(async (user) => {
  if (!user) { window.location.href = 'login.html'; return; }

  const [plan, empresa] = await Promise.all([
    getPlanEmpresa(user.uid),
    getEmpresa(user.uid),
  ]);

  document.getElementById('hEmp').textContent = empresa?.nombre || user.email;

  const fbEl   = document.getElementById('fbStatus');
  const fbSide = document.getElementById('fbStatusSide');
  const setFb  = (txt, cls) => {
    [fbEl, fbSide].forEach(el => { if (el) { el.textContent = txt; el.className = `fb-status ${cls}`; } });
  };
  setFb('Firebase: conectado ✓', 'ok');

  if (!plan.multiview) {
    document.getElementById('upgradeWall').classList.remove('hidden');
    return;
  }

  initMultiCam(user.uid, plan, setFb);
});

// ════════════════════════════════════════════════════════════════════════════
function initMultiCam(empresaId, plan, setFb) {
  const NUM  = plan.camaras; // 1–3
  const grid = document.getElementById('camsGrid');
  grid.innerHTML = '';

  // ── Construir celdas HTML ────────────────────────────────────────────────
  for (let i = 0; i < NUM; i++) {
    grid.insertAdjacentHTML('beforeend', `
      <div class="cell" id="cell${i}">
        <div class="cell-hdr">
          <span class="cell-num">CAM ${i + 1}</span>
          <span class="cell-name" id="name${i}">Sin señal</span>
          <span class="cell-badge off" id="badge${i}">OFFLINE</span>
          <div class="cell-cdot" id="cdot${i}"></div>
          <button class="cell-fs" id="fsBtn${i}" title="Pantalla completa">
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5">
              <path d="M1 4V1h3M8 1h3v3M11 8v3H8M4 11H1V8"/>
            </svg>
          </button>
          <button class="cell-collapse" id="colBtn${i}" title="Minimizar cámara">
            <svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.8">
              <path d="M2 6h8M5 3l-3 3 3 3"/>
            </svg>
          </button>
        </div>
        <div class="cell-wrap" id="wrap${i}">
          <video id="vid${i}" class="cell-video" autoplay playsinline muted style="display:none;"></video>
          <canvas id="cvs${i}" class="cell-canvas"></canvas>
          <div class="cell-nosig" id="nosig${i}">
            <svg viewBox="0 0 48 48" fill="none" stroke="currentColor" stroke-width="1.2">
              <rect x="4" y="8" width="40" height="28" rx="3"/>
              <path d="M16 36v4M32 36v4M12 40h24"/>
              <line x1="14" y1="18" x2="34" y2="18" stroke-dasharray="4 3"/>
            </svg>
            <span>Sin señal de video</span>
            <small>Abrí ⚙ para conectar esta cámara</small>
          </div>
          <div class="cell-alert" id="ovl${i}">
            <div class="cell-alert-border"></div>
            <span class="cell-alert-txt" id="atxt${i}">⚠ ALERTA</span>
          </div>
        </div>
        <div class="cell-panel" id="panel${i}">
          <div class="panel-inner">
            <div class="p-row">
              <span class="p-lbl">Fuente</span>
              <button class="p-tab on"  data-cam="${i}" data-src="webcam">📷 Webcam</button>
              <button class="p-tab"     data-cam="${i}" data-src="webrtc">🌐 IP/RTC</button>
              <button class="p-tab"     data-cam="${i}" data-src="file">📁 Archivo</button>
              <input type="file" id="file${i}" accept="video/*" style="display:none"/>
            </div>
            <div class="p-row" id="devRow${i}">
              <span class="p-lbl">Cámara</span>
              <select class="p-sel" id="sel${i}"><option value="">Detectando…</option></select>
            </div>
            <div class="p-row" id="rtcRow${i}" style="display:none;">
              <span class="p-lbl">Protocolo</span>
              <select class="p-sel" id="rtcProto${i}" style="max-width:120px;">
                <option value="auto">Auto</option>
                <option value="webrtc">WebRTC</option>
                <option value="whep">WHEP</option>
                <option value="mjpeg">MJPEG</option>
                <option value="hls">HLS</option>
              </select>
            </div>
            <div class="p-row" id="rtcUrlRow${i}" style="display:none;">
              <span class="p-lbl">URL</span>
              <input class="p-url" id="rtcUrl${i}" type="text" placeholder="http://192.168.1.x:8083/api/webrtc?src=cam${i + 1}"/>
              <button class="p-btn" id="rtcBtn${i}">Conectar</button>
            </div>
            <!-- NUEVA FILA: IP y credenciales básicas -->
              <div class="p-row" id="ipBasicRow${i}" style="display:none;">
                <span class="p-lbl" style="width:35px;">IP</span>
                <input class="p-url" id="ipAddr${i}" type="text" 
                      placeholder="192.168.1.100" 
                      style="flex:1.2; min-width:100px;">
                <input class="p-url" id="ipUser${i}" type="text" 
                      placeholder="admin" value="admin" 
                      style="flex:0.8; min-width:70px;">
                <input class="p-url" id="ipPass${i}" type="password" 
                      placeholder="contraseña" 
                      style="flex:0.8; min-width:70px;">
              </div>
            <div class="p-divider"></div>
            <div class="p-slider-row">
              <span class="p-lbl">Movim.</span>
              <input type="range" class="p-slider" id="sldMov${i}" min="10" max="100" value="50"/>
              <span class="p-sval" id="sldMovVal${i}">50</span>
            </div>
            <div class="p-slider-row">
              <span class="p-lbl">Zona</span>
              <input type="range" class="p-slider" id="sldDwell${i}" min="1" max="10" value="3"/>
              <span class="p-sval" id="sldDwellVal${i}">3s</span>
            </div>
            <div class="p-slider-row">
              <span class="p-lbl">Cooldown</span>
              <input type="range" class="p-slider" id="sldCool${i}" min="2" max="30" value="8"/>
              <span class="p-sval" id="sldCoolVal${i}">8s</span>
            </div>
          </div>
        </div>
        <div class="cell-foot" id="foot${i}">
          <button class="fb" id="zoneBtn${i}" disabled>+ Zona</button>
          <button class="fb" id="editBtn${i}" disabled>✏ Editar</button>
          <button class="fb" id="detectBtn${i}" disabled>▶ Analizar</button>
          <button class="fb" id="stopBtn${i}" disabled title="Detener cámara" style="color:var(--danger,#ff3a3a);border-color:rgba(255,58,58,.25);">⏹</button>
          <span class="cell-fps" id="fps${i}">—</span>
          <button class="fb-cfg" id="cfgBtn${i}" title="Configurar fuente">⚙</button>
        </div>
      </div>`);
  }

  // ── Colapso de celdas (minimizar/expandir) ───────────────────────
  // Click en el boton ⇔ de una celda la minimiza a franja vertical.
  // Click en la franja la vuelve a expandir.
  // Siempre queda al menos 1 camara expandida.
  (function setupCollapse() {
    function countCollapsed() {
      return document.querySelectorAll('#camsGrid .cell.collapsed').length;
    }
    function toggleCollapse(i) {
      if (NUM === 1) return; // con 1 sola camara no tiene sentido
      const cell = document.getElementById('cell' + i);
      if (!cell) return;
      const willCollapse = !cell.classList.contains('collapsed');
      if (willCollapse && countCollapsed() >= NUM - 1) return; // dejar al menos 1 expandida
      cell.classList.toggle('collapsed', willCollapse);
      // flex-mode solo en desktop (>=901px); en mobile el colapso es por height
      const isDesktop = window.matchMedia('(min-width: 901px)').matches;
      grid.classList.toggle('flex-mode', isDesktop && countCollapsed() > 0);
    }

    for (let i = 0; i < NUM; i++) {
      // Boton de colapso en el header
      const btn = document.getElementById('colBtn' + i);
      if (btn) {
        btn.addEventListener('click', (e) => { e.stopPropagation(); toggleCollapse(i); });
      }
      // Click en la celda colapsada para expandirla
      const cell = document.getElementById('cell' + i);
      if (cell) {
        cell.addEventListener('click', (e) => {
          if (!cell.classList.contains('collapsed')) return;
          if (e.target.closest('.cell-collapse')) return;
          toggleCollapse(i);
        });
      }
    }
  })();

    // ── Objetos de estado por cámara ─────────────────────────────────────────
  const cams = Array.from({ length: NUM }, (_, i) => ({
    idx:       i,
    video:     document.getElementById(`vid${i}`),
    canvas:    document.getElementById(`cvs${i}`),
    wrap:      document.getElementById(`wrap${i}`),
    nosig:     document.getElementById(`nosig${i}`),
    cell:      document.getElementById(`cell${i}`),
    badge:     document.getElementById(`badge${i}`),
    nameEl:    document.getElementById(`name${i}`),
    alertOvl:  document.getElementById(`ovl${i}`),
    alertTxt:  document.getElementById(`atxt${i}`),
    fpsEl:     document.getElementById(`fps${i}`),
    detectBtn: document.getElementById(`detectBtn${i}`),
    zoneBtn:   document.getElementById(`zoneBtn${i}`),
    editBtn:   document.getElementById(`editBtn${i}`),
    stopBtn:   document.getElementById(`stopBtn${i}`),
    source:    null,   // CameraSource
    zm:        null,
    de:        null,
    am:        null,
    detecting: false,
    animId:    null,
    alertTimer:null,
    _ovlTimer: null,
    srcType:   'webcam',
  }));

  let sessionEvents = [];
  let activeAlerts  = 0;

  // ── [FIX 5] Motores: AlertManager con trigger limpio ─────────────────────
  cams.forEach(cam => {
    cam.zm = new ZoneManager(cam.canvas);
    cam.am = new AlertManager(cam.canvas);

    // Override limpio del trigger — sin double-wrap
    cam.am.trigger = function(type, sev) {
      let snapshot = '';
      try {
        const tmp = document.createElement('canvas');
        tmp.width  = cam.canvas.width  || 640;
        tmp.height = cam.canvas.height || 480;
        const tc   = tmp.getContext('2d');
        if (cam.video && cam.video.readyState >= 2 && !cam.video.paused)
          tc.drawImage(cam.video, 0, 0, tmp.width, tmp.height);
        else {
          tc.fillStyle = '#0a1018';
          tc.fillRect(0, 0, tmp.width, tmp.height);
        }
        tc.drawImage(cam.canvas, 0, 0);
        snapshot = tmp.toDataURL('image/jpeg', 0.75);
      } catch {}

      const event = {
        id:        `mc-${Date.now()}-${cam.idx}`,
        type, severity: sev,
        timestamp: new Date().toISOString(),
        timeStr:   new Date().toLocaleTimeString('es-UY', { hour12: false }),
        snapshot,
      };
      this.events = this.events || [];
      this.events.unshift(event);
      if (this.events.length > 200) this.events = this.events.slice(0, 200);

      // Overlay visual en la celda
      const dur = sev === 'high' ? 4500 : sev === 'medium' ? 3000 : 2000;
      cam.alertOvl.classList.add('on');
      cam.alertTxt.textContent = `⚠ ${type}`;
      clearTimeout(cam._ovlTimer);
      cam._ovlTimer = setTimeout(() => cam.alertOvl.classList.remove('on'), dur);

      // Push al historial + Firebase
      pushEvent(cam.idx, type, sev, snapshot);

      return event;
    };

    cam.de = new DetectionEngine(cam.canvas, cam.zm, cam.am, {
      movementThreshold: 50, dwellTime: 3, cooldown: 6, storeType: 'generico',
    });

    // [FIX 1] CameraSource con callbacks correctos
    cam.source = new CameraSource(cam.video, cam.canvas, cam.wrap, {
      onReady: (label) => {
        cam.nosig.style.display = 'none';
        cam.nameEl.textContent  = label.length > 26 ? label.slice(0, 24) + '…' : label;
        cam.badge.textContent   = 'LIVE';
        cam.badge.className     = 'cell-badge live';
        cam.detectBtn.disabled  = false;
        cam.zoneBtn.disabled    = false;
        if (cam.stopBtn) cam.stopBtn.disabled = false;
        if (cam.zm?.zones.length > 0) cam.editBtn.disabled = false;
        const t = cam.source?.type;
        if (t === 'webrtc' || t === 'mjpeg' || t === 'hls') cam.srcType = 'webrtc';
        else if (t === 'file') cam.srcType = 'file';
        else if (t === 'webcam') cam.srcType = 'webcam';
        setSysStatus(`CAM ${cam.idx + 1} activa`, 'ok', 2500);
        updateStats();
      },
      onError: (msg) => {
        if (cam.source?.isReady) {
          toast(`CAM ${cam.idx + 1}: ${msg}`, 'warn', 5000);
          setSysStatus(`CAM ${cam.idx + 1}: aviso`, 'warn', 4000);
        } else {
          setOffline(cam, msg);
          toast(`CAM ${cam.idx + 1}: ${msg}`, 'err', 5000);
          setSysStatus(`CAM ${cam.idx + 1}: error de conexión`, 'err', 5000);
        }
      },
      onStopped: () => {
        // Solo llama setOffline si no fue intencional (lo maneja stopCam)
      },
    });
  });

  // ── Fullscreen por cámara ────────────────────────────────────────────────
  cams.forEach(cam => {
    const btn = document.getElementById(`fsBtn${cam.idx}`);
    if (!btn) return;
    btn.addEventListener('click', () => {
      if (!document.fullscreenElement) {
        (cam.cell.requestFullscreen?.() || cam.cell.webkitRequestFullscreen?.());
      } else {
        (document.exitFullscreen?.() || document.webkitExitFullscreen?.());
      }
    });
  });

  document.addEventListener('fullscreenchange', () => {
    cams.forEach(cam => {
      const btn = document.getElementById(`fsBtn${cam.idx}`);
      if (!btn) return;
      const isFs = document.fullscreenElement === cam.cell;
      btn.innerHTML = isFs
        ? `<svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M4 1H1v3M8 1h3v3M11 8v3H8M4 11H1V8"/></svg>`
        : `<svg viewBox="0 0 12 12" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 4V1h3M8 1h3v3M11 8v3H8M4 11H1V8"/></svg>`;
      btn.title = isFs ? 'Salir de pantalla completa' : 'Pantalla completa';
    });
  });

  // ── Panel de configuración ───────────────────────────────────────────────
  function closeAllPanels() {
    document.querySelectorAll('.cell-panel').forEach(p => p.classList.remove('open'));
    document.querySelectorAll('.fb-cfg').forEach(b => b.classList.remove('open'));
  }

  cams.forEach(cam => {
    const btn   = document.getElementById(`cfgBtn${cam.idx}`);
    const panel = document.getElementById(`panel${cam.idx}`);
    btn.addEventListener('click', e => {
      e.stopPropagation();
      const was = panel.classList.contains('open');
      closeAllPanels();
      if (!was) {
        panel.classList.add('open');
        btn.classList.add('open');
        loadDevicesForCam(cam.idx);
      }
    });
  });

  // [FIX 7] El click fuera cierra panel, pero NO si fue en el select o inputs del panel
  document.addEventListener('click', e => {
    if (e.target.closest('.cell-panel') || e.target.closest('.fb-cfg')) return;
    closeAllPanels();
  });

  // ── Cargar lista de cámaras en el select de una celda ───────────────────
  async function loadDevicesForCam(idx) {
    const selEl = document.getElementById(`sel${idx}`);
    if (!selEl) return;

    const devSel = new DeviceSelector(selEl, async (deviceId) => {
      if (deviceId === '__disconnected__') {
        stopCam(cams[idx]);
        toast(`CAM ${idx+1}: camara desconectada. Elige otra.`, 'warn', 5000);
        return;
      }
      if (cams[idx].srcType === 'webcam') {
        await cams[idx].source.startWebcam(deviceId, `cam${idx}`);
      }
    });
    const preferred = await devSel.populate(`cam${idx}`);
    return preferred;
  }

  // ── Tabs de fuente ───────────────────────────────────────────────────────
  document.querySelectorAll('.p-tab').forEach(btn => {
    btn.addEventListener('click', async () => {
      const idx = +btn.dataset.cam;
      const src = btn.dataset.src;
      const cam = cams[idx];
      const prevSrcType = cam.srcType;
      cam.srcType = src;

      document.querySelectorAll(`.p-tab[data-cam="${idx}"]`).forEach(b => b.classList.remove('on'));
      btn.classList.add('on');

      document.getElementById(`devRow${idx}`).style.display    = src === 'webcam' ? '' : 'none';
      document.getElementById(`rtcRow${idx}`).style.display     = src === 'webrtc' ? '' : 'none';
      document.getElementById(`rtcUrlRow${idx}`).style.display  = src === 'webrtc' ? '' : 'none';

      if (src === 'webcam') {
        // Detener cámara IP o archivo si estaba activa
        if (cam.source?.isReady && prevSrcType !== 'webcam') stopCam(cam);
        document.getElementById(`ipBasicRow${idx}`).style.display = 'none';
        const deviceId = await loadDevicesForCam(idx);
        if (deviceId) await cam.source.startWebcam(deviceId, `cam${idx}`);

      } else if (src === 'webrtc') {
        // Detener webcam si estaba activa; conservar cámara IP ya conectada
        if (cam.source?.isReady && prevSrcType === 'webcam') stopCam(cam);
        document.getElementById(`ipBasicRow${idx}`).style.display = 'flex';
        document.getElementById(`rtcRow${idx}`).style.display     = '';
        document.getElementById(`rtcUrlRow${idx}`).style.display  = '';

      } else if (src === 'file') {
        // Detener fuente activa si no era archivo
        if (cam.source?.isReady && prevSrcType !== 'file') stopCam(cam);
        document.getElementById(`ipBasicRow${idx}`).style.display = 'none';
        document.getElementById(`file${idx}`).click();
      }
    }); // end click
  }); // end p-tab forEach

  // ── WebRTC y Archivo por celda ───────────────────────────────────────────
  cams.forEach(cam => {
    const i = cam.idx;

    // Placeholder dinámico según protocolo seleccionado
    const protoSel = document.getElementById(`rtcProto${i}`);
    const urlInput = document.getElementById(`rtcUrl${i}`);
    const proxyPlaceholders = {
      auto:   `http://192.168.1.x:8083/api/webrtc?src=cam${i + 1}`,
      webrtc: `http://192.168.1.x:8083/api/webrtc?src=cam${i + 1}`,
      whep:   `http://192.168.1.x:8889/cam${i + 1}/whep`,
      mjpeg:  `http://usuario:clave@192.168.1.x/video.cgi`,
      hls:    `http://192.168.1.x/stream.m3u8`,
    };
    protoSel?.addEventListener('change', () => {
      if (urlInput) urlInput.placeholder = proxyPlaceholders[protoSel.value] || proxyPlaceholders.auto;
    });

    document.getElementById(`rtcBtn${i}`).onclick = () => {
      const url      = document.getElementById(`rtcUrl${i}`).value.trim();
      const protocol = document.getElementById(`rtcProto${i}`)?.value || 'auto';
      if (!url) { toast(`CAM ${i + 1}: Ingresá la URL de la cámara`, 'warn'); return; }
      setSysStatus(`CAM ${i + 1}: conectando…`, 'loading');
      closeAllPanels();
      cam.source.startIP(url, protocol);
    };

    // Enter en URL también conecta
    document.getElementById(`rtcUrl${i}`)?.addEventListener('keydown', e => {
      if (e.key === 'Enter') document.getElementById(`rtcBtn${i}`).click();
    });

    document.getElementById(`file${i}`).onchange = e => {
      const f = e.target.files[0]; if (!f) return;
      stopCam(cam);
      cam.source.startFile(f);
      closeAllPanels();
    };
  });

  // ── Helpers de estado de cámara ──────────────────────────────────────────
  function setOffline(cam, msg = 'Sin señal') {
    if (cam.detecting) toggleDetect(cam);
    cam.nameEl.textContent  = msg;
    cam.badge.textContent   = 'OFFLINE';
    cam.badge.className     = 'cell-badge off';
    cam.nosig.style.display = '';
    cam.video.style.display = 'none';
    cam.detectBtn.disabled  = true;
    cam.zoneBtn.disabled    = true;
    if (cam.editBtn) cam.editBtn.disabled = true;
    if (cam.stopBtn) cam.stopBtn.disabled = true;
    updateStats();
  }

  function stopCam(cam) {
    if (cam.detecting) toggleDetect(cam);
    cam.source.stop();
    setOffline(cam);
  }

  // ── Zona y detección por cámara ──────────────────────────────────────────
  cams.forEach(cam => {
    cam.detectBtn.addEventListener('click', () => toggleDetect(cam));

    // Botón detener — para cámara activa sin importar la fuente
    cam.stopBtn?.addEventListener('click', () => {
      if (!cam.source?.isReady) return;
      stopCam(cam);
      toast(`CAM ${cam.idx + 1} detenida`, 'info', 2000);
    });

    cam.zoneBtn.addEventListener('click', async () => {
      if (!cam.source.isReady) return;
      const nombre = await openZoneModal();
      if (!nombre) return;
      cam.zm.enableDraw(nombre);
      cam.zoneBtn.classList.add('on');
      cam.zoneBtn.textContent = '✓ Dibujando…';
      toast(`Dibujá "${nombre}" en CAM ${cam.idx + 1}. Doble clic para cerrar.`, 'info', 5000);
      cam.zm.onZoneChange(() => {
        cam.zoneBtn.classList.remove('on');
        cam.zoneBtn.textContent = '+ Zona';
        if (cam.zm.zones.length > 0) cam.editBtn.disabled = false;
      });
    });

    cam.editBtn.addEventListener('click', () => {
      if (!cam.source.isReady || !cam.zm.zones.length) return;
      const editing = cam.editBtn.classList.contains('on');
      if (!editing) {
        cam.zm.enableEdit();
        cam.editBtn.classList.add('on');
        cam.editBtn.textContent = '✓ Listo';
      } else {
        cam.zm.disableEdit();
        cam.editBtn.classList.remove('on');
        cam.editBtn.textContent = '✏ Editar';
      }
    });

    // Sliders de sensibilidad por cámara
    [['Mov', 'movementThreshold'], ['Dwell', 'dwellTime'], ['Cool', 'cooldown']].forEach(([name, cfgKey]) => {
      const sld = document.getElementById(`sld${name}${cam.idx}`);
      const val = document.getElementById(`sld${name}Val${cam.idx}`);
      if (!sld) return;
      sld.addEventListener('input', () => {
        const v = +sld.value;
        val.textContent = name === 'Mov' ? v : v + 's';
        cam.de.updateConfig({ [cfgKey]: v });
      });
    });
  });

  function toggleDetect(cam) {
    if (!cam.source.isReady) return;
    cam.detecting = !cam.detecting;
    if (cam.detecting) {
      cam.de.start();
      startLoop(cam);
      cam.detectBtn.classList.add('on');
      cam.detectBtn.textContent = '■ Detener';
      cam.badge.textContent = 'ANALIZ.';
      cam.badge.className   = 'cell-badge live';
      setSysStatus(`CAM ${cam.idx + 1}: análisis IA activo`, 'ok', 3000);
    } else {
      cam.de.stop();
      if (cam.animId) { cancelAnimationFrame(cam.animId); cam.animId = null; }
      cam.detectBtn.classList.remove('on');
      cam.detectBtn.textContent = '▶ Analizar';
      cam.badge.textContent = 'LIVE';
      setSysStatus(`CAM ${cam.idx + 1}: análisis detenido`, 'warn', 2000);
    }
  }

  function startLoop(cam) {
    let last = 0;
    const loop = async ts => {
      if (!cam.detecting) return;
      if (ts - last >= 66) { // ~15 FPS
        last = ts;
        // [FIX 2] Solo procesar si el video tiene frame real
        if (cam.video.readyState >= 2 && cam.video.videoWidth > 0 && !cam.video.paused) {
          await cam.de.processFrame(cam.video);
        }
        if (cam.fpsEl) cam.fpsEl.textContent = cam.de.currentFPS + ' FPS';
      }
      cam.animId = requestAnimationFrame(loop);
    };
    cam.animId = requestAnimationFrame(loop);
  }

  // ── Eventos ───────────────────────────────────────────────────────────────
  function pushEvent(camIdx, tipo, sev, snapshot) {
    const now = new Date();
    const ev  = {
      id:      `mc-${Date.now()}-${camIdx}`,
      camIdx, tipo, sev, snapshot,
      timeStr: now.toLocaleTimeString('es-UY', { hour12: false }),
    };
    sessionEvents.unshift(ev);
    if (sessionEvents.length > 200) sessionEvents = sessionEvents.slice(0, 200);
    renderEvent(ev);
    updateStats();
    flashAlert(cams[camIdx], tipo, sev);
    guardarEvento(empresaId, { tipo, severidad: sev, snapshot, camaraIdx: camIdx })
      .catch(err => { if (err) setFb?.('Firebase: error al guardar', 'err'); });
  }

  function flashAlert(cam, tipo, sev) {
    const dur = { low: 2000, medium: 3000, high: 4500 }[sev] || 3000;
    cam.cell.classList.add('alerting');
    cam.alertOvl.classList.add('on');
    cam.alertTxt.textContent = `⚠ ${tipo}`;
    cam.badge.textContent = 'ALERTA';
    cam.badge.className   = 'cell-badge alert';
    activeAlerts++;
    updateAlertBadge();
    clearTimeout(cam.alertTimer);
    cam.alertTimer = setTimeout(() => {
      cam.cell.classList.remove('alerting');
      cam.alertOvl.classList.remove('on');
      if (cam.source.isReady) {
        cam.badge.textContent = cam.detecting ? 'ANALIZ.' : 'LIVE';
        cam.badge.className   = 'cell-badge live';
      }
      activeAlerts = Math.max(0, activeAlerts - 1);
      updateAlertBadge();
    }, dur);
  }

  function updateAlertBadge() {
    const el = document.getElementById('hAlerts');
    if (!el) return;
    document.getElementById('hAlertsNum').textContent =
      activeAlerts > 0 ? `${activeAlerts} alerta${activeAlerts > 1 ? 's' : ''}` : '0 alertas';
    el.className = 'hdr-alerts' + (activeAlerts > 0 ? ' on' : '');
  }

  function renderEvent(ev) {
    const list  = document.getElementById('evList');
    const empty = list.querySelector('.ev-empty');
    if (empty) empty.remove();
    const cls  = ev.sev === 'high' ? 'h' : ev.sev === 'medium' ? 'm' : 'l';
    const item = document.createElement('div');
    item.className = `ev-item ${cls}`;
    item.innerHTML = `
      <img class="ev-thumb" src="${ev.snapshot || ''}" alt=""/>
      <div class="ev-info">
        <div class="ev-type">${ev.tipo}</div>
        <div class="ev-meta">
          <span>${ev.timeStr}</span>
          <span class="ev-cam">CAM ${ev.camIdx + 1}</span>
        </div>
      </div>`;
    item.addEventListener('click', () => openModal(ev));
    list.insertBefore(item, list.firstChild);
    const items = list.querySelectorAll('.ev-item');
    if (items.length > 60) items[items.length - 1].remove();
  }

  function updateStats() {
    const active = cams.filter(c => c.source?.isReady).length;
    document.getElementById('statTotal').textContent   = sessionEvents.length;
    document.getElementById('statCams').textContent    = active + '/3';
    document.getElementById('statAlertsN').textContent = activeAlerts;
    document.getElementById('stateEvents').textContent = sessionEvents.length;
    document.getElementById('stateAlerts').textContent = activeAlerts;
    document.getElementById('stateCams').textContent   = active + ' / ' + NUM;
  }

  // ── Modal de detalle ──────────────────────────────────────────────────────
  function openModal(ev) {
    document.getElementById('modalTitle').textContent = ev.tipo;
    const img = document.getElementById('modalImg');
    img.src = ev.snapshot || '';
    img.style.display = ev.snapshot ? 'block' : 'none';
    document.getElementById('modalMeta').innerHTML =
      `ID: ${ev.id}<br>Hora: ${ev.timeStr}<br>Severidad: ${ev.sev.toUpperCase()}<br>Cámara: ${ev.camIdx + 1}`;
    document.getElementById('modalBack').classList.remove('hidden');
  }
  document.getElementById('modalClose').onclick = () =>
    document.getElementById('modalBack').classList.add('hidden');
  document.getElementById('modalBack').onclick = e => {
    if (e.target.id === 'modalBack') document.getElementById('modalBack').classList.add('hidden');
  };

  // ── CSV ───────────────────────────────────────────────────────────────────
  document.getElementById('btnCSV').addEventListener('click', () => {
    if (!sessionEvents.length) { toast('Sin eventos para exportar.', 'warn'); return; }
    const csv = [
      '# SSIP Multi-Cam Export',
      `# ${new Date().toLocaleString('es-UY')}`,
      'ID,Hora,Camara,Tipo,Severidad',
      ...sessionEvents.map(e =>
        [e.id, e.timeStr, `Cam ${e.camIdx + 1}`, `"${e.tipo}"`, e.sev].join(',')
      ),
    ].join('\n');
    const a = document.createElement('a');
    a.href = URL.createObjectURL(new Blob([csv], { type: 'text/csv' }));
    a.download = `ssip_multicam_${new Date().toISOString().slice(0, 10)}.csv`;
    a.click();
  });

  // ── Init: cargar modelos IA ───────────────────────────────────────────────
  (async () => {
    const loader     = document.getElementById('modelLoader');
    const poseBar    = document.getElementById('mlPoseBar');
    const poseStatus = document.getElementById('mlPoseStatus');
    const objBar     = document.getElementById('mlObjBar');
    const objStatus  = document.getElementById('mlObjStatus');
    const statusTxt  = document.getElementById('mlStatusTxt');

    // Animación de progreso mientras carga
    let posePct = 0;
    setSysStatus('Cargando modelo IA…', 'loading');
    const poseAnim = setInterval(() => {
      posePct = Math.min(posePct + Math.random() * 4, 85);
      poseBar.style.width = posePct + '%';
    }, 200);

    // Inicializar motor de detección de cam 0 (carga el modelo compartido)
    let poseOk = false;
    try {
      await cams[0].de.init();
      poseOk = true;
    } catch (e) {
      statusTxt.textContent = '⚠ Error cargando modelo: ' + e.message;
    }
    // El resto de cámaras comparten la sesión ONNX ya cargada
    for (let i = 1; i < cams.length; i++) {
      try { await cams[i].de.init(); } catch {}
    }

    clearInterval(poseAnim);
    poseBar.style.width = '100%';
    if (poseOk) {
      poseBar.classList.add('done');
      poseStatus.textContent = '✓ listo';
      poseStatus.style.color = 'var(--green)';
      setSysStatus('Modelo IA listo ✓', 'ok', 3000);
    } else {
      poseBar.classList.add('err');
      poseStatus.textContent = '✗ error';
      poseStatus.style.color = 'var(--danger)';
      setSysStatus('Error cargando modelo IA', 'err');
    }

    setTimeout(() => {
      objBar.style.width = '100%';
      objBar.classList.add(poseOk ? 'done' : 'err');
      objStatus.textContent = poseOk ? '✓ listo' : 'no disponible';
      objStatus.style.color = poseOk ? 'var(--green)' : 'var(--warn)';
      statusTxt.textContent = poseOk
        ? '✓ Sistema listo — conectando cámara 1…'
        : '⚠ Verificá los archivos .onnx en el servidor';
    }, 400);

    const stateModelEl = document.getElementById('stateModel');
    if (stateModelEl) {
      stateModelEl.textContent = poseOk ? 'Listo ✓' : 'Error';
      stateModelEl.className   = `state-val ${poseOk ? 'ok' : 'err'}`;
    }

    // ── [v5.0] SELECTOR TIPO DE LOCAL ────────────────────────────────────
    const storeSelect = document.getElementById('storeTypeSelect');
    if (storeSelect) {
      // Poblar opciones desde store-profiles.js
      storeSelect.innerHTML = '';
      for (const p of listProfiles()) {
        const opt = document.createElement('option');
        opt.value       = p.key;
        opt.textContent = `${p.icon} ${p.name}`;
        storeSelect.appendChild(opt);
      }
      storeSelect.value = 'generico';

      storeSelect.addEventListener('change', () => {
        const type    = storeSelect.value;
        const label   = storeSelect.options[storeSelect.selectedIndex].text;

        // Aplicar perfil a TODOS los engines activos
        cams.forEach(cam => {
          const profile = cam.de.setStoreType(type);
          // Sincronizar slider dwellTime de cada celda con el valor del perfil
          if (profile?.dwellTime) {
            const sld = document.getElementById(`sldDwell${cam.idx}`);
            const val = document.getElementById(`sldDwellVal${cam.idx}`);
            if (sld) sld.value = profile.dwellTime;
            if (val) val.textContent = profile.dwellTime + 's';
            cam.de.updateConfig({ dwellTime: profile.dwellTime });
          }
        });

        // Mostrar perfil en Estado del Sistema
        if (stateModelEl) {
          stateModelEl.textContent = `Listo ✓ · ${label}`;
          stateModelEl.className   = 'state-val ok';
        }

        toast(`Perfil aplicado a todas las cámaras: ${label}`, 'ok', 2500);
      });
    }

    const hideDelay = poseOk ? 1000 : 3500;
    setTimeout(() => {
      if (loader) {
        loader.style.transition = 'opacity .5s';
        loader.style.opacity    = '0';
        setTimeout(() => loader.classList.add('hidden'), 500);
      }
    }, hideDelay);

    if (!poseOk) return;

    // Auto-arrancar cam 0 con la cámara preferida
    const deviceId0 = await loadDevicesForCam(0);
    if (deviceId0) {
      await cams[0].source.startWebcam(deviceId0, 'cam0');
    }

    // Tour
    setTimeout(() => window._startMcTour?.(), 1200);
  })();

} // end initMultiCam - last