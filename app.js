/**
 * app.js — SSIP v4.0 — monitor.html
 * Maneja: selector de camara, zonas, deteccion IA, navegacion entre paginas.
 */

import { CameraSource, DeviceSelector, savePref } from './camera-manager.js';
import { listProfiles } from './store-profiles.js';

export async function initApp({ ZoneManager, DetectionEngine, AlertManager }) {

  // DOM refs
  const video          = document.getElementById('videoElement');
  const canvas         = document.getElementById('overlayCanvas');
  const noSignal       = document.getElementById('noSignal');
  const canvasWrapper  = document.getElementById('canvasWrapper');
  const btnDrawZone    = document.getElementById('btnDrawZone');
  const btnClearZone   = document.getElementById('btnClearZone');
  const btnDetection   = document.getElementById('btnDetection');
  const btnStop        = document.getElementById('btnStop');
  const btnPause       = document.getElementById('btnPause');
  const btnCamToggle   = document.getElementById('btnCamToggle');
  const fpsDisplay     = document.getElementById('fpsDisplay');
  const zoneHint       = document.getElementById('zoneHint');
  const stateVideo     = document.getElementById('stateVideo');
  const stateDetect    = document.getElementById('stateDetection');
  const stateZone      = document.getElementById('stateZone');
  const stateModel     = document.getElementById('stateModel');
  const sliderMovement = document.getElementById('sliderMovement');
  const sliderDwell    = document.getElementById('sliderDwell');
  const sliderCooldown = document.getElementById('sliderCooldown');
  const valMovement    = document.getElementById('valMovement');
  const valDwell       = document.getElementById('valDwell');
  const valCooldown    = document.getElementById('valCooldown');

  // Estado
  let videoReady      = false;
  let detectionActive = false;
  let drawingZone     = false;
  let animFrameId     = null;
  let camPaused       = false;
  let srcType         = 'webcam';
  let _ssTimer        = null;

  // Motores
  const zoneManager  = new ZoneManager(canvas);
  const alertManager = new AlertManager(canvas);
  const detection    = new DetectionEngine(canvas, zoneManager, alertManager, {
    movementThreshold: 50, dwellTime: 3, cooldown: 6, storeType: 'generico',
  });
  detection.onDetection = (_, sev) => { if (sev !== 'info') _flashStatus(sev); };

  // CameraSource
  const camSrc = new CameraSource(video, canvas, canvasWrapper, {
    onReady(label) {
      noSignal?.classList.add('hidden');
      videoReady = true; camPaused = false;
      if (camSrc.type === 'webrtc' || camSrc.type === 'mjpeg' || camSrc.type === 'hls') srcType = 'webrtc';
      stateVideo.textContent = label;
      stateVideo.className   = 'state-val ok';
      alertManager.setOnline?.();
      _syncCamBtn(true);
      _updateControls();
      _setSysStatus('Señal de video activa', 'ok', 2500);
    },
    onError(msg) {
      if (videoReady) { _toast(msg, 'warn', 6000); _setSysStatus('Aviso de video', 'warn', 4000); return; }
      noSignal?.classList.remove('hidden');
      const p   = noSignal?.querySelector('p');
      const sub = noSignal?.querySelector('.no-signal-sub');
      if (p)   p.textContent   = msg;
      if (sub) sub.textContent = 'Verifica la fuente de video';
      stateVideo.textContent = 'Sin señal';
      stateVideo.className   = 'state-val err';
      _toast(msg, 'err', 6000);
      _setSysStatus('Error de conexión', 'err', 6000);
    },
    onStopped() {
      if (detectionActive) _stopDetection();
      videoReady = false; camPaused = false;
      noSignal?.classList.remove('hidden');
      stateVideo.textContent = '—';
      stateVideo.className   = 'state-val';
      alertManager.setOffline?.();
      _syncCamBtn(false);
      _updateControls();
      _setSysStatus('Cámara detenida', 'warn', 2000);
    },
  });

   window.camSrc = camSrc; 

  // ── BOTÓN PARAR (mata la fuente completamente) ────────────────────────────
  btnStop?.addEventListener('click', () => {
    if (!camSrc.isReady) return;
    if (detectionActive) _stopDetection();
    camSrc.stop();
    _toast('Cámara detenida', 'info', 2000);
  });

  // ── BOTÓN PAUSAR/REANUDAR ─────────────────────────────────────────────────
  btnPause?.addEventListener('click', () => {
    if (!videoReady) return;
    camPaused = !camPaused;
    camPaused ? video.pause() : video.play().catch(() => {});
    _syncPauseBtn(camPaused);
    _syncCamBtn(!camPaused);
  });
   
  // Reloj
  setInterval(() => {
    const el = document.getElementById('headerTime');
    if (el) el.textContent = new Date().toLocaleTimeString('es-UY', { hour12: false });
  }, 1000);

  // Canvas resize
  new ResizeObserver(() => {
    const r = canvasWrapper.getBoundingClientRect();
    if (r.width > 0 && r.height > 0) { canvas.width = r.width; canvas.height = r.height; }
  }).observe(canvasWrapper);

  // ── TABS DE FUENTE ────────────────────────────────────────────────────────
  // NOTA: el manejo de clases active y visibilidad del panel webrtcConfig
  // lo hace monitor.html. app.js solo gestiona el arranque/parada de la cámara.
  document.querySelectorAll('.source-tab').forEach(tab => {
    tab.addEventListener('click', async () => {
      const newSrc  = tab.dataset.source;
      const prevSrc = srcType;
      srcType = newSrc;

      // Al cambiar fuente: detener detección, resetear pausa
      if (newSrc !== prevSrc) {
        if (detectionActive) _stopDetection();
        if (camPaused) {
          camPaused = false;
          _syncPauseBtn(false);
          _syncCamBtn(true);
        }
        _setSysStatus('Cambiando fuente…', 'loading');
      }

      if (newSrc === 'webcam') {
        if (prevSrc !== 'webcam') { camSrc.stop(); await _launchWebcam(); }
      } else if (newSrc === 'webrtc' || newSrc === 'ip') {
        // Detener webcam al pasar a IP; conservar IP ya conectada
        if (prevSrc === 'webcam') camSrc.stop();
      } else if (newSrc === 'file') {
        if (prevSrc === 'webcam') camSrc.stop();
      }
    });
  });

  
  // setupIpPanel es manejado íntegramente por monitor.html
  // (evita listeners duplicados en ipConnectBtn / ipDiscoverBtn)

  
  
  
  // ── WEBCAM ────────────────────────────────────────────────────────────────
  async function _launchWebcam() {
    // Construir selector si no existe
    let selEl = document.getElementById('camSelector');
    if (!selEl) {
      const wrap = document.createElement('div');
      wrap.id = 'camSelectorWrap';
      wrap.style.cssText = 'display:flex;align-items:center;gap:8px;margin-top:6px;';
      wrap.innerHTML = `
        <label style="font-size:10px;letter-spacing:1.5px;color:#5a7a90;text-transform:uppercase;white-space:nowrap;">Camara</label>
        <select id="camSelector" style="flex:1;padding:5px 10px;background:#0c1118;border:1px solid #1a2535;
          border-radius:5px;color:#c8dde8;font-size:12px;font-family:'Share Tech Mono',monospace;
          cursor:pointer;outline:none;"></select>`;
      const bar = document.querySelector('.source-bar') || document.querySelector('.video-controls');
      bar?.appendChild(wrap);
      selEl = document.getElementById('camSelector');
    }

    const ds = new DeviceSelector(selEl, async (deviceId) => {
      if (deviceId === '__disconnected__') {
        camSrc.stop();
        _toast('La camara seleccionada se desconecto. Elige otra.', 'warn', 5000);
        return;
      }
      if (srcType === 'webcam') await camSrc.startWebcam(deviceId, 'monitor');
    });

    const preferred = await ds.populate('monitor');
    if (preferred) await camSrc.startWebcam(preferred, 'monitor');
  }

  // ── IP CAMERA ─────────────────────────────────────────────────────────────
  (function buildIPPanel() {
    const cfg = document.getElementById('ipConfig');
    if (!cfg) return;

    // Rutas por marca para auto-construir la URL desde IP + credenciales
    const BRAND_PATHS = {
      hikvision: { mjpeg: '/Streaming/Channels/101/httppreview', hls: '/Streaming/Channels/101/httpFlv' },
      dahua:     { mjpeg: '/cgi-bin/mjpg/video.cgi?channel=0&subtype=1' },
      axis:      { mjpeg: '/axis-cgi/mjpg/video.cgi' },
      tplink:    { webrtc: '/api/webrtc?src=main' },
      generic:   { mjpeg: '/video.cgi', hls: '/stream/index.m3u8' },
    };

    // Si el panel nuevo ya está en el HTML (tiene ipConnectBtn), solo registrar listeners.
    // Si no, construir el panel legacy.
    const hasNewPanel = !!document.getElementById('ipConnectBtn');

    if (!hasNewPanel) {
      const PH = {
        auto:   'http://192.168.1.x:8083/api/webrtc?src=cam',
        webrtc: 'http://192.168.1.x:8083/api/webrtc?src=cam1',
        whep:   'http://192.168.1.x:8889/cam1/whep',
        mjpeg:  'http://usuario:clave@192.168.1.x/video.cgi',
        hls:    'http://192.168.1.x/stream/cam.m3u8',
      };
      const HELP = {
        auto:   'La URL se analiza automaticamente para detectar el protocolo.',
        webrtc: 'Compatible con go2rtc y mediamtx. Puerto tipico: 8083.',
        whep:   'Protocolo moderno. Compatible con mediamtx y camaras recientes.',
        mjpeg:  'Para camaras IP baratas, Hikvision, Dahua, Axis. Incluir usuario:clave@ si hace falta.',
        hls:    'Stream .m3u8. Nativo en Safari; Chrome/Firefox usan hls.js automaticamente.',
      };
      cfg.innerHTML = `
        <div style="display:flex;flex-direction:column;gap:6px;padding:6px 0;">
          <div style="display:flex;gap:6px;align-items:center;">
            <label style="font-size:10px;letter-spacing:1.5px;color:#5a7a90;text-transform:uppercase;white-space:nowrap;">Protocolo</label>
            <select id="ipProtocol" style="padding:5px 8px;background:#0c1118;border:1px solid #1a2535;border-radius:5px;color:#c8dde8;font-size:11px;font-family:'Share Tech Mono',monospace;cursor:pointer;outline:none;">
              <option value="auto">Auto-detectar</option>
              <option value="webrtc">WebRTC (go2rtc/mediamtx)</option>
              <option value="whep">WHEP</option>
              <option value="mjpeg">MJPEG directo</option>
              <option value="hls">HLS (.m3u8)</option>
            </select>
          </div>
          <div style="display:flex;gap:6px;align-items:center;">
            <input id="ipUrl" type="text" placeholder="${PH.auto}"
              style="flex:1;padding:6px 10px;background:#0c1118;border:1px solid #1a2535;border-radius:5px;
                color:#c8dde8;font-size:11px;font-family:'Share Tech Mono',monospace;outline:none;transition:border-color .2s;"
              onfocus="this.style.borderColor='#00d4ff'" onblur="this.style.borderColor='#1a2535'"/>
            <button id="ipConnectBtn"
              style="padding:6px 14px;background:rgba(0,212,255,.08);border:1px solid #00d4ff;border-radius:5px;
                color:#00d4ff;font-size:11px;font-weight:700;cursor:pointer;font-family:'Barlow',sans-serif;white-space:nowrap;transition:background .15s;">
              Conectar →</button>
          </div>
          <div id="ipHelp" style="font-size:10px;color:#3a5468;line-height:1.5;"></div>
        </div>`;
      const proto = document.getElementById('ipProtocol');
      const urlIn = document.getElementById('ipUrl');
      const help  = document.getElementById('ipHelp');
      proto.addEventListener('change', () => { urlIn.placeholder = PH[proto.value]; help.textContent = HELP[proto.value]; });
    }

    // ── Función central: obtener URL y conectar ───────────────────────────────
    function _doConnect() {
      const ipAddr = document.getElementById('ipAddress')?.value.trim() || '';
      const ipUser = document.getElementById('ipUser')?.value.trim()    || '';
      const ipPass = document.getElementById('ipPass')?.value.trim()    || '';
      const brand  = document.getElementById('ipBrand')?.value          || 'all';
      const proto  = document.getElementById('ipProtocol')?.value       || 'auto';
      let   url    = document.getElementById('ipUrl')?.value.trim()     || '';

      // Si hay IP y la URL está vacía → auto-construir
      if (ipAddr && !url) {
        const creds = (ipUser || ipPass)
          ? `${encodeURIComponent(ipUser)}:${encodeURIComponent(ipPass)}@` : '';
        const paths = BRAND_PATHS[brand] || BRAND_PATHS.generic;
        if      (proto === 'mjpeg'  && paths.mjpeg)  url = `http://${creds}${ipAddr}${paths.mjpeg}`;
        else if (proto === 'hls'    && paths.hls)    url = `http://${creds}${ipAddr}${paths.hls}`;
        else if (proto === 'webrtc' && paths.webrtc) url = `http://${ipAddr}${paths.webrtc}`;
        else url = `http://${creds}${ipAddr}/video`;
        const urlField = document.getElementById('ipUrl');
        if (urlField) urlField.value = url;
      }

      if (!url) { _toast('Ingresa la IP o URL de la cámara', 'warn'); return; }
      const statusEl = document.getElementById('ipConnectionStatus');
      if (statusEl) { statusEl.textContent = '⏳ Conectando…'; statusEl.style.color = '#00d4ff'; }
      _setSysStatus('Conectando cámara IP…', 'loading');
      camSrc.startIP(url, proto);
    }

    // ── Botón Conectar ────────────────────────────────────────────────────────
    document.getElementById('ipConnectBtn')?.addEventListener('click', _doConnect);

    // ── Botón Descubrir ───────────────────────────────────────────────────────
    document.getElementById('ipDiscoverBtn')?.addEventListener('click', () => {
      const ipAddr = document.getElementById('ipAddress')?.value.trim();
      if (!ipAddr) { _toast('Ingresa la IP de la cámara primero', 'warn'); return; }
      const ipUser = document.getElementById('ipUser')?.value.trim() || '';
      const ipPass = document.getElementById('ipPass')?.value.trim() || '';
      const creds  = (ipUser || ipPass) ? `${encodeURIComponent(ipUser)}:${encodeURIComponent(ipPass)}@` : '';
      const statusEl = document.getElementById('ipConnectionStatus');
      if (statusEl) { statusEl.textContent = '🔍 Buscando…'; statusEl.style.color = '#ffaa00'; }
      _toast(`Probando conexión con ${ipAddr}…`, 'info', 3000);
      _setSysStatus(`Descubriendo ${ipAddr}…`, 'loading');
      camSrc.startIP(`http://${creds}${ipAddr}/video`, 'auto');
    });

    // ── Botón Proxy ───────────────────────────────────────────────────────────
    document.getElementById('connectProxyBtn')?.addEventListener('click', () => {
      const url = document.getElementById('ipUrl')?.value.trim();
      if (!url) { _toast('Ingresa la URL de la cámara primero', 'warn'); return; }
      _toast('Modo proxy activo', 'info', 3000);
      _setSysStatus('Conectando vía proxy…', 'loading');
      camSrc.startIP(url, 'mjpeg');
    });

    // ── Auto-completar URL al cambiar marca/protocolo/IP ─────────────────────
    const autoFill = () => {
      const ipAddr = document.getElementById('ipAddress')?.value.trim();
      if (!ipAddr) return;
      const brand  = document.getElementById('ipBrand')?.value  || 'all';
      const proto  = document.getElementById('ipProtocol')?.value || 'auto';
      const ipUser = document.getElementById('ipUser')?.value.trim() || '';
      const ipPass = document.getElementById('ipPass')?.value.trim() || '';
      const creds  = (ipUser || ipPass) ? `${encodeURIComponent(ipUser)}:${encodeURIComponent(ipPass)}@` : '';
      const paths  = BRAND_PATHS[brand] || BRAND_PATHS.generic;
      let url = '';
      if      (proto === 'mjpeg'  && paths.mjpeg)  url = `http://${creds}${ipAddr}${paths.mjpeg}`;
      else if (proto === 'hls'    && paths.hls)    url = `http://${creds}${ipAddr}${paths.hls}`;
      else if (proto === 'webrtc' && paths.webrtc) url = `http://${ipAddr}${paths.webrtc}`;
      const urlField = document.getElementById('ipUrl');
      if (urlField && url) urlField.value = url;
    };
    document.getElementById('ipBrand')?.addEventListener('change', autoFill);
    document.getElementById('ipProtocol')?.addEventListener('change', autoFill);
    document.getElementById('ipAddress')?.addEventListener('blur', autoFill);

    // Enter en cualquier campo del panel dispara la conexión
    ['ipAddress', 'ipUser', 'ipPass', 'ipUrl'].forEach(id => {
      document.getElementById(id)?.addEventListener('keydown', e => {
        if (e.key === 'Enter') _doConnect();
      });
    });
  })();

  // ── ARCHIVO ───────────────────────────────────────────────────────────────
  document.getElementById('btnFileSelect')?.addEventListener('click', () =>
    document.getElementById('fileInput')?.click()
  );
  document.getElementById('fileInput')?.addEventListener('change', e => {
    const f = e.target.files[0]; if (!f) return;
    camSrc.startFile(f);
    _buildVideoBar(f.name);
    e.target.value = '';
  });

  function _buildVideoBar(name) {
    document.getElementById('videoBar')?.remove();
    const bar = document.createElement('div');
    bar.id = 'videoBar';
    bar.style.cssText = 'position:absolute;bottom:0;left:0;right:0;z-index:15;background:linear-gradient(transparent,rgba(6,9,13,.95));padding:8px 12px 10px;display:flex;align-items:center;gap:10px;font-family:"Barlow",sans-serif;';
    bar.innerHTML = `
      <button id="vbPlay" style="${_vbBtn()}">▶</button>
      <div style="flex:1;display:flex;align-items:center;gap:8px;">
        <input id="vbSeek" type="range" min="0" max="100" value="0" style="flex:1;height:3px;accent-color:#00d4ff;cursor:pointer;"/>
        <span id="vbTime" style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#3a5468;white-space:nowrap;">0:00</span>
      </div>
      <button id="vbStop" style="${_vbBtn('#ff3d3d')}" title="Cerrar">✕</button>
      <span style="font-size:10px;color:#3a5468;max-width:110px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;" title="${name}">${name}</span>`;
    canvasWrapper.appendChild(bar);
    document.getElementById('vbPlay').onclick = () => {
      if (video.paused) { video.play(); document.getElementById('vbPlay').textContent='⏸'; }
      else              { video.pause(); document.getElementById('vbPlay').textContent='▶'; }
    };
    document.getElementById('vbStop').onclick = () => { camSrc.stop(); bar.remove(); };
    const seek = document.getElementById('vbSeek');
    seek.addEventListener('input', () => { video.currentTime=(seek.value/100)*(video.duration||0); });
    video.addEventListener('timeupdate', () => {
      if (!video.duration) return;
      seek.value=(video.currentTime/video.duration)*100;
      const m=Math.floor(video.currentTime/60), s=Math.floor(video.currentTime%60);
      const t=document.getElementById('vbTime'); if (t) t.textContent=`${m}:${s.toString().padStart(2,'0')}`;
    });
  }
  function _vbBtn(c='#3a5468') {
    return `background:rgba(0,0,0,.5);border:1px solid ${c};border-radius:5px;color:${c};width:28px;height:28px;cursor:pointer;font-size:12px;display:flex;align-items:center;justify-content:center;flex-shrink:0;`;
  }

  // ── BOTON EN VIVO / PAUSADO ───────────────────────────────────────────────
  btnCamToggle?.addEventListener('click', () => {
    if (!videoReady) return;
    camPaused = !camPaused;
    camPaused ? video.pause() : video.play().catch(() => {});
    _syncCamBtn(!camPaused);
  });
  function _syncCamBtn(on) {
    if (!btnCamToggle) return;
    const icon  = btnCamToggle.querySelector('#camToggleIcon');
    const label = btnCamToggle.querySelector('#camToggleLabel');
    btnCamToggle.style.background  = on?'rgba(0,255,148,.1)':'rgba(255,61,61,.1)';
    btnCamToggle.style.borderColor = on?'#00ff94':'#ff3d3d';
    btnCamToggle.style.color       = on?'#00ff94':'#ff3d3d';
    if (icon)  icon.textContent  = on?'●':'◼';
    if (label) label.textContent = on?'EN VIVO':'PAUSADO';
  }

  // ── ZONAS ─────────────────────────────────────────────────────────────────
  _buildZoneModal();

  zoneManager.onZoneChange(zones => {
    const n = zones.length;
    stateZone.textContent = n>0?`${n} zona(s) ✓`:'No definida';
    stateZone.className   = `state-val ${n>0?'ok':''}`;
    zoneHint?.classList.add('hidden');
    drawingZone = false; btnDrawZone?.classList.remove('active');
    _renderZoneList(zones);
    document.getElementById('btnEditZone') && (document.getElementById('btnEditZone').disabled = zones.length===0);
    if (btnClearZone) btnClearZone.disabled = zones.length===0;
  });

  btnDrawZone?.addEventListener('click', async () => {
    if (drawingZone) {
      zoneManager.disableDraw(); drawingZone=false;
      zoneHint?.classList.add('hidden'); btnDrawZone.classList.remove('active'); return;
    }
    if (zoneManager.zones.length>=6) { _toast('Maximo 6 zonas. Elimina una primero.','warn'); return; }
    const nombre = await _openZoneModal();
    if (!nombre) return;
    drawingZone=true; zoneManager.enableDraw(nombre);
    zoneHint?.classList.remove('hidden'); btnDrawZone.classList.add('active');
    _toast(`Dibuja "${nombre}" haciendo clic en el video. Doble clic para cerrar.`,'info',5000);
  });

  document.getElementById('btnEditZone')?.addEventListener('click', function() {
    if (!zoneManager.zones.length) return;
    const ed = this.classList.contains('active');
    if (!ed) { zoneManager.enableEdit(); this.classList.add('active'); const s=this.querySelector('span'); if(s) s.textContent='Terminar Edicion'; }
    else     { zoneManager.disableEdit(); this.classList.remove('active'); const s=this.querySelector('span'); if(s) s.textContent='Editar Zona'; _toast('Edicion guardada','ok',2000); }
  });

  btnClearZone?.addEventListener('click', () => {
    if (!zoneManager.zones.length) return;
    _confirm('Eliminar todas las zonas?', () => {
      zoneManager.clearAllZones();
      canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
      _updateControls();
    });
  });

  window._rmZone = id => zoneManager.removeZone(id);

  function _renderZoneList(zones) {
    const c=document.getElementById('zoneList'), e=document.getElementById('zoneEmpty'), n=document.getElementById('zoneCount');
    if (!c) return;
    if (n) n.textContent=`${zones.length} / 6`;
    if (e) e.style.display=zones.length?'none':'block';
    if (!zones.length) { c.innerHTML=''; return; }
    const COLS=['#00d4ff','#00ff94','#ffb800','#bf5af2','#ff6b35','#00e5ff'];
    c.innerHTML=zones.map(z=>`
      <div style="display:flex;align-items:center;gap:8px;padding:6px 9px;background:#111820;border:1px solid #1e2d3d;border-left:3px solid ${COLS[z.colorIdx%6]};border-radius:5px;">
        <div style="width:7px;height:7px;border-radius:50%;background:${COLS[z.colorIdx%6]};flex-shrink:0;"></div>
        <span style="flex:1;font-size:11px;color:#c8dde8;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-family:'Share Tech Mono',monospace;">${z.name}</span>
        <button onclick="window._rmZone('${z.id}')" style="background:none;border:none;color:#2e4558;cursor:pointer;font-size:13px;padding:0 3px;"
          onmouseenter="this.style.color='#ff3d3d'" onmouseleave="this.style.color='#2e4558'">✕</button>
      </div>`).join('');
  }

  // ── DETECCION IA ──────────────────────────────────────────────────────────
  stateModel.textContent='Cargando...'; stateModel.className='state-val warn';
  _setSysStatus('Cargando modelo IA…', 'loading');
  try {
    await detection.init();
    stateModel.textContent='Listo ✓'; stateModel.className='state-val ok';
    _setSysStatus('Modelo IA listo ✓', 'ok', 3000);
  } catch(e) {
    stateModel.textContent='Error'; stateModel.className='state-val err';
    _setSysStatus('Error al cargar modelo IA', 'err', 6000);
    console.error('Error cargando modelo IA:', e);
  }

  btnDetection?.addEventListener('click', () => { if (videoReady) _toggleDetection(); });

  function _toggleDetection() {
    if (!videoReady) return;
    if (!detectionActive && zoneManager.zones.length===0) { _toast('Define al menos una zona critica antes de iniciar el analisis.','warn'); return; }
    detectionActive = !detectionActive;
    if (detectionActive) {
      detection.start(); _startLoop();
      btnDetection?.classList.add('active');
      const s=btnDetection?.querySelector('span'); if(s) s.textContent='Detener Analisis';
      stateDetect.textContent='Activo ✓'; stateDetect.className='state-val ok';
      _setSysStatus('Análisis IA en curso', 'ok', 3000);
    } else { _stopDetection(); }
  }

  function _stopDetection() {
    detection.stop();
    if (animFrameId) { cancelAnimationFrame(animFrameId); animFrameId=null; }
    detectionActive=false;
    btnDetection?.classList.remove('active');
    const s=btnDetection?.querySelector('span'); if(s) s.textContent='Iniciar Analisis';
    stateDetect.textContent='Detenido'; stateDetect.className='state-val';
    _setSysStatus('Análisis detenido', 'warn', 2000);
  }

  let lastFT=0;
  function _startLoop() {
    async function loop(ts) {
      if (!detectionActive) return;
      if (ts-lastFT>=66) {
        lastFT=ts;
        if (video.readyState>=2&&video.videoWidth>0&&!video.paused) await detection.processFrame(video);
        if (fpsDisplay) fpsDisplay.textContent=`${detection.currentFPS} FPS`;
      }
      animFrameId=requestAnimationFrame(loop);
    }
    animFrameId=requestAnimationFrame(loop);
  }

  // Sliders
  [[sliderMovement,valMovement,50,'movementThreshold'],[sliderDwell,valDwell,3,'dwellTime'],[sliderCooldown,valCooldown,6,'cooldown']].forEach(([sld,val,def,key])=>{
    if (!sld) return;
    sld.value=def; if(val) val.textContent=def;
    sld.addEventListener('input',()=>{ const v=+sld.value; if(val) val.textContent=v; detection.updateConfig({[key]:v}); });
  });

  document.getElementById('btnExport')?.addEventListener('click',()=>alertManager.exportCSV?.());
  document.getElementById('btnLogout')?.addEventListener('click',async()=>{ const {logout}=await import('./firebase-config.js'); logout(); });

  // ── [v5.0] SELECTOR TIPO DE LOCAL ────────────────────────────────────────
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
    // Estilo consistente con el resto del panel
    storeSelect.style.cssText = [
      'padding:5px 10px',
      'background:#0c1118',
      'border:1px solid #1a2535',
      'border-radius:5px',
      'color:#c8dde8',
      'font-size:11px',
      "font-family:'Share Tech Mono',monospace",
      'cursor:pointer',
      'outline:none',
      'height:32px',
    ].join(';');
    storeSelect.addEventListener('change', () => {
      const type    = storeSelect.value;
      const label   = storeSelect.options[storeSelect.selectedIndex].text;
      const profile = detection.setStoreType(type);
      // Sincronizar slider dwellTime con el valor del perfil
      if (sliderDwell && profile?.dwellTime) {
        sliderDwell.value = profile.dwellTime;
        if (valDwell) valDwell.textContent = profile.dwellTime;
        detection.updateConfig({ dwellTime: profile.dwellTime });
      }
      // Mostrar perfil activo en Estado del Sistema
      if (stateModel) {
        stateModel.textContent = `Listo ✓ · ${label}`;
        stateModel.className   = 'state-val ok';
      }
      _toast(`Perfil: ${label}`, 'ok', 2000);
    });
  }

  // ── [v5.0] CLIC DERECHO → MARCAR EMPLEADO ────────────────────────────────
  canvas.addEventListener('contextmenu', (e) => {
    e.preventDefault();
    if (!detectionActive) return;
    const rect = canvas.getBoundingClientRect();
    const nx = ((e.clientX - rect.left) * (canvas.width  / rect.width))  / canvas.width;
    const ny = ((e.clientY - rect.top)  * (canvas.height / rect.height)) / canvas.height;
    const hit = detection.getTracks().find(t =>
      nx >= t.bbox.nx1 && nx <= t.bbox.nx2 &&
      ny >= t.bbox.ny1 && ny <= t.bbox.ny2
    );
    if (!hit) { _toast('Clic derecho sobre una persona para marcarla', 'info', 1800); return; }
    if (hit.isEmployee) {
      detection.markCustomer(hit.id);
      _toast(`Track #${hit.id} → CLIENTE (monitoreo activo)`, 'warn', 2500);
    } else {
      detection.markEmployee(hit.id);
      _toast(`Track #${hit.id} → EMPLEADO 👷 (alertas desactivadas)`, 'ok', 2500);
    }
  });

  // ── HELPERS UI ────────────────────────────────────────────────────────────
  function _updateControls() {
    if(btnDrawZone)  btnDrawZone.disabled  = !videoReady;
    if(btnDetection) btnDetection.disabled = !videoReady;
    if(btnCamToggle) btnCamToggle.disabled = !camSrc.isReady;
    if(btnStop)      btnStop.disabled      = !camSrc.isReady;
    if(btnPause)     btnPause.disabled     = !videoReady;
    const hz=zoneManager.zones.length>0;
    const btnEZ=document.getElementById('btnEditZone');
    if(btnEZ)        btnEZ.disabled        = !hz;
    if(btnClearZone) btnClearZone.disabled = !hz;
    // Reset pause button when camera stops
    if (!videoReady && btnPause) _syncPauseBtn(false);
  }

  function _syncPauseBtn(paused) {
    if (!btnPause) return;
    const icon  = btnPause.querySelector('svg') || btnPause;
    const span  = btnPause.querySelector('span');
    if (paused) {
      btnPause.classList.add('active');
      btnPause.style.borderColor = 'var(--warn, #ffaa00)';
      btnPause.style.color       = 'var(--warn, #ffaa00)';
      if (span) span.textContent = 'Reanudar';
    } else {
      btnPause.classList.remove('active');
      btnPause.style.borderColor = '';
      btnPause.style.color       = '';
      if (span) span.textContent = 'Pausar';
    }
  }

  function _flashStatus(sev) {
    const b=document.getElementById('sensStatus'); if(!b) return;
    b.textContent=sev==='high'?'⚠ ALERTA':'⚠ AVISO';
    b.style.color=sev==='high'?'var(--danger,#ff3d3d)':'var(--warn,#ffaa00)';
    clearTimeout(b._t); b._t=setTimeout(()=>{ if(detectionActive){b.textContent='ACTIVO';b.style.color='var(--ok,#00e676)';} },3000);
  }

  function _toast(msg, type='info', ms=3500) {
    const cols={info:'#00c8ff',warn:'#ffaa00',ok:'#00e676',err:'#ff3d3d'};
    const n=document.createElement('div');
    n.style.cssText=`position:fixed;top:70px;right:20px;z-index:9999;background:#0d1520;
      border:1px solid ${cols[type]||cols.info};border-radius:10px;padding:13px 18px;
      max-width:340px;font-family:'Barlow',sans-serif;font-size:13px;color:#c8dde8;
      box-shadow:0 8px 32px rgba(0,0,0,.6);line-height:1.5;
      animation:_tIn .22s ease;`;
    n.innerHTML=`<style>@keyframes _tIn{from{opacity:0;transform:translateX(18px)}to{opacity:1;transform:translateX(0)}}</style>${msg}`;
    document.body.appendChild(n);
    setTimeout(()=>{n.style.opacity='0';n.style.transition='opacity .4s';},ms);
    setTimeout(()=>n.remove(),ms+450);
  }

  // ── Status bar del sidebar ────────────────────────────────────────────────
  function _setSysStatus(msg, type = 'loading', autoDismissMs = 0) {
    const bar  = document.getElementById('sysStatus');
    const text = document.getElementById('sysStatusMsg');
    if (!bar || !text) return;
    clearTimeout(_ssTimer);
    bar.className    = `sys-status ${type}`;
    text.textContent = msg;
    if (autoDismissMs > 0) {
      _ssTimer = setTimeout(() => {
        bar.style.opacity = '0';
        setTimeout(() => { bar.classList.add('hidden'); bar.style.opacity = ''; }, 400);
      }, autoDismissMs);
    }
  }
  function _hideSysStatus() {
    const bar = document.getElementById('sysStatus');
    if (!bar) return;
    clearTimeout(_ssTimer);
    bar.style.opacity = '0';
    setTimeout(() => { bar.classList.add('hidden'); bar.style.opacity = ''; }, 400);
  }

  function _confirm(msg, onOk) {
    const ov=document.createElement('div');
    ov.style.cssText='position:fixed;inset:0;z-index:9998;background:rgba(0,0,0,.7);display:flex;align-items:center;justify-content:center;font-family:"Barlow",sans-serif;';
    ov.innerHTML=`<div style="background:#0d1520;border:1px solid rgba(255,61,61,.4);border-radius:12px;padding:24px 26px;width:300px;">
      <div style="font-size:13px;color:#c8dde8;margin-bottom:20px;line-height:1.6;">${msg}</div>
      <div style="display:flex;gap:10px;">
        <button id="cNo"  style="flex:1;padding:9px;background:transparent;border:1px solid #1a2535;border-radius:6px;color:#5a7a90;font-size:12px;font-weight:600;cursor:pointer;">Cancelar</button>
        <button id="cYes" style="flex:1;padding:9px;background:rgba(255,61,61,.1);border:1px solid #ff3d3d;border-radius:6px;color:#ff3d3d;font-size:12px;font-weight:700;cursor:pointer;">Eliminar</button>
      </div></div>`;
    document.body.appendChild(ov);
    ov.querySelector('#cYes').onclick=()=>{ov.remove();onOk();};
    ov.querySelector('#cNo').onclick=()=>ov.remove();
    ov.onclick=e=>{if(e.target===ov)ov.remove();};
  }

  function _buildZoneModal() {
    if (document.getElementById('zoneModal')) return;
    const m=document.createElement('div');
    m.id='zoneModal';
    m.style.cssText='position:fixed;inset:0;z-index:5000;background:rgba(0,0,0,.75);display:none;align-items:center;justify-content:center;font-family:"Barlow",sans-serif;';
    m.innerHTML=`
      <div style="background:#0d1520;border:1px solid rgba(0,212,255,.4);border-radius:14px;padding:28px;width:340px;box-shadow:0 20px 60px rgba(0,0,0,.8);animation:zmIn .25s ease;">
        <div style="font-size:13px;font-weight:700;color:#00d4ff;letter-spacing:2px;text-transform:uppercase;margin-bottom:6px;">Nueva zona critica</div>
        <div style="font-size:12px;color:#5a7a90;margin-bottom:18px;line-height:1.6;">
          Asigna un nombre y luego dibuja el contorno haciendo clic en el video.<br>
          <span style="color:#00e676;font-size:11px;">Si incluye "caja", "pago" o "cobro" es zona de pago.</span></div>
        <label style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3a5468;display:block;margin-bottom:7px;">Nombre</label>
        <input id="zmInput" type="text" placeholder="Ej: Caja, Estanteria, Entrada"
          style="width:100%;padding:11px 13px;background:rgba(0,0,0,.35);border:1px solid #1a2535;border-radius:7px;
          color:#c8dde8;font-family:'Share Tech Mono',monospace;font-size:13px;outline:none;box-sizing:border-box;transition:border-color .2s;"/>
        <div id="zmSugs" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;"></div>
        <div style="display:flex;gap:10px;margin-top:20px;">
          <button id="zmCancel" style="flex:1;padding:10px;background:transparent;border:1px solid #1a2535;border-radius:7px;color:#5a7a90;font-size:12px;font-weight:600;cursor:pointer;">Cancelar</button>
          <button id="zmOk" style="flex:2;padding:10px;background:rgba(0,212,255,.12);border:1px solid #00d4ff;border-radius:7px;color:#00d4ff;font-size:12px;font-weight:700;cursor:pointer;letter-spacing:1px;">Dibujar →</button>
        </div>
      </div>
      <style>@keyframes zmIn{from{opacity:0;transform:scale(.93)}to{opacity:1;transform:scale(1)}}</style>`;
    document.body.appendChild(m);
    const inp=document.getElementById('zmInput');
    inp.addEventListener('focus',()=>inp.style.borderColor='#00d4ff');
    inp.addEventListener('blur', ()=>inp.style.borderColor='#1a2535');
    ['Caja/Cobro','Pago','Estanteria','Deposito','Entrada','Cocina'].forEach(s=>{
      const b=document.createElement('button');
      b.textContent=s; b.style.cssText='padding:3px 10px;background:rgba(0,212,255,.06);border:1px solid #1a2535;border-radius:20px;color:#5a7a90;font-size:11px;cursor:pointer;transition:all .15s;';
      b.onmouseenter=()=>{b.style.borderColor='#00d4ff';b.style.color='#00d4ff';};
      b.onmouseleave=()=>{b.style.borderColor='#1a2535';b.style.color='#5a7a90';};
      b.onclick=()=>inp.value=s;
      document.getElementById('zmSugs').appendChild(b);
    });
  }

  function _openZoneModal() {
    return new Promise(resolve=>{
      const m=document.getElementById('zoneModal'); m.style.display='flex';
      const inp=document.getElementById('zmInput'),ok=document.getElementById('zmOk'),can=document.getElementById('zmCancel');
      inp.value=''; setTimeout(()=>inp.focus(),80);
      const fin=v=>{m.style.display='none';ok.removeEventListener('click',onOk);can.removeEventListener('click',onCan);inp.removeEventListener('keydown',onKey);resolve(v);};
      const onOk=()=>fin(inp.value.trim()||null);
      const onCan=()=>fin(null);
      const onKey=e=>{if(e.key==='Enter')onOk();if(e.key==='Escape')onCan();};
      ok.addEventListener('click',onOk); can.addEventListener('click',onCan); inp.addEventListener('keydown',onKey);
      m.addEventListener('click',e=>{if(e.target===m)onCan();},{once:true});
    });
  }

  // Init
  _updateControls();
  // Activar tab webcam y arrancar
  const webcamTab = document.querySelector('.source-tab[data-source="webcam"]');
  if (webcamTab) webcamTab.click();
  else setTimeout(()=>_launchWebcam(), 300);

  console.log('%cSSIP v4.0 monitor.html listo','color:#00d4ff;font-weight:bold');
}