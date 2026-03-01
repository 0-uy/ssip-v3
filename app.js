/**
 * app.js — Orquestador SSIP v2.1
 * + Modal bonito para nombre de zona
 * + Controles de video (play/pause/borrar/progreso)
 * + Polígono con render loop siempre activo
 */

export async function initApp({ ZoneManager, DetectionEngine, AlertManager }) {

  const video         = document.getElementById('videoElement');
  const canvas        = document.getElementById('overlayCanvas');
  const noSignal      = document.getElementById('noSignal');
  const canvasWrapper = document.getElementById('canvasWrapper');
  const btnDrawZone   = document.getElementById('btnDrawZone');
  const btnClearZone  = document.getElementById('btnClearZone');
  const btnDetection  = document.getElementById('btnDetection');
  const fpsDisplay    = document.getElementById('fpsDisplay');
  const zoneHint      = document.getElementById('zoneHint');
  const stateVideo    = document.getElementById('stateVideo');
  const stateDetect   = document.getElementById('stateDetection');
  const stateZone     = document.getElementById('stateZone');
  const stateModel    = document.getElementById('stateModel');
  const sliderMovement = document.getElementById('sliderMovement');
  const sliderDwell    = document.getElementById('sliderDwell');
  const sliderCooldown = document.getElementById('sliderCooldown');
  const valMovement    = document.getElementById('valMovement');
  const valDwell       = document.getElementById('valDwell');
  const valCooldown    = document.getElementById('valCooldown');

  let videoReady=false, detectionActive=false, drawingZone=false;
  let currentSource='webcam', animFrameId=null, mediaStream=null;

  const zoneManager  = new ZoneManager(canvas);
  const alertManager = new AlertManager(canvas);
  const detection    = new DetectionEngine(canvas, zoneManager, alertManager, {
    movementThreshold: 50, dwellTime: 3, cooldown: 6,
  });

  detection.onDetection = (type, severity) => {
    // 'info' = entrada de zona, solo visual, no llena historial
    // 'medium' / 'high' = alerta real del sistema de scoring
    if (severity === 'info') return; // ya se muestra en canvas, no hacer nada más
    updateStateUI(type, severity);
  };

  /* ══ Modal bonito para nombre de zona ════════════════ */
  function createZoneModal() {
    if (document.getElementById('zoneModal')) return;
    const m = document.createElement('div');
    m.id = 'zoneModal';
    m.style.cssText = `
      position:fixed;inset:0;z-index:5000;background:rgba(0,0,0,0.75);
      display:none;align-items:center;justify-content:center;
      font-family:'Barlow',sans-serif;
    `;
    m.innerHTML = `
      <div style="background:#0d1520;border:1px solid rgba(0,212,255,0.4);border-radius:14px;
        padding:28px 28px 22px;width:340px;box-shadow:0 20px 60px rgba(0,0,0,0.8);
        animation:modalIn .25s ease;">
        <div style="font-size:13px;font-weight:700;color:#00d4ff;letter-spacing:2px;
          text-transform:uppercase;margin-bottom:6px;">Nueva zona crítica</div>
        <div style="font-size:12px;color:#5a7a90;margin-bottom:18px;line-height:1.6;">
          Poné un nombre descriptivo para esta zona.<br>
          Después vas a dibujar el contorno haciendo clic sobre el video.<br>
          <span style="color:#00e676;font-size:11px;">💳 Si el nombre incluye "caja", "pago" o "cobro" → se crea como zona de pago (objetos que pasen por ahí no generan alerta).</span>
        </div>
        <label style="font-size:10px;letter-spacing:2px;text-transform:uppercase;color:#3a5468;display:block;margin-bottom:7px;">Nombre de la zona</label>
        <input id="zoneNameInput" type="text" placeholder="Ej: Zona Caja, Estantería, Cocina"
          style="width:100%;padding:11px 13px;background:rgba(0,0,0,0.35);
          border:1px solid #1a2535;border-radius:7px;color:#c8dde8;
          font-family:'Share Tech Mono',monospace;font-size:13px;outline:none;
          transition:border-color .2s;"/>
        <div id="zoneSuggestions" style="display:flex;gap:6px;flex-wrap:wrap;margin-top:10px;"></div>
        <div style="display:flex;gap:10px;margin-top:20px;">
          <button id="zoneModalCancel" style="flex:1;padding:10px;background:transparent;
            border:1px solid #1a2535;border-radius:7px;color:#5a7a90;font-size:12px;
            font-weight:600;cursor:pointer;font-family:'Barlow',sans-serif;">Cancelar</button>
          <button id="zoneModalOk" style="flex:2;padding:10px;background:rgba(0,212,255,0.12);
            border:1px solid #00d4ff;border-radius:7px;color:#00d4ff;font-size:12px;
            font-weight:700;cursor:pointer;font-family:'Barlow',sans-serif;
            letter-spacing:1px;">Dibujar zona →</button>
        </div>
      </div>
      <style>@keyframes modalIn{from{opacity:0;transform:scale(.93)}to{opacity:1;transform:scale(1)}}</style>`;
    document.body.appendChild(m);

    // Sugerencias de nombres
    // Las que contienen "caja", "pago", "cobro" etc. se detectan como zona de pago (borde verde 💳)
    const suggestions = ['Caja / Cobro 💳','Pago 💳','Estantería','Depósito','Entrada','Cocina'];
    const sugDiv = document.getElementById('zoneSuggestions');
    suggestions.forEach(s => {
      const btn = document.createElement('button');
      btn.textContent = s;
      btn.style.cssText = `padding:3px 10px;background:rgba(0,212,255,0.06);border:1px solid #1a2535;
        border-radius:20px;color:#5a7a90;font-size:11px;cursor:pointer;font-family:'Barlow',sans-serif;
        transition:all .15s;`;
      btn.onmouseenter = () => { btn.style.borderColor='#00d4ff'; btn.style.color='#00d4ff'; };
      btn.onmouseleave = () => { btn.style.borderColor='#1a2535'; btn.style.color='#5a7a90'; };
      btn.onclick = () => { document.getElementById('zoneNameInput').value = s; };
      sugDiv.appendChild(btn);
    });

    const input = document.getElementById('zoneNameInput');
    input.style.borderColor = '#1a2535';
    input.addEventListener('focus',  () => input.style.borderColor='#00d4ff');
    input.addEventListener('blur',   () => input.style.borderColor='#1a2535');
    input.addEventListener('keydown', e => { if (e.key==='Enter') document.getElementById('zoneModalOk').click(); });
  }

  function openZoneModal() {
    return new Promise(resolve => {
      const m = document.getElementById('zoneModal');
      m.style.display = 'flex';
      const input  = document.getElementById('zoneNameInput');
      const okBtn  = document.getElementById('zoneModalOk');
      const cancel = document.getElementById('zoneModalCancel');
      input.value  = '';
      setTimeout(() => input.focus(), 100);

      const ok = () => {
        const val = input.value.trim() || `Zona ${zoneManager.zones.length + 1}`;
        m.style.display = 'none';
        cleanup();
        resolve(val);
      };
      const canc = () => {
        m.style.display = 'none';
        cleanup();
        resolve(null);
      };
      function cleanup() { okBtn.removeEventListener('click',ok); cancel.removeEventListener('click',canc); }
      okBtn.addEventListener('click', ok);
      cancel.addEventListener('click', canc);
      m.addEventListener('click', e => { if (e.target===m) canc(); }, { once:true });
    });
  }

  /* ══ Zona change handler ══════════════════════════════ */
  zoneManager.onZoneChange(zones => {
    const n = zones.length;
    stateZone.textContent = n>0 ? `${n} zona(s) ✓` : 'No definida';
    stateZone.className   = `state-val ${n>0?'ok':''}`;
    zoneHint.classList.add('hidden');
    drawingZone = false;
    btnDrawZone.classList.remove('active');
    renderZoneList(zones);
    // Habilitar botones de editar/limpiar cuando hay zonas
    const btnEZ = document.getElementById('btnEditZone');
    if(btnEZ) btnEZ.disabled = zones.length === 0;
    if(btnClearZone) btnClearZone.disabled = zones.length === 0;
  });

  /* ══ Lista de zonas en sidebar ════════════════════════ */
  function renderZoneList(zones) {
    const container = document.getElementById('zoneList');
    const empty     = document.getElementById('zoneEmpty');
    const counter   = document.getElementById('zoneCount');
    if (!container) return;

    if (counter) counter.textContent = zones.length + ' / 6';
    if (empty)   empty.style.display = zones.length ? 'none' : 'block';

    if (!zones.length) { container.innerHTML = ''; return; }

    const COLS=['#00d4ff','#00ff94','#ffb800','#bf5af2','#ff6b35','#00e5ff'];
    container.innerHTML = zones.map(z=>`
      <div style="display:flex;align-items:center;gap:8px;padding:6px 9px;
        background:var(--bg-card,#111820);border:1px solid var(--border,#1e2d3d);
        border-left:3px solid ${COLS[z.colorIdx%COLS.length]};border-radius:5px;
        flex-shrink:0;">
        <div style="width:7px;height:7px;border-radius:50%;background:${COLS[z.colorIdx%COLS.length]};flex-shrink:0;box-shadow:0 0 5px ${COLS[z.colorIdx%COLS.length]};"></div>
        <span style="flex:1;font-size:11px;color:#c8dde8;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;font-family:'Share Tech Mono',monospace;">${z.name}</span>
        <button onclick="window._rmZone('${z.id}')" title="Eliminar zona"
          style="background:none;border:none;color:#2e4558;cursor:pointer;font-size:13px;
          padding:0 3px;line-height:1;transition:color .15s;"
          onmouseenter="this.style.color='#ff3d3d'" onmouseleave="this.style.color='#2e4558'">✕</button>
      </div>`).join('');
  }
  window._rmZone = id => zoneManager.removeZone(id);

  /* ══ Botón dibujar zona ═══════════════════════════════ */
  createZoneModal();

  btnDrawZone.addEventListener('click', async () => {
    if (drawingZone) {
      zoneManager.disableDraw();
      drawingZone=false; zoneHint.classList.add('hidden');
      btnDrawZone.classList.remove('active'); return;
    }
    if (zoneManager.zones.length >= 6) { showNotif('Máximo 6 zonas. Eliminá una primero.','warn'); return; }

    const nombre = await openZoneModal();
    if (!nombre) return;

    drawingZone = true;
    zoneManager.enableDraw(nombre);
    zoneHint.classList.remove('hidden');
    btnDrawZone.classList.add('active');
    showNotif(`Dibujá "${nombre}" haciendo clic en el video. Doble clic para cerrar.`, 'info', 5000);
  });

  // Botón editar zona
  const btnEditZone = document.getElementById('btnEditZone');
  if (btnEditZone) {
    btnEditZone.addEventListener('click', () => {
      if (!zoneManager.zones.length) { showNotif('No hay zonas para editar.','warn'); return; }
      const isEditing = btnEditZone.classList.contains('active');
      if (isEditing) {
        zoneManager.disableEdit();
        btnEditZone.classList.remove('active');
        btnEditZone.querySelector('span').textContent = 'Editar Zona';
        showNotif('Edición guardada ✓', 'ok', 2000);
      } else {
        zoneManager.enableEdit();
        btnEditZone.classList.add('active');
        btnEditZone.querySelector('span').textContent = 'Terminar Edición';
        showNotif('Arrastrá los puntos blancos para mover los vértices de la zona.', 'info', 5000);
      }
    });
  }

  btnClearZone.addEventListener('click', () => {
    if (!zoneManager.zones.length) return;
    showConfirm('¿Eliminar todas las zonas?', () => {
      zoneManager.clearAllZones();
      canvas.getContext('2d').clearRect(0,0,canvas.width,canvas.height);
      updateControlsState();
    });
  });

  /* ══ Notificación bonita (reemplaza alert/confirm) ═══ */
  function showNotif(msg, type='info', duration=3500) {
    const colors = { info:'#00d4ff', warn:'#ffb800', ok:'#00ff94', err:'#ff3d3d' };
    const n = document.createElement('div');
    n.style.cssText = `
      position:fixed;top:70px;right:20px;z-index:9999;
      background:#0d1520;border:1px solid ${colors[type]||colors.info};
      border-radius:10px;padding:13px 18px;max-width:320px;
      font-family:'Barlow',sans-serif;font-size:13px;color:#c8dde8;
      box-shadow:0 8px 32px rgba(0,0,0,0.6);
      animation:nIn .25s ease;line-height:1.5;`;
    n.innerHTML = `<style>@keyframes nIn{from{opacity:0;transform:translateX(20px)}to{opacity:1;transform:translateX(0)}}</style>${msg}`;
    document.body.appendChild(n);
    setTimeout(()=>n.style.opacity='0', duration);
    setTimeout(()=>n.remove(), duration+400);
  }

  function showConfirm(msg, onOk) {
    const overlay = document.createElement('div');
    overlay.style.cssText='position:fixed;inset:0;z-index:9998;background:rgba(0,0,0,0.7);display:flex;align-items:center;justify-content:center;font-family:"Barlow",sans-serif;';
    overlay.innerHTML=`
      <div style="background:#0d1520;border:1px solid rgba(255,61,61,0.4);border-radius:12px;padding:24px 26px;width:300px;box-shadow:0 16px 48px rgba(0,0,0,0.7);">
        <div style="font-size:13px;color:#c8dde8;margin-bottom:20px;line-height:1.6;">${msg}</div>
        <div style="display:flex;gap:10px;">
          <button id="cNo"  style="flex:1;padding:9px;background:transparent;border:1px solid #1a2535;border-radius:6px;color:#5a7a90;font-size:12px;font-weight:600;cursor:pointer;font-family:'Barlow',sans-serif;">Cancelar</button>
          <button id="cYes" style="flex:1;padding:9px;background:rgba(255,61,61,0.1);border:1px solid #ff3d3d;border-radius:6px;color:#ff3d3d;font-size:12px;font-weight:700;cursor:pointer;font-family:'Barlow',sans-serif;">Eliminar</button>
        </div>
      </div>`;
    document.body.appendChild(overlay);
    document.getElementById('cYes').onclick=()=>{ overlay.remove(); onOk(); };
    document.getElementById('cNo').onclick =()=>{ overlay.remove(); };
    overlay.onclick=e=>{ if(e.target===overlay) overlay.remove(); };
  }

  /* ══ Reloj ════════════════════════════════════════════ */
  setInterval(()=>{
    const el=document.getElementById('headerTime');
    if(el) el.textContent=new Date().toLocaleTimeString('es-UY',{hour12:false});
  },1000);

  /* ══ Tabs fuente ══════════════════════════════════════ */
  document.querySelectorAll('.source-tab').forEach(tab=>{
    tab.addEventListener('click',()=>{
      document.querySelectorAll('.source-tab').forEach(t=>t.classList.remove('active'));
      tab.classList.add('active');
      currentSource=tab.dataset.source;
      document.getElementById('webrtcConfig')?.classList.add('hidden');
      document.getElementById('fileConfig')?.classList.add('hidden');
      if(currentSource==='webcam') startWebcam();
      else if(currentSource==='webrtc') document.getElementById('webrtcConfig')?.classList.remove('hidden');
      else document.getElementById('fileConfig')?.classList.remove('hidden');
    });
  });

  /* ══ Webcam ═══════════════════════════════════════════ */
  // Selector de cámara
  let _allCameras = [];
  let _selectedCamId = null;

  async function buildCameraSelector() {
    try {
      // Primero pedir permiso para ver los labels
      await navigator.mediaDevices.getUserMedia({video:true,audio:false});
      const devices = await navigator.mediaDevices.enumerateDevices();
      _allCameras = devices.filter(d => d.kind === 'videoinput');

      let sel = document.getElementById('camSelector');
      if (!sel) {
        const bar = document.querySelector('.source-bar');
        const wrap = document.createElement('div');
        wrap.style.cssText = 'display:flex;align-items:center;gap:8px;margin-top:6px;';
        wrap.innerHTML = `
          <label style="font-size:10px;letter-spacing:1.5px;color:var(--text-secondary,#5a7a90);text-transform:uppercase;white-space:nowrap;">Cámara:</label>
          <select id="camSelector" style="
            flex:1;padding:5px 10px;background:#0c1118;border:1px solid #1a2535;
            border-radius:5px;color:#c8dde8;font-size:12px;
            font-family:'Share Tech Mono',monospace;cursor:pointer;outline:none;">
          </select>`;
        bar.appendChild(wrap);
        sel = document.getElementById('camSelector');
        sel.addEventListener('change', () => {
          _selectedCamId = sel.value;
          startWebcam();
        });
      }

      sel.innerHTML = _allCameras.map((cam, i) => {
        const label = cam.label || `Cámara ${i+1}`;
        return `<option value="${cam.deviceId}">${label}</option>`;
      }).join('');

      // Preferir cámara externa (USB) — buscar la que NO sea "integrated" o "facetime"
      const external = _allCameras.find(c =>
        c.label && !/integrated|facetime|built.?in|internal/i.test(c.label)
      );
      if (external && !_selectedCamId) {
        _selectedCamId = external.deviceId;
        sel.value = external.deviceId;
      } else if (!_selectedCamId && _allCameras.length > 0) {
        // Si solo hay una o no encontró externa, usar la última (USB suele ser la última)
        const preferred = _allCameras[_allCameras.length - 1];
        _selectedCamId = preferred.deviceId;
        sel.value = preferred.deviceId;
      }
    } catch(e) {
      console.warn('No se pudieron listar cámaras:', e);
    }
  }

  async function startWebcam() {
    stopCurrentStream();
    try {
      await buildCameraSelector();
      const constraints = {
        video: {
          width:{ideal:1280}, height:{ideal:720}, frameRate:{ideal:30,max:30},
          ..._selectedCamId ? { deviceId: { exact: _selectedCamId } } : {},
        },
        audio: false,
      };
      mediaStream = await navigator.mediaDevices.getUserMedia(constraints);
      video.srcObject = mediaStream; video.classList.remove('hidden');
      // Mostrar nombre de cámara activa
      const track = mediaStream.getVideoTracks()[0];
      onVideoReady(track?.label || 'Webcam'); hideVideoBar();
    } catch(e) {
      setVideoError('Sin acceso a webcam. Verificar permisos.');
      console.error(e);
    }
  }

  /* ══ WebRTC ═══════════════════════════════════════════ */
  document.getElementById('btnConnect')?.addEventListener('click',()=>{
    const url=document.getElementById('webrtcUrl')?.value.trim();
    if(url) startWebRTC(url);
  });
  async function startWebRTC(url) {
    stopCurrentStream();
    try {
      const pc=new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'}]});
      pc.ontrack=evt=>{
        if(evt.streams?.[0]){video.srcObject=evt.streams[0];video.classList.remove('hidden');onVideoReady('IP Camera');hideVideoBar();}
      };
      pc.addTransceiver('video',{direction:'recvonly'});
      const offer=await pc.createOffer(); await pc.setLocalDescription(offer);
      const resp=await fetch(url,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({data:btoa(offer.sdp)})});
      const body=await resp.text(); const b64=JSON.parse(body).answer??body;
      await pc.setRemoteDescription({type:'answer',sdp:atob(b64)});
      window._ssip_pc=pc;
    } catch(e){setVideoError(`Error WebRTC: ${e.message}`);}
  }

  /* ══ Archivo con controles ════════════════════════════ */
  document.getElementById('btnFileSelect')?.addEventListener('click',()=>document.getElementById('fileInput')?.click());
  document.getElementById('fileInput')?.addEventListener('change',e=>{
    const file=e.target.files[0]; if(!file) return;
    stopCurrentStream();
    video.srcObject=null; video.src=URL.createObjectURL(file);
    video.loop=true; video.classList.remove('hidden'); video.play();
    onVideoReady(`${file.name}`);
    showVideoBar(file.name);
  });

  /* ══ Barra de controles de video ══════════════════════ */
  function showVideoBar(filename) {
    let bar = document.getElementById('videoBar');
    if (!bar) {
      bar = document.createElement('div');
      bar.id = 'videoBar';
      bar.style.cssText=`
        position:absolute;bottom:0;left:0;right:0;z-index:15;
        background:linear-gradient(transparent,rgba(6,9,13,0.95));
        padding:8px 12px 10px;display:flex;align-items:center;gap:10px;
        font-family:'Barlow',sans-serif;`;
      canvasWrapper.appendChild(bar);
    }
    bar.innerHTML=`
      <button id="vbPlay" title="Play/Pausa" style="${vBtnStyle()}">▶</button>
      <div style="flex:1;display:flex;align-items:center;gap:8px;">
        <input id="vbSeek" type="range" min="0" max="100" value="0"
          style="flex:1;height:3px;accent-color:#00d4ff;cursor:pointer;"/>
        <span id="vbTime" style="font-family:'Share Tech Mono',monospace;font-size:10px;color:#3a5468;white-space:nowrap;">0:00</span>
      </div>
      <button id="vbDelete" title="Quitar video" style="${vBtnStyle('#ff3d3d')}">✕</button>
      <span style="font-size:10px;color:#3a5468;max-width:120px;overflow:hidden;white-space:nowrap;text-overflow:ellipsis;" title="${filename}">${filename}</span>`;

    document.getElementById('vbPlay').onclick=()=>{
      if(video.paused){video.play();document.getElementById('vbPlay').textContent='⏸';}
      else{video.pause();document.getElementById('vbPlay').textContent='▶';}
    };
    document.getElementById('vbDelete').onclick=()=>{
      stopCurrentStream(); hideVideoBar();
      showNotif('Video eliminado','info',2000);
    };
    const seek=document.getElementById('vbSeek');
    seek.addEventListener('input',()=>{ video.currentTime=(seek.value/100)*video.duration||0; });
    video.addEventListener('timeupdate',()=>{
      if(!video.duration) return;
      seek.value=(video.currentTime/video.duration)*100;
      const m=Math.floor(video.currentTime/60), s=Math.floor(video.currentTime%60);
      const tm=document.getElementById('vbTime');
      if(tm) tm.textContent=`${m}:${s.toString().padStart(2,'0')}`;
    });
  }

  function vBtnStyle(color='#3a5468') {
    return `background:rgba(0,0,0,0.5);border:1px solid ${color};border-radius:5px;
      color:${color};width:28px;height:28px;cursor:pointer;font-size:12px;
      display:flex;align-items:center;justify-content:center;flex-shrink:0;`;
  }

  function hideVideoBar() {
    document.getElementById('videoBar')?.remove();
  }

  /* ══ Helpers video ════════════════════════════════════ */
  function onVideoReady(src) {
    video.addEventListener('loadedmetadata',resizeCanvas,{once:true});
    video.play().catch(()=>{});
    noSignal?.classList.add('hidden');
    videoReady=true;
    stateVideo.textContent=src; stateVideo.className='state-val ok';
    alertManager.setOnline?.(); updateControlsState();
  }
  function setVideoError(msg) {
    noSignal?.classList.remove('hidden');
    const p=noSignal?.querySelector('p'); if(p) p.textContent=msg;
    stateVideo.textContent='Error'; stateVideo.className='state-val err';
  }
  function stopCurrentStream() {
    if(detectionActive) toggleDetection();
    mediaStream?.getTracks?.().forEach(t=>t.stop());
    window._ssip_pc?.close(); window._ssip_pc=null;
    video.srcObject=null; video.src=''; video.classList.add('hidden');
    noSignal?.classList.remove('hidden'); videoReady=false;
    stateVideo.textContent='—'; stateVideo.className='state-val';
    alertManager.setOffline?.(); updateControlsState();
  }

  /* ══ Canvas resize ════════════════════════════════════ */
  function resizeCanvas() {
    const r=canvasWrapper.getBoundingClientRect();
    canvas.width=r.width; canvas.height=r.height;
  }
  new ResizeObserver(resizeCanvas).observe(canvasWrapper);
  resizeCanvas();

  /* ══ MediaPipe init ════════════════════════════════════ */
  stateModel.textContent='Cargando…'; stateModel.className='state-val warn';
  try {
    await detection.init();
    stateModel.textContent='Listo ✓'; stateModel.className='state-val ok';
  } catch(e) {
    stateModel.textContent='Error'; stateModel.className='state-val err';
  }

  /* ══ Toggle detección ══════════════════════════════════ */
  btnDetection.addEventListener('click',()=>{ if(videoReady) toggleDetection(); });
  function toggleDetection() {
    if(!videoReady) return;
    if(!detectionActive && zoneManager.zones.length===0) {
      showNotif('Definí al menos una zona crítica primero.','warn'); return;
    }
    detectionActive=!detectionActive;
    if(detectionActive) {
      detection.start(); startProcessingLoop();
      btnDetection.classList.add('active');
      const s=btnDetection.querySelector('span'); if(s) s.textContent='Detener Análisis';
      stateDetect.textContent='Activo ✓'; stateDetect.className='state-val ok';
    } else {
      detection.stop();
      if(animFrameId){cancelAnimationFrame(animFrameId);animFrameId=null;}
      btnDetection.classList.remove('active');
      const s=btnDetection.querySelector('span'); if(s) s.textContent='Iniciar Análisis';
      stateDetect.textContent='Detenido'; stateDetect.className='state-val';
    }
  }

  /* ══ Loop 15 FPS ══════════════════════════════════════ */
  let lastFT=0;
  function startProcessingLoop() {
    async function loop(ts) {
      if(!detectionActive) return;
      if(ts-lastFT >= 1000/15) {
        lastFT=ts;
        if(video.readyState>=2) await detection.processFrame(video);
        if(fpsDisplay) fpsDisplay.textContent=`${detection.currentFPS} FPS`;
      }
      animFrameId=requestAnimationFrame(loop);
    }
    animFrameId=requestAnimationFrame(loop);
  }

  /* ══ Sliders con defaults mejorados ═══════════════════ */
  const defaults={sliderMovement:50,sliderDwell:3,sliderCooldown:6};
  for(const [id,val] of Object.entries(defaults)){
    const el=document.getElementById(id); if(el) el.value=val;
  }
  if(valMovement) valMovement.textContent=50;
  if(valDwell)    valDwell.textContent=3;
  if(valCooldown) valCooldown.textContent=6;

  sliderMovement?.addEventListener('input',()=>{
    const v=+sliderMovement.value; if(valMovement) valMovement.textContent=v;
    detection.updateConfig({movementThreshold:v});
  });
  sliderDwell?.addEventListener('input',()=>{
    const v=+sliderDwell.value; if(valDwell) valDwell.textContent=v;
    detection.updateConfig({dwellTime:v});
  });
  sliderCooldown?.addEventListener('input',()=>{
    const v=+sliderCooldown.value; if(valCooldown) valCooldown.textContent=v;
    detection.updateConfig({cooldown:v});
  });

  /* ══ Export CSV ═══════════════════════════════════════ */
  document.getElementById('btnExport')?.addEventListener('click',()=>alertManager.exportCSV?.());

  /* ══ Logout ═══════════════════════════════════════════ */
  document.getElementById('btnLogout')?.addEventListener('click',async()=>{
    const {logout}=await import('./firebase-config.js'); logout();
  });

  function updateControlsState() {
    const hasZones = zoneManager.zones.length > 0;
    if(btnDrawZone)  btnDrawZone.disabled  = !videoReady;
    if(btnDetection) btnDetection.disabled = !videoReady;
    const btnEZ = document.getElementById('btnEditZone');
    if(btnEZ)        btnEZ.disabled        = !hasZones;
    if(btnClearZone) btnClearZone.disabled  = !hasZones;
  }
  function updateStateUI(type, severity) {
    // Actualizar badge de detección en sidebar cuando hay alerta
    if (!type) return;
    const badge = document.getElementById('sensStatus');
    if (badge) {
      badge.textContent = severity === 'high' ? '⚠ ALERTA' : 'ACTIVO';
      badge.style.color  = severity === 'high' ? 'var(--danger,#ff3d3d)' : 'var(--ok,#00e676)';
      // Volver a ACTIVO después de 3s
      clearTimeout(badge._t);
      badge._t = setTimeout(() => {
        if (detectionActive) {
          badge.textContent = 'ACTIVO';
          badge.style.color = 'var(--ok,#00e676)';
        }
      }, 3000);
    }
  }
  updateControlsState();
  setTimeout(()=>startWebcam(),500);

  console.log('%cSSIP v2.1 · Multi-zona · Bolsillos · Brazos cruzados','color:#00d4ff;font-weight:bold');
}