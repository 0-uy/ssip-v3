/**
 * camera-manager.js — SSIP v4.0
 *
 * FILOSOFIA: el usuario elige la camara, el sistema la conecta.
 * Sin cambios automaticos entre camaras. Si falla, mensaje claro.
 * El usuario puede cambiar, desconectar y reconectar libremente.
 *
 * FUENTES: webcam USB/integrada/virtual, IP WebRTC/WHEP/MJPEG/HLS, archivo.
 */

const PREF_KEY = 'ssip_cam_v4';
function _p()   { try { return JSON.parse(sessionStorage.getItem(PREF_KEY)||'{}'); } catch { return {}; } }
function _sp(o) { try { sessionStorage.setItem(PREF_KEY, JSON.stringify(o)); } catch {} }
export function savePref(slot, id) { const o=_p(); o[slot]=id; _sp(o); }
export function loadPref(slot)     { return _p()[slot]||null; }

const _live = new Set();
let _ub = false;
function _bindUnload() {
  if (_ub) return; _ub = true;
  const fn = () => { _live.forEach(s=>s.getTracks().forEach(t=>t.stop())); _live.clear(); };
  window.addEventListener('pagehide', fn);
  window.addEventListener('beforeunload', fn);
}

export async function listCameras() {
  try { const t=await navigator.mediaDevices.getUserMedia({video:true,audio:false}); t.getTracks().forEach(t=>t.stop()); } catch {}
  try { return (await navigator.mediaDevices.enumerateDevices()).filter(d=>d.kind==='videoinput'); } catch { return []; }
}

export function pickCamera(cams, slot='default') {
  if (!cams.length) return null;
  const saved = loadPref(slot);
  if (saved) { const f=cams.find(c=>c.deviceId===saved); if (f) return f; }
  const ext = cams.find(c=>c.label&&!/integrated|facetime|built.?in|internal|isight/i.test(c.label));
  return ext||cams[cams.length-1];
}

export class DeviceSelector {
  constructor(el, onChange) { this.el=el; this.onChange=onChange; this._slot='default'; this._bound=false; }

  async populate(slot='default') {
    this._slot=slot;
    const cams = await listCameras();
    if (!cams.length) { this.el.innerHTML='<option value="">Sin camaras detectadas</option>'; this.el.disabled=true; return null; }
    this.el.disabled=false;
    this.el.innerHTML=cams.map((c,i)=>`<option value="${c.deviceId}">${c.label||`Camara ${i+1}`}</option>`).join('');
    const pref=pickCamera(cams,slot);
    if (pref) this.el.value=pref.deviceId;
    if (!this._bound) {
      this._bound=true;
      this.el.addEventListener('change',()=>{ savePref(this._slot,this.el.value); this.onChange(this.el.value); });
    }
    navigator.mediaDevices.ondevicechange=async()=>{
      const cur=this.el.value;
      const fresh=await listCameras();
      this.el.innerHTML=fresh.map((c,i)=>`<option value="${c.deviceId}">${c.label||`Camara ${i+1}`}</option>`).join('');
      if (fresh.find(c=>c.deviceId===cur)) this.el.value=cur;
      else if (fresh.length) this.onChange('__disconnected__');
    };
    return pref?.deviceId||null;
  }
}

export class CameraSource {
  constructor(videoEl, canvasEl, wrapEl, cbs={}) {
    this.video=videoEl; this.canvas=canvasEl; this.wrap=wrapEl;
    this.onReady  =cbs.onReady  ||(()=>{}); this.onError  =cbs.onError  ||(()=>{});
    this.onStopped=cbs.onStopped||(()=>{});
    this._stream=null; this._pc=null; this._hls=null; this._mjpeg=null;
    this._type=null; this._stopping=false;
    _bindUnload();
    this._ro=new ResizeObserver(()=>this._resize());
    this._ro.observe(wrapEl);
  }

  get isReady() { return this._type!==null; }
  get type()    { return this._type; }

  _resize() {
    const r=this.wrap.getBoundingClientRect();
    if (r.width>0&&r.height>0) { this.canvas.width=r.width; this.canvas.height=r.height; }
  }
  _show() { this.video.style.display=''; this.video.classList.remove('hidden'); }
  _hide() { this.video.style.display='none'; this.video.classList.add('hidden'); }

  _waitReady(ms=9000) {
    const v=this.video;
    if (v.readyState>=2) return Promise.resolve();
    return new Promise((res,rej)=>{
      const tid=setTimeout(()=>{ cleanup(); (v.srcObject||v.src)?res():rej(new Error('timeout')); },ms);
      const ok=()=>{ cleanup(); res(); };
      const bad=()=>{ cleanup(); rej(new Error('error')); };
      function cleanup() {
        clearTimeout(tid);
        v.removeEventListener('loadedmetadata',ok); v.removeEventListener('canplay',ok);
        v.removeEventListener('playing',ok); v.removeEventListener('error',bad);
      }
      v.addEventListener('loadedmetadata',ok); v.addEventListener('canplay',ok);
      v.addEventListener('playing',ok); v.addEventListener('error',bad);
    });
    
  }

  async startWebcam(deviceId=null, slot='default') {
    this._stopping=false; this.stop();
    // CRITICO: visible ANTES de asignar srcObject (fix Chrome display:none bug)
    this._show();
    const sets=[
      deviceId?{video:{deviceId:{exact:deviceId},width:{ideal:1280},height:{ideal:720},frameRate:{ideal:30}},audio:false}
              :{video:{width:{ideal:1280},height:{ideal:720}},audio:false},
      deviceId?{video:{deviceId:{exact:deviceId}},audio:false}:{video:true,audio:false},
    ];
    let stream=null, lastErr=null;
    for (const c of sets) {
      try { stream=await navigator.mediaDevices.getUserMedia(c); break; }
      catch(e) { lastErr=e; if (['NotAllowedError','NotFoundError','NotReadableError'].includes(e.name)) break; }
    }
    if (!stream) { this._hide(); this.onError(this._camErr(lastErr)); return; }
    try {
      this._stream=stream; _live.add(stream);
      this.video.srcObject=stream;
      this.video.play().catch(()=>{});
      await this._waitReady();
      this._type='webcam';
      this._resize(); requestAnimationFrame(()=>this._resize());
      const track=stream.getVideoTracks()[0];
      const label=track?.label||'Webcam';
      if (deviceId) savePref(slot,deviceId);
      track?.addEventListener('ended',()=>{
        if (this._stopping) return;
        this._cleanup(); this.onStopped();
        this.onError('Camara desconectada. Podes volver a conectarla o elegir otra en el selector.');
      });
      this.onReady(label);
    } catch(e) { this._cleanup(); this.onError(`No se pudo iniciar la camara: ${e.message}`); }
  }

  _camErr(e) {
    if (!e) return 'Error desconocido.';
    const m={
      NotReadableError:     'Esta camara esta siendo usada por otra aplicacion. Cerrar Zoom, Teams, Meet, OBS u otras pestanas y volver a intentar.',
      NotAllowedError:      'Permiso de camara denegado. Hacer clic en el icono de camara en la barra de direccion y seleccionar Permitir.',
      NotFoundError:        'Camara no encontrada. Verificar que este bien conectada.',
      OverconstrainedError: 'La camara no soporta esa configuracion. Intentarlo de nuevo.',
      AbortError:           'Conexion interrumpida. Volver a intentarlo.',
    };
    return m[e.name]||`Error al acceder a la camara (${e.name}): ${e.message}`;
  }

  async startIP(url, protocol='auto') {
    this._stopping=false; this.stop();
    if (!url||!/^https?:\/\//i.test(url)) { this.onError('URL invalida. Debe comenzar con http:// o https://'); return; }
    if (protocol==='auto') {
      if (/\.m3u8/i.test(url))                         protocol='hls';
      else if (/mjpeg|\.cgi|axis-cgi|snapshot|nphMotionJpeg/i.test(url)) protocol='mjpeg';
      else if (/whep/i.test(url))                       protocol='whep';
      else                                               protocol='webrtc';
    }
    if (protocol==='mjpeg') { this._mjpegStart(url); return; }
    if (protocol==='hls')   { this._hlsStart(url);   return; }
    await this._rtcStart(url, protocol==='whep'?'whep':'webrtc');
  }

  async _rtcStart(url, mode) {
    let done=false;
    try {
      const pc=new RTCPeerConnection({iceServers:[{urls:'stun:stun.l.google.com:19302'},{urls:'stun:stun1.l.google.com:19302'}]});
      this._pc=pc;
      const tid=setTimeout(()=>{ if(done)return; pc.close(); this._pc=null;
        this.onError('La camara IP no respondio en 15s. Verificar URL y que la camara este encendida.'); },15000);
      pc.ontrack=async(evt)=>{
        if(done||!evt.streams?.[0])return; done=true; clearTimeout(tid);
        this._show(); this.video.srcObject=evt.streams[0];
        this.video.play().catch(()=>{}); await this._waitReady().catch(()=>{});
        this._type='webrtc'; this._resize(); requestAnimationFrame(()=>this._resize());
        this.onReady('IP Camera');
      };
      pc.oniceconnectionstatechange=()=>{
        if(done)return;
        if(pc.iceConnectionState==='failed'){clearTimeout(tid); this.onError('Conexion WebRTC fallida. Verificar que la camara este en la misma red.');}
      };
      pc.addTransceiver('video',{direction:'recvonly'}); pc.addTransceiver('audio',{direction:'recvonly'});
      const offer=await pc.createOffer(); await pc.setLocalDescription(offer);
      if (mode==='whep') {
        const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/sdp'},body:offer.sdp});
        if(!r.ok) throw new Error(`HTTP ${r.status}`);
        await pc.setRemoteDescription({type:'answer',sdp:await r.text()});
      } else {
        const r=await fetch(url,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:new URLSearchParams({data:btoa(offer.sdp)})});
        if(!r.ok) throw new Error(`HTTP ${r.status}`);
        const body=await r.text(); let sdp;
        try { const j=JSON.parse(body); sdp=j.answer??j.sdp??body; } catch { sdp=body; }
        await pc.setRemoteDescription({type:'answer',sdp:atob(sdp)});
      }
    } catch(e) { if(!done) this.onError(`Error conectando camara IP: ${e.message}`); }
  }

  _mjpegStart(url) {
    this._hide();
    let img=this.wrap.querySelector('.ssip-mjpeg');
    if (!img) {
      img=document.createElement('img'); img.className='ssip-mjpeg';
      img.style.cssText='position:absolute;inset:0;width:100%;height:100%;object-fit:contain;background:#000;z-index:1;display:none;';
      this.wrap.appendChild(img);
    }
    this._mjpeg=img; img.style.display='';
    img.src=url+(url.includes('?')?'&':'?')+'_t='+Date.now();
    img.onload =()=>{ this._type='mjpeg'; this._resize(); this.onReady('IP Camera (MJPEG)'); };
    img.onerror=()=>this.onError('No se pudo conectar al stream MJPEG. Verificar URL y credenciales.');
  }

  _hlsStart(url) {
    this._show();
    if (this.video.canPlayType('application/vnd.apple.mpegurl')) {
      this.video.src=url; this.video.play().catch(()=>{});
      this._waitReady().then(()=>{ this._type='hls'; this._resize(); this.onReady('IP Camera (HLS)'); })
                       .catch(()=>this.onError('Error reproduciendo HLS.'));
      return;
    }
    const boot=()=>{
      if (!window.Hls?.isSupported()) { this.onError('HLS no soportado. Usar Safari o elegir otro protocolo.'); return; }
      const hls=new window.Hls({enableWorker:false}); this._hls=hls;
      hls.loadSource(url); hls.attachMedia(this.video);
      hls.on(window.Hls.Events.MANIFEST_PARSED,()=>{ this.video.play().catch(()=>{}); this._type='hls'; this._resize(); this.onReady('IP Camera (HLS)'); });
      hls.on(window.Hls.Events.ERROR,(_,d)=>{ if(d.fatal) this.onError(`Error HLS: ${d.details}`); });
    };
    if (window.Hls) { boot(); return; }
    const s=document.createElement('script'); s.src='https://cdn.jsdelivr.net/npm/hls.js@latest/dist/hls.min.js';
    s.onload=boot; s.onerror=()=>this.onError('No se pudo cargar el reproductor HLS.');
    document.head.appendChild(s);
  }

  startFile(file) {
    this._stopping=false; this.stop(); this._show();
    const url=URL.createObjectURL(file);
    const label=file.name.length>32?file.name.slice(0,30)+'...':file.name;
    this.video.srcObject=null; this.video.src=url; this.video.loop=true;
    this.video.addEventListener('loadedmetadata',()=>{ this.video.play().catch(()=>{}); this._type='file'; this._resize(); this.onReady(label); },{once:true});
    this.video.onerror=()=>{ URL.revokeObjectURL(url); this.onError(`No se pudo reproducir "${file.name}". Formato no soportado.`); };
  }

  stop() {
    this._stopping=true;
    const was=this.isReady;
    this._cleanup();
    if (was) this.onStopped();
  }

  _cleanup() {
    if (this._stream) { this._stream.getTracks().forEach(t=>t.stop()); _live.delete(this._stream); this._stream=null; }
    if (this._pc)     { this._pc.close(); this._pc=null; }
    if (this._hls)    { this._hls.destroy(); this._hls=null; }
    if (this._mjpeg)  { this._mjpeg.src=''; this._mjpeg.style.display='none'; this._mjpeg=null; }
    if (this.video.src?.startsWith('blob:')) URL.revokeObjectURL(this.video.src);
    this.video.srcObject=null; this.video.src=''; this.video.loop=false;
    this._hide(); this._type=null;
  }

  destroy() {
    this._cleanup();
    if (this._ro) { this._ro.disconnect(); this._ro=null; }
    try { navigator.mediaDevices.ondevicechange=null; } catch {}
  }
}