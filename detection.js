/**
 * detection.js — SSIP v5.0
 * ═══════════════════════════════════════════════════════════════════════════
 * Modelos: yolov8n-pose.onnx (17 kp, hasta 8 personas)
 *          yolov8n.onnx       (80 clases COCO, hasta 20 objetos)
 *
 * ═══ NOVEDADES v5.0 ════════════════════════════════════════════════════════
 *
 *  [MULTI-STORE]   Perfiles por tipo de local (farmacia, joyería, supermercado…)
 *                  Cada perfil ajusta umbrales, objetos vigilados y sensibilidad.
 *                  import { getProfile } from './store-profiles.js'
 *
 *  [EMPLOYEE]      Distinción empleado / cliente:
 *                  · Manual:  engine.markEmployee(trackId)
 *                  · Auto:    track >5 min sin actividad sospechosa → empleado silencioso
 *                  Empleados reciben: umbral dwell x3, O1/O2/K/D deshabilitados,
 *                  no suman score, badge verde 👷.
 *
 *  [YOLO-AGNOSTIC] ObjTracker rastrea objetos por posición, no por label.
 *                  Elimina falsos positivos por label-flipping frame a frame
 *                  (taza→botella→taza). Alertas usan "OBJETO PEQUEÑO", no "taza".
 *                  Conf mínima por familia de objetos.
 *
 *  [ACCOMPLICE]    Análisis grupal cada 2 frames:
 *     N — DISTRACTOR: Persona B cerca de mostrador/caja MIENTRAS A tiene postContact.
 *     W — PANTALLA HUMANA: Persona B se interpone entre cámara y Persona A en zona.
 *
 *  [TRAYECTORIA T] Persona que llega al frame y en <1.5s ya tiene mano en zona.
 *
 *  [SCORE v2]      Decay inteligente. Si sale del frame con score >50 → alerta
 *                  "SOSPECHOSO SALIÓ". Si vuelve, hereda 60% del score anterior.
 *
 * ═══ COMPORTAMIENTOS ══════════════════════════════════════════════════════
 *  Z1  Mano en zona (debounce N frames)     LOW
 *  Z2  Permanencia en zona                  HIGH
 *  Z3  Mano escapa al torso desde zona      HIGH
 *  P1  Mano en bolsillo                     HIGH
 *  P2  Brazos cruzados + muñeca oculta      HIGH
 *  O1  Contacto objeto en zona (400ms+)     LOW   ← ya no HIGH inmediato
 *  O2  Objeto tomado (desaparición estable) HIGH
 *  A   Caja → bolsillo/manga                HIGH
 *  B   Bajo manga (post-contacto)           HIGH
 *  C   Bag stuffing                         HIGH
 *  D   Bajo ropa torso ampliado             HIGH
 *  E   Merodeo (3+ accesos sin compra)      MEDIUM
 *  F   Traspaso entre personas              HIGH
 *  G   Arrebato rápido (300-700ms)          HIGH
 *  H   Escaneo previo (cabeza gira)         MEDIUM
 *  I   Cuerpo como pantalla                 HIGH
 *  J   Agacharse y ocultar                  HIGH
 *  K   Cadera / bermuda (post-contacto)     HIGH
 *  N   Distractor (cómplice en mostrador)   HIGH
 *  S   Robo confirmado — score              HIGH
 *  T   Trayectoria directa a zona           LOW
 *  W   Pantalla humana (cómplice bloquea)   HIGH
 */

import { getProfile, getFamily, BAG_IDS, ALERT_IDS } from './store-profiles.js';

// ─────────────────────────────────────────────────────────────────────────────
//  CONSTANTES
// ─────────────────────────────────────────────────────────────────────────────
const POSE_MODEL = './yolov8n-pose.onnx';
const OBJ_MODEL  = './yolov8n.onnx';
const INPUT_W    = 640;
const INPUT_H    = 640;
const CONF_POSE  = 0.30;
const CONF_OBJ   = 0.35;
const KP_THRESH  = 0.25;
const IOU_THRESH = 0.45;
const OBJ_VIS_WINDOW   = 14;   // frames de historial de visibilidad
const SAME_OBJ_IOU     = 0.28; // IOU para considerar mismo objeto aunque cambie label
const MIN_BROWSE_MS    = 1500; // tiempo mínimo de "browsing" antes de contactar zona
const AUTO_EMPLOYEE_MIN = 5;   // minutos para auto-detectar empleado
const SCREEN_MAX_DIST   = 0.35;
const DISTRACTOR_PAY_DIST = 0.30;
const EXIT_SCORE_MEMORY_MS = 30000;

const KP = {
  NOSE:0, L_EYE:1, R_EYE:2, L_EAR:3, R_EAR:4,
  L_SHOULDER:5, R_SHOULDER:6, L_ELBOW:7, R_ELBOW:8,
  L_WRIST:9, R_WRIST:10, L_HIP:11, R_HIP:12,
  L_KNEE:13, R_KNEE:14, L_ANKLE:15, R_ANKLE:16,
};

const BONES = [
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],
];

let _poseSession = null;
let _objSession  = null;
let _posePromise = null;
let _objPromise  = null;

const _ok  = p => p && p.c >= KP_THRESH;
const _d   = (ax, ay, bx, by) => Math.hypot(ax - bx, ay - by);
const _mid = (a, b) => ({ x: (a.x + b.x) / 2, y: (a.y + b.y) / 2 });

// ─────────────────────────────────────────────────────────────────────────────
//  ObjTracker — rastrea objetos por posición (agnóstico al label de YOLO)
// ─────────────────────────────────────────────────────────────────────────────
class ObjTracker {
  constructor() {
    this._objs   = {};   // { stableId: obj }
    this._nextId = 0;
  }

  update(dets) {
    const matched = new Set();

    for (const [id, obj] of Object.entries(this._objs)) {
      let bestIou = SAME_OBJ_IOU;
      let bestDet = null, bestIdx = -1;
      for (let i = 0; i < dets.length; i++) {
        if (matched.has(i)) continue;
        const iou = this._iou(obj.bbox, dets[i]);
        // Misma familia O IOU alto: mismo objeto aunque label cambie
        const sameFam = dets[i].family?.key === obj.family?.key;
        if (iou > bestIou || (sameFam && iou > 0.15)) {
          bestIou = iou; bestDet = dets[i]; bestIdx = i;
        }
      }
      obj.history.push(bestDet !== null);
      if (obj.history.length > OBJ_VIS_WINDOW) obj.history.shift();
      if (bestDet) {
        obj.bbox   = { nx1: bestDet.nx1, ny1: bestDet.ny1, nx2: bestDet.nx2, ny2: bestDet.ny2 };
        obj.cls    = bestDet.cls;
        obj.label  = bestDet.label;
        obj.family = bestDet.family;
        obj.conf   = bestDet.conf;
        obj.visible = true;
        obj.lastSeen = Date.now();
        matched.add(bestIdx);
      } else {
        obj.visible = false;
      }
    }

    for (let i = 0; i < dets.length; i++) {
      if (matched.has(i)) continue;
      const d = dets[i];
      const id = `o${this._nextId++}`;
      this._objs[id] = {
        id, cls: d.cls, family: d.family, label: d.label, conf: d.conf,
        bbox: { nx1: d.nx1, ny1: d.ny1, nx2: d.nx2, ny2: d.ny2 },
        history: [true], visible: true, lastSeen: Date.now(), contactStart: null,
      };
    }

    // Limpiar objetos no vistos en >5s
    for (const id of Object.keys(this._objs))
      if (Date.now() - this._objs[id].lastSeen > 5000) delete this._objs[id];
  }

  get visible()      { return Object.values(this._objs).filter(o => o.visible); }
  get alertVisible() { return this.visible.filter(o => o.family && ALERT_IDS.has(o.cls)); }

  disappearedAfterContact(objId) {
    const obj = this._objs[objId];
    if (!obj || obj.history.length < 6) return false;
    const half   = Math.floor(obj.history.length / 2);
    const before = obj.history.slice(0, half);
    const after  = obj.history.slice(half);
    const visBefore = before.filter(Boolean).length / before.length;
    const absAfter  = after.filter(v => !v).length  / after.length;
    return visBefore >= 0.60 && absAfter >= 0.60;
  }

  markContact(objId) {
    const obj = this._objs[objId];
    if (obj) {
      obj.contactStart = Date.now();
      obj.history = new Array(Math.floor(OBJ_VIS_WINDOW * 0.7)).fill(true);
    }
  }

  _iou(a, b) {
    const ix1 = Math.max(a.nx1, b.nx1), iy1 = Math.max(a.ny1, b.ny1);
    const ix2 = Math.min(a.nx2, b.nx2), iy2 = Math.min(a.ny2, b.ny2);
    const I = Math.max(0, ix2-ix1) * Math.max(0, iy2-iy1);
    return I / ((a.nx2-a.nx1)*(a.ny2-a.ny1) + (b.nx2-b.nx1)*(b.ny2-b.ny1) - I + 1e-6);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
//  DetectionEngine
// ─────────────────────────────────────────────────────────────────────────────
export class DetectionEngine {
  constructor(canvas, zoneManager, alertManager, config = {}) {
    this.canvas       = canvas;
    this.ctx          = canvas.getContext('2d');
    this.zoneManager  = zoneManager;
    this.alertManager = alertManager;
    this._profile     = getProfile(config.storeType || 'generico');
    this.config = {
      movementThreshold: config.movementThreshold ?? 50,
      dwellTime:         config.dwellTime         ?? this._profile.dwellTime,
      cooldown:          config.cooldown          ?? 8,
      storeType:         config.storeType         ?? 'generico',
    };
    this.active        = false;
    this._off          = document.createElement('canvas');
    this._off.width    = INPUT_W;
    this._off.height   = INPUT_H;
    this._offCtx       = this._off.getContext('2d', { willReadFrequently: true });
    this._tracks       = [];
    this._nextId       = 0;
    this._maxHistory   = 30;
    this._objDets      = [];
    this._objTracker   = new ObjTracker();
    this._interactions = {};
    this._lastAlert    = {};
    this._fpsFrames    = 0;
    this._fpsLast      = performance.now();
    this.currentFPS    = 0;
    this._renderLoopId = null;
    this._lastDets     = [];
    this.onDetection   = null;
    this._employeeIds  = new Set();
    this._exitScores   = [];
    console.log(`%c✓ SSIP v5.0 — ${this._profile.icon} ${this._profile.name}`, 'color:#00d4ff;font-weight:bold');
  }

  // ── API pública ─────────────────────────────────────────────────────────────
  markEmployee(trackId) {
    this._employeeIds.add(trackId);
    const t = this._tracks.find(t => t.id === trackId);
    if (t) { t.isEmployee = true; t.suspicionScore = 0; t.badges = []; }
  }
  markCustomer(trackId) {
    this._employeeIds.delete(trackId);
    const t = this._tracks.find(t => t.id === trackId);
    if (t) t.isEmployee = false;
  }
  getTracks() {
    return this._tracks.map(t => ({
      id: t.id, isEmployee: t.isEmployee,
      score: Math.round(t.suspicionScore),
      bbox: { nx1: t.nx1, ny1: t.ny1, nx2: t.nx2, ny2: t.ny2 },
    }));
  }
  setStoreType(type) {
    this._profile = getProfile(type);
    this.config.storeType = type;
    console.log(`%c🏪 Perfil: ${this._profile.icon} ${this._profile.name}`, 'color:#00d4ff');
    return this._profile;  // permite sincronizar sliders en app.js
  }

  // ── Init ────────────────────────────────────────────────────────────────────
  async init() {
    if (!_posePromise) _posePromise = this._loadModel(POSE_MODEL, 'pose');
    if (!_objPromise)  _objPromise  = this._loadModel(OBJ_MODEL,  'obj');
    const [pR, oR] = await Promise.allSettled([_posePromise, _objPromise]);
    if (pR.status === 'rejected') throw new Error('No se pudo cargar yolov8n-pose.onnx');
    if (oR.status === 'rejected') console.warn('%c⚠ yolov8n.onnx no disponible', 'color:#ffaa00');
    this._startRenderLoop();
  }
  async _loadModel(path, name) {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime no cargado');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    for (const ep of ['webgl', 'wasm']) {
      try {
        const s = await ort.InferenceSession.create(path, { executionProviders: [ep], graphOptimizationLevel: 'all' });
        if (name === 'pose') _poseSession = s; else _objSession = s;
        console.log(`%c✓ YOLOv8n-${name} (${ep.toUpperCase()})`, 'color:#00e676;font-weight:bold');
        return;
      } catch(e) { console.warn(`ONNX [${name}/${ep}]:`, e.message); }
    }
    throw new Error(`No se pudo cargar ${path}`);
  }
  _startRenderLoop() {
    const loop = () => {
      if (!this.active) {
        this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
        this.zoneManager.drawZone(false);
        this.zoneManager.drawPreview();
        if (this._lastDets.length) this._drawDetections(this._lastDets);
      }
      this._renderLoopId = requestAnimationFrame(loop);
    };
    this._renderLoopId = requestAnimationFrame(loop);
  }

  // ── Pipeline ────────────────────────────────────────────────────────────────
  async processFrame(video) {
    if (!this.active || !_poseSession) return;
    this._fpsFrames++;
    const now = performance.now();
    if (now - this._fpsLast >= 1000) { this.currentFPS = this._fpsFrames; this._fpsFrames = 0; this._fpsLast = now; }
    let tensor, meta;
    try { [tensor, meta] = this._preprocess(video); } catch { return; }
    let poseDets = [];
    try {
      const out = await _poseSession.run({ images: tensor });
      poseDets = this._postprocessPose(out.output0 || out[Object.keys(out)[0]], meta);
    } catch(e) { console.warn('Pose:', e.message); }
    let objDets = [];
    if (_objSession) {
      try {
        const out = await _objSession.run({ images: tensor });
        objDets = this._postprocessObj(out.output0 || out[Object.keys(out)[0]], meta);
      } catch(e) { console.warn('Obj:', e.message); }
    }
    if (typeof tensor?.dispose === 'function') tensor.dispose();
    this._objTracker.update(objDets);
    this._objDets  = objDets;
    this._lastDets = poseDets;
    this._updateTracks(poseDets, Date.now());
    if (this._fpsFrames % 2 === 0) this._analyzeGroup(Date.now());
    this._render();
  }

  // ── Pre/post proceso ────────────────────────────────────────────────────────
  _preprocess(video) {
    const vw = video.videoWidth || video.width || 640;
    const vh = video.videoHeight || video.height || 480;
    const scale = Math.min(INPUT_W/vw, INPUT_H/vh);
    const nw = Math.round(vw*scale), nh = Math.round(vh*scale);
    const dx = (INPUT_W-nw)/2, dy = (INPUT_H-nh)/2;
    this._offCtx.fillStyle = '#808080';
    this._offCtx.fillRect(0,0,INPUT_W,INPUT_H);
    this._offCtx.drawImage(video, dx, dy, nw, nh);
    const px = this._offCtx.getImageData(0,0,INPUT_W,INPUT_H).data;
    const N  = INPUT_W*INPUT_H;
    const f32 = new Float32Array(3*N);
    for (let i=0; i<N; i++) {
      f32[i]     = px[i*4]   /255;
      f32[N+i]   = px[i*4+1] /255;
      f32[2*N+i] = px[i*4+2] /255;
    }
    return [new ort.Tensor('float32', f32, [1,3,INPUT_H,INPUT_W]), {dx,dy,scale,vw,vh}];
  }
  _postprocessPose(output, {dx,dy,scale,vw,vh}) {
    const data=output.data, S=output.dims[2], dets=[];
    for (let i=0;i<S;i++) {
      const conf=data[4*S+i]; if (conf<CONF_POSE) continue;
      const cx=data[0*S+i],cy=data[1*S+i],bw=data[2*S+i],bh=data[3*S+i];
      const n=v=>Math.max(0,Math.min(1,v));
      const nx1=n((cx-bw/2-dx)/(vw*scale)),ny1=n((cy-bh/2-dy)/(vh*scale));
      const nx2=n((cx+bw/2-dx)/(vw*scale)),ny2=n((cy+bh/2-dy)/(vh*scale));
      const kps=[];
      for (let k=0;k<17;k++) kps.push({
        x:n((data[(5+k*3)*S+i]-dx)/(vw*scale)),
        y:n((data[(5+k*3+1)*S+i]-dy)/(vh*scale)),
        c:data[(5+k*3+2)*S+i],
      });
      dets.push({conf,kps,nx1,ny1,nx2,ny2});
    }
    return this._nms(dets).slice(0,8);
  }
  _postprocessObj(output, {dx,dy,scale,vw,vh}) {
    const data=output.data, S=output.dims[2], dets=[];
    for (let i=0;i<S;i++) {
      let bestCls=-1, bestConf=CONF_OBJ;
      for (let c=0;c<80;c++) { const sc=data[(4+c)*S+i]; if (sc>bestConf){bestConf=sc;bestCls=c;} }
      if (bestCls<0) continue;
      const family=getFamily(bestCls);
      if (!family||bestConf<family.minConf) continue;
      const cx=data[0*S+i],cy=data[1*S+i],bw=data[2*S+i],bh=data[3*S+i];
      const n=v=>Math.max(0,Math.min(1,v));
      dets.push({cls:bestCls,conf:bestConf,label:family.label,family,
        nx1:n((cx-bw/2-dx)/(vw*scale)),ny1:n((cy-bh/2-dy)/(vh*scale)),
        nx2:n((cx+bw/2-dx)/(vw*scale)),ny2:n((cy+bh/2-dy)/(vh*scale)),
      });
    }
    return this._nms(dets).slice(0,20);
  }
  _nms(dets) {
    if (!dets.length) return [];
    dets.sort((a,b)=>b.conf-a.conf);
    const keep=[], drop=new Set();
    for (let i=0;i<dets.length;i++) {
      if (drop.has(i)) continue; keep.push(dets[i]);
      for (let j=i+1;j<dets.length;j++) if (!drop.has(j)&&this._iou(dets[i],dets[j])>IOU_THRESH) drop.add(j);
    }
    return keep;
  }
  _iou(a,b) {
    const ix1=Math.max(a.nx1,b.nx1),iy1=Math.max(a.ny1,b.ny1);
    const ix2=Math.min(a.nx2,b.nx2),iy2=Math.min(a.ny2,b.ny2);
    const I=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
    return I/((a.nx2-a.nx1)*(a.ny2-a.ny1)+(b.nx2-b.nx1)*(b.ny2-b.ny1)-I+1e-6);
  }

  // ── Tracking ────────────────────────────────────────────────────────────────
  _makeTrack(d, now) {
    return {
      id:this._nextId++, kps:d.kps, nx1:d.nx1,ny1:d.ny1,nx2:d.nx2,ny2:d.ny2,
      missed:0, history:[{kps:d.kps,t:now}], firstSeen:now,
      isEmployee:false, staffZoneTime:0,
      inZoneWrist:{}, dwellStart:{}, zoneEntryFrames:{},
      pocketL:0, pocketR:0, crossedArms:0,
      cajaExit:{}, postContact:null,
      zoneVisits:{}, visitedPay:false,
      noseXHist:[], bodyScreen:0, crouchHide:0, hipConcealment:0,
      directTrajFired:false, firstZoneEntry:null,
      suspicionScore:0, scoreEvidence:[], badges:[],
    };
  }
  _updateTracks(dets, now) {
    const matched=new Set();
    for (const t of this._tracks) {
      let best=-1, bestIou=0.10;
      for (let i=0;i<dets.length;i++) {
        if (matched.has(i)) continue;
        const iou=this._iou(t,dets[i]);
        if (iou>bestIou){best=i;bestIou=iou;}
      }
      if (best>=0) {
        const d=dets[best];
        Object.assign(t,{kps:d.kps,nx1:d.nx1,ny1:d.ny1,nx2:d.nx2,ny2:d.ny2,missed:0});
        t.history.push({kps:d.kps,t:now});
        if (t.history.length>this._maxHistory) t.history.shift();
        matched.add(best);
      } else { t.missed=(t.missed||0)+1; }
    }
    // Tracks que desaparecen con score alto → alerta
    for (const t of this._tracks.filter(t=>(t.missed||0)>=10)) {
      if (t.suspicionScore>=50&&!t.isEmployee) {
        this._exitScores.push({
          score:t.suspicionScore, evidence:t.scoreEvidence.slice(-3),
          timestamp:now, cx:(t.nx1+t.nx2)/2, cy:(t.ny1+t.ny2)/2,
        });
        this._fire(`exit_${t.id}`,
          `SOSPECHOSO SALIÓ — SCORE ${Math.round(t.suspicionScore)} | ${t.scoreEvidence.slice(-2).join(' + ')}`,
          'medium', 5000);
      }
    }
    this._tracks=this._tracks.filter(t=>(t.missed||0)<10);
    this._exitScores=this._exitScores.filter(e=>now-e.timestamp<EXIT_SCORE_MEMORY_MS);
    for (let i=0;i<dets.length;i++) {
      if (matched.has(i)) continue;
      const nt=this._makeTrack(dets[i],now);
      if (this._employeeIds.has(nt.id)) nt.isEmployee=true;
      // Heredar score si volvió rápido
      const cx=(dets[i].nx1+dets[i].nx2)/2, cy=(dets[i].ny1+dets[i].ny2)/2;
      const prev=this._exitScores.find(e=>_d(cx,cy,e.cx,e.cy)<0.20);
      if (prev) { nt.suspicionScore=prev.score*0.6; nt.scoreEvidence=prev.evidence; }
      this._tracks.push(nt);
    }
    for (const t of this._tracks) if (!t.missed) this._analyze(t,now);
  }

  // ── Análisis por track ──────────────────────────────────────────────────────
  _analyze(t, now) {
    const k=t.kps;
    const lw=k[KP.L_WRIST],rw=k[KP.R_WRIST];
    const lh=k[KP.L_HIP],  rh=k[KP.R_HIP];
    const le=k[KP.L_ELBOW],re=k[KP.R_ELBOW];
    const ls=k[KP.L_SHOULDER],rs=k[KP.R_SHOULDER];
    const nose=k[KP.NOSE];
    t.badges=[];
    this._decayScore(t);
    this._checkAutoEmployee(t,now);
    if (t.isEmployee) { t.badges.push('👷'); this._detectZoneDwellOnly(t,lw,rw,now); return; }
    const P=this._profile;
    this._detectZone(t,lw,rw,lh,rh,now);
    this._detectPocket(t,lw,lh,ls,'L');
    this._detectPocket(t,rw,rh,rs,'R');
    this._detectCrossedArms(t,le,re,lw,rw,ls,rs,lh,rh);
    this._detectHandObj(t,lw,rw,now);
    this._checkCajaHeist(t,lw,rw,lh,rh,le,re,now);
    this._checkPostContact(t,lw,rw,le,re,ls,rs,lh,rh,now);
    if (P.behaviors.cadera)      this._checkHipConcealment(t,lw,rw,lh,rh,now);
    if (P.behaviors.merodeo)     this._checkProwling(t,now);
    if (P.behaviors.escaneo)     this._checkScanBehavior(t,nose,now);
    if (P.behaviors.pantalla)    this._checkBodyScreen(t,nose);
    if (P.behaviors.agachado)    this._checkCrouchHide(t,nose,ls,rs,lh,rh);
    if (P.behaviors.trayectoria) this._checkDirectTrajectory(t,now);
    this._checkSuspicionScore(t,now);
  }

  // ── [EMPLOYEE] Auto-detección ───────────────────────────────────────────────
  _checkAutoEmployee(t, now) {
    if (t.isEmployee||t.suspicionScore>20) return;
    if ((now-t.firstSeen)/60000>=AUTO_EMPLOYEE_MIN&&t.suspicionScore<5) {
      t.isEmployee=true; this._employeeIds.add(t.id);
      console.log(`%c👷 Track #${t.id} auto-empleado`, 'color:#00e676');
    }
  }
  _detectZoneDwellOnly(t,lw,rw,now) {
    for (const [w,side] of [[lw,'L'],[rw,'R']]) {
      if (!_ok(w)) continue;
      const zones=this.zoneManager.getZonesForPoint(w.x,w.y);
      for (const zone of zones) {
        const key=`${side}_${zone.id}`;
        if (!t.dwellStart[key]) t.dwellStart[key]=now;
        if ((now-t.dwellStart[key])/1000>=this.config.dwellTime*3) {
          t.dwellStart[key]=now+this.config.dwellTime*3000;
          this._fire(`emp_dw_${key}`,`EMPLEADO — PERMANENCIA INUSUAL EN ${zone.name.toUpperCase()}`,'medium',30000);
        }
      }
    }
  }

  // ── Z1 Z2 Z3 ────────────────────────────────────────────────────────────────
  _detectZone(t,lw,rw,lh,rh,now) {
    const P=this._profile;
    for (const [w,side] of [[lw,'L'],[rw,'R']]) {
      if (!_ok(w)) {
        for (const key of Object.keys(t.inZoneWrist)) if (key.startsWith(side+'_')) {
          t.inZoneWrist[key]=false; t.dwellStart[key]=null; t.zoneEntryFrames[key]=0;
        }
        continue;
      }
      const zones=this.zoneManager.getZonesForPoint(w.x,w.y);
      for (const zone of zones) {
        const key=`${side}_${zone.id}`;
        t.zoneEntryFrames[key]=(t.zoneEntryFrames[key]||0)+1;
        if (!t.inZoneWrist[key]) {
          if (t.zoneEntryFrames[key]>=P.zoneEntryFrames) {
            t.inZoneWrist[key]=true; t.dwellStart[key]=now;
            zone.alert=true; setTimeout(()=>{if(zone)zone.alert=false;},2000);
            this._fire(`ze_${key}`,`MANO EN ${zone.name.toUpperCase()}`,'low',1500);
            this._recordVisit(t,zone,now);
            if (zone.type==='pago') t.visitedPay=true;
            if (!t.firstZoneEntry) t.firstZoneEntry=now;
          }
        } else {
          const elapsed=(now-(t.dwellStart[key]||now))/1000;
          if (elapsed>=this.config.dwellTime) {
            t.dwellStart[key]=now+this.config.dwellTime*1000;
            this._fire(`dw_${key}`,`PERMANENCIA — ${zone.name.toUpperCase()}`,'high',this.config.cooldown*1000);
          }
          if (t.history.length>=6) this._detectEscape(t,side,zone,lh,rh);
          t.badges.push('⚠ EN ZONA');
        }
      }
      if (zones.length===0) {
        for (const key of Object.keys(t.inZoneWrist)) {
          if (!key.startsWith(side+'_')||!t.inZoneWrist[key]) continue;
          t.inZoneWrist[key]=false; t.dwellStart[key]=null; t.zoneEntryFrames[key]=0;
          const zId=key.slice(2);
          const z=this.zoneManager.zones.find(z=>z.id===zId);
          if (z?.type==='pago'&&_ok(w)) t.cajaExit[`${side}_${zId}`]={t:now,wristY:w.y};
        }
        for (const key of Object.keys(t.zoneEntryFrames))
          if (key.startsWith(side+'_')&&!t.inZoneWrist[key]) t.zoneEntryFrames[key]=0;
      }
    }
  }
  _detectEscape(t,side,zone,lh,rh) {
    if (!lh||!rh) return;
    const mid=_mid(lh,rh), hLen=t.history.length;
    const old=t.history[Math.max(0,hLen-6)], cur=t.history[hLen-1];
    if (!old||!cur) return;
    const idx=side==='L'?KP.L_WRIST:KP.R_WRIST;
    const pw=old.kps[idx], cw=cur.kps[idx];
    if (!_ok(pw)||!_ok(cw)) return;
    if (!this.zoneManager.getZonesForPoint(pw.x,pw.y).some(z=>z.id===zone.id)) return;
    const pd=_d(pw.x,pw.y,mid.x,mid.y), cd=_d(cw.x,cw.y,mid.x,mid.y);
    if (cd<pd*0.65&&pd>0.08)
      this._fire(`esc_${zone.id}_${side}`,`OBJETO OCULTADO — ${zone.name.toUpperCase()}`,'high',this.config.cooldown*1000);
  }

  // ── P1 P2 ───────────────────────────────────────────────────────────────────
  _detectPocket(t,wrist,hip,shoulder,side) {
    if (!hip||!shoulder) return;
    const hx=hip.x,hy=hip.y;
    let pocket=false;
    if (!wrist||wrist.c<0.20)   pocket=true;
    else if (wrist.c<0.50)      pocket=Math.abs(wrist.x-hx)<0.18&&Math.abs(wrist.y-hy)<0.22;
    else pocket=wrist.y>hy-0.05&&Math.abs(wrist.x-hx)<0.15&&Math.abs(wrist.y-hy)<0.19;
    const sk=side==='L'?'pocketL':'pocketR';
    if (pocket) {
      t[sk]++;
      if (t[sk]>=12){t[sk]=0;this._fire(`pkt_${side}_${t.id}`,`MANO ${side==='L'?'IZQ.':'DER.'} EN BOLSILLO`,'high',this.config.cooldown*1000);}
      if (t[sk]>6) t.badges.push('⚠ BOLSILLO');
    } else t[sk]=Math.max(0,t[sk]-2);
  }
  _detectCrossedArms(t,le,re,lw,rw,ls,rs,lh,rh) {
    if (!le||!re||!ls||!rs||!lh||!rh) return;
    const mx=(ls.x+rs.x)/2,my=(ls.y+rs.y)/2,hy=(lh.y+rh.y)/2;
    const ok=Math.abs(le.x-mx)<0.20&&Math.abs(re.x-mx)<0.20&&le.x>mx&&re.x<mx
           &&le.y>my&&le.y<hy+0.08&&re.y>my&&re.y<hy+0.08
           &&((!lw||lw.c<0.40)||(!rw||rw.c<0.40));
    if (ok) {
      t.crossedArms++;
      if (t.crossedArms>=15){t.crossedArms=0;this._fire(`cross_${t.id}`,'BRAZOS CRUZADOS — POSIBLE OCULTAMIENTO','high',this.config.cooldown*1000);}
      if (t.crossedArms>8) t.badges.push('⚠ CRUZADO');
      if (t.postContact&&!t.postContact.fired) this._addScore(t,this._B('brazoscruzados'),'BRAZOS CRUZADOS');
    } else t.crossedArms=Math.max(0,t.crossedArms-2);
  }

  // ── O1 O2 G F ────────────────────────────────────────────────────────────────
  _detectHandObj(t,lw,rw,now) {
    const alertObjs=this._objTracker.alertVisible;
    if (!alertObjs.length) return;
    const enabledFams=new Set(this._profile.families);
    for (const [w,side] of [[lw,'L'],[rw,'R']]) {
      if (!_ok(w)) continue;
      for (const obj of alertObjs) {
        if (!enabledFams.has(obj.family?.key)) continue;
        const m=0.06;
        const touching=w.x>=obj.bbox.nx1-m&&w.x<=obj.bbox.nx2+m&&w.y>=obj.bbox.ny1-m&&w.y<=obj.bbox.ny2+m;
        const intKey=`${t.id}_${obj.id}_${side}`;
        if (touching) {
          if (!this._interactions[intKey]){this._interactions[intKey]={startT:now,objId:obj.id,label:obj.label,cls:obj.cls};this._objTracker.markContact(obj.id);}
          const dur=now-this._interactions[intKey].startT;
          if (dur>=this._profile.contactMinMs) {
            const zones=this.zoneManager.getZonesForPoint((obj.bbox.nx1+obj.bbox.nx2)/2,(obj.bbox.ny1+obj.bbox.ny2)/2);
            if (zones.length>0) {
              this._fire(`oz_${intKey}`,`CONTACTO: ${obj.label} EN ${zones[0].name.toUpperCase()}`,'low',3000);
              this._addScore(t,this._B('contacto'),`CONTACTO ${obj.label}`);
            }
          }
        } else if (this._interactions[intKey]) {
          const d=this._interactions[intKey];
          delete this._interactions[intKey];
          const dur=now-d.startT;
          if (dur<200) continue;
          if (!this._objTracker.disappearedAfterContact(d.objId)) continue;
          const nearby=this._countNearby(t,0.22);
          if (nearby>0&&this._profile.behaviors.traspaso) {
            this._fire(`hof_${t.id}_${obj.id}`,`TRASPASO: ${d.label} (${nearby} persona cerca)`,'high',this.config.cooldown*1000);
            this._addScore(t,this._B('traspaso'),'TRASPASO');
          } else if (dur<=this._profile.grabMaxMs) {
            this._fire(`grab_${t.id}_${obj.id}_${side}`,`ARREBATO: ${d.label}`,'high',this.config.cooldown*1000);
            this._addScore(t,this._B('arrebato'),'ARREBATO');
          } else {
            const zn=this._getObjZone(obj.bbox);
            this._fire(`og_${t.id}_${obj.id}_${side}`,`OBJETO TOMADO${zn}`,'high',this.config.cooldown*1000);
            this._addScore(t,this._B('objetoTomado'),`TOMADO ${d.label}`);
          }
          if (_ok(w)) {
            const elbow=side==='L'?t.kps[KP.L_ELBOW]:t.kps[KP.R_ELBOW];
            t.postContact={disappearT:now,label:d.label,cls:d.cls,side,wristY0:w.y,elbowY0:_ok(elbow)?elbow.y:null,fired:false};
          }
        }
      }
      for (const k of Object.keys(this._interactions))
        if (k.startsWith(`${t.id}_`)&&now-this._interactions[k].startT>8000) delete this._interactions[k];
    }
  }
  _getObjZone(bbox) {
    const cx=(bbox.nx1+bbox.nx2)/2, cy=(bbox.ny1+bbox.ny2)/2;
    const z=this.zoneManager.getZonesForPoint(cx,cy);
    return z.length>0?` EN ${z[0].name.toUpperCase()}`:'';
  }
  _countNearby(t,maxDist) {
    const cx=(t.nx1+t.nx2)/2,cy=(t.ny1+t.ny2)/2;
    let n=0;
    for (const o of this._tracks) if (o.id!==t.id&&!o.missed&&_d(cx,cy,(o.nx1+o.nx2)/2,(o.ny1+o.ny2)/2)<maxDist) n++;
    return n;
  }

  // ── A Caja heist ─────────────────────────────────────────────────────────────
  _checkCajaHeist(t,lw,rw,lh,rh,le,re,now) {
    for (const [w,elbow,hip,side] of [[lw,le,lh,'L'],[rw,re,rh,'R']]) {
      for (const [key,state] of Object.entries(t.cajaExit)) {
        if (!key.startsWith(side+'_')) continue;
        if (now-state.t>2000){delete t.cajaExit[key];continue;}
        if (!_ok(w)) continue;
        if (_ok(hip)&&w.y>state.wristY+0.06&&Math.abs(w.x-hip.x)<0.15&&Math.abs(w.y-hip.y)<0.18) {
          this._fire(`cj_pkt_${key}`,'CAJA → BOLSILLO: POSIBLE EXTRACCIÓN','high',this.config.cooldown*1000);
          delete t.cajaExit[key]; continue;
        }
        if (_ok(elbow)&&w.y<state.wristY-0.07&&w.y<elbow.y-0.04) {
          this._fire(`cj_slv_${key}`,'CAJA → MANGA: POSIBLE EXTRACCIÓN','high',this.config.cooldown*1000);
          delete t.cajaExit[key]; continue;
        }
      }
    }
  }

  // ── B C D post-contact ────────────────────────────────────────────────────────
  _checkPostContact(t,lw,rw,le,re,ls,rs,lh,rh,now) {
    const pc=t.postContact; if (!pc||pc.fired) return;
    if (now-pc.disappearT>this._profile.postContactMs){t.postContact=null;return;}
    const w=pc.side==='L'?lw:rw, elbow=pc.side==='L'?le:re;
    if (!_ok(w)) return;
    const hcc=this._profile.hipConcealConf??0.55;
    if (w.c<hcc) this._addScore(t,20,'WRIST OCULTA');
    // [B] MANGA
    if (this._profile.behaviors.manga&&pc.elbowY0!==null&&_ok(elbow)) {
      if (w.y<pc.wristY0-0.07&&w.y<elbow.y-0.04) {
        this._fire(`slv_${t.id}_${pc.cls}`,`MANGA — ${pc.label} BAJO MANGA`,'high',this.config.cooldown*1000);
        this._addScore(t,this._B('manga'),'BAJO MANGA'); pc.fired=true; t.postContact=null; return;
      }
    }
    // [C] BAG STUFFING
    if (this._profile.behaviors.bagStuffing) {
      const nearBag=this._objTracker.visible.find(o=>BAG_IDS.has(o.cls)&&_d(w.x,w.y,(o.bbox.nx1+o.bbox.nx2)/2,(o.bbox.ny1+o.bbox.ny2)/2)<0.14);
      if (nearBag) {
        this._fire(`bag_${t.id}_${pc.cls}`,`BOLSO — ${pc.label} EN BOLSO`,'high',this.config.cooldown*1000);
        this._addScore(t,this._B('bagStuffing'),'BAG STUFFING'); pc.fired=true; t.postContact=null; return;
      }
    }
    // [D] BAJO ROPA — zona ampliada
    if (_ok(ls)&&_ok(rs)&&_ok(lh)&&_ok(rh)) {
      const bL=Math.min(ls.x,rs.x,lh.x,rh.x), bR=Math.max(ls.x,rs.x,lh.x,rh.x), bw=(bR-bL);
      const tx1=bL-bw*0.15, tx2=bR+bw*0.15;
      const ty1=Math.min(ls.y,rs.y), ty2=Math.max(lh.y,rh.y)+0.12;
      if (w.x>tx1&&w.x<tx2&&w.y>ty1&&w.y<ty2&&w.c<hcc) {
        this._fire(`trso_${t.id}_${pc.cls}`,`ROPA — ${pc.label} BAJO ROPA`,'high',this.config.cooldown*1000);
        this._addScore(t,this._B('bajoropa'),'BAJO ROPA'); pc.fired=true; t.postContact=null; return;
      }
    }
  }

  // ── K Cadera/bermuda ──────────────────────────────────────────────────────────
  _checkHipConcealment(t,lw,rw,lh,rh,now) {
    const pc=t.postContact; if (!pc||pc.fired) return;
    if (now-pc.disappearT>this._profile.postContactMs) return;
    const w=pc.side==='L'?lw:rw, hip=pc.side==='L'?lh:rh;
    if (!_ok(w)||!_ok(hip)) return;
    const nearHip=_d(w.x,w.y,hip.x,hip.y)<0.22;
    const atLevel=w.y>=hip.y-0.08&&w.y<=hip.y+0.20;
    const moved=pc.wristY0!==undefined?Math.abs(w.y-pc.wristY0)>0.06:true;
    if (nearHip&&atLevel&&moved) {
      t.hipConcealment++;
      this._addScore(t,5,`WRIST CADERA ${pc.side}`);
      if (t.hipConcealment>=5) {
        t.hipConcealment=0;
        const cl=w.c<(this._profile.hipConcealConf??0.55)?'MANO OCULTA':'MANO VISIBLE';
        this._fire(`hip_${t.id}_${pc.cls}`,`CADERA ${pc.side==='L'?'IZQ':'DER'} — ${pc.label} (${cl})`,'high',this.config.cooldown*1000);
        this._addScore(t,this._B('cadera'),'CADERA'); pc.fired=true; t.postContact=null; t.badges.push('⚠ CADERA');
      } else if (t.hipConcealment>2) t.badges.push('⚠ CADERA');
    } else t.hipConcealment=Math.max(0,t.hipConcealment-1);
  }

  // ── E Merodeo ─────────────────────────────────────────────────────────────────
  _recordVisit(t,zone,now) {
    if (zone.type==='pago') return;
    if (!t.zoneVisits[zone.id]) t.zoneVisits[zone.id]=[];
    t.zoneVisits[zone.id].push(now);
    t.zoneVisits[zone.id]=t.zoneVisits[zone.id].filter(ts=>now-ts<90000);
  }
  _checkProwling(t,now) {
    for (const [zId,tss] of Object.entries(t.zoneVisits)) {
      if (tss.length<3||t.visitedPay) continue;
      const z=this.zoneManager.zones.find(z=>z.id===zId);
      this._fire(`prl_${t.id}_${zId}`,`MERODEO — ${tss.length} ACCESOS SIN COMPRA EN ${z?.name?.toUpperCase()||'ZONA'}`,'medium',this.config.cooldown*1500);
      this._addScore(t,this._B('merodeo'),'MERODEO'); t.badges.push('⚠ MERODEO');
    }
  }

  // ── H Escaneo ────────────────────────────────────────────────────────────────
  _checkScanBehavior(t,nose,now) {
    if (!_ok(nose)) return;
    t.noseXHist.push({x:nose.x,t:now});
    t.noseXHist=t.noseXHist.filter(p=>now-p.t<1500);
    if (t.noseXHist.length<6) return;
    const xs=t.noseXHist.map(p=>p.x), mean=xs.reduce((a,b)=>a+b,0)/xs.length;
    const std=Math.sqrt(xs.reduce((a,x)=>a+(x-mean)**2,0)/xs.length);
    if (std<0.06) return;
    const inZone=Object.values(t.inZoneWrist).some(v=>v);
    const cx=(t.nx1+t.nx2)/2,cy=(t.ny1+t.ny2)/2;
    const nearObj=this._objTracker.alertVisible.some(o=>_d(cx,cy,(o.bbox.nx1+o.bbox.nx2)/2,(o.bbox.ny1+o.bbox.ny2)/2)<0.30);
    if (!inZone&&!nearObj) return;
    this._fire(`scan_${t.id}`,'ESCANEO — COMPORTAMIENTO PREVIO A HURTO','medium',this.config.cooldown*1000);
    this._addScore(t,this._B('escaneo'),'ESCANEO'); t.badges.push('⚠ ESCANEO'); t.noseXHist=[];
  }

  // ── I Pantalla ─────────────────────────────────────────────────────────────
  _checkBodyScreen(t,nose) {
    const nH=!nose||nose.c<KP_THRESH, wZ=Object.values(t.inZoneWrist).some(v=>v);
    if (nH&&wZ) {
      t.bodyScreen++;
      if (t.bodyScreen>=10){t.bodyScreen=0;this._fire(`bsc_${t.id}`,'CUERPO COMO PANTALLA — DE ESPALDAS EN ZONA','high',this.config.cooldown*1000);}
      if (t.bodyScreen>5){t.badges.push('⚠ PANTALLA');this._addScore(t,this._B('pantalla'),'PANTALLA');}
    } else t.bodyScreen=Math.max(0,t.bodyScreen-2);
  }

  // ── J Agachado ────────────────────────────────────────────────────────────────
  _checkCrouchHide(t,nose,ls,rs,lh,rh) {
    if (!t.postContact||t.postContact.fired||!_ok(nose)||!_ok(ls)||!_ok(rs)) return;
    const sY=(ls.y+rs.y)/2, hY=_ok(lh)&&_ok(rh)?(lh.y+rh.y)/2:sY+0.3;
    if (nose.y>(sY+hY)/2+0.08) {
      t.crouchHide++;
      if (t.crouchHide>=8){
        t.crouchHide=0;
        this._fire(`crch_${t.id}_${t.postContact.cls}`,`AGACHADO — ${t.postContact.label} ZONA BAJA`,'high',this.config.cooldown*1000);
        this._addScore(t,this._B('agachado'),'AGACHADO'); t.postContact.fired=true; t.badges.push('⚠ AGACHADO');
      }
    } else t.crouchHide=Math.max(0,t.crouchHide-2);
  }

  // ── T Trayectoria directa ────────────────────────────────────────────────────
  _checkDirectTrajectory(t,now) {
    if (t.directTrajFired||!t.firstZoneEntry) return;
    const ms=t.firstZoneEntry-t.firstSeen;
    if (ms<MIN_BROWSE_MS&&ms>0) {
      t.directTrajFired=true;
      const zn=this._getFirstZoneName(t);
      this._fire(`traj_${t.id}`,`ACCESO DIRECTO${zn} — SIN BROWSING`,'low',this.config.cooldown*1000);
      this._addScore(t,this._B('trayectoria'),'TRAYECTORIA DIRECTA'); t.badges.push('⚠ DIRECTO');
    } else if (ms>=MIN_BROWSE_MS) t.directTrajFired=true;
  }
  _getFirstZoneName(t) {
    for (const key of Object.keys(t.inZoneWrist)) {
      const z=this.zoneManager.zones.find(z=>z.id===key.slice(2));
      if (z) return ` A ${z.name.toUpperCase()}`;
    }
    return '';
  }

  // ── N W Análisis grupal ──────────────────────────────────────────────────────
  _analyzeGroup(now) {
    if (this._tracks.length<2) return;
    const active=this._tracks.filter(t=>!t.missed&&!t.isEmployee);
    // [N] DISTRACTOR
    if (this._profile.behaviors.distractor) {
      const stealers=active.filter(t=>t.postContact&&!t.postContact.fired);
      const distractors=active.filter(t=>{
        if (stealers.includes(t)) return false;
        const nearPay=this.zoneManager.zones.filter(z=>z.type==='pago').some(z=>{
          const cx=z.points.reduce((s,p)=>s+p.x,0)/z.points.length;
          const cy=z.points.reduce((s,p)=>s+p.y,0)/z.points.length;
          return _d((t.nx1+t.nx2)/2,(t.ny1+t.ny2)/2,cx,cy)<DISTRACTOR_PAY_DIST;
        });
        return nearPay||((t.ny1+t.ny2)/2<0.25);
      });
      for (const s of stealers) {
        if (!distractors.length) continue;
        this._fire(`dist_${s.id}`,`CÓMPLICE DISTRACTOR — ${distractors.length} persona${distractors.length>1?'s':''} en mostrador`,'high',this.config.cooldown*1000);
        this._addScore(s,this._B('distractor'),'CÓMPLICE DISTRACTOR'); s.badges.push('⚠ CÓMPLICE');
      }
    }
    // [W] PANTALLA HUMANA
    for (const tA of active) {
      if (!Object.values(tA.inZoneWrist).some(v=>v)) continue;
      for (const tB of active) {
        if (tB.id===tA.id) continue;
        const aC={x:(tA.nx1+tA.nx2)/2,y:(tA.ny1+tA.ny2)/2};
        const bC={x:(tB.nx1+tB.nx2)/2,y:(tB.ny1+tB.ny2)/2};
        if (bC.y<aC.y-0.10&&bC.x>=tA.nx1-0.10&&bC.x<=tA.nx2+0.10&&_d(aC.x,aC.y,bC.x,bC.y)<SCREEN_MAX_DIST) {
          this._fire(`wall_${tA.id}_${tB.id}`,'PANTALLA HUMANA — CÓMPLICE BLOQUEANDO VISTA','high',this.config.cooldown*1000);
          this._addScore(tA,25,'PANTALLA HUMANA'); tA.badges.push('⚠ BLOQUEADO'); tB.badges.push('⚠ CÓMPLICE');
          break;
        }
      }
    }
  }

  // ── Score ─────────────────────────────────────────────────────────────────────
  _addScore(t,pts,reason) {
    if (t.isEmployee) return;
    t.suspicionScore=Math.min(100,t.suspicionScore+pts);
    if (reason&&!t.scoreEvidence.includes(reason)){t.scoreEvidence.push(reason);if(t.scoreEvidence.length>8)t.scoreEvidence.shift();}
  }
  _decayScore(t) {
    if (!t.postContact&&Object.values(t.inZoneWrist).every(v=>!v)) t.suspicionScore=Math.max(0,t.suspicionScore-2);
    if (t.suspicionScore===0) t.scoreEvidence=[];
  }
  _checkSuspicionScore(t,now) {
    const th=this._profile.scoreThreshold;
    if (t.suspicionScore>=th) {
      this._fire(`score_${t.id}`,`ROBO CONFIRMADO — SCORE ${Math.round(t.suspicionScore)}/100 | ${t.scoreEvidence.slice(-3).join(' + ')}`,'high',this.config.cooldown*1000);
      t.scoreEvidence=[]; t.suspicionScore=th*0.35;
    }
    if (t.suspicionScore>=th*0.55) t.badges.push(`⚠ ${Math.round(t.suspicionScore)}pts`);
  }
  _B(key) { return this._profile.scoreBonus?.[key]??15; }

  // ── Fire ──────────────────────────────────────────────────────────────────────
  _fire(key,type,severity,coolMs) {
    const now=Date.now();
    if (now-(this._lastAlert[key]||0)<coolMs) return;
    this._lastAlert[key]=now;
    if (this.onDetection)  this.onDetection(type,severity);
    if (this.alertManager) this.alertManager.trigger(type,severity);
  }

  // ── Render ────────────────────────────────────────────────────────────────────
  _render() {
    this.ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
    this.zoneManager.drawZone(this.zoneManager.zones.some(z=>z.alert));
    this.zoneManager.drawPreview();
    this._drawDetections(this._lastDets);
  }
  _drawDetections(poseDets) {
    const ctx=this.ctx, cw=this.canvas.width, ch=this.canvas.height;
    // Objetos
    for (const obj of this._objTracker.alertVisible) {
      const {nx1,ny1,nx2,ny2}=obj.bbox;
      const x1=nx1*cw,y1=ny1*ch,x2=nx2*cw,y2=ny2*ch;
      const isBag=BAG_IDS.has(obj.cls);
      const col=isBag?'rgba(191,90,242,0.9)':'rgba(255,170,0,0.85)';
      ctx.save(); ctx.strokeStyle=col; ctx.lineWidth=1.8;
      ctx.setLineDash([4,3]); ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
      const lbl=`${obj.label} ${Math.round(obj.conf*100)}%`;
      const lw2=ctx.measureText(lbl).width+6;
      ctx.font='9px "Share Tech Mono",monospace';
      ctx.fillStyle=isBag?'rgba(191,90,242,0.15)':'rgba(255,170,0,0.15)'; ctx.fillRect(x1,y1-14,lw2,13);
      ctx.fillStyle=col; ctx.fillText(lbl,x1+3,y1-4); ctx.restore();
    }
    // Personas
    for (const det of poseDets) {
      const k=det.kps, x1=det.nx1*cw,y1=det.ny1*ch,x2=det.nx2*cw,y2=det.ny2*ch;
      const track=this._tracks.find(t=>!t.missed&&this._iou(t,det)>0.3);
      const isEmp=track?.isEmployee;
      const inZone=track&&Object.values(track.inZoneWrist||{}).some(v=>v);
      const hasPost=track?.postContact&&!track.postContact.fired;
      const scanning=track?.badges?.includes('⚠ ESCANEO');
      const hipHide=(track?.hipConcealment??0)>2;
      const hasCom=track?.badges?.some(b=>b.includes('CÓMPLICE')||b.includes('BLOQUEADO'));
      const boxCol=isEmp?'rgba(0,230,118,0.6)':hasCom?'#ff6b35':inZone?'#ff3d3d':hipHide?'#ff6b35':hasPost?'#ffaa00':scanning?'#bf5af2':'rgba(0,200,255,0.45)';
      ctx.save();
      ctx.strokeStyle=boxCol; ctx.lineWidth=(inZone||hasPost)?2:1.5;
      ctx.strokeRect(x1,y1,x2-x1,y2-y1);
      ctx.fillStyle=boxCol; ctx.font='10px "Share Tech Mono",monospace';
      ctx.fillText(`${isEmp?'👷':''}${Math.round(det.conf*100)}%`,x1+3,y1-3);
      if (track&&track.suspicionScore>25&&!isEmp) {
        const th=this._profile.scoreThreshold, sc=track.suspicionScore;
        ctx.fillStyle=sc>=th*0.8?'#ff3d3d':sc>=th*0.5?'#ffaa00':'#ffee58';
        ctx.font='bold 9px "Share Tech Mono",monospace';
        ctx.fillText(`${Math.round(sc)}pts`,x1+3,y2-5);
      }
      ctx.restore();
      // Esqueleto
      ctx.save(); ctx.lineWidth=1.8;
      for (const [a,b] of BONES) {
        const pa=k[a],pb=k[b]; if (!_ok(pa)||!_ok(pb)) continue;
        ctx.beginPath(); ctx.moveTo(pa.x*cw,pa.y*ch); ctx.lineTo(pb.x*cw,pb.y*ch);
        ctx.strokeStyle=isEmp?'rgba(0,230,118,0.4)':'rgba(0,200,255,0.5)'; ctx.globalAlpha=0.75; ctx.stroke();
      }
      ctx.globalAlpha=1;
      for (let i=0;i<17;i++) {
        const p=k[i]; if (!_ok(p)) continue;
        const isW=i===KP.L_WRIST||i===KP.R_WRIST, isH=i===KP.L_HIP||i===KP.R_HIP;
        const inZ=isW&&this.zoneManager.getZonesForPoint(p.x,p.y).length>0;
        const onO=isW&&this._objTracker.alertVisible.some(o=>{const m=0.06;return p.x>=o.bbox.nx1-m&&p.x<=o.bbox.nx2+m&&p.y>=o.bbox.ny1-m&&p.y<=o.bbox.ny2+m;});
        ctx.beginPath(); ctx.arc(p.x*cw,p.y*ch,isW?6:isH?4:3,0,Math.PI*2);
        ctx.fillStyle=isEmp?'rgba(0,230,118,0.8)':inZ?'#ff3d3d':isW?'#ffb800':isH?'#bf5af2':'rgba(255,255,255,0.7)';
        ctx.fill();
        if ((inZ||onO)&&!isEmp) {
          ctx.beginPath(); ctx.arc(p.x*cw,p.y*ch,11,0,Math.PI*2);
          ctx.strokeStyle=inZ?'#ff3d3d':'#ffb800'; ctx.lineWidth=1.5;
          ctx.globalAlpha=0.5+0.5*Math.sin(Date.now()/200); ctx.stroke(); ctx.globalAlpha=1;
        }
        if (isH&&(track?.hipConcealment??0)>0) {
          ctx.beginPath(); ctx.arc(p.x*cw,p.y*ch,13,0,Math.PI*2);
          ctx.strokeStyle='#ff6b35'; ctx.lineWidth=1.5;
          ctx.globalAlpha=0.3+0.4*Math.sin(Date.now()/250); ctx.stroke(); ctx.globalAlpha=1;
        }
      }
      ctx.restore();
      // Badges
      if (track?.badges?.length) {
        ctx.save(); ctx.font='bold 9px "Share Tech Mono",monospace';
        let bx=det.nx1*cw; const by=det.ny2*ch+13;
        for (const badge of track.badges) {
          ctx.fillStyle=badge.includes('ZONA')?'#ff3d3d':badge.includes('MERODEO')?'#ffaa00':badge.includes('ESCANEO')?'#bf5af2':badge.includes('CADERA')||badge.includes('CÓMPLICE')||badge.includes('BLOQUEADO')?'#ff6b35':badge.includes('pts')?'#ff3d3d':badge==='👷'?'#00e676':'rgba(255,58,58,0.9)';
          ctx.fillText(badge,bx,by); bx+=ctx.measureText(badge).width+8;
        }
        ctx.restore();
      }
      // Indicador SEGUIMIENTO
      if (hasPost) {
        const w=track.postContact.side==='L'?k[KP.L_WRIST]:k[KP.R_WRIST];
        if (_ok(w)) {
          ctx.save(); ctx.beginPath(); ctx.arc(w.x*cw,w.y*ch,14,0,Math.PI*2);
          ctx.strokeStyle='#ffaa00'; ctx.lineWidth=2;
          ctx.globalAlpha=0.4+0.4*Math.sin(Date.now()/150); ctx.stroke(); ctx.globalAlpha=1;
          ctx.font='bold 8px "Share Tech Mono",monospace'; ctx.fillStyle='#ffaa00';
          ctx.fillText('SEGUIMIENTO',w.x*cw-28,w.y*ch-17); ctx.restore();
        }
      }
    }
  }

  // ── Control ───────────────────────────────────────────────────────────────────
  start() {
    this.active=true; this._lastAlert={}; this._interactions={};
    for (const t of this._tracks) Object.assign(t,{
      inZoneWrist:{},dwellStart:{},zoneEntryFrames:{},pocketL:0,pocketR:0,crossedArms:0,
      cajaExit:{},postContact:null,zoneVisits:{},visitedPay:false,
      noseXHist:[],bodyScreen:0,crouchHide:0,hipConcealment:0,
      directTrajFired:false,firstZoneEntry:null,
      suspicionScore:0,scoreEvidence:[],badges:[],
    });
  }
  stop()          { this.active=false; }
  updateConfig(c) { Object.assign(this.config,c); if (c.storeType) this.setStoreType(c.storeType); }
  destroy()       { if (this._renderLoopId) cancelAnimationFrame(this._renderLoopId); }
}