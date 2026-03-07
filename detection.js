/**
 * detection.js — SSIP v8.1
 * FIX v8.1:
 *   [F1] Label inteligente — baja conf muestra "OBJETO?" en vez de "BOLSO/MOCHILA"
 *   [F2] Seg solo corre cuando hay postContact activo (ahorra CPU)
 *   [F3] Seg cada 5fr, MP cada 3fr, en frames distintos (no se superponen)
 *   [F4] Contacto ignorado si obj.conf < 0.42 (evita falsos positivos fondos)
 *   [F5] Color tenue para detecciones dudosas (gris vs púrpura llamativo)
 * ═══════════════════════════════════════════════════════════════════════════
 * Modelos: yolo26n-pose.onnx  (17 kp, hasta 8 personas)
 *          yolo26n.onnx       (80 clases COCO — fallback si YOLOE no disponible)
 *          yoloe26n.onnx      [NUEVO v6] 1200+ categorías LVIS re-parametrizado,
 *                             postproceso idéntico a yolo26n, overhead cero.
 *                             Si existe en servidor, reemplaza yolo26n automáticamente.
 *          MediaPipe Hand     [NUEVO v6] 21 landmarks por mano (pinch grip, palma,
 *                             orientación). Carga lazy desde CDN, opcional.
 *
 * ═══ MEJORAS v6.0 ═══════════════════════════════════════════════════════════
 *
 *  [M1] YOLOE AUTO-UPGRADE
 *       Intenta cargar yoloe26n.onnx primero. Si falla → cae a yolo26n.onnx.
 *       Sin cambios en el postproceso: mismo tensor [1,84,8400] COCO-compatible.
 *       Con YOLOE: papel higiénico, ropa, comida, herramientas, etc. son detectados.
 *
 *  [M2] MEDIAPIPE HAND LANDMARKS — 21 puntos por mano
 *       Carga asíncrona lazy desde CDN. Si no carga, el sistema funciona igual.
 *       Nuevos análisis habilitados cuando disponible:
 *         · Pinch grip confirmado (pulgar + índice < umbral): +score contacto real
 *         · Palma cerrada/abierta: diferencia tomar vs soltar
 *         · Orientación palma hacia adentro (ocultamiento activo)
 *         · Velocidad de cierre de mano: rápido = arrebato, lento = pick
 *
 *  [M3] FILTRO DE INTENCIÓN POST-SALIDA DE ZONA
 *       Soluciona el falso positivo del v5.3 zone-exit.
 *       La dirección de la mano al salir de zona determina si activar postContact:
 *         · Mano va hacia cuerpo (torso/cadera/bolso) → postContact activado
 *         · Mano va hacia afuera (carrito, estante opuesto, abajo) → ignorado
 *
 *  [M4] VENTANA TEMPORAL AMPLIADA — 90 frames (~6s a 15fps)
 *       El historial de tracks sube de 30 a 90 frames para detectar
 *       secuencias completas de hurto (escaneo→aproxi→contacto→ocultamiento).
 *
 *  [M5] SCORE DE SECUENCIA COMPLETA — multiplicador 1.5×
 *       Si se detectan escaneo→zona→postContact dentro de 30s, el score
 *       se multiplica por 1.5 en lugar de sumar independiente.
 *       Premia las secuencias completas y reduce ruido de eventos aislados.
 *
 *  [M6] VELOCIDAD DE MANO — detección de arrebato mejorada
 *       Calcula velocidad de muñeca entre frames consecutivos.
 *       Velocidad alta en zona + postContact = arrebato confirmado.
 *       Score diferenciado: lento pick vs rápido grab.
 *
 *  [M7] ANÁLISIS DE GRUPO MEJORADO
 *       Detección de formación en V (bloqueo de ángulo de cámara con 3+personas).
 *       Sincronización de comportamiento: si 2 personas hacen escaneo simultáneo
 *       cerca de la misma zona → alerta de coordinación.
 *
 * ═══ MEJORAS v7.0 ═══════════════════════════════════════════════════════════
 *
 *  [M8] SMOOTHING + INTERPOLACIÓN DE KEYPOINTS (inspirado en PoseLift WACV2025)
 *       EMA de 3 frames sobre cada keypoint antes de cualquier análisis.
 *       Interpolación lineal de hasta 4 frames cuando KP desaparece temporalmente.
 *       Elimina el ruido de cámara de baja calidad sin bajar thresholds.
 *
 *  [M9] CLASIFICACIÓN DE INTENCIÓN — Estado de Alta Vigilancia
 *       Al detectar escaneo → 8s de Alta Vigilancia activa.
 *       Cualquier gesto sospechoso en esa ventana → score ×3.
 *       Implementa: "cualquier gesto después de mirada de vigilancia multiplica por 3"
 *
 *  [M10] FILTRO DINÁMICO DE FALSAS BOLSAS
 *        Descarta objetos BAG que llevan >45 frames sin moverse (fondo estático).
 *        Descarta BAG cuyo tamaño es <8% de la altura de la persona (micro-objeto).
 *        Soluciona el "todo detecta como mochila" de cámaras de baja calidad.
 *
 *  [M3+] LÓGICA POST-CONTACTO OBLIGATORIA MEJORADA
 *        Bolsillo sin postContact previo → score conservador (no multiplica ×3).
 *        Bolsillo con postContact confirmado → multiplica full por vigilancia.
 *        Implementa: "si no tocó primero, el bolsillo es probablemente el teléfono"
 *
 * ═══ MEJORAS v8.0 ═══════════════════════════════════════════════════════════
 *
 *  [M11] SEGMENTACIÓN DE SILUETA — YOLOv8n-seg.onnx
 *        Carga lazy igual que MediaPipe. Si el .onnx no existe → ignorado.
 *        Cada persona detectada por pose recibe su máscara de segmentación.
 *        SilhouetteTracker por track:
 *          · Al momento del postContact activo: captura máscara SNAPSHOT
 *          · Frames siguientes: compara con snapshot región por región
 *          · Si la silueta del torso/cadera CRECIÓ → algo fue incorporado
 *          · SIL_GROW_THRESH: área relativa que debe crecer para confirmar
 *          · Nuevo comportamiento: SIL — Ocultamiento por silueta (+35 pts)
 *        Ventaja clave: detecta ocultamiento de objetos pequeños que los
 *        keypoints no pueden capturar (barra de chocolate, perfume, etc.)
 *        porque el contorno del cuerpo cambia aunque los puntos no.
 *
 * ═══ COMPORTAMIENTOS (total 26) ══════════════════════════════════════════
 *  Z1  Mano en zona (debounce N frames)          LOW
 *  Z2  Permanencia en zona                       HIGH
 *  Z3  Mano escapa al torso desde zona           HIGH
 *  P1  Mano en bolsillo                          HIGH
 *  P2  Brazos cruzados + muñeca oculta           HIGH
 *  O1  Contacto objeto en zona (400ms+)          LOW
 *  O2  Objeto tomado (desaparición estable)      HIGH
 *  A   Caja → bolsillo/manga                     HIGH
 *  B   Bajo manga (post-contacto)                HIGH
 *  C   Bag stuffing                              HIGH
 *  D   Bajo ropa torso ampliado                  HIGH
 *  E   Merodeo (3+ accesos sin compra)           MEDIUM
 *  F   Traspaso entre personas                   HIGH
 *  G   Arrebato rápido (300-700ms)               HIGH
 *  H   Escaneo previo (cabeza gira)              MEDIUM
 *  I   Cuerpo como pantalla                      HIGH
 *  J   Agacharse y ocultar                       HIGH
 *  K   Cadera / bermuda (post-contacto)          HIGH
 *  N   Distractor (cómplice en mostrador)        HIGH
 *  S   Robo confirmado — score                   HIGH
 *  T   Trayectoria directa a zona                LOW
 *  W   Pantalla humana (cómplice bloquea)        HIGH
 *  MP  Pinch grip confirmado (MediaPipe)         HIGH   [NUEVO v6]
 *  SQ  Secuencia completa detectada              HIGH   [NUEVO v6]
 *  GR  Coordinación grupal (escaneo sincrónico)  HIGH   [NUEVO v6]
 *  SIL Ocultamiento por silueta (seg. YOLOv8-seg) HIGH   [NUEVO v8]
 */

import { getProfile, getFamily, BAG_IDS, ALERT_IDS } from './store-profiles.js';

// ─────────────────────────────────────────────────────────────────────────────
//  CONSTANTES
// ─────────────────────────────────────────────────────────────────────────────
const POSE_MODEL  = './yolo26n-pose.onnx';
const OBJ_MODEL   = './yoloe26n.onnx';       // [M1] YOLOE 1200+ clases (fallback: yolo26n.onnx)
const OBJ_FALLBACK = './yolo26n.onnx';
const INPUT_W    = 640;
const INPUT_H    = 640;
const CONF_POSE  = 0.25;  // [CAL] bajado de 0.30 — permite detectar personas con lighting bajo
const CONF_OBJ   = 0.30;  // [CAL] 0.30 — permite detectar objetos parcialmente visibles
const KP_THRESH  = 0.18;  // [CAL] bajado de 0.25 — cámaras de baja calidad dan wrist 0.18-0.24
const IOU_THRESH = 0.45;
const OBJ_VIS_WINDOW    = 14;
const SAME_OBJ_IOU      = 0.28;
const MIN_BROWSE_MS     = 1500;
const AUTO_EMPLOYEE_MIN = 5;
const SCREEN_MAX_DIST   = 0.35;
const DISTRACTOR_PAY_DIST = 0.30;
const EXIT_SCORE_MEMORY_MS = 30000;
const MAX_HISTORY       = 90;                // [M4] 90 frames (~6s a 15fps)
const SEQ_WINDOW_MS     = 30000;             // [M5] ventana para secuencia completa
const VIGILANCE_WINDOW_MS = 8000;          // [M9] ventana de alta vigilancia post-escaneo
const VIGILANCE_MULTIPLIER = 3.0;          // [M9] multiplicador de score en alta vigilancia
const SEQ_MULTIPLIER    = 1.5;              // [M5] multiplicador por secuencia completa
const HAND_PINCH_DIST   = 0.06;             // [M2] distancia normalizada pulgar-índice = pinch
const BAG_STATIC_FRAMES = 45;              // [M10] frames sin movimiento para descartar bolsa estática
const BAG_MIN_SCALE     = 0.08;            // [M10] escala mínima bolsa vs cuerpo persona
const ZONE_EXIT_BODY_RATIO = 0.55;           // [M3] ratio para filtro de intención
const KP_SMOOTH_FRAMES  = 3;               // [M8] frames para suavizado de keypoints
const KP_INTERP_MAX_GAP = 4;              // [M8] max frames sin KP antes de interpolar

// [M11] Segmentación de silueta
const SEG_MODEL        = './yolov8n-seg.onnx'; // carga lazy, opcional
const SEG_MASK_SIZE    = 160;                  // resolución de la máscara de salida
const SEG_CONF         = 0.40;                 // confianza mínima para persona seg
const SIL_GROW_THRESH  = 0.045;               // área relativa que debe crecer (4.5% del bbox)
const SIL_FRAMES_WAIT  = 4;                    // frames a esperar antes de comparar
const SIL_REGION_TORSO = [0.20, 0.80];        // franja Y del torso dentro del bbox [top%, bot%]
const SIL_REGION_HIP   = [0.55, 1.00];        // franja Y de cadera dentro del bbox

// MediaPipe CDN
// v0.10.14 — último release estable con WASM y ESM correctos
const MP_VISION_CDN  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14';
// ESM entry point — se importa con import() dinámico, evita el SyntaxError de script tag
const MP_VISION_ESM  = 'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14/vision_bundle.mjs';
const MP_HAND_MODEL  = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task';

// MediaPipe Hand landmark indices
const MH = {
  WRIST:0, THUMB_CMC:1, THUMB_MCP:2, THUMB_IP:3, THUMB_TIP:4,
  INDEX_MCP:5, INDEX_PIP:6, INDEX_DIP:7, INDEX_TIP:8,
  MIDDLE_MCP:9, MIDDLE_PIP:10, MIDDLE_DIP:11, MIDDLE_TIP:12,
  RING_MCP:13, RING_PIP:14, RING_DIP:15, RING_TIP:16,
  PINKY_MCP:17, PINKY_PIP:18, PINKY_DIP:19, PINKY_TIP:20,
};

let _handLandmarker = null;   // MediaPipe HandLandmarker (carga lazy)
let _mpReady        = false;
let _mpLoading      = false;

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

let _poseSession  = null;
let _objSession   = null;
let _posePromise  = null;
let _objPromise   = null;
let _objModelUsed = null;   // 'yoloe' | 'yolo' | null
let _segSession   = null;   // [M11] YOLOv8n-seg session (opcional)
let _segLoading   = false;
let _segReady     = false;

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
    this._maxHistory   = MAX_HISTORY;  // [M4] 90 frames
    this._objDets      = [];
    this._objTracker   = new ObjTracker();
    this._interactions = {};
    this._lastAlert    = {};
    this._fpsFrames    = 0;
    this._fpsLast      = performance.now();
    this.currentFPS    = 0;
    this._renderLoopId = null;
    this._lastDets     = [];
    this._lastMpHands  = [];
    this._lastMpHandedness = [];
    this.onDetection   = null;
    this._employeeIds  = new Set();
    this._exitScores   = [];
    this._lastSegMasks = [];   // [M11] máscaras seg del último frame
    console.log(`%c✓ SSIP v8.1 — ${this._profile.icon} ${this._profile.name} | YOLOE+MediaPipe`, 'color:#00d4ff;font-weight:bold');
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

  // Retorna { total, inZone, byZone: { zoneName: count } }
  getZoneCounts() {
    const byZone = {};
    let inZone = 0;
    for (const t of this._tracks) {
      if (t.missed || t.isEmployee) continue;
      const active = Object.entries(t.inZoneWrist)
        .filter(([, v]) => v)
        .map(([key]) => {
          const zId = key.slice(2); // quitar "L_" o "R_"
          return this.zoneManager.zones.find(z => z.id === zId);
        })
        .filter(Boolean);
      if (active.length > 0) {
        inZone++;
        const seen = new Set();
        for (const z of active) {
          if (!seen.has(z.id)) {
            seen.add(z.id);
            byZone[z.name] = (byZone[z.name] || 0) + 1;
          }
        }
      }
    }
    return { total: this._tracks.filter(t => !t.missed && !t.isEmployee).length, inZone, byZone };
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

    // [M1] YOLOE auto-upgrade: intenta yoloe26n.onnx, cae a yolo26n.onnx
    if (!_objPromise) {
      _objPromise = this._loadModel(OBJ_MODEL, 'obj').then(() => {
        _objModelUsed = 'yoloe';
        console.log('%c🔥 YOLOE 1200+ clases ACTIVO', 'color:#00ff94;font-weight:bold');
      }).catch(() => {
        console.warn('%c⚠ yoloe26n.onnx no encontrado → fallback yolo26n.onnx (80 clases COCO)', 'color:#ffaa00');
        return this._loadModel(OBJ_FALLBACK, 'obj').then(() => {
          _objModelUsed = 'yolo';
        });
      });
    }

    const [pR, oR] = await Promise.allSettled([_posePromise, _objPromise]);
    if (pR.status === 'rejected') throw new Error('No se pudo cargar yolo26n-pose.onnx');
    if (oR.status === 'rejected') console.warn('%c⚠ Modelo de objetos no disponible', 'color:#ffaa00');

    // [M2] MediaPipe Hand — carga lazy en background, no bloquea
    this._loadMediaPipeHands();
    // [M11] YOLOv8n-seg — carga lazy en background, no bloquea
    this._loadSegModel();
    this._startRenderLoop();
  }

  // [M2] MediaPipe Hand — carga asíncrona sin bloquear el pipeline principal
  async _loadMediaPipeHands() {
    if (_mpLoading || _mpReady) return;
    _mpLoading = true;
    try {
      // ── Estrategia de carga: import() dinámico ESM ────────────────
      // vision_bundle.js/.mjs usan "export" internamente → NO se pueden
      // cargar con <script> clásico (da SyntaxError). La única solución
      // compatible con módulos ES es import() dinámico.
      let FilesetResolverCls, HandLandmarkerCls;

      if (typeof FilesetResolver !== 'undefined' && typeof HandLandmarker !== 'undefined') {
        // Ya estaban disponibles globalmente (cargados por el HTML)
        FilesetResolverCls = FilesetResolver;
        HandLandmarkerCls  = HandLandmarker;
      } else {
        // import() dinámico — funciona desde cualquier contexto module
        const mp = await import(MP_VISION_ESM);
        FilesetResolverCls = mp.FilesetResolver;
        HandLandmarkerCls  = mp.HandLandmarker;
      }

      if (!FilesetResolverCls) throw new Error('FilesetResolver no disponible tras import');

      const vision = await FilesetResolverCls.forVisionTasks(`${MP_VISION_CDN}/wasm`);
      _handLandmarker = await HandLandmarkerCls.createFromOptions(vision, {
        baseOptions: { modelAssetPath: MP_HAND_MODEL, delegate: 'GPU' },
        runningMode: 'VIDEO',
        numHands: 4,
        minHandDetectionConfidence: 0.5,
        minHandPresenceConfidence:  0.5,
        minTrackingConfidence:      0.5,
      });
      _mpReady = true;
      console.log('%c🖐 MediaPipe Hand Landmarks ACTIVO (21pts/mano)', 'color:#bf5af2;font-weight:bold');
    } catch(e) {
      _mpLoading = false;
      console.info(`%cℹ MediaPipe Hand no disponible (${e.message}) — sin análisis de grip`, 'color:#888');
    }
  }
  async _loadModel(path, name) {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime no cargado');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    for (const ep of ['webgl', 'wasm']) {
      try {
        const s = await ort.InferenceSession.create(path, { executionProviders: [ep], graphOptimizationLevel: 'all' });
        if (name === 'pose') _poseSession = s; else _objSession = s;
        console.log(`%c✓ YOLO26n-${name} (${ep.toUpperCase()})`, 'color:#00e676;font-weight:bold');
        return;
      } catch(e) { console.warn(`ONNX [${name}/${ep}]:`, e.message); }
    }
    throw new Error(`No se pudo cargar ${path} — verificá que el archivo esté en el servidor`);
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

    // [M11] YOLOv8n-seg — corre cada 5 frames y solo si hay postContact activo
    // (no tiene sentido calcular silueta si nadie está en estado sospechoso)
    const _hasPostContact = this._tracks.some(t => t.postContact && !t.postContact.fired);
    if (_segReady && _segSession && this._fpsFrames % 5 === 0 && _hasPostContact) {
      try {
        const segOut = await _segSession.run({ images: tensor });
        this._lastSegMasks = this._postprocessSeg(segOut, meta, poseDets);
        this._updateSilhouetteTracks(Date.now());
      } catch(e) { this._lastSegMasks = []; }
    }

    // [M2] MediaPipe Hand — corre cada 3 frames, en frames distintos a Seg
    if (_mpReady && _handLandmarker && this._fpsFrames % 3 === 1) {
      try {
        const mpResult = _handLandmarker.detectForVideo(video, performance.now());
        this._lastMpHands = mpResult?.landmarks || [];
        this._lastMpHandedness = mpResult?.handedness || [];
      } catch(e) { this._lastMpHands = []; }
    }

    this._updateTracks(poseDets, Date.now());

    // [M2] Actualizar análisis de grip por track
    if (_mpReady && this._lastMpHands?.length) {
      this._updateMpGrip(Date.now());
    }

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
      // [M4] historial ampliado + [M5] secuencia + [M6] velocidad + [M2] MP
      wristVelHist:{L:[],R:[]},    // velocidad de muñeca por frame
      seqState:{scan:0,zone:0,post:0},  // timestamps para secuencia completa
      seqBonusFired:false,         // evitar doble multiplicador
      mpGripConf:{L:0,R:0},        // confianza de pinch grip por mano (MediaPipe)
      mpPalmIn:{L:false,R:false},  // palma orientada hacia adentro (ocultamiento)
      // [M8] Suavizado de keypoints
      kpSmooth: Array.from({length:17},()=>({x:0,y:0,c:0,n:0})),
      kpLastValid: Array.from({length:17},()=>null), // último KP válido para interpolación
      kpMissingFrames: new Array(17).fill(0),        // frames consecutivos sin KP
      // [M9] Estado de alta vigilancia
      vigilanceUntil: 0,           // timestamp hasta cuando está en alta vigilancia
      vigilanceCount: 0,           // cuántas veces entró en vigilancia
      // [M10] Filtro de bolsa estática
      bagStaticFrames: {},         // { objId: frames sin movimiento }
      // [M11] Silueta
      silSnapshot:     null,       // { torsoArea, hipArea, frame } — capturado al activar postContact
      silWaitFrames:   0,          // frames esperados desde snapshot
      silFired:        false,      // evitar doble disparo
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
    // [M8] Suavizar keypoints antes de cualquier análisis
    const k = this._smoothKps(t, t.kps);
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
    this._trackWristVelocity(t,lw,rw,now);   // [M6]
    this._checkMpGrip(t,lw,rw,now);           // [M2]
    this._checkSilhouette(t,now);              // [M11]
    this._checkSequenceBonus(t,now);           // [M5]
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
          this._fire(`emp_dw_${t.id}_${key}`,`EMPLEADO — PERMANENCIA INUSUAL EN ${zone.name.toUpperCase()}`,'medium',30000);
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
          // [CAL] Mínimo 2 frames para confirmar entrada — con cámaras lentas
          // el perfil puede pedir 3-4 pero 2 es el mínimo efectivo
          const minFrames = Math.max(2, P.zoneEntryFrames);
          if (t.zoneEntryFrames[key]>=minFrames) {
            t.inZoneWrist[key]=true; t.dwellStart[key]=now;
            zone.alert=true; setTimeout(()=>{if(zone)zone.alert=false;},2000);
            this._fire(`ze_${t.id}_${key}`,`MANO EN ${zone.name.toUpperCase()}`,'low',1500);
            this._recordVisit(t,zone,now);
            if (zone.type==='pago') t.visitedPay=true;
            if (!t.firstZoneEntry) t.firstZoneEntry=now;
          }
        } else {
          const elapsed=(now-(t.dwellStart[key]||now))/1000;
          if (elapsed>=this.config.dwellTime) {
            t.dwellStart[key]=now+this.config.dwellTime*1000;
            this._fire(`dw_${t.id}_${key}`,`PERMANENCIA — ${zone.name.toUpperCase()}`,'high',this.config.cooldown*1000);
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

          // [M3] Filtro de intención post-salida de zona
          // Activa postContact solo si la mano se mueve HACIA el cuerpo, no hacia afuera.
          if (z && z.type !== 'pago' && !t.postContact && _ok(w)) {
            const bodyCenter = this._getBodyCenter(t);
            const intentScore = this._calcExitIntent(t, side, w, bodyCenter, now);
            // intentScore > 0 = mano va hacia cuerpo → es sospechoso
            // intentScore < 0 = mano va hacia afuera → cliente normal examinando
            if (intentScore >= 0) {
              const elbow = side==='L' ? t.kps[KP.L_ELBOW] : t.kps[KP.R_ELBOW];
              t.postContact = {
                disappearT:   now,
                label:        `OBJETO EN ${z.name.toUpperCase()}`,
                cls:          -1,
                side,
                wristY0:      w.y,
                elbowY0:      _ok(elbow) ? elbow.y : null,
                fired:        false,
                fromZoneExit: true,
                intentScore,
              };
              if (t.seqState.zone === 0) t.seqState.zone = now; // [M5]
            }
          }
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
      this._fire(`esc_${t.id}_${zone.id}_${side}`,`OBJETO OCULTADO — ${zone.name.toUpperCase()}`,'high',this.config.cooldown*1000);
  }

  // ── P1 P2 ───────────────────────────────────────────────────────────────────
  _detectPocket(t,wrist,hip,shoulder,side) {
    // [CAL] Usamos bbox del track como fallback si hip/shoulder son de baja confianza
    const hipX  = (hip&&hip.c>0.12)  ? hip.x  : (t.nx1+t.nx2)/2;
    const hipY  = (hip&&hip.c>0.12)  ? hip.y  : t.ny1+(t.ny2-t.ny1)*0.70;
    const shdY  = (shoulder&&shoulder.c>0.12) ? shoulder.y : t.ny1+(t.ny2-t.ny1)*0.30;
    let pocket=false;
    // Muñeca invisible o de muy baja confianza → posición desconocida → bolsillo probable
    if (!wrist||wrist.c<0.15)   pocket=true;
    // Muñeca de baja confianza → zona ampliada
    else if (wrist.c<0.45)      pocket=Math.abs(wrist.x-hipX)<0.22&&Math.abs(wrist.y-hipY)<0.28;
    // Muñeca de buena confianza → zona precisa
    else pocket=wrist.y>hipY-0.05&&wrist.y<hipY+0.25&&Math.abs(wrist.x-hipX)<0.18;
    const sk=side==='L'?'pocketL':'pocketR';
    if (pocket) {
      t[sk]++;
      if (t[sk]>=10){
        t[sk]=0;
        this._fire(`pkt_${side}_${t.id}`,`MANO ${side==='L'?'IZQ.':'DER.'} EN BOLSILLO`,'high',this.config.cooldown*1000);
        // [CAL] Suma score aunque no haya postContact activo
        // Bolsillo es sospechoso de por sí — especialmente si hay score previo
        // [M3+M9] Bolsillo sin postContact previo → evidencia más débil
        // Con postContact confirmado → la vigilancia multiplica full
        const hasPostCtx = !!t.postContact;
        const pocketBase = t.suspicionScore > 10 ? 20 : 10;
        if (hasPostCtx) {
          this._addScoreVigilant(t, pocketBase, `BOLSILLO ${side}`, now); // [M9] full ×3
        } else {
          // Sin contacto previo → bolsillo podría ser teléfono → ×1.5 máximo
          const conservativePts = Math.round(pocketBase * (now < t.vigilanceUntil ? 1.5 : 1));
          this._addScore(t, conservativePts, `BOLSILLO ${side} (sin contacto previo)`);
        }
      }
      if (t[sk]>5) t.badges.push('⚠ BOLSILLO');
    } else t[sk]=Math.max(0,t[sk]-2);
  }
  _detectCrossedArms(t,le,re,lw,rw,ls,rs,lh,rh) {
    if (!le||!re||!ls||!rs) return; // [CAL] hip no requerido
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
        // [FIX] Ignorar objetos de muy baja confianza para evitar falsos positivos
        if (obj.conf < 0.42) continue;
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
        this._addScoreVigilant(t,this._B('manga'),'BAJO MANGA',now); pc.fired=true; // [M9]
        if (t.seqState.post===0) t.seqState.post=now; // [M5]
        t.postContact=null; return;
      }
    }
    // [C] BAG STUFFING
    if (this._profile.behaviors.bagStuffing) {
      // [M10] Solo bolsas reales (no estáticas ni micro-objetos)
      const nearBag=this._objTracker.visible.find(o=>BAG_IDS.has(o.cls)&&_d(w.x,w.y,(o.bbox.nx1+o.bbox.nx2)/2,(o.bbox.ny1+o.bbox.ny2)/2)<0.14&&this._isRealBag(o,t));
      if (nearBag) {
        this._fire(`bag_${t.id}_${pc.cls}`,`BOLSO — ${pc.label} EN BOLSO`,'high',this.config.cooldown*1000);
        this._addScoreVigilant(t,this._B('bagStuffing'),'BAG STUFFING',now); pc.fired=true; // [M9]
        if (t.seqState.post===0) t.seqState.post=now; // [M5]
        t.postContact=null; return;
      }
    }
    // [D] BAJO ROPA — zona ampliada
    if (_ok(ls)&&_ok(rs)&&_ok(lh)&&_ok(rh)) {
      const bL=Math.min(ls.x,rs.x,lh.x,rh.x), bR=Math.max(ls.x,rs.x,lh.x,rh.x), bw=(bR-bL);
      const tx1=bL-bw*0.15, tx2=bR+bw*0.15;
      const ty1=Math.min(ls.y,rs.y), ty2=Math.max(lh.y,rh.y)+0.12;
      if (w.x>tx1&&w.x<tx2&&w.y>ty1&&w.y<ty2&&w.c<hcc) {
        this._fire(`trso_${t.id}_${pc.cls}`,`ROPA — ${pc.label} BAJO ROPA`,'high',this.config.cooldown*1000);
        this._addScore(t,this._B('bajoropa'),'BAJO ROPA'); pc.fired=true;
        if (t.seqState.post===0) t.seqState.post=now; // [M5]
        t.postContact=null; return;
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
    if (t.seqState.scan === 0) t.seqState.scan = now; // [M5] marca inicio de secuencia
    this._enterVigilance(t, now); // [M9] activa alta vigilancia por 8s
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

    // [M7] COORDINACIÓN GRUPAL — escaneo simultáneo cerca de misma zona
    if (active.length >= 2) {
      const scanners = active.filter(t => t.badges.includes('⚠ ESCANEO') || (now - t.seqState.scan < 3000));
      if (scanners.length >= 2) {
        const s0 = scanners[0], s1 = scanners[1];
        const dist = _d((s0.nx1+s0.nx2)/2,(s0.ny1+s0.ny2)/2,(s1.nx1+s1.nx2)/2,(s1.ny1+s1.ny2)/2);
        if (dist < 0.4) {
          this._fire(`grp_scan_${s0.id}_${s1.id}`,
            `COORDINACIÓN — ${scanners.length} PERSONAS ESCANEANDO SIMULTÁNEAMENTE`, 'high',
            this.config.cooldown * 2000);
          for (const s of scanners) {
            this._addScore(s, 20, 'ESCANEO COORDINADO');
            s.badges.push('⚠ COORDINADO');
          }
        }
      }
    }

    // [M7] FORMACIÓN EN V — 3+ personas bloqueando ángulo de cámara
    if (active.length >= 3) {
      const sorted = [...active].sort((a,b)=>((a.nx1+a.nx2)/2)-((b.nx1+b.nx2)/2));
      const leftC  = {x:(sorted[0].nx1+sorted[0].nx2)/2,  y:(sorted[0].ny1+sorted[0].ny2)/2};
      const rightC = {x:(sorted[sorted.length-1].nx1+sorted[sorted.length-1].nx2)/2,
                      y:(sorted[sorted.length-1].ny1+sorted[sorted.length-1].ny2)/2};
      const span   = rightC.x - leftC.x;
      if (span > 0.5) {
        const midOnes = sorted.slice(1,-1).filter(t=>{
          const cx=(t.nx1+t.nx2)/2, cy=(t.ny1+t.ny2)/2;
          return cy < Math.max(leftC.y,rightC.y) - 0.08;
        });
        if (midOnes.length >= 1) {
          this._fire(`vform_${now}`, 'FORMACIÓN EN V — POSIBLE BLOQUEO DE CÁMARA', 'high',
            this.config.cooldown * 3000);
          for (const t of active) { this._addScore(t, 15, 'FORMACIÓN EN V'); }
        }
      }
    }
  }

  // ── [M3] Cuerpo center y filtro de intención ──────────────────────────────────
  _getBodyCenter(t) {
    const lh=t.kps[KP.L_HIP], rh=t.kps[KP.R_HIP];
    const ls=t.kps[KP.L_SHOULDER], rs=t.kps[KP.R_SHOULDER];
    if (_ok(lh)&&_ok(rh)) return _mid(lh,rh);
    if (_ok(ls)&&_ok(rs)) return {x:(ls.x+rs.x)/2, y:(ls.y+rs.y)/2+0.2};
    return {x:(t.nx1+t.nx2)/2, y:(t.ny1+t.ny2)/2};
  }

  // Calcula si la mano al salir de zona va hacia el cuerpo (>0) o hacia afuera (<0)
  _calcExitIntent(t, side, wristNow, bodyCenter, now) {
    // Comparar posición actual de muñeca con últimas N frames
    const hist = t.history;
    const pastIdx = Math.max(0, hist.length - 8);
    const past = hist[pastIdx];
    if (!past) return 0; // sin historial suficiente → asumir neutro
    const wIdx = side==='L' ? KP.L_WRIST : KP.R_WRIST;
    const pastW = past.kps[wIdx];
    if (!_ok(pastW)) return 0;

    // Vector de movimiento de la muñeca
    const dx = wristNow.x - pastW.x;
    const dy = wristNow.y - pastW.y;

    // Vector hacia el cuerpo
    const toBx = bodyCenter.x - wristNow.x;
    const toBy = bodyCenter.y - wristNow.y;
    const toBLen = Math.hypot(toBx, toBy) + 1e-6;

    // Producto punto normalizado: >0 = muñeca se mueve hacia cuerpo
    const dot = (dx*toBx + dy*toBy) / (Math.hypot(dx,dy)+1e-6) / toBLen;

    // También considerar velocidad: movimiento muy lento → neutro
    const vel = Math.hypot(dx, dy);
    if (vel < 0.015) return 0;

    return dot; // [-1, 1]
  }

  // ── [M6] Velocidad de muñeca ──────────────────────────────────────────────────
  _trackWristVelocity(t, lw, rw, now) {
    for (const [w, side] of [[lw,'L'],[rw,'R']]) {
      if (!_ok(w)) { t.wristVelHist[side]=[]; continue; }
      t.wristVelHist[side].push({x:w.x, y:w.y, t:now});
      if (t.wristVelHist[side].length > 8) t.wristVelHist[side].shift();
    }
  }

  // Retorna velocidad promedio de muñeca en últimos frames (unidades normalizadas/s)
  _getWristVelocity(t, side) {
    const hist = t.wristVelHist[side];
    if (hist.length < 2) return 0;
    const a = hist[0], b = hist[hist.length-1];
    const dt = (b.t - a.t) / 1000;
    if (dt < 0.01) return 0;
    return _d(a.x, a.y, b.x, b.y) / dt;
  }

  // ── [M2] MediaPipe Grip analysis ─────────────────────────────────────────────
  // Asocia los resultados de MP a cada track según proximidad de muñeca
  _updateMpGrip(now) {
    if (!this._lastMpHands?.length) return;
    for (const t of this._tracks) {
      if (t.missed || t.isEmployee) continue;
      t.mpGripConf = {L:0, R:0};
      t.mpPalmIn   = {L:false, R:false};

      for (let hi = 0; hi < this._lastMpHands.length; hi++) {
        const hand = this._lastMpHands[hi];
        const handedness = this._lastMpHandedness?.[hi]?.[0]?.categoryName || 'Right';
        const side = handedness === 'Left' ? 'R' : 'L'; // MP invierte para selfie

        const mpWrist = hand[MH.WRIST];
        if (!mpWrist) continue;

        // Asociar esta mano al track más cercano
        const lw = t.kps[KP.L_WRIST], rw = t.kps[KP.R_WRIST];
        const refW = side==='L' ? lw : rw;
        if (!_ok(refW)) continue;
        const dist = _d(mpWrist.x, mpWrist.y, refW.x, refW.y);
        if (dist > 0.15) continue; // no es la mano de este track

        // Pinch grip: distancia pulgar-índice
        const thumbTip = hand[MH.THUMB_TIP], indexTip = hand[MH.INDEX_TIP];
        if (thumbTip && indexTip) {
          const pinchDist = _d(thumbTip.x, thumbTip.y, indexTip.x, indexTip.y);
          t.mpGripConf[side] = Math.max(0, 1 - pinchDist / HAND_PINCH_DIST);
        }

        // Orientación de palma: normal de la palma apunta hacia el cuerpo
        const wrist = hand[MH.WRIST], middleMcp = hand[MH.MIDDLE_MCP];
        const ringMcp = hand[MH.RING_MCP];
        if (wrist && middleMcp && ringMcp) {
          // Vector palma: wrist → middle_mcp
          const px = middleMcp.x - wrist.x, py = middleMcp.y - wrist.y;
          // Si la palma apunta hacia abajo/izquierda (interior) → ocultamiento
          t.mpPalmIn[side] = py > 0.05 || Math.abs(px) < 0.1;
        }
      }
    }
  }

  // Verifica pinch grip y palma y suma score si hay postContact
  _checkMpGrip(t, lw, rw, now) {
    if (!_mpReady) return;
    for (const side of ['L','R']) {
      const grip = t.mpGripConf[side];
      if (grip < 0.6) continue; // no hay pinch confirmado
      const w = side==='L' ? lw : rw;
      if (!_ok(w)) continue;

      // Pinch en zona → contacto real confirmado
      const inZone = this.zoneManager.getZonesForPoint(w.x, w.y).length > 0;
      if (inZone) {
        this._addScore(t, 12, 'PINCH GRIP EN ZONA');
        t.badges.push('✋ GRIP');
      }

      // Pinch post-contacto con palma hacia adentro → ocultamiento activo
      if (t.postContact && !t.postContact.fired && t.mpPalmIn[side]) {
        this._fire(`mpgrip_${t.id}_${side}`,
          `GRIP CONFIRMADO — OBJETO EN MANO (palma oculta)`, 'high',
          this.config.cooldown * 1000);
        this._addScore(t, this._B('contacto') + 15, 'GRIP MEDIAPIPE');
        t.postContact.mpConfirmed = true;
        t.badges.push('✋ OCULTO');
      }
    }
  }

  // ── [M5] Bonus de secuencia completa ─────────────────────────────────────────
  _checkSequenceBonus(t, now) {
    if (t.seqBonusFired || t.isEmployee) return;
    const { scan, zone, post } = t.seqState;
    if (!scan || !zone || !post) return;
    // Secuencia completa: escaneo → zona → ocultamiento dentro de la ventana
    const span = post - scan;
    if (span > 0 && span <= SEQ_WINDOW_MS && zone >= scan && post >= zone) {
      const bonus = Math.round(t.suspicionScore * (SEQ_MULTIPLIER - 1));
      t.suspicionScore = Math.min(100, t.suspicionScore + bonus);
      t.seqBonusFired  = true;
      t.scoreEvidence.push('SECUENCIA COMPLETA');
      this._fire(`seq_${t.id}`,
        `SECUENCIA COMPLETA — ESCANEO → ZONA → OCULTAMIENTO (${Math.round(span/1000)}s)`,
        'high', this.config.cooldown * 1000);
      t.badges.push('🔴 SECUENCIA');
      console.log(`%c🎯 Track #${t.id} secuencia completa en ${Math.round(span/1000)}s, bonus +${bonus}pts`,
        'color:#ff3a3a;font-weight:bold');
    }
  }

  // ── Score ─────────────────────────────────────────────────────────────────────
  _addScore(t,pts,reason) {
    if (t.isEmployee) return;
    t.suspicionScore=Math.min(100,t.suspicionScore+pts);
    if (reason&&!t.scoreEvidence.includes(reason)){t.scoreEvidence.push(reason);if(t.scoreEvidence.length>8)t.scoreEvidence.shift();}
  }
  _decayScore(t) {
    if (!t.postContact&&Object.values(t.inZoneWrist).every(v=>!v)) {
      // [FIX v5.1] Decay proporcional al score: si score alto y sin actividad, baja más rápido
      const rate = t.suspicionScore > 50 ? 6 : t.suspicionScore > 25 ? 3 : 2;
      t.suspicionScore=Math.max(0,t.suspicionScore-rate);
    }
    if (t.suspicionScore===0) t.scoreEvidence=[];
  }
  _checkSuspicionScore(t,now) {
    const th=this._profile.scoreThreshold;
    if (t.suspicionScore>=th) {
      this._fire(`score_${t.id}`,`ROBO CONFIRMADO — SCORE ${Math.round(t.suspicionScore)}/100 | ${t.scoreEvidence.slice(-3).join(' + ')}`,'high',this.config.cooldown*1000);
      t.scoreEvidence=[]; t.suspicionScore=th*0.15; // [FIX v5.1] evita re-trigger rápido
    }
    if (t.suspicionScore>=th*0.55) t.badges.push(`⚠ ${Math.round(t.suspicionScore)}pts`);
  }
  // ─────────────────────────────────────────────────────────────────
  // [M11] CARGA DEL MODELO DE SEGMENTACIÓN
  // ─────────────────────────────────────────────────────────────────
  async _loadSegModel() {
    if (_segLoading || _segReady) return;
    _segLoading = true;
    try {
      if (typeof ort === 'undefined') throw new Error('ort no disponible');
      for (const ep of ['webgl', 'wasm']) {
        try {
          _segSession = await ort.InferenceSession.create(SEG_MODEL, {
            executionProviders: [ep],
            graphOptimizationLevel: 'all',
          });
          _segReady = true;
          console.log(`%c⬟ YOLOv8n-seg ACTIVO (${ep.toUpperCase()}) — Silueta habilitada`, 'color:#ff6b35;font-weight:bold');
          return;
        } catch(e) { /* intentar siguiente EP */ }
      }
      throw new Error('no se pudo cargar yolov8n-seg.onnx');
    } catch(e) {
      _segLoading = false;
      console.info('%cℹ YOLOv8n-seg no disponible — detección de silueta desactivada', 'color:#888');
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // [M11] POSTPROCESO DE SEGMENTACIÓN
  // YOLOv8-seg output: [1, 116, 8400] + [1, 32, 160, 160]
  //   output0: detecciones (4 bbox + 1 conf + N clases + 32 mask coeffs)
  //   output1: prototipas de máscara 32×160×160
  // Solo procesamos detecciones de clase 0 (persona) con conf > SEG_CONF
  // Devuelve array de { nx1,ny1,nx2,ny2, mask: Uint8Array 160×160 }
  // ─────────────────────────────────────────────────────────────────
  _postprocessSeg(segOut, meta, poseDets) {
    try {
      const keys    = Object.keys(segOut);
      // output0 = detecciones, output1 = proto masks
      const det0    = segOut[keys[0]];
      const proto   = segOut[keys[1]];
      if (!det0 || !proto) return [];

      const detData  = det0.data;
      const S        = det0.dims[2];          // 8400 anchors
      const protoData= proto.data;
      const PH = proto.dims[2], PW = proto.dims[3]; // 160×160
      const NC_START = 4 + 1 + 80;            // offset coeficientes de máscara (tras bbox+conf+80cls)
      const { dx, dy, scale, vw, vh } = meta;
      const n = v => Math.max(0, Math.min(1, v));

      const results = [];
      for (let i = 0; i < S; i++) {
        // conf global = obj_conf * max_cls_conf (clase 0 = persona)
        const objConf = detData[4 * S + i];
        if (objConf < SEG_CONF) continue;
        const clsConf = detData[5 * S + i]; // clase 0 = persona
        if (clsConf < SEG_CONF) continue;

        const cx = detData[0*S+i], cy = detData[1*S+i];
        const bw = detData[2*S+i], bh = detData[3*S+i];
        const nx1 = n((cx-bw/2-dx)/(vw*scale)), ny1 = n((cy-bh/2-dy)/(vh*scale));
        const nx2 = n((cx+bw/2-dx)/(vw*scale)), ny2 = n((cy+bh/2-dy)/(vh*scale));

        // Extraer 32 coeficientes de máscara para este anchor
        const coeffs = new Float32Array(32);
        for (let k = 0; k < 32; k++) {
          coeffs[k] = detData[(NC_START + k) * S + i];
        }

        // Combinar coeficientes con prototipos: mask = sigmoid(sum_k(c_k * proto_k))
        const mask = new Uint8Array(PH * PW);
        for (let py = 0; py < PH; py++) {
          for (let px = 0; px < PW; px++) {
            let val = 0;
            for (let k = 0; k < 32; k++) {
              val += coeffs[k] * protoData[k * PH * PW + py * PW + px];
            }
            // sigmoid: 1/(1+e^-x) > 0.5 → píxel dentro de máscara
            mask[py * PW + px] = (1 / (1 + Math.exp(-val))) > 0.5 ? 1 : 0;
          }
        }

        results.push({ nx1, ny1, nx2, ny2, mask, PW, PH });
      }
      return this._nmsSeg(results);
    } catch(e) {
      console.warn('[SEG] postprocess error:', e.message);
      return [];
    }
  }

  // NMS básico para máscaras — elimina duplicados por IOU alto
  _nmsSeg(dets) {
    dets.sort((a, b) => (b.nx2-b.nx1)*(b.ny2-b.ny1) - (a.nx2-a.nx1)*(a.ny2-a.ny1));
    const keep = [], drop = new Set();
    for (let i = 0; i < dets.length; i++) {
      if (drop.has(i)) continue;
      keep.push(dets[i]);
      for (let j = i+1; j < dets.length; j++) {
        if (!drop.has(j) && this._iou(dets[i], dets[j]) > 0.5) drop.add(j);
      }
    }
    return keep.slice(0, 8);
  }

  // ─────────────────────────────────────────────────────────────────
  // [M11] ASOCIAR MÁSCARAS A TRACKS y actualizar estado silueta
  // Busca la máscara de seg más cercana al bbox del track
  // Extrae el área de silueta en región torso y cadera por separado
  // ─────────────────────────────────────────────────────────────────
  _updateSilhouetteTracks(now) {
    if (!this._lastSegMasks?.length) return;
    for (const t of this._tracks) {
      if (t.missed || t.isEmployee) continue;

      // Encontrar máscara que mejor solapa con bbox del track
      let bestMask = null, bestIou = 0.20;
      for (const m of this._lastSegMasks) {
        const iou = this._iou(t, m);
        if (iou > bestIou) { bestIou = iou; bestMask = m; }
      }
      if (!bestMask) continue;

      // Calcular área de silueta en región torso y cadera
      const areas = this._calcSilAreas(bestMask, t);
      t._silCurrent = areas;

      // Si postContact acaba de activarse y no hay snapshot → capturar baseline
      if (t.postContact && !t.postContact.fired && !t.silSnapshot) {
        t.silSnapshot    = { ...areas, capturedAt: now };
        t.silWaitFrames  = 0;
        t.silFired       = false;
      }
      // Si hay snapshot, incrementar contador de espera
      if (t.silSnapshot) t.silWaitFrames++;
    }
  }

  // Extrae qué fracción del bbox del track está cubierta por la máscara
  // en la región del torso y en la región de la cadera por separado.
  // Retorna { torsoArea: float [0,1], hipArea: float [0,1] }
  _calcSilAreas(segMask, track) {
    const { mask, PW, PH } = segMask;
    const bx1 = segMask.nx1, by1 = segMask.ny1;
    const bw  = segMask.nx2 - bx1, bh = segMask.ny2 - by1;

    // Región torso: franja vertical SIL_REGION_TORSO dentro del bbox
    const [tTop, tBot] = SIL_REGION_TORSO;
    const [hTop, hBot] = SIL_REGION_HIP;

    let torsoOn = 0, torsoTotal = 0, hipOn = 0, hipTotal = 0;

    for (let py = 0; py < PH; py++) {
      // Posición normalizada de este pixel dentro del bbox de la máscara
      const relY = py / PH;
      const isTorso = relY >= tTop && relY <= tBot;
      const isHip   = relY >= hTop && relY <= hBot;
      if (!isTorso && !isHip) continue;

      for (let px = 0; px < PW; px++) {
        const val = mask[py * PW + px];
        if (isTorso) { torsoTotal++; if (val) torsoOn++; }
        if (isHip)   { hipTotal++;   if (val) hipOn++;   }
      }
    }

    return {
      torsoArea: torsoTotal > 0 ? torsoOn / torsoTotal : 0,
      hipArea:   hipTotal   > 0 ? hipOn   / hipTotal   : 0,
    };
  }

  // ─────────────────────────────────────────────────────────────────
  // [M11] CHECK SILHOUETTE — llamado desde _analyze por track
  // Compara área actual vs snapshot. Si creció más de SIL_GROW_THRESH
  // en torso o cadera → algo fue incorporado al cuerpo → ALERTA SIL
  // ─────────────────────────────────────────────────────────────────
  _checkSilhouette(t, now) {
    if (!_segReady) return;
    if (!t.silSnapshot || t.silFired) return;
    if (!t.postContact || t.postContact.fired) {
      // postContact terminó sin alerta → limpiar snapshot
      t.silSnapshot = null; return;
    }
    if (t.silWaitFrames < SIL_FRAMES_WAIT) return; // esperar baseline estable

    const cur = t._silCurrent;
    if (!cur) return;

    const snap   = t.silSnapshot;
    const dTorso = cur.torsoArea - snap.torsoArea;
    const dHip   = cur.hipArea   - snap.hipArea;

    const torsoGrew = dTorso > SIL_GROW_THRESH;
    const hipGrew   = dHip   > SIL_GROW_THRESH;

    if (torsoGrew || hipGrew) {
      const region  = torsoGrew && hipGrew ? 'TORSO Y CADERA'
                    : torsoGrew ? 'TORSO' : 'CADERA';
      const pctT    = (dTorso * 100).toFixed(1);
      const pctH    = (dHip   * 100).toFixed(1);
      const label   = t.postContact?.label || 'OBJETO';

      this._fire(
        `sil_${t.id}`,
        `SILUETA — ${label} OCULTO EN ${region} (+${torsoGrew?pctT:pctH}% área)`,
        'high',
        this.config.cooldown * 1000
      );
      this._addScoreVigilant(t, 35, 'OCULT. SILUETA', now);
      t.silFired = true;
      t.postContact.silConfirmed = true;
      t.badges.push('⬟ SIL');

      // Marcar secuencia si aplica
      if (t.seqState.post === 0) t.seqState.post = now;

      console.log(
        `%c⬟ Track #${t.id} — SILUETA CRECIÓ: torso Δ${pctT}% hip Δ${pctH}%`,
        'color:#ff6b35;font-weight:bold'
      );
    }

    // Limpiar si postContact expiró
    if (now - t.postContact.disappearT > this._profile.postContactMs) {
      t.silSnapshot = null;
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // [M11] RENDER DE SILUETA — dibuja contorno de máscara sobre canvas
  // Solo si track tiene postContact activo o silFired (modo debug)
  // ─────────────────────────────────────────────────────────────────
  _drawSegMasks() {
    if (!_segReady || !this._lastSegMasks?.length) return;
    const ctx = this.ctx;
    const cw = this.canvas.width, ch = this.canvas.height;

    for (const m of this._lastSegMasks) {
      // Encontrar track asociado
      const track = this._tracks.find(t => !t.missed && this._iou(t, m) > 0.20);
      if (!track) continue;
      const hasPost   = track.postContact && !track.postContact.fired;
      const silFired  = track.silFired;
      if (!hasPost && !silFired) continue; // solo dibujar cuando es relevante

      const { mask, PW, PH } = m;
      const x1 = m.nx1 * cw, y1 = m.ny1 * ch;
      const mw  = (m.nx2 - m.nx1) * cw, mh = (m.ny2 - m.ny1) * ch;

      // Dibujar máscara semi-transparente sobre el área del track
      const col = silFired ? 'rgba(255,107,53,0.28)' : 'rgba(255,171,0,0.15)';
      ctx.save();
      ctx.beginPath();
      // Rasterizar máscara como grid de celdas
      const cellW = mw / PW, cellH = mh / PH;
      for (let py = 0; py < PH; py++) {
        for (let px = 0; px < PW; px++) {
          if (mask[py * PW + px]) {
            ctx.rect(x1 + px*cellW, y1 + py*cellH, cellW, cellH);
          }
        }
      }
      ctx.fillStyle = col;
      ctx.fill();

      // Borde de la silueta
      if (silFired) {
        ctx.strokeStyle = 'rgba(255,107,53,0.7)';
        ctx.lineWidth = 1.5;
        ctx.setLineDash([3, 3]);
        ctx.strokeRect(x1, y1, mw, mh);
        ctx.setLineDash([]);
        ctx.font = 'bold 9px "Share Tech Mono",monospace';
        ctx.fillStyle = '#ff6b35';
        ctx.fillText('⬟ SILUETA', x1 + 3, y1 - 4);
      }
      ctx.restore();

      // Marcar regiones torso/cadera analizadas con líneas sutiles
      if (hasPost) {
        ctx.save();
        const [tTop, tBot] = SIL_REGION_TORSO;
        const [hTop, hBot] = SIL_REGION_HIP;
        ctx.strokeStyle = 'rgba(255,171,0,0.25)';
        ctx.lineWidth = 1;
        ctx.setLineDash([2, 4]);
        // línea torso-top
        ctx.beginPath(); ctx.moveTo(x1, y1+mh*tTop); ctx.lineTo(x1+mw, y1+mh*tTop); ctx.stroke();
        // línea torso-bot / hip-top
        ctx.beginPath(); ctx.moveTo(x1, y1+mh*hTop); ctx.lineTo(x1+mw, y1+mh*hTop); ctx.stroke();
        ctx.setLineDash([]);
        ctx.restore();
      }
    }
  }

  _B(key) { return this._profile.scoreBonus?.[key]??15; }

  // ─────────────────────────────────────────────────────────────────
  // [M8] SMOOTHING + INTERPOLACIÓN DE KEYPOINTS
  // Aplicar antes de usar kps en análisis. Reduce ruido de cámara.
  // Inspirado en PoseLift WACV2025 (suavizado + interpolación lineal)
  // ─────────────────────────────────────────────────────────────────
  _smoothKps(t, rawKps) {
    const smoothed = [];
    for (let i = 0; i < 17; i++) {
      const raw = rawKps[i];
      const sm  = t.kpSmooth[i];
      const lv  = t.kpLastValid[i];

      if (raw && raw.c >= KP_THRESH) {
        // KP válido: actualizar suavizado EMA (exponential moving average)
        const alpha = 1 / KP_SMOOTH_FRAMES;
        if (sm.n === 0) {
          sm.x = raw.x; sm.y = raw.y; sm.c = raw.c;
        } else {
          sm.x = sm.x * (1-alpha) + raw.x * alpha;
          sm.y = sm.y * (1-alpha) + raw.y * alpha;
          sm.c = sm.c * (1-alpha) + raw.c * alpha;
        }
        sm.n++;
        t.kpLastValid[i] = {x: sm.x, y: sm.y, c: sm.c};
        t.kpMissingFrames[i] = 0;
        smoothed.push({x: sm.x, y: sm.y, c: sm.c});
      } else {
        // KP ausente: intentar interpolar desde último válido
        t.kpMissingFrames[i]++;
        if (lv && t.kpMissingFrames[i] <= KP_INTERP_MAX_GAP) {
          // Mantener última posición válida con confianza degradada
          const decayedConf = lv.c * (1 - t.kpMissingFrames[i] / KP_INTERP_MAX_GAP);
          smoothed.push({x: lv.x, y: lv.y, c: Math.max(0, decayedConf)});
        } else {
          smoothed.push(raw || {x:0, y:0, c:0});
        }
      }
    }
    return smoothed;
  }

  // ─────────────────────────────────────────────────────────────────
  // [M9] CLASIFICACIÓN DE INTENCIÓN — Estado de Alta Vigilancia
  // Si el sistema detecta escaneo, activa alta vigilancia por 8s.
  // Cualquier gesto sospechoso en esa ventana multiplica score ×3
  // ─────────────────────────────────────────────────────────────────
  _enterVigilance(t, now) {
    t.vigilanceUntil = now + VIGILANCE_WINDOW_MS;
    t.vigilanceCount++;
    t.badges.push('🔍 VIGILANCIA');
    console.log(`%c🔍 Track #${t.id} — ALTA VIGILANCIA activada (escaneo #${t.vigilanceCount})`,
      'color:#bf5af2;font-weight:bold');
  }

  // Suma score aplicando multiplicador si está en alta vigilancia
  _addScoreVigilant(t, pts, reason, now) {
    if (t.isEmployee) return;
    const inVigilance = now && now < t.vigilanceUntil;
    const multiplied  = inVigilance ? Math.round(pts * VIGILANCE_MULTIPLIER) : pts;
    const suffix      = inVigilance ? ` [×${VIGILANCE_MULTIPLIER} VIGILANCIA]` : '';
    this._addScore(t, multiplied, reason + suffix);
    if (inVigilance && multiplied > pts) {
      t.badges.push(`⚡ ×${VIGILANCE_MULTIPLIER}`);
    }
  }

  // ─────────────────────────────────────────────────────────────────
  // [M10] FILTRO DINÁMICO DE FALSAS BOLSAS
  // Descarta objetos BAG que llevan demasiados frames sin moverse
  // o que son demasiado pequeños relativos al cuerpo de la persona.
  // ─────────────────────────────────────────────────────────────────
  _isRealBag(obj, t) {
    // Filtro por escala: bolsa debe ser razonablemente grande vs persona
    const personH  = t.ny2 - t.ny1;
    const bagH     = obj.bbox.ny2 - obj.bbox.ny1;
    const bagW     = obj.bbox.nx2 - obj.bbox.nx1;
    if (personH > 0 && (bagH / personH < BAG_MIN_SCALE || bagW / personH < BAG_MIN_SCALE)) {
      return false; // demasiado pequeña → probablemente error de YOLO
    }

    // Filtro por movimiento: si no se ha movido en N frames → bolsa de fondo estático
    const key = obj.id;
    if (!t.bagStaticFrames[key]) {
      t.bagStaticFrames[key] = { frames: 0, lastX: obj.bbox.nx1, lastY: obj.bbox.ny1 };
    }
    const bs = t.bagStaticFrames[key];
    const moved = Math.hypot(obj.bbox.nx1 - bs.lastX, obj.bbox.ny1 - bs.lastY) > 0.005;
    if (moved) {
      bs.frames = 0; bs.lastX = obj.bbox.nx1; bs.lastY = obj.bbox.ny1;
    } else {
      bs.frames++;
    }
    if (bs.frames >= BAG_STATIC_FRAMES) return false; // estática = fondo
    return true;
  }

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
    this._drawSegMasks();          // [M11] silueta debajo de las detecciones
    this._drawDetections(this._lastDets);
    this._drawModelBadges();
  }

  // [M1][M2] Badges de modelos activos en esquina superior izquierda
  _drawModelBadges() {
    const ctx = this.ctx;
    ctx.save();
    ctx.font = 'bold 9px "Share Tech Mono",monospace';
    let y = 14;
    if (_objModelUsed === 'yoloe') {
      ctx.fillStyle = 'rgba(0,255,148,0.85)';
      ctx.fillText('⬡ YOLOE 1200+', 6, y); y += 13;
    } else if (_objModelUsed === 'yolo') {
      ctx.fillStyle = 'rgba(255,170,0,0.70)';
      ctx.fillText('⬡ YOLO 80cls', 6, y); y += 13;
    }
    if (_mpReady) {
      ctx.fillStyle = 'rgba(191,90,242,0.85)';
      ctx.fillText('🖐 MP-HAND', 6, y); y += 13;
    }
    if (_segReady) {
      ctx.fillStyle = 'rgba(255,107,53,0.85)';
      ctx.fillText('⬟ SEG-SIL', 6, y); y += 13;
    }
    ctx.restore();
  }
 _drawDetections(poseDets) {
  const ctx=this.ctx, cw=this.canvas.width, ch=this.canvas.height;

  // Objetos
  for (const obj of this._objTracker.alertVisible) {
    const {nx1,ny1,nx2,ny2}=obj.bbox;
    const x1=nx1*cw,y1=ny1*ch,x2=nx2*cw,y2=ny2*ch;

    const isBag=BAG_IDS.has(obj.cls);

    // FIX: declarar primero
    const rawLabel = obj.label;
    const confPct  = Math.round(obj.conf * 100);
    const isBagLow = isBag && obj.conf < 0.62;
    const isAnyLow = obj.conf < 0.45;

    // Color por certeza
    const col = isBagLow  ? 'rgba(140,140,160,0.6)'
              : isBag     ? 'rgba(191,90,242,0.9)'
              : isAnyLow  ? 'rgba(180,130,0,0.65)'
              :             'rgba(255,170,0,0.85)';

    ctx.save();
    ctx.strokeStyle=col;
    ctx.lineWidth=1.8;

    ctx.setLineDash([4,3]);
    ctx.strokeRect(x1,y1,x2-x1,y2-y1);
    ctx.setLineDash([]);

    // Label inteligente
    const dispLabel = isBagLow  ? `OBJETO? ${confPct}%`
                    : isAnyLow  ? `${rawLabel}? ${confPct}%`
                    :             `${rawLabel} ${confPct}%`;

    const lbl = dispLabel;

    ctx.font='9px "Share Tech Mono",monospace';
    const lw2=ctx.measureText(lbl).width+6;

    ctx.fillStyle=isBag
      ? 'rgba(191,90,242,0.15)'
      : 'rgba(255,170,0,0.15)';

    ctx.fillRect(x1,y1-14,lw2,13);

    ctx.fillStyle=col;
    ctx.fillText(lbl,x1+3,y1-4);

    ctx.restore();
  }

  // Personas
  for (const det of poseDets) {

    const k=det.kps;

    const x1=det.nx1*cw;
    const y1=det.ny1*ch;
    const x2=det.nx2*cw;
    const y2=det.ny2*ch;

    const track=this._tracks.find(t=>!t.missed && this._iou(t,det)>0.3);

    const isEmp=track?.isEmployee;
    const inZone=track && Object.values(track.inZoneWrist||{}).some(v=>v);

    const hasPost=track?.postContact && !track.postContact.fired;

    const scanning=track?.badges?.includes('⚠ ESCANEO');

    const silGrow=track?.badges?.some(b=>b.includes('SIL'));

    const hipHide=(track?.hipConcealment??0)>2;

    const hasCom=track?.badges?.some(b=>
      b.includes('CÓMPLICE') || b.includes('BLOQUEADO')
    );

    const boxCol =
      isEmp ? 'rgba(0,230,118,0.6)' :
      hasCom ? '#ff6b35' :
      inZone ? '#ff3d3d' :
      hipHide ? '#ff6b35' :
      silGrow ? '#ff6b35' :
      hasPost ? '#ffaa00' :
      scanning ? '#bf5af2' :
      'rgba(0,200,255,0.45)';

    ctx.save();

    ctx.strokeStyle=boxCol;
    ctx.lineWidth=(inZone||hasPost)?2:1.5;

    ctx.strokeRect(x1,y1,x2-x1,y2-y1);

    ctx.fillStyle=boxCol;
    ctx.font='10px "Share Tech Mono",monospace';

    ctx.fillText(`${isEmp?'👷':''}${Math.round(det.conf*100)}%`,x1+3,y1-3);

    if (track && track.suspicionScore>25 && !isEmp) {

      const th=this._profile.scoreThreshold;
      const sc=track.suspicionScore;

      ctx.fillStyle =
        sc>=th*0.8 ? '#ff3d3d' :
        sc>=th*0.5 ? '#ffaa00' :
        '#ffee58';

      ctx.font='bold 9px "Share Tech Mono",monospace';

      ctx.fillText(`${Math.round(sc)}pts`,x1+3,y2-5);
    }

    ctx.restore();

    // Esqueleto
    ctx.save();
    ctx.lineWidth=1.8;

    for (const [a,b] of BONES) {

      const pa=k[a];
      const pb=k[b];

      if (!_ok(pa) || !_ok(pb)) continue;

      ctx.beginPath();
      ctx.moveTo(pa.x*cw,pa.y*ch);
      ctx.lineTo(pb.x*cw,pb.y*ch);

      ctx.strokeStyle =
        isEmp
        ? 'rgba(0,230,118,0.4)'
        : 'rgba(0,200,255,0.5)';

      ctx.globalAlpha=0.75;

      ctx.stroke();
    }

    ctx.globalAlpha=1;

    for (let i=0;i<17;i++) {

      const p=k[i];
      if (!_ok(p)) continue;

      const isW=i===KP.L_WRIST||i===KP.R_WRIST;
      const isH=i===KP.L_HIP||i===KP.R_HIP;

      const inZ=isW &&
        this.zoneManager.getZonesForPoint(p.x,p.y).length>0;

      const onO=isW &&
        this._objTracker.alertVisible.some(o=>{
          const m=0.06;
          return (
            p.x>=o.bbox.nx1-m &&
            p.x<=o.bbox.nx2+m &&
            p.y>=o.bbox.ny1-m &&
            p.y<=o.bbox.ny2+m
          );
        });

      ctx.beginPath();

      ctx.arc(
        p.x*cw,
        p.y*ch,
        isW?6:isH?4:3,
        0,
        Math.PI*2
      );

      ctx.fillStyle =
        isEmp ? 'rgba(0,230,118,0.8)' :
        inZ ? '#ff3d3d' :
        isW ? '#ffb800' :
        isH ? '#bf5af2' :
        'rgba(255,255,255,0.7)';

      ctx.fill();

      if ((inZ||onO) && !isEmp) {

        ctx.beginPath();

        ctx.arc(p.x*cw,p.y*ch,11,0,Math.PI*2);

        ctx.strokeStyle=inZ?'#ff3d3d':'#ffb800';
        ctx.lineWidth=1.5;

        ctx.globalAlpha=
          0.5+0.5*Math.sin(Date.now()/200);

        ctx.stroke();

        ctx.globalAlpha=1;
      }

      if (isH && (track?.hipConcealment??0)>0) {

        ctx.beginPath();

        ctx.arc(p.x*cw,p.y*ch,13,0,Math.PI*2);

        ctx.strokeStyle='#ff6b35';
        ctx.lineWidth=1.5;

        ctx.globalAlpha=
          0.3+0.4*Math.sin(Date.now()/250);

        ctx.stroke();

        ctx.globalAlpha=1;
      }
    }

    ctx.restore();
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
      wristVelHist:{L:[],R:[]},seqState:{scan:0,zone:0,post:0},seqBonusFired:false,
      mpGripConf:{L:0,R:0},mpPalmIn:{L:false,R:false},
      kpSmooth:Array.from({length:17},()=>({x:0,y:0,c:0,n:0})),
      kpLastValid:Array.from({length:17},()=>null),kpMissingFrames:new Array(17).fill(0),
      vigilanceUntil:0,vigilanceCount:0,bagStaticFrames:{},silSnapshot:null,silWaitFrames:0,silFired:false,
    });
  }
  stop()          { this.active=false; }
  updateConfig(c) { Object.assign(this.config,c); if (c.storeType) this.setStoreType(c.storeType); }
  destroy()       { if (this._renderLoopId) cancelAnimationFrame(this._renderLoopId); }
}