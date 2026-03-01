/**
 * detection.js — SSIP v4.0  orIGIN
 * ─────────────────────────────────────────────────────────────────────────
 * Modelos requeridos en la carpeta del proyecto:
 *   yolov8n-pose.onnx  →  https://huggingface.co/Xenova/yolov8-pose-onnx/resolve/main/yolov8n-pose.onnx
 *   yolov8n.onnx       →  https://huggingface.co/Xenova/yolov8-onnx/resolve/main/yolov8n.onnx
 *
 * Detecta:
 *   · Múltiples personas (hasta 8) con esqueleto completo
 *   · Manos en zonas + permanencia + patrón escape al torso
 *   · Manos ocultas / en bolsillo
 *   · Brazos cruzados
 *   · Objetos (80 clases COCO)
 *   · Interacción mano–objeto: mano toca objeto → objeto desaparece → ALERTA
 */

const POSE_MODEL = './yolov8n-pose.onnx';
const OBJ_MODEL  = './yolov8n.onnx';
const INPUT_W    = 640;
const INPUT_H    = 640;
const CONF_POSE  = 0.30;
const CONF_OBJ   = 0.35;
const KP_THRESH  = 0.25;
const IOU_THRESH = 0.45;

const KP = {
  NOSE:0,L_EYE:1,R_EYE:2,L_EAR:3,R_EAR:4,
  L_SHOULDER:5,R_SHOULDER:6,L_ELBOW:7,R_ELBOW:8,
  L_WRIST:9,R_WRIST:10,L_HIP:11,R_HIP:12,
  L_KNEE:13,R_KNEE:14,L_ANKLE:15,R_ANKLE:16,
};

const BONES = [
  [5,6],[5,7],[7,9],[6,8],[8,10],
  [5,11],[6,12],[11,12],
  [11,13],[13,15],[12,14],[14,16],
];

const OBJ_CLASSES = {
  24:'mochila',25:'paraguas',26:'bolso',27:'corbata',28:'valija',
  39:'botella',40:'copa',41:'taza',42:'tenedor',43:'cuchillo',
  44:'cuchara',45:'tazón',46:'banana',47:'manzana',
  56:'silla',57:'sofá',63:'laptop',64:'mouse',65:'control',
  66:'teclado',67:'celular',73:'libro',74:'reloj',75:'jarrón',76:'tijera',
};

const ALERT_CLASSES = new Set([24,26,28,39,41,43,63,67,73,74,75,76]);

let _poseSession = null;
let _objSession  = null;
let _posePromise = null;
let _objPromise  = null;

export class DetectionEngine {
  constructor(canvas, zoneManager, alertManager, config = {}) {
    this.canvas       = canvas;
    this.ctx          = canvas.getContext('2d');
    this.zoneManager  = zoneManager;
    this.alertManager = alertManager;
    this.config = {
      movementThreshold: config.movementThreshold ?? 50,
      dwellTime:         config.dwellTime         ?? 3,
      cooldown:          config.cooldown          ?? 8,
    };
    this.active = false;
    this._off    = document.createElement('canvas');
    this._off.width = INPUT_W; this._off.height = INPUT_H;
    this._offCtx = this._off.getContext('2d', { willReadFrequently: true });
    this._tracks      = [];
    this._nextId      = 0;
    this._maxHistory  = 30;
    this._objDets     = [];
    this._interactions = {};
    this._lastAlert   = {};
    this._fpsFrames   = 0;
    this._fpsLast     = performance.now();
    this.currentFPS   = 0;
    this._renderLoopId = null;
    this._lastDets    = [];
    this.onDetection  = null;
  }

  async init() {
    if (!_posePromise) _posePromise = this._loadModel(POSE_MODEL, 'pose');
    if (!_objPromise)  _objPromise  = this._loadModel(OBJ_MODEL, 'obj');
    const [poseResult] = await Promise.allSettled([_posePromise, _objPromise]);
    if (poseResult.status === 'rejected') throw new Error('No se pudo cargar yolov8n-pose.onnx');
    this._startRenderLoop();
  }

  async _loadModel(path, name) {
    if (typeof ort === 'undefined') throw new Error('ONNX Runtime no cargado');
    ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.18.0/dist/';
    for (const ep of ['webgl','wasm']) {
      try {
        const s = await ort.InferenceSession.create(path, {
          executionProviders: [ep], graphOptimizationLevel: 'all',
        });
        if (name === 'pose') _poseSession = s;
        else                 _objSession  = s;
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
        if (this._lastDets.length) this._drawDetections(this._lastDets, this._objDets);
      }
      this._renderLoopId = requestAnimationFrame(loop);
    };
    this._renderLoopId = requestAnimationFrame(loop);
  }

  async processFrame(video) {
    if (!this.active || !_poseSession) return;
    this._fpsFrames++;
    const now = performance.now();
    if (now - this._fpsLast >= 1000) {
      this.currentFPS = this._fpsFrames; this._fpsFrames = 0; this._fpsLast = now;
    }
    let tensor, meta;
    try { [tensor, meta] = this._preprocess(video); } catch { return; }

    let poseDets = [];
    try {
      const out = await _poseSession.run({ images: tensor });
      poseDets = this._postprocessPose(out.output0 || out[Object.keys(out)[0]], meta);
    } catch(e) { console.warn('Pose error:', e.message); }

    let objDets = [];
    if (_objSession) {
      try {
        const out = await _objSession.run({ images: tensor });
        objDets = this._postprocessObj(out.output0 || out[Object.keys(out)[0]], meta);
      } catch(e) { console.warn('Obj error:', e.message); }
    }

    if (typeof tensor?.dispose === 'function') tensor.dispose();

    this._updateTracks(poseDets, Date.now());
    this._objDets  = objDets;
    this._lastDets = poseDets;
    this._render();
  }

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
    const N = INPUT_W*INPUT_H, f32 = new Float32Array(3*N);
    for (let i=0;i<N;i++) {
      f32[i]=px[i*4]/255; f32[N+i]=px[i*4+1]/255; f32[2*N+i]=px[i*4+2]/255;
    }
    return [new ort.Tensor('float32',f32,[1,3,INPUT_H,INPUT_W]),{dx,dy,scale,vw,vh}];
  }

  _postprocessPose(output, {dx,dy,scale,vw,vh}) {
    const data=output.data, S=output.dims[2], dets=[];
    for (let i=0;i<S;i++) {
      const conf=data[4*S+i]; if(conf<CONF_POSE) continue;
      const cx=data[0*S+i],cy=data[1*S+i],bw=data[2*S+i],bh=data[3*S+i];
      const nx1=Math.max(0,Math.min(1,(cx-bw/2-dx)/(vw*scale)));
      const ny1=Math.max(0,Math.min(1,(cy-bh/2-dy)/(vh*scale)));
      const nx2=Math.max(0,Math.min(1,(cx+bw/2-dx)/(vw*scale)));
      const ny2=Math.max(0,Math.min(1,(cy+bh/2-dy)/(vh*scale)));
      const kps=[];
      for (let k=0;k<17;k++) kps.push({
        x:Math.max(0,Math.min(1,(data[(5+k*3)*S+i]-dx)/(vw*scale))),
        y:Math.max(0,Math.min(1,(data[(5+k*3+1)*S+i]-dy)/(vh*scale))),
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
      for (let c=0;c<80;c++) { const sc=data[(4+c)*S+i]; if(sc>bestConf){bestConf=sc;bestCls=c;} }
      if(bestCls<0||!OBJ_CLASSES[bestCls]) continue;
      const cx=data[0*S+i],cy=data[1*S+i],bw=data[2*S+i],bh=data[3*S+i];
      dets.push({
        cls:bestCls, conf:bestConf, label:OBJ_CLASSES[bestCls],
        nx1:Math.max(0,Math.min(1,(cx-bw/2-dx)/(vw*scale))),
        ny1:Math.max(0,Math.min(1,(cy-bh/2-dy)/(vh*scale))),
        nx2:Math.max(0,Math.min(1,(cx+bw/2-dx)/(vw*scale))),
        ny2:Math.max(0,Math.min(1,(cy+bh/2-dy)/(vh*scale))),
      });
    }
    return this._nms(dets).slice(0,20);
  }

  _nms(dets) {
    if(!dets.length) return [];
    dets.sort((a,b)=>b.conf-a.conf);
    const keep=[], drop=new Set();
    for(let i=0;i<dets.length;i++) {
      if(drop.has(i)) continue; keep.push(dets[i]);
      for(let j=i+1;j<dets.length;j++)
        if(!drop.has(j)&&this._iou(dets[i],dets[j])>IOU_THRESH) drop.add(j);
    }
    return keep;
  }

  _iou(a,b) {
    const ix1=Math.max(a.nx1,b.nx1),iy1=Math.max(a.ny1,b.ny1);
    const ix2=Math.min(a.nx2,b.nx2),iy2=Math.min(a.ny2,b.ny2);
    const I=Math.max(0,ix2-ix1)*Math.max(0,iy2-iy1);
    return I/((a.nx2-a.nx1)*(a.ny2-a.ny1)+(b.nx2-b.nx1)*(b.ny2-b.ny1)-I+1e-6);
  }

  _updateTracks(dets, now) {
    const matched=new Set();
    for(const t of this._tracks) {
      let best=-1,bestIou=0.10;
      for(let i=0;i<dets.length;i++) {
        if(matched.has(i)) continue;
        const iou=this._iou(t,dets[i]);
        if(iou>bestIou){best=i;bestIou=iou;}
      }
      if(best>=0) {
        const d=dets[best];
        Object.assign(t,{kps:d.kps,nx1:d.nx1,ny1:d.ny1,nx2:d.nx2,ny2:d.ny2,missed:0});
        t.history.push({kps:d.kps,t:now});
        if(t.history.length>this._maxHistory) t.history.shift();
        matched.add(best);
      } else { t.missed=(t.missed||0)+1; }
    }
    this._tracks=this._tracks.filter(t=>(t.missed||0)<10);
    for(let i=0;i<dets.length;i++) {
      if(!matched.has(i)) {
        const d=dets[i];
        this._tracks.push({
          id:this._nextId++,kps:d.kps,
          nx1:d.nx1,ny1:d.ny1,nx2:d.nx2,ny2:d.ny2,
          missed:0,history:[{kps:d.kps,t:now}],
          inZoneWrist:{},dwellStart:{},pocketL:0,pocketR:0,crossedArms:0,
        });
      }
    }
    for(const t of this._tracks) if(!t.missed) this._analyze(t,now);
  }

  _analyze(t, now) {
    const k=t.kps;
    const lw=k[KP.L_WRIST],rw=k[KP.R_WRIST];
    const lh=k[KP.L_HIP],  rh=k[KP.R_HIP];
    const le=k[KP.L_ELBOW],re=k[KP.R_ELBOW];
    const ls=k[KP.L_SHOULDER],rs=k[KP.R_SHOULDER];

    // 1. Manos en zona
    for(const [w,side] of [[lw,'L'],[rw,'R']]) {
      if(!w||w.c<KP_THRESH) {
        for(const key of Object.keys(t.inZoneWrist))
          if(key.startsWith(side+'_')){ t.inZoneWrist[key]=false; t.dwellStart[key]=null; }
        continue;
      }
      const zones=this.zoneManager.getZonesForPoint(w.x,w.y);
      for(const zone of zones) {
        const key=`${side}_${zone.id}`;
        if(!t.inZoneWrist[key]) {
          t.inZoneWrist[key]=true; t.dwellStart[key]=now;
          zone.alert=true; setTimeout(()=>{if(zone)zone.alert=false;},2000);
          this._fire(`zone_enter_${key}`,`MANO EN ${zone.name.toUpperCase()}`,'low',3000);
        }
        const elapsed=(now-(t.dwellStart[key]||now))/1000;
        if(elapsed>=this.config.dwellTime) {
          t.dwellStart[key]=now+this.config.dwellTime*1000;
          this._fire(`dwell_${key}`,`PERMANENCIA — ${zone.name.toUpperCase()}`,'high',this.config.cooldown*1000);
        }
        if(t.history.length>=10) this._detectEscape(t,side,zone,lh,rh);
      }
      if(zones.length===0)
        for(const key of Object.keys(t.inZoneWrist))
          if(key.startsWith(side+'_')){ t.inZoneWrist[key]=false; t.dwellStart[key]=null; }
    }

    // 2. Bolsillos
    this._detectPocket(t,lw,lh,ls,'L');
    this._detectPocket(t,rw,rh,rs,'R');

    // 3. Brazos cruzados
    this._detectCrossedArms(t,le,re,lw,rw,ls,rs,lh,rh);

    // 4. Interacción mano–objeto
    this._detectHandObj(t,lw,rw,now);
  }

  _detectEscape(t,side,zone,lh,rh) {
    if(!lh||!rh) return;
    const tx=(lh.x+rh.x)/2, ty=(lh.y+rh.y)/2;
    const old=t.history[t.history.length-8], cur=t.history[t.history.length-1];
    if(!old||!cur) return;
    const idx=side==='L'?KP.L_WRIST:KP.R_WRIST;
    const pw=old.kps[idx], cw=cur.kps[idx];
    if(!pw||!cw||pw.c<KP_THRESH||cw.c<KP_THRESH) return;
    const wasIn=this.zoneManager.getZonesForPoint(pw.x,pw.y).some(z=>z.id===zone.id);
    if(!wasIn) return;
    const pd=Math.hypot(pw.x-tx,pw.y-ty), cd=Math.hypot(cw.x-tx,cw.y-ty);
    if(cd<pd*0.65&&pd>0.08)
      this._fire(`escape_${zone.id}_${side}`,`OBJETO OCULTADO — ${zone.name.toUpperCase()}`,'high',this.config.cooldown*1000);
  }

  _detectPocket(t,wrist,hip,shoulder,side) {
    if(!hip||!shoulder) return;
    const hx=hip.x, hy=hip.y;
    let pocket=false;
    if(!wrist||wrist.c<0.20)        pocket=true;
    else if(wrist.c<0.50)           pocket=Math.abs(wrist.x-hx)<0.18&&Math.abs(wrist.y-hy)<0.22;
    else pocket=wrist.y>hy-0.05&&Math.abs(wrist.x-hx)<0.15&&Math.abs(wrist.y-hy)<0.19;
    const sk=side==='L'?'pocketL':'pocketR';
    if(pocket){ t[sk]++;
      if(t[sk]>=12){ t[sk]=0;
        this._fire(`pocket_${side}_${t.id}`,`MANO ${side==='L'?'IZQ.':'DER.'} OCULTA`,'high',this.config.cooldown*1000);
      }
    } else t[sk]=Math.max(0,t[sk]-2);
  }

  _detectCrossedArms(t,le,re,lw,rw,ls,rs,lh,rh) {
    if(!le||!re||!ls||!rs||!lh||!rh) return;
    const mx=(ls.x+rs.x)/2, my=(ls.y+rs.y)/2, hy=(lh.y+rh.y)/2;
    const ok=Math.abs(le.x-mx)<0.20&&Math.abs(re.x-mx)<0.20&&le.x>mx&&re.x<mx
           &&le.y>my&&le.y<hy+0.08&&re.y>my&&re.y<hy+0.08
           &&((!lw||lw.c<0.40)||(!rw||rw.c<0.40));
    if(ok){ t.crossedArms++;
      if(t.crossedArms>=15){ t.crossedArms=0;
        this._fire(`crossed_${t.id}`,'BRAZOS CRUZADOS — POSIBLE OCULTAMIENTO','high',this.config.cooldown*1000);
      }
    } else t.crossedArms=Math.max(0,t.crossedArms-2);
  }

  _detectHandObj(t,lw,rw,now) {
    if(!this._objDets.length) return;
    for(const [w,side] of [[lw,'L'],[rw,'R']]) {
      if(!w||w.c<KP_THRESH) continue;
      for(const obj of this._objDets) {
        if(!ALERT_CLASSES.has(obj.cls)) continue;
        const m=0.06;
        const touching=w.x>=obj.nx1-m&&w.x<=obj.nx2+m&&w.y>=obj.ny1-m&&w.y<=obj.ny2+m;
        const intKey=`${t.id}_${obj.cls}_${side}`;
        if(touching) {
          if(!this._interactions[intKey])
            this._interactions[intKey]={startT:now,objBox:{nx1:obj.nx1,ny1:obj.ny1,nx2:obj.nx2,ny2:obj.ny2},label:obj.label};
          const zones=this.zoneManager.getZonesForPoint((obj.nx1+obj.nx2)/2,(obj.ny1+obj.ny2)/2);
          if(zones.length>0)
            this._fire(`obj_zone_${intKey}`,`${obj.label.toUpperCase()} EN ${zones[0].name.toUpperCase()}`,'high',this.config.cooldown*1000);
        } else if(this._interactions[intKey]) {
          const d=this._interactions[intKey]; delete this._interactions[intKey];
          const dur=now-d.startT; if(dur<300) continue;
          const stillThere=this._objDets.some(o=>o.cls===obj.cls&&this._iou(o,d.objBox)>0.30);
          if(!stillThere&&dur>800)
            this._fire(`obj_gone_${t.id}_${obj.cls}_${side}`,`OBJETO TOMADO: ${d.label.toUpperCase()}`,'high',this.config.cooldown*1000);
        }
      }
      // Limpiar interacciones viejas
      for(const k of Object.keys(this._interactions))
        if(k.startsWith(`${t.id}_`)&&now-this._interactions[k].startT>8000) delete this._interactions[k];
    }
  }

  _fire(key, type, severity, coolMs) {
    const now=Date.now();
    if(now-(this._lastAlert[key]||0)<coolMs) return;
    this._lastAlert[key]=now;
    if(this.onDetection) this.onDetection(type,severity);
    if(this.alertManager) this.alertManager.trigger(type,severity);
  }

  _render() {
    const ctx=this.ctx;
    ctx.clearRect(0,0,this.canvas.width,this.canvas.height);
    const anyAlert=this.zoneManager.zones.some(z=>z.alert);
    this.zoneManager.drawZone(anyAlert);
    this.zoneManager.drawPreview();
    this._drawDetections(this._lastDets,this._objDets);
  }

  _drawDetections(poseDets, objDets) {
    const ctx=this.ctx, cw=this.canvas.width, ch=this.canvas.height;

    // Objetos
    for(const obj of (objDets||[])) {
      const x1=obj.nx1*cw,y1=obj.ny1*ch,x2=obj.nx2*cw,y2=obj.ny2*ch;
      const alert=ALERT_CLASSES.has(obj.cls);
      const col=alert?'rgba(255,170,0,0.85)':'rgba(160,160,160,0.5)';
      ctx.save();
      ctx.strokeStyle=col; ctx.lineWidth=alert?1.8:1;
      ctx.setLineDash([4,3]); ctx.strokeRect(x1,y1,x2-x1,y2-y1); ctx.setLineDash([]);
      ctx.font='9px "Share Tech Mono",monospace';
      const lbl=`${obj.label} ${Math.round(obj.conf*100)}%`;
      const lw2=ctx.measureText(lbl).width+6;
      ctx.fillStyle=alert?'rgba(255,170,0,0.15)':'rgba(40,40,40,0.4)';
      ctx.fillRect(x1,y1-14,lw2,13);
      ctx.fillStyle=col; ctx.fillText(lbl,x1+3,y1-4);
      ctx.restore();
    }

    // Personas
    for(const det of poseDets) {
      const k=det.kps;
      const x1=det.nx1*cw,y1=det.ny1*ch,x2=det.nx2*cw;
      const track=this._tracks.find(t=>!t.missed&&this._iou(t,det)>0.3);
      const inZone=track&&Object.values(track.inZoneWrist||{}).some(v=>v);

      ctx.save();
      ctx.strokeStyle=inZone?'#ff3d3d':'rgba(0,200,255,0.45)';
      ctx.lineWidth=inZone?2:1.5;
      ctx.strokeRect(x1,y1,x2-x1,(det.ny2-det.ny1)*ch);
      ctx.fillStyle=inZone?'#ff3d3d':'rgba(0,200,255,0.7)';
      ctx.font='10px "Share Tech Mono",monospace';
      ctx.fillText(`${Math.round(det.conf*100)}%`,x1+3,y1-3);
      ctx.restore();

      // Esqueleto
      ctx.save(); ctx.lineWidth=1.8;
      for(const [a,b] of BONES) {
        const pa=k[a],pb=k[b];
        if(!pa||!pb||pa.c<KP_THRESH||pb.c<KP_THRESH) continue;
        ctx.beginPath(); ctx.moveTo(pa.x*cw,pa.y*ch); ctx.lineTo(pb.x*cw,pb.y*ch);
        ctx.strokeStyle='rgba(0,200,255,0.5)'; ctx.globalAlpha=0.75; ctx.stroke();
      }
      ctx.globalAlpha=1;

      // Keypoints
      for(let i=0;i<17;i++) {
        const p=k[i]; if(!p||p.c<KP_THRESH) continue;
        const isWrist=i===KP.L_WRIST||i===KP.R_WRIST;
        const isHip=i===KP.L_HIP||i===KP.R_HIP;
        const inZ=isWrist&&this.zoneManager.getZonesForPoint(p.x,p.y).length>0;
        const touchingObj=isWrist&&(objDets||[]).some(o=>{
          const m=0.06;
          return ALERT_CLASSES.has(o.cls)&&p.x>=o.nx1-m&&p.x<=o.nx2+m&&p.y>=o.ny1-m&&p.y<=o.ny2+m;
        });
        ctx.beginPath();
        ctx.arc(p.x*cw,p.y*ch,isWrist?6:isHip?4:3,0,Math.PI*2);
        ctx.fillStyle=inZ?'#ff3d3d':isWrist?'#ffb800':isHip?'#bf5af2':'rgba(255,255,255,0.7)';
        ctx.fill();
        if(inZ||touchingObj) {
          ctx.beginPath();
          ctx.arc(p.x*cw,p.y*ch,11,0,Math.PI*2);
          ctx.strokeStyle=inZ?'#ff3d3d':'#ffb800';
          ctx.lineWidth=1.5; ctx.globalAlpha=0.5+0.5*Math.sin(Date.now()/200);
          ctx.stroke(); ctx.globalAlpha=1;
        }
      }
      ctx.restore();

      // Badges
      if(track) {
        const badges=[];
        if(track.pocketL>8||track.pocketR>8) badges.push({text:'⚠ BOLSILLO',color:'rgba(255,58,58,0.9)'});
        if(track.crossedArms>8) badges.push({text:'⚠ CRUZADO',color:'rgba(255,170,0,0.9)'});
        if(Object.values(track.inZoneWrist||{}).some(v=>v)) badges.push({text:'⚠ EN ZONA',color:'#ff3d3d'});
        ctx.save(); ctx.font='bold 9px "Share Tech Mono",monospace';
        let bx=det.nx1*cw; const by=det.ny2*ch+13;
        for(const b of badges){ ctx.fillStyle=b.color; ctx.fillText(b.text,bx,by); bx+=ctx.measureText(b.text).width+8; }
        ctx.restore();
      }
    }
  }

  start() {
    this.active=true; this._lastAlert={}; this._interactions={};
    for(const t of this._tracks) Object.assign(t,{inZoneWrist:{},dwellStart:{},pocketL:0,pocketR:0,crossedArms:0});
  }
  stop()          { this.active=false; }
  updateConfig(c) { Object.assign(this.config,c); }
  destroy()       { if(this._renderLoopId) cancelAnimationFrame(this._renderLoopId); }
}