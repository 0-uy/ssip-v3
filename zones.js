/**
 * zones.js — SSIP v2.2   - ORIGINAL 
 * + Polígono editable: arrastrar vértices después de dibujar
 * + Fix resize: coordenadas normalizadas se mantienen
 * + Instrucción en pantalla durante dibujo
 */

const ZONE_COLORS = [
  { border: '#00d4ff', fill: 'rgba(0,212,255,0.07)' },
  { border: '#00ff94', fill: 'rgba(0,255,148,0.07)' },
  { border: '#ffb800', fill: 'rgba(255,184,0,0.07)'  },
  { border: '#bf5af2', fill: 'rgba(191,90,242,0.07)' },
  { border: '#ff6b35', fill: 'rgba(255,107,53,0.07)' },
  { border: '#00e5ff', fill: 'rgba(0,229,255,0.07)'  },
];

// Palabras clave que identifican automáticamente una zona de pago
const PAY_KEYWORDS = /caja|pago|cobro|checkout|registro|factura|ticket|pos|cajero/i;

// Colores especiales para zona de pago (verde)
const PAY_ZONE_COLOR = { border: '#00e676', fill: 'rgba(0,230,118,0.07)' };

const VERTEX_RADIUS   = 8;   // área clickeable para arrastrar
const VERTEX_SNAP     = 15;  // píxeles para snap al primer punto

export class ZoneManager {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx    = canvas.getContext('2d');
    this.zones  = [];
    this.maxZones = 6;

    // Dibujo polígono
    this._drawing     = false;
    this._enabled     = false;
    this._points      = [];
    this._mouseX      = 0;
    this._mouseY      = 0;
    this._pendingName = 'Zona';
    this._pendingType = 'alerta'; // 'alerta' | 'pago'

    // Edición de vértices
    this._editMode    = false;
    this._dragging    = null;  // { zoneId, pointIdx }
    this._hovering    = null;  // { zoneId, pointIdx }

    this._onZoneChange = null;

    // Trash icon hover state
    this._trashHoverZoneId = null;  // zone id actualmente mostrando el tarro
    this._trashHideTimer   = null;  // timer para ocultarlo
    this._trashRects       = [];    // [{zoneId, x, y, size}] — posiciones clickeables
    this._bTrashMove       = this._onTrashMove.bind(this);
    this._bTrashClick      = this._onTrashClick.bind(this);
    this.canvas.addEventListener('mousemove', this._bTrashMove);
    this.canvas.addEventListener('click',     this._bTrashClick);

    this._bClick  = this._onClick.bind(this);
    this._bMove   = this._onMove.bind(this);
    this._bDbl    = this._onDblClick.bind(this);
    this._bKey    = this._onKey.bind(this);
    this._bDown   = this._onMouseDown.bind(this);
    this._bUp     = this._onMouseUp.bind(this);
    this._bMoveEd = this._onMoveEdit.bind(this);
  }

  /* ── Activar dibujo ─────────────────────────────────── */
  enableDraw(name = 'Zona', type = null) {
    if (this.zones.length >= this.maxZones) return false;
    this._enabled = true;
    this._drawing = false;
    this._points  = [];
    this._pendingName = name;
    // Auto-detectar tipo por nombre si no se pasa explícitamente
    this._pendingType = type ?? (PAY_KEYWORDS.test(name) ? 'pago' : 'alerta');
    this._disableEdit();
    this.canvas.style.cursor = 'crosshair';
    this.canvas.addEventListener('click',    this._bClick);
    this.canvas.addEventListener('mousemove',this._bMove);
    this.canvas.addEventListener('dblclick', this._bDbl);
    window.addEventListener('keydown',       this._bKey);
    return true;
  }

  disableDraw() {
    this._enabled = false;
    this._drawing = false;
    this._points  = [];
    this.canvas.style.cursor = 'default';
    this.canvas.removeEventListener('click',    this._bClick);
    this.canvas.removeEventListener('mousemove',this._bMove);
    this.canvas.removeEventListener('dblclick', this._bDbl);
    window.removeEventListener('keydown',       this._bKey);
  }

  /* ── Activar edición de vértices ────────────────────── */
  enableEdit() {
    this._editMode = true;
    this.canvas.style.cursor = 'default';
    this.canvas.addEventListener('mousedown', this._bDown);
    this.canvas.addEventListener('mousemove', this._bMoveEd);
    window.addEventListener('mouseup', this._bUp);
  }

  disableEdit() { this._disableEdit(); }

  _disableEdit() {
    this._editMode  = false;
    this._dragging  = null;
    this._hovering  = null;
    this.canvas.removeEventListener('mousedown', this._bDown);
    this.canvas.removeEventListener('mousemove', this._bMoveEd);
    window.removeEventListener('mouseup', this._bUp);
  }

  /* ── Zona management ────────────────────────────────── */
  removeZone(id) {
    this.zones = this.zones.filter(z => z.id !== id);
    if (this._onZoneChange) this._onZoneChange(this.zones);
  }

  clearAllZones() {
    this.zones = [];
    if (this._onZoneChange) this._onZoneChange(this.zones);
  }

  onZoneChange(fn) { this._onZoneChange = fn; }

  /* ── Point-in-polygon (ray casting) ────────────────── */
  _pointInPoly(x, y, pts) {
    let inside = false;
    for (let i=0, j=pts.length-1; i<pts.length; j=i++) {
      const xi=pts[i].x, yi=pts[i].y, xj=pts[j].x, yj=pts[j].y;
      if (((yi>y)!==(yj>y)) && (x < (xj-xi)*(y-yi)/(yj-yi)+xi)) inside=!inside;
    }
    return inside;
  }

  isInZone(xN, yN) {
    return this.zones.some(z => this._pointInPoly(xN, yN, z.points));
  }

  getZonesForPoint(xN, yN) {
    return this.zones.filter(z => this._pointInPoly(xN, yN, z.points));
  }

  // Retorna true si existe al menos una zona de tipo 'pago'
  // Usado por detection.js para la lógica adaptativa de alertas
  hasPayZone() {
    return this.zones.some(z => z.type === 'pago');
  }

  get zone() { return this.zones[0] || null; }

  /* ── Render zonas finalizadas ────────────────────────── */
  drawZone(isAlert = false) {
    for (const zone of this.zones) this._drawPoly(zone, isAlert && zone.alert);
    if (this._editMode) this._drawEditHandles();
  }

  _drawPoly(zone, isAlert) {
    if (!zone.points || zone.points.length < 2) return;
    const ctx=this.ctx, cw=this.canvas.width, ch=this.canvas.height;

    // Zona de pago → color verde fijo; zona de alerta → colores normales
    const isPay  = zone.type === 'pago';
    const color  = isPay ? PAY_ZONE_COLOR : ZONE_COLORS[zone.colorIdx % ZONE_COLORS.length];
    const sc     = isAlert ? '#ff3d3d' : color.border;
    const fc     = isAlert ? 'rgba(255,61,61,0.1)' : color.fill;

    ctx.save();
    ctx.beginPath();
    ctx.moveTo(zone.points[0].x*cw, zone.points[0].y*ch);
    for (let i=1; i<zone.points.length; i++) ctx.lineTo(zone.points[i].x*cw, zone.points[i].y*ch);
    ctx.closePath();
    ctx.fillStyle=fc; ctx.fill();
    ctx.strokeStyle=sc; ctx.lineWidth=2; ctx.setLineDash([6,3]); ctx.stroke(); ctx.setLineDash([]);

    // Vértices (solo si no está en modo edición — los handles se dibujan aparte)
    if (!this._editMode) {
      ctx.fillStyle=sc;
      for (const p of zone.points) {
        ctx.beginPath(); ctx.arc(p.x*cw, p.y*ch, 4, 0, Math.PI*2); ctx.fill();
      }
    }

    // Etiqueta centroide
    const cx = zone.points.reduce((s,p)=>s+p.x,0)/zone.points.length;
    const cy = zone.points.reduce((s,p)=>s+p.y,0)/zone.points.length;
    const label = isPay ? `💳 ${zone.name}` : zone.name;
    ctx.font='bold 11px "Share Tech Mono",monospace';
    const lw=ctx.measureText(label).width;
    ctx.fillStyle=sc; ctx.fillRect(cx*cw-lw/2-5, cy*ch-18, lw+10, 16);
    ctx.fillStyle='#000'; ctx.fillText(label, cx*cw-lw/2, cy*ch-5);

    // Tarro de basura (solo cuando el mouse está cerca de esta zona)
    if (this._trashHoverZoneId === zone.id) {
      const TRASH_SIZE = 22;
      const tx = cx * cw + lw / 2 + 10;  // a la derecha de la etiqueta
      const ty = cy * ch - 18;

      // Fondo del botón
      ctx.save();
      ctx.globalAlpha = 0.92;
      ctx.fillStyle   = 'rgba(255,58,58,0.85)';
      ctx.strokeStyle = '#ff3a3a';
      ctx.lineWidth   = 1;
      ctx.beginPath();
      ctx.roundRect(tx, ty, TRASH_SIZE, TRASH_SIZE, 5);
      ctx.fill(); ctx.stroke();

      // Icono tarro (SVG path dibujado con canvas)
      ctx.globalAlpha = 1;
      ctx.strokeStyle = '#fff';
      ctx.lineWidth   = 1.5;
      ctx.lineCap     = 'round';
      const ox = tx + TRASH_SIZE / 2, oy = ty + TRASH_SIZE / 2;
      // tapa
      ctx.beginPath(); ctx.moveTo(ox - 5, oy - 5); ctx.lineTo(ox + 5, oy - 5); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ox - 2, oy - 7); ctx.lineTo(ox + 2, oy - 7); ctx.stroke();
      // cuerpo
      ctx.beginPath();
      ctx.moveTo(ox - 4, oy - 4); ctx.lineTo(ox - 4, oy + 5);
      ctx.lineTo(ox + 4, oy + 5); ctx.lineTo(ox + 4, oy - 4);
      ctx.closePath(); ctx.stroke();
      // líneas internas
      ctx.beginPath(); ctx.moveTo(ox - 1.5, oy - 2); ctx.lineTo(ox - 1.5, oy + 3); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(ox + 1.5, oy - 2); ctx.lineTo(ox + 1.5, oy + 3); ctx.stroke();
      ctx.restore();

      // Registrar área clickeable
      const existing = this._trashRects.findIndex(r => r.zoneId === zone.id);
      const rect = { zoneId: zone.id, x: tx, y: ty, size: TRASH_SIZE };
      if (existing >= 0) this._trashRects[existing] = rect;
      else this._trashRects.push(rect);
    } else {
      // Limpiar rect si ya no se muestra
      this._trashRects = this._trashRects.filter(r => r.zoneId !== zone.id);
    }

    ctx.restore();
  }

  /* ── Handles de edición ──────────────────────────────── */
  _drawEditHandles() {
    const ctx=this.ctx, cw=this.canvas.width, ch=this.canvas.height;
    ctx.save();
    for (const zone of this.zones) {
      const color = ZONE_COLORS[zone.colorIdx % ZONE_COLORS.length];
      for (let i=0; i<zone.points.length; i++) {
        const p = zone.points[i];
        const px=p.x*cw, py=p.y*ch;
        const isHover  = this._hovering?.zoneId===zone.id && this._hovering?.idx===i;
        const isDrag   = this._dragging?.zoneId===zone.id && this._dragging?.idx===i;

        // Outer glow
        if (isHover || isDrag) {
          ctx.globalAlpha=0.3;
          ctx.fillStyle=color.border;
          ctx.beginPath(); ctx.arc(px, py, VERTEX_RADIUS+5, 0, Math.PI*2); ctx.fill();
          ctx.globalAlpha=1;
        }
        // Handle circle
        ctx.fillStyle  = isDrag ? '#fff' : (isHover ? color.border : '#0c1520');
        ctx.strokeStyle= color.border;
        ctx.lineWidth  = 2;
        ctx.beginPath(); ctx.arc(px, py, VERTEX_RADIUS, 0, Math.PI*2);
        ctx.fill(); ctx.stroke();
      }
    }

    // Instrucción modo edición
    ctx.font='12px "Barlow",sans-serif';
    ctx.fillStyle='rgba(0,212,255,0.8)';
    ctx.fillText('✎ Modo edición — arrastrá los puntos para mover la zona', 12, 28);
    ctx.restore();
  }

  /* ── Preview del polígono en construcción ────────────── */
  drawPreview() {
    if (!this._enabled || this._points.length === 0) return;
    const ctx=this.ctx, cw=this.canvas.width, ch=this.canvas.height;
    const color = ZONE_COLORS[this.zones.length % ZONE_COLORS.length];

    ctx.save();
    // Líneas ya trazadas
    ctx.strokeStyle=color.border; ctx.lineWidth=2; ctx.setLineDash([4,3]);
    ctx.beginPath();
    ctx.moveTo(this._points[0].x*cw, this._points[0].y*ch);
    for (let i=1; i<this._points.length; i++) ctx.lineTo(this._points[i].x*cw, this._points[i].y*ch);
    // Línea al cursor
    ctx.lineTo(this._mouseX, this._mouseY);
    ctx.stroke(); ctx.setLineDash([]);

    // Relleno semi-transparente del área ya trazada
    if (this._points.length >= 2) {
      ctx.beginPath();
      ctx.moveTo(this._points[0].x*cw, this._points[0].y*ch);
      for (let i=1; i<this._points.length; i++) ctx.lineTo(this._points[i].x*cw, this._points[i].y*ch);
      ctx.closePath();
      ctx.fillStyle=color.fill; ctx.fill();
    }

    // Puntos colocados
    for (let i=0; i<this._points.length; i++) {
      const p=this._points[i];
      ctx.fillStyle=color.border;
      ctx.beginPath(); ctx.arc(p.x*cw, p.y*ch, 5, 0, Math.PI*2); ctx.fill();
      // Número de punto
      ctx.fillStyle='#000'; ctx.font='bold 9px sans-serif';
      ctx.fillText(i+1, p.x*cw-3, p.y*ch+3);
    }

    // Círculo "snap" en primer punto
    if (this._points.length >= 3) {
      ctx.strokeStyle='rgba(255,255,255,0.8)'; ctx.lineWidth=1.5;
      ctx.beginPath(); ctx.arc(this._points[0].x*cw, this._points[0].y*ch, VERTEX_SNAP, 0, Math.PI*2);
      ctx.stroke();
    }

    // Instrucciones en pantalla
    ctx.fillStyle='rgba(0,212,255,0.9)'; ctx.font='12px "Barlow",sans-serif';
    const pts=this._points.length;
    const hint = pts < 3
      ? `Clic para agregar punto (${pts} de mín. 3)`
      : `${pts} puntos — Doble clic, Enter o clic en ○ para cerrar`;
    ctx.fillText(hint, 12, 28);
    if (pts > 0) ctx.fillText('Backspace = borrar último punto · Escape = cancelar', 12, 46);

    ctx.restore();
  }

  /* ── Edición: mouse down — buscar vértice ────────────── */
  _getCanvasCoords(e) {
    const r=this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX-r.left)*(this.canvas.width /r.width),
      y: (e.clientY-r.top) *(this.canvas.height/r.height),
    };
  }

  _findVertex(cx, cy) {
    const cw=this.canvas.width, ch=this.canvas.height;
    for (const zone of this.zones) {
      for (let i=0; i<zone.points.length; i++) {
        const px=zone.points[i].x*cw, py=zone.points[i].y*ch;
        if (Math.sqrt((cx-px)**2+(cy-py)**2) <= VERTEX_RADIUS+4) {
          return { zoneId: zone.id, idx: i };
        }
      }
    }
    return null;
  }

  _onMouseDown(e) {
    if (!this._editMode) return;
    const {x,y}=this._getCanvasCoords(e);
    const hit=this._findVertex(x,y);
    if (hit) { this._dragging=hit; this.canvas.style.cursor='grabbing'; }
  }

  _onMoveEdit(e) {
    if (!this._editMode) return;
    const {x,y}=this._getCanvasCoords(e);
    const cw=this.canvas.width, ch=this.canvas.height;

    if (this._dragging) {
      const zone=this.zones.find(z=>z.id===this._dragging.zoneId);
      if (zone) {
        zone.points[this._dragging.idx]={
          x: Math.max(0,Math.min(1, x/cw)),
          y: Math.max(0,Math.min(1, y/ch)),
        };
      }
      this.canvas.style.cursor='grabbing';
    } else {
      const hit=this._findVertex(x,y);
      this._hovering=hit;
      this.canvas.style.cursor=hit?'grab':'default';
    }
  }

  _onMouseUp() {
    if (this._dragging) {
      this._dragging=null;
      this.canvas.style.cursor='grab';
      if (this._onZoneChange) this._onZoneChange(this.zones);
    }
  }

  /* ── Dibujo: mouse events ────────────────────────────── */
  _getDrawCoords(e) {
    const r=this.canvas.getBoundingClientRect();
    return {
      x: (e.clientX-r.left)*(this.canvas.width /r.width),
      y: (e.clientY-r.top) *(this.canvas.height/r.height),
    };
  }

  _onClick(e) {
    if (!this._enabled) return;
    e.preventDefault();
    const {x,y}=this._getDrawCoords(e);
    const cw=this.canvas.width, ch=this.canvas.height;

    // Snap al primer punto para cerrar
    if (this._points.length >= 3) {
      const p0=this._points[0];
      const dx=x-p0.x*cw, dy=y-p0.y*ch;
      if (Math.sqrt(dx*dx+dy*dy) < VERTEX_SNAP) { this._finalize(); return; }
    }
    this._points.push({ x: x/cw, y: y/ch });
  }

  _onMove(e) {
    const {x,y}=this._getDrawCoords(e);
    this._mouseX=x; this._mouseY=y;
  }

  _onDblClick(e) {
    e.preventDefault();
    if (this._points.length >= 3) this._finalize();
  }

  _onKey(e) {
    if (e.key==='Enter'    && this._points.length>=3) this._finalize();
    if (e.key==='Escape')  { this.disableDraw(); if(this._onZoneChange)this._onZoneChange(this.zones); }
    if (e.key==='Backspace'&& this._points.length>0) this._points.pop();
  }

  _finalize() {
    if (this._points.length < 3) return;
    this.zones.push({
      id:       `zone-${Date.now()}`,
      name:     this._pendingName,
      type:     this._pendingType,   // 'alerta' | 'pago'
      points:   [...this._points],
      colorIdx: this.zones.length,
      alert:    false,
    });
    this.disableDraw();
    if (this._onZoneChange) this._onZoneChange(this.zones);
  }

  /* ── Trash icon: hover detection ────────────────────────────────── */
  _onTrashMove(e) {
    if (this._drawing || this._editMode) return;
    const { x, y } = this._getCanvasCoords(e);
    const cw = this.canvas.width, ch = this.canvas.height;

    // ¿Está sobre el icono tarro?
    const hitTrash = this._trashRects.find(r =>
      x >= r.x && x <= r.x + r.size &&
      y >= r.y && y <= r.y + r.size
    );

    // ¿Está cerca del centroide de alguna zona? (radio 48px)
    const nearZone = !hitTrash && this.zones.find(z => {
      const cx = z.points.reduce((s, p) => s + p.x, 0) / z.points.length * cw;
      const cy = z.points.reduce((s, p) => s + p.y, 0) / z.points.length * ch;
      return Math.sqrt((x - cx) ** 2 + (y - cy) ** 2) < 48;
    });

    const target = hitTrash?.zoneId || nearZone?.id || null;

    if (target) {
      clearTimeout(this._trashHideTimer);
      this._trashHoverZoneId = target;
      this.canvas.style.cursor = hitTrash ? 'pointer' : '';
    } else if (this._trashHoverZoneId) {
      clearTimeout(this._trashHideTimer);
      this._trashHideTimer = setTimeout(() => {
        this._trashHoverZoneId = null;
      }, 500);
    }
  }

  _onTrashClick(e) {
    if (this._drawing || this._editMode) return;
    const { x, y } = this._getCanvasCoords(e);
    const hit = this._trashRects.find(r =>
      x >= r.x && x <= r.x + r.size &&
      y >= r.y && y <= r.y + r.size
    );
    if (hit) {
      e.stopPropagation();
      this.removeZone(hit.zoneId);
      this._trashHoverZoneId = null;
      this._trashRects = [];
    }
  }

  /* ── Firebase ────────────────────────────────────────── */
  toFirebaseData() {
    return this.zones.map(z=>({
      id:       z.id,
      name:     z.name,
      type:     z.type || 'alerta',   // backward compat
      points:   z.points,
      colorIdx: z.colorIdx,
    }));
  }

  loadFromFirebase(data) {
    this.zones = data.map(z=>({
      ...z,
      type:  z.type || (PAY_KEYWORDS.test(z.name) ? 'pago' : 'alerta'), // retrocompat
      alert: false,
    }));
    if (this._onZoneChange) this._onZoneChange(this.zones);
  }
}