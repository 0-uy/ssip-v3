// ═══════════════════════════════════════════════════════════
//  firebase-config.js — Configuración central de Firebase
//  SSIP v3.0 · Multi-tenant · Planes · Auto-borrado 48hs
// ═══════════════════════════════════════════════════════════

import { initializeApp } from "https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js";
import { getAuth, signInWithEmailAndPassword, signOut, onAuthStateChanged }
  from "https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js";
import { getFirestore, collection, doc, addDoc, setDoc, getDoc, getDocs,
         deleteDoc, query, where, orderBy, updateDoc, serverTimestamp }
  from "https://www.gstatic.com/firebasejs/10.7.1/firebase-firestore.js";

// ─── CREDENCIALES FIREBASE ───────────────────────────────
const firebaseConfig = {
  apiKey:            "AIzaSyC4lNSyk8VvQGu0Nb7zfvUsfc34K-xtUzk",
  authDomain:        "ssip-seguridad.firebaseapp.com",
  projectId:         "ssip-seguridad",
  storageBucket:     "ssip-seguridad.firebasestorage.app",
  messagingSenderId: "905868905769",
  appId:             "1:905868905769:web:cf9ff453f99b18e1cdfb38"
};

const app  = initializeApp(firebaseConfig);
const auth = getAuth(app);
const db   = getFirestore(app);

// ─── ADMIN ───────────────────────────────────────────────
const ADMIN_EMAIL = "ssip.admi@gmail.com";

// ─── DEFINICIÓN DE PLANES ────────────────────────────────
export const PLANES = {
  basico: {
    id:          'basico',
    nombre:      'Básico',
    camaras:     1,           // máx cámaras simultáneas
    multiview:   false,       // acceso a multicam.html
    historial:   48,          // horas de historial
    color:       '#00d4ff',
    badge:       'BÁSICO',
  },
  pro: {
    id:          'pro',
    nombre:      'Profesional',
    camaras:     3,
    multiview:   true,
    historial:   48,
    color:       '#ffb800',
    badge:       'PRO',
  },
};

// ═══════════════════════════════════════════════════════════
//  AUTH
// ═══════════════════════════════════════════════════════════

export async function login(email, password) {
  const cred = await signInWithEmailAndPassword(auth, email, password);
  return cred.user;
}

export async function logout() {
  await signOut(auth);
  window.location.href = 'login.html';
}

export function onAuth(callback) {
  return onAuthStateChanged(auth, callback);
}

export function currentUser() {
  return auth.currentUser;
}

export function isAdmin(user) {
  return user && user.email === ADMIN_EMAIL;
}

// ═══════════════════════════════════════════════════════════
//  EMPRESAS
// ═══════════════════════════════════════════════════════════

export async function getEmpresa(uid) {
  const snap = await getDoc(doc(db, 'empresas', uid));
  return snap.exists() ? { id: snap.id, ...snap.data() } : null;
}

export async function getAllEmpresas() {
  const snap = await getDocs(collection(db, 'empresas'));
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

export async function createEmpresa(uid, data) {
  await setDoc(doc(db, 'empresas', uid), {
    ...data,
    plan:     data.plan || 'basico',
    creadoEn: serverTimestamp(),
    activo:   true,
  });
}

export async function updateEmpresa(uid, data) {
  await updateDoc(doc(db, 'empresas', uid), data);
}

// ─── Helper: obtener el plan activo de un usuario ────────
export async function getPlanEmpresa(uid) {
  const empresa = await getEmpresa(uid);
  const planId  = empresa?.plan || 'basico';
  return PLANES[planId] || PLANES.basico;
}

// ═══════════════════════════════════════════════════════════
//  EVENTOS — auto-borrado 48hs desde el cliente
// ═══════════════════════════════════════════════════════════

export async function guardarEvento(empresaId, evento) {
  const ahora = Date.now();
  await addDoc(collection(db, 'eventos'), {
    empresaId,
    tipo:      evento.tipo,
    severidad: evento.severidad,
    snapshot:  evento.snapshot || '',
    camaraIdx: evento.camaraIdx ?? 0,   // ← NEW: índice de cámara (0,1,2)
    favorito:  false,
    tsMs:      ahora,
    expiraEn:  ahora + (48 * 60 * 60 * 1000),
  });
}

export async function getEventos(empresaId) {
  await limpiarEventosViejos(empresaId);
  const hace48h = Date.now() - (48 * 60 * 60 * 1000);
  const q = query(
    collection(db, 'eventos'),
    where('empresaId', '==', empresaId),
    where('tsMs', '>', hace48h),
    orderBy('tsMs', 'desc')
  );
  const snap = await getDocs(q);
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

export async function limpiarEventosViejos(empresaId) {
  const hace48h = Date.now() - (48 * 60 * 60 * 1000);
  const q = query(
    collection(db, 'eventos'),
    where('empresaId', '==', empresaId),
    where('tsMs', '<', hace48h)
  );
  const snap = await getDocs(q);
  await Promise.all(snap.docs.map(d => deleteDoc(d.ref)));
}

export async function toggleFavorito(eventoId, valorActual) {
  await updateDoc(doc(db, 'eventos', eventoId), { favorito: !valorActual });
}

export async function deleteEvento(eventoId) {
  await deleteDoc(doc(db, 'eventos', eventoId));
}

// ═══════════════════════════════════════════════════════════
//  ZONAS — configuración por empresa y por cámara
// ═══════════════════════════════════════════════════════════

// camaraIdx: 0 = cam principal, 1 = cam 2, 2 = cam 3
export async function getZonas(empresaId, camaraIdx = 0) {
  const snap = await getDocs(
    collection(db, 'empresas', empresaId, `zonas_cam${camaraIdx}`)
  );
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

export async function saveZona(empresaId, zonaId, data, camaraIdx = 0) {
  await setDoc(
    doc(db, 'empresas', empresaId, `zonas_cam${camaraIdx}`, zonaId),
    { ...data, actualizadoEn: serverTimestamp() }
  );
}

export async function deleteZona(empresaId, zonaId, camaraIdx = 0) {
  await deleteDoc(
    doc(db, 'empresas', empresaId, `zonas_cam${camaraIdx}`, zonaId)
  );
}

// Retrocompat: zonas de cámara principal (índice 0)
export async function getZonasPrincipal(empresaId) {
  // Intenta nueva colección primero, cae a la vieja
  const zonas = await getZonas(empresaId, 0);
  if (zonas.length > 0) return zonas;
  const snap = await getDocs(
    collection(db, 'empresas', empresaId, 'zonas')
  );
  return snap.docs.map(d => ({ id: d.id, ...d.data() }));
}

export { db, auth };