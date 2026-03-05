/**
 * store-profiles.js — SSIP v5.0
 * ─────────────────────────────────────────────────────────────────────
 * Perfil de detección por tipo de local.
 * Cada perfil ajusta:
 *   · Sensibilidad de contacto y permanencia
 *   · Qué objetos vigilar (familias YOLO-agnósticas)
 *   · Score threshold para confirmar robo
 *   · Qué comportamientos habilitar/deshabilitar
 *   · Textos de alerta genéricos (no "taza", sino "objeto pequeño")
 *
 * Uso en detection.js:
 *   import { getProfile } from './store-profiles.js';
 *   const profile = getProfile('farmacia');
 */

// ─────────────────────────────────────────────────────────────────────
//  FAMILIAS DE OBJETOS — agnósticas al label específico de YOLO
//  Agrupa clases que se confunden entre sí (taza/botella/jarrón, etc.)
// ─────────────────────────────────────────────────────────────────────
export const OBJ_FAMILIES = {
  SMALL: {
    ids:    new Set([39, 41, 44, 45, 75, 76]),  // botella, taza, cuchara, tazón, reloj, jarrón, tijera
    label:  'OBJETO PEQUEÑO',
    minConf: 0.42,   // más exigente: son los que más se confunden
  },
  MEDIUM: {
    ids:    new Set([40, 42, 43, 46, 47, 67, 73]),  // copa, tenedor, cuchillo, frutas, celular, libro
    label:  'OBJETO',
    minConf: 0.38,
  },
  BAG: {
    ids:    new Set([24, 26, 28]),                  // mochila, bolso, valija
    label:  'BOLSO/MOCHILA',
    minConf: 0.35,
  },
  TECH: {
    ids:    new Set([63, 64, 65, 66]),              // laptop, mouse, control, teclado
    label:  'DISPOSITIVO',
    minConf: 0.33,
  },
  JEWELRY: {
    ids:    new Set([74]),                          // reloj (único de joyería en COCO)
    label:  'JOYA/RELOJ',
    minConf: 0.48,
  },
};

/** Dado un cls de YOLO, retorna la familia o null si no es de alerta */
export function getFamily(cls) {
  for (const [key, fam] of Object.entries(OBJ_FAMILIES)) {
    if (fam.ids.has(cls)) return { key, ...fam };
  }
  return null;
}

/** Clases de bolsos (para bag-stuffing) */
export const BAG_IDS = OBJ_FAMILIES.BAG.ids;

/** Todas las clases de alerta */
export const ALERT_IDS = new Set([
  ...OBJ_FAMILIES.SMALL.ids,
  ...OBJ_FAMILIES.MEDIUM.ids,
  ...OBJ_FAMILIES.BAG.ids,
  ...OBJ_FAMILIES.TECH.ids,
  ...OBJ_FAMILIES.JEWELRY.ids,
]);

// ─────────────────────────────────────────────────────────────────────
//  PERFILES POR TIPO DE LOCAL
// ─────────────────────────────────────────────────────────────────────

const PROFILES = {

  // ── Genérico (fallback) ──────────────────────────────────────────
  generico: {
    name:           'Genérico',
    icon:           '🏪',
    dwellTime:      4,       // segundos en zona para alerta de permanencia
    contactMinMs:   400,     // ms mínimo de contacto para O1
    grabMaxMs:      700,     // ms máximo para considerar "arrebato"
    scoreThreshold: 72,
    postContactMs:  4000,
    zoneEntryFrames: 3,
    hipConcealConf: 0.55,
    families:       ['SMALL','MEDIUM','BAG','TECH'],
    behaviors: {
      merodeo:      true,
      escaneo:      true,
      pantalla:     true,
      cadera:       true,
      manga:        true,
      agachado:     true,
      bagStuffing:  true,
      traspaso:     true,
      distractor:   true,
      trayectoria:  true,
    },
  },

  // ── Supermercado / Almacén / Minimercado ─────────────────────────
  supermercado: {
    name:           'Supermercado / Almacén',
    icon:           '🛒',
    dwellTime:      5,
    contactMinMs:   500,     // más tolerante: los clientes legítimos revisan productos
    grabMaxMs:      700,
    scoreThreshold: 75,
    postContactMs:  4000,
    zoneEntryFrames: 4,      // más frames para confirmar entrada (mucho tráfico)
    hipConcealConf: 0.55,
    families:       ['SMALL','MEDIUM','BAG'],
    behaviors: {
      merodeo:      true,
      escaneo:      true,
      pantalla:     true,
      cadera:       true,
      manga:        true,
      agachado:     true,
      bagStuffing:  true,
      traspaso:     true,
      distractor:   true,
      trayectoria:  false,   // demasiado tráfico → muchos falsos positivos
    },
    // Puntos de score ajustados
    scoreBonus: {
      contacto:     10,      // menos puntos por contacto (es normal revisar productos)
      objetoTomado: 35,
      bagStuffing:  45,
    },
  },

  // ── Farmacia ─────────────────────────────────────────────────────
  farmacia: {
    name:           'Farmacia',
    icon:           '💊',
    dwellTime:      3,
    contactMinMs:   350,
    grabMaxMs:      600,
    scoreThreshold: 68,
    postContactMs:  4000,
    zoneEntryFrames: 3,
    hipConcealConf: 0.52,
    families:       ['SMALL','MEDIUM'],
    behaviors: {
      merodeo:      true,
      escaneo:      true,
      pantalla:     true,
      cadera:       true,
      manga:        true,
      agachado:     true,
      bagStuffing:  true,
      traspaso:     true,
      distractor:   true,
      trayectoria:  true,
    },
  },

  // ── Kiosco / Cafetería ───────────────────────────────────────────
  kiosco: {
    name:           'Kiosco / Cafetería',
    icon:           '☕',
    dwellTime:      2,       // local chico, permanencia corta es sospechosa
    contactMinMs:   300,
    grabMaxMs:      600,
    scoreThreshold: 65,
    postContactMs:  3000,
    zoneEntryFrames: 2,
    hipConcealConf: 0.52,
    families:       ['SMALL','MEDIUM','BAG'],
    behaviors: {
      merodeo:      true,
      escaneo:      true,
      pantalla:     true,
      cadera:       true,
      manga:        true,
      agachado:     true,
      bagStuffing:  true,
      traspaso:     true,
      distractor:   false,   // local chico, una sola persona por vez
      trayectoria:  true,
    },
    scoreBonus: {
      contacto:     20,      // más puntos: en kiosco no debería haber mucho contacto
    },
  },

  // ── Joyería ──────────────────────────────────────────────────────
  joyeria: {
    name:           'Joyería',
    icon:           '💎',
    dwellTime:      2,       // cualquier permanencia cerca de vitrinas es sospechosa
    contactMinMs:   200,     // contacto brevísimo ya es relevante
    grabMaxMs:      500,
    scoreThreshold: 55,      // muy sensible — los objetos valen mucho
    postContactMs:  5000,    // ventana más larga para detectar destino
    zoneEntryFrames: 2,
    hipConcealConf: 0.60,    // más sensible a manos ocultas
    families:       ['JEWELRY','SMALL','MEDIUM'],
    behaviors: {
      merodeo:      true,
      escaneo:      true,
      pantalla:     true,
      cadera:       true,
      manga:        true,
      agachado:     false,   // en joyería el cliente no se agacha al piso
      bagStuffing:  false,   // improbable en joyería (vitrinas con asistente)
      traspaso:     true,
      distractor:   true,    // técnica muy usada en joyerías
      trayectoria:  true,
    },
    scoreBonus: {
      contacto:     25,      // cualquier contacto en joyería es relevante
      objetoTomado: 55,
      escaneo:      20,
    },
  },

  // ── Tienda de Ropa ───────────────────────────────────────────────
  ropa: {
    name:           'Tienda de Ropa',
    icon:           '👕',
    dwellTime:      6,       // los clientes pasan tiempo mirando → umbral más alto
    contactMinMs:   600,
    grabMaxMs:      800,
    scoreThreshold: 78,
    postContactMs:  5000,    // ropa puede meterse bajo ropa (layering) → más tiempo
    zoneEntryFrames: 4,
    hipConcealConf: 0.50,
    families:       ['BAG'],  // YOLO no detecta ropa como clase de alerta
    behaviors: {
      merodeo:      true,
      escaneo:      true,
      pantalla:     true,
      cadera:       true,    // layering: meterse ropa bajo ropa
      manga:        true,    // manga: meterse prenda en la manga
      agachado:     true,    // agacharse para esconder en bolso
      bagStuffing:  true,    // lo más común en tiendas de ropa
      traspaso:     true,
      distractor:   true,
      trayectoria:  false,
    },
    scoreBonus: {
      bagStuffing:  50,      // muy probable en ropa
      manga:        40,
    },
  },

  // ── Bazar / Tienda variada ───────────────────────────────────────
  bazar: {
    name:           'Bazar / Tienda variada',
    icon:           '🏬',
    dwellTime:      4,
    contactMinMs:   400,
    grabMaxMs:      700,
    scoreThreshold: 70,
    postContactMs:  4000,
    zoneEntryFrames: 3,
    hipConcealConf: 0.53,
    families:       ['SMALL','MEDIUM','BAG','TECH'],
    behaviors: {
      merodeo:      true,
      escaneo:      true,
      pantalla:     true,
      cadera:       true,
      manga:        true,
      agachado:     true,
      bagStuffing:  true,
      traspaso:     true,
      distractor:   true,
      trayectoria:  true,
    },
  },

  // ── Depósito / Bodega (acceso solo empleados) ────────────────────
  deposito: {
    name:           'Depósito / Bodega',
    icon:           '📦',
    dwellTime:      2,
    contactMinMs:   300,
    grabMaxMs:      600,
    scoreThreshold: 60,
    postContactMs:  4000,
    zoneEntryFrames: 2,
    hipConcealConf: 0.50,
    families:       ['SMALL','MEDIUM','BAG','TECH'],
    behaviors: {
      merodeo:      false,   // si está en depósito, la sola presencia es sospechosa
      escaneo:      true,
      pantalla:     true,
      cadera:       true,
      manga:        true,
      agachado:     true,
      bagStuffing:  true,
      traspaso:     true,
      distractor:   false,
      trayectoria:  false,
    },
    scoreBonus: {
      contacto:     20,      // cualquier contacto en depósito es relevante
    },
  },

  // ── Cocina / Área de preparación ────────────────────────────────
  cocina: {
  name:           'Cocina / Área de preparación',
  icon:           '🍳',
  dwellTime:      8,
  contactMinMs:   800,
  grabMaxMs:      1000,
  scoreThreshold: 82,        // un poco más alto
  postContactMs:  3000,
  zoneEntryFrames: 5,
  hipConcealConf: 0.60,       // más estricto para ocultamiento
  families:       ['SMALL','MEDIUM','TECH'],
  behaviors: {
    merodeo:      false,
    escaneo:      false,
    pantalla:     false,
    cadera:       true,
    manga:        true,      // ← ACTIVADO
    agachado:     false,
    bagStuffing:  true,
    traspaso:     true,      // ← útil en robo interno
    distractor:   false,
    trayectoria:  false,
  },
  scoreBonus: {
    contacto:     5,         // manipular es normal
    cadera:       45,
    manga:        45,
    bagStuffing:  55,
    traspaso:     45,
  },
},
};

// Alias y variantes de nombres
const ALIASES = {
  minimercado: 'supermercado',
  'mini mercado': 'supermercado',
  almacen:    'supermercado',
  almacén:    'supermercado',
  cafeteria:  'kiosco',
  cafetería:  'kiosco',
  cafe:       'kiosco',
  café:       'kiosco',
  tienda:     'bazar',
  farmacia:   'farmacia',
  joyeria:    'joyeria',
  joyería:    'joyeria',
  ropa:       'ropa',
  vestimenta: 'ropa',
  deposito:   'deposito',
  depósito:   'deposito',
  bodega:     'deposito',
  cocina:     'cocina',
  kiosko:     'kiosco',
  kiosco:     'kiosco',
};

/**
 * Retorna el perfil para un tipo de local dado.
 * Si no reconoce el tipo, retorna el genérico.
 * @param {string} type - nombre del tipo de local
 * @returns {object} perfil completo con defaults rellenados
 */
export function getProfile(type = 'generico') {
  const normalized = (type || '').toLowerCase().trim();
  const key = ALIASES[normalized] || normalized;
  const base = PROFILES[key] || PROFILES.generico;
  const generic = PROFILES.generico;

  // Merge con genérico como fallback para campos faltantes
  return {
    ...generic,
    ...base,
    behaviors:  { ...generic.behaviors,  ...(base.behaviors  || {}) },
    scoreBonus: { ...SCORE_BONUS_DEFAULTS, ...(base.scoreBonus || {}) },
  };
}

export function listProfiles() {
  return Object.entries(PROFILES).map(([key, p]) => ({
    key,
    name: p.name,
    icon: p.icon,
  }));
}

// Puntos de score por defecto (pueden sobreescribirse en cada perfil)
const SCORE_BONUS_DEFAULTS = {
  contacto:          15,
  objetoTomado:      40,
  arrebato:          55,
  traspaso:          50,
  bajoropa:          35,
  cadera:            35,
  manga:             35,
  agachado:          30,
  bagStuffing:       45,
  pantalla:          25,
  escaneo:           10,
  merodeo:           20,
  brazoscruzados:    15,
  distractor:        30,
  trayectoria:        8,
};

console.log('%c✅ store-profiles.js v5.0 cargado', 'color:#00e676;font-weight:bold');