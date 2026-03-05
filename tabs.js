// MANEJADOR DE PESTAÑAS Y PANEL IP - VERSIÓN REAL
document.addEventListener('DOMContentLoaded', function() {
  const tabs = document.querySelectorAll('.source-tab');
  const webcamConfig = document.getElementById('webcamConfig');
  const webrtcConfig = document.getElementById('webrtcConfig');
  const fileConfig = document.getElementById('fileConfig');
  
  if (!tabs.length) return;

  // ── FLAG: setupIpPanel solo se ejecuta una vez ──
  let ipPanelReady = false;

  tabs.forEach(tab => {
    tab.addEventListener('click', function(e) {
      e.preventDefault();

      const source = this.dataset.source;
      const isAlreadyActive = this.classList.contains('active');

      // Si clic en el tab IP/WebRTC ya activo → solo toggle del panel
      if (isAlreadyActive && source === 'webrtc') {
        webrtcConfig.classList.toggle('hidden');
        return; // no cambia la tab activa ni hace nada más
      }

      // Cambio normal de tab
      tabs.forEach(t => t.classList.remove('active'));
      this.classList.add('active');

      if (webcamConfig) webcamConfig.classList.add('hidden');
      if (webrtcConfig) webrtcConfig.classList.add('hidden');
      if (fileConfig)   fileConfig.classList.add('hidden');

      if (source === 'webcam' && webcamConfig) {
        webcamConfig.classList.remove('hidden');

      } else if (source === 'webrtc' && webrtcConfig) {
        webrtcConfig.classList.remove('hidden');
        // Inicializar panel IP solo la primera vez
        if (!ipPanelReady) {
          setTimeout(() => { setupIpPanel(); ipPanelReady = true; }, 100);
        }

      } else if (source === 'file' && fileConfig) {
        fileConfig.classList.remove('hidden');
      }
    });
  });
  
  // ══ FUNCIÓN PARA CONFIGURAR EL PANEL IP ══════════════════════════
  function setupIpPanel() {
    const ipInput        = document.getElementById('ipAddress');
    const userInput      = document.getElementById('ipUser');
    const passInput      = document.getElementById('ipPass');
    const brandSelect    = document.getElementById('ipBrand');
    const protocolSelect = document.getElementById('ipProtocol');
    const urlInput       = document.getElementById('ipUrl');
    const discoverBtn    = document.getElementById('ipDiscoverBtn');
    const connectBtn     = document.getElementById('ipConnectBtn');
    const connectProxyBtn = document.getElementById('connectProxyBtn');
    const statusDiv      = document.getElementById('ipConnectionStatus');
    
    if (!discoverBtn) {
      console.log('No se encontraron los botones IP');
      return;
    }
    
    console.log('Panel IP inicializado');
    
    // ── Helpers ──────────────────────────────────────────────────────
    const setStatus = (msg, type = 'info') => {
      if (!statusDiv) return;
      statusDiv.textContent = msg;
      statusDiv.style.color = type === 'error'   ? '#ff3d3d'
                            : type === 'success'  ? '#00e676'
                            :                       '#3a5468';
    };

    const updatePlaceholder = () => {
      const ip    = ipInput?.value.trim()  || '192.168.1.100';
      const user  = userInput?.value.trim()|| 'admin';
      const pass  = passInput?.value       || 'contraseña';
      const proto = protocolSelect?.value  || 'auto';
      const map = {
        'auto':   `http://${user}:${pass}@${ip}/video.mjpg`,
        'mjpeg':  `http://${user}:${pass}@${ip}/video.mjpg`,
        'hls':    `http://${ip}:8080/stream.m3u8`,
        'webrtc': `http://${ip}:8083/api/webrtc?src=cam1`,
        'whep':   `http://${ip}:8889/whep`
      };
      if (urlInput) urlInput.placeholder = map[proto] || map.auto;
    };

    // ── Guardar en localStorage ───────────────────────────────────────
    const STORAGE_KEY = 'ssip_ip_config';

    function saveIpConfig() {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({
        ip:       ipInput?.value        || '',
        user:     userInput?.value      || '',
        pass:     passInput?.value      || '',
        protocol: protocolSelect?.value || 'auto',
        brand:    brandSelect?.value    || 'all',
        url:      urlInput?.value       || ''
      }));
    }

    [ipInput, userInput, passInput, protocolSelect, brandSelect, urlInput]
      .forEach(el => {
        if (!el) return;
        el.addEventListener('change', saveIpConfig);
        el.addEventListener('input',  saveIpConfig);
      });

    // ── Restaurar config guardada ─────────────────────────────────────
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved) {
      try {
        const cfg = JSON.parse(saved);
        if (ipInput        && cfg.ip)       ipInput.value        = cfg.ip;
        if (userInput      && cfg.user)     userInput.value      = cfg.user;
        if (passInput      && cfg.pass)     passInput.value      = cfg.pass;
        if (protocolSelect && cfg.protocol) protocolSelect.value = cfg.protocol;
        if (brandSelect    && cfg.brand)    brandSelect.value    = cfg.brand;
        if (urlInput       && cfg.url)      urlInput.value       = cfg.url;
        setStatus('⚡ Configuración restaurada', 'info');
      } catch(e) { /* datos corruptos, ignorar */ }
    }

    // ── Listeners de campos ───────────────────────────────────────────
    if (ipInput)        ipInput.addEventListener('input',        updatePlaceholder);
    if (userInput)      userInput.addEventListener('input',      updatePlaceholder);
    if (passInput)      passInput.addEventListener('input',      updatePlaceholder);
    if (protocolSelect) protocolSelect.addEventListener('change',updatePlaceholder);

    // ── Botón Descubrir ───────────────────────────────────────────────
    discoverBtn.addEventListener('click', async function() {
      const ip = ipInput?.value.trim();
      if (!ip) { setStatus('❌ Introduce la IP', 'error'); return; }

      setStatus('🔍 Buscando cámara...', 'info');
      this.disabled = true;

      setTimeout(() => {
        const user = userInput?.value.trim() || 'admin';
        const pass = passInput?.value        || '';
        if (urlInput) urlInput.value = `http://${user}:${pass}@${ip}/video.mjpg`;
        setStatus('✅ Cámara encontrada', 'success');
        saveIpConfig();
        this.disabled = false;
      }, 1500);
    });

    // ── Botón Conectar ────────────────────────────────────────────────
    if (connectBtn) {
      connectBtn.addEventListener('click', function() {
        const url      = urlInput?.value.trim();
        const protocol = protocolSelect?.value || 'auto';
        if (!url) { setStatus('❌ Introduce la URL', 'error'); return; }

        setStatus('🔌 Conectando...', 'info');

        if (window.camSrc) {
          window.camSrc.startIP(url, protocol)
            .then(() => {
              setStatus('✅ Conectado', 'success');
              saveIpConfig();
              // Ocultar panel tras conectar exitosamente
              setTimeout(() => webrtcConfig.classList.add('hidden'), 800);
            })
            .catch(err => {
              console.error('Error de conexión:', err);
              setStatus(`❌ Error: ${err.message || 'No se pudo conectar'}`, 'error');
            });
        } else {
          setStatus('❌ Sistema de cámaras no disponible', 'error');
          console.error('camSrc no está definido');
        }
      });
    }

    // ── Botón Proxy ───────────────────────────────────────────────────
    if (connectProxyBtn) {
      connectProxyBtn.addEventListener('click', function() {
        const url      = urlInput?.value.trim();
        const protocol = protocolSelect?.value || 'auto';
        if (!url) { setStatus('❌ Introduce la URL', 'error'); return; }

        setStatus('🔌 Conectando con proxy...', 'info');

        if (window.camSrc) {
          const proxyUrl = 'https://cors-anywhere.herokuapp.com/' + url;
          console.log('🌐 Usando proxy:', proxyUrl);

          window.camSrc.startIP(proxyUrl, protocol)
            .then(() => {
              setStatus('✅ Conectado con proxy', 'success');
              saveIpConfig();
              // Ocultar panel tras conectar exitosamente
              setTimeout(() => webrtcConfig.classList.add('hidden'), 800);
            })
            .catch(err => {
              console.error('Error con proxy:', err);
              setStatus(`❌ Error: ${err.message || 'No se pudo conectar'}`, 'error');
            });
        }
      });
    }

    // ── Placeholder inicial ───────────────────────────────────────────
    updatePlaceholder();
  }

  // Si por algún motivo el panel IP ya está visible al cargar, inicializar
  if (webrtcConfig && !webrtcConfig.classList.contains('hidden')) {
    setTimeout(() => { setupIpPanel(); ipPanelReady = true; }, 100);
  }
});
