/* Visión Facial IA — Frontend
 * Captura la cámara, envía fotogramas JPEG por WebSocket al backend Python
 * y dibuja los resultados (cajas, puntos clave, malla) sobre el video.
 */

const $ = (id) => document.getElementById(id);
const servidorConfigurado = typeof window.VF_API_URL === "string" ? window.VF_API_URL.trim() : "";

// ------------------------------------------------------------------ estado
const estado = {
  servidor: localStorage.getItem("vf_servidor") ||
    servidorConfigurado || (location.protocol.startsWith("http") ? location.origin : "http://localhost:8000"),
  ws: null,
  stream: null,
  corriendo: false,
  esperando: false,
  tEnvio: 0,
  ultimoEnvio: 0,
  modo: "acceso",
  opciones: { umbral: 0.5, fpsMax: 15, ancho: 480, cajas: true, puntos: true, malla: false, espejo: true },
  ultimoResultado: null,
  marcasFps: [],
  reintentos: 0,
  despliegue: null,
};

const ESTADOS = {
  SIN_ROSTRO: { titulo: "Sin rostro", color: "#64748b" },
  OBSERVADO: { titulo: "Observado", color: "#ef4444" },
  VERIFICANDO: { titulo: "Verificando", color: "#f59e0b" },
  VALIDADO: { titulo: "Validado", color: "#22c55e" },
};
const COLOR_SECUNDARIO = "#22d3ee";

const video = $("video");
const overlay = $("overlay");
const ctx = overlay.getContext("2d");
const capCanvas = document.createElement("canvas");
const capCtx = capCanvas.getContext("2d");

// ------------------------------------------------------------------ utilidades UI
let avisoTimer;
function aviso(texto, tipo = "info") {
  const el = $("aviso");
  el.textContent = texto;
  el.className = `aviso visible ${tipo === "error" ? "error" : ""}`;
  clearTimeout(avisoTimer);
  avisoTimer = setTimeout(() => (el.className = "aviso"), 3500);
}

function chip(id, clase, texto) {
  const el = $(id);
  el.classList.remove("ok", "error", "alerta");
  if (clase) el.classList.add(clase);
  el.querySelector("b").textContent = texto;
}

const wsUrl = () => estado.servidor.replace(/^http/, "ws").replace(/\/$/, "") + "/ws/stream";
const apiUrl = (ruta) => estado.servidor.replace(/\/$/, "") + ruta;

// ------------------------------------------------------------------ WebSocket
function conectar() {
  if (estado.ws && estado.ws.readyState <= 1) estado.ws.close();
  const ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";
  estado.ws = ws;
  chip("chipServidor", "alerta", "conectando…");

  ws.onopen = () => {
    estado.reintentos = 0;
    chip("chipServidor", "ok", "conectado");
    enviarConfig();
  };
  ws.onclose = () => {
    chip("chipServidor", "error", "desconectado");
    estado.esperando = false;
    if (estado.ws !== ws) return; // se reemplazó por una conexión nueva
    const espera = Math.min(1000 * 2 ** estado.reintentos++, 10000);
    setTimeout(() => estado.ws === ws && conectar(), espera);
  };
  ws.onerror = () => chip("chipServidor", "error", "error");
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    switch (msg.tipo) {
      case "bienvenida":
        $("chipSesion").querySelector("b").textContent = msg.sesion;
        chip("chipMotor", msg.motor_listo ? "ok" : "error", msg.motor_listo ? "listo" : "no disponible");
        if (!msg.motor_listo) aviso(`Motor de IA no disponible: ${msg.error_motor || "revise el servidor"}`, "error");
        break;
      case "resultado":
        alRecibirResultado(msg);
        break;
      case "reiniciado":
        aviso("Verificación reiniciada.");
        break;
      case "error":
        estado.esperando = false;
        aviso(msg.mensaje, "error");
        break;
    }
  };
}

function enviarConfig() {
  if (estado.ws?.readyState !== WebSocket.OPEN) return;
  estado.ws.send(JSON.stringify({
    tipo: "config", modo: estado.modo, umbral: estado.opciones.umbral, malla: estado.opciones.malla,
  }));
}

// ------------------------------------------------------------------ cámara
async function listarCamaras() {
  try {
    const dispositivos = await navigator.mediaDevices.enumerateDevices();
    const sel = $("selCamara");
    const actual = sel.value;
    sel.innerHTML = '<option value="">Predeterminada</option>';
    dispositivos.filter((d) => d.kind === "videoinput").forEach((d, i) => {
      const o = document.createElement("option");
      o.value = d.deviceId;
      o.textContent = d.label || `Cámara ${i + 1}`;
      sel.appendChild(o);
    });
    sel.value = actual;
  } catch { /* sin permisos todavía */ }
}

async function iniciarCamara() {
  if (!navigator.mediaDevices?.getUserMedia) {
    aviso("Este navegador no permite acceder a la cámara (use https o localhost).", "error");
    return;
  }
  try {
    const deviceId = $("selCamara").value;
    estado.stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: deviceId ? { deviceId: { exact: deviceId } } : { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    });
  } catch (e) {
    aviso(e.name === "NotAllowedError" ? "Permiso de cámara denegado." : `No se pudo abrir la cámara: ${e.message}`, "error");
    return;
  }
  video.srcObject = estado.stream;
  await video.play();
  $("lienzo").style.aspectRatio = `${video.videoWidth} / ${video.videoHeight}`;
  $("lienzo").classList.add("activo");
  estado.corriendo = true;
  estado.esperando = false;
  actualizarBotonCamara();
  $("btnCaptura").disabled = false;
  listarCamaras();
  if (!estado.ws || estado.ws.readyState > 1) conectar();
  requestAnimationFrame(bucle);
}

function detenerCamara() {
  estado.corriendo = false;
  estado.stream?.getTracks().forEach((t) => t.stop());
  estado.stream = null;
  video.srcObject = null;
  $("lienzo").classList.remove("activo");
  $("btnCaptura").disabled = true;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  pintarBanda({ estado: "SIN_ROSTRO", motivos: ["Cámara detenida."] });
  actualizarBotonCamara();
}

function actualizarBotonCamara() {
  const b = $("btnCamara");
  b.textContent = estado.corriendo ? "Detener cámara" : "Iniciar cámara";
  b.classList.toggle("detener", estado.corriendo);
}

// ------------------------------------------------------------------ bucle de envío
function bucle() {
  if (!estado.corriendo) return;
  const ahora = performance.now();
  // Vigilante: si el servidor no responde en 3 s, se libera el turno
  if (estado.esperando && ahora - estado.tEnvio > 3000) estado.esperando = false;

  const intervalo = 1000 / estado.opciones.fpsMax;
  if (!estado.esperando && estado.ws?.readyState === WebSocket.OPEN && ahora - estado.ultimoEnvio >= intervalo
      && video.readyState >= 2) {
    enviarFotograma();
  }
  requestAnimationFrame(bucle);
}

function enviarFotograma() {
  const w = estado.opciones.ancho;
  const h = Math.round((video.videoHeight / video.videoWidth) * w);
  capCanvas.width = w;
  capCanvas.height = h;
  capCtx.drawImage(video, 0, 0, w, h);
  estado.esperando = true;
  estado.ultimoEnvio = estado.tEnvio = performance.now();
  capCanvas.toBlob((blob) => {
    if (blob && estado.ws?.readyState === WebSocket.OPEN) estado.ws.send(blob);
    else estado.esperando = false;
  }, "image/jpeg", 0.75);
}

// ------------------------------------------------------------------ resultados
function alRecibirResultado(res) {
  const ahora = performance.now();
  estado.esperando = false;
  estado.ultimoResultado = res;
  const latencia = Math.round(ahora - estado.tEnvio);

  estado.marcasFps.push(ahora);
  while (estado.marcasFps.length && ahora - estado.marcasFps[0] > 1000) estado.marcasFps.shift();

  $("hudFps").textContent = estado.marcasFps.length;
  $("hudLat").textContent = latencia;
  $("mRostros").textContent = res.rostros.length;
  $("mConf").textContent = res.rostros[0] ? `${Math.round(res.rostros[0].confianza * 100)}%` : "—";
  $("mInf").textContent = `${res.tiempo_inferencia_ms} ms`;
  $("mParp").textContent = res.validacion.parpadeos;

  if (estado.corriendo) dibujar(res);
  pintarBanda(res.validacion);
  pintarChecklist(res.validacion.verificaciones);
  pintarBarras(res.rostros[0]);
  if (res.evento) cargarEventos();
}

// ------------------------------------------------------------------ dibujo
function ajustarOverlay() {
  const dpr = window.devicePixelRatio || 1;
  const { clientWidth: w, clientHeight: h } = overlay;
  if (overlay.width !== Math.round(w * dpr) || overlay.height !== Math.round(h * dpr)) {
    overlay.width = Math.round(w * dpr);
    overlay.height = Math.round(h * dpr);
  }
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { w, h };
}

function dibujar(res, destino = ctx, dims = null) {
  const { w, h } = dims || ajustarOverlay();
  const espejo = estado.opciones.espejo && !dims;
  // Área real donde se muestra el fotograma (object-fit: contain deja bandas a los lados o arriba)
  const vw = video.videoWidth || w, vh = video.videoHeight || h;
  const escala = Math.min(w / vw, h / vh);
  const dw = vw * escala, dh = vh * escala;
  const ox = (w - dw) / 2, oy = (h - dh) / 2;
  const X = (x) => ox + (espejo ? 1 - x : x) * dw;
  const Y = (y) => oy + y * dh;
  destino.clearRect(0, 0, w, h);
  const colorPrincipal = ESTADOS[res.validacion.estado]?.color || COLOR_SECUNDARIO;

  res.rostros.forEach((r, i) => {
    const color = i === 0 ? colorPrincipal : COLOR_SECUNDARIO;
    const c = r.caja;
    const x0 = espejo ? X(c.x + c.ancho) : X(c.x);
    const y0 = Y(c.y), bw = c.ancho * dw, bh = c.alto * dh;

    if (r.malla && estado.opciones.malla) {
      destino.fillStyle = "rgba(34, 211, 238, 0.55)";
      for (const [px, py] of r.malla) destino.fillRect(X(px) - 0.8, Y(py) - 0.8, 1.6, 1.6);
    }

    if (estado.opciones.cajas) {
      esquinas(destino, x0, y0, bw, bh, color);
      const etiqueta = `Rostro ${i + 1} · ${Math.round(r.confianza * 100)}%`;
      destino.font = "600 12px system-ui, sans-serif";
      const tw = destino.measureText(etiqueta).width + 12;
      destino.fillStyle = color;
      destino.fillRect(x0, y0 - 22, tw, 20);
      destino.fillStyle = "#04121a";
      destino.fillText(etiqueta, x0 + 6, y0 - 8);
    }

    if (estado.opciones.puntos) {
      for (const p of r.puntos_clave) {
        destino.beginPath();
        destino.arc(X(p.x), Y(p.y), 3.5, 0, Math.PI * 2);
        destino.fillStyle = color;
        destino.fill();
        destino.lineWidth = 1.5;
        destino.strokeStyle = "#04121a";
        destino.stroke();
      }
    }
  });
}

function esquinas(c, x, y, w, h, color) {
  const l = Math.min(w, h) * 0.22;
  c.strokeStyle = color;
  c.lineWidth = 3;
  c.lineCap = "round";
  c.beginPath();
  c.moveTo(x, y + l); c.lineTo(x, y); c.lineTo(x + l, y);
  c.moveTo(x + w - l, y); c.lineTo(x + w, y); c.lineTo(x + w, y + l);
  c.moveTo(x + w, y + h - l); c.lineTo(x + w, y + h); c.lineTo(x + w - l, y + h);
  c.moveTo(x + l, y + h); c.lineTo(x, y + h); c.lineTo(x, y + h - l);
  c.stroke();
  c.lineWidth = 1;
  c.globalAlpha = 0.35;
  c.strokeRect(x, y, w, h);
  c.globalAlpha = 1;
}

// ------------------------------------------------------------------ paneles
function pintarBanda(val) {
  const info = ESTADOS[val.estado] || ESTADOS.SIN_ROSTRO;
  $("banda").dataset.estado = val.estado;
  $("bandaTitulo").textContent = info.titulo;
  let msg = val.motivos?.[0];
  if (!msg) msg = estado.modo === "acceso" ? "Persona real verificada: rostro frontal y nítido." : "Rostro detectado correctamente.";
  $("bandaMensaje").textContent = msg;
}

function pintarChecklist(lista) {
  const ul = $("checklist");
  if (!lista?.length) {
    ul.innerHTML = '<li class="vacio">Sin rostro en escena.</li>';
    return;
  }
  ul.innerHTML = lista.map((v) => `<li class="${v.ok ? "ok" : ""}">${v.etiqueta}</li>`).join("");
}

function barra(clave, valor, texto, { centro = false, max = 1 } = {}) {
  const el = document.querySelector(`.barra[data-k="${clave}"]`);
  const i = el.querySelector("i");
  el.classList.toggle("centro", centro);
  if (valor == null) {
    i.style.width = "0"; i.style.left = "0";
    el.querySelector("b").textContent = "—";
    return;
  }
  if (centro) { // barra simétrica para ángulos (-max..max)
    const v = Math.max(-max, Math.min(max, valor)) / max;
    i.style.left = `${50 + Math.min(0, v) * 50}%`;
    i.style.width = `${Math.abs(v) * 50}%`;
  } else {
    i.style.left = "0";
    i.style.width = `${Math.max(0, Math.min(1, valor / max)) * 100}%`;
  }
  el.querySelector("b").textContent = texto;
}

function pintarBarras(r) {
  const p = r?.pose, q = r?.calidad, g = r?.gestos || {};
  const grados = (v) => `${Number(v).toFixed(1)}°`;
  barra("yaw", p?.yaw, p ? grados(p.yaw) : "", { centro: true, max: 45 });
  barra("pitch", p?.pitch, p ? grados(p.pitch) : "", { centro: true, max: 45 });
  barra("roll", p?.roll, p ? grados(p.roll) : "", { centro: true, max: 45 });
  barra("brillo", q?.brillo, q ? `${Math.round(q.brillo)}` : "", { max: 255 });
  barra("nitidez", q?.nitidez, q ? `${Math.round(q.nitidez)}` : "", { max: 300 });
  const sonrisa = r ? ((g.mouthSmileLeft || 0) + (g.mouthSmileRight || 0)) / 2 : null;
  barra("sonrisa", sonrisa, sonrisa != null ? `${Math.round(sonrisa * 100)}%` : "");
}

// ------------------------------------------------------------------ bitácora (API REST)
async function cargarEventos() {
  try {
    const [evs, est] = await Promise.all([
      fetch(apiUrl("/api/eventos?limite=50")).then((r) => r.json()),
      fetch(apiUrl("/api/estadisticas")).then((r) => r.json()),
    ]);
    const tb = $("tablaEventos");
    tb.innerHTML = evs.length
      ? evs.map((e) => `<tr>
          <td>${e.id}</td><td>${e.fecha.replace("T", " ")}</td><td><code>${e.sesion}</code></td>
          <td>${e.modo === "acceso" ? "Control de acceso" : "Detección"}</td>
          <td><span class="etiqueta ${e.estado}">${ESTADOS[e.estado]?.titulo || e.estado}</span></td>
          <td>${e.num_rostros}</td><td>${e.confianza != null ? Math.round(e.confianza * 100) + "%" : "—"}</td>
          <td>${e.motivos.join(" · ") || "—"}</td></tr>`).join("")
      : '<tr><td colspan="8" class="vacio">Sin eventos registrados.</td></tr>';
    $("resumen").innerHTML = `
      <span>Total: <b>${est.total_eventos}</b></span>
      <span>Validados hoy: <b>${est.validados_hoy}</b></span>
      <span>Observados: <b>${est.por_estado.OBSERVADO || 0}</b></span>`;
  } catch {
    $("resumen").innerHTML = "<span>No se pudo leer la bitácora.</span>";
  }
}

// ------------------------------------------------------------------ aviso de actualización
function mostrarBanner(texto, pendiente) {
  $("bannerTexto").textContent = texto;
  $("bannerActualizacion").classList.toggle("pendiente", pendiente);
  $("btnRecargar").hidden = !pendiente;
  $("bannerActualizacion").hidden = false;
}

function revisarDespliegue(s) {
  if (!s.despliegue) return;
  if (!estado.despliegue) {
    // Primera consulta: si la última versión vista era otra, el sistema se actualizó
    estado.despliegue = s.despliegue;
    let anterior = null;
    try { anterior = localStorage.getItem("vf_despliegue"); localStorage.setItem("vf_despliegue", s.despliegue); } catch {}
    if (anterior !== s.despliegue) {
      mostrarBanner(`El sistema se actualizó a la versión ${s.version}.`, false);
    }
  } else if (s.despliegue !== estado.despliegue) {
    // La página quedó abierta mientras se publicaba una versión nueva
    mostrarBanner("Hay una actualización del sistema disponible. Recargue para usar la versión nueva.", true);
  }
}

async function verificarSalud() {
  try {
    const s = await fetch(apiUrl("/api/salud")).then((r) => r.json());
    chip("chipMotor", s.motor_listo ? "ok" : "error", s.motor_listo ? "listo" : "no disponible");
    revisarDespliegue(s);
  } catch {
    chip("chipMotor", "error", "sin respuesta");
  }
}

// ------------------------------------------------------------------ captura
function capturar() {
  if (!estado.ultimoResultado) return;
  const c = document.createElement("canvas");
  c.width = video.videoWidth;
  c.height = video.videoHeight;
  const g = c.getContext("2d");
  g.drawImage(video, 0, 0);
  dibujar(estado.ultimoResultado, g, { w: c.width, h: c.height });
  g.font = "600 16px system-ui"; g.fillStyle = "#fff";
  g.fillText(`${new Date().toLocaleString("es-PE")} · ${estado.ultimoResultado.validacion.estado}`, 12, c.height - 14);
  const a = document.createElement("a");
  a.download = `captura_${Date.now()}.png`;
  a.href = c.toDataURL("image/png");
  a.click();
}

// ------------------------------------------------------------------ eventos de la interfaz
function enlazar() {
  const alternarCamara = () => (estado.corriendo ? detenerCamara() : iniciarCamara());
  $("btnCamara").onclick = alternarCamara;
  $("btnIniciarReposo").onclick = iniciarCamara;
  $("btnCaptura").onclick = capturar;
  $("btnReiniciar").onclick = () => estado.ws?.readyState === 1 && estado.ws.send(JSON.stringify({ tipo: "reiniciar" }));
  $("btnActualizar").onclick = cargarEventos;
  $("btnLimpiar").onclick = async () => {
    if (!confirm("¿Eliminar todos los eventos de la bitácora?")) return;
    await fetch(apiUrl("/api/eventos"), { method: "DELETE" });
    cargarEventos();
  };
  $("selCamara").onchange = () => estado.corriendo && (detenerCamara(), iniciarCamara());

  document.querySelectorAll(".segmentado button").forEach((b) => {
    b.onclick = () => {
      document.querySelectorAll(".segmentado button").forEach((x) => x.classList.toggle("activo", x === b));
      estado.modo = b.dataset.modo;
      enviarConfig();
    };
  });

  $("rngUmbral").oninput = (e) => {
    estado.opciones.umbral = +e.target.value;
    $("outUmbral").textContent = estado.opciones.umbral.toFixed(2);
  };
  $("rngUmbral").onchange = enviarConfig;
  $("rngFps").oninput = (e) => {
    estado.opciones.fpsMax = +e.target.value;
    $("outFps").textContent = e.target.value;
  };
  $("selResolucion").onchange = (e) => (estado.opciones.ancho = +e.target.value);
  $("chkCajas").onchange = (e) => (estado.opciones.cajas = e.target.checked);
  $("chkPuntos").onchange = (e) => (estado.opciones.puntos = e.target.checked);
  $("chkMalla").onchange = (e) => { estado.opciones.malla = e.target.checked; enviarConfig(); };
  $("chkEspejo").onchange = (e) => {
    estado.opciones.espejo = e.target.checked;
    video.classList.toggle("espejo", e.target.checked);
  };
  video.classList.toggle("espejo", estado.opciones.espejo);

  $("txtServidor").value = estado.servidor;
  $("btnServidor").onclick = () => {
    estado.servidor = $("txtServidor").value.trim() || estado.servidor;
    localStorage.setItem("vf_servidor", estado.servidor);
    conectar();
    verificarSalud();
    cargarEventos();
  };

  window.addEventListener("resize", () => estado.ultimoResultado && estado.corriendo && dibujar(estado.ultimoResultado));
  setInterval(() => ($("hudReloj").textContent = new Date().toLocaleTimeString("es-PE")), 1000);
  setInterval(cargarEventos, 10000);
  setInterval(verificarSalud, 60000);
  $("btnRecargar").onclick = () => location.reload();
  $("btnCerrarBanner").onclick = () => ($("bannerActualizacion").hidden = true);
}

enlazar();
conectar();
verificarSalud();
cargarEventos();
