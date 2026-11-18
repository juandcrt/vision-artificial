"""API del sistema de detección facial en tiempo real.

Endpoints:
  GET    /api/salud          Estado del servidor y del motor de visión
  GET    /api/config         Parámetros de validación vigentes
  POST   /api/detectar       Procesa una imagen (multipart) -> rostros + validación
  GET    /api/eventos        Bitácora de eventos (más recientes primero)
  GET    /api/estadisticas   Resumen de eventos
  DELETE /api/eventos        Limpia la bitácora
  WS     /ws/stream          Flujo en tiempo real (frames JPEG binarios -> resultados JSON)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from dataclasses import asdict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Query, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from .config import CONFIG, Config
from .eventos import RegistroEventos
from .vision.validacion import OBSERVADO, VALIDADO, EstadoSesion, evaluar

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [%(name)s] %(message)s")
log = logging.getLogger("api")

VERSION = "1.3.0"
MODOS = ("deteccion", "acceso")
ESTADOS_A_REGISTRAR = (VALIDADO, OBSERVADO)
SEGUNDOS_ENTRE_EVENTOS = 5.0


def _huella_codigo(cfg: Config) -> str:
    """Identificador del despliegue: cambia cuando cambia el código del backend o del frontend.
    El navegador lo compara para avisar que el sistema se actualizó."""
    h = hashlib.sha256()
    for raiz in (Path(__file__).resolve().parent, cfg.dir_frontend):
        if not raiz.exists():
            continue
        for ruta in sorted(raiz.rglob("*")):
            if ruta.is_file() and "__pycache__" not in ruta.parts:
                h.update(str(ruta.relative_to(raiz)).encode())
                h.update(ruta.read_bytes())
    return h.hexdigest()[:12]


def _cargar_motor(cfg: Config):
    from .vision.modelos import asegurar_modelos
    from .vision.motor import MotorVision

    det, lmk = asegurar_modelos(cfg)
    return MotorVision(det, lmk, cfg)


def crear_app(motor=None, registro: RegistroEventos | None = None, cfg: Config = CONFIG,
              servir_frontend: bool = True, cargar_motor: bool = True) -> FastAPI:
    """Fábrica de la aplicación (permite inyectar un motor simulado en las pruebas)."""
    candado_motor = asyncio.Lock()

    async def obtener_motor():
        """Carga el motor una sola vez. También se llama bajo demanda porque algunos
        entornos serverless (Vercel) pueden no ejecutar el evento de arranque."""
        if app.state.motor is None and cargar_motor:
            async with candado_motor:
                if app.state.motor is None:
                    try:
                        app.state.motor = await run_in_threadpool(_cargar_motor, cfg)
                        app.state.error_motor = None
                    except Exception as exc:  # el servidor sigue funcionando e informa el error
                        log.exception("No se pudo iniciar el motor de visión")
                        app.state.error_motor = str(exc)
        return app.state.motor

    @asynccontextmanager
    async def ciclo_vida(app: FastAPI):
        await obtener_motor()
        yield
        if app.state.motor is not None and hasattr(app.state.motor, "cerrar"):
            app.state.motor.cerrar()

    app = FastAPI(title="Visión Facial IA", version=VERSION, lifespan=ciclo_vida,
                  description="Detección facial en tiempo real con MediaPipe y OpenCV")
    app.state.motor = motor
    app.state.error_motor = None
    app.state.registro = registro or RegistroEventos(cfg.ruta_bd, cfg.zona_horaria)
    app.state.inicio = time.time()
    app.state.despliegue = _huella_codigo(cfg)
    app.add_middleware(CORSMiddleware, allow_origins=list(cfg.origenes_cors),
                       allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------------ REST
    @app.get("/api/salud")
    def salud():
        return {
            "estado": "ok" if app.state.motor is not None else "degradado",
            "motor_listo": app.state.motor is not None,
            "error_motor": app.state.error_motor,
            "version": VERSION,
            "despliegue": app.state.despliegue,
            "activo_desde_s": round(time.time() - app.state.inicio, 1),
        }

    @app.get("/api/config")
    def configuracion():
        d = asdict(cfg)
        for k in ("dir_modelos", "dir_frontend", "ruta_bd", "url_detector", "url_landmarker", "origenes_cors"):
            d.pop(k, None)
        return d

    @app.post("/api/detectar")
    async def detectar(
        imagen: UploadFile = File(..., description="Imagen JPEG o PNG"),
        umbral: float = Query(cfg.umbral_deteccion, ge=0.0, le=1.0),
        malla: bool = Query(False, description="Incluir los 478 puntos faciales"),
        modo: str = Query("deteccion", pattern="^(deteccion|acceso)$"),
    ):
        motor = await obtener_motor()
        if motor is None:
            raise HTTPException(503, f"Motor de visión no disponible: {app.state.error_motor}")
        datos = await imagen.read()
        if not datos:
            raise HTTPException(400, "Imagen vacía.")
        try:
            img = motor.decodificar(datos)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        res = await run_in_threadpool(motor.procesar, img, umbral, malla)
        # Petición aislada: no hay historial, así que no se exige estabilidad entre frames
        sesion = EstadoSesion(frames_validos=cfg.frames_estables)
        res["validacion"] = evaluar(res["rostros"], sesion, modo, cfg)
        return res

    @app.get("/api/eventos")
    def eventos(limite: int = Query(50, ge=1, le=500)):
        return app.state.registro.listar(limite)

    @app.get("/api/estadisticas")
    def estadisticas():
        return app.state.registro.estadisticas()

    @app.delete("/api/eventos")
    def limpiar_eventos():
        return {"eliminados": app.state.registro.limpiar()}

    # ------------------------------------------------------------------ WebSocket
    @app.websocket("/ws/stream")
    async def flujo(ws: WebSocket):
        await ws.accept()
        sesion_id = uuid.uuid4().hex[:8]
        estado = EstadoSesion()
        opciones = {"umbral": cfg.umbral_deteccion, "malla": False, "modo": "acceso"}
        ultimo_evento: dict[str, float] = {}
        motor_listo = await obtener_motor() is not None
        await ws.send_json({"tipo": "bienvenida", "sesion": sesion_id,
                            "motor_listo": motor_listo,
                            "error_motor": app.state.error_motor})
        log.info("Cliente conectado (sesión %s)", sesion_id)
        try:
            while True:
                msg = await ws.receive()
                if msg["type"] == "websocket.disconnect":
                    break

                # --- Mensajes de control (JSON) ---
                if msg.get("text") is not None:
                    try:
                        data = json.loads(msg["text"])
                    except json.JSONDecodeError:
                        await ws.send_json({"tipo": "error", "mensaje": "JSON inválido."})
                        continue
                    tipo = data.get("tipo")
                    if tipo == "config":
                        if "umbral" in data:
                            opciones["umbral"] = min(max(float(data["umbral"]), 0.0), 1.0)
                        if "malla" in data:
                            opciones["malla"] = bool(data["malla"])
                        if data.get("modo") in MODOS:
                            opciones["modo"] = data["modo"]
                        await ws.send_json({"tipo": "config_ok", "opciones": opciones})
                    elif tipo == "reiniciar":
                        estado.reiniciar()
                        await ws.send_json({"tipo": "reiniciado"})
                    elif tipo == "ping":
                        await ws.send_json({"tipo": "pong", "t": data.get("t")})
                    continue

                # --- Fotogramas (binario JPEG) ---
                datos = msg.get("bytes")
                if not datos:
                    continue
                motor = app.state.motor
                if motor is None:
                    await ws.send_json({"tipo": "error", "mensaje": f"Motor no disponible: {app.state.error_motor}"})
                    continue
                t0 = time.perf_counter()
                try:
                    img = motor.decodificar(datos)
                    res = await run_in_threadpool(motor.procesar, img, opciones["umbral"], opciones["malla"])
                except Exception as exc:
                    await ws.send_json({"tipo": "error", "mensaje": str(exc)})
                    continue

                val = evaluar(res["rostros"], estado, opciones["modo"], cfg)
                evento = None
                ahora = time.time()
                if (val["estado"] != estado.ultimo_estado and val["estado"] in ESTADOS_A_REGISTRAR
                        and ahora - ultimo_evento.get(val["estado"], 0) >= SEGUNDOS_ENTRE_EVENTOS):
                    ultimo_evento[val["estado"]] = ahora
                    conf = res["rostros"][0]["confianza"] if res["rostros"] else None
                    evento = await run_in_threadpool(
                        app.state.registro.registrar, sesion_id, opciones["modo"], val["estado"],
                        len(res["rostros"]), conf, val["motivos"])
                estado.ultimo_estado = val["estado"]

                res.update({
                    "tipo": "resultado",
                    "validacion": val,
                    "evento": evento,
                    "tiempo_servidor_ms": round((time.perf_counter() - t0) * 1000, 1),
                })
                await ws.send_json(res)
        except WebSocketDisconnect:
            pass
        finally:
            log.info("Cliente desconectado (sesión %s)", sesion_id)

    # ------------------------------------------------------------------ Frontend
    if servir_frontend and cfg.dir_frontend.exists():
        app.mount("/", StaticFiles(directory=str(cfg.dir_frontend), html=True), name="frontend")

    return app


app = crear_app()
