"""Reglas de validación para control de acceso y prueba de vivacidad (parpadeo).

Importante: este módulo valida que haya UNA persona real, frontal y bien
capturada frente a la cámara. NO identifica quién es (eso requiere un modelo
de reconocimiento / embeddings y un padrón con consentimiento).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from ..config import CONFIG, Config

# Estados posibles
SIN_ROSTRO = "SIN_ROSTRO"
OBSERVADO = "OBSERVADO"      # hay rostro pero falla alguna condición
VERIFICANDO = "VERIFICANDO"  # todo bien, falta vivacidad o estabilidad
VALIDADO = "VALIDADO"        # rostro vivo, frontal y de buena calidad


@dataclass
class EstadoSesion:
    """Estado por cliente (cada conexión WebSocket tiene el suyo)."""
    ojo_cerrado: bool = False
    parpadeos: int = 0
    ultimo_parpadeo: float = 0.0
    frames_validos: int = 0
    ultimo_estado: str = SIN_ROSTRO
    creado: float = field(default_factory=time.time)

    def reiniciar(self) -> None:
        self.ojo_cerrado = False
        self.parpadeos = 0
        self.ultimo_parpadeo = 0.0
        self.frames_validos = 0
        self.ultimo_estado = SIN_ROSTRO


def actualizar_parpadeo(sesion: EstadoSesion, gestos: dict, cfg: Config = CONFIG, ahora: float | None = None) -> None:
    """Cuenta un parpadeo en la transición ojos cerrados -> abiertos."""
    ahora = time.time() if ahora is None else ahora
    if not gestos:
        return
    cierre = (gestos.get("eyeBlinkLeft", 0.0) + gestos.get("eyeBlinkRight", 0.0)) / 2
    if cierre >= cfg.umbral_ojo_cerrado:
        sesion.ojo_cerrado = True
    elif sesion.ojo_cerrado and cierre < cfg.umbral_ojo_cerrado * 0.6:  # histéresis
        sesion.ojo_cerrado = False
        sesion.parpadeos += 1
        sesion.ultimo_parpadeo = ahora


def evaluar(rostros: list[dict], sesion: EstadoSesion | None = None, modo: str = "acceso",
            cfg: Config = CONFIG, ahora: float | None = None) -> dict:
    """Aplica las reglas y devuelve estado, lista de verificaciones y motivos."""
    ahora = time.time() if ahora is None else ahora
    sesion = sesion or EstadoSesion()

    if not rostros:
        sesion.frames_validos = 0
        return {"estado": SIN_ROSTRO, "verificaciones": [], "motivos": ["No se detecta ningún rostro."],
                "parpadeos": sesion.parpadeos}

    principal = rostros[0]
    actualizar_parpadeo(sesion, principal.get("gestos") or {}, cfg, ahora)
    cal = principal.get("calidad") or {}
    pose = principal.get("pose")

    checks: list[tuple[str, str, bool, str]] = [
        ("unico", "Un solo rostro", len(rostros) == 1,
         f"Se detectan {len(rostros)} rostros; debe haber solo una persona."),
        ("confianza", f"Confianza ≥ {cfg.umbral_acceso:.0%}", principal["confianza"] >= cfg.umbral_acceso,
         f"Confianza baja ({principal['confianza']:.0%})."),
        ("distancia", "Distancia adecuada", cal.get("tamano", 0) >= cfg.tamano_min_rostro,
         "Acérquese a la cámara."),
        ("iluminacion", "Iluminación correcta", cfg.brillo_min <= cal.get("brillo", 0) <= cfg.brillo_max,
         "Iluminación insuficiente." if cal.get("brillo", 0) < cfg.brillo_min else "Exceso de luz / contraluz."),
        ("nitidez", "Imagen nítida", cal.get("nitidez", 0) >= cfg.nitidez_min,
         "Imagen borrosa: quédese quieto."),
        ("frontal", "Mirada frontal",
         pose is not None and abs(pose["yaw"]) <= cfg.yaw_max and abs(pose["pitch"]) <= cfg.pitch_max,
         "Mire de frente a la cámara."),
    ]
    vivo = sesion.ultimo_parpadeo > 0 and (ahora - sesion.ultimo_parpadeo) <= cfg.ventana_vivacidad_s
    checks.append(("vivacidad", "Prueba de vida (parpadeo)", vivo, "Parpadee para confirmar que es una persona real."))

    if modo != "acceso":  # en modo detección solo se informan, no se exige vivacidad
        checks = checks[:-1]

    verificaciones = [{"clave": c, "etiqueta": e, "ok": ok} for c, e, ok, _ in checks]
    fallas_basicas = [m for c, _, ok, m in checks if not ok and c != "vivacidad"]
    motivos = [m for _, _, ok, m in checks if not ok]

    if fallas_basicas:
        sesion.frames_validos = 0
        estado = OBSERVADO
    else:
        sesion.frames_validos += 1
        if modo == "acceso" and not vivo:
            estado = VERIFICANDO
        elif sesion.frames_validos < cfg.frames_estables:
            estado = VERIFICANDO
            motivos = ["Mantenga la posición unos instantes…"]
        else:
            estado = VALIDADO
            motivos = []

    return {"estado": estado, "verificaciones": verificaciones, "motivos": motivos,
            "parpadeos": sesion.parpadeos}
