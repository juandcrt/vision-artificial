"""Configuración central del sistema (se puede sobrescribir con variables de entorno)."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

RAIZ_BACKEND = Path(__file__).resolve().parent.parent
RAIZ_PROYECTO = RAIZ_BACKEND.parent

# En Vercel el código es de solo lectura: modelos y bitácora van a /tmp (efímero)
EN_VERCEL = bool(os.getenv("VERCEL"))
RAIZ_DATOS = Path("/tmp/vision-facial-ia") if EN_VERCEL else RAIZ_BACKEND


def _env_float(nombre: str, defecto: float) -> float:
    try:
        return float(os.getenv(nombre, defecto))
    except ValueError:
        return defecto


def _env_int(nombre: str, defecto: int) -> int:
    try:
        return int(os.getenv(nombre, defecto))
    except ValueError:
        return defecto


@dataclass(frozen=True)
class Config:
    # --- Rutas ---
    dir_modelos: Path = Path(os.getenv("VF_DIR_MODELOS", RAIZ_DATOS / "modelos"))
    dir_frontend: Path = RAIZ_PROYECTO / "frontend"
    ruta_bd: Path = RAIZ_DATOS / "eventos.db"

    # --- Modelos MediaPipe (API Tasks) ---
    modelo_detector: str = "blaze_face_short_range.tflite"
    url_detector: str = (
        "https://storage.googleapis.com/mediapipe-models/face_detector/"
        "blaze_face_short_range/float16/latest/blaze_face_short_range.tflite"
    )
    modelo_landmarker: str = "face_landmarker.task"
    url_landmarker: str = (
        "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
        "face_landmarker/float16/latest/face_landmarker.task"
    )

    # --- Detección ---
    max_rostros: int = field(default_factory=lambda: _env_int("VF_MAX_ROSTROS", 5))
    umbral_deteccion: float = field(default_factory=lambda: _env_float("VF_UMBRAL", 0.5))
    ancho_max_entrada: int = 960  # se redimensiona si llega una imagen más grande
    # Si la malla facial confirma el rostro, se acepta con este margen por debajo del umbral
    # (con mascarilla o lentes la confianza del detector baja aunque la cara sea clara)
    margen_con_malla: float = 0.2

    # --- Validación (modo control de acceso) ---
    umbral_acceso: float = field(default_factory=lambda: _env_float("VF_UMBRAL_ACCESO", 0.75))
    tamano_min_rostro: float = 0.15   # ancho del rostro / ancho de la imagen
    brillo_min: float = 60.0           # media de gris del rostro (0-255)
    brillo_max: float = 210.0
    nitidez_min: float = 40.0          # varianza del Laplaciano (rostro normalizado a 128 px)
    yaw_max: float = 20.0              # grados (giro izquierda/derecha)
    pitch_max: float = 20.0            # grados (arriba/abajo)
    roll_max: float = 20.0             # grados (cabeza inclinada hacia un hombro)
    umbral_ojo_cerrado: float = 0.45   # blendshape eyeBlink
    ventana_vivacidad_s: float = 8.0   # el parpadeo debe ocurrir en esta ventana
    frames_estables: int = 5           # frames válidos consecutivos para aprobar

    # --- Mascarilla (solo informativa: no cambia la validación) ---
    mascarilla_fraccion_rojez: float = 0.5     # un píxel es piel si su Cr-Cb >= esta fracción del entrecejo...
    mascarilla_rojez_min: float = 8.0          # ...pero nunca menos que esto (mascarillas: Cr-Cb <= 0)
    mascarilla_rojez_max: float = 25.0         # ...ni más que esto (labios y sombras siguen contando)
    mascarilla_piel_min: float = 0.35          # con menos piel que esto en boca/barbilla -> mascarilla

    # --- Servidor ---
    zona_horaria: str = field(default_factory=lambda: os.getenv("VF_ZONA_HORARIA", "America/Lima"))
    origenes_cors: tuple[str, ...] = ("*",)


CONFIG = Config()
