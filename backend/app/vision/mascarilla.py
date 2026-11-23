"""Detección de mascarilla con OpenCV a partir de los puntos clave de BlazeFace.

Idea: la piel es rojiza (Cr > Cb en YCrCb) aun con poca luz o cámaras de colores apagados,
mientras que las mascarillas celestes, blancas o negras no lo son. Si la zona de boca y
barbilla casi no tiene píxeles rojizos, está cubierta. El entrecejo, que la mascarilla no
suele tapar, sirve para ajustar el umbral al tono de cada persona. Solo se usa el color,
no el brillo, para tolerar sombras.
No necesita un modelo adicional y tarda menos de 1 ms por rostro.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..config import CONFIG, Config


def _parche(img: np.ndarray, cx: float, cy: float, ancho: float, alto: float) -> np.ndarray | None:
    h, w = img.shape[:2]
    x0, x1 = int(max(cx - ancho / 2, 0)), int(min(cx + ancho / 2, w))
    y0, y1 = int(max(cy - alto / 2, 0)), int(min(cy + alto / 2, h))
    if x1 - x0 < 3 or y1 - y0 < 3:
        return None
    return img[y0:y1, x0:x1]


def detectar_mascarilla(img_bgr: np.ndarray, puntos_clave: list[dict], cfg: Config = CONFIG) -> dict | None:
    """Devuelve {"detectada": bool, "piel_inferior": fracción 0-1} o None si faltan puntos."""
    p = {k["nombre"]: k for k in puntos_clave}
    if not all(n in p for n in ("ojo_derecho", "ojo_izquierdo", "nariz", "boca")):
        return None
    h, w = img_bgr.shape[:2]
    ojo_d = np.array([p["ojo_derecho"]["x"] * w, p["ojo_derecho"]["y"] * h])
    ojo_i = np.array([p["ojo_izquierdo"]["x"] * w, p["ojo_izquierdo"]["y"] * h])
    nariz = np.array([p["nariz"]["x"] * w, p["nariz"]["y"] * h])
    boca = np.array([p["boca"]["x"] * w, p["boca"]["y"] * h])
    d = float(np.linalg.norm(ojo_d - ojo_i))  # distancia entre ojos: escala del rostro
    if d < 12:
        return None

    centro_ojos = (ojo_d + ojo_i) / 2
    referencia = _parche(img_bgr, centro_ojos[0], centro_ojos[1] - 0.05 * d, 0.30 * d, 0.25 * d)
    alto_boca = max(boca[1] - nariz[1], 0.2 * d)
    inferior = _parche(img_bgr, boca[0], boca[1] + 0.15 * alto_boca, 0.90 * d, 1.30 * alto_boca)
    if referencia is None or inferior is None:
        return None

    # "Rojez" de cada píxel = Cr - Cb. La piel siempre tira a rojo (positivo) incluso con cámaras de
    # colores apagados; las mascarillas celestes, blancas o negras quedan en cero o negativo.
    rojez_ref = _rojez(referencia)
    rojez_inf = _rojez(inferior)
    # Umbral relativo a la piel de esta persona (la mediana ignora la montura de lentes), con un mínimo
    # fijo: si la referencia quedó sobre la mascarilla, igual se exige un tono claramente rojizo.
    umbral = float(np.clip(cfg.mascarilla_fraccion_rojez * np.median(rojez_ref),
                           cfg.mascarilla_rojez_min, cfg.mascarilla_rojez_max))
    piel = float((rojez_inf >= umbral).mean())
    return {"detectada": piel < cfg.mascarilla_piel_min, "piel_inferior": round(piel, 3)}


def _rojez(parche_bgr: np.ndarray) -> np.ndarray:
    ycrcb = cv2.cvtColor(parche_bgr, cv2.COLOR_BGR2YCrCb).reshape(-1, 3).astype(np.float32)
    return ycrcb[:, 1] - ycrcb[:, 2]
