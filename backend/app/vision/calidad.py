"""Métricas de calidad de imagen calculadas con OpenCV sobre la región del rostro."""
from __future__ import annotations

import math

import cv2
import numpy as np


def recortar_rostro(img_bgr: np.ndarray, caja: dict) -> np.ndarray | None:
    """Recorta la caja normalizada {x, y, ancho, alto} (0-1) de la imagen."""
    h, w = img_bgr.shape[:2]
    x0 = max(int(caja["x"] * w), 0)
    y0 = max(int(caja["y"] * h), 0)
    x1 = min(int((caja["x"] + caja["ancho"]) * w), w)
    y1 = min(int((caja["y"] + caja["alto"]) * h), h)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    return img_bgr[y0:y1, x0:x1]


def medir_calidad(img_bgr: np.ndarray, caja: dict) -> dict:
    """Brillo (media de gris), contraste (desv. estándar) y nitidez (varianza del Laplaciano).

    El recorte se normaliza a 128 px de ancho para que la nitidez sea comparable
    sin importar la distancia a la cámara.
    """
    roi = recortar_rostro(img_bgr, caja)
    if roi is None:
        return {"brillo": 0.0, "contraste": 0.0, "nitidez": 0.0, "tamano": float(caja["ancho"])}
    gris = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    escala = 128 / gris.shape[1]
    gris = cv2.resize(gris, (128, max(int(gris.shape[0] * escala), 1)), interpolation=cv2.INTER_AREA)
    return {
        "brillo": round(float(gris.mean()), 1),
        "contraste": round(float(gris.std()), 1),
        "nitidez": round(float(cv2.Laplacian(gris, cv2.CV_64F).var()), 1),
        "tamano": round(float(caja["ancho"]), 3),
    }


def pose_desde_matriz(matriz: np.ndarray) -> dict:
    """Ángulos de Euler (grados) a partir de la matriz de transformación facial 4x4."""
    r = np.asarray(matriz, dtype=float)[:3, :3]
    sy = math.hypot(r[2, 1], r[2, 2])
    pitch = math.degrees(math.atan2(r[2, 1], r[2, 2]))
    yaw = math.degrees(math.atan2(-r[2, 0], sy))
    roll = math.degrees(math.atan2(r[1, 0], r[0, 0]))
    return {"yaw": round(yaw, 1), "pitch": round(pitch, 1), "roll": round(roll, 1)}


def iou(a: dict, b: dict) -> float:
    """Intersección sobre unión entre dos cajas normalizadas."""
    ax1, ay1 = a["x"] + a["ancho"], a["y"] + a["alto"]
    bx1, by1 = b["x"] + b["ancho"], b["y"] + b["alto"]
    iw = max(0.0, min(ax1, bx1) - max(a["x"], b["x"]))
    ih = max(0.0, min(ay1, by1) - max(a["y"], b["y"]))
    inter = iw * ih
    union = a["ancho"] * a["alto"] + b["ancho"] * b["alto"] - inter
    return inter / union if union > 0 else 0.0
