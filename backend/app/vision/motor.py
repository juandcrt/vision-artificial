"""Motor de visión: detección facial + puntos faciales con MediaPipe (API Tasks) y OpenCV."""
from __future__ import annotations

import logging
import math
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from ..config import CONFIG, Config
from .calidad import iou, medir_calidad, pose_desde_matriz
from .mascarilla import detectar_mascarilla

log = logging.getLogger("vision.motor")

# Nombres de los 6 puntos clave que entrega BlazeFace
NOMBRES_PUNTOS = ["ojo_derecho", "ojo_izquierdo", "nariz", "boca", "oreja_derecha", "oreja_izquierda"]
BLENDSHAPES_INTERES = ("eyeBlinkLeft", "eyeBlinkRight", "jawOpen", "mouthSmileLeft", "mouthSmileRight")
# Los mismos 6 puntos tomados de la malla de 478 (centro del iris, punta de nariz, centro de
# la boca, tragos). Son más precisos que los del detector, sobre todo con lentes o mascarilla.
INDICES_MALLA = {"ojo_derecho": (468,), "ojo_izquierdo": (473,), "nariz": (1,), "boca": (13, 14),
                 "oreja_derecha": (234,), "oreja_izquierda": (454,)}


def compensar_luz(img_bgr: np.ndarray, objetivo: float, gamma_min: float) -> np.ndarray:
    """Aclara una imagen oscura con corrección gamma para que su brillo medio se acerque al objetivo.

    Las webcams bajan la exposición por su cuenta (p. ej. con una pared clara detrás) y la cara
    queda tan oscura que el detector la pierde. La gamma aclara las sombras sin saturar las luces.
    """
    media = float(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).mean())
    if media >= objetivo or media < 1:
        return img_bgr
    g = max(math.log(objetivo / 255) / math.log(media / 255), gamma_min)
    tabla = (np.linspace(0, 1, 256) ** g * 255).astype(np.uint8)
    return cv2.LUT(img_bgr, tabla)


def _extraer_mallas(res_lmk) -> list[dict]:
    """Mallas de FaceLandmarker con su caja, gestos y pose."""
    mallas = []
    for i, puntos in enumerate(res_lmk.face_landmarks or []):
        xs = [p.x for p in puntos]
        ys = [p.y for p in puntos]
        caja = {"x": min(xs), "y": min(ys), "ancho": max(xs) - min(xs), "alto": max(ys) - min(ys)}
        gestos = {}
        if res_lmk.face_blendshapes and i < len(res_lmk.face_blendshapes):
            gestos = {
                c.category_name: round(float(c.score), 3)
                for c in res_lmk.face_blendshapes[i]
                if c.category_name in BLENDSHAPES_INTERES
            }
        pose = None
        if res_lmk.facial_transformation_matrixes and i < len(res_lmk.facial_transformation_matrixes):
            pose = pose_desde_matriz(res_lmk.facial_transformation_matrixes[i])
        mallas.append({"caja": caja, "puntos": puntos, "gestos": gestos, "pose": pose, "usada": False})
    return mallas


def _puntos_desde_malla(puntos) -> list[dict]:
    return [{"nombre": nombre,
             "x": round(sum(puntos[i].x for i in idx) / len(idx), 4),
             "y": round(sum(puntos[i].y for i in idx) / len(idx), 4)}
            for nombre, idx in INDICES_MALLA.items()]


def _con_margen(caja: dict, margen_x: float, margen_y: float) -> dict:
    """Amplía una caja normalizada (fracción de su tamaño por lado) sin salirse de la imagen."""
    x0 = max(caja["x"] - caja["ancho"] * margen_x, 0.0)
    y0 = max(caja["y"] - caja["alto"] * margen_y, 0.0)
    x1 = min(caja["x"] + caja["ancho"] * (1 + margen_x), 1.0)
    y1 = min(caja["y"] + caja["alto"] * (1 + margen_y), 1.0)
    return {"x": round(x0, 4), "y": round(y0, 4), "ancho": round(x1 - x0, 4), "alto": round(y1 - y0, 4)}


class MotorVision:
    """Envuelve FaceDetector (cajas + 6 puntos) y FaceLandmarker (478 puntos, gestos y pose)."""

    def __init__(self, ruta_detector: Path, ruta_landmarker: Path, cfg: Config = CONFIG):
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions, vision

        self._mp = mp
        self.cfg = cfg
        self._lock = threading.Lock()  # los grafos de MediaPipe no son reentrantes

        self.detector = vision.FaceDetector.create_from_options(
            vision.FaceDetectorOptions(
                base_options=BaseOptions(model_asset_path=str(ruta_detector)),
                running_mode=vision.RunningMode.IMAGE,
                min_detection_confidence=0.3,  # filtramos después con el umbral del usuario
            )
        )
        self.landmarker = vision.FaceLandmarker.create_from_options(
            vision.FaceLandmarkerOptions(
                base_options=BaseOptions(model_asset_path=str(ruta_landmarker)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=cfg.max_rostros,
                min_face_detection_confidence=0.3,
                output_face_blendshapes=True,
                output_facial_transformation_matrixes=True,
            )
        )
        log.info("Motor de visión listo (MediaPipe %s, OpenCV %s)", mp.__version__, cv2.__version__)

    # ------------------------------------------------------------------ utilidades
    @staticmethod
    def decodificar(datos: bytes) -> np.ndarray:
        """Bytes JPEG/PNG -> imagen BGR de OpenCV."""
        arr = np.frombuffer(datos, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            raise ValueError("No se pudo decodificar la imagen (use JPEG o PNG).")
        return img

    def _preparar(self, img_bgr: np.ndarray) -> np.ndarray:
        h, w = img_bgr.shape[:2]
        if w > self.cfg.ancho_max_entrada:
            esc = self.cfg.ancho_max_entrada / w
            img_bgr = cv2.resize(img_bgr, (int(w * esc), int(h * esc)), interpolation=cv2.INTER_AREA)
        return img_bgr

    # ------------------------------------------------------------------ inferencia
    def procesar(self, img_bgr: np.ndarray, umbral: float | None = None, incluir_malla: bool = False) -> dict:
        """Procesa un fotograma y devuelve rostros con caja, confianza, puntos, pose, gestos y calidad.

        Todas las coordenadas van normalizadas (0-1) respecto a la imagen recibida,
        así el frontend puede dibujarlas a cualquier resolución.
        """
        t0 = time.perf_counter()
        umbral = self.cfg.umbral_deteccion if umbral is None else umbral
        img_bgr = self._preparar(img_bgr)
        h, w = img_bgr.shape[:2]
        img_det = compensar_luz(img_bgr, self.cfg.brillo_objetivo_deteccion, self.cfg.gamma_min)
        rgb = cv2.cvtColor(img_det, cv2.COLOR_BGR2RGB)
        imagen_mp = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))

        with self._lock:
            res_det = self.detector.detect(imagen_mp)
            res_lmk = self.landmarker.detect(imagen_mp)

        # --- Malla facial (FaceLandmarker) ---
        mallas = _extraer_mallas(res_lmk)

        # --- Detecciones (FaceDetector) ---
        rostros = []
        for det in res_det.detections or []:
            score = float(det.categories[0].score) if det.categories else 0.0
            if score < umbral - self.cfg.margen_con_malla:
                continue
            bb = det.bounding_box
            caja = {
                "x": round(bb.origin_x / w, 4),
                "y": round(bb.origin_y / h, 4),
                "ancho": round(bb.width / w, 4),
                "alto": round(bb.height / h, 4),
            }
            # Asociar la malla con mayor solapamiento
            mejor, mejor_iou = None, 0.25
            for m in mallas:
                if m["usada"]:
                    continue
                v = iou(caja, m["caja"])
                if v > mejor_iou:
                    mejor, mejor_iou = m, v
            if score < umbral and mejor is None:
                continue  # confianza baja y sin malla que lo confirme
            rostro = {
                "id": len(rostros),
                "confianza": round(score, 3),
                "caja": caja,
                "puntos_clave": [
                    {"nombre": NOMBRES_PUNTOS[k] if k < len(NOMBRES_PUNTOS) else f"p{k}",
                     "x": round(kp.x, 4), "y": round(kp.y, 4)}
                    for k, kp in enumerate(det.keypoints or [])
                ],
                "calidad": medir_calidad(img_bgr, caja),
                "pose": None,
                "gestos": {},
                "malla": None,
            }
            if mejor is not None:
                mejor["usada"] = True
                if len(mejor["puntos"]) > max(i for idx in INDICES_MALLA.values() for i in idx):
                    rostro["puntos_clave"] = _puntos_desde_malla(mejor["puntos"])
                # La malla cubre la cara entera (frente a barbilla); la caja del detector deja fuera
                # la frente y con mascarilla puede quedar desplazada. Calidad y distancia siguen
                # midiéndose con la caja del detector, para la que están calibrados los umbrales.
                rostro["caja"] = _con_margen(mejor["caja"], 0.06, 0.08)
                rostro["pose"] = mejor["pose"]
                rostro["gestos"] = mejor["gestos"]
                rostro["num_puntos_malla"] = len(mejor["puntos"])
                if incluir_malla:
                    rostro["malla"] = [[round(p.x, 4), round(p.y, 4)] for p in mejor["puntos"]]
            rostro["mascarilla"] = detectar_mascarilla(img_bgr, rostro["puntos_clave"], self.cfg)
            rostros.append(rostro)

        rostros.sort(key=lambda r: r["caja"]["ancho"], reverse=True)  # el más cercano primero
        for i, r in enumerate(rostros):
            r["id"] = i

        return {
            "ancho": w,
            "alto": h,
            "rostros": rostros,
            "tiempo_inferencia_ms": round((time.perf_counter() - t0) * 1000, 1),
        }

    def cerrar(self) -> None:
        for tarea in (self.detector, self.landmarker):
            try:
                tarea.close()
            except Exception:  # pragma: no cover
                pass
