"""Pruebas de la lógica de validación, calidad y API (con un motor simulado, sin cámara ni modelos)."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Config
from app.eventos import RegistroEventos
from app.main import crear_app
from app.vision.calidad import iou, medir_calidad, pose_desde_matriz
from app.vision.validacion import (OBSERVADO, SIN_ROSTRO, VALIDADO, VERIFICANDO, EstadoSesion, evaluar)

CFG = Config()


def rostro(confianza=0.95, ancho=0.3, brillo=120, nitidez=150, yaw=0, pitch=0, roll=0, blink=0.0):
    return {
        "id": 0, "confianza": confianza,
        "caja": {"x": 0.3, "y": 0.2, "ancho": ancho, "alto": ancho * 1.2},
        "puntos_clave": [], "malla": None,
        "calidad": {"brillo": brillo, "contraste": 40, "nitidez": nitidez, "tamano": ancho},
        "pose": {"yaw": yaw, "pitch": pitch, "roll": roll},
        "gestos": {"eyeBlinkLeft": blink, "eyeBlinkRight": blink},
    }


# ---------------------------------------------------------------- validación
def test_sin_rostro():
    assert evaluar([], EstadoSesion())["estado"] == SIN_ROSTRO


def test_multiples_rostros_observado():
    r = evaluar([rostro(), rostro()], EstadoSesion())
    assert r["estado"] == OBSERVADO
    assert any("2 rostros" in m for m in r["motivos"])


@pytest.mark.parametrize("kw,texto", [
    ({"brillo": 20}, "Iluminación"), ({"nitidez": 5}, "borrosa"),
    ({"ancho": 0.05}, "Acérquese"), ({"yaw": 40}, "frente"), ({"confianza": 0.5}, "Confianza"),
    ({"roll": 30}, "Enderece"), ({"roll": -30}, "Enderece"),
])
def test_fallas_de_calidad(kw, texto):
    r = evaluar([rostro(**kw)], EstadoSesion())
    assert r["estado"] == OBSERVADO and any(texto in m for m in r["motivos"])


def test_flujo_vivacidad_hasta_validado():
    s, t = EstadoSesion(), 1000.0
    # Sin parpadeo -> VERIFICANDO
    assert evaluar([rostro()], s, ahora=t)["estado"] == VERIFICANDO
    # Cierra y abre los ojos -> cuenta un parpadeo
    evaluar([rostro(blink=0.8)], s, ahora=t + 0.1)
    evaluar([rostro(blink=0.05)], s, ahora=t + 0.2)
    assert s.parpadeos == 1
    estados = [evaluar([rostro()], s, ahora=t + 0.3 + i * 0.1)["estado"] for i in range(CFG.frames_estables)]
    assert estados[-1] == VALIDADO
    # Fuera de la ventana de vivacidad vuelve a pedir parpadeo
    assert evaluar([rostro()], s, ahora=t + 60)["estado"] == VERIFICANDO


def test_modo_deteccion_no_exige_vivacidad():
    s = EstadoSesion(frames_validos=CFG.frames_estables)
    r = evaluar([rostro()], s, modo="deteccion")
    assert r["estado"] == VALIDADO
    assert all(v["clave"] != "vivacidad" for v in r["verificaciones"])


# ---------------------------------------------------------------- calidad
def test_metricas_calidad():
    img = np.full((200, 200, 3), 128, np.uint8)
    lisa = medir_calidad(img, {"x": 0.25, "y": 0.25, "ancho": 0.5, "alto": 0.5})
    assert lisa["brillo"] == pytest.approx(128, abs=1) and lisa["nitidez"] < 1
    ruido = np.random.default_rng(0).integers(0, 255, (200, 200, 3), dtype=np.uint8)
    assert medir_calidad(ruido, {"x": 0.25, "y": 0.25, "ancho": 0.5, "alto": 0.5})["nitidez"] > 100


def test_pose_identidad_y_iou():
    assert pose_desde_matriz(np.eye(4)) == {"yaw": 0.0, "pitch": 0.0, "roll": 0.0}
    a = {"x": 0, "y": 0, "ancho": 1, "alto": 1}
    assert iou(a, a) == 1 and iou(a, {"x": 2, "y": 2, "ancho": 1, "alto": 1}) == 0


# ---------------------------------------------------------------- API
class MotorFalso:
    """Simula el motor: devuelve un rostro frontal; alterna el parpadeo según un contador."""
    def __init__(self):
        self.n = 0

    decodificar = staticmethod(lambda datos: _decodificar(datos))

    def procesar(self, img, umbral=None, incluir_malla=False):
        self.n += 1
        blink = 0.9 if self.n == 2 else 0.0
        r = rostro(blink=blink)
        if incluir_malla:
            r["malla"] = [[0.5, 0.5]] * 478
        return {"ancho": img.shape[1], "alto": img.shape[0], "rostros": [r], "tiempo_inferencia_ms": 1.0}


def _decodificar(datos):
    img = cv2.imdecode(np.frombuffer(datos, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("No se pudo decodificar la imagen (use JPEG o PNG).")
    return img


def _jpeg():
    ok, buf = cv2.imencode(".jpg", np.full((240, 320, 3), 100, np.uint8))
    return buf.tobytes()


@pytest.fixture
def cliente(tmp_path):
    app = crear_app(motor=MotorFalso(), registro=RegistroEventos(tmp_path / "e.db"), servir_frontend=False)
    with TestClient(app) as c:
        yield c


def test_salud_y_config(cliente):
    assert cliente.get("/api/salud").json()["motor_listo"] is True
    assert "umbral_acceso" in cliente.get("/api/config").json()


def test_detectar_rest(cliente):
    r = cliente.post("/api/detectar?malla=true", files={"imagen": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 200
    d = r.json()
    assert len(d["rostros"]) == 1 and len(d["rostros"][0]["malla"]) == 478
    assert d["validacion"]["estado"] == VALIDADO


def test_detectar_imagen_invalida(cliente):
    r = cliente.post("/api/detectar", files={"imagen": ("f.jpg", b"no-es-imagen", "image/jpeg")})
    assert r.status_code == 400


def test_websocket_flujo_completo(cliente):
    with cliente.websocket_connect("/ws/stream") as ws:
        assert ws.receive_json()["tipo"] == "bienvenida"
        ws.send_json({"tipo": "config", "modo": "acceso", "umbral": 0.6})
        assert ws.receive_json()["opciones"]["umbral"] == 0.6
        estados = []
        for _ in range(3 + CFG.frames_estables):
            ws.send_bytes(_jpeg())
            res = ws.receive_json()
            assert res["tipo"] == "resultado"
            estados.append(res["validacion"]["estado"])
        assert estados[0] == VERIFICANDO and estados[-1] == VALIDADO
        ws.send_json({"tipo": "reiniciar"})
        assert ws.receive_json()["tipo"] == "reiniciado"
    eventos = cliente.get("/api/eventos").json()
    assert eventos and eventos[0]["estado"] == VALIDADO
    assert cliente.get("/api/estadisticas").json()["validados_hoy"] == 1


def test_motor_no_disponible(tmp_path):
    app = crear_app(motor=None, registro=RegistroEventos(tmp_path / "e.db"), servir_frontend=False,
                    cargar_motor=False)
    app.state.motor = None
    c = TestClient(app)  # sin "with": no se ejecuta el arranque, el motor queda en None
    r = c.post("/api/detectar", files={"imagen": ("f.jpg", _jpeg(), "image/jpeg")})
    assert r.status_code == 503
