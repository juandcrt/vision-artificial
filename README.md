# Visión Facial IA — Detección de rostros en tiempo real

Aplicación web que detecta rostros en tiempo real con visión computacional e inteligencia artificial.
El **frontend** (HTML, CSS y JavaScript) muestra la cámara, dibuja las detecciones y ofrece el panel de control;
el **backend** (Python + FastAPI) procesa cada fotograma con **MediaPipe** y **OpenCV** y expone los resultados
por **API REST** y **WebSocket**.

Aplicaciones: seguridad, control de acceso y monitoreo inteligente.

---

## 1. Arquitectura

```
┌──────────────────────── NAVEGADOR (frontend/) ────────────────────────┐
│  getUserMedia → <video> ─┐                                              │
│                          ├─ cada fotograma: canvas → JPEG (320–640 px)  │
│  <canvas> superpuesto ◄──┼─ dibuja cajas, puntos clave, malla           │
│  Panel: modo, umbral,    │  Panel de validación, métricas, bitácora     │
│  FPS, interruptores      │                                              │
└──────────────┬───────────┴───────────────────────▲─────────────────────┘
               │ WebSocket /ws/stream (binario)     │ JSON con resultados
               ▼                                    │
┌──────────────────────── SERVIDOR (backend/) ───────────────────────────┐
│ FastAPI (app/main.py)                                                   │
│   ├─ vision/motor.py      MediaPipe FaceDetector  → caja + 6 puntos     │
│   │                       MediaPipe FaceLandmarker→ 478 puntos, gestos, │
│   │                                                  pose de la cabeza   │
│   ├─ vision/calidad.py    OpenCV: brillo, contraste, nitidez (Laplaciano)│
│   ├─ vision/mascarilla.py OpenCV: piel en boca/barbilla vs. entrecejo     │
│   ├─ vision/validacion.py Reglas de acceso + prueba de vida (parpadeo)  │
│   └─ eventos.py           Bitácora SQLite (solo metadatos)              │
└─────────────────────────────────────────────────────────────────────────┘
```

**Flujo de un fotograma:** el navegador captura → reduce a 480 px → JPEG → WebSocket → el backend decodifica
con OpenCV → MediaPipe detecta rostros y puntos → OpenCV mide la calidad → el motor de reglas decide el estado →
el backend devuelve JSON → el navegador dibuja y actualiza los paneles. El envío es con *contrapresión*
(no se manda un fotograma nuevo hasta recibir la respuesta anterior), así el sistema no se satura en equipos lentos.

## 2. Requisitos

| Elemento | Versión |
|---|---|
| Python | 3.10 a 3.12 recomendado (probado también en 3.13) |
| Navegador | Chrome, Edge o Firefox actualizados |
| Cámara web | Cualquiera compatible con el navegador |
| Internet | Solo la **primera vez**, para descargar los modelos (~4 MB) |

## 3. Instalación y ejecución

### Windows (doble clic)
Ejecute **`iniciar_windows.bat`**. Crea el entorno virtual, instala dependencias, descarga los modelos y abre
`http://localhost:8000`.

### Manual (Windows / Linux / macOS)
```bash
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate     Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
python descargar_modelos.py          # opcional: el servidor también lo hace al iniciar
uvicorn app.main:app --host 0.0.0.0 --port 8000
```
Abra **http://localhost:8000**, pulse **Iniciar cámara** y conceda el permiso.

> **Acceso desde otro equipo de la red:** los navegadores solo permiten la cámara en `localhost` o en `https`.
> Para usarlo desde otra PC/celular configure HTTPS (por ejemplo con un proxy Nginx o
> `uvicorn ... --ssl-keyfile clave.pem --ssl-certfile cert.pem`).

### Variables de entorno opcionales
| Variable | Defecto | Descripción |
|---|---|---|
| `VF_UMBRAL` | 0.5 | Confianza mínima para reportar un rostro |
| `VF_UMBRAL_ACCESO` | 0.75 | Confianza mínima en control de acceso |
| `VF_MAX_ROSTROS` | 5 | Máximo de rostros analizados con malla |

El resto de umbrales (brillo, nitidez, ángulos, ventana de vivacidad) está en `backend/app/config.py`.

## 4. Uso de la interfaz

- **Modo Detección:** detecta y describe todos los rostros (caja, confianza, puntos, pose, calidad).
- **Modo Control de acceso:** además exige una sola persona, buena calidad, mirada frontal y **parpadeo** (prueba de vida).
- **Umbral de confianza:** descarta detecciones débiles.
- **FPS máximos / Resolución de análisis:** equilibran fluidez vs. precisión según la PC.
- **Malla facial (478):** dibuja los puntos faciales de MediaPipe Face Landmarker.
- **Mascarilla:** la etiqueta de cada rostro indica *Mascarilla detectada* o *Sin mascarilla*. Se compara el
  tono de piel del entrecejo con el de la zona de boca y barbilla (OpenCV, espacio YCrCb). Es informativa:
  no cambia las reglas de validación. Una mascarilla de color piel podría no detectarse.
- **Capturar:** descarga un PNG del fotograma con las detecciones dibujadas.
- **Registro de eventos:** cambios de estado (Validado / Observado) con hora, sesión y motivo.

### Estados
| Estado | Significado |
|---|---|
| ⚪ Sin rostro | No hay nadie frente a la cámara |
| 🔴 Observado | Hay rostro pero falla una condición (varios rostros, oscuridad, borroso, lejos, de perfil, cabeza inclinada, confianza baja) |
| 🟡 Verificando | Todo correcto; falta el parpadeo o mantener la posición unos instantes |
| 🟢 Validado | Persona real, frontal, nítida y estable |

### Reglas de validación (control de acceso)
| Verificación | Criterio | Fuente |
|---|---|---|
| Un solo rostro | exactamente 1 detección | MediaPipe FaceDetector |
| Confianza | ≥ 0.75 | MediaPipe FaceDetector |
| Distancia | ancho del rostro ≥ 15 % de la imagen | Caja de detección |
| Iluminación | brillo medio del rostro entre 60 y 210 | OpenCV (escala de grises) |
| Nitidez | varianza del Laplaciano ≥ 40 | OpenCV |
| Mirada frontal | giro e inclinación ≤ 20° | Matriz de transformación facial (MediaPipe) |
| Cabeza recta | rotación lateral (roll) ≤ 20° | Matriz de transformación facial (MediaPipe) |
| Prueba de vida | parpadeo en los últimos 8 s | Blendshapes `eyeBlinkLeft/Right` (MediaPipe) |
| Estabilidad | 5 fotogramas válidos seguidos | Motor de reglas |

## 5. API

Documentación interactiva automática en **http://localhost:8000/docs** (Swagger).

| Método | Ruta | Descripción |
|---|---|---|
| GET | `/api/salud` | Estado del servidor y del motor de IA |
| GET | `/api/config` | Umbrales de validación vigentes |
| POST | `/api/detectar` | Analiza una imagen (multipart, campo `imagen`). Parámetros: `umbral`, `malla`, `modo` |
| GET | `/api/eventos?limite=50` | Bitácora de eventos |
| GET | `/api/estadisticas` | Totales por estado y validados del día |
| DELETE | `/api/eventos` | Limpia la bitácora |
| WS | `/ws/stream` | Flujo en tiempo real |

**Ejemplo REST:**
```bash
curl -F "imagen=@foto.jpg" "http://localhost:8000/api/detectar?malla=false&modo=deteccion"
```

**Respuesta (resumen):**
```json
{
  "ancho": 640, "alto": 480, "tiempo_inferencia_ms": 18.2,
  "rostros": [{
    "id": 0, "confianza": 0.94,
    "caja": {"x": 0.31, "y": 0.22, "ancho": 0.28, "alto": 0.37},
    "puntos_clave": [{"nombre": "ojo_derecho", "x": 0.39, "y": 0.33}, "..."],
    "pose": {"yaw": -4.1, "pitch": 6.3, "roll": 1.2},
    "gestos": {"eyeBlinkLeft": 0.03, "eyeBlinkRight": 0.04, "jawOpen": 0.01},
    "calidad": {"brillo": 118.5, "contraste": 41.2, "nitidez": 132.7, "tamano": 0.28},
    "mascarilla": {"detectada": false, "piel_inferior": 0.93},
    "malla": null
  }],
  "validacion": {"estado": "VALIDADO", "motivos": [], "parpadeos": 0,
                 "verificaciones": [{"clave": "unico", "etiqueta": "Un solo rostro", "ok": true}]}
}
```
Las coordenadas están **normalizadas (0 a 1)**, así cualquier cliente puede dibujarlas a su resolución.

**Protocolo WebSocket `/ws/stream`:**
1. El servidor envía `{"tipo":"bienvenida","sesion":"a1b2c3d4","motor_listo":true}`.
2. El cliente configura con texto JSON: `{"tipo":"config","modo":"acceso","umbral":0.5,"malla":false}`.
3. El cliente envía cada fotograma como **mensaje binario JPEG**; el servidor responde con el mismo JSON de
   `/api/detectar` más `tipo:"resultado"`, `evento` (si hubo cambio de estado) y `tiempo_servidor_ms`.
4. `{"tipo":"reiniciar"}` reinicia la prueba de vida; `{"tipo":"ping"}` responde `pong`.

## 6. Pruebas

```bash
cd backend
pytest -q
```
16 pruebas cubren reglas de validación, prueba de vida, métricas de calidad, endpoints REST y el flujo
WebSocket completo (con un motor simulado, sin necesidad de cámara ni modelos).

## 6.1 Publicación en Vercel

Toda la aplicación (API, WebSocket y frontend) se publica como **un solo proyecto de Vercel** usando una imagen
de contenedor (`Dockerfile.vercel`, preset `container` en `vercel.json`). El contenedor instala las bibliotecas
de sistema que MediaPipe necesita (EGL/OpenGL) y descarga los modelos durante la construcción.

```bash
vercel deploy          # vista previa
vercel deploy --prod   # producción
```

El frontend se conecta al mismo dominio (`frontend/config.js` deja `VF_API_URL` vacío). Si el servidor está en
otro lugar, se puede cambiar en el panel **Conexión al servidor** de la interfaz.

Limitaciones en Vercel: la bitácora SQLite vive en `/tmp` y se pierde cuando la instancia se recicla, y las
conexiones WebSocket se cierran al llegar a la duración máxima de la función (el navegador se reconecta solo).

## 7. Estructura

```
vision-facial-ia/
├── backend/
│   ├── app/
│   │   ├── main.py            API FastAPI (REST + WebSocket) y servidor del frontend
│   │   ├── config.py          Umbrales y rutas
│   │   ├── eventos.py         Bitácora SQLite
│   │   └── vision/
│   │       ├── motor.py       MediaPipe FaceDetector + FaceLandmarker
│   │       ├── calidad.py     OpenCV: brillo, nitidez, pose, IoU
│   │       ├── mascarilla.py  Detección de mascarilla
│   │       ├── validacion.py  Reglas de acceso y prueba de vida
│   │       └── modelos.py     Descarga automática de modelos
│   ├── tests/test_sistema.py
│   ├── descargar_modelos.py
│   └── requirements.txt
├── frontend/
│   ├── index.html
│   ├── css/estilos.css
│   └── js/app.js
├── iniciar_windows.bat
└── iniciar.sh
```

## 8. Alcance y limitaciones (importante)

- **Detecta y valida, no identifica.** El sistema confirma que hay *una persona real, frontal y bien
  capturada*, pero no sabe *quién es*. Para reconocimiento de identidad se necesita un modelo de
  *embeddings* faciales (p. ej. InsightFace/ArcFace o `face_recognition`) y un padrón de personas registradas.
- **La prueba de vida por parpadeo es básica.** Detiene fotos impresas, pero un video reproducido en una pantalla
  podría superarla. Para uso real se recomienda un modelo antisuplantación (*anti-spoofing*) o retos aleatorios
  (girar la cabeza, sonreír).
- Los modelos de MediaPipe usados son de **corto alcance** (rostros a menos de ~2 m), adecuados para un punto
  de acceso frente a la cámara; para vigilancia a distancia conviene un detector de largo alcance.
- La API no tiene autenticación: en producción agregue tokens (por ejemplo JWT) y HTTPS.

## 9. Protección de datos personales (Perú)

La imagen del rostro es un dato biométrico, y el Reglamento de la Ley N.º 29733 aprobado por
**D.S. N.º 016-2024-JUS** (vigente desde marzo de 2025) lo trata como **dato sensible**.
Por diseño, este sistema **no guarda imágenes ni vectores biométricos**: procesa los fotogramas en memoria y
solo registra metadatos (hora, estado, confianza). Si se amplía a reconocimiento de identidad o se almacenan
imágenes, se debe contar con consentimiento expreso de los titulares, informar la finalidad, aplicar medidas de
seguridad y evaluar el impacto, conforme a la norma. Verifique la normativa vigente antes de implementarlo
en un entorno real.

## 10. Próximas mejoras sugeridas

1. Reconocimiento de identidad con embeddings + registro de personas autorizadas (con consentimiento).
2. Modelo antisuplantación y retos aleatorios de prueba de vida.
3. Autenticación de la API y panel de administración.
4. Soporte para cámaras IP (RTSP) procesadas directamente en el backend con `cv2.VideoCapture`.
5. Exportación de la bitácora a Excel/CSV y alertas (correo o WhatsApp) ante eventos observados.
