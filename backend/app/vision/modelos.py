"""Descarga y verificación de los modelos de MediaPipe."""
from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

from ..config import CONFIG, Config

log = logging.getLogger("vision.modelos")


def _descargar(url: str, destino: Path) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    temporal = destino.with_suffix(destino.suffix + ".parcial")
    log.info("Descargando modelo %s ...", destino.name)
    with urllib.request.urlopen(url, timeout=60) as resp, open(temporal, "wb") as f:
        while bloque := resp.read(1 << 16):
            f.write(bloque)
    temporal.replace(destino)
    log.info("Modelo guardado en %s (%.1f MB)", destino, destino.stat().st_size / 1e6)


def asegurar_modelos(cfg: Config = CONFIG) -> tuple[Path, Path]:
    """Devuelve las rutas de los modelos; los descarga si no existen."""
    det = cfg.dir_modelos / cfg.modelo_detector
    lmk = cfg.dir_modelos / cfg.modelo_landmarker
    for ruta, url in ((det, cfg.url_detector), (lmk, cfg.url_landmarker)):
        if not ruta.exists() or ruta.stat().st_size == 0:
            _descargar(url, ruta)
    return det, lmk
