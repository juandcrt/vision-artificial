"""Bitácora de eventos en SQLite (solo metadatos: NO se guardan imágenes ni biometría)."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime
from pathlib import Path


class RegistroEventos:
    def __init__(self, ruta: Path | str):
        self._lock = threading.Lock()
        Path(ruta).parent.mkdir(parents=True, exist_ok=True)
        self._con = sqlite3.connect(str(ruta), check_same_thread=False)
        self._con.execute(
            """CREATE TABLE IF NOT EXISTS eventos (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   fecha TEXT NOT NULL,
                   sesion TEXT NOT NULL,
                   modo TEXT NOT NULL,
                   estado TEXT NOT NULL,
                   num_rostros INTEGER NOT NULL,
                   confianza REAL,
                   motivos TEXT
               )"""
        )
        self._con.commit()

    def registrar(self, sesion: str, modo: str, estado: str, num_rostros: int,
                  confianza: float | None, motivos: list[str]) -> dict:
        fecha = datetime.now().isoformat(timespec="seconds")
        with self._lock:
            cur = self._con.execute(
                "INSERT INTO eventos (fecha, sesion, modo, estado, num_rostros, confianza, motivos) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (fecha, sesion, modo, estado, num_rostros, confianza, json.dumps(motivos, ensure_ascii=False)),
            )
            self._con.commit()
        return {"id": cur.lastrowid, "fecha": fecha, "sesion": sesion, "modo": modo, "estado": estado,
                "num_rostros": num_rostros, "confianza": confianza, "motivos": motivos}

    def listar(self, limite: int = 50) -> list[dict]:
        with self._lock:
            filas = self._con.execute(
                "SELECT id, fecha, sesion, modo, estado, num_rostros, confianza, motivos "
                "FROM eventos ORDER BY id DESC LIMIT ?", (limite,)
            ).fetchall()
        claves = ("id", "fecha", "sesion", "modo", "estado", "num_rostros", "confianza", "motivos")
        salida = []
        for f in filas:
            d = dict(zip(claves, f))
            d["motivos"] = json.loads(d["motivos"] or "[]")
            salida.append(d)
        return salida

    def estadisticas(self) -> dict:
        with self._lock:
            total = self._con.execute("SELECT COUNT(*) FROM eventos").fetchone()[0]
            por_estado = dict(self._con.execute("SELECT estado, COUNT(*) FROM eventos GROUP BY estado").fetchall())
            hoy = datetime.now().date().isoformat()
            validados_hoy = self._con.execute(
                "SELECT COUNT(*) FROM eventos WHERE estado = 'VALIDADO' AND fecha >= ?", (hoy,)
            ).fetchone()[0]
        return {"total_eventos": total, "por_estado": por_estado, "validados_hoy": validados_hoy}

    def limpiar(self) -> int:
        with self._lock:
            n = self._con.execute("DELETE FROM eventos").rowcount
            self._con.commit()
        return n
