#!/usr/bin/env bash
set -e
cd "$(dirname "$0")/backend"

if [ ! -d .venv ]; then
  echo "[1/3] Creando entorno virtual..."
  python3 -m venv .venv
fi
source .venv/bin/activate

echo "[2/3] Instalando dependencias..."
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "[3/3] Servidor en http://localhost:8000"
exec uvicorn app.main:app --host 0.0.0.0 --port 8000
