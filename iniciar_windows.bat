@echo off
chcp 65001 >nul
title Vision Facial IA
cd /d "%~dp0backend"

if not exist .venv (
    echo [1/3] Creando entorno virtual...
    python -m venv .venv || (echo No se encontro Python. Instale Python 3.10 a 3.12 desde python.org & pause & exit /b 1)
)
call .venv\Scripts\activate.bat

echo [2/3] Instalando dependencias...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt || (echo Error instalando dependencias & pause & exit /b 1)

echo [3/3] Iniciando servidor en http://localhost:8000
start "" http://localhost:8000
uvicorn app.main:app --host 0.0.0.0 --port 8000
pause
