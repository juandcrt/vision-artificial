"""Descarga anticipada de los modelos de MediaPipe (opcional: el servidor también lo hace al arrancar)."""
from app.vision.modelos import asegurar_modelos

if __name__ == "__main__":
    det, lmk = asegurar_modelos()
    print(f"Modelos listos:\n  {det}\n  {lmk}")
