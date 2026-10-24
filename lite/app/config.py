"""Configuración: las claves de API llegan por Docker Secrets (fichero) o, en desarrollo, por variable de entorno."""
import os
from dataclasses import dataclass
from pathlib import Path


def _secret(name: str) -> str:
    """Lee NAME_FILE (ruta a un secreto) o, si no existe, NAME. Falla al arrancar si falta: sin clave no hay servicio."""
    file_path = os.environ.get(f"{name}_FILE")
    value = Path(file_path).read_text(encoding="utf-8").strip() if file_path else os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Falta {name} (o {name}_FILE)")
    return value


@dataclass(frozen=True)
class Settings:
    verification_api_key: str
    detection_api_key: str
    # Coeficientes de la transformación distancia -> similitud: (tanh((a - d) * b) + 1) / 2. Son los de CompreFace para FaceNet 20180402-114759.
    similarity_a: float
    similarity_b: float
    max_image_bytes: int


def load_settings() -> Settings:
    verification_key = _secret("VERIFICATION_API_KEY")
    detection_key = _secret("DETECTION_API_KEY")
    if verification_key == detection_key:
        raise RuntimeError("Las claves de verificación y de detección deben ser distintas")
    return Settings(
        verification_api_key=verification_key,
        detection_api_key=detection_key,
        similarity_a=float(os.environ.get("SIMILARITY_A", "1.1817961")),
        similarity_b=float(os.environ.get("SIMILARITY_B", "5.291995557")),
        max_image_bytes=int(os.environ.get("MAX_IMAGE_BYTES", str(5 * 1024 * 1024))),
    )
