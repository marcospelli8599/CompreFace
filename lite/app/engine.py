"""Contrato del motor de caras. La implementación real (ONNX) y la de pruebas cumplen la misma interfaz."""
from dataclasses import dataclass
from typing import Optional, Protocol

import numpy as np


@dataclass(frozen=True)
class Face:
    # Caja en píxeles de la imagen original.
    x_min: int
    y_min: int
    x_max: int
    y_max: int
    probability: float
    # Cinco puntos (x, y): ojo izquierdo, ojo derecho, nariz, comisura izquierda y comisura derecha de la boca.
    landmarks: list[tuple[float, float]]
    embedding: np.ndarray
    # Rango de edad aparente (bajo, alto), si el motor lo estima.
    age: Optional[tuple[int, int]] = None


class FaceEngine(Protocol):
    def find_faces(self, image: np.ndarray, with_age: bool = False) -> list[Face]:
        """Devuelve las caras de una imagen BGR (OpenCV), de mayor a menor probabilidad."""
        ...
