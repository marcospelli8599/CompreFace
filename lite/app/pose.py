"""Giro de la cabeza a partir de cinco puntos faciales, con solvePnP.

Es el cálculo de CompreFace (PoseEstimatorMixin, Apache 2.0) tal cual, incluidas sus particularidades (cámara con el centro
óptico en (alto/2, ancho/2), divisiones enteras al deducir la barbilla y el signo de pitch y roll): Vero ajusta sus umbrales de
giro a estos valores, así que hay que reproducirlos y no «corregirlos».
"""
from typing import Optional, Sequence

import cv2
import numpy as np

# Modelo 3D genérico de una cara, en este orden: ojo izq., ojo der., nariz, comisura izq., comisura der., barbilla.
_MODEL_3D = np.array(
    [
        [-8.0, 9.0, -8.0],
        [8.0, 9.0, -8.0],
        [0.0, 1.0, 0.0],
        [-5.0, -4.0, -8.0],
        [5.0, -4.0, -8.0],
        [0.0, -11.0, -4.0],
    ],
    dtype=np.float64,
)


def estimate_pose(landmarks: Sequence[Sequence[int]], image_size: tuple[int, int]) -> Optional[dict[str, float]]:
    """landmarks: ojo izq., ojo der., nariz, comisura izq., comisura der., en píxeles enteros. image_size: (alto, ancho)."""
    (le_x, le_y), (re_x, re_y), nose, (ml_x, ml_y), (mr_x, mr_y) = (tuple(int(c) for c in p) for p in landmarks)
    bridge = ((le_x + re_x) // 2, (le_y + re_y) // 2)
    mouth = ((ml_x + mr_x) // 2, (ml_y + mr_y) // 2)
    chin = (mouth[0] + (mouth[0] - bridge[0]) // 2, mouth[1] + (mouth[1] - bridge[1]) // 2)
    points_2d = np.array([(le_x, le_y), (re_x, re_y), nose, (ml_x, ml_y), (mr_x, mr_y), chin], dtype=np.float64)

    height, width = image_size
    focal = float(width)
    camera = np.array([[focal, 1, height / 2], [0, focal, width / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(_MODEL_3D, points_2d, camera, np.zeros((4, 1), dtype=np.float64))
    if not ok:
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    angles = cv2.RQDecomp3x3(matrix)[0]
    return {
        "pitch": float(np.sign(angles[0]) * 180 - angles[0]),
        "yaw": float(angles[1]),
        "roll": float(np.sign(angles[2]) * 180 - angles[2]),
    }
