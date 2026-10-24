"""Giro de la cabeza a partir de cinco puntos faciales, con solvePnP. Adaptado de CompreFace (PoseEstimatorMixin, Apache 2.0)."""
from typing import Optional

import cv2
import numpy as np

# Modelo 3D genérico de una cara (ojos, nariz, comisuras de la boca y barbilla).
_MODEL_3D = np.array(
    [
        [-8.0, 9.0, -8.0],  # ojo izquierdo
        [8.0, 9.0, -8.0],  # ojo derecho
        [0.0, 1.0, 0.0],  # nariz
        [-5.0, -4.0, -8.0],  # comisura izquierda
        [5.0, -4.0, -8.0],  # comisura derecha
        [0.0, -11.0, -4.0],  # barbilla
    ],
    dtype=np.float64,
)


def estimate_pose(landmarks: list[tuple[float, float]], image_size: tuple[int, int]) -> Optional[dict[str, float]]:
    """landmarks: ojo izq., ojo der., nariz, comisura izq., comisura der. image_size: (alto, ancho). Devuelve grados."""
    left_eye, right_eye, nose, mouth_left, mouth_right = (np.array(p, dtype=np.float64) for p in landmarks)
    nose_bridge = (left_eye + right_eye) / 2
    mouth_center = (mouth_left + mouth_right) / 2
    chin = mouth_center + (mouth_center - nose_bridge) / 2
    points_2d = np.array([left_eye, right_eye, nose, mouth_left, mouth_right, chin], dtype=np.float64)

    height, width = image_size
    focal = float(width)
    camera = np.array([[focal, 0, width / 2], [0, focal, height / 2], [0, 0, 1]], dtype=np.float64)
    ok, rotation, _ = cv2.solvePnP(_MODEL_3D, points_2d, camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_ITERATIVE)
    if not ok:
        return None
    matrix, _ = cv2.Rodrigues(rotation)
    angles, *_ = cv2.RQDecomp3x3(matrix)
    pitch, yaw, roll = (float(a) for a in angles)
    return {"pitch": pitch, "yaw": yaw, "roll": roll}
