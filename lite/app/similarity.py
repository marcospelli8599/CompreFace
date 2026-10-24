import numpy as np


def similarity(source: np.ndarray, target: np.ndarray, a: float, b: float) -> float:
    """Misma transformación que CompreFace (EuclideanDistanceClassifier): (tanh((a - distancia) * b) + 1) / 2, en [0, 1]."""
    distance = float(np.linalg.norm(np.asarray(source, dtype=np.float64) - np.asarray(target, dtype=np.float64)))
    return float((np.tanh((a - distance) * b) + 1.0) / 2.0)
