"""Motor de caras con los mismos modelos y el mismo procesado que CompreFace (compilación FaceNet por defecto), ejecutados con
ONNX Runtime en lugar de TensorFlow 1:

  detección y cinco puntos   MTCNN (pnet, rnet, onet), umbrales y márgenes de CompreFace
  embedding                  FaceNet 20180402-114759 (VGGFace2), 512 dimensiones
  edad                       Levi y Hassner (Inception v3), ocho tramos de edad

Los modelos .onnx salen de tools/convert_models.py y se montan en MODELS_DIR (por defecto /models).
"""
import os
from pathlib import Path
from typing import Callable, Optional

import cv2
import numpy as np
import onnxruntime as ort

from .engine import Face
from .mtcnn_onnx import MTCNN
from .resize import squish_img

# ONNX Runtime trae telemetría (métricas de uso hacia Microsoft). No lleva imágenes, pero un servicio que procesa caras no debe
# enviar nada fuera: se desactiva antes de crear ninguna sesión.
ort.disable_telemetry_events()

# Ajustes de CompreFace (facenet.FaceDetector) para esta compilación.
IMAGE_SIZE = 160
FACE_MIN_SIZE = 20
SCALE_FACTOR = 0.709
DET_PROB_THRESHOLD = 0.85
STEPS_THRESHOLD = [0.9436513301, 0.7059968943, 0.5506904359]
LEFT_MARGIN = 0.2125984251968504
RIGHT_MARGIN = 0.2230769230769231
TOP_MARGIN = 0.10526315789473684
BOTTOM_MARGIN = 0.09868421052631579
IMG_LENGTH_LIMIT = int(os.environ.get("IMG_LENGTH_LIMIT", "640"))
AGE_LABELS = ((0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100))


def _session(path: Path) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.intra_op_num_threads = int(os.environ.get("ONNX_THREADS", "2"))
    return ort.InferenceSession(str(path), sess_options=options, providers=["CPUExecutionProvider"])


def _runner(session: ort.InferenceSession) -> Callable[[np.ndarray], list[np.ndarray]]:
    input_name = session.get_inputs()[0].name

    def run(batch: np.ndarray) -> list[np.ndarray]:
        return session.run(None, {input_name: batch.astype(np.float32)})

    return run


def prewhiten(img: np.ndarray) -> np.ndarray:
    """Misma normalización que facenet.prewhiten y agegender.helpers.prewhiten."""
    mean = np.mean(img)
    std = np.std(img)
    std_adj = np.maximum(std, 1.0 / np.sqrt(img.size))
    return np.multiply(np.subtract(img, mean), 1 / std_adj)


class OnnxEngine:
    def __init__(self, models_dir: Optional[str] = None):
        directory = Path(models_dir or os.environ.get("MODELS_DIR", "/models"))
        self._mtcnn = MTCNN(
            _runner(_session(directory / "mtcnn_pnet.onnx")),
            _runner(_session(directory / "mtcnn_rnet.onnx")),
            _runner(_session(directory / "mtcnn_onet.onnx")),
            min_face_size=FACE_MIN_SIZE,
            steps_threshold=STEPS_THRESHOLD,
            scale_factor=SCALE_FACTOR,
        )
        self._facenet = _runner(_session(directory / "facenet.onnx"))
        self._age = _runner(_session(directory / "age.onnx"))

    # --- detección -----------------------------------------------------------------------------------------------------

    def _detect(self, rgb: np.ndarray, det_prob_threshold: float) -> list[dict]:
        height, width = rgb.shape[:2]
        downscale = None
        small = rgb
        if max(width, height) > IMG_LENGTH_LIMIT:
            downscale = IMG_LENGTH_LIMIT / max(width, height)
            small = cv2.resize(rgb, (round(width * downscale), round(height * downscale)), interpolation=cv2.INTER_AREA)
        upscale = 1 / downscale if downscale else 1

        small_h, small_w = small.shape[:2]
        found = []
        for item in self._mtcnn.detect_faces(small):
            x, y, w, h = item["box"]
            keypoints = item["keypoints"]
            x_min = int(np.maximum(x - LEFT_MARGIN * w, 0))
            y_min = int(np.maximum(y - TOP_MARGIN * h, 0))
            x_max = int(np.minimum(x + w + RIGHT_MARGIN * w, small_w))
            y_max = int(np.minimum(y + h + BOTTOM_MARGIN * h, small_h))
            points = np.array(
                [keypoints[name] for name in ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")], dtype=float
            )
            probability = float(item["confidence"])
            if probability <= det_prob_threshold:
                continue
            found.append(
                {
                    "box": (int(x_min * upscale), int(y_min * upscale), int(x_max * upscale), int(y_max * upscale)),
                    "points": (points * upscale).astype(int),
                    "probability": probability,
                }
            )
        found.sort(key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), reverse=True)
        return found

    # --- modelos -------------------------------------------------------------------------------------------------------

    def _crop(self, rgb: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
        x_min, y_min, x_max, y_max = box
        return squish_img(np.ascontiguousarray(rgb[y_min:y_max, x_min:x_max, :]), IMAGE_SIZE)

    def _embedding(self, face_img: np.ndarray) -> np.ndarray:
        return self._facenet(prewhiten(face_img)[np.newaxis])[0][0]

    def _age_range(self, face_img: np.ndarray) -> tuple[int, int]:
        probabilities = self._age(prewhiten(face_img)[np.newaxis])[0][0]
        return AGE_LABELS[int(np.argmax(probabilities))]

    # --- contrato ------------------------------------------------------------------------------------------------------

    def find_faces(self, image: np.ndarray, with_age: bool = False, det_prob_threshold: float = DET_PROB_THRESHOLD) -> list[Face]:
        rgb = np.ascontiguousarray(image[:, :, ::-1])
        faces = []
        for item in self._detect(rgb, det_prob_threshold):
            x_min, y_min, x_max, y_max = item["box"]
            crop = self._crop(rgb, item["box"])
            faces.append(
                Face(
                    x_min=x_min,
                    y_min=y_min,
                    x_max=x_max,
                    y_max=y_max,
                    probability=item["probability"],
                    landmarks=[(int(px), int(py)) for px, py in item["points"]],
                    embedding=self._embedding(crop),
                    age=self._age_range(crop) if with_age else None,
                )
            )
        return faces
