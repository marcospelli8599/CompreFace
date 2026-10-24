"""Resultados de REFERENCIA: ejecuta los modelos originales de CompreFace (TensorFlow, MTCNN en Keras) sobre unas imágenes y
guarda lo que producen. tools/compare_reference.py compara después el motor ONNX con este fichero.

Se ejecuta en el contenedor de conversión (TensorFlow 2.10); ver convert.Dockerfile.

  /compreface/embedding-calculator/...   código de CompreFace (MTCNN, red de edad)
  /models/facenet, /models/age           modelos originales
  /images/*.jpg|png                      imágenes a procesar
  /out/reference.json                    salida
"""
import glob
import importlib.util
import json
import os
import sys

import cv2
import imageio
import numpy as np
import tensorflow as tf
from skimage import transform

EC = "/compreface/embedding-calculator"
sys.path.insert(0, f"{EC}/srcext")
from mtcnn.mtcnn import MTCNN  # noqa: E402

tf1 = tf.compat.v1
IMG_LENGTH_LIMIT = 640
MARGINS = dict(left=0.2125984251968504, right=0.2230769230769231, top=0.10526315789473684, bottom=0.09868421052631579)
AGE_LABELS = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]


def load_helpers():
    spec = importlib.util.spec_from_file_location("agegender_helpers", f"{EC}/src/services/facescan/plugins/agegender/helpers.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def facenet_session():
    graph_def = tf1.GraphDef()
    with open("/models/facenet/20180402-114759.pb", "rb") as f:
        graph_def.ParseFromString(f.read())
    graph = tf1.Graph()
    with graph.as_default():
        tf1.import_graph_def(graph_def, name="")
        session = tf1.Session(graph=graph)
    return graph, session


def age_session(helpers):
    graph = tf1.Graph()
    with graph.as_default():
        session = tf1.Session(graph=graph)
        images = tf1.placeholder(tf.float32, [None, 160, 160, 3])
        logits = helpers.inception_v3(8, images)
        softmax = tf.nn.softmax(logits)
        tf1.global_variables_initializer()
        tf1.train.Saver().restore(session, "/models/age/checkpoint-14999")
    return session, images, softmax


def main():
    helpers = load_helpers()
    detector = MTCNN(min_face_size=20, steps_threshold=[0.9436513301, 0.7059968943, 0.5506904359], scale_factor=0.709)
    fn_graph, fn_session = facenet_session()
    age_sess, age_in, age_out = age_session(helpers)

    results = {}
    for path in sorted(glob.glob("/images/*.jpg")):
        img = imageio.imread(path)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        img = img[:, :, 0:3]
        height, width = img.shape[:2]
        scaled, upscale = img, 1.0
        if max(width, height) > IMG_LENGTH_LIMIT:
            coefficient = IMG_LENGTH_LIMIT / max(width, height)
            scaled = cv2.resize(img, (round(width * coefficient), round(height * coefficient)), interpolation=cv2.INTER_AREA)
            upscale = 1 / coefficient
        sh, sw = scaled.shape[:2]

        faces = []
        for item in detector.detect_faces(scaled):
            x, y, w, h = item["box"]
            box = (
                int(np.maximum(x - MARGINS["left"] * w, 0)),
                int(np.maximum(y - MARGINS["top"] * h, 0)),
                int(np.minimum(x + w + MARGINS["right"] * w, sw)),
                int(np.minimum(y + h + MARGINS["bottom"] * h, sh)),
            )
            names = ("left_eye", "right_eye", "nose", "mouth_left", "mouth_right")
            points = np.array([item["keypoints"][n] for n in names], dtype=float)
            if item["confidence"] <= 0.85:
                continue
            ox = tuple(int(c * upscale) for c in box)
            crop = transform.resize(img[ox[1]:ox[3], ox[0]:ox[2], :], (160, 160))
            white = helpers.prewhiten(crop)
            embedding = fn_session.run(
                fn_graph.get_tensor_by_name("embeddings:0"),
                {
                    fn_graph.get_tensor_by_name("input:0"): white[np.newaxis],
                    fn_graph.get_tensor_by_name("phase_train:0"): False,
                },
            )[0]
            probabilities = age_sess.run(age_out, {age_in: white[np.newaxis]})[0]
            faces.append(
                {
                    "box": ox,
                    "probability": float(item["confidence"]),
                    "landmarks": (points * upscale).astype(int).tolist(),
                    "embedding": embedding.tolist(),
                    "age_probabilities": probabilities.tolist(),
                    "age": AGE_LABELS[int(np.argmax(probabilities))],
                }
            )
        faces.sort(key=lambda f: (f["box"][2] - f["box"][0]) * (f["box"][3] - f["box"][1]), reverse=True)
        results[os.path.basename(path)] = faces
        print(os.path.basename(path), len(faces), "caras")

    with open("/out/reference.json", "w") as f:
        json.dump(results, f)


if __name__ == "__main__":
    main()
