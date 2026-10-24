"""Diagnostico de la edad, parte 1 (entorno TensorFlow): ejecuta MTCNN y los modelos ORIGINALES sobre fotos de LFW y guarda los
recortes de 160x160 con las probabilidades de edad y los embeddings que da TensorFlow. Salida: /out/diag_ref.npz"""
import glob
import json
import os
import random
import sys

sys.path.insert(0, "/compreface/lite/tools")
import cv2  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
from skimage import transform  # noqa: E402

import reference_outputs as R  # noqa: E402

N = int(os.environ.get("N_IMAGES", "400"))
rng = random.Random(7)
people = sorted(os.listdir("/data/lfw"))
picks = []
for p in rng.sample(people, min(N, len(people))):
    picks.append(rng.choice(sorted(glob.glob(f"/data/lfw/{p}/*.jpg"))))

helpers = R.load_helpers()
detector = R.MTCNN(min_face_size=20, steps_threshold=[0.9436513301, 0.7059968943, 0.5506904359], scale_factor=0.709)
fn_graph, fn_session = R.facenet_session()
age_sess, age_in, age_out = R.age_session(helpers)

crops, probs, embs, meta = [], [], [], []
for k, path in enumerate(picks):
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    img = img[:, :, 0:3]
    height, width = img.shape[:2]
    scaled, upscale = img, 1.0
    if max(width, height) > R.IMG_LENGTH_LIMIT:
        c = R.IMG_LENGTH_LIMIT / max(width, height)
        scaled = cv2.resize(img, (round(width * c), round(height * c)), interpolation=cv2.INTER_AREA)
        upscale = 1 / c
    sh, sw = scaled.shape[:2]
    found = []
    for item in detector.detect_faces(scaled):
        if item["confidence"] <= 0.85:
            continue
        x, y, w, h = item["box"]
        box = (int(np.maximum(x - R.MARGINS["left"] * w, 0)), int(np.maximum(y - R.MARGINS["top"] * h, 0)),
               int(np.minimum(x + w + R.MARGINS["right"] * w, sw)), int(np.minimum(y + h + R.MARGINS["bottom"] * h, sh)))
        ox = tuple(int(v * upscale) for v in box)
        found.append(ox)
    found.sort(key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    for index, ox in enumerate(found):
        crop = transform.resize(img[ox[1]:ox[3], ox[0]:ox[2], :], (160, 160))
        white = helpers.prewhiten(crop)
        p = age_sess.run(age_out, {age_in: white[np.newaxis]})[0]
        e = fn_session.run(fn_graph.get_tensor_by_name("embeddings:0"),
                           {fn_graph.get_tensor_by_name("input:0"): white[np.newaxis], fn_graph.get_tensor_by_name("phase_train:0"): False})[0]
        crops.append(crop.astype(np.float32)); probs.append(p); embs.append(e)
        meta.append({"path": path, "index": index, "box": list(ox), "faces_in_image": len(found)})
    if (k + 1) % 50 == 0:
        print(k + 1, "imagenes,", len(crops), "caras", flush=True)

np.savez_compressed("/out/diag_ref.npz", crops=np.array(crops), probs=np.array(probs), embs=np.array(embs))
json.dump(meta, open("/out/diag_meta.json", "w"))
print("guardado:", len(crops), "caras de", len(picks), "imagenes")
