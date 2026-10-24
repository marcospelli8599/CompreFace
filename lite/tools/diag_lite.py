"""Diagnostico de la edad, parte 2 (motor ONNX): con los MISMOS recortes que dio TensorFlow compara probabilidades de edad y
embeddings, y despues pasa las imagenes por la deteccion completa de lite para ver si el recorte difiere."""
import json
import sys

import cv2
import numpy as np

sys.path.insert(0, "/srv")
from app.onnx_engine import AGE_LABELS, OnnxEngine, prewhiten  # noqa: E402

d = np.load("/out/diag_ref.npz")
meta = json.load(open("/out/diag_meta.json"))
crops, tf_probs, tf_embs = d["crops"].astype(np.float64), d["probs"], d["embs"]
engine = OnnxEngine("/models")

on_probs = np.array([engine._age(prewhiten(c)[np.newaxis])[0][0] for c in crops])
on_embs = np.array([engine._facenet(prewhiten(c)[np.newaxis])[0][0] for c in crops])

print(f"=== PARTE 1: mismo recorte por el modelo original y el convertido ({len(crops)} caras) ===")
diff = np.abs(on_probs - tf_probs)
print(f"edad, probabilidades: diferencia maxima {diff.max():.2e}   media {diff.mean():.2e}")
a, b = tf_probs.argmax(1), on_probs.argmax(1)
print(f"edad, mismo tramo: {int((a == b).sum())}/{len(a)}")
for i in np.where(a != b)[0][:8]:
    top = np.sort(tf_probs[i])[::-1][:2]
    print(f"   cara {i}: TF {AGE_LABELS[a[i]]} p={top[0]:.3f} (2o {top[1]:.3f})  ONNX {AGE_LABELS[b[i]]} p={on_probs[i].max():.3f}")
cos = (tf_embs * on_embs).sum(1) / (np.linalg.norm(tf_embs, axis=1) * np.linalg.norm(on_embs, axis=1))
print(f"embedding: coseno minimo {cos.min():.6f}   diferencia maxima {np.abs(tf_embs - on_embs).max():.2e}")

print("\n=== PARTE 2: deteccion completa de lite frente a la de TensorFlow ===")
by_path = {}
for i, m in enumerate(meta):
    by_path.setdefault(m["path"], []).append(i)
n_img = n_same_count = 0
age_total = age_same = age_diff_crop = age_diff_model = 0
iou_list = []
for path, idxs in by_path.items():
    img = cv2.imread(path)
    faces = engine.find_faces(img, with_age=True)
    n_img += 1
    if len(faces) == len(idxs):
        n_same_count += 1
    for j, i in enumerate(idxs[: len(faces)]):
        f = faces[j]
        ref = meta[i]["box"]
        x1, y1, x2, y2 = max(ref[0], f.x_min), max(ref[1], f.y_min), min(ref[2], f.x_max), min(ref[3], f.y_max)
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        union = (ref[2] - ref[0]) * (ref[3] - ref[1]) + (f.x_max - f.x_min) * (f.y_max - f.y_min) - inter
        iou_list.append(inter / union)
        tf_age = AGE_LABELS[a[i]]
        age_total += 1
        if f.age == tf_age:
            age_same += 1
        elif tuple(ref) == (f.x_min, f.y_min, f.x_max, f.y_max):
            age_diff_model += 1    # misma caja y distinta edad: no puede ser el recorte
        else:
            age_diff_crop += 1     # la caja ya difiere
print(f"imagenes con el mismo numero de caras: {n_same_count}/{n_img}")
print(f"caja: IoU medio {np.mean(iou_list):.4f}   cajas identicas {sum(1 for v in iou_list if v > 0.9999)}/{len(iou_list)}   IoU minimo {min(iou_list):.3f}")
print(f"edad de la canalizacion completa: igual {age_same}/{age_total}   distinta con caja identica {age_diff_model}   distinta con caja distinta {age_diff_crop}")
