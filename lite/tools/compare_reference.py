"""Compara el motor ONNX con los resultados de referencia de CompreFace (tools/reference_outputs.py).

  python tools/compare_reference.py <reference.json> <carpeta con los .onnx> <carpeta con las imágenes>

Para cada imagen compara: número de caras, cajas y puntos faciales, embedding (diferencia máxima y coseno) y tramo de edad.
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.onnx_engine import OnnxEngine  # noqa: E402


def main(reference_path: str, models_dir: str, images_dir: str) -> int:
    reference = json.loads(Path(reference_path).read_text())
    engine = OnnxEngine(models_dir)
    problems = 0
    worst = {"box": 0, "landmarks": 0, "embedding": 0.0, "age_probabilities": 0.0}
    min_cosine = 1.0
    total_faces = 0
    for name, expected in reference.items():
        rgb = np.array(Image.open(Path(images_dir) / name).convert("RGB"))
        bgr = np.ascontiguousarray(rgb[:, :, ::-1])
        actual = engine.find_faces(bgr, with_age=True)
        if len(actual) != len(expected):
            print(f"{name}: caras {len(actual)} (referencia {len(expected)})")
            problems += 1
            continue
        for got, want in zip(actual, expected):
            total_faces += 1
            box_diff = max(abs(a - b) for a, b in zip((got.x_min, got.y_min, got.x_max, got.y_max), want["box"]))
            lm_diff = int(np.abs(np.array(got.landmarks) - np.array(want["landmarks"])).max())
            e_got, e_want = np.array(got.embedding), np.array(want["embedding"])
            emb_diff = float(np.abs(e_got - e_want).max())
            cosine = float(e_got @ e_want / (np.linalg.norm(e_got) * np.linalg.norm(e_want)))
            same_age = tuple(got.age) == tuple(want["age"])
            worst["box"] = max(worst["box"], box_diff)
            worst["landmarks"] = max(worst["landmarks"], lm_diff)
            worst["embedding"] = max(worst["embedding"], emb_diff)
            min_cosine = min(min_cosine, cosine)
            if box_diff or lm_diff or emb_diff > 1e-3 or not same_age:
                problems += 1
                print(f"{name}: caja Δ{box_diff} puntos Δ{lm_diff} embedding Δ{emb_diff:.2e} coseno {cosine:.6f} edad {'=' if same_age else 'DISTINTA'}")
    print(f"\n{len(reference)} imágenes, {total_faces} caras comparadas")
    print(f"Peor diferencia: caja {worst['box']} px, puntos {worst['landmarks']} px, embedding {worst['embedding']:.2e}, coseno mínimo {min_cosine:.7f}")
    print("RESULTADO:", "IGUAL" if problems == 0 else f"{problems} diferencias")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:4]))
