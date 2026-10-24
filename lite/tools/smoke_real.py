"""Prueba de humo con el motor real a través de la API (sin red): misma persona, personas distintas, edad y giro.

  python tools/smoke_real.py <carpeta con los .onnx> <carpeta con las imágenes de ejemplo de CompreFace>

Necesita httpx (pip install httpx). Las imágenes de ejemplo son las de CompreFace (001-006_A = misma persona, 007-008_B = otra,
009_C = otra).
"""
import itertools
import sys
import time
from pathlib import Path

from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from app.onnx_engine import OnnxEngine  # noqa: E402

VERIFY, DETECT = "k-verify-0123456789", "k-detect-0123456789"


def main(models_dir: str, images_dir: str) -> None:
    started = time.time()
    engine = OnnxEngine(models_dir)
    client = TestClient(create_app(engine, Settings(VERIFY, DETECT, 1.1817961, 5.291995557, 5 * 1024 * 1024)))
    print(f"modelos cargados en {time.time() - started:.1f} s")

    def load(name: str) -> bytes:
        return (Path(images_dir) / name).read_bytes()

    def verify(a: str, b: str) -> float:
        response = client.post(
            "/api/v1/verification/verify",
            headers={"x-api-key": VERIFY},
            files={"source_image": (a, load(a), "image/jpeg"), "target_image": (b, load(b), "image/jpeg")},
        )
        assert response.status_code == 200, response.text
        return response.json()["result"][0]["face_matches"][0]["similarity"]

    same = [verify(a, b) for a, b in itertools.combinations(["001_A.jpg", "002_A.jpg", "003_A.jpg", "004_A.jpg", "005_A.jpg", "006_A.jpg"], 2)]
    same_b = [verify("007_B.jpg", "008_B.jpg")]
    different = [verify(a, b) for a in ("001_A.jpg", "002_A.jpg", "003_A.jpg") for b in ("007_B.jpg", "009_C.jpg")]
    print(f"misma persona (A-A, {len(same)} pares): mín {min(same):.3f} media {sum(same)/len(same):.3f}")
    print(f"misma persona (B-B): {same_b[0]:.3f}")
    print(f"personas distintas ({len(different)} pares): máx {max(different):.3f} media {sum(different)/len(different):.3f}")

    for name in ("001_A.jpg", "004_A.jpg", "007_B.jpg", "009_C.jpg"):
        started = time.time()
        response = client.post(
            "/api/v1/detection/detect?face_plugins=pose,age&limit=2",
            headers={"x-api-key": DETECT},
            files={"file": (name, load(name), "image/jpeg")},
        )
        item = response.json()["result"][0]
        pose = item["pose"]
        print(f"{name}: edad {item['age']['low']}-{item['age']['high']} yaw {pose['yaw']:.1f} pitch {pose['pitch']:.1f} ({(time.time()-started)*1000:.0f} ms)")

    none = client.post("/api/v1/detection/detect", headers={"x-api-key": DETECT}, files={"file": ("017_0.jpg", load("017_0.jpg"), "image/jpeg")})
    print("sin caras:", none.status_code, none.json())


if __name__ == "__main__":
    main(*sys.argv[1:3])
