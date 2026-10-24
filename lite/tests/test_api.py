import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.engine import Face
from app.main import create_app
from app.pose import estimate_pose
from app.similarity import similarity

VERIFY_KEY = "verify-key-0123456789"
DETECT_KEY = "detect-key-0123456789"
SETTINGS = Settings(VERIFY_KEY, DETECT_KEY, 1.1, 3.0, 1024 * 1024)

FRONTAL = [(40.0, 50.0), (80.0, 50.0), (60.0, 70.0), (45.0, 90.0), (75.0, 90.0)]


def png(color: int = 128) -> bytes:
    ok, data = cv2.imencode(".png", np.full((120, 120, 3), color, dtype=np.uint8))
    assert ok
    return data.tobytes()


def face(embedding, age=None, landmarks=FRONTAL) -> Face:
    return Face(10, 10, 100, 110, 0.99, landmarks, np.array(embedding, dtype=np.float64), age)


class FakeEngine:
    """Devuelve las caras que se le indican, en orden, una lista por imagen recibida."""

    def __init__(self, *per_call):
        self.calls = list(per_call)
        self.age_requests = []

    def find_faces(self, image, with_age=False):
        self.age_requests.append(with_age)
        return self.calls.pop(0)


def client(engine) -> TestClient:
    return TestClient(create_app(engine, SETTINGS))


def verify(c, key=VERIFY_KEY):
    headers = {"x-api-key": key} if key else {}
    return c.post(
        "/api/v1/verification/verify",
        headers=headers,
        files={"source_image": ("a.png", png(), "image/png"), "target_image": ("b.png", png(), "image/png")},
    )


def detect(c, key=DETECT_KEY, params="face_plugins=pose,age&limit=2"):
    headers = {"x-api-key": key} if key else {}
    return c.post(f"/api/v1/detection/detect?{params}", headers=headers, files={"file": ("f.png", png(), "image/png")})


# --- autenticación -----------------------------------------------------------------------------------------------------

def test_verify_requires_its_own_key():
    c = client(FakeEngine())
    assert verify(c, key=None).status_code == 401
    assert verify(c, key="otra").status_code == 401
    assert verify(c, key=DETECT_KEY).status_code == 401


def test_detect_requires_its_own_key():
    c = client(FakeEngine())
    assert detect(c, key=None).status_code == 401
    assert detect(c, key=VERIFY_KEY).status_code == 401


def test_healthcheck_is_open():
    assert client(FakeEngine()).get("/healthcheck").json() == {"status": "OK"}


def test_no_docs_exposed():
    c = client(FakeEngine())
    assert c.get("/docs").status_code == 404
    assert c.get("/openapi.json").status_code == 404


# --- verify -----------------------------------------------------------------------------------------------------------

def test_verify_same_embedding_is_high_similarity():
    c = client(FakeEngine([face([0.1] * 8)], [face([0.1] * 8)]))
    body = verify(c).json()
    assert body["result"][0]["face_matches"][0]["similarity"] > 0.99


def test_verify_different_embedding_is_low_similarity():
    c = client(FakeEngine([face([0.0] * 8)], [face([1.0] * 8)]))
    assert verify(c).json()["result"][0]["face_matches"][0]["similarity"] < 0.05


def test_verify_no_face_returns_code_28():
    response = verify(client(FakeEngine([], [face([0.0])])))
    assert response.status_code == 400 and response.json()["code"] == 28
    response = verify(client(FakeEngine([face([0.0])], [])))
    assert response.status_code == 400 and response.json()["code"] == 28


def test_verify_many_faces_in_source_returns_code_31():
    response = verify(client(FakeEngine([face([0.0]), face([1.0])], [face([0.0])])))
    assert response.status_code == 400 and response.json()["code"] == 31


def test_verify_rejects_garbage_image():
    c = client(FakeEngine())
    response = c.post(
        "/api/v1/verification/verify",
        headers={"x-api-key": VERIFY_KEY},
        files={"source_image": ("a.png", b"no soy una imagen", "image/png"), "target_image": ("b.png", png(), "image/png")},
    )
    assert response.status_code == 400 and response.json()["code"] == 21


def test_verify_missing_part_is_400():
    c = client(FakeEngine())
    response = c.post("/api/v1/verification/verify", headers={"x-api-key": VERIFY_KEY}, files={"source_image": ("a.png", png(), "image/png")})
    assert response.status_code == 400


# --- detect -----------------------------------------------------------------------------------------------------------

def test_detect_returns_pose_and_age():
    engine = FakeEngine([face([0.0], age=(25, 32))])
    body = detect(client(engine)).json()
    item = body["result"][0]
    assert set(item["pose"]) == {"pitch", "yaw", "roll"}
    assert item["age"] == {"probability": 1.0, "low": 25, "high": 32}
    assert engine.age_requests == [True]


def test_detect_without_age_plugin_does_not_ask_the_engine_for_age():
    engine = FakeEngine([face([0.0], age=(25, 32))])
    item = detect(client(engine), params="face_plugins=pose").json()["result"][0]
    assert "age" not in item and "pose" in item
    assert engine.age_requests == [False]


def test_detect_respects_limit():
    engine = FakeEngine([face([0.0]), face([1.0]), face([2.0])])
    assert len(detect(client(engine), params="face_plugins=pose&limit=2").json()["result"]) == 2


def test_detect_no_face_returns_code_28():
    response = detect(client(FakeEngine([])))
    assert response.status_code == 400 and response.json()["code"] == 28


# --- pieza a pieza ----------------------------------------------------------------------------------------------------

def test_similarity_is_in_unit_range():
    for distance in (0.0, 0.5, 1.1, 2.0, 5.0):
        value = similarity(np.zeros(4), np.array([distance, 0, 0, 0]), 1.1, 3.0)
        assert 0.0 <= value <= 1.0
    assert similarity(np.zeros(4), np.zeros(4), 1.1, 3.0) > similarity(np.zeros(4), np.ones(4), 1.1, 3.0)


def test_pose_frontal_has_small_yaw():
    pose = estimate_pose(FRONTAL, (120, 120))
    assert pose is not None and abs(pose["yaw"]) < 20


def test_pose_yaw_changes_with_head_turn():
    # La nariz desplazada hacia un lado respecto al centro de los ojos simula un giro.
    turned = [(40.0, 50.0), (80.0, 50.0), (75.0, 70.0), (45.0, 90.0), (75.0, 90.0)]
    frontal_yaw = estimate_pose(FRONTAL, (120, 120))["yaw"]
    turned_yaw = estimate_pose(turned, (120, 120))["yaw"]
    assert abs(turned_yaw - frontal_yaw) > 10


def test_settings_reject_equal_keys(monkeypatch):
    from app.config import load_settings

    monkeypatch.setenv("VERIFICATION_API_KEY", "misma")
    monkeypatch.setenv("DETECTION_API_KEY", "misma")
    with pytest.raises(RuntimeError):
        load_settings()


def test_settings_read_secret_files(tmp_path, monkeypatch):
    from app.config import load_settings

    (tmp_path / "v").write_text("clave-verificacion\n")
    (tmp_path / "d").write_text("clave-deteccion\n")
    monkeypatch.setenv("VERIFICATION_API_KEY_FILE", str(tmp_path / "v"))
    monkeypatch.setenv("DETECTION_API_KEY_FILE", str(tmp_path / "d"))
    monkeypatch.delenv("VERIFICATION_API_KEY", raising=False)
    monkeypatch.delenv("DETECTION_API_KEY", raising=False)
    settings = load_settings()
    assert settings.verification_api_key == "clave-verificacion" and settings.detection_api_key == "clave-deteccion"
