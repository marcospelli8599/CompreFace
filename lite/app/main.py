"""API compatible con el subconjunto de CompreFace que usa Vero: /verification/verify y /detection/detect."""
import hmac
from typing import Optional

import cv2
import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import Settings, load_settings
from .engine import Face, FaceEngine
from .pose import estimate_pose
from .similarity import similarity

# Códigos de error de CompreFace que Vero distingue (ver compreface.client.ts).
CODE_NO_FACE = 28
CODE_TOO_MANY_FACES = 31
CODE_BAD_IMAGE = 21
CODE_MISSING = 20


class ApiError(Exception):
    def __init__(self, status: int, code: int, message: str):
        self.status, self.code, self.message = status, code, message


def _check_key(provided: Optional[str], expected: str) -> None:
    """Comparación en tiempo constante; sin clave o con otra clave: 401."""
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        raise ApiError(401, 0, "Invalid API key")


def _decode(upload: UploadFile, limit: int) -> np.ndarray:
    data = upload.file.read(limit + 1)
    if len(data) > limit:
        raise ApiError(400, CODE_BAD_IMAGE, "Image is too large")
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ApiError(400, CODE_BAD_IMAGE, "Unsupported image")
    return image


def _box(face: Face) -> dict:
    return {
        "probability": round(face.probability, 5),
        "x_min": face.x_min,
        "y_min": face.y_min,
        "x_max": face.x_max,
        "y_max": face.y_max,
    }


def create_app(engine: FaceEngine, settings: Optional[Settings] = None) -> FastAPI:
    settings = settings or load_settings()
    app = FastAPI(title="compreface-lite", docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(ApiError)
    async def _api_error(_: Request, error: ApiError) -> JSONResponse:
        return JSONResponse(status_code=error.status, content={"message": error.message, "code": error.code})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, error: RequestValidationError) -> JSONResponse:
        return JSONResponse(status_code=400, content={"message": "Missing or invalid request part", "code": CODE_MISSING})

    @app.get("/healthcheck")
    def healthcheck() -> dict:
        return {"status": "OK"}

    @app.post("/api/v1/verification/verify")
    def verify(
        source_image: UploadFile = File(...),
        target_image: UploadFile = File(...),
        x_api_key: Optional[str] = Header(default=None),
    ) -> dict:
        _check_key(x_api_key, settings.verification_api_key)
        source_img = _decode(source_image, settings.max_image_bytes)
        target_img = _decode(target_image, settings.max_image_bytes)

        source_faces = engine.find_faces(source_img)
        if not source_faces:
            raise ApiError(400, CODE_NO_FACE, "No face is found in the given image")
        if len(source_faces) > 1:
            raise ApiError(400, CODE_TOO_MANY_FACES, "There is more than one face in the given image")
        target_faces = engine.find_faces(target_img)
        if not target_faces:
            raise ApiError(400, CODE_NO_FACE, "No face is found in the given image")

        matches = [
            {
                "box": _box(face),
                "similarity": round(
                    similarity(source_faces[0].embedding, face.embedding, settings.similarity_a, settings.similarity_b), 5
                ),
            }
            for face in target_faces
        ]
        matches.sort(key=lambda m: m["similarity"], reverse=True)
        return {"result": [{"source_image_face": {"box": _box(source_faces[0])}, "face_matches": matches}]}

    @app.post("/api/v1/detection/detect")
    def detect(
        file: UploadFile = File(...),
        face_plugins: str = Query(default=""),
        limit: int = Query(default=0, ge=0, le=20),
        x_api_key: Optional[str] = Header(default=None),
    ) -> dict:
        _check_key(x_api_key, settings.detection_api_key)
        image = _decode(file, settings.max_image_bytes)
        plugins = {p.strip() for p in face_plugins.split(",") if p.strip()}

        faces = engine.find_faces(image, with_age="age" in plugins)
        if not faces:
            raise ApiError(400, CODE_NO_FACE, "No face is found in the given image")
        if limit:
            faces = faces[:limit]

        height, width = image.shape[:2]
        result = []
        for face in faces:
            item: dict = {"box": _box(face)}
            if "pose" in plugins:
                pose = estimate_pose(face.landmarks, (height, width))
                if pose is not None:
                    item["pose"] = pose
            if "age" in plugins and face.age is not None:
                item["age"] = {"probability": 1.0, "low": face.age[0], "high": face.age[1]}
            result.append(item)
        return {"result": result}

    return app
