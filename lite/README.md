# compreface-lite

Servicio mínimo de reconocimiento facial, compatible con el subconjunto de la API de CompreFace que usa Vero:

- `POST /api/v1/verification/verify` (`source_image`, `target_image`; cabecera `x-api-key` de verificación) -> similitud.
- `POST /api/v1/detection/detect?face_plugins=pose,age&limit=2` (`file`; cabecera `x-api-key` de detección) -> caja, giro de la cabeza (`pose`) y rango de edad (`age`).
- `GET /healthcheck`.

Sin panel de administración ni base de datos. Las dos claves de API distintas llegan por Docker Secrets
(`VERIFICATION_API_KEY_FILE`, `DETECTION_API_KEY_FILE`) o, en desarrollo, por `VERIFICATION_API_KEY` y `DETECTION_API_KEY`.
Cada ruta solo acepta su clave (401 si no).

Errores con el mismo formato que CompreFace: `{"message": ..., "code": N}`; `28` = no hay cara, `31` = varias caras en la imagen de
origen, `21` = imagen no válida, `20` = falta una parte de la petición.

## Estado

Esqueleto con contrato, autenticación, similitud, pose y pruebas (`pytest`, motor simulado). **Falta el motor real**
(`app/engine.py`: detección, cinco puntos faciales, embedding y, si se resuelve, edad) con modelos de licencia comercial
clara, cargados con ONNX Runtime. La pose usa los cinco puntos con `solvePnP` (adaptado de CompreFace, Apache 2.0).

## Pruebas

    pip install -r requirements-dev.txt
    pytest
