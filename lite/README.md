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

## Motor

Los mismos modelos y el mismo procesado que la compilación FaceNet (la de por defecto) de CompreFace, ejecutados con ONNX Runtime
en lugar de TensorFlow 1 (`app/onnx_engine.py`):

| | Modelo |
|---|---|
| Detección y cinco puntos | MTCNN (pnet, rnet, onet), umbrales y márgenes de CompreFace |
| Embedding | FaceNet 20180402-114759 (entrenado con VGGFace2), 512 dimensiones |
| Edad | Levi y Hassner (Inception v3), ocho tramos: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100 |
| Pose | `solvePnP` sobre los cinco puntos, tal cual CompreFace (`app/pose.py`) |
| Similitud | `(tanh((1.1817961 - distancia) * 5.291995557) + 1) / 2` |

**Licencias de los modelos:** el código de CompreFace es Apache 2.0, pero los pesos tienen su propio origen (datasets VGGFace2 y
CASIA para FaceNet; sin licencia declarada para la red de edad). Se usan los mismos que ya corría CompreFace; no se ha
resuelto su uso comercial.

## Modelos

Los `.onnx` no van en el repositorio ni en la imagen: se generan una vez a partir de los modelos originales de CompreFace
(`tools/convert_models.py`, dentro del contenedor de `tools/convert.Dockerfile`) y se montan en `/models`.

## Comprobaciones hechas

- `pytest`: contrato, claves, errores, similitud y pose (motor simulado).
- `tools/reference_outputs.py` + `tools/compare_reference.py`: los modelos originales (TensorFlow) y los ONNX sobre las 18
  imágenes de ejemplo de CompreFace (46 caras). Mismo número de caras en todas las imágenes; 43 de 46 caras idénticas (caja,
  puntos y embedding por debajo de 1e-3); 3 con diferencias de 1-3 px en la caja o los puntos (redondeos de coma flotante en
  MTCNN), que mueven un poco el recorte: coseno del embedding >= 0,97 y un tramo de edad distinto en una cara.
- `tools/smoke_real.py`: misma persona 0,970-0,996; personas distintas <= 0,215; edad 25-32 en las tres personas de ejemplo
  (30, 25 y 30 años reales).

## Comparación con CompreFace 1.2.0 (oct 2026)

`tools/compare/`: las mismas llamadas de Vero contra el CompreFace real y contra este servicio, con fotos de LFW y con fotos de
prueba de PRO (solo se miraron números). Enviando los mismos píxeles (PNG, fotos de 250 a 1280 px):

| | Resultado |
|---|---|
| Similitud (verificar) | misma persona: diferencia máx. 0,002; 0 decisiones distintas con 0,85 / 0,6 / 0,5 (de 199 a 220 comparaciones) |
| Fotos de PRO (133 parejas) | diferencia 0,0000; caras y error iguales en 29/29; edad igual 17/17 |
| Caras y errores | mismo número de caras 120-138/138; mismos códigos (28 sin cara, 31 varias caras) |
| Giro de la cabeza | diferencia media 0,00-0,01 grados, máx. 0,07 |
| Edad | mismo tramo 116/116 con fotos de 1280 px; 0 diferencias en el filtro de menores |
| Memoria | ~0,4-0,6 GB frente a ~2,1 GB (core + api + admin) |

Lo que explicó las diferencias que aparecieron al principio, y está corregido o documentado:

- **Recorte a 160x160** (`app/resize.py`): scikit-image 0.17.2 (el de CompreFace) cuantiza el suavizado a enteros de 8 bits antes
  de interpolar; las versiones nuevas no. Error medio frente a CompreFace: 3e-3 con el resize actual, 1e-6 con la réplica. Afectaba
  a cualquier recorte reducido (fotos de móvil).
- **Decodificador JPEG**: CompreFace usa Pillow 8.3.2 / libjpeg 9; aquí libjpeg-turbo. Con JPEG los píxeles difieren ~0,3/255 de
  media y, sobre eso, el modelo de edad cambia de tramo en ~6 % de las caras y la similitud se mueve (en fotos de PRO enviadas
  como JPEG: 2 de 120 parejas cruzan 0,85). Es ruido propio del modelo con cualquier decodificador; no se replica libjpeg 9.
- Imagen ilegible: se responde 28, como CompreFace. Sin rotar por EXIF, como `imageio`.
- El modelo de edad convertido a ONNX coincide con el original: 445 caras, mismo tramo en todas (`tools/diag_ref.py`, `tools/diag_lite.py`).

## Pruebas

    pip install -r requirements-dev.txt
    pytest
