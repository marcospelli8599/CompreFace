# Comparación de compreface-lite con el CompreFace real

Lanza las dos llamadas que hace Vero (verificar y detectar con pose y edad) contra **CompreFace 1.2.0** (misma configuración que
producción) y contra **compreface-lite**, con las mismas fotos, y resume las diferencias. Fotos: LFW (conjunto público).

1. `docker volume create litemodels` y copiar dentro los `.onnx` (`docker cp onnx/. <contenedor>:/models`).
2. `docker volume create lfwdata` y copiar `lfw.tgz` a `/data` (SHA-256 `055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0`).
3. `docker compose up -d --no-deps compreface-postgres-db compreface-core compreface-api lite` (sin `admin`/`fe`: ahorra memoria).
4. Claves de CompreFace: su alta por la API de administración falla en este banco (el token se emite pero no autoriza), así que se usan las
   que trae la instalación: detección `00000000-0000-0000-0000-000000000003` y verificación `00000000-0000-0000-0000-000000000004`.
   Guardarlas en `keys.json`: `{"VERIFICATION": "...0004", "DETECTION": "...0003"}`.
5. `docker build -t cfcmp-harness .` y `docker run --rm --network cfcmp_default -v lfwdata:/data cfcmp-harness python compare.py keys.json 100`
   (`FRAC=1.0` para enviar todas las fotos a 1280 px; `analyze.py` da el detalle).

Notas aprendidas:

- Con **JPEG** no coinciden del todo: CompreFace decodifica con Pillow 8.3.2 / libjpeg 9 y aquí se usa libjpeg-turbo. Mueve los píxeles
  ~0,3 de 255 de media, y el modelo de edad cambia de tramo en ~6 % de las caras. Con **PNG** (sin pérdida) los dos reciben los mismos
  píxeles y la comparación es la real: por eso `compare.py` envía PNG.
- CompreFace elige el lector de imagen por la **extensión del nombre** del archivo: hay que mandar `f.png`, `source.jpg`... (sin extensión da 21).
- El `core` de CompreFace necesita ~1,5-2,5 GB; en un Docker de 4 GB usar `UWSGI_PROCESSES=1` y no arrancar `admin`.

`../diag_ref.py` (entorno TensorFlow de `convert.Dockerfile`) y `../diag_lite.py` comparan el modelo de edad original y el convertido
sobre los mismos recortes (445 caras: mismo tramo en todas, diferencia máxima de probabilidad 8e-5).
