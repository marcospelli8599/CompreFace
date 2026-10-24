# compreface-lite: servicio de verificación y detección facial compatible con lo que usa Vero de CompreFace.
#   docker build -t compreface-lite lite
# Los modelos .onnx NO van en la imagen: se montan en /models (los genera tools/convert_models.py).
FROM python:3.12-slim

RUN useradd --system --uid 10001 --no-create-home app
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app ./app

ENV MODELS_DIR=/models \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
USER 10001
EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/healthcheck', timeout=3).status == 200 else 1)"
# Cada proceso carga los modelos en memoria; WORKERS sube las peticiones simultáneas por réplica.
CMD ["sh", "-c", "exec uvicorn app.server:app --host 0.0.0.0 --port 8080 --workers ${WORKERS:-2}"]
