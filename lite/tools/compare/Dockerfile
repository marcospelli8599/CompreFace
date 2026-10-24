FROM python:3.12-slim
RUN pip install --no-cache-dir requests==2.33.1 pillow==12.2.0 numpy==2.5.4
WORKDIR /h
COPY compare.py analyze.py ./
# keys.json (claves de CompreFace) se añade con: docker cp keys.json <contenedor>:/h/keys.json
