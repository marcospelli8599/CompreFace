# Entorno de un solo uso para convertir los modelos de CompreFace (TensorFlow 1) a ONNX y sacar los resultados de referencia con
# los modelos originales. No forma parte de la imagen del servicio.
# TensorFlow 2.10 a propósito: tf_slim (red de edad) necesita las capas "legacy" que TensorFlow 2.11+ ya no trae.
#   docker build -f lite/tools/convert.Dockerfile -t compreface-lite-convert lite/tools
FROM python:3.10-slim
RUN pip install --no-cache-dir "tensorflow-cpu==2.10.1" "tf2onnx==1.14.0" "onnx==1.12.0" "numpy<1.24" "protobuf<3.20" \
    "tf_slim==1.1.0" "opencv-python-headless==4.8.1.78" "scikit-image==0.19.3" "imageio==2.31.1" attrs cached_property
RUN mkdir -p /out /models /images
WORKDIR /compreface
CMD ["python", "/compreface/lite/tools/convert_models.py"]
