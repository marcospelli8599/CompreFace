"""Punto de entrada: uvicorn app.server:app"""
from .main import create_app
from .onnx_engine import OnnxEngine

app = create_app(OnnxEngine())
