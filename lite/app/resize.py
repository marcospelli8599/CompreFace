"""Recorte de cara a 160x160 con el mismo resultado que CompreFace.

CompreFace llama a `skimage.transform.resize(recorte, (160, 160))` con scikit-image 0.17.2. En esa versión el suavizado
gaussiano previo se aplica sobre la imagen de 8 bits y su resultado se queda en enteros; las versiones nuevas convierten
antes a decimales y no cuantizan. La diferencia (menos de un nivel de gris) basta para que el modelo de edad cambie de tramo
en algunas caras, justo cuando el recorte se reduce (fotos de móvil), así que se reproduce el procedimiento antiguo.
Comprobado contra el resize de la imagen de CompreFace 1.2.0: error medio 1e-6 (frente a 3e-3 con el resize actual).
"""
import numpy as np
from scipy import ndimage as ndi
from skimage.transform import AffineTransform, warp

SIZE = 160


def squish_img(img: np.ndarray, size: int = SIZE) -> np.ndarray:
    """`img`: recorte RGB (alto, ancho, 3) de 8 bits. Devuelve (size, size, 3) en decimales de 0 a 1."""
    height, width = img.shape[:2]
    factors = np.array([height / size, width / size, 1.0])
    sigma = np.maximum(0, (factors - 1) / 2)
    smoothed = ndi.gaussian_filter(img, sigma, mode="mirror")  # sobre enteros: queda cuantizado, como en scikit-image 0.17.2
    # Los píxeles de salida (x, y) leen la entrada en factor * (x + 0.5) - 0.5 (el píxel 0 está en 0.5).
    fy, fx = factors[0], factors[1]
    matrix = np.array([[fx, 0.0, 0.5 * fx - 0.5], [0.0, fy, 0.5 * fy - 0.5], [0.0, 0.0, 1.0]])
    return warp(smoothed, AffineTransform(matrix=matrix), output_shape=(size, size, 3), order=1, mode="reflect", cval=0, clip=True, preserve_range=False)
