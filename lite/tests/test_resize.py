import numpy as np

from app.resize import SIZE, squish_img


def _gradient(height: int, width: int) -> np.ndarray:
    row = np.linspace(0, 255, width, dtype=np.uint8)
    return np.stack([np.tile(row, (height, 1))] * 3, axis=2)


def test_output_shape_and_range():
    for height, width in ((120, 150), (160, 160), (500, 420), (900, 800)):
        out = squish_img(_gradient(height, width))
        assert out.shape == (SIZE, SIZE, 3)
        assert out.dtype == np.float64
        assert 0.0 <= out.min() and out.max() <= 1.0


def test_exact_size_is_just_scaled_to_zero_one():
    img = _gradient(SIZE, SIZE)
    np.testing.assert_allclose(squish_img(img), img / 255.0, atol=1e-12)


def test_constant_image_stays_constant_when_downscaling():
    out = squish_img(np.full((640, 480, 3), 117, dtype=np.uint8))
    np.testing.assert_allclose(out, 117 / 255.0, atol=1e-9)


def test_downscaling_keeps_the_gradient_direction():
    out = squish_img(_gradient(480, 480))
    columns = out[80, :, 0]
    assert np.all(np.diff(columns) >= -1e-9)  # sigue creciendo de izquierda a derecha
    assert columns[0] < 0.05 and columns[-1] > 0.95
