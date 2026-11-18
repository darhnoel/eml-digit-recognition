import numpy as np
import pytest
from PIL import Image, ImageDraw

from eml_study.image_input import image_to_pixels


def _bar_image(dark_on_light=True):
    paper, ink = (255, 0) if dark_on_light else (30, 230)
    img = Image.new("L", (120, 160), paper)
    ImageDraw.Draw(img).rectangle([55, 20, 65, 140], fill=ink)  # a "1"
    return img


@pytest.mark.parametrize("dark_on_light", [True, False])
def test_values_shape_range_and_centred_stroke(dark_on_light):
    v = image_to_pixels(_bar_image(dark_on_light)).reshape(8, 8)
    assert v.shape == (8, 8) and v.min() >= 0 and v.max() <= 16
    assert v[:, 3:5].sum() > 0.8 * v.sum()  # the vertical bar lands in the middle columns


def test_light_and_dark_versions_match():
    assert np.array_equal(image_to_pixels(_bar_image(True)), image_to_pixels(_bar_image(False)))


def test_blank_image_rejected():
    with pytest.raises(ValueError):
        image_to_pixels(Image.new("L", (50, 50), 255))


def test_mnist_format_centred_in_20px_box():
    from eml_study.image_input import image_to_mnist

    v = image_to_mnist(_bar_image()).reshape(28, 28)
    assert v.shape == (28, 28) and v.min() >= 0 and v.max() <= 255
    rows, cols = np.nonzero(v > 30)
    assert rows.max() - rows.min() + 1 <= 21  # longest side fits the 20 px box
    ys, xs = np.mgrid[0:28, 0:28]
    assert abs((ys * v).sum() / v.sum() - 14) < 1.0 and abs((xs * v).sum() / v.sum() - 14) < 1.0
