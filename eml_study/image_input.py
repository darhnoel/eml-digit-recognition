"""Turn a real image of one handwritten digit into the 64 values the formula reads.

The training digits (UCI optdigits, scikit-learn's load_digits) were made from
32x32 black-and-white bitmaps: each 4x4 block was counted, giving 8x8 values
0..16. A photo or drawing is put through the same steps:

1. grayscale; ink is whatever is darker than the paper (inverted automatically
   for light ink on a dark background);
2. threshold halfway between paper and ink brightness;
3. crop to the ink's bounding box, pad it to a centred square;
4. resize to 32x32, threshold again, count ink in each 4x4 block -> 0..16.

Best results: one digit per image, dark pen on plain paper, little else in frame.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageOps


def _ink_array(path_or_image) -> np.ndarray:
    """Grayscale image as ink strength 0..1 (1 = full ink), paper -> 0."""
    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    a = np.asarray(ImageOps.exif_transpose(img).convert("L"), dtype=np.float64)
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if border.mean() < a.mean():
        a = 255.0 - a
    paper, ink_level = np.percentile(a, 90), np.percentile(a, 1)
    if paper - ink_level < 20:
        raise ValueError("no digit found: the image has almost no contrast")
    ink = np.clip((paper - a) / (paper - ink_level), 0.0, 1.0)
    ink[ink < 0.25] = 0.0  # drop faint paper texture and shadows
    return ink


def image_to_mnist(path_or_image) -> np.ndarray:
    """Return 784 values 0..255 (row by row), prepared the way MNIST was.

    MNIST digits were scaled (keeping their shape, with smooth grey edges) to fit
    a 20x20 box, then placed in a 28x28 frame so their centre of mass is in the
    middle.
    """
    ink = _ink_array(path_or_image)
    rows, cols = np.nonzero(ink > 0.5)
    if len(rows) == 0:
        raise ValueError("no digit found")
    crop = ink[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
    h, w = crop.shape
    scale = 20.0 / max(h, w)
    nh, nw = max(1, round(h * scale)), max(1, round(w * scale))
    small = np.asarray(
        Image.fromarray((crop * 255).astype(np.uint8)).resize((nw, nh), Image.Resampling.LANCZOS),
        dtype=np.float64,
    ).clip(0, 255)
    frame = np.zeros((28, 28))
    ys, xs = np.mgrid[0:nh, 0:nw]
    total = small.sum()
    cy, cx = (ys * small).sum() / total, (xs * small).sum() / total
    top = int(round(14 - cy))
    left = int(round(14 - cx))
    top, left = min(max(top, 0), 28 - nh), min(max(left, 0), 28 - nw)
    frame[top : top + nh, left : left + nw] = small
    return frame.ravel()


def image_to_pixels(path_or_image) -> np.ndarray:
    """Return 64 float values in 0..16 (row by row), like load_digits."""
    img = path_or_image if isinstance(path_or_image, Image.Image) else Image.open(path_or_image)
    img = ImageOps.exif_transpose(img).convert("L")
    a = np.asarray(img, dtype=np.float64)

    # ink = darker than the paper; flip if the picture is light-on-dark
    border = np.concatenate([a[0], a[-1], a[:, 0], a[:, -1]])
    if border.mean() < a.mean():
        a = 255.0 - a
    paper = np.percentile(a, 90)
    ink_level = np.percentile(a, 1)
    if paper - ink_level < 20:
        raise ValueError("no digit found: the image has almost no contrast")
    ink = a < (paper + ink_level) / 2

    rows, cols = np.nonzero(ink)
    r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
    crop = ink[r0:r1, c0:c1]
    h, w = crop.shape
    side = int(max(h, w) * 1.1) + 1  # small margin, like the training bitmaps
    square = np.zeros((side, side), dtype=np.uint8)
    square[(side - h) // 2 : (side - h) // 2 + h, (side - w) // 2 : (side - w) // 2 + w] = crop * 255

    small = np.asarray(Image.fromarray(square).resize((32, 32), Image.Resampling.LANCZOS), dtype=np.float64)
    bits = (small >= 128).astype(np.float64)
    return bits.reshape(8, 4, 8, 4).sum(axis=(1, 3)).ravel()
