"""Section 4.3 checks: does the MNIST E-formula still work on photo-like input?

1. 1,000 MNIST test digits are enlarged 10x, placed off-centre on a light
   background with Gaussian noise (sigma = 8), then prepared again with the
   MNIST-style pipeline (eml_study.image_input.image_to_mnist).
2. The digits 0-9 printed in Pillow's built-in sans-serif font.

Both are read with the formula file only. Run from the project root:
    PYTHONPATH=. uv run python paper/check_real_images.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from eml_study.image_input import image_to_mnist
from eml_study.mnist_data import load
from eml_study.read_digit import load_formula, run_formula

ROOT = Path(__file__).resolve().parent.parent


def photo_like(X, rng):
    out = []
    for px in X:
        big = np.asarray(
            Image.fromarray(px.reshape(28, 28).astype(np.uint8)).resize((280, 280), Image.Resampling.BICUBIC),
            dtype=float,
        )
        canvas = np.full((400, 360), 235.0)
        r, c = rng.integers(0, 120), rng.integers(0, 80)
        canvas[r : r + 280, c : c + 280] -= big * 0.85
        canvas = np.clip(canvas + rng.normal(0, 8, canvas.shape), 0, 255).astype(np.uint8)
        out.append(image_to_mnist(Image.fromarray(canvas)))
    return np.stack(out)


def printed_digits():
    font = ImageFont.load_default(size=150)
    out = []
    for d in range(10):
        img = Image.new("L", (200, 220), 255)
        ImageDraw.Draw(img).text((50, 10), str(d), fill=0, font=font, stroke_width=6, stroke_fill=0)
        out.append(image_to_mnist(img))
    return np.stack(out)


def main():
    steps, scores, _ = load_formula(ROOT / "results/mnist/formula.txt")
    X, y = load("test")
    rng = np.random.default_rng(1)
    idx = rng.choice(len(X), 1000, replace=False)
    direct = float((run_formula(steps, scores, X[idx]).argmax(1) == y[idx]).mean())
    photos = float((run_formula(steps, scores, photo_like(X[idx], rng)).argmax(1) == y[idx]).mean())
    pred = run_formula(steps, scores, printed_digits()).argmax(1)
    out = {
        "mnist_1000_original_pixels": direct,
        "mnist_1000_photo_like_through_pipeline": photos,
        "printed_digits_predictions": pred.tolist(),
        "printed_digits_correct": int((pred == np.arange(10)).sum()),
    }
    (ROOT / "paper/data/real_images.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
