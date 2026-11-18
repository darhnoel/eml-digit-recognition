"""Read a handwritten digit using only an E-formula file.

No neural network and no training code: this reads a formula file, where every
line is one E(x, y) = exp(x) - ln(y), plugs in the pixel values, computes
score_0 .. score_9, and answers with the largest.

Two formulas exist:
    results/mnist/formula.txt   28x28 MNIST digits (784 pixels)  [default]
    results/ocr/formula.txt     8x8 scikit-learn digits (64 pixels)

Run:
    python -m eml_study.read_digit                      # demo on 5 test digits
    python -m eml_study.read_digit --image photo.png    # real images (png/jpg),
        one handwritten digit each; several files can be given
    python -m eml_study.read_digit --formula results/ocr/formula.txt ...
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np

LINE = re.compile(r"^(\w+) = E\((\w+), (\w+)\)$")
SCORE = re.compile(r"^score_(\d) = (\w+)$")
DEFAULT_FORMULA = "results/mnist/formula.txt"


def load_formula(path=DEFAULT_FORMULA):
    """Return (steps, scores, n_vars): steps are (name, x, y) for name = E(x, y)."""
    steps, scores, n_vars = [], {}, 0
    for line in Path(path).read_text().splitlines():
        if m := LINE.match(line):
            steps.append(m.groups())
            for v in m.groups()[1:]:
                if v[0] == "p":
                    n_vars = max(n_vars, int(v[1:]))
        elif m := SCORE.match(line):
            scores[int(m.group(1))] = m.group(2)
    return steps, [scores[k] for k in range(10)], n_vars


def run_formula(steps, scores, pixels):
    """Evaluate the formula. pixels: one image (n_vars,) or many (n, n_vars).

    Returns real scores, shape (10,) or (n, 10).
    """
    P = np.atleast_2d(np.asarray(pixels, dtype=np.complex128))
    value = {"1": np.ones(len(P), dtype=np.complex128)}
    for i in range(P.shape[1]):
        value[f"p{i + 1}"] = P[:, i]
    with np.errstate(all="ignore"):
        for name, x, y in steps:  # each line: name = E(x, y) = exp(x) - ln(y)
            value[name] = np.exp(value[x]) - np.log(value[y])
    out = np.stack([value[s].real for s in scores], axis=1)
    return out[0] if np.ndim(pixels) == 1 else out


def show(pixels):
    """ASCII picture of what the formula sees (28x28 shown at 14x14)."""
    v = np.asarray(pixels, dtype=float)
    side = int(round(len(v) ** 0.5))
    img = v.reshape(side, side) / v.max() if v.max() > 0 else v.reshape(side, side)
    if side == 28:
        img = img.reshape(14, 2, 14, 2).mean(axis=(1, 3))
    shades = " .:-=+*#%@"
    return "\n".join("   " + "".join(shades[min(9, int(x * 9.99))] * 2 for x in r) for r in img)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--formula", default=DEFAULT_FORMULA)
    ap.add_argument("--image", nargs="+", help="image files (png/jpg), or .txt grids of pixel values")
    args = ap.parse_args(argv)
    steps, scores, n_vars = load_formula(args.formula)
    print(f"loaded formula: {len(steps):,} E's, {n_vars} pixels, 10 scores\n")

    if args.image:
        from eml_study.image_input import image_to_mnist, image_to_pixels

        convert = image_to_mnist if n_vars == 784 else image_to_pixels
        images = []
        for path in args.image:
            px = np.loadtxt(path).ravel() if path.endswith(".txt") else convert(path)
            images.append((px, None, path))
    elif n_vars == 784:
        from eml_study.mnist_data import load

        Xte, yte = load("test")
        images = [(Xte[i], yte[i], f"MNIST test digit {i}") for i in range(5)]
    else:
        from eml_study.ocr_compile import load_split

        _, _, Xte, yte = load_split()
        images = [(Xte[i], yte[i], f"test digit {i}") for i in range(5)]

    all_scores = run_formula(steps, scores, np.stack([p for p, _, _ in images]))
    for (pixels, truth, label), s in zip(images, all_scores):
        print(f"{label}  (as the formula sees it):")
        print(show(pixels))
        print("   scores:", " ".join(f"{k}:{v:+.1f}" for k, v in enumerate(s)))
        tail = "" if truth is None else f"   (true digit: {truth})"
        print(f"   formula says: {int(s.argmax())}{tail}\n")


if __name__ == "__main__":
    main()
