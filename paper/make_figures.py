"""Generate the data figures and benchmark numbers used in paper/main.tex.

Run from the project root:  uv run python paper/make_figures.py
Writes paper/figures/*.pdf and paper/data/*.json.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from eml_study.eml_compiler import EML, ONE, VAR  # noqa: E402
from eml_study.mnist_compile import train_mlp  # noqa: E402
from eml_study.mnist_data import load  # noqa: E402
from eml_study.ocr_compile import compile_mlp, forward_np  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIG, DATA = ROOT / "figures", ROOT / "data"
FIG.mkdir(exist_ok=True)
DATA.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix", "font.size": 9, "axes.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.02,
})


def wave_plan(B, roots):
    order = B._reachable(roots)
    level = {}
    for k in order:
        level[k] = 1 + max(level[B.a[k]], level[B.b[k]]) if B.kind[k] == EML else 0
    levels: dict[int, list[int]] = {}
    for k in order:
        if B.kind[k] == EML:
            levels.setdefault(level[k], []).append(k)
    pos = {k: i for i, k in enumerate(order)}
    plan = [
        (np.array([pos[k] for k in levels[L]]), np.array([pos[B.a[k]] for k in levels[L]]),
         np.array([pos[B.b[k]] for k in levels[L]]))
        for L in sorted(levels)
    ]
    ones = [pos[k] for k in order if B.kind[k] == ONE]
    vars_ = [(pos[k], B.a[k]) for k in order if B.kind[k] == VAR]
    rows = np.array([pos[r] for r in roots])

    def run(X):
        V = np.empty((len(order), len(X)), dtype=np.complex128)
        V[ones] = 1.0
        for p, c in vars_:
            V[p] = X[:, c]
        with np.errstate(all="ignore"):
            for idx, a, b in plan:
                V[idx] = np.exp(V[a]) - np.log(V[b])
        return V[rows].T

    widths = [len(levels[L]) for L in sorted(levels)]
    return run, widths


def bench(f, reps):
    f()
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


def main():
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    params = train_mlp(Xtr, ytr)
    B, roots, q = compile_mlp(params)
    run_waves, widths = wave_plan(B, roots)

    # correctness of the wave evaluator on all 10,000 test digits (in chunks)
    Z = np.concatenate([run_waves(Xte[i:i + 500]) for i in range(0, len(Xte), 500)])
    ref = forward_np(q, Xte)
    acc = float((Z.real.argmax(1) == yte).mean())

    # --- per-level widths (reported in the text) ----------------------------------
    (DATA / "level_widths.json").write_text(json.dumps(widths))
    # (the per-level widths are reported in the text; no figure)

    # --- Figure: test digits with the formula's predictions -------------------
    rng = np.random.default_rng(3)
    pred = Z.real.argmax(1)
    wrong = np.nonzero(pred != yte)[0]
    picks = list(rng.choice(np.nonzero(pred == yte)[0], 14, replace=False)) + list(rng.choice(wrong, 2, replace=False))
    fig, axes = plt.subplots(2, 8, figsize=(6.8, 2.0))
    for ax, i in zip(axes.ravel(), picks):
        ax.imshow(Xte[i].reshape(28, 28), cmap="Greys", vmin=0, vmax=255)
        ok = pred[i] == yte[i]
        ax.set_title(f"E: {pred[i]}" + ("" if ok else f" (true {yte[i]})"), fontsize=8,
                     color="black" if ok else "#b5452a")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_visible(False)
    fig.savefig(FIG / "examples.pdf")
    plt.close(fig)

    # --- Benchmarks ---------------------------------------------------------------
    one = Xte[:1]
    z = np.random.default_rng(0).normal(size=1_000_000).astype(np.complex128)
    r = z.real.copy()
    with np.errstate(all="ignore"):
        t_e_op = bench(lambda: np.exp(z) - np.log(z), 5) / 1e6
    t_mac = bench(lambda: r * 1.5 + r, 5) / 1e6
    out = {
        "accuracy_wave_evaluator_10000": acc,
        "max_abs_logit_error_10000": float(np.abs(Z.real - ref).max()),
        "levels": len(widths),
        "level_width_min": int(min(widths)),
        "level_width_median": float(np.median(widths)),
        "level_width_max": int(max(widths)),
        "e_nodes": int(sum(widths)),
        "network_us_per_image_single": bench(lambda: forward_np(q, one), 2000) * 1e6,
        "network_us_per_image_batch10000": bench(lambda: forward_np(q, Xte), 5) / len(Xte) * 1e6,
        "eml_sequential_ms_single": bench(lambda: B.evaluate_lowmem(roots, one), 3) * 1e3,
        "eml_waves_ms_single": bench(lambda: run_waves(one), 5) * 1e3,
        "eml_waves_ms_per_image_batch100": bench(lambda: run_waves(Xte[:100]), 3) / 100 * 1e3,
        "ns_per_complex_E": t_e_op * 1e9,
        "ns_per_real_multiply_add": t_mac * 1e9,
        "multiply_adds_network": int(q[0].size + q[2].size),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "torch": torch.__version__,
    }
    (DATA / "mnist_benchmark.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
