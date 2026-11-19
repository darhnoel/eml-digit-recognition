"""Table 5: the original MNIST expression against the compact ones (Section 6).

Three expressions: the original (Section 5); the same network compiled with
the compact compiler; and the pruned, merged expression. For each: size,
depth, accuracy, and time per image in NumPy, in the browser evaluator
(webapp/eml.js under Node) and in eml-eval (Rust). Rust must give the same
answer as NumPy on every test image.
Writes paper/data/compact.json.

Run from the project root, after `uv run python -m eml_study.compact_compile`:
    uv run python paper/compact_table.py
"""

import json
import subprocess
import time
from pathlib import Path

import numpy as np

from eml_study.compact_compile import compile_compact
from eml_study.formula import Formula
from eml_study.mnist_compile import train_mlp
from eml_study.mnist_data import load
from eml_study.networks import MLP

ROOT = Path(__file__).resolve().parent.parent
FORMULAS = {"original": ROOT / "results/mnist/formula.txt", "compact": ROOT / "results/mnist/formula_compact.txt"}


def seconds_per_call(f, reps: int) -> float:
    f()  # warm-up
    start = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - start) / reps


def javascript(path: Path, X: np.ndarray) -> float:
    """Seconds per image for webapp/eml.js, one image at a time."""
    script = f"""
import {{ parseFormula, evaluate }} from {json.dumps(str(ROOT / "webapp/eml.js"))};
import {{ readFileSync }} from "node:fs";
const prog = parseFormula(readFileSync({json.dumps(str(path))}, "utf8"));
const X = JSON.parse(readFileSync(0, "utf8"));
for (const x of X.slice(0, 5)) evaluate(prog, x);
const start = performance.now();
for (const x of X) evaluate(prog, x);
console.log((performance.now() - start) / X.length);"""
    run = subprocess.run(["node", "--input-type=module", "-e", script], input=json.dumps(X.astype(int).tolist()),
                         capture_output=True, text=True, check=True)
    return float(run.stdout) / 1e3


def rust(path: Path, reference: Path | None = None) -> dict:
    cmd = ["cargo", "run", "--release", "--quiet", "--manifest-path", str(ROOT / "eml-eval/Cargo.toml"), "--",
           str(path), "--json", "--mnist", str(ROOT / "data/mnist")]
    if reference:
        cmd += ["--reference", str(reference)]
    return json.loads(subprocess.run(cmd, capture_output=True, text=True, check=True).stdout)


def describe(formula: Formula, X, y) -> dict:
    scores = formula.scores(X)
    return {"E_nodes": len(formula), "depth": formula.depth, "accuracy": float((scores.argmax(1) == y).mean())}


def main():
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    out = {}

    # the original network, compiled compactly (no pruning, no merging)
    same_net, _ = compile_compact(MLP(*train_mlp(Xtr, ytr)))
    out["same_network_compact"] = describe(same_net, Xte, yte)
    print("same network, compact compiler", out["same_network_compact"], flush=True)

    for name, path in FORMULAS.items():
        formula = Formula.read(path)
        ref = ROOT / "results/compact/numpy_scores.f64" if name == "compact" else None
        r = rust(path, ref)
        if ref:
            assert r["same_answer_as_numpy"] == len(Xte), "Rust and NumPy disagree"
        out[name] = {
            **describe(formula, Xte, yte),
            "file_kb": round(path.stat().st_size / 1024),
            "us_numpy_batch": seconds_per_call(lambda f=formula: f.scores(Xte[:1000]), 2) / 1000 * 1e6,
            "us_javascript": javascript(path, Xte[:100]) * 1e6,
            "us_rust_one_core": r["us_per_digit_one_core"],
            "us_rust_all_cores": r["us_per_digit_all_cores"],
            "threads": r["threads"],
            "E_per_image_rust": r["e_count_per_digit"],
        }
        print(name, out[name], flush=True)

    net = MLP.load(ROOT / "results/compact/network.npz")
    out["network_us_numpy_batch"] = seconds_per_call(lambda: net(Xte), 20) / len(Xte) * 1e6
    (ROOT / "paper/data/compact.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
