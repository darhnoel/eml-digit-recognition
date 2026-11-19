"""Table 4: the MNIST expression after each compression step (Section 6),
and the final step repeated with two more random seeds.

Every row is a formula evaluated as a formula on all 10,000 test images;
"same" is the share of images on which it predicts the same class as the
(quantized) network it was compiled from.
Writes paper/data/compression_steps.json (a few minutes).

Run from the project root:  PYTHONPATH=. uv run python paper/compression_steps.py
"""

import json
from pathlib import Path

import numpy as np

from eml_study.compact_compile import NETWORK, build, compile_compact, probe_images, stages
from eml_study.mnist_compile import train_mlp
from eml_study.mnist_data import load
from eml_study.networks import MLP

ROOT = Path(__file__).resolve().parent.parent


def row(step, formula, net, Xte, yte):
    scores = formula.scores(Xte)
    out = {
        "step": step,
        "E_nodes": len(formula),
        "depth": formula.depth,
        "weights": net.n_weights,
        "pixels": int(np.count_nonzero(np.abs(net.W1).sum(0))),
        "accuracy": float((scores.argmax(1) == yte).mean()),
        "same_as_network": float((scores.argmax(1) == net(Xte).argmax(1)).mean()),
    }
    print(json.dumps(out), flush=True)
    return out


def main():
    Xtr, ytr = load("train")
    Xte, yte = load("test")
    probes = probe_images(Xtr)
    original = json.loads((ROOT / "results/mnist/summary.json").read_text())
    rows = [{"step": "original (Section 5)", "E_nodes": original["dag_eml_nodes"], "depth": original["max_depth"],
             "weights": 25408, "pixels": 784, "accuracy": original["accuracy_pure_eml"],
             "same_as_network": original["eml_vs_quantized_prediction_agreement"]}]

    same, q = compile_compact(MLP(*train_mlp(Xtr, ytr)))
    rows.append(row("compact compiler, same network", same, q, Xte, yte))

    steps = stages(Xtr, ytr, seed=0)
    _, pruned = next(steps)
    compiled, q = compile_compact(pruned)
    rows.append(row("pruned network (64 hidden, 2% of layer 1)", compiled, q, Xte, yte))
    rows.append(row("merging equal subexpressions (v2)", compiled.merge_equal(probes), q, Xte, yte))
    ordered, q = compile_compact(pruned, order_signs=True)
    rows.append(row("ordered signs", ordered.merge_equal(probes), q, Xte, yte))
    for name, net in steps:
        _, merged, q = build(net, probes)
        rows.append(row(name, merged, q, Xte, yte))
    final = net
    saved = MLP.load(NETWORK)
    assert all(np.array_equal(a, b) for a, b in zip(final, saved)), "the pipeline no longer gives the saved network"

    seeds = []
    for seed in (1, 2):
        *_, (_, net) = stages(Xtr, ytr, seed=seed)
        _, merged, q = build(net, probes)
        seeds.append({"seed": seed, **row(f"final, seed {seed}", merged, q, Xte, yte)})
    out = {"steps": rows, "other_seeds": seeds}
    (ROOT / "paper/data/compression_steps.json").write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
