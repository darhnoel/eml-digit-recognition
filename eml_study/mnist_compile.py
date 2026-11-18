"""Compile a trained MNIST classifier (28x28 handwritten digits) into pure EML.

1. Train a 784 -> HIDDEN -> 10 tanh MLP on the 60,000 MNIST training digits.
2. Compile it into E-formulas whose only leaves are 1 and the 784 raw pixel
   values (0..255), with every weight built from 1 (9-bit mantissa).
3. Evaluate the formulas on all 10,000 test digits and write the formula file.

Run: python -m eml_study.mnist_compile [--hidden 32] [--out-dir results/mnist]
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import eml_study  # noqa: F401  (float64 default)
from eml_study.mnist_data import load
from eml_study.ocr_compile import compile_mlp, forward_np


def train_mlp(Xtr, ytr, hidden=32, epochs=15, batch=128, seed=0):
    torch.manual_seed(seed)
    X = torch.from_numpy(Xtr / 255.0)
    y = torch.from_numpy(ytr)
    W1 = (torch.randn(hidden, 784) * (1 / 28)).requires_grad_(True)
    b1 = torch.zeros(hidden, requires_grad=True)
    W2 = (torch.randn(10, hidden) * (1 / hidden**0.5)).requires_grad_(True)
    b2 = torch.zeros(10, requires_grad=True)
    opt = torch.optim.Adam([W1, b1, W2, b2], lr=2e-3, weight_decay=1e-5)
    gen = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        perm = torch.randperm(len(X), generator=gen)
        for i in range(0, len(X), batch):
            idx = perm[i : i + batch]
            opt.zero_grad()
            logits = torch.tanh(X[idx] @ W1.T + b1) @ W2.T + b2
            torch.nn.functional.cross_entropy(logits, y[idx]).backward()
            opt.step()
    # fold the /255 input scaling into W1 so the formula reads raw pixels
    return (W1.detach().numpy() / 255.0, b1.detach().numpy(), W2.detach().numpy(), b2.detach().numpy())


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--out-dir", default="results/mnist")
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    Xtr, ytr = load("train")
    Xte, yte = load("test")
    t0 = time.time()
    params = train_mlp(Xtr, ytr, hidden=args.hidden)
    acc_float = (forward_np(params, Xte).argmax(1) == yte).mean()
    print(f"trained network: test accuracy {acc_float:.4f} ({time.time() - t0:.0f}s)", flush=True)

    t1 = time.time()
    B, roots, qparams = compile_mlp(params)
    ref_q = forward_np(qparams, Xte)
    acc_quant = (ref_q.argmax(1) == yte).mean()
    stats = B.stats(roots)
    per_class = [B.stats([r]) for r in roots]
    print(f"compiled: {stats['dag_eml_nodes']:,} E's, depth {stats['max_depth']} ({time.time() - t1:.0f}s)", flush=True)

    t2 = time.time()
    Z = np.concatenate([B.evaluate_lowmem(roots, Xte[i : i + 2000]) for i in range(0, len(Xte), 2000)])
    t3 = time.time()
    acc_eml = (Z.real.argmax(1) == yte).mean()
    n_lines = B.write_formula(
        roots, out / "formula.txt", n_vars=784,
        header=f"MNIST 28x28, 784 -> {args.hidden} -> 10 tanh network compiled to E; "
               f"test accuracy {acc_eml:.4f}",
    )
    summary = {
        "test_samples": int(len(yte)),
        "hidden_units": args.hidden,
        "network_parameters": int(sum(p.size for p in params)),
        "accuracy_float_network": float(acc_float),
        "accuracy_quantized_network": float(acc_quant),
        "accuracy_pure_eml": float(acc_eml),
        "eml_vs_quantized_prediction_agreement": float((Z.real.argmax(1) == ref_q.argmax(1)).mean()),
        "max_abs_logit_error": float(np.abs(Z.real - ref_q).max()),
        "max_abs_imag_part": float(np.abs(Z.imag).max()),
        "non_finite_outputs": int((~np.isfinite(Z)).sum()),
        "dag_eml_nodes": stats["dag_eml_nodes"],
        "formula_file_lines": n_lines,
        "expanded_tree_nodes_one_output_min": min(s["tree_size_total"] for s in per_class),
        "expanded_tree_nodes_one_output_max": max(s["tree_size_total"] for s in per_class),
        "max_depth": stats["max_depth"],
        "seconds_train": round(t1 - t0, 1),
        "seconds_compile": round(t2 - t1, 1),
        "seconds_eval_eml_10000": round(t3 - t2, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    lines = ["# MNIST compiled to pure EML", ""]
    lines += [f"- **{k}**: {v:,}" if isinstance(v, int) else f"- **{k}**: {v}" for k, v in summary.items()]
    (out / "report.md").write_text("\n".join(lines) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
