"""Train every (unit, task, learning rate) with 100 seeds batched together.

Run: python -m eml_study.train [--family single|two] [--quick] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

import eml_study  # noqa: F401  (float64 default)
from eml_study.family import FAMILIES, SINGLE, Family
from eml_study.units import Activation

LRS = (1e-3, 1e-2, 1e-1)
FULL_SEEDS = list(range(100))
FULL_STEPS = 3000
QUICK_SEEDS = [0, 1]
QUICK_STEPS = 200


@dataclass
class RunRecord:
    unit: str
    task: str
    lr: float
    seed: int
    train_acc: float
    test_acc: float
    success: bool
    final_loss: float
    nan_flag: bool
    pole_guard_frac: float
    exp_clamp_hits: int
    clip_step_frac: float
    min_abs_s: float | None
    max_abs_gprime: float | None
    w1: float
    w2: float
    b: float
    t: float
    k: float
    c1: float | None = None
    c2: float | None = None
    d: float | None = None


FIELDS = [f.name for f in dataclasses.fields(RunRecord)]


def init_params(activation, X_train: torch.Tensor, seeds, family: Family = SINGLE) -> torch.Tensor:
    unit = activation.name if isinstance(activation, Activation) else activation
    n_w = family.n_params - 2
    rows = []
    for seed in seeds:
        gen = torch.Generator().manual_seed(seed)
        w = torch.randn(n_w, generator=gen)
        t = torch.quantile(family.init_g(unit, w, X_train), 0.5)
        rows.append(torch.cat([w, torch.stack([t, torch.tensor(1.0)])]))
    return torch.stack(rows)


def train_batch(
    activation,
    task: str,
    lr: float,
    seeds,
    steps: int = FULL_STEPS,
    clip_norm: float = 10.0,
    init: torch.Tensor | None = None,
    family: Family | str = SINGLE,
) -> list[RunRecord]:
    fam = FAMILIES[family] if isinstance(family, str) else family
    unit = activation.name if isinstance(activation, Activation) else activation
    seeds = list(seeds)
    Xtr_np, ytr_np = fam.make_task(task, "train")
    Xte_np, yte_np = fam.make_task(task, "test")
    Xtr, ytr = torch.from_numpy(Xtr_np), torch.from_numpy(ytr_np)
    Xte, yte = torch.from_numpy(Xte_np), torch.from_numpy(yte_np)
    n = len(seeds)

    start = init.clone() if init is not None else init_params(unit, Xtr, seeds, fam)
    assert start.shape == (n, fam.n_params)
    params = start.clone().requires_grad_(True)
    opt = torch.optim.Adam([params], lr=lr)

    frozen = torch.zeros(n, dtype=torch.bool)
    clipped = torch.zeros(n, dtype=torch.long)
    pole_hits = torch.zeros(n, dtype=torch.long)
    clamp_hits = torch.zeros(n, dtype=torch.long)
    ytr_b = ytr.expand(n, -1)

    for _ in range(steps):
        opt.zero_grad(set_to_none=False)
        aux, z = fam.forward(unit, params, Xtr)
        loss_per = F.binary_cross_entropy_with_logits(z, ytr_b, reduction="none").mean(1)
        loss_per.sum().backward()
        with torch.no_grad():
            grad = params.grad
            bad = ~torch.isfinite(loss_per) | ~torch.isfinite(grad).all(dim=1)
            frozen |= bad
            grad[frozen] = 0.0
            norms = grad.norm(dim=1)
            over = norms > clip_norm
            scale = torch.where(over, clip_norm / norms, torch.ones_like(norms))
            grad.mul_(scale[:, None])
            clipped += (over & ~frozen).long()
            saved = params.detach().clone()
        opt.step()
        with torch.no_grad():
            params[frozen] = saved[frozen]
            p, c = fam.guards(unit, aux)
            live = (~frozen).long()
            pole_hits += p * live
            clamp_hits += c * live

    with torch.no_grad():
        aux_tr, z_tr = fam.forward(unit, params, Xtr)
        _, z_te = fam.forward(unit, params, Xte)
        loss_per = F.binary_cross_entropy_with_logits(z_tr, ytr_b, reduction="none").mean(1)
        nonfinite = ~torch.isfinite(z_tr).all(1) | ~torch.isfinite(z_te).all(1)
        nan_flag = frozen | nonfinite
        train_acc = ((z_tr > 0).double() == ytr).double().mean(1)
        test_acc = ((z_te > 0).double() == yte).double().mean(1)
        diag = fam.diag(unit, aux_tr)
        final = params.detach()

    records = []
    for i, seed in enumerate(seeds):
        tr, te, nf = train_acc[i].item(), test_acc[i].item(), bool(nan_flag[i])
        p = final[i].tolist()
        if fam.n_params == 5:
            w = dict(w1=p[0], w2=p[1], b=p[2], t=p[3], k=p[4])
        else:
            w = dict(w1=p[0], w2=p[1], b=p[2], c1=p[3], c2=p[4], d=p[5], t=p[6], k=p[7])
        records.append(
            RunRecord(
                unit=unit,
                task=task,
                lr=lr,
                seed=seed,
                train_acc=tr,
                test_acc=te,
                success=(tr >= 0.99 and te >= 0.95 and not nf),
                final_loss=loss_per[i].item(),
                nan_flag=nf,
                pole_guard_frac=pole_hits[i].item() / (steps * len(Xtr)),
                exp_clamp_hits=int(clamp_hits[i]),
                clip_step_frac=clipped[i].item() / steps,
                min_abs_s=diag[0][i].item() if diag is not None else None,
                max_abs_gprime=diag[1][i].item() if diag is not None else None,
                **w,
            )
        )
    return records


def params_of(r: RunRecord) -> list[float]:
    if r.c1 is None:
        return [r.w1, r.w2, r.b, r.t, r.k]
    return [r.w1, r.w2, r.b, r.c1, r.c2, r.d, r.t, r.k]


def write_csv(rows: list[RunRecord], path, fields=None) -> None:
    fields = fields or (FIELDS if any(r.c1 is not None for r in rows) else FIELDS[:-3])
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            d = {k: v for k, v in dataclasses.asdict(r).items() if k in fields}
            w.writerow({k: ("" if v is None else repr(v) if isinstance(v, float) else v) for k, v in d.items()})


def _parse(field: str, v: str):
    if v == "":
        return None
    if field in ("unit", "task"):
        return v
    if field in ("seed", "exp_clamp_hits"):
        return int(v)
    if field in ("success", "nan_flag"):
        return v == "True"
    return float(v)


def read_csv(path) -> list[RunRecord]:
    with open(path, newline="") as fh:
        return [RunRecord(**{k: _parse(k, v) for k, v in row.items()}) for row in csv.DictReader(fh)]


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--family", choices=sorted(FAMILIES), default="single")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args(argv)
    fam = FAMILIES[args.family]
    seeds, steps = (QUICK_SEEDS, QUICK_STEPS) if args.quick else (FULL_SEEDS, FULL_STEPS)
    out = Path(args.out_dir or ("results" if fam.name == "single" else "results/two_input"))
    out.mkdir(parents=True, exist_ok=True)
    rows: list[RunRecord] = []
    jobs = [(u, t, lr) for u in fam.units for t in fam.tasks for lr in LRS]
    t0 = time.time()
    for i, (u, t, lr) in enumerate(jobs, 1):
        rows += train_batch(u, t, lr, seeds, steps=steps, family=fam)
        done = sum(r.success for r in rows[-len(seeds):])
        print(f"[{i:3d}/{len(jobs)}] {u:10s} {t:12s} lr={lr:g}  success {done}/{len(seeds)}"
              f"  ({time.time() - t0:.0f}s)", flush=True)
    write_csv(rows, out / "results.csv")
    print(f"wrote {len(rows)} rows to {out / 'results.csv'}")


if __name__ == "__main__":
    main()
