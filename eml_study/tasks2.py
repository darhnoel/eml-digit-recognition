"""Tasks for the two-input study. Labels: 1 = class A, 0 = class B."""

from __future__ import annotations

import math

import numpy as np

from eml_study.tasks import MARGIN, _moons_task, _xor_task

TASKS2 = [
    "U1_linear", "U2_xor", "U3_disc", "U4_parabola", "U5_wave", "U6_wedge",
    "U7_moons", "C1_ring", "C2_checker",
]
CONTROL_TASKS2 = {"C1_ring", "C2_checker"}
SANITY_TASKS2 = {"U1_linear"}
EVALUATED_TASKS2 = set(TASKS2) - CONTROL_TASKS2 - SANITY_TASKS2


def label_and_distance2(name: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class-A indicator and (first-order) distance to the true boundary."""
    u1, u2 = X[:, 0], X[:, 1]
    if name == "U1_linear":
        f = u1 + 0.5 * u2 - 0.1
        return (f > 0).astype(float), np.abs(f) / math.sqrt(1.25)
    if name == "U3_disc":
        r = np.hypot(u1, u2)
        return (r < 0.6).astype(float), np.abs(r - 0.6)
    if name == "U4_parabola":
        f = u2 - (1.5 * u1**2 - 0.5)
        return (f > 0).astype(float), np.abs(f) / np.sqrt(1 + (3 * u1) ** 2)
    if name == "U5_wave":
        f = u2 - 0.4 * np.sin(3 * u1)
        return (f > 0).astype(float), np.abs(f) / np.sqrt(1 + (1.2 * np.cos(3 * u1)) ** 2)
    if name == "U6_wedge":
        f = 0.15 + 0.5 * (u1 + 1) - np.abs(u2)
        return (f > 0).astype(float), np.abs(f) / math.sqrt(1.25)
    if name == "C1_ring":
        r = np.hypot(u1, u2)
        return ((r > 0.35) & (r < 0.7)).astype(float), np.minimum(np.abs(r - 0.35), np.abs(r - 0.7))
    if name == "C2_checker":
        i = np.clip(np.floor((u1 + 1) * 1.5), 0, 2)
        j = np.clip(np.floor((u2 + 1) * 1.5), 0, 2)
        lines = np.array([-1 / 3, 1 / 3])
        d1 = np.min(np.abs(u1[:, None] - lines[None, :]), axis=1)
        d2 = np.min(np.abs(u2[:, None] - lines[None, :]), axis=1)
        return ((i + j) % 2 == 0).astype(float), np.minimum(d1, d2)
    raise ValueError(f"no uniform rule for {name}")


def _uniform_task2(name: str, n: int, rng: np.random.Generator):
    kept, total = [], 0
    while total < n:
        X = rng.uniform(-1.0, 1.0, size=(4 * n, 2))
        _, dist = label_and_distance2(name, X)
        X = X[dist >= MARGIN]
        kept.append(X)
        total += len(X)
    X = np.concatenate(kept)[:n]
    y, _ = label_and_distance2(name, X)
    return X, y


def make_task2(name: str, split: str, n: int = 400, data_seed: int = 0):
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    rng = np.random.default_rng(data_seed if split == "train" else data_seed + 1)
    if name == "U2_xor":
        return _xor_task(n, rng)
    if name == "U7_moons":
        return _moons_task(n, rng)
    if name in TASKS2:
        return _uniform_task2(name, n, rng)
    raise ValueError(f"unknown task {name}")
