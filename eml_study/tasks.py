"""Synthetic 2-D tasks. Labels: 1 = class A, 0 = class B."""

from __future__ import annotations

import math

import numpy as np

TASKS = ["T1_linear", "T2_xor", "T3_band", "T4_stripes", "T5_circles", "T6_moons"]
CONTROL_TASKS = {"T5_circles", "T6_moons"}
EVALUATED_TASKS = {"T2_xor", "T3_band", "T4_stripes"}
SANITY_TASKS = {"T1_linear"}

MARGIN = 0.02
T4_BOUNDARIES = (-0.6, 0.0, 0.9)
SQRT2 = math.sqrt(2.0)


def label_and_distance(name: str, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Class-A indicator and distance to the nearest true boundary (uniform tasks)."""
    u1, u2 = X[:, 0], X[:, 1]
    if name == "T1_linear":
        f = u1 + 0.5 * u2 - 0.1
        return (f > 0).astype(float), np.abs(f) / math.sqrt(1.25)
    if name == "T3_band":
        d = np.abs(u1 - u2)
        return (d < 0.4).astype(float), np.abs(d - 0.4) / SQRT2
    if name == "T4_stripes":
        p = (u1 + u2) / SQRT2
        b0, b1, b2 = T4_BOUNDARIES
        a = (p < b0) | ((p >= b1) & (p < b2))
        dist = np.min(np.abs(p[:, None] - np.array(T4_BOUNDARIES)[None, :]), axis=1)
        return a.astype(float), dist
    if name == "T5_circles":
        r = np.hypot(u1, u2)
        return (r < 0.6).astype(float), np.abs(r - 0.6)
    raise ValueError(f"no uniform rule for {name}")


def _uniform_task(name: str, n: int, rng: np.random.Generator):
    kept = []
    total = 0
    while total < n:
        X = rng.uniform(-1.0, 1.0, size=(4 * n, 2))
        _, dist = label_and_distance(name, X)
        X = X[dist >= MARGIN]
        kept.append(X)
        total += len(X)
    X = np.concatenate(kept)[:n]
    y, _ = label_and_distance(name, X)
    return X, y


def _xor_task(n: int, rng: np.random.Generator):
    per = n // 4
    parts, labels = [], []
    for cx, cy in ((0.5, 0.5), (-0.5, -0.5), (-0.5, 0.5), (0.5, -0.5)):
        r = 0.15 * np.sqrt(rng.uniform(0, 1, per))
        a = 2 * math.pi * rng.uniform(0, 1, per)
        parts.append(np.stack([cx + r * np.cos(a), cy + r * np.sin(a)], axis=1))
        labels.append(np.full(per, 1.0 if cx * cy > 0 else 0.0))
    return np.concatenate(parts), np.concatenate(labels)


def moons_points(theta_upper: np.ndarray, theta_lower: np.ndarray):
    upper = np.stack([np.cos(theta_upper), np.sin(theta_upper)], axis=1)
    lower = np.stack([1 - np.cos(theta_lower), 0.5 - np.sin(theta_lower)], axis=1)
    shift = np.array([0.5, 0.25])
    return 0.6 * (upper - shift), 0.6 * (lower - shift)


def _moons_task(n: int, rng: np.random.Generator):
    per = n // 2
    up, lo = moons_points(rng.uniform(0, math.pi, per), rng.uniform(0, math.pi, per))
    X = np.concatenate([up, lo])
    y = np.concatenate([np.ones(per), np.zeros(per)])
    return X, y


def make_task(name: str, split: str, n: int = 400, data_seed: int = 0):
    """Return (X, y) as float64 NumPy arrays of shapes (n, 2) and (n,)."""
    if split not in ("train", "test"):
        raise ValueError("split must be 'train' or 'test'")
    rng = np.random.default_rng(data_seed if split == "train" else data_seed + 1)
    if name == "T2_xor":
        return _xor_task(n, rng)
    if name == "T6_moons":
        return _moons_task(n, rng)
    if name in TASKS:
        return _uniform_task(name, n, rng)
    raise ValueError(f"unknown task {name}")
