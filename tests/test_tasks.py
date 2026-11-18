import math

import numpy as np
import pytest

from eml_study.tasks import MARGIN, TASKS, label_and_distance, make_task

UNIFORM = ["T1_linear", "T3_band", "T4_stripes", "T5_circles"]


@pytest.mark.parametrize("name", TASKS)
def test_shape_range_and_determinism(name):
    X, y = make_task(name, "train")
    assert X.shape == (400, 2) and y.shape == (400,)
    assert X.dtype == np.float64
    assert np.all(np.abs(X) <= 1.0)
    X2, y2 = make_task(name, "train")
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    Xt, _ = make_task(name, "test")
    assert not np.array_equal(X, Xt)


@pytest.mark.parametrize("name", TASKS)
def test_class_balance(name):
    _, y = make_task(name, "train")
    assert 0.2 <= y.mean() <= 0.8


@pytest.mark.parametrize("name", UNIFORM)
def test_margin(name):
    for split in ("train", "test"):
        X, y = make_task(name, split)
        labels, dist = label_and_distance(name, X)
        assert np.all(dist >= MARGIN)
        assert np.array_equal(labels, y)


@pytest.mark.parametrize("name", ["T2_xor", "T6_moons"])
def test_class_separation(name):
    X, y = make_task(name, "train")
    A, B = X[y == 1], X[y == 0]
    d = np.sqrt(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1)).min()
    assert d >= 0.3 - 1e-9


def test_t4_pattern():
    X, y = make_task("T4_stripes", "train")
    p = (X[:, 0] + X[:, 1]) / math.sqrt(2)
    order = np.argsort(p)
    labels = y[order]
    changes = np.nonzero(np.diff(labels))[0]
    assert len(changes) == 3
    assert labels[0] == 1  # A first
    for idx, boundary in zip(changes, (-0.6, 0.0, 0.9)):
        mid = (p[order][idx] + p[order][idx + 1]) / 2
        assert abs(mid - boundary) <= MARGIN + 0.05
