import numpy as np
import pytest

from eml_study.tasks import MARGIN
from eml_study.tasks2 import TASKS2, label_and_distance2, make_task2

UNIFORM = [t for t in TASKS2 if t not in ("U2_xor", "U7_moons")]


@pytest.mark.parametrize("name", TASKS2)
def test_shape_determinism_balance(name):
    X, y = make_task2(name, "train")
    assert X.shape == (400, 2) and np.all(np.abs(X) <= 1.0)
    X2, y2 = make_task2(name, "train")
    assert np.array_equal(X, X2) and np.array_equal(y, y2)
    assert not np.array_equal(X, make_task2(name, "test")[0])
    assert 0.15 <= y.mean() <= 0.85


@pytest.mark.parametrize("name", UNIFORM)
def test_margin_and_labels(name):
    X, y = make_task2(name, "train")
    labels, dist = label_and_distance2(name, X)
    assert np.all(dist >= MARGIN) and np.array_equal(labels, y)
