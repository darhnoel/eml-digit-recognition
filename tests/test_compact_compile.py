import numpy as np
import pytest

from eml_study.compact_compile import compile_compact, compile_ternary, quantize
from eml_study.networks import MLP

RNG = np.random.default_rng(0)
PIXELS = RNG.integers(0, 256, (40, 784)).astype(float)


def sparse_net(hidden=5, keep=0.03) -> MLP:
    W1 = RNG.normal(0, 3, (hidden, 784)) * (RNG.random((hidden, 784)) < keep) / 255
    return MLP(W1, RNG.normal(size=hidden), RNG.normal(size=(10, hidden)), RNG.normal(size=10))


@pytest.mark.parametrize("w", [1e-4, 0.3, -0.75, 1.0, -37.5])
def test_quantize_keeps_9_significant_bits(w):
    n, k = quantize(w)
    assert 256 <= n < 512 and abs(n * 2.0**-k - abs(w)) <= 2.0**-k / 2


def test_compact_formula_computes_its_quantized_network():
    formula, q = compile_compact(sparse_net())
    assert np.allclose(formula.scores(PIXELS), q(PIXELS), atol=1e-9)


def test_ternary_formula_computes_its_quantized_network():
    T = RNG.choice([-1, 0, 0, 0, 0, 1], size=(4, 784))
    net = MLP(RNG.uniform(1, 5, (4, 1)) * T / 255, RNG.normal(size=4), RNG.normal(size=(10, 4)), RNG.normal(size=10))
    formula, q = compile_ternary(net)
    assert np.allclose(formula.scores(PIXELS), q(PIXELS), atol=1e-9)
