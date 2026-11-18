import numpy as np

from eml_study.ocr_compile import compile_mlp, forward_np


def test_compiled_mlp_matches_quantized_network():
    rng = np.random.default_rng(0)
    params = (rng.normal(0, 0.05, (4, 64)), rng.normal(0, 0.5, 4),
              rng.normal(0, 1.0, (10, 4)), rng.normal(0, 0.5, 10))
    B, roots, q = compile_mlp(params)
    X = rng.integers(0, 17, size=(20, 64)).astype(float)
    X[:, :5] = 0.0  # zero pixels must work (ln 0 = -inf path)
    Z = B.evaluate(roots, X)
    assert np.allclose(Z.real, forward_np(q, X), atol=1e-10)
    assert np.abs(Z.imag).max() < 1e-10
    kinds = set(B.kind)
    assert kinds <= {0, 1, 2}  # leaves are only 1 and pixels
