from fractions import Fraction

import numpy as np

from eml_study.eml_compiler import Builder

X = np.array([[0.0, 1.5], [-2.5, 0.0], [3.0, -4.0], [1e-8, 2.0], [0.7, -0.3], [16.0, 16.0]])


def _eval(b, node):
    return b.evaluate([node], X)[:, 0]


def test_basic_ops_with_zeros_and_negatives():
    b = Builder()
    x, y = b.var(0), b.var(1)
    cases = {
        b.exp(x): np.exp(X[:, 0]),
        b.sub(x, y): X[:, 0] - X[:, 1],
        b.add(x, y): X[:, 0] + X[:, 1],
        b.neg(y): -X[:, 1],
        b.mul(x, y): X[:, 0] * X[:, 1],
    }
    for node, want in cases.items():
        got = _eval(b, node)
        assert np.allclose(got.real, want, rtol=1e-12, atol=1e-12)
        assert np.abs(got.imag).max() < 1e-12


def test_zero_constant():
    b = Builder()
    assert np.allclose(_eval(b, b.zero), 0.0)


def test_integers_and_fractions_from_one():
    b = Builder()
    for n in (1, 2, 3, 7, 16, 255, 511, 700):
        assert abs(_eval(b, b.integer(n))[0].real - n) < 1e-9 * n
    f = Fraction(123, 512)
    assert abs(_eval(b, b.ln_fraction(f))[0].real - np.log(123 / 512)) < 1e-12


def test_large_integers_rejected():
    import pytest

    with pytest.raises(ValueError):
        Builder().integer(4096)


def test_quantized_log_constants():
    b = Builder()
    for w in (0.0123, -0.5, 3.7, -41.0, 1e-4, 0.999):
        node, q = b.ln_abs_quantized(w)
        assert abs(q - abs(w)) / abs(w) < 2**-8
        assert abs(_eval(b, node)[0].real - np.log(q)) < 1e-12
        assert abs(_eval(b, b.exp(node))[0].real - q) < 1e-12 * q


def test_leaves_are_only_one_and_vars():
    b = Builder()
    node = b.mul(b.var(0), b.integer(5))
    kinds = {b.kind[i] for i in range(len(b.kind))}
    assert kinds <= {0, 1, 2}
    assert np.allclose(_eval(b, node).real, 5 * X[:, 0])


def test_stats_counts_tree_and_depth():
    b = Builder()
    e = b.exp(b.var(0))  # eml(x, 1): one eml node, two leaves
    s = b.stats([e])
    assert s["dag_eml_nodes"] == 1 and s["tree_size_total"] == 3 and s["max_depth"] == 1


def test_lowmem_evaluation_matches_and_formula_file(tmp_path):
    b = Builder()
    x, y = b.var(0), b.var(1)
    roots = [b.add(b.mul(x, y), b.integer(3)), b.sub(x, y)]
    assert np.allclose(b.evaluate_lowmem(roots, X), b.evaluate(roots, X), equal_nan=True)
    n = b.write_formula(roots, tmp_path / "f.txt", n_vars=2)
    text = (tmp_path / "f.txt").read_text()
    assert n == text.count(" = E(") and "score_1 = e" in text


def test_mnist_idx_parser():
    from eml_study.mnist_data import parse_idx

    raw = bytes([0, 0, 8, 3]) + (2).to_bytes(4, "big") + (2).to_bytes(4, "big") + (3).to_bytes(4, "big") + bytes(range(12))
    assert parse_idx(raw).shape == (2, 2, 3)
