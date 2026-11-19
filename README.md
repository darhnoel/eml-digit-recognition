# Reading handwritten digits with a single binary operator

**Paper:** [paper/main.pdf](paper/main.pdf) · **Live demo:** [darhnoel.github.io/eml-digit-recognition/webapp](https://darhnoel.github.io/eml-digit-recognition/webapp/)

Odrzywołek ([arXiv:2603.21852](https://arxiv.org/abs/2603.21852)) showed that
the operator

```
E(x, y) = exp(x) − ln(y)
```

together with the constant `1` can build every function on a scientific
calculator. This repository shows what that means for a real perception task:
a handwritten-digit classifier written as **one expression whose only operator
is `E` and whose only leaves are `1` and the pixels**.

| | 8×8 digits | MNIST 28×28 |
|---|---|---|
| Accuracy of the pure E-expression | 98.33% | **96.35%** (10,000 test images) |
| Same prediction as the source network | 100% | 100% |
| Distinct `E` nodes | 17,613 | 258,842 |
| Depth | 134 | 152 |

A smaller expression (paper, Section 6): pruning the network, coarse 4-bit
weights, a more compact compiler and merging equal subexpressions give
**7,207 `E` nodes at 96.13%**, 36× fewer
([`results/mnist/formula_compact.txt`](results/mnist/formula_compact.txt)).

The expression is not trained directly. A small neural network is trained as
usual and then **compiled**: every `+`, `×`, `tanh` and weight is rewritten
into nested `E`s, with every weight built from `1`. The result is exact, but
much larger and slower than the network. See the paper for the details,
including two numerical limits of EML arithmetic.

## Try it in the browser

```sh
python3 -m http.server 8000      # from the repository root
# open http://localhost:8000/webapp/  and draw a digit
```

The page evaluates the E-formula in JavaScript: about 4 ms per digit for the
original formula and 0.3 ms for the compact one (switch between them on the
page); see [webapp/](webapp/).

## The expression itself

[`results/mnist/formula.txt`](results/mnist/formula.txt) is the full MNIST
classifier, one `E` per line:

```
e1 = E(1, 1)        # = e
e2 = E(e1, 1)       # = e^e
e3 = E(1, e2)       # = 0
e4 = E(1, p1)       # = e − ln p1
...
score_7 = e…        # the largest of score_0 … score_9 is the digit
```

## Run it

Requires Python 3.12 and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
uv run python -m eml_study.read_digit                    # read 5 MNIST test digits with the formula file only
uv run python -m eml_study.read_digit --image digit.jpg  # read your own photo of a handwritten digit
```

Reproduce everything:

```sh
uv run python -m eml_study.mnist_data          # download MNIST (≈12 MB) into data/
uv run python -m eml_study.mnist_compile       # train, compile, evaluate  → results/mnist/
uv run python -m eml_study.ocr_compile         # same for 8×8 digits      → results/ocr/
PYTHONPATH=. uv run python paper/make_figures.py       # figures and timings (Section 5)
PYTHONPATH=. uv run python paper/check_real_images.py  # photo-like and printed digits (Section 5.3)
uv run python -m eml_study.compact_compile     # the compact formula (Section 6) → results/mnist/formula_compact.txt
PYTHONPATH=. uv run python paper/compression_steps.py  # Table 4: the formula after each step, and two more seeds
PYTHONPATH=. uv run python paper/frontier.py           # Figure 4: size against accuracy (a few minutes; --plot to redraw)
PYTHONPATH=. uv run python paper/compact_table.py      # Table 5, including the Rust and browser timings
cargo test --release --manifest-path eml-eval/Cargo.toml   # Rust evaluator
uv run pytest                                  # Python tests
node webapp/test_eml.mjs && node webapp/test_preprocess.mjs && node webapp/test_photo.mjs   # browser demo
```

## Layout

| Path | What it is |
|---|---|
| `eml_study/eml_compiler.py` | Builds pure EML expressions (exp, ln, +, −, ×, constants from 1) and evaluates them |
| `eml_study/mnist_compile.py`, `ocr_compile.py` | Train a network and compile it to EML |
| `eml_study/read_digit.py`, `image_input.py` | Read digits (or photos) using only a formula file |
| `eml_study/networks.py`, `compact_compile.py`, `formula.py` | Pruned networks, the compact compiler, and simplifying a formula (Section 6) |
| `eml-eval/` | Fast evaluator in Rust: `cargo run --release --manifest-path eml-eval/Cargo.toml -- results/mnist/formula_compact.txt` |
| `paper/` | LaTeX source, figures, the PDF, and the scripts and data behind every number |
| `results/` | The compiled formula files and their summaries (`mnist/`, `ocr/`) |
| `webapp/` | Browser demo: draw a digit, read by the formula in JavaScript |

## Versions

- **v3**: Section 6 extended: ordered signs, 4-bit weights with retraining and
  more pruning take the compact expression from 13,999 to 7,207 nodes at the
  same accuracy (a new table lists every step, with two more random seeds).
- **v2**: adds Section 6 (a 13,999-node expression, 18× smaller), the Rust
  evaluator, and a switch between the two formulas in the demo; fixes signed
  zeros in the browser evaluator.
- **v1**: the paper as first published (tag `v1`).

## License

MIT, see [LICENSE](LICENSE).
