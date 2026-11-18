# EML Digit Reader (web)

Draw a digit; it is read by the pure E-formula `results/mnist/formula.txt`
(258,842 nested `E(x, y) = exp(x) − ln(y)`), evaluated in JavaScript.
No neural-network code runs on the page.

A switch selects a second formula, `results/mnist/formula_compact.txt`
(13,999 E's, 96.13% on the MNIST test set, from a pruned network with a more
compact compiler); it reads a digit in about 0.35 ms instead of about 4 ms.

Run from the repository root (the page loads `../results/mnist/formula.txt`):

```sh
python3 -m http.server 8000
# open http://localhost:8000/webapp/
```

Files: `eml.js` parses and evaluates the formula (complex arithmetic with
IEEE special values, `ln 0 = −∞`); `preprocess.js` prepares a drawing or photo
the way MNIST was prepared; `index.html` is the page.

Checks (Node 18+): `node webapp/test_eml.mjs` compares the JavaScript
evaluator with the Python one on 25 MNIST test digits; `node
webapp/test_preprocess.mjs` runs enlarged test digits through the image
preparation and the formula; `node webapp/test_photo.mjs` checks that
photos are prepared exactly as in `eml_study/image_input.py`.
