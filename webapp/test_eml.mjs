// Check the browser evaluator against the Python (NumPy) evaluator.
// Run from the repo root:  node webapp/test_eml.mjs
import { readFileSync } from "node:fs";
import { parseFormula, evaluate } from "./eml.js";

const prog = parseFormula(readFileSync("results/mnist/formula.txt", "utf8"));
const fx = JSON.parse(readFileSync("webapp/test_fixture.json", "utf8"));
let worst = 0;
let agree = 0;
const t0 = performance.now();
fx.pixels.forEach((px, i) => {
  const s = evaluate(prog, px);
  s.forEach((v, k) => { worst = Math.max(worst, Math.abs(v - fx.scores[i][k])); });
  const argmax = (arr) => arr.indexOf(Math.max(...arr));
  if (argmax(s) === argmax(fx.scores[i])) agree++;
});
const ms = (performance.now() - t0) / fx.pixels.length;
console.log(`nodes ${prog.nodes}, pixels ${prog.nVars}`);
console.log(`same prediction as Python: ${agree}/${fx.pixels.length}, max score difference ${worst.toExponential(2)}`);
console.log(`${ms.toFixed(1)} ms per image in JavaScript`);
if (agree !== fx.pixels.length || !(worst < 1e-9)) { console.error("MISMATCH"); process.exit(1); }
