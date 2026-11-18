// End-to-end check without a browser: enlarge MNIST test digits to 280x280
// "drawings" (dark ink on white), prepare them with toMnist, read with the formula.
// Run from the repo root:  node webapp/test_preprocess.mjs
import { readFileSync } from "node:fs";
import { parseFormula, evaluate } from "./eml.js";
import { toMnist } from "./preprocess.js";

const prog = parseFormula(readFileSync("results/mnist/formula.txt", "utf8"));
const fx = JSON.parse(readFileSync("webapp/test_fixture.json", "utf8"));
const S = 280;
let right = 0;
fx.pixels.forEach((px, i) => {
  const data = new Uint8ClampedArray(S * S * 4);
  for (let y = 0; y < S; y++) {
    for (let x = 0; x < S; x++) {
      const v = 255 - px[Math.floor(y / 10) * 28 + Math.floor(x / 10)]; // dark ink on white
      const o = 4 * (y * S + x);
      data[o] = data[o + 1] = data[o + 2] = v;
      data[o + 3] = 255;
    }
  }
  const input = toMnist({ width: S, height: S, data });
  const s = evaluate(prog, input);
  if (s.indexOf(Math.max(...s)) === fx.labels[i]) right++;
});
console.log(`drawn-and-prepared digits read correctly: ${right}/${fx.pixels.length}`);
if (right < fx.pixels.length - 2) process.exit(1);
