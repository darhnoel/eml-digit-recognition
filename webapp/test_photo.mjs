// Photo path: compare JavaScript preparation with Python's (eml_study/image_input.py)
// on 12 sample images, and the formula's answers on both.
// Run from the repo root:  node webapp/test_photo.mjs
import { readFileSync } from "node:fs";
import { parseFormula, evaluate } from "./eml.js";
import { toMnist } from "./preprocess.js";

const prog = parseFormula(readFileSync("results/mnist/formula.txt", "utf8"));
const fx = JSON.parse(readFileSync("webapp/test_photo_fixture.json", "utf8"));
const argmax = (a) => a.indexOf(Math.max(...a));
let same = 0;
for (const f of fx) {
  const data = new Uint8ClampedArray(f.w * f.h * 4);
  Buffer.from(f.gray_b64, "base64").forEach((v, i) => { data[4 * i] = data[4 * i + 1] = data[4 * i + 2] = v; data[4 * i + 3] = 255; });
  const js = toMnist({ width: f.w, height: f.h, data });
  const diff = Math.max(...js.map((v, i) => Math.abs(v - f.python[i])));
  const a = argmax(evaluate(prog, js));
  const b = argmax(evaluate(prog, f.python));
  if (a === b) same++;
  console.log(`${f.name.padEnd(14)} js=${a} python=${b}  max pixel diff ${diff.toFixed(1)}`);
}
console.log(`same answer: ${same}/${fx.length}`);
if (same !== fx.length) process.exit(1);
