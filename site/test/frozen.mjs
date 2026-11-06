// Frozen priors in the browser vs the paper's records: 16 fresh rooms x 30 trials, familiar goal, per seed.
//   node site/test/frozen.mjs REF.json   (REF: {arm: {seed: mean successes out of 30}})
import { readFileSync } from "node:fs";
import { Prior, prepare } from "../js/sim.js";
import { probe, frozenAgent } from "../js/protocol.js";
const D = new URL("../data/", import.meta.url), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), REF = JSON.parse(readFileSync(process.argv[2])), all = [];
for (const arm of Object.keys(REF)) {
  const d = [];
  for (const [s, rec] of Object.entries(REF[arm])) {
    const act = frozenAgent(new Prior(bin(`models/${arm}_${s}.bin`)));
    const js = E.rooms.reduce((t, r, k) => t + probe(E, r, r.familiar_goal, act, 97 + k + 31 * +s).filter(e => e.success).length, 0) / E.rooms.length;
    d.push(js - rec); all.push(js - rec);
  }
  const m = d.reduce((a, b) => a + b) / d.length; console.log(arm.padEnd(22), `browser - recorded: ${(100 * m / 30).toFixed(1)} pp per seed  [${d.map(x => (100 * x / 30).toFixed(1)).join(", ")}]`);
}
const m = all.reduce((a, b) => a + b) / all.length, sd = Math.sqrt(all.reduce((a, b) => a + (b - m) ** 2, 0) / (all.length - 1));
console.log(`all: ${(100 * m / 30).toFixed(2)} pp ± ${(100 * 2.045 * sd / Math.sqrt(all.length) / 30).toFixed(2)} (95% CI, n=${all.length})`);
