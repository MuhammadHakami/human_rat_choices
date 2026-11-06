// The learned agents the page shows (PPO and the stage-3 brain circuit after learning the moved goal) against the paper's
// recorded probes (moved goal, trained room, 30 trials):  node site/test/learned.mjs
import { readFileSync } from "node:fs";
import { Prior, Circuit, prepare } from "../js/sim.js";
import { probe, frozenAgent, circuitAgent } from "../js/protocol.js";
const D = new URL("../data/", import.meta.url), J = f => JSON.parse(readFileSync(new URL(f, D)));
const bin = f => { const b = readFileSync(new URL(f, D)); return b.buffer.slice(b.byteOffset, b.byteOffset + b.byteLength); };
const E = prepare(J("env.json")), C = J("circuits.json"), d = { ppo: [], bl3: [] };
for (const L of C.learned) {
  const room = E.rooms.find(r => r.configuration === 100 + 2 * L.pair), act = L.kind === "ppo" ? frozenAgent(new Prior(bin(`learned/ppo_${L.prior}_${L.seed}.bin`)))
    : (() => { const c = new Circuit(new Prior(bin(`models/${L.prior}_${L.seed}.bin`)), bin(`kc/${L.seed}.bin`), C.configs.none, C.gate_half0), w = new Float32Array(bin(`learned/bl3_${L.prior}_${L.seed}.bin`));
               c.Wa.set(w.subarray(0, 4 * 2048)); c.Wv.set(w.subarray(4 * 2048)); return circuitAgent(c); })();
  const js = probe(E, room, room.moved_goal, act, 5 + L.seed).filter(e => e.success).length; d[L.kind].push(js - L.recorded);
  console.log(L.kind.padEnd(4), L.prior.padEnd(20), L.seed, "recorded", String(L.recorded).padStart(2), "browser", String(js).padStart(2));
}
for (const k in d) { const v = d[k], m = v.reduce((a, b) => a + b) / v.length, s = Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / (v.length - 1));
  console.log(k, `browser - recorded: ${(100 * m / 30).toFixed(1)} pp ± ${(100 * 2.145 * s / Math.sqrt(v.length) / 30).toFixed(1)} (95% CI, n=${v.length})`); }
